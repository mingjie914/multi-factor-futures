"""Frozen effective-factor subset selection for the intraday daily contract.

This workflow consumes the current admitted effective-factor library (not a
fixed-size candidate pool), computes the frozen admission interval with its
required warm-up, clusters all daily signals together by default, and builds
an exact nested portfolio path under the production construction recipe. It
never mutates the effective library and never uses post-cutoff data for
selection. The input count is always discovered from the configured effective
library path, so a future library version needs no workflow change.
"""
from __future__ import annotations

import csv
from itertools import combinations
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from core.config import load_config
from core.date_policy import factor_admission_start, research_cutoff
from core.registry import list_registered
from core.sectors import SECTOR_MAP
from optimization.costs import SimpleFuturesCost
from optimization.factor_weighting import (
    prepare_complete_history,
    rank_information_coefficients,
)
from pipeline.runner import PipelineRunner
from research.artifacts import sha256_file
from research.effective_factor_library import load_library
from research.governance import factor_family
from research.historical_portfolio_search import (
    PortfolioEvaluator,
    PortfolioRecipe,
    performance_metrics,
    robust_summary,
    robustness_key,
)


SELECTION_SCHEMA_VERSION = 3
CLUSTER_CORRELATION_THRESHOLD = 0.50
MIN_CROSS_SECTION = 10
N_IS_SEGMENTS = 4
COMPACT_MAX_FACTORS = 12
PORTFOLIO_SEARCH_MAX_FACTORS = 24
PORTFOLIO_SEARCH_EXACT_WIDTH = 16
PORTFOLIO_SEARCH_PATIENCE = 3
PORTFOLIO_SEARCH_SHARPE_TOLERANCE = 0.10


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty selection table: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _rank_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, method="average", pct=True)


def _daily_spearman_ic(
    factor: pd.DataFrame,
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    for date in dates:
        x = pd.to_numeric(factor.loc[date], errors="coerce")
        y = pd.to_numeric(returns.loc[date], errors="coerce")
        mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) < MIN_CROSS_SECTION:
            continue
        xr = x.loc[mask].rank(method="average")
        yr = y.loc[mask].rank(method="average")
        corr = xr.corr(yr)
        if pd.notna(corr) and np.isfinite(float(corr)):
            values[pd.Timestamp(date)] = float(corr)
    return pd.Series(values, dtype=float).sort_index()


def _segments(dates: pd.DatetimeIndex) -> list[pd.DatetimeIndex]:
    chunks = np.array_split(np.asarray(dates), N_IS_SEGMENTS)
    return [pd.DatetimeIndex(chunk) for chunk in chunks if len(chunk)]


def _metric(values: pd.Series) -> tuple[float, float, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    ir = mean / std if np.isfinite(std) and std > 0.0 else float("nan")
    return mean, float(values.gt(0.0).mean()), ir


def _exposure_correlation(
    ranks: dict[str, pd.DataFrame], names: Iterable[str]
) -> pd.DataFrame:
    series = {}
    for name in names:
        frame = ranks[name]
        # Pandas 3 defaults to the new stack implementation, where the
        # legacy ``dropna`` argument is rejected.  The old implementation is
        # intentional here because the correlation panel must retain the
        # rectangular date×instrument missing-value positions.  Keep the
        # fallback for the project's supported Pandas 1.5+ range.
        try:
            stacked = frame.stack(dropna=False, future_stack=False)
        except TypeError:
            stacked = frame.stack(dropna=False)
        series[name] = stacked
    panel = pd.DataFrame(series)
    corr = panel.corr(min_periods=MIN_CROSS_SECTION * 3).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    return corr.clip(-1.0, 1.0)


def _cluster(corr: pd.DataFrame, names: list[str]) -> dict[str, int]:
    if len(names) == 1:
        return {names[0]: 1}
    values = corr.reindex(index=names, columns=names).to_numpy(dtype=float)
    values = np.nan_to_num(np.abs(values), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(values, 1.0)
    distance = np.clip(1.0 - values, 0.0, 1.0)
    tree = linkage(squareform(distance, checks=False), method="complete")
    labels = fcluster(
        tree,
        t=1.0 - CLUSTER_CORRELATION_THRESHOLD,
        criterion="distance",
    )
    return {name: int(label) for name, label in zip(names, labels)}


def _representative_key(row: dict) -> tuple:
    return (
        float(row.get("segment_positive_ratio", 0.0)),
        float(row.get("worst_segment_mean_ic", -np.inf)),
        float(row.get("mean_ic", -np.inf)),
        float(row.get("coverage", 0.0)),
        -float(row.get("rank_churn", np.inf)),
        str(row["factor"]),
    )


def _compact_representatives(rows: list[dict], max_count: int) -> list[str]:
    """Keep the strongest distinct-cluster representatives up to one limit."""
    ranked = sorted(rows, key=_representative_key, reverse=True)
    return [str(row["factor"]) for row in ranked[:max_count]]


def _segment_bounds(dates: pd.DatetimeIndex, count: int = 4) -> list[tuple]:
    chunks = np.array_split(np.asarray(dates), int(count))
    return [
        (pd.Timestamp(chunk[0]), pd.Timestamp(chunk[-1]))
        for chunk in chunks if len(chunk)
    ]


def _combined_ic_key(
    ic: pd.DataFrame,
    factors: Iterable[str],
    segments: list[tuple],
) -> tuple[float, float, float]:
    combined = ic[list(factors)].mean(axis=1, skipna=True)
    means: list[float] = []
    sharpes: list[float] = []
    for start, end in segments:
        values = combined.loc[start:end].dropna()
        if len(values) < 20:
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=1))
        means.append(mean)
        sharpes.append(
            mean / std * np.sqrt(252.0) if std > 0.0 else -10.0
        )
    if not sharpes:
        return (0.0, -10.0, -10.0)
    return (
        float(np.mean(np.asarray(means) > 0.0)),
        float(np.min(sharpes)),
        float(np.median(sharpes)),
    )


def _recommended_nested_candidate(
    path: list[dict],
    *,
    sharpe_tolerance: float = PORTFOLIO_SEARCH_SHARPE_TOLERANCE,
) -> dict:
    """Choose the smallest near-best robust member of one nested path."""
    if not path:
        raise ValueError("nested portfolio path is empty")
    best_positive = max(float(row["positive_segment_ratio"]) for row in path)
    stable = [
        row for row in path
        if float(row["positive_segment_ratio"]) == best_positive
    ]
    best_worst = max(float(row["worst_sharpe"]) for row in stable)
    stable = [
        row for row in stable
        if float(row["worst_sharpe"]) >= best_worst - float(sharpe_tolerance)
    ]
    best_median = max(float(row["median_sharpe"]) for row in stable)
    stable = [
        row for row in stable
        if float(row["median_sharpe"]) >= best_median - float(sharpe_tolerance)
    ]
    return min(
        stable,
        key=lambda row: (
            int(row["factor_count"]),
            float(row["annual_turnover"]),
            tuple(-value for value in robustness_key(row)),
        ),
    )


def _production_recipe(config) -> PortfolioRecipe:
    policy = config.production_portfolio
    return PortfolioRecipe(
        factor_weight=str(policy.factor_weight_method),
        top_n=int(policy.top_n_per_side),
        sector_cap=int(policy.sector_count_cap),
        asset_weight=str(policy.asset_weight_method),
        asset_min_fraction=float(policy.asset_min_fraction),
        asset_max_fraction=float(policy.asset_max_fraction),
        gross_exposure=float(policy.gross_exposure),
        asset_max_overrides=tuple(sorted(policy.asset_max_overrides.items())),
        sector_weight_caps=tuple(sorted(policy.sector_weight_caps.items())),
    )


def _configured_cost_model(config) -> SimpleFuturesCost:
    policy = config.costs
    return SimpleFuturesCost(
        turnover_cost_rate=float(policy.turnover_cost_rate),
        annual_fee=float(policy.annual_fee),
        annual_roll_cost=float(policy.annual_roll_cost),
        periods_per_year=float(policy.periods_per_year),
        cost_stage=str(policy.cost_stage),
    )


def _run_nested_portfolio_search(
    *,
    evaluator: PortfolioEvaluator,
    portfolio_ic: pd.DataFrame,
    representatives: list[str],
    recipe: PortfolioRecipe,
    segments: list[tuple[pd.Timestamp, pd.Timestamp]],
    max_factors: int = PORTFOLIO_SEARCH_MAX_FACTORS,
    exact_width: int = PORTFOLIO_SEARCH_EXACT_WIDTH,
    patience: int = PORTFOLIO_SEARCH_PATIENCE,
) -> tuple[list[dict], list[dict], str]:
    """Build one exact, nested path without fixing the final factor count."""
    pool = list(dict.fromkeys(map(str, representatives)))
    if len(pool) < 2:
        raise ValueError("nested portfolio search requires at least two factors")
    if int(exact_width) < 1 or int(patience) < 1 or int(max_factors) < 2:
        raise ValueError("nested portfolio search limits must be positive")

    selection_dates = portfolio_ic.loc[segments[0][0]:segments[-1][1]].index

    def icir_history_available(factors: tuple[str, ...]) -> bool:
        values = portfolio_ic[list(factors)]
        started = False
        for date in selection_dates:
            history = prepare_complete_history(
                values.loc[values.index < date].tail(60),
                minimum_observations=30,
            )
            available = history.shape[1] >= 2
            if started and not available:
                return False
            started = started or available
        return started

    def cheap_candidates(candidates: Iterable[tuple[str, ...]]) -> list[tuple]:
        rows = [
            (_combined_ic_key(portfolio_ic, factors, segments), tuple(factors))
            for factors in candidates
            if icir_history_available(tuple(factors))
        ]
        return sorted(rows, key=lambda row: (row[0], row[1]), reverse=True)

    evaluations: list[dict] = []

    def evaluate(
        factors: tuple[str, ...],
        *,
        step: int,
        added_factor: str,
        cheap_key: tuple[float, float, float],
    ) -> dict | None:
        row = {
            "step": int(step),
            "factor_count": len(factors),
            "added_factor": str(added_factor),
            "factors": list(factors),
            "cheap_positive_segment_ratio": float(cheap_key[0]),
            "cheap_worst_ic_sharpe": float(cheap_key[1]),
            "cheap_median_ic_sharpe": float(cheap_key[2]),
            "status": "rejected_runtime",
            "error": "",
            "segment_count": 0,
            "positive_segment_ratio": 0.0,
            "worst_sharpe": -10.0,
            "median_sharpe": -10.0,
            "median_annual_return": -1.0,
            "worst_drawdown": -1.0,
            "annual_turnover": float("nan"),
            "full_annual_return": float("nan"),
            "full_sharpe": float("nan"),
            "full_max_drawdown": float("nan"),
            "full_total_return": float("nan"),
        }
        try:
            ledger = evaluator.ledger(factors, recipe)
        except RuntimeError as exc:
            row["error"] = str(exc)
            evaluations.append(row)
            return None
        robust = robust_summary(ledger, segments, initial_anchor=True)
        full = performance_metrics(ledger["net_return"], initial_anchor=True)
        row.update({
            "status": "evaluated",
            **robust,
            "annual_turnover": (
                float(ledger["executed_traded_notional"].iloc[1:].mean() * 252.0)
                if len(ledger) > 1 else float("nan")
            ),
            "full_annual_return": float(full.get("annual_return", np.nan)),
            "full_sharpe": float(full.get("sharpe", np.nan)),
            "full_max_drawdown": float(full.get("max_drawdown", np.nan)),
            "full_total_return": float(full.get("total_return", np.nan)),
        })
        evaluations.append(row)
        return row

    def exact_candidates(
        shortlisted: list[tuple], *, step: int
    ) -> list[dict]:
        rows: list[dict] = []
        for cheap_key, factors in shortlisted:
            row = evaluate(
                factors,
                step=step,
                added_factor=("|".join(factors) if step == 2 else factors[-1]),
                cheap_key=cheap_key,
            )
            if row is not None:
                rows.append(row)
            if len(rows) >= int(exact_width):
                break
        return rows

    initial_rows = exact_candidates(
        cheap_candidates(combinations(pool, 2)), step=2
    )
    if not initial_rows:
        raise RuntimeError("no executable two-factor portfolio survived exact evaluation")
    selected_row = max(
        initial_rows,
        key=lambda row: (
            *robustness_key(row),
            -float(row["annual_turnover"]),
            tuple(row["factors"]),
        ),
    )
    path = [dict(selected_row)]
    selected = tuple(selected_row["factors"])
    best_key = robustness_key(selected_row)
    stale_steps = 0
    stop_reason = "candidate_pool_exhausted"
    search_limit = min(len(pool), int(max_factors))
    for size in range(3, search_limit + 1):
        remaining = [name for name in pool if name not in selected]
        shortlisted = cheap_candidates(
            tuple((*selected, candidate)) for candidate in remaining
        )
        exact_rows = exact_candidates(shortlisted, step=size)
        if not exact_rows:
            stop_reason = f"no_executable_extension_at_size_{size}"
            break
        selected_row = max(
            exact_rows,
            key=lambda row: (
                *robustness_key(row),
                -float(row["annual_turnover"]),
                str(row["added_factor"]),
            ),
        )
        path.append(dict(selected_row))
        selected = tuple(selected_row["factors"])
        current_key = robustness_key(selected_row)
        if current_key > best_key:
            best_key = current_key
            stale_steps = 0
        else:
            stale_steps += 1
        if stale_steps >= int(patience):
            stop_reason = f"no_robustness_improvement_for_{int(patience)}_sizes"
            break
    else:
        if search_limit < len(pool):
            stop_reason = f"computational_safety_cap_{search_limit}"
    return path, evaluations, stop_reason


def _load_library(
    config,
    *,
    source_run_dir: str | None = None,
    allowed_horizons: tuple[int, ...] | None = None,
) -> tuple[Path, list[dict]]:
    if source_run_dir is not None:
        root = Path(source_run_dir)
        if not root.is_absolute():
            root = Path(__file__).resolve().parents[1] / root
        root = root.resolve()
        summary_path = root / "validation_summary.json"
        contract_path = root / "run_contract.json"
        passed_path = root / "passed_factors.csv"
        if not summary_path.exists() or not contract_path.exists() or not passed_path.exists():
            raise FileNotFoundError(
                "common-horizon selection requires a finalized validation run: "
                f"{root}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        horizon_mode = str(summary.get("horizon_mode", ""))
        horizon = int(summary.get("common_horizon", 0) or 0)
        if (
            horizon_mode != "common_horizon"
            or horizon < 1
            or (allowed_horizons is not None and horizon not in allowed_horizons)
        ):
            raise ValueError(
                "selection source is not the requested common-horizon run: "
                f"mode={horizon_mode!r}, horizon={horizon}"
            )
        if contract.get("horizon_policy", {}).get("common_horizon") != horizon:
            raise ValueError("validation run and summary horizon contracts disagree")
        frame = pd.read_csv(passed_path, encoding="utf-8-sig")
        if "final_pass" not in frame or "factor" not in frame:
            raise ValueError("passed_factors.csv is missing the final-pass schema")
        frame = frame.loc[frame["final_pass"].astype(bool)].copy()
        if frame.empty:
            raise ValueError("common-horizon validation run has no passed factors")
        rows = []
        for record in frame.to_dict(orient="records"):
            name = str(record["factor"])
            ic = float(record.get("is_ic", 0.0) or 0.0)
            rows.append({
                "factor": name,
                "status": "validation_passed",
                "signal_frequency": "daily",
                "input_bar_frequency": "",
                "family": str(record.get("family", "") or factor_family(name)),
                "registered_horizons": str(record.get("registered_horizons", "")),
                "best_period": horizon,
                "direction": 1 if ic >= 0.0 else -1,
                "source_run": root.name,
                "oos_ic": record.get("oos_ic"),
            })
        names = {str(row["factor"]) for row in rows}
        if len(names) != len(rows):
            raise ValueError("common-horizon passed factors contain duplicates")
        return passed_path.resolve(), rows

    path = Path(config.factor_library.path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    payload = load_library(path)
    factors = [
        row for row in payload.get("factors", [])
        if isinstance(row, dict) and row.get("status") == "effective"
    ]
    if not factors:
        raise ValueError("effective factor library has no effective members")
    names = {str(row.get("factor")) for row in factors}
    if len(names) != len(factors):
        raise ValueError("effective factor library contains duplicate names")
    for row in factors:
        period = int(row.get("best_period", 0) or 0)
        if period < 1 or (
            allowed_horizons is not None and period not in allowed_horizons
        ):
            raise ValueError(
                f"factor {row.get('factor')!r} has invalid best period {period}"
            )
        if str(row.get("signal_frequency")) != "daily":
            raise ValueError(
                f"factor {row.get('factor')!r} is not daily: "
                f"{row.get('signal_frequency')!r}"
            )
    return path.resolve(), factors


def run_effective_factor_selection(
    *,
    run_id: str,
    config_path: str = "config/default.yaml",
    max_compact_factors: int = COMPACT_MAX_FACTORS,
    source_run_dir: str | None = None,
    common_horizon: int | None = None,
    group_by_best_period: bool = False,
) -> Path:
    """Run governed effective-library subset selection and write one immutable run."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(run_id)):
        raise ValueError("run_id only allows letters, numbers, dot, underscore and hyphen")
    if int(max_compact_factors) < 1:
        raise ValueError("max_compact_factors must be positive")
    if source_run_dir is not None:
        if common_horizon is None or int(common_horizon) < 1:
            raise ValueError(
                "common-horizon selection requires a positive common_horizon"
            )
        selection_horizons = (int(common_horizon),)
    else:
        selection_horizons = None

    project = Path(__file__).resolve().parents[1]
    output = project / "runs" / "factor_selection" / str(run_id)
    config = load_config(config_path)
    library_path, library_rows = _load_library(
        config,
        source_run_dir=source_run_dir,
        allowed_horizons=selection_horizons,
    )
    if selection_horizons is None:
        selection_horizons = tuple(sorted({
            int(row["best_period"]) for row in library_rows
        }))
    output.mkdir(parents=True, exist_ok=False)
    runner = PipelineRunner(config=config)
    selection_start = factor_admission_start(config)
    cutoff = research_cutoff(config)
    warmup_days = int(config.validation_policy.warmup_days_by_frequency["daily"])
    factor_start = selection_start - pd.Timedelta(days=warmup_days)
    selection_dates = pd.DatetimeIndex(
        runner.data_manager.get_calendar(selection_start, cutoff)
    )
    requested_dates = pd.DatetimeIndex(
        runner.data_manager.get_calendar(factor_start, cutoff)
    )
    if len(selection_dates) < 80:
        raise ValueError("factor-set selection requires at least 80 trading days")
    universe = pd.Index(config.universe)
    names = sorted(str(row["factor"]) for row in library_rows)
    factor_compute_started = time.perf_counter()
    raw = runner.factor_engine.compute_factors(
        names, requested_dates, universe, parallel=False, chunk_size=64
    )
    factor_compute_seconds = time.perf_counter() - factor_compute_started
    factor_timings = list(runner.factor_engine.computation_timings)
    if set(raw) != set(names):
        raise ValueError("factor engine did not return all admitted factor names")
    registry = list_registered("factor").get("factor", {})
    missing_registry = sorted(set(names) - set(registry))
    if missing_registry:
        raise ValueError("effective factors are not registered: " + ", ".join(missing_registry))
    for row in library_rows:
        name = str(row["factor"])
        declared_input = str(
            getattr(registry[name], "input_bar_frequency", "1min")
        )
        stored_input = str(row.get("input_bar_frequency", "") or "")
        if stored_input and stored_input != declared_input:
            raise ValueError(
                f"factor {name!r} input bar metadata {stored_input!r} does not "
                f"match registered implementation {declared_input!r}"
            )
        row["input_bar_frequency"] = declared_input
    directions = {str(row["factor"]): int(row["direction"]) for row in library_rows}
    periods = {str(row["factor"]): int(row["best_period"]) for row in library_rows}

    returns = {
        horizon: runner.data_manager.get_forward_returns(
            requested_dates, universe, period=horizon
        )
        for horizon in selection_horizons
    }
    segments = _segments(selection_dates)
    all_diagnostics: list[dict] = []
    ranks: dict[str, pd.DataFrame] = {}
    full_ranks: dict[str, pd.DataFrame] = {}
    for row in library_rows:
        name = str(row["factor"])
        horizon = periods[name]
        raw_rank = _rank_frame(raw[name])
        direction = directions[name]
        full_ranks[name] = raw_rank if direction == 1 else (1.0 - raw_rank)
        ranks[name] = full_ranks[name].loc[selection_dates]
        ic = _daily_spearman_ic(
            ranks[name],
            returns[horizon].loc[selection_dates],
            selection_dates,
        )
        segment_values = []
        for segment in segments:
            mean, positive_ratio, ir = _metric(ic.reindex(segment))
            segment_values.append((mean, positive_ratio, ir))
        means = [item[0] for item in segment_values if np.isfinite(item[0])]
        positive_ratios = [item[1] for item in segment_values if np.isfinite(item[1])]
        prior = ranks[name].diff().abs().stack().dropna()
        row_out = {
            "factor": name,
            "horizon": horizon,
            "signal_frequency": row["signal_frequency"],
            "direction": direction,
            "family": str(row.get("family", "") or factor_family(name)),
            "coverage": float(ic.notna().mean()) if len(selection_dates) else 0.0,
            "mean_ic": float(ic.mean()) if not ic.empty else float("nan"),
            "ic_std": float(ic.std(ddof=1)) if len(ic) > 1 else float("nan"),
            "ic_pos_ratio": float(ic.gt(0.0).mean()) if not ic.empty else float("nan"),
            "segment_positive_ratio": float(np.mean(np.asarray(means) > 0.0)) if means else 0.0,
            "worst_segment_mean_ic": float(np.min(means)) if means else float("nan"),
            "median_segment_mean_ic": float(np.median(means)) if means else float("nan"),
            "rank_churn": float(prior.mean()) if not prior.empty else float("nan"),
        }
        if source_run_dir is not None:
            observed_oos_ic = row.get("oos_ic")
            row_out.update({
                "observed_oos_ic": observed_oos_ic,
                "observed_oos_same_direction": (
                    bool(float(observed_oos_ic) * direction > 0.0)
                    if observed_oos_ic is not None and pd.notna(observed_oos_ic)
                    else None
                ),
                "observed_oos_used_for_selection": False,
            })
        for idx, (mean, positive_ratio, ir) in enumerate(segment_values, 1):
            row_out[f"segment_{idx}_mean_ic"] = mean
            row_out[f"segment_{idx}_positive_ratio"] = positive_ratio
            row_out[f"segment_{idx}_ic_ratio"] = ir
        all_diagnostics.append(row_out)

    cluster_rows: list[dict] = []
    factor_sets: dict[str, dict] = {}
    correlation_files: dict[str, str] = {}
    if group_by_best_period or source_run_dir is not None:
        groups = [
            (f"h{horizon}", sorted(name for name in names if periods[name] == horizon))
            for horizon in selection_horizons
        ]
    else:
        groups = [("mixed", names)]
    for group, group_names in groups:
        corr = _exposure_correlation(ranks, group_names)
        corr_path = output / f"exposure_correlation_{group}.csv"
        corr.to_csv(corr_path, encoding="utf-8-sig")
        correlation_files[group] = corr_path.name
        clusters = _cluster(corr, group_names)
        rows_in_group = [
            row for row in all_diagnostics if str(row["factor"]) in set(group_names)
        ]
        rows_by_name = {str(row["factor"]): row for row in rows_in_group}
        for row in rows_in_group:
            row["cluster_id"] = int(clusters[str(row["factor"])])
        for name in group_names:
            cluster_rows.append({
                "group": group,
                "factor": name,
                "best_period": periods[name],
                "cluster_id": int(clusters[name]),
                "cluster_size": int(sum(value == clusters[name] for value in clusters.values())),
                "is_representative": False,
                "family": rows_by_name[name]["family"],
                "mean_ic": rows_by_name[name]["mean_ic"],
                "segment_positive_ratio": rows_by_name[name]["segment_positive_ratio"],
                "rank_churn": rows_by_name[name]["rank_churn"],
            })
        representatives: list[str] = []
        for cluster_id in sorted(set(clusters.values())):
            members = [
                rows_by_name[name] for name in group_names
                if clusters[name] == cluster_id
            ]
            chosen = max(members, key=_representative_key)
            representatives.append(str(chosen["factor"]))
            for row in cluster_rows:
                if row["group"] == group and row["cluster_id"] == cluster_id and row["factor"] == chosen["factor"]:
                    row["is_representative"] = True
        representatives = sorted(representatives, key=lambda name: _representative_key(rows_by_name[name]), reverse=True)
        factor_sets[f"balanced_core_{group}"] = {
            "group": group,
            "purpose": "all cluster representatives; portfolio-search candidate pool",
            "factors": representatives,
        }
        # Retain the old bounded view only for the explicit common-horizon
        # observation workflow. It is not a current strategy candidate.
        if source_run_dir is not None:
            compact = _compact_representatives(
                [rows_by_name[name] for name in representatives],
                int(max_compact_factors),
            )
            factor_sets[f"compact_core_{group}"] = {
                "group": group,
                "purpose": "legacy bounded common-horizon observation",
                "factors": compact,
            }

    aggregate_names = ["balanced_core"]
    if source_run_dir is not None:
        aggregate_names.append("compact_core")
    for name in aggregate_names:
        by_group = [
            factor_sets[f"{name}_{group}"]["factors"]
            for group, _ in groups
        ]
        factor_sets[name] = {
            "groups": {
                group: values
                for (group, _), values in zip(groups, by_group)
            },
            "factors": sorted(set().union(*map(set, by_group))),
            "purpose": (
                "cluster representatives; best_period is evidence only"
                if name == "balanced_core" else
                "legacy common-horizon observation only"
            ),
        }

    nested_path: list[dict] = []
    candidate_evaluations: list[dict] = []
    nested_stop_reason: str | None = None
    portfolio_search_seconds = 0.0
    extra_files: list[str] = []
    if source_run_dir is None and not group_by_best_period:
        portfolio_started = time.perf_counter()
        close = runner.data_manager.get("close", requested_dates, universe)
        daily_ret, close_tradable = runner.data_manager.prepare_close_data(close)
        full_portfolio_ic = rank_information_coefficients(
            full_ranks,
            daily_ret,
            minimum_cross_section=3,
        )
        portfolio_ic = full_portfolio_ic
        schedule = runner.data_manager.get_contract_schedule(
            requested_dates, universe
        )
        adapter = SimpleNamespace(
            cal=requested_dates,
            u=list(universe),
            ranks=full_ranks,
            ic=full_portfolio_ic,
            daily_ret=daily_ret,
            close_tradable=close_tradable,
            contract_schedule=schedule,
            env=SimpleNamespace(
                sector_of={name: SECTOR_MAP.get(name, "other") for name in universe}
            ),
        )
        recipe = _production_recipe(config)
        if (
            recipe.constraints.minimum_risk_observations
            != int(config.production_portfolio.minimum_risk_observations)
            or not np.isclose(
                recipe.constraints.covariance_shrinkage,
                float(config.production_portfolio.covariance_shrinkage),
            )
        ):
            raise ValueError(
                "PortfolioRecipe risk defaults no longer match production_portfolio"
            )
        evaluator = PortfolioEvaluator(
            adapter,
            start=selection_dates[0],
            end=selection_dates[-1],
            cost_model=_configured_cost_model(config),
            ic_window=int(config.production_portfolio.ic_window),
            risk_lookback_calendar_days=int(
                config.production_portfolio.risk_lookback_calendar_days
            ),
        )
        representatives = factor_sets["balanced_core_mixed"]["factors"]
        nested_path, candidate_evaluations, nested_stop_reason = (
            _run_nested_portfolio_search(
                evaluator=evaluator,
                portfolio_ic=portfolio_ic,
                representatives=representatives,
                recipe=recipe,
                segments=_segment_bounds(selection_dates),
            )
        )
        recommended = _recommended_nested_candidate(nested_path)
        factor_sets["recommended_core"] = {
            "purpose": (
                "smallest near-best robust member of the nested fixed-recipe path; "
                "membership still requires independent post-selection validation"
            ),
            "factor_count": int(recommended["factor_count"]),
            "factors": list(recommended["factors"]),
            "selection_metrics": {
                key: recommended[key]
                for key in (
                    "positive_segment_ratio", "worst_sharpe", "median_sharpe",
                    "median_annual_return", "worst_drawdown", "annual_turnover",
                    "full_annual_return", "full_sharpe", "full_max_drawdown",
                    "full_total_return",
                )
            },
        }
        factor_sets["nested_candidates"] = {
            "purpose": "complete nested path retained for independent validation",
            "stop_reason": nested_stop_reason,
            "candidates": [
                {
                    "factor_count": int(row["factor_count"]),
                    "factors": list(row["factors"]),
                    **{
                        key: row[key]
                        for key in (
                            "positive_segment_ratio", "worst_sharpe",
                            "median_sharpe", "median_annual_return",
                            "worst_drawdown", "annual_turnover", "full_sharpe",
                        )
                    },
                }
                for row in nested_path
            ],
        }
        portfolio_search_seconds = time.perf_counter() - portfolio_started
        for filename, rows_to_write in (
            ("nested_candidate_path.csv", nested_path),
            ("portfolio_candidate_evaluations.csv", candidate_evaluations),
        ):
            serializable = [
                {**row, "factors": "|".join(row["factors"])}
                for row in rows_to_write
            ]
            _write_csv(output / filename, serializable)
            extra_files.append(filename)

    _write_csv(output / "factor_diagnostics.csv", all_diagnostics)
    _write_csv(output / "factor_clusters.csv", cluster_rows)
    _write_json(output / "factor_sets.json", factor_sets)
    summary = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "workflow": "effective_factor_subset_selection",
        "library_path": str(library_path),
        "source_run_dir": str(Path(source_run_dir).resolve()) if source_run_dir else None,
        "selection_mode": (
            "common_horizon" if source_run_dir else
            "best_period_sleeves" if group_by_best_period else
            "mixed_daily_nested_portfolio"
        ),
        "common_horizon": int(common_horizon) if common_horizon is not None else None,
        "library_count": len(names),
        "data_source": config.data.source,
        "research_cutoff": cutoff.date().isoformat(),
        "warmup": [
            factor_start.date().isoformat(),
            (selection_start - pd.Timedelta(days=1)).date().isoformat(),
        ],
        "selection_sample": [
            selection_start.date().isoformat(),
            cutoff.date().isoformat(),
            len(selection_dates),
        ],
        "post_cutoff_data_used": False,
        "input_bar_frequencies": {
            frequency: sum(
                str(row["input_bar_frequency"]) == frequency
                for row in library_rows
            )
            for frequency in sorted({
                str(row["input_bar_frequency"]) for row in library_rows
            })
        },
        "signal_frequency": "daily",
        "horizon_unit": "daily bars / trading days",
        "horizon_counts": {
            str(horizon): sum(periods[name] == horizon for name in names)
            for horizon in selection_horizons
        },
        "best_period_role": "admission_evidence_only",
        "cluster_correlation": {"metric": "direction-adjusted daily cross-sectional rank exposure", "method": "complete_linkage", "threshold_abs_corr": CLUSTER_CORRELATION_THRESHOLD},
        "oos_used_for_selection": False,
        "portfolio_search": (
            {
                "enabled": True,
                "recipe": recipe.to_dict(),
                "ic_horizon": 1,
                "risk_lookback_calendar_days": int(
                    config.production_portfolio.risk_lookback_calendar_days
                ),
                "minimum_risk_observations": int(
                    config.production_portfolio.minimum_risk_observations
                ),
                "covariance_shrinkage": float(
                    config.production_portfolio.covariance_shrinkage
                ),
                "segments": 4,
                "initial_factor_count": 2,
                "final_factor_count_is_fixed": False,
                "maximum_explored_factor_count": PORTFOLIO_SEARCH_MAX_FACTORS,
                "exact_shortlist_width": PORTFOLIO_SEARCH_EXACT_WIDTH,
                "no_improvement_patience": PORTFOLIO_SEARCH_PATIENCE,
                "near_best_sharpe_tolerance": PORTFOLIO_SEARCH_SHARPE_TOLERANCE,
                "stop_reason": nested_stop_reason,
                "recommended_factor_count": int(
                    factor_sets["recommended_core"]["factor_count"]
                ),
            }
            if nested_path else {"enabled": False}
        ),
        "performance": {
            "factor_compute_seconds": factor_compute_seconds,
            "portfolio_search_seconds": portfolio_search_seconds,
            "factor_seconds": sum(float(row["seconds"]) for row in factor_timings),
            "shared_overhead_seconds": max(
                0.0,
                factor_compute_seconds
                - sum(float(row["seconds"]) for row in factor_timings),
            ),
            "factor_timings": sorted(
                factor_timings,
                key=lambda row: (-float(row["seconds"]), str(row["factor"])),
            ),
        },
        "factor_sets": {
            key: value for key, value in factor_sets.items()
            if key in {"balanced_core", "compact_core", "recommended_core"}
        },
        "correlation_files": correlation_files,
    }
    _write_json(output / "selection_summary.json", summary)
    files = [
        "factor_diagnostics.csv", "factor_clusters.csv", "factor_sets.json",
        "selection_summary.json", *correlation_files.values(), *extra_files,
    ]
    contract = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "run_id": str(run_id),
        "workflow": "effective-factor-subset-selection",
        "selection_contract": summary,
        "files": {name: {"sha256": sha256_file(output / name)} for name in files},
    }
    _write_json(output / "run_contract.json", contract)
    runner.factor_engine.clear_cache()
    return output
