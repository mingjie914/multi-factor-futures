"""Frozen effective-factor subset selection for the intraday daily contract.

This workflow consumes the current admitted effective-factor library (not a
fixed-size candidate pool), computes an independent long-history interval with
required warm-up, records daily-signal clusters, and evaluates multiple forward
search paths under the production construction recipe. It
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
from core.period import iter_overlapping_chunks
from core.registry import list_registered
from core.sectors import FRAMEWORK_UNIVERSE, portfolio_selection_group_for
from optimization.costs import SimpleFuturesCost
from optimization.factor_weighting import (
    rank_information_coefficients,
)
from pipeline.runner import PipelineRunner
from research.artifacts import sha256_file
from research.effective_factor_library import load_library
from research.governance import factor_family
from research.portfolio_experiment_support import FactorPanelRunner
from research.historical_portfolio_search import (
    PortfolioEvaluator,
    PortfolioRecipe,
    performance_metrics,
    robust_summary,
    robustness_key,
)


SELECTION_SCHEMA_VERSION = 4
CLUSTER_CORRELATION_THRESHOLD = 0.50
MIN_CROSS_SECTION = 10
N_IS_SEGMENTS = 4
COMPACT_MAX_FACTORS = 12
PORTFOLIO_SEARCH_MAX_FACTORS = 20
PORTFOLIO_SEARCH_EXACT_WIDTH = 3
PORTFOLIO_SEARCH_BEAM_WIDTH = 20
PORTFOLIO_CAP_GROUPS = {
    name: portfolio_selection_group_for(name) for name in FRAMEWORK_UNIVERSE
}


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




def _portfolio_shortlist(rows: list[dict], limit: int = 5) -> list[dict]:
    """Keep diverse, non-dominated net-performance candidates without size bias."""
    valid = [r for r in rows if r["status"] == "evaluated"
             and r["segment_count"] >= 4 and r["positive_segment_ratio"] >= 0.6
             and r["median_sharpe"] > 0 and r["full_annual_return"] > 0]
    def objectives(row):
        return (*robustness_key(row), -float(row["annual_turnover"]))
    frontier = [r for r in valid if not any(
        all(a >= b for a, b in zip(objectives(other), objectives(r)))
        and any(a > b for a, b in zip(objectives(other), objectives(r)))
        for other in valid if other is not r)]
    chosen = []
    for row in sorted(frontier, key=objectives, reverse=True):
        members = set(row["factors"])
        if all(len(members & set(r["factors"])) / len(members | set(r["factors"])) < 0.85
               for r in chosen):
            chosen.append(row)
        if len(chosen) == limit:
            break
    return chosen


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


def _run_portfolio_search(
    *, evaluator: PortfolioEvaluator, portfolio_ic: pd.DataFrame,
    representatives: list[str], recipe: PortfolioRecipe, segments: list[tuple],
    max_factors: int = PORTFOLIO_SEARCH_MAX_FACTORS,
    exact_width: int = PORTFOLIO_SEARCH_EXACT_WIDTH,
    beam_width: int = PORTFOLIO_SEARCH_BEAM_WIDTH,
    output: Path | None = None,
) -> tuple[list[dict], list[dict], str]:
    """Multi-path forward proposals; exact net portfolios judge every size.

    Historical blocks are development evidence conditional on today's admitted
    pool. They are not independent out-of-sample folds.
    """
    pool = sorted(set(representatives))
    if len(pool) < 2 or min(exact_width, beam_width) < 1 or max_factors < 2:
        raise ValueError("portfolio search requires at least two factors and positive widths")
    panel = portfolio_ic[pool]
    arrays = [panel.loc[left:right].to_numpy(dtype=float) for left, right in segments]
    positions = {name: i for i, name in enumerate(pool)}
    def proxy(members):
        ids = [positions[n] for n in members]
        means, sharpes = [], []
        for values in arrays:
            x = values[:, ids]
            count = np.isfinite(x).sum(axis=1)
            combined = np.divide(np.nansum(x, axis=1), count,
                                 out=np.full(len(x), np.nan), where=count > 0)
            combined = combined[np.isfinite(combined)]
            if len(combined) < 20:
                return (0.0, -10.0, -10.0)
            mean, std = combined.mean(), combined.std(ddof=1)
            means.append(mean)
            sharpes.append(mean / std * np.sqrt(252) if std > 0 else -10.0)
        return (float(np.mean(np.asarray(means) > 0)), float(min(sharpes)), float(np.median(sharpes)))
    beam = []
    preferred_parent = None
    evaluations, path = [], []
    limit = min(max_factors, len(pool))
    for size in range(2, limit + 1):
        proposals = (set(combinations(pool, 2)) if size == 2 else
                     {tuple(sorted((*members, n))) for members in beam for n in pool if n not in members})
        ranked = sorted(((proxy(m), m) for m in proposals), reverse=True)
        # Reserve one expansion of the previous exact winner. Merely sorting
        # its parent earlier would not affect the next global proxy ranking.
        if preferred_parent is not None:
            continuation = next((r for r in ranked if set(preferred_parent).issubset(r[1])), None)
            if continuation is not None:
                ranked.remove(continuation)
                ranked.insert(0, continuation)
        # Preserve multiple starting points; no permanent cluster representatives.
        beam = []
        for key, members in ranked:
            if all(len(set(members) & set(other)) / len(set(members) | set(other)) < 0.95
                   for other in beam):
                beam.append(members)
            if len(beam) >= beam_width:
                break
        if not beam:
            break
        exact_rows = []
        for members in beam:
            if any(len(set(members) & set(other["factors"])) /
                   len(set(members) | set(other["factors"])) >= 0.85 for other in exact_rows):
                continue
            started = time.perf_counter()
            row = dict(step=size, factor_count=size, added_factor="", factors=list(members),
                       status="rejected_runtime", error="", seconds=0.0,
                       segment_count=0, positive_segment_ratio=0.0, worst_sharpe=-10.0,
                       median_sharpe=-10.0, median_annual_return=-1.0, worst_drawdown=-1.0,
                       annual_turnover=float("nan"), full_annual_return=float("nan"),
                       full_sharpe=float("nan"), full_max_drawdown=float("nan"),
                       full_total_return=float("nan"))
            try:
                ledger = evaluator.ledger(members, recipe)
                if any(len(ledger.loc[left:right]) < 20 for left, right in segments):
                    raise RuntimeError("insufficient portfolio history in a declared block")
                metrics = performance_metrics(ledger["net_return"], initial_anchor=True)
                row.update(status="evaluated", **robust_summary(ledger, segments, initial_anchor=True),
                           annual_turnover=float(ledger["executed_traded_notional"].iloc[1:].mean() * 252),
                           full_annual_return=metrics["annual_return"], full_sharpe=metrics["sharpe"],
                           full_max_drawdown=metrics["max_drawdown"], full_total_return=metrics["total_return"])
                exact_rows.append(row)
            except RuntimeError as exc:
                row["error"] = str(exc)
            row["seconds"] = time.perf_counter() - started
            evaluations.append(row)
            if output is not None:
                _write_json(output / "portfolio_search_progress.json", evaluations)
            print(f"portfolio size={size} status={row['status']} seconds={row['seconds']:.1f}", flush=True)
            if len(exact_rows) >= exact_width:
                break
        if exact_rows:
            winner = max(exact_rows, key=robustness_key)
            path.append(winner)
            preferred_parent = tuple(winner["factors"])
            # Exact portfolio evidence influences the next size, while other
            # proxy paths survive even when three consecutive sizes deteriorate.
            beam = list(dict.fromkeys([tuple(r["factors"]) for r in
                        sorted(exact_rows, key=robustness_key, reverse=True)] + beam))[:beam_width]
    return path, evaluations, ("candidate_pool_exhausted" if limit == len(pool)
                              else f"declared_compute_budget_size_{limit}")


def _run_budgeted_pool_search(*, evaluator, portfolio_ic, pool, recipe, segments,
                              clusters, db, output, budget=512, beam_width=6,
                              round_width=16):
    """Explicit research-only multi-start search; never changes default selection.

    Every factor gets an exact seed-pair attempt. IC only orders proposals;
    exact net backtests choose the moving beam and the final Pareto archive.
    Growth lanes survive temporary deterioration. Size is not a stop rule.
    """
    names = sorted(set(pool))
    if len(names) < 2 or budget < len(names) or not 1 <= beam_width <= round_width:
        raise ValueError("budget must cover the full pool and widths must be positive")
    # IC at T contains T+1 returns. Drop the last selection-day label even if
    # the caller supplied a panel extending into the observation period.
    panel = portfolio_ic.loc[:evaluator.end, names].iloc[:-1]
    arrays = [panel.loc[left:right].to_numpy(dtype=float) for left, right in segments]
    positions, proxies = {n: i for i, n in enumerate(names)}, {}
    def proxy(members):
        if members not in proxies:
            means, sharpes = [], []
            for values in arrays:
                x = values[:, [positions[n] for n in members]]
                count = np.isfinite(x).sum(axis=1)
                y = np.divide(np.nansum(x, axis=1), count,
                              out=np.full(len(x), np.nan), where=count > 0)
                y = y[np.isfinite(y)]
                if len(y) < 20:
                    proxies[members] = (0.0, -10.0, -10.0)
                    break
                mean, std = y.mean(), y.std(ddof=1)
                means.append(mean)
                sharpes.append(mean / std * np.sqrt(252) if std > 0 else -10.0)
            else:
                proxies[members] = (float(np.mean(np.asarray(means) > 0)),
                                    float(min(sharpes)), float(np.median(sharpes)))
        return proxies[members]
    db.execute("""CREATE TABLE IF NOT EXISTS pool_results (
        signature VARCHAR PRIMARY KEY, result VARCHAR, dates DATE[],
        nav DOUBLE[], net_returns DOUBLE[])""")
    cached = {key: json.loads(value) for key, value in db.execute(
        "SELECT signature,result FROM pool_results").fetchall()}
    evaluations, visited, path = [], set(), []
    def evaluate(members, phase, iteration):
        members = tuple(sorted(members))
        if members in visited or len(evaluations) >= budget:
            return None
        key = json.dumps(members)
        row = cached.get(key)
        if row is None:
            started, cpu = time.perf_counter(), time.process_time()
            row = {"factors": list(members), "factor_count": len(members),
                   "phase": phase, "step": iteration, "status": "rejected_runtime",
                   "error": "", "segment_count": 0, "positive_segment_ratio": 0.0,
                   "worst_sharpe": -10.0, "median_sharpe": -10.0,
                   "median_annual_return": -1.0, "worst_drawdown": -1.0,
                   "annual_turnover": None, "full_annual_return": None}
            dates, nav, returns = [], [], []
            try:
                ledger = evaluator.ledger(members, recipe)
                if ledger.index.max() > evaluator.end:
                    raise AssertionError("selection ledger escaped the cutoff")
                metrics = performance_metrics(ledger["net_return"], initial_anchor=True)
                if any(len(ledger.loc[left:right]) < 20 for left, right in segments):
                    raise RuntimeError("insufficient history in a declared block")
                row.update(status="evaluated", **robust_summary(ledger, segments, initial_anchor=True),
                           annual_turnover=float(ledger["executed_traded_notional"].iloc[1:].mean()*252),
                           **{f"full_{k}": v for k, v in metrics.items()})
                dates, nav, returns = list(ledger.index.date), ledger["nav"].tolist(), ledger["net_return"].tolist()
            except (RuntimeError, ValueError) as exc:
                row["error"] = str(exc)
            row.update(seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu)
            db.execute("INSERT INTO pool_results VALUES (?,?,?,?,?)",
                       [key, json.dumps(row), dates, nav, returns])
            cached[key] = row
            evaluator.clear_transient_caches()
        visited.add(members)
        evaluations.append(row)
        _write_json(output / "pool_progress.json", {
            "attempted": len(evaluations), "budget": budget,
            "cached_total": len(cached), "phase": phase, "round": iteration,
            "evaluated": sum(r["status"] == "evaluated" for r in evaluations),
            "covered_factors": len({n for r in evaluations for n in r["factors"]})})
        print(f"pool {len(evaluations)}/{budget} {phase} {len(members)}f {row['status']}", flush=True)
        return row
    def choose(rows, count, existing=()):
        selected = list(existing)
        for row in sorted((r for r in rows if r and r["status"] == "evaluated"),
                          key=lambda r: (*robustness_key(r), -r["annual_turnover"]), reverse=True):
            members = tuple(row["factors"])
            if all(len(set(members)&set(other))/len(set(members)|set(other)) < .95 for other in selected):
                selected.append(members)
            if len(selected) >= count:
                break
        return selected
    pairs = sorted(combinations(names, 2), key=lambda m: (proxy(m), m), reverse=True)
    seeds = []
    for name in names:
        pair = next((m for m in pairs if name in m and clusters[m[0]] != clusters[m[1]]),
                    next(m for m in pairs if name in m))
        if pair not in seeds:
            seeds.append(pair)
    seed_rows = [evaluate(m, "seed_coverage", 0) for m in seeds]
    beam = choose(seed_rows, beam_width)
    iteration = 0
    while beam and len(evaluations) < budget:
        iteration += 1
        proposals, growth = set(), set()
        for parent in beam:
            added = {tuple(sorted((*parent, n))) for n in names if n not in parent} - visited
            proposals.update(added)
            if added:
                growth.add(max(added, key=lambda m: (proxy(m), m)))
            for old in parent:
                remainder = tuple(n for n in parent if n != old)
                if len(remainder) >= 2:
                    proposals.add(remainder)
                proposals.update(tuple(sorted((*remainder, n))) for n in names if n not in parent)
        proposals -= visited
        if not proposals:
            break
        ordered = sorted(growth, key=lambda m: (proxy(m), m), reverse=True)
        ordered += sorted(proposals-growth, key=lambda m: (proxy(m), m), reverse=True)
        rows = [evaluate(m, "grow" if m in growth else "add_delete_swap", iteration)
                for m in ordered[:round_width] if len(evaluations) < budget]
        grown = choose([r for r in rows if r and r["phase"] == "grow"], max(1, beam_width//2))
        beam = choose(rows, beam_width, grown)
        if rows and any(r and r["status"] == "evaluated" for r in rows):
            path.append(max((r for r in rows if r and r["status"] == "evaluated"), key=robustness_key))
    reason = "exact_backtest_budget_exhausted" if len(evaluations) >= budget else "reachable_proposals_exhausted"
    _write_json(output / "pool_selection.json", {
        "stop_reason": reason, "budget": budget, "attempted": len(evaluations),
        "scope": names, "shortlist": _portfolio_shortlist(evaluations), "path": path,
        "observation_used_for_selection": False})
    return path, evaluations, reason


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
    selection_start: str = "2016-03-31",
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
    selection_start = (factor_admission_start(config) if source_run_dir is not None
                       else pd.Timestamp(selection_start))
    cutoff = research_cutoff(config)
    warmup_days = int(config.validation_policy.warmup_days_by_frequency["daily"])
    factor_start = selection_start - pd.Timedelta(days=warmup_days)
    source_fingerprint = runner.data_manager.source.checkpoint_source_fingerprint(factor_start, cutoff)
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
    from factors.library.intraday import clear_transient_data_caches
    raw = {}
    for target_dates, chunk_dates in iter_overlapping_chunks(requested_dates, 400, 128):
        part = FactorPanelRunner._compute_part(
            runner.factor_engine, names, chunk_dates, universe, raw
        )
        for name, frame in part.items():
            if name not in raw:
                raw[name] = pd.DataFrame(np.nan, index=requested_dates, columns=universe)
            raw[name].loc[target_dates] = frame.reindex(index=target_dates, columns=universe)
        runner.factor_engine.clear_cache()
        clear_transient_data_caches()
        print(f"factor panel through {target_dates[-1].date()}: {len(raw)}/{len(names)}", flush=True)
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
                sector_of={name: PORTFOLIO_CAP_GROUPS[name] for name in universe}
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
        # Clusters describe redundancy, but no member is permanently excluded.
        representatives = names
        history_segments = [
            (max(selection_dates[0], pd.Timestamp(left)), min(cutoff, pd.Timestamp(right)))
            for left, right in (("2016-03-31", "2019-12-31"), ("2020-01-01", "2021-12-31"),
                                ("2022-01-01", "2023-12-31"), ("2024-01-01", "2024-12-31"),
                                ("2025-01-01", str(cutoff.date())))
            if pd.Timestamp(right) >= selection_dates[0] and pd.Timestamp(left) <= cutoff
        ]
        nested_path, candidate_evaluations, nested_stop_reason = (
            _run_portfolio_search(
                evaluator=evaluator,
                portfolio_ic=portfolio_ic,
                representatives=representatives,
                recipe=recipe,
                segments=history_segments,
                output=output,
            )
        )
        shortlist = _portfolio_shortlist(candidate_evaluations)
        factor_sets["portfolio_candidates"] = {
            "purpose": "diverse Pareto candidates; historical development evidence only",
            "candidates": shortlist,
            "qualified_count": len(shortlist),
        }
        candidate_returns = pd.DataFrame({
            f"candidate_{i}_{row['factor_count']}f": evaluator.ledger(row["factors"], recipe)["net_return"]
            for i, row in enumerate(shortlist, 1)
        })
        if not candidate_returns.empty:
            candidate_returns.to_csv(output / "candidate_returns.csv")
            candidate_returns.corr().to_csv(output / "candidate_return_correlation.csv")
            extra_files.extend(["candidate_returns.csv", "candidate_return_correlation.csv"])
            block_metrics = []
            for i, row in enumerate(shortlist, 1):
                ledger = evaluator.ledger(row["factors"], recipe)
                for left, right in history_segments:
                    values = ledger.loc[left:right, "net_return"]
                    block_metrics.append({
                        "candidate": f"candidate_{i}_{row['factor_count']}f",
                        "start": str(left.date()), "end": str(right.date()),
                        **performance_metrics(values, initial_anchor=values.index[0] == ledger.index[0]),
                    })
            _write_csv(output / "candidate_block_metrics.csv", block_metrics)
            extra_files.append("candidate_block_metrics.csv")
            from run_portfolio_workflow import _write_comparison_plot
            labels = {name: f"候选{i}（{row['factor_count']}因子）"
                      for i, (name, row) in enumerate(zip(candidate_returns, shortlist), 1)}
            plot_rows = []
            for name, row in zip(candidate_returns, shortlist):
                metrics = performance_metrics(candidate_returns[name], initial_anchor=True)
                plot_rows.append({"strategy": labels[name], **metrics,
                                  "volatility": metrics["annual_volatility"],
                                  "annualized_turnover": row["annual_turnover"]})
            _write_comparison_plot(
                output, (1 + candidate_returns).cumprod().rename(columns=labels), plot_rows,
                title=f"候选组合净值对比（{selection_dates[0]:%Y-%m-%d}至{cutoff:%Y-%m-%d}）\n历史开发样本；统一组合配方与成本，非独立样本外",
            )
            extra_files.append("nav_comparison.png")
        recommended = shortlist[0] if shortlist else None
        factor_sets["recommended_core"] = {
            "purpose": (
                "first robust candidate for display only; no automatic promotion; "
                "membership still requires independent post-selection validation"
            ),
            "factor_count": int(recommended["factor_count"]) if recommended else 0,
            "factors": list(recommended["factors"]) if recommended else [],
            "selection_metrics": ({
                key: recommended[key]
                for key in (
                    "positive_segment_ratio", "worst_sharpe", "median_sharpe",
                    "median_annual_return", "worst_drawdown", "annual_turnover",
                    "full_annual_return", "full_sharpe", "full_max_drawdown",
                    "full_total_return",
                )
            } if recommended else {}),
        }
        factor_sets["size_candidates"] = {
            "purpose": "best exact candidate at every explored size; paths need not be nested",
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
            ("size_candidate_path.csv", nested_path),
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
            "mixed_daily_multipath_portfolio"
        ),
        "common_horizon": int(common_horizon) if common_horizon is not None else None,
        "library_count": len(names),
        "data_source": config.data.source,
        "data_sha256": source_fingerprint,
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
        "historical_evidence_role": "development_stability_conditional_on_current_admitted_pool",
        "admission_window": [str(factor_admission_start(config).date()), str(cutoff.date())],
        "library_sha256": sha256_file(library_path),
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
                "cap_groups": {name: PORTFOLIO_CAP_GROUPS[name] for name in universe},
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
                "segments": [[str(a.date()), str(b.date())] for a, b in history_segments],
                "initial_factor_count": 2,
                "final_factor_count_is_fixed": False,
                "maximum_explored_factor_count": PORTFOLIO_SEARCH_MAX_FACTORS,
                "exact_shortlist_width": PORTFOLIO_SEARCH_EXACT_WIDTH,
                "beam_width": PORTFOLIO_SEARCH_BEAM_WIDTH,
                "size_penalty": False,
                "early_stop_on_nonimprovement": False,
                "proxy_role": "H1 IC proposal only; exact net portfolios determine shortlist",
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
            if key in {"balanced_core", "compact_core", "recommended_core", "portfolio_candidates"}
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
