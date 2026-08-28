"""Frozen effective-factor subset selection for the intraday daily contract.

This workflow consumes the current admitted effective-factor library (not a
fixed-size candidate pool), computes only the locked IS window (with its
required warm-up), clusters factors within their approved forward horizon, and
writes durable diagnostics plus parallel factor sets. It never mutates the
effective library and never uses post-cutoff data for selection. The input
count is always discovered from the configured effective library path, so a
future library version is handled without changing this workflow.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from core.config import load_config
from core.date_policy import factor_validation_window, research_cutoff
from core.registry import list_registered
from factors.processor import build_processing_context
from pipeline.runner import PipelineRunner


SELECTION_SCHEMA_VERSION = 1
HORIZONS = (5, 10, 20)
CLUSTER_CORRELATION_THRESHOLD = 0.50
MIN_CROSS_SECTION = 10
N_IS_SEGMENTS = 3
COMPACT_MAX_PER_HORIZON = 12


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _factor_family(name: str) -> str:
    """Small transparent family taxonomy used for diagnostics, not hard gates."""
    text = str(name).lower()
    families = (
        ("timing_session", ("overnight", "gap", "opening", "seasonality", "early_late", "time_")),
        ("liquidity_flow", ("volume", "turnover", "amihud", "kyle", "spread", "impact", "flow", "torrent", "signed_volume", "micro_leverage")),
        ("risk_distribution", ("vol", "variance", "cvar", "skew", "kurt", "tail", "parkinson", "semivariance", "quarticity", "jump", "amplitude", "smile", "dispersion")),
        ("price_trend_shape", ("momentum", "trend", "slope", "hurst", "run", "breakout", "peak", "ridge", "drip", "stone", "candle", "position", "delay", "reversal")),
        ("dependence_structure", ("corr", "covariance", "coupling", "elasticity", "entropy", "autocorr", "partial", "herding")),
        ("market_participation", ("participation", "order", "large_order", "overconfidence", "positioning")),
    )
    for family, tokens in families:
        if any(token in text for token in tokens):
            return family
    return "intraday_other"


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
    if len(rows) <= max_count:
        return [str(row["factor"]) for row in rows]
    ranked = sorted(rows, key=_representative_key, reverse=True)
    selected: list[dict] = []
    family_counts: dict[str, int] = {}
    # Preserve family breadth first; then fill by the same transparent score.
    for row in ranked:
        family = str(row.get("family", "intraday_other"))
        if family_counts.get(family, 0) >= 2:
            continue
        selected.append(row)
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(selected) >= max_count:
            break
    if len(selected) < max_count:
        chosen = {row["factor"] for row in selected}
        selected.extend(row for row in ranked if row["factor"] not in chosen)
    return [str(row["factor"]) for row in selected[:max_count]]


def _load_library(
    config,
    *,
    source_run_dir: str | None = None,
    allowed_horizons: tuple[int, ...] = HORIZONS,
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
        if horizon_mode != "common_horizon" or horizon not in allowed_horizons:
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
                "frequency": "daily_intraday",
                "registered_horizons": str(record.get("registered_horizons", "")),
                "selected_period": horizon,
                "approved_periods": [horizon],
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
    payload = json.loads(path.read_text(encoding="utf-8"))
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
        periods = tuple(int(value) for value in row.get("approved_periods", []))
        if len(periods) != 1 or periods[0] not in allowed_horizons:
            raise ValueError(
                f"factor {row.get('factor')!r} has ambiguous approved periods {periods}"
            )
        if str(row.get("frequency")) != "daily_intraday":
            raise ValueError(
                f"factor {row.get('factor')!r} is not daily_intraday: {row.get('frequency')!r}"
            )
    return path.resolve(), factors


def run_effective_factor_selection(
    *,
    run_id: str,
    config_path: str = "config/default.yaml",
    max_compact_per_horizon: int = COMPACT_MAX_PER_HORIZON,
    source_run_dir: str | None = None,
    common_horizon: int | None = None,
) -> Path:
    """Run governed effective-library subset selection and write one immutable run."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(run_id)):
        raise ValueError("run_id only allows letters, numbers, dot, underscore and hyphen")
    if int(max_compact_per_horizon) < 1:
        raise ValueError("max_compact_per_horizon must be positive")
    if source_run_dir is not None:
        if common_horizon is None or int(common_horizon) < 1:
            raise ValueError(
                "common-horizon selection requires a positive common_horizon"
            )
        selection_horizons = (int(common_horizon),)
    else:
        selection_horizons = HORIZONS

    project = Path(__file__).resolve().parents[1]
    output = project / "runs" / "factor_selection" / str(run_id)
    output.mkdir(parents=True, exist_ok=False)

    config = load_config(config_path)
    library_path, library_rows = _load_library(
        config,
        source_run_dir=source_run_dir,
        allowed_horizons=selection_horizons,
    )
    runner = PipelineRunner(config=config)
    window = factor_validation_window(
        config, runner.data_manager, frequency="daily_intraday"
    )
    cutoff = research_cutoff(config)
    if window.oos_end != cutoff:
        raise ValueError(
            f"selection window ends {window.oos_end.date()}, not research cutoff {cutoff.date()}"
        )
    is_dates = pd.DatetimeIndex(
        runner.data_manager.get_calendar(window.is_start, window.is_end)
    )
    requested_dates = pd.DatetimeIndex(
        runner.data_manager.get_calendar(window.factor_start, window.is_end)
    )
    universe = pd.Index(config.universe)
    names = sorted(str(row["factor"]) for row in library_rows)
    context = build_processing_context(
        runner.data_manager,
        requested_dates,
        universe,
        config.universe_selection,
    )
    raw = runner.factor_engine.compute_factors(
        names, requested_dates, universe, parallel=False, chunk_size=64
    )
    if set(raw) != set(names):
        raise ValueError("factor engine did not return all admitted factor names")
    processed = runner.processor.process_batch(raw, context)
    registry = list_registered("factor").get("factor", {})
    missing_registry = sorted(set(names) - set(registry))
    if missing_registry:
        raise ValueError("effective factors are not registered: " + ", ".join(missing_registry))
    directions = {str(row["factor"]): int(row["direction"]) for row in library_rows}
    periods = {str(row["factor"]): int(row["approved_periods"][0]) for row in library_rows}

    returns = {
        horizon: runner.data_manager.get_forward_returns(
            requested_dates, universe, period=horizon
        )
        for horizon in selection_horizons
    }
    segments = _segments(is_dates)
    all_diagnostics: list[dict] = []
    oriented_ic: dict[str, pd.Series] = {}
    ranks: dict[str, pd.DataFrame] = {}
    for row in library_rows:
        name = str(row["factor"])
        horizon = periods[name]
        frame = processed[name].loc[is_dates]
        direction = directions[name]
        directed = frame * float(direction)
        ranks[name] = _rank_frame(directed)
        ic = _daily_spearman_ic(directed, returns[horizon].loc[is_dates], is_dates)
        oriented_ic[name] = ic
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
            "frequency": row["frequency"],
            "direction": direction,
            "family": _factor_family(name),
            "coverage": float(ic.notna().mean()) if len(is_dates) else 0.0,
            "mean_ic": float(ic.mean()) if not ic.empty else float("nan"),
            "ic_std": float(ic.std(ddof=1)) if len(ic) > 1 else float("nan"),
            "ic_pos_ratio": float(ic.gt(0.0).mean()) if not ic.empty else float("nan"),
            "segment_positive_ratio": float(np.mean(np.asarray(means) > 0.0)) if means else 0.0,
            "worst_segment_mean_ic": float(np.min(means)) if means else float("nan"),
            "median_segment_mean_ic": float(np.median(means)) if means else float("nan"),
            "rank_churn": float(prior.mean()) if not prior.empty else float("nan"),
            "oos_ic": row.get("oos_ic"),
            "oos_same_direction": bool(row.get("oos_ic", 0.0) * direction > 0.0),
            "oos_gate_only": True,
        }
        for idx, (mean, positive_ratio, ir) in enumerate(segment_values, 1):
            row_out[f"segment_{idx}_mean_ic"] = mean
            row_out[f"segment_{idx}_positive_ratio"] = positive_ratio
            row_out[f"segment_{idx}_ic_ratio"] = ir
        all_diagnostics.append(row_out)

    diagnostics_by_horizon: dict[int, list[dict]] = {}
    cluster_rows: list[dict] = []
    factor_sets: dict[str, dict] = {}
    correlation_files: dict[str, str] = {}
    for horizon in selection_horizons:
        names_h = sorted(name for name in names if periods[name] == horizon)
        corr = _exposure_correlation(ranks, names_h)
        corr_path = output / f"exposure_correlation_h{horizon}.csv"
        corr.to_csv(corr_path, encoding="utf-8-sig")
        correlation_files[str(horizon)] = corr_path.name
        clusters = _cluster(corr, names_h)
        rows_h = [row for row in all_diagnostics if int(row["horizon"]) == horizon]
        rows_by_name = {str(row["factor"]): row for row in rows_h}
        for row in rows_h:
            row["cluster"] = int(clusters[str(row["factor"])])
        diagnostics_by_horizon[horizon] = rows_h
        for name in names_h:
            cluster_rows.append({
                "horizon": horizon,
                "factor": name,
                "cluster": int(clusters[name]),
                "cluster_size": int(sum(value == clusters[name] for value in clusters.values())),
                "is_representative": False,
                "family": rows_by_name[name]["family"],
                "mean_ic": rows_by_name[name]["mean_ic"],
                "segment_positive_ratio": rows_by_name[name]["segment_positive_ratio"],
                "rank_churn": rows_by_name[name]["rank_churn"],
            })
        representatives: list[str] = []
        for cluster_id in sorted(set(clusters.values())):
            members = [rows_by_name[name] for name in names_h if clusters[name] == cluster_id]
            chosen = max(members, key=_representative_key)
            representatives.append(str(chosen["factor"]))
            for row in cluster_rows:
                if row["horizon"] == horizon and row["cluster"] == cluster_id and row["factor"] == chosen["factor"]:
                    row["is_representative"] = True
        representatives = sorted(representatives, key=lambda name: _representative_key(rows_by_name[name]), reverse=True)
        compact = _compact_representatives(
            [rows_by_name[name] for name in representatives], int(max_compact_per_horizon)
        )
        factor_sets[f"balanced_core_h{horizon}"] = {
            "horizon": horizon,
            "purpose": "all same-horizon cluster representatives; no OOS ranking",
            "factors": representatives,
        }
        factor_sets[f"compact_core_h{horizon}"] = {
            "horizon": horizon,
            "purpose": "bounded family-diverse representative subset",
            "factors": compact,
        }

    for name, spec in ("balanced_core", "balanced_core_h"), ("compact_core", "compact_core_h"):
        by_horizon = [
            factor_sets[f"{name}_h{horizon}"]["factors"]
            for horizon in selection_horizons
        ]
        factor_sets[name] = {
            "horizons": {
                str(horizon): values
                for horizon, values in zip(selection_horizons, by_horizon)
            },
            "factors": sorted(set().union(*map(set, by_horizon))),
            "purpose": "parallel factor set; horizon assignment remains authoritative",
        }

    _write_csv(output / "factor_diagnostics.csv", all_diagnostics)
    _write_csv(output / "factor_clusters.csv", cluster_rows)
    _write_json(output / "factor_sets.json", factor_sets)
    summary = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "workflow": "effective_factor_subset_selection",
        "library_path": str(library_path),
        "source_run_dir": str(Path(source_run_dir).resolve()) if source_run_dir else None,
        "selection_mode": "common_horizon" if source_run_dir else "effective_library",
        "common_horizon": int(common_horizon) if common_horizon is not None else None,
        "library_count": len(names),
        "data_source": config.data.source,
        "research_cutoff": cutoff.date().isoformat(),
        "warmup": [window.factor_start.date().isoformat(), (window.is_start - pd.Timedelta(days=1)).date().isoformat()],
        "is": [window.is_start.date().isoformat(), window.is_end.date().isoformat(), window.is_bars],
        "oos": [window.oos_start.date().isoformat(), window.oos_end.date().isoformat(), window.oos_bars],
        "factor_frequency": "daily_intraday (1min-derived daily output)",
        "horizon_unit": "daily bars / trading days",
        "horizon_counts": {
            str(horizon): sum(periods[name] == horizon for name in names)
            for horizon in selection_horizons
        },
        "cluster_correlation": {"metric": "direction-adjusted daily cross-sectional rank exposure", "method": "complete_linkage", "threshold_abs_corr": CLUSTER_CORRELATION_THRESHOLD},
        "oos_used_for_selection": False,
        "factor_sets": {key: value for key, value in factor_sets.items() if key in {"balanced_core", "compact_core"}},
        "correlation_files": correlation_files,
    }
    _write_json(output / "selection_summary.json", summary)
    files = [
        "factor_diagnostics.csv", "factor_clusters.csv", "factor_sets.json",
        "selection_summary.json", *correlation_files.values(),
    ]
    contract = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "run_id": str(run_id),
        "workflow": "effective-factor-subset-selection",
        "selection_contract": summary,
        "files": {name: {"sha256": _sha256(output / name)} for name in files},
    }
    _write_json(output / "run_contract.json", contract)
    runner.factor_engine.clear_cache()
    return output
