"""Frozen effective-factor subset selection for the intraday daily contract.

This workflow consumes the current admitted effective-factor library (not a
fixed-size candidate pool), computes only the locked IS window (with its
required warm-up), clusters all daily signals together by default, and
writes durable diagnostics plus parallel factor sets. It never mutates the
effective library and never uses post-cutoff data for selection. The input
count is always discovered from the configured effective library path, so a
future library version is handled without changing this workflow.
"""
from __future__ import annotations

import csv
import json
import math
import re
import time
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
from research.artifacts import sha256_file
from research.effective_factor_library import load_library
from research.governance import factor_family


SELECTION_SCHEMA_VERSION = 2
CLUSTER_CORRELATION_THRESHOLD = 0.50
MIN_CROSS_SECTION = 10
N_IS_SEGMENTS = 3
COMPACT_MAX_FACTORS = 12


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
    factor_compute_started = time.perf_counter()
    raw = runner.factor_engine.compute_factors(
        names, requested_dates, universe, parallel=False, chunk_size=64
    )
    factor_compute_seconds = time.perf_counter() - factor_compute_started
    factor_timings = list(runner.factor_engine.computation_timings)
    if set(raw) != set(names):
        raise ValueError("factor engine did not return all admitted factor names")
    processed = runner.processor.process_batch(raw, context)
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
            "signal_frequency": row["signal_frequency"],
            "direction": direction,
            "family": str(row.get("family", "") or factor_family(name)),
            "coverage": float(ic.notna().mean()) if len(is_dates) else 0.0,
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
        compact = _compact_representatives(
            [rows_by_name[name] for name in representatives], int(max_compact_factors)
        )
        factor_sets[f"balanced_core_{group}"] = {
            "group": group,
            "purpose": "all cluster representatives; no OOS ranking",
            "factors": representatives,
        }
        factor_sets[f"compact_core_{group}"] = {
            "group": group,
            "purpose": "bounded strongest distinct-cluster representatives",
            "factors": compact,
        }

    for name in ("balanced_core", "compact_core"):
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
            "purpose": "mixed daily factor set; best_period is evidence only",
        }

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
            "mixed_daily"
        ),
        "common_horizon": int(common_horizon) if common_horizon is not None else None,
        "library_count": len(names),
        "data_source": config.data.source,
        "research_cutoff": cutoff.date().isoformat(),
        "warmup": [window.factor_start.date().isoformat(), (window.is_start - pd.Timedelta(days=1)).date().isoformat()],
        "selection_sample": [
            window.is_start.date().isoformat(),
            window.is_end.date().isoformat(),
            window.is_bars,
        ],
        "excluded_post_selection_observation": [
            window.oos_start.date().isoformat(),
            window.oos_end.date().isoformat(),
            window.oos_bars,
        ],
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
        "performance": {
            "factor_compute_seconds": factor_compute_seconds,
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
        "files": {name: {"sha256": sha256_file(output / name)} for name in files},
    }
    _write_json(output / "run_contract.json", contract)
    runner.factor_engine.clear_cache()
    return output
