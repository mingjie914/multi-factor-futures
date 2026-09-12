"""IDE entrypoint for durable single/multi-strategy backtests and comparison.

The readable strategy library names parallel factor subsets and strategies;
each strategy points to a method YAML; shared-default subsets live in the catalog.
``RUN_PREFERRED`` runs the unique preferred observation strategy by default.
``RUN_AND_COMPARE`` compares active peers: only the factor set varies and every
selected strategy uses the single configured production recipe. The explicit
``RUN_AND_COMPARE_CONFIGURED`` branch is reserved for deliberate
model/optimizer comparisons from each strategy YAML.
"""
from __future__ import annotations

import hashlib
import json
import os
import runpy
import shutil
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from backtest.metrics import TRADING_DAYS_PER_YEAR, compute_all_metrics, compute_turnover_metrics
from core.config import ProcessingStepConfig, load_config, load_strategy_library
from core.date_policy import research_cutoff
from pipeline.runner import PipelineRunner
from research.artifacts import sha256_file
from research.effective_factor_library import (
    effective_factor_names,
    validate_effective_factor_membership,
)


class PortfolioWorkflow(Enum):
    VALIDATE_CONFIGURATIONS = "validate_configurations"
    RUN_PREFERRED = "run_preferred"
    # Default comparison route: only the factor subset varies; every selected
    # strategy uses config/default.yaml::production_portfolio.
    RUN_AND_COMPARE = "run_and_compare"
    # Explicit challenger route: honor each strategy YAML's configured model,
    # risk and optimizer instead of the production recipe.
    RUN_AND_COMPARE_CONFIGURED = "run_and_compare_configured"
    # Explicitly rerun archived 6f/8f/13f definitions through the current
    # production ledger without adding them to ordinary peer comparisons.
    RUN_AND_COMPARE_SNAPSHOT_AUDIT = "run_and_compare_snapshot_audit"
    # Compare non-archived catalog peers under the same production recipe.
    RUN_AND_COMPARE_ALL = "run_and_compare_all"
    # Explicit research-pool comparison: common-H5 passed factors are routed
    # through the same default production recipe as the ordinary peers.  This
    # is independent evidence and does not require admitting them to the
    # effective library.
    RUN_AND_COMPARE_COMMON_H5 = "run_and_compare_common_h5"
    # Same five-bar evidence with a five-bar IC used by the production
    # weighting history.  This is a sensitivity comparison, not the default.
    RUN_AND_COMPARE_COMMON_H5_MATCHED = "run_and_compare_common_h5_matched"
    # Training-only factor-membership search under the fixed production recipe;
    # the locked OOS is diagnostic and never changes the chosen set.
    SEARCH_COMMON_H5_FACTOR_SET = "search_common_h5_factor_set"
    RUN_AND_COMPARE_COMMON_H5_SEARCH = "run_and_compare_common_h5_search"


# ============================== IDE SETTINGS ==============================
WORKFLOW = PortfolioWorkflow.RUN_PREFERRED

CATALOG_PATH = "config/strategy_library.yaml"
RUN_ID: str | None = None
# Default to the framework observation start; assign an explicit date here
# only for a deliberately bounded comparison. Admission dates are independent.
COMPARISON_START = str(load_config("config/default.yaml").date_range.start)
# The legacy observation route uses the production-style research ledger and
# its own availability checks.  It must not be silently converted into the
# generic model pipeline or patched with a comparison-only fill operation.
LEGACY_COMPARISON_FILLNA_ZERO = False
# The production-style evaluator needs enough prior daily bars for the 60-day
# IC history, 90-day risk window and intraday rolling features.  This is panel
# preheat only; the reported comparison still begins at COMPARISON_START.
LEGACY_PANEL_BUFFER_DAYS = 365
# Empty means all non-archived catalog entries.  During a controlled peer
# comparison this can name a small parallel set without mutating the catalog.
STRATEGY_IDS: tuple[str, ...] = ()
# Archived factor snapshots are only admitted by the explicit audit branch.
SNAPSHOT_AUDIT_IDS: tuple[str, ...] = (
    "snapshot_6f_icir",
    "snapshot_8f_icir",
    "snapshot_13f_icir",
)
# Empty selects every non-archived peer. Archived 6f remains audit-only.
ALL_STRATEGY_IDS: tuple[str, ...] = ()
# RUN_AND_COMPARE_COMMON_H5: the source selection run is immutable evidence;
# the comparison uses COMPARISON_START and observes through the latest source
# date, exactly like other confirmed-combination backtests.
COMMON_H5_SELECTION_RUN_DIR = (
    "runs/factor_selection/20260826_common_h5_subset_selection"
)
COMMON_H5_COMPARISON_RUN_ID: str | None = None
COMMON_H5_MATCHED_COMPARISON_RUN_ID: str | None = None
# SEARCH_COMMON_H5_FACTOR_SET consumes the finalized 50-factor H5 validation
# evidence and the existing cluster baselines.  It never mutates either library.
COMMON_H5_SEARCH_RUN_ID: str | None = None
COMMON_H5_VALIDATION_RUN_DIR = (
    "runs/factor_validation/20260826_intraday588_common_h5_is126_oos42_cutoff_20260515"
)
COMMON_H5_SEARCH_BASELINE_DIR = (
    "runs/factor_selection/20260903_common_h5_subset_selection_native_liq_events"
)
COMMON_H5_FACTOR_SEARCH_RUN_DIR = (
    "runs/portfolio_factor_search/20260903_common_h5_combination_search_v2"
)
COMMON_H5_SEARCH_COMPARISON_RUN_ID: str | None = None
# Keep 5 for selection/deployment horizon consistency; set 1 only for the
# explicit default-IC sensitivity control.  The factor set remains frozen.
COMMON_H5_SEARCH_IC_HORIZON = 5
# ========================================================================


PROJECT_ROOT = Path(__file__).resolve().parent


def _peak_working_set_mib() -> float | None:
    """Return this process's peak RSS on Windows without an extra dependency."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("page_fault_count", wintypes.DWORD),
            ("peak_working_set_size", ctypes.c_size_t),
            ("working_set_size", ctypes.c_size_t),
            *(  # Remaining fields are required by the Windows structure.
                (name, ctypes.c_size_t)
                for name in (
                    "quota_peak_paged_pool_usage", "quota_paged_pool_usage",
                    "quota_peak_non_paged_pool_usage", "quota_non_paged_pool_usage",
                    "pagefile_usage", "peak_pagefile_usage",
                )
            ),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    query = kernel.K32GetProcessMemoryInfo
    query.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    query.restype = wintypes.BOOL
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not query(
        kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        return None
    return counters.peak_working_set_size / (1024.0 * 1024.0)

STRATEGY_LABELS = {
    "current_single_baseline": "旧10因子观察策略",
    "intraday_balanced_ridge": "日内平衡因子集",
    "intraday_compact_ridge": "日内紧凑因子集",
    "snapshot_6f_icir": "历史 6f 因子集",
    "snapshot_8f_icir": "历史 8f 因子集",
    "snapshot_13f_icir": "历史 13f 因子集",
    "common_h5_balanced": "共同H5平衡因子集",
    "common_h5_compact": "共同H5紧凑因子集",
    "common_h5_combination": "共同H5组合互补因子集",
}


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return (candidate if candidate.is_absolute() else PROJECT_ROOT / candidate).resolve()


def _load_factor_definition(path: Path) -> dict:
    """Load an immutable snapshot definition without treating it as an entrypoint."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"factor definition not found: {path}")
    payload = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.suffix.lower() == ".json"
        else runpy.run_path(str(path))
    )
    raw_factors = payload.get("FACTORS")
    if not isinstance(raw_factors, dict) or not raw_factors:
        raise ValueError(f"snapshot factor definition must expose FACTORS: {path}")
    factors = [str(name) for name in raw_factors]
    directions = {}
    for name, direction in raw_factors.items():
        direction = int(direction)
        if direction not in {-1, 1}:
            raise ValueError(f"invalid direction for {name!r} in {path}")
        directions[str(name)] = direction
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "factors": factors,
        "directions": directions,
    }


def _assignments(config, mode: str) -> dict[int, list[str]]:
    if mode == "single":
        return {int(config.backtest.holding_period): list(config.factors)}
    assignments: dict[int, list[str]] = {}
    if not config.sub_portfolios:
        raise ValueError("multi strategy requires non-empty sub_portfolios")
    for sleeve in config.sub_portfolios:
        if not sleeve.factors:
            raise ValueError(f"sub-portfolio {sleeve.name!r} has no factors")
        assignments.setdefault(int(sleeve.holding_period), []).extend(sleeve.factors)
    return assignments


def _validated_specs():
    catalog_path = _resolve(CATALOG_PATH)
    catalog = load_strategy_library(catalog_path)
    factor_sets = {entry.id: entry for entry in catalog.factor_sets}
    library_path = _resolve(catalog.effective_factor_library)
    effective_names = set(effective_factor_names(library_path))
    snapshot_audit = WORKFLOW is PortfolioWorkflow.RUN_AND_COMPARE_SNAPSHOT_AUDIT
    all_strategy_compare = WORKFLOW is PortfolioWorkflow.RUN_AND_COMPARE_ALL
    audit_workflow = snapshot_audit or all_strategy_compare
    default_production = (
        _default_production_config().production_portfolio
        if WORKFLOW in {
            PortfolioWorkflow.VALIDATE_CONFIGURATIONS,
            PortfolioWorkflow.RUN_PREFERRED,
            PortfolioWorkflow.RUN_AND_COMPARE,
            PortfolioWorkflow.RUN_AND_COMPARE_SNAPSHOT_AUDIT,
            PortfolioWorkflow.RUN_AND_COMPARE_ALL,
        }
        else None
    )
    for factor_set in catalog.factor_sets:
        if factor_set.status != "active":
            continue
        unknown = sorted(set(factor_set.factors) - effective_names)
        if unknown:
            raise ValueError(
                f"factor set {factor_set.id!r} contains non-effective factors: {unknown}"
            )
    validated = []
    requested_ids = (
        set(ALL_STRATEGY_IDS)
        if all_strategy_compare
        else
        set(SNAPSHOT_AUDIT_IDS)
        if snapshot_audit
        else {str(value) for value in STRATEGY_IDS}
    )
    if WORKFLOW is PortfolioWorkflow.RUN_PREFERRED and not requested_ids:
        requested_ids = {entry.id for entry in catalog.strategies if entry.status == "preferred"}
        if len(requested_ids) != 1:
            raise ValueError("default portfolio requires exactly one preferred strategy")
    STRATEGY_LABELS.update({entry.id: entry.name for entry in catalog.strategies if entry.name})
    seen_ids: set[str] = set()
    for strategy in catalog.strategies:
        seen_ids.add(strategy.id)
        if strategy.status == "archived" and not (
            audit_workflow and strategy.id in requested_ids
        ):
            continue
        if requested_ids and strategy.id not in requested_ids:
            continue
        path = _resolve(strategy.config_path)
        config = load_config(path)
        if strategy.source == "effective_library" and path == _resolve("config/default.yaml"):
            # Shared method config; the catalog owns each reusable factor subset.
            factor_set = factor_sets[strategy.factor_set_id]
            config.factors = list(factor_set.factors)
            config.factor_library.path = str(library_path)
            config.factor_library.enforce_effective_membership = True
            frozen = factor_set.selection_context.get("directions")
            if frozen and frozen != _effective_factor_directions(config, config.factors):
                raise ValueError(f"strategy {strategy.id!r} frozen directions differ from current library")
        snapshot_definition = None
        if strategy.factor_definition_path:
            snapshot_definition = _load_factor_definition(
                _resolve(strategy.factor_definition_path)
            )
            config.factors = list(snapshot_definition["factors"])
            config.factor_library.enforce_effective_membership = False
        # All peers share the comparison range without rewriting their YAMLs.
        config.date_range.start = COMPARISON_START
        if strategy.source == "legacy_observation" and LEGACY_COMPARISON_FILLNA_ZERO:
            if not any(step.type == "fillna" for step in config.processing):
                config.processing.append(
                    ProcessingStepConfig(type="fillna", params={"method": "zero"})
                )
        if config.data.source != "duckdb_futures":
            raise ValueError(
                f"strategy {strategy.id!r} must use the framework's certified "
                "duckdb_futures runtime source"
            )
        if default_production is not None and (
            _config_dict(config.production_portfolio)
            != _config_dict(default_production)
        ):
            raise ValueError(
                f"strategy {strategy.id!r} overrides production_portfolio; "
                "default comparison changes factor sets only"
            )
        assignments = _assignments(config, strategy.mode)
        configured_factors = set().union(*map(set, assignments.values()))
        if strategy.source == "effective_library":
            if not config.factor_library.enforce_effective_membership:
                raise ValueError(
                    f"effective-library strategy {strategy.id!r} must enable "
                    "factor_library.enforce_effective_membership"
                )
            if _resolve(config.factor_library.path) != library_path:
                raise ValueError(
                    f"strategy {strategy.id!r} does not use the catalog's "
                    "effective factor library"
                )
            expected_factors = set(factor_sets[strategy.factor_set_id].factors)
            if configured_factors != expected_factors:
                raise ValueError(
                    f"strategy {strategy.id!r} factors do not match factor set "
                    f"{strategy.factor_set_id!r}"
                )
            validate_effective_factor_membership(
                library_path, assignments
            )
        validated.append((strategy, path, config))
    if not validated:
        raise ValueError("strategy library has no preferred or observing strategies")
    unknown_ids = sorted(requested_ids - seen_ids)
    if unknown_ids:
        raise ValueError(f"strategy IDs are not in the catalog: {unknown_ids}")
    return catalog_path, catalog, validated


def _config_dict(config) -> dict:
    return config.model_dump() if hasattr(config, "model_dump") else config.dict()


def _strategy_label(strategy_id: str) -> str:
    return STRATEGY_LABELS.get(str(strategy_id), str(strategy_id))


def _comparison_metrics(combined) -> dict:
    """Add NAV-day and turnover diagnostics to the framework metrics."""
    metrics = dict(combined.metrics)
    nav = pd.Series(combined.nav, dtype=float).dropna().sort_index()
    returns = nav.pct_change(fill_method=None).iloc[1:]
    metrics["positive_day_ratio"] = (
        float(returns.gt(0.0).mean()) if not returns.empty else 0.0
    )
    turnover = pd.Series(
        getattr(combined, "turnover", pd.Series(dtype=float)), dtype=float
    ).reindex(nav.index)
    intervals = turnover.iloc[1:]
    metrics.update(compute_turnover_metrics(intervals) if intervals.notna().all() else
                   dict.fromkeys(compute_turnover_metrics(pd.Series(dtype=float))))
    return metrics


def _write_comparison_plot(
    output: Path,
    nav_table: pd.DataFrame,
    rows: list[dict],
    *,
    title: str | None = None,
    cutoff: pd.Timestamp | None = None,
    background_batches=None,
    background_limits: tuple[float, float] | None = None,
) -> None:
    """Use the framework's Chinese plotting conventions for peer comparison."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False
    if nav_table.empty:
        return

    extra_rows = max(len(rows) - 9, 0)
    fig, (ax, metrics_ax) = plt.subplots(
        2, 1, figsize=(15, 10 + 0.27 * extra_rows),
        gridspec_kw={"height_ratios": [5.0, 1.35 + 0.22 * extra_rows]}
    )
    colors = [
        "#1a73e8", "#e8710a", "#1e8e3e", "#d93025", "#9334e6", "#00897b",
        "#795548", "#d81b60", "#607d8b", "#bcbd22", "#17becf", "#ff9896", "#9467bd",
    ]
    for idx, name in enumerate(nav_table.columns):
        series = nav_table[name].dropna()
        if series.empty:
            continue
        ax.plot(
            series.index,
            series.values,
            label=_strategy_label(name),
            color=colors[idx % len(colors)],
            linewidth=1.6,
        )
    # Keep failed strategies visible in the same comparison figure without
    # inventing a NAV line; the metrics table below carries the full reason.
    plotted = set(nav_table.columns)
    for row in rows:
        name = str(row.get("strategy", ""))
        if name not in plotted and str(row.get("status", "")).startswith("failed_"):
            ax.plot(
                [], [], linestyle="--", color="#777777",
                label=_strategy_label(name) + "（未形成）",
            )
    if cutoff is not None and nav_table.index.min() <= cutoff < nav_table.index.max():
        ax.axvline(cutoff, color="#555555", linestyle=":", linewidth=1.3,
                   label=f"研究截止日 {cutoff:%Y-%m-%d}")
        ax.axvspan(cutoff, nav_table.index.max(), color="#e8f1fc", alpha=0.45,
                   label="模拟实盘观察期（截止日之后）")
    ax.set_title(
        title or f"平行策略净值对比（{nav_table.index.min():%Y-%m-%d}至{nav_table.index.max():%Y-%m-%d}）",
        fontsize=14,
    )
    ax.set_ylabel("归一化净值")
    ax.set_xlabel("交易日期")
    ax.legend(loc="upper left", ncol=3 if len(rows) > 6 else 2)
    ax.grid(True, alpha=0.3)

    columns = (
        ("annual_return", "年化收益", "pct"),
        ("sharpe", "夏普", "num"),
        ("max_drawdown", "最大回撤", "pct"),
        ("volatility", "年化波动", "pct"),
        ("total_return", "总收益", "pct"),
        ("calmar", "卡玛", "num"),
        ("annualized_turnover", "年化换手(倍/年)", "num"),
    )
    table_rows, labels = [], []
    for row in rows:
        failed = str(row.get("status", "")).startswith("failed_")
        labels.append(
            _strategy_label(row["strategy"]) + ("（未形成）" if failed else "")
        )
        values = []
        for key, _label, kind in columns:
            if failed or row.get(key) is None or pd.isna(row.get(key)):
                values.append("—")
            else:
                value = float(row[key])
                values.append(f"{value:.2%}" if kind == "pct" else f"{value:.2f}")
        table_rows.append(values)
    metrics_ax.axis("off")
    table = metrics_ax.table(
        cellText=table_rows,
        rowLabels=labels,
        colLabels=[label for _key, label, _kind in columns],
        cellLoc="center",
        rowLoc="center",
        loc="center",
        bbox=[0.03, 0.0, 0.94, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.35)
    if background_batches is not None:
        # Stream every background curve into the existing Chinese report
        # canvas. Do not retain millions of Line2D artists or sample away jobs.
        from matplotlib.collections import LineCollection
        from matplotlib.dates import date2num
        from PIL import Image
        import numpy as np

        if background_limits is not None:
            low, high = background_limits
            margin = max((high - low) * 0.03, 0.01)
            ax.set_ylim(low - margin, high + margin)
        fig.set_dpi(150)
        fig.tight_layout()
        fig.canvas.draw()
        for batch in background_batches:
            if batch.empty:
                continue
            dates = date2num(pd.DatetimeIndex(batch.index).to_pydatetime())
            values = batch.to_numpy(dtype=float).T
            segments = np.stack((np.broadcast_to(dates, values.shape), values), axis=-1)
            collection = LineCollection(segments, colors="#777777", linewidths=0.45, alpha=0.025)
            ax.add_collection(collection, autolim=False)
            ax.draw_artist(collection)
            collection.remove()
        for line in ax.lines:
            ax.draw_artist(line)
        if ax.get_legend() is not None:
            ax.draw_artist(ax.get_legend())
        Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).save(output / "nav_comparison.png")
    else:
        fig.tight_layout()
        fig.savefig(output / "nav_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _segment_rows(
    strategy: str,
    status: str,
    factor_set_id: str,
    nav: pd.Series,
    turnover: pd.Series,
    cutoff: pd.Timestamp,
) -> list[dict]:
    """Compute full/pre-cutoff/forward metrics without changing the backtest."""
    nav = pd.Series(nav, dtype=float).dropna().sort_index()
    nav.index = pd.DatetimeIndex(nav.index)
    turnover = pd.Series(turnover, dtype=float).reindex(nav.index)
    rows: list[dict] = []
    segments = (
        ("full_observation", pd.Series(True, index=nav.index)),
        ("research_through_cutoff", nav.index <= cutoff),
        ("forward_observation_after_cutoff", nav.index > cutoff),
    )
    for label, mask in segments:
        segment = nav.loc[mask]
        if segment.empty:
            continue
        first = int(nav.index.get_loc(segment.index[0]))
        # Segment dates identify holding days. Retain the preceding close as
        # an anchor so the first post-cutoff return/trade is not discarded.
        anchored = pd.concat([nav.iloc[first - 1:first], segment]) if first else segment
        if len(anchored) < 2:
            continue
        normalized = anchored / float(anchored.iloc[0])
        returns = normalized.pct_change(fill_method=None).iloc[1:]
        metrics = compute_all_metrics(normalized, returns=returns)
        # ``compute_all_metrics.win_rate`` is reserved for trade-signal
        # ledgers; this report has daily NAV returns, so expose the relevant
        # positive-day ratio explicitly instead of presenting a misleading 0.
        metrics["positive_day_ratio"] = (
            float(returns.gt(0.0).mean()) if not returns.empty else 0.0
        )
        intervals = turnover.loc[returns.index]
        metrics.update(compute_turnover_metrics(intervals) if intervals.notna().all() else
                       dict.fromkeys(compute_turnover_metrics(pd.Series(dtype=float))))
        rows.append({
            "strategy": strategy,
            "status": status,
            "factor_set_id": factor_set_id,
            "segment": label,
            "start": segment.index[0].date().isoformat(),
            "end": segment.index[-1].date().isoformat(),
            "observations": int(len(segment)),
            "return_intervals": int(len(returns)),
            "anchor_date": anchored.index[0].date().isoformat(),
            **metrics,
        })
    return rows


def _legacy_recipe(config):
    """Build the shared production recipe from a fully loaded config."""
    from research.historical_portfolio_search import PortfolioRecipe

    portfolio = config.production_portfolio

    def _pairs(value) -> tuple[tuple[str, float], ...]:
        return tuple(sorted(
            (str(key), float(item))
            for key, item in dict(value or {}).items()
        ))

    return PortfolioRecipe(
        factor_weight=str(portfolio.factor_weight_method),
        top_n=int(portfolio.top_n_per_side),
        sector_cap=int(portfolio.sector_count_cap),
        asset_weight=str(portfolio.asset_weight_method),
        asset_min_fraction=float(portfolio.asset_min_fraction),
        asset_max_fraction=float(portfolio.asset_max_fraction),
        gross_exposure=float(portfolio.gross_exposure),
        asset_max_overrides=_pairs(portfolio.asset_max_overrides),
        sector_weight_caps=_pairs(portfolio.sector_weight_caps),
    )


def _default_production_config():
    """Load the one portfolio-method contract used by the default route."""
    return load_config(_resolve("config/default.yaml"))


def _effective_factor_directions(config, factors: list[str]) -> dict[str, int]:
    """Load frozen directions when a strategy explicitly uses the effective library."""
    if not bool(config.factor_library.enforce_effective_membership):
        return {}
    path = _resolve(config.factor_library.path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("factors", []) if isinstance(payload, dict) else []
    directions = {
        str(record["factor"]): int(record["direction"])
        for record in records
        if isinstance(record, dict) and "factor" in record and "direction" in record
    }
    missing = sorted(set(factors) - set(directions))
    if missing:
        raise ValueError(
            "effective-library production route has no frozen direction for: "
            + ", ".join(missing)
        )
    return {name: directions[name] for name in factors}


def _build_shared_production_panel(
    specs, *, ic_horizon: int = 1, checkpoint_dir: Path | None = None
):
    """Compute the union factor panel once for one peer-comparison run."""
    from research.portfolio_experiment_support import (
        FactorPanelRunner,
        latest_local_date,
    )

    factors = list(dict.fromkeys(
        str(name)
        for _strategy, _config_path, config in specs
        for name in config.factors
    ))
    if not factors:
        raise ValueError("production comparison has no factors")
    latest = pd.Timestamp(latest_local_date()).normalize()
    configured_ends = [
        latest
        if str(config.date_range.end) == "latest_available"
        else min(pd.Timestamp(config.date_range.end).normalize(), latest)
        for _strategy, _config_path, config in specs
    ]
    return FactorPanelRunner(
        factors,
        start=(
            pd.Timestamp(COMPARISON_START)
            - pd.Timedelta(days=LEGACY_PANEL_BUFFER_DAYS)
        ),
        end=max(configured_ends),
        ic_horizon=int(ic_horizon),
        checkpoint_dir=checkpoint_dir,
    )


def _run_production_portfolio(
    config,
    strategy_dir: Path,
    *,
    strategy_id: str = "current_single_baseline",
    route: str = "legacy_production_portfolio",
    factor_directions: dict[str, int] | None = None,
    factor_direction_source: dict | None = None,
    panel_runner_override=None,
    ic_horizon: int = 1,
):
    """Run a factor set through the shared configured production ledger.

    ``config/default.yaml`` remains the single source for the recipe, while
    the generic model/risk PipelineRunner remains available for the explicit
    configured-candidate route.  The default IDE comparison calls this helper
    for every selected peer, so only the factor subset changes.
    """
    from backtest.engine import BacktestResult
    from backtest.metrics import compute_all_metrics
    from research.historical_portfolio_search import (
        CausalEligibilityEnvironment,
        PortfolioEvaluator,
    )
    from research.portfolio_experiment_support import (
        FactorPanelRunner,
        configured_futures_cost_model,
        latest_local_date,
    )

    start = pd.Timestamp(COMPARISON_START)
    latest = pd.Timestamp(latest_local_date()).normalize()
    configured_end = str(config.date_range.end)
    end = (
        latest
        if configured_end == "latest_available"
        else min(pd.Timestamp(configured_end).normalize(), latest)
    )
    panel_start = start - pd.Timedelta(days=LEGACY_PANEL_BUFFER_DAYS)
    factors = list(config.factors)
    # The default route deliberately ignores any production-method override in
    # a candidate YAML.  This keeps the comparison variable at the factor set
    # and makes config/default.yaml the single method source of truth.
    production_config = _default_production_config()
    recipe = _legacy_recipe(production_config)
    direction_source = factor_direction_source
    if factor_directions is None:
        factor_directions = _effective_factor_directions(config, factors)
    if factor_directions and direction_source is None:
        direction_path = _resolve(config.factor_library.path)
        direction_source = {
            "path": str(direction_path),
            "sha256": hashlib.sha256(direction_path.read_bytes()).hexdigest(),
        }
    if factor_directions:
        missing_directions = sorted(set(factors) - set(factor_directions))
        if missing_directions:
            raise ValueError(
                "production route has no direction for: "
                + ", ".join(missing_directions)
            )

    ic_horizon = int(ic_horizon)
    if ic_horizon < 1:
        raise ValueError("ic_horizon must be positive")
    if panel_runner_override is not None and int(
        getattr(panel_runner_override, "ic_horizon", 1)
    ) != ic_horizon:
        raise ValueError("shared panel IC horizon does not match the requested route")
    if panel_runner_override is None:
        panel_runner = FactorPanelRunner(
            factors,
            start=panel_start,
            end=end,
            factor_directions=factor_directions,
            ic_horizon=ic_horizon,
        )
    else:
        # Load the concrete-contract schedule once on the shared owner before
        # creating a shallow strategy view. Factor values are never recomputed;
        # directions and IC remain strategy-specific.
        panel_runner_override.get_contract_schedule()
        panel_runner = panel_runner_override.for_factors(
            factors,
            factor_directions=factor_directions,
        )
    panel_runner.get_contract_schedule()
    panel_runner.env = CausalEligibilityEnvironment(
        panel_runner.cal,
        panel_runner.daily_ret,
        panel_runner.env.sector_of,
    )
    evaluator = PortfolioEvaluator(
        panel_runner,
        start=start,
        end=end,
        cost_model=configured_futures_cost_model(),
        ic_window=int(production_config.production_portfolio.ic_window),
        risk_lookback_calendar_days=int(
            production_config.production_portfolio.risk_lookback_calendar_days
        ),
    )
    weights = evaluator.weights(factors, recipe)
    ledger = evaluator.ledger_from_weights(weights)
    nav = pd.Series(ledger["nav"], dtype=float).sort_index()
    returns = pd.Series(ledger["net_return"], dtype=float).reindex(nav.index)
    # Both turnover and executed_traded_notional are normalized weight ratios,
    # not currency amounts. Gross leverage is already included in the weights.
    turnover = pd.Series(ledger["turnover"], dtype=float).reindex(nav.index)
    metrics = compute_all_metrics(nav, returns=returns.iloc[1:])
    result = BacktestResult(
        nav=nav,
        weights_history=weights,
        metrics=metrics,
        turnover=turnover,
        costs=pd.Series(
            ledger["trade_cost"].to_numpy(dtype=float)
            + ledger["holding_cost"].to_numpy(dtype=float),
            index=ledger.index,
            name="cost",
        ),
    )
    shared_panel_meta = {
        "shared": panel_runner_override is not None,
        "computed_factor_count": len(
            getattr(panel_runner_override, "raw_ranks", {})
        ) if panel_runner_override is not None else len(factors),
    }
    result.save(strategy_dir, metadata={
        "route": route,
        "strategy": strategy_id,
        "factor_count": len(factors),
        "factor_direction_source": direction_source,
        "panel_start": panel_start.date().isoformat(),
        "comparison_start": start.date().isoformat(),
        "observation_end": end.date().isoformat(),
        "recipe": recipe.to_dict(),
        "ic_horizon": ic_horizon,
        "factor_panel": shared_panel_meta,
        "production_config_path": str(_resolve("config/default.yaml")),
        "production_config_sha256": hashlib.sha256(
            _resolve("config/default.yaml").read_bytes()
        ).hexdigest(),
    })
    ledger.to_csv(strategy_dir / "production_ledger.csv", encoding="utf-8-sig")
    from optimization.costs import evaluate_cost_sensitivity
    (strategy_dir / "cost_diagnostics.json").write_text(
        json.dumps(evaluate_cost_sensitivity(ledger, initial_anchor=True),
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (strategy_dir / "production_recipe.json").write_text(
        json.dumps({
            "route": route,
            "panel_start": panel_start.date().isoformat(),
            "comparison_start": start.date().isoformat(),
            "observation_end": end.date().isoformat(),
            "factors": factors,
            "factor_direction_source": direction_source,
            "recipe": recipe.to_dict(),
            "ic_horizon": ic_horizon,
            "factor_panel": shared_panel_meta,
            "production_config_path": str(_resolve("config/default.yaml")),
            "production_config_sha256": hashlib.sha256(
                _resolve("config/default.yaml").read_bytes()
            ).hexdigest(),
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result, {
        "route": route,
        "panel_start": panel_start.date().isoformat(),
        "comparison_start": start.date().isoformat(),
        "observation_end": end.date().isoformat(),
        "factor_direction_source": direction_source,
        "recipe": recipe.to_dict(),
        "ic_horizon": ic_horizon,
        "factor_panel": shared_panel_meta,
        "production_config_path": str(_resolve("config/default.yaml")),
        "production_config_sha256": hashlib.sha256(
            _resolve("config/default.yaml").read_bytes()
        ).hexdigest(),
    }


def _write_segment_report(
    output: Path,
    strategy_results: list[tuple[object, object, object]],
    cutoff: pd.Timestamp,
    *,
    production_method_compare: bool = False,
    ic_horizon: int = 1,
    failures: list[dict] | None = None,
) -> None:
    failures = list(failures or [])
    rows: list[dict] = []
    for strategy, combined, _config in strategy_results:
        rows.extend(
            _segment_rows(
                strategy.id,
                strategy.status,
                strategy.factor_set_id,
                combined.nav,
                getattr(combined, "turnover", pd.Series(dtype=float)),
                cutoff,
            )
        )
    if not rows and not failures:
        return
    if rows:
        table = pd.DataFrame(rows)
        table.to_csv(
            output / "segment_comparison.csv", index=False, encoding="utf-8-sig"
        )
    metric_columns = (
        "annual_return", "sharpe", "max_drawdown", "volatility",
        "total_return", "positive_day_ratio", "avg_turnover",
        "annualized_turnover", "total_turnover",
    )
    lines = [
        "# 组合回测报告",
        "",
        f"- 回测区间：{COMPARISON_START} 至数据源最新完整交易日。",
        f"- 研究截止日：{cutoff.date().isoformat()}（不参与回测选择之后的新增研究）",
        (
            (
                "- 本次默认生产方法比较中，所有选定策略均采用 "
                "config/default.yaml::production_portfolio 的同一默认方法："
                "因子加权、选池、板块上限、品种分配及敞口参数以运行合同为准；只改变因子集合。"
            )
            if production_method_compare and int(ic_horizon) == 1
            else (
                f"- 本次显式 H{int(ic_horizon)} IC 敏感性比较中，所有选定策略均采用 "
                "同一冻结配方，参数以运行合同为准；"
                "仅将 IC 历史标签改为 H{0}，不作为默认方法。".format(int(ic_horizon))
                if production_method_compare
                else
                "- 旧10因子采用 config/default.yaml::production_portfolio 的默认方法："
                "参数以运行合同为准；其独立配置未改写。"
            )
        ),
        "- `research_through_cutoff`：用于查看研究截止日前的历史表现。",
        f"- `forward_observation_after_cutoff`：{cutoff.date().isoformat()} 之后的数据，仅作为确定组合的真实环境观察。",
        "- 信号时序：T 日收盘形成目标，下一交易日生效；收益按前一日有效权重乘 T 日收盘到收盘收益。",
        "- 价格口径：因子与连续收益使用因果的点时主连比例后复权价格；交易转换单独使用具体合约计划计量换手和换月成本。",
        "- `positive_day_ratio` 是日度净值收益为正的比例；`win_rate` 仅适用于独立交易信号账本，本报告不将其误作日度胜率。",
        "- 分段按收益所属日归属，截止后第一日的收益和执行换手均计入；前一收盘只作净值锚点。年化换手单位为倍/年，包含零成交持有日。缺失交易账本标为不可得，不伪造零换手。",
        "- 生产路径的cost_diagnostics.json使用通用static_turnover_rate_v1静态费率压力方法；不修改基线净值、默认费用或策略成员。",
        "",
        "| strategy | segment | start | end | annual_return | sharpe | max_drawdown | volatility | total_return | positive_day_ratio | avg_turnover | annualized_turnover | total_turnover |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        cells = [row["strategy"], row["segment"], row["start"], row["end"]]
        for column in metric_columns:
            value = row.get(column)
            cells.append("—" if value is None or pd.isna(value) else f"{float(value):.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    if failures:
        lines.extend([
            "",
            "## 未形成完整净值的快照",
            "",
            "以下方案因当前数据/因子可用性门禁失败，未生成虚假净值或指标：",
            "",
            "| strategy | error_type | reason |",
            "|---|---|---|",
        ])
        for failure in failures:
            lines.append(
                "| {strategy} | {error_type} | {message} |".format(**failure)
            )
    (output / "portfolio_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_and_compare() -> Path:
    run_started = time.perf_counter()
    catalog_path, catalog, specs = _validated_specs()
    run_id = RUN_ID or datetime.now().strftime("%Y%m%d_%H%M%S")
    output = _resolve(catalog.output_root) / run_id
    production_method_compare = WORKFLOW in {
        PortfolioWorkflow.RUN_PREFERRED,
        PortfolioWorkflow.RUN_AND_COMPARE,
        PortfolioWorkflow.RUN_AND_COMPARE_SNAPSHOT_AUDIT,
        PortfolioWorkflow.RUN_AND_COMPARE_ALL,
    }
    checkpoint_dir = output / ".factor_panel_checkpoint"
    if output.exists():
        contract_path = output / "run_contract.json"
        completed = False
        if contract_path.is_file():
            try:
                completed = json.loads(
                    contract_path.read_text(encoding="utf-8")
                ).get("status") == "complete"
            except (json.JSONDecodeError, OSError):
                pass
        resumable = (
            RUN_ID is not None
            and production_method_compare
            and checkpoint_dir.is_dir()
            and not completed
        )
        if not resumable:
            raise FileExistsError(output)
    else:
        output.mkdir(parents=True)
    if production_method_compare:
        checkpoint_dir.mkdir(exist_ok=True)
    rows, navs, configs = [], {}, {}
    failures: list[dict] = []
    strategy_results: list[tuple[object, object, object]] = []
    cutoffs: set[pd.Timestamp] = set()
    panel_started = time.perf_counter()
    shared_panel_runner = None
    if production_method_compare:
        shared_panel_runner = _build_shared_production_panel(
            specs, checkpoint_dir=checkpoint_dir
        )
    performance = {
        "schema_version": 1,
        "shared_factor_panel_seconds": (
            time.perf_counter() - panel_started
            if shared_panel_runner is not None else 0.0
        ),
        "shared_factor_panel": (
            {
                "factor_count": len(shared_panel_runner.raw_ranks),
                "checkpoint_loaded_factor_count": int(
                    shared_panel_runner.checkpoint_loaded_factor_count
                ),
                "computed_factor_count": int(
                    shared_panel_runner.computed_factor_count
                ),
                "profile": shared_panel_runner.performance_profile(),
            }
            if shared_panel_runner is not None else None
        ),
        "strategies": [],
    }

    for strategy, config_path, config in specs:
        strategy_started = time.perf_counter()
        name, mode = strategy.id, strategy.mode
        strategy_dir = output / name
        config.backtest.report_dir = str(strategy_dir)
        cutoffs.add(research_cutoff(config))
        snapshot_definition = None
        try:
            snapshot_definition = (
                _load_factor_definition(_resolve(strategy.factor_definition_path))
                if strategy.factor_definition_path
                else None
            )
            if strategy.source == "legacy_observation" or production_method_compare:
                # The old observation strategy is a durable production recipe, not
                # a generic single-pipeline model experiment.  Keep this explicit
                # so the two semantically different routes cannot be conflated.
                result, route_meta = _run_production_portfolio(
                    config,
                    strategy_dir,
                    strategy_id=name,
                    route=(
                        "default_production_portfolio"
                        if production_method_compare
                        else "legacy_production_portfolio"
                    ),
                    factor_directions=(
                        snapshot_definition["directions"]
                        if snapshot_definition is not None else None
                    ),
                    factor_direction_source=(
                        {
                            "path": snapshot_definition["path"],
                            "sha256": snapshot_definition["sha256"],
                        }
                        if snapshot_definition is not None else None
                    ),
                    panel_runner_override=shared_panel_runner,
                )
                runner = None
                if catalog.plot:
                    result.plot(save_dir=str(strategy_dir))
                combined = result
            else:
                runner = PipelineRunner(config=config)  # resolves latest available date
                result = (
                    runner.run_full_pipeline()
                    if mode == "single"
                    else runner.run_multi_portfolio()
                )
                result.save(strategy_dir, metadata={
                    "strategy": name,
                    "status": strategy.status,
                    "factor_set_id": strategy.factor_set_id,
                    "mode": mode,
                    "config_path": str(config_path),
                    "observation_end": runner.config.date_range.end,
                })
                route_meta = {
                    "route": "pipeline_runner",
                    "mode": mode,
                    "observation_end": runner.config.date_range.end,
                }
                if catalog.plot:
                    result.plot(save_dir=str(strategy_dir))
                combined = result if mode == "single" else result.combined_result
        except Exception as exc:
            if WORKFLOW not in {
                PortfolioWorkflow.RUN_AND_COMPARE_SNAPSHOT_AUDIT,
                PortfolioWorkflow.RUN_AND_COMPARE_ALL,
            }:
                raise
            failure = {
                "strategy": name,
                "error_type": type(exc).__name__,
                "message": str(exc).replace("|", "/"),
            }
            failures.append(failure)
            strategy_dir.mkdir(parents=True, exist_ok=True)
            (strategy_dir / "failure.json").write_text(
                json.dumps({
                    **failure,
                    "config_path": str(config_path),
                    "config_sha256": hashlib.sha256(
                        config_path.read_bytes()
                    ).hexdigest(),
                    "factor_definition": (
                        {
                            "path": snapshot_definition["path"],
                            "sha256": snapshot_definition["sha256"],
                        }
                        if snapshot_definition is not None else None
                    ),
                }, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            rows.append({
                "strategy": name,
                "status": "failed_data_quality",
                "factor_set_id": strategy.factor_set_id,
                "start": COMPARISON_START,
                "end": str(config.date_range.end),
                "error": str(exc),
            })
            configs[name] = {
                "path": str(config_path),
                "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "execution": {
                    "route": "default_production_portfolio",
                    "status": "failed",
                },
                "factor_definition": (
                    {
                        "path": snapshot_definition["path"],
                        "sha256": snapshot_definition["sha256"],
                    }
                    if snapshot_definition is not None else None
                ),
                "resolved": _config_dict(config),
                "failure": failure,
            }
            performance["strategies"].append({
                "strategy": name,
                "status": "failed",
                "seconds": time.perf_counter() - strategy_started,
            })
            continue
        strategy_results.append((strategy, combined, config))
        navs[name] = combined.nav / float(combined.nav.iloc[0])
        metrics = _comparison_metrics(combined)
        observation_start = (
            route_meta.get("comparison_start", config.date_range.start)
        )
        observation_end = route_meta.get(
            "observation_end", config.date_range.end
        )
        rows.append({
            "strategy": name,
            "status": strategy.status,
            "factor_set_id": strategy.factor_set_id,
            "start": observation_start,
            "end": observation_end,
            **metrics,
        })
        configs[name] = {
            "path": str(config_path),
            "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "execution": route_meta,
            "factor_definition": (
                {
                    "path": snapshot_definition["path"],
                    "sha256": snapshot_definition["sha256"],
                }
                if snapshot_definition is not None else None
            ),
            "resolved": _config_dict(config),
        }
        performance["strategies"].append({
            "strategy": name,
            "status": "complete",
            "seconds": time.perf_counter() - strategy_started,
        })

    comparison = pd.DataFrame(rows).set_index("strategy")
    comparison.to_csv(output / "comparison.csv", encoding="utf-8-sig")
    nav_table = pd.DataFrame(navs)
    nav_table.to_csv(output / "nav_comparison.csv")
    (output / "run_contract.json").write_text(json.dumps({
        "schema_version": 1,
        "status": "finalizing",
        "run_id": run_id,
        "comparison_start": COMPARISON_START,
        "comparison_policy": {
            "construction_route": (
                "default_production_portfolio"
                if production_method_compare
                else "catalog_configured"
            ),
            "shared_factor_panel": (
                {
                    "enabled": True,
                    "computed_factor_count": len(shared_panel_runner.raw_ranks),
                    "checkpoint_loaded_factor_count": int(
                        shared_panel_runner.checkpoint_loaded_factor_count
                    ),
                    "newly_computed_factor_count": int(
                        shared_panel_runner.computed_factor_count
                    ),
                }
                if shared_panel_runner is not None
                else {"enabled": False}
            ),
            "legacy_route": "legacy_production_portfolio",
            "legacy_missing_exposure": "fillna_zero"
            if LEGACY_COMPARISON_FILLNA_ZERO else "fail_closed",
        },
        "strategy_library": {
            "path": str(catalog_path),
            "sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
            "snapshot": _config_dict(catalog),
        },
        "strategies": configs,
        "failures": failures,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if len(cutoffs) != 1:
        raise ValueError(
            "strategy catalog contains multiple research cutoffs; refusing "
            "to write an ambiguous comparison report"
        )
    if catalog.plot:
        _write_comparison_plot(output, nav_table, rows, cutoff=next(iter(cutoffs)))
    _write_segment_report(
        output,
        strategy_results,
        next(iter(cutoffs)),
        production_method_compare=(
            production_method_compare
        ),
        failures=failures,
    )
    if failures:
        (output / "snapshot_failures.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    performance["total_seconds"] = time.perf_counter() - run_started
    performance["process_peak_working_set_mib"] = _peak_working_set_mib()
    (output / "performance.json").write_text(
        json.dumps(performance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # Bind the durable comparison artifacts to the same immutable run
    # contract.  Missing optional plots are simply omitted; the CSV/Markdown
    # report hashes are always recorded when those files were produced.
    contract_path = output / "run_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    artifact_names = (
        "comparison.csv",
        "nav_comparison.csv",
        "segment_comparison.csv",
        "portfolio_report.md",
        "nav_comparison.png",
        "snapshot_failures.json",
        "performance.json",
    )
    contract["artifacts"] = {
        name: {"sha256": hashlib.sha256((output / name).read_bytes()).hexdigest()}
        for name in artifact_names
        if (output / name).exists()
    }
    contract["status"] = "complete"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if checkpoint_dir.is_dir():
        shutil.rmtree(checkpoint_dir)
    return output


def _load_common_h5_selection() -> tuple[dict, dict[str, list[str]], dict[str, int]]:
    """Load one finalized common-H5 selection without touching the library."""
    root = _resolve(COMMON_H5_SELECTION_RUN_DIR)
    summary_path = root / "selection_summary.json"
    sets_path = root / "factor_sets.json"
    contract_path = root / "run_contract.json"
    if not all(path.exists() for path in (summary_path, sets_path, contract_path)):
        raise FileNotFoundError(
            "common-H5 comparison requires a finalized selection run: "
            f"{root}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if summary.get("selection_mode") != "common_horizon" or int(
        summary.get("common_horizon", 0) or 0
    ) != 5:
        raise ValueError("selection source is not a common-H5 evidence run")
    if contract.get("selection_contract", {}).get("common_horizon") != 5:
        raise ValueError("selection contract and summary disagree on H5")
    payload = json.loads(sets_path.read_text(encoding="utf-8"))
    sets = {}
    for key, label in (
        ("balanced_core", "common_h5_balanced"),
        ("compact_core", "common_h5_compact"),
    ):
        factors = payload.get(key, {}).get("factors", [])
        if not factors or len(factors) != len(set(factors)):
            raise ValueError(f"invalid factor set in common-H5 selection: {key}")
        sets[label] = [str(name) for name in factors]
    passed_path = _resolve(
        str(Path(summary["source_run_dir"]) / "passed_factors.csv")
    )
    if not passed_path.exists():
        raise FileNotFoundError(f"common-H5 validation detail not found: {passed_path}")
    passed = pd.read_csv(passed_path, encoding="utf-8-sig")
    required = {"factor", "final_pass", "is_ic"}
    if not required.issubset(passed.columns):
        raise ValueError("common-H5 validation detail has an unexpected schema")
    passed = passed.loc[
        passed["final_pass"].map(
            lambda value: str(value).strip().lower() in {"true", "1", "yes"}
        )
    ]
    directions = {
        str(row.factor): (1 if float(row.is_ic) >= 0.0 else -1)
        for row in passed.itertuples(index=False)
    }
    selected = set().union(*map(set, sets.values()))
    if not selected.issubset(directions):
        raise ValueError("common-H5 factor set contains a factor without frozen direction")
    summary["selection_run_dir"] = str(root)
    summary["selection_run_sha256"] = sha256_file(summary_path)
    summary["validation_detail_sha256"] = sha256_file(passed_path)
    return summary, sets, directions


def run_common_h5_compare(
    *,
    ic_horizon: int = 1,
    run_id_override: str | None = None,
    factor_search_run_dir: str | None = None,
) -> Path:
    """Compare H5-selected peers and current peers under one production recipe.

    The common-H5 candidates are deliberately not admitted to the effective
    library.  The shared panel is only a performance optimization; each row
    retains its own factor directions and source hashes in the run contract.
    """
    run_started = time.perf_counter()
    ic_horizon = int(ic_horizon)
    if ic_horizon < 1:
        raise ValueError("ic_horizon must be positive")
    selection_summary, h5_sets, h5_directions = _load_common_h5_selection()
    default_config = load_config(_resolve("config/default.yaml"))
    catalog_path = _resolve(CATALOG_PATH)
    catalog = load_strategy_library(catalog_path)
    catalog_by_id = {entry.id: entry for entry in catalog.strategies}
    current_ids = ("intraday_balanced_ridge", "intraday_compact_ridge")
    missing = [name for name in current_ids if name not in catalog_by_id]
    if missing:
        raise ValueError(f"current peer strategies are missing from the catalog: {missing}")

    candidates: list[dict] = [{
        "id": "current_single_baseline",
        "label": "旧10因子观察策略",
        "config": default_config,
        "factor_source_config": str(_resolve("config/default.yaml")),
        "factors": list(default_config.factors),
        "factor_set_id": "legacy_10f",
        "directions": None,
        "direction_source": None,
    }]
    library_path = _resolve(catalog.effective_factor_library)
    library_source = {
        "path": str(library_path),
        "sha256": hashlib.sha256(library_path.read_bytes()).hexdigest(),
    }
    for strategy_id in current_ids:
        entry = catalog_by_id[strategy_id]
        config_path = _resolve(entry.config_path)
        config = load_config(config_path)
        factors = list(config.factors)
        candidates.append({
            "id": strategy_id,
            "label": STRATEGY_LABELS[strategy_id],
            "config": config,
            "factor_source_config": str(config_path),
            "factors": factors,
            "factor_set_id": entry.factor_set_id,
            "directions": _effective_factor_directions(config, factors),
            "direction_source": library_source,
        })
    validation_source = {
        "path": str(_resolve(selection_summary["source_run_dir"])),
        "summary_sha256": selection_summary["selection_run_sha256"],
        "detail_sha256": selection_summary["validation_detail_sha256"],
        "horizon": 5,
    }
    for strategy_id in ("common_h5_balanced", "common_h5_compact"):
        candidates.append({
            "id": strategy_id,
            "label": STRATEGY_LABELS[strategy_id],
            "config": load_config(_resolve("config/default.yaml")),
            "factor_source_config": str(_resolve(COMMON_H5_SELECTION_RUN_DIR)),
            "factors": h5_sets[strategy_id],
            "factor_set_id": strategy_id,
            "directions": {
                name: h5_directions[name] for name in h5_sets[strategy_id]
            },
            "direction_source": validation_source,
        })
    factor_search_source = None
    if factor_search_run_dir:
        search_root = _resolve(factor_search_run_dir)
        search_path = search_root / "search_summary.json"
        if not search_path.is_file():
            raise FileNotFoundError(
                f"combination-aware factor search is incomplete: {search_root}"
            )
        search = json.loads(search_path.read_text(encoding="utf-8"))
        if (
            search.get("workflow") != "combination_aware_factor_search"
            or bool(search.get("oos_used_for_selection", True))
            or int(search.get("common_horizon", 0)) != 5
            or str(search.get("research_cutoff")) != str(research_cutoff(default_config).date())
        ):
            raise ValueError("combination-aware factor search contract is invalid")
        search_factors = [str(name) for name in search["winner"]["factors"]]
        if not search_factors or not set(search_factors).issubset(h5_directions):
            raise ValueError("factor-search winner is outside the common-H5 pool")
        factor_search_source = {
            "path": str(search_root),
            "sha256": sha256_file(search_path),
            "oos_used_for_selection": False,
        }
        candidates.append({
            "id": "common_h5_combination",
            "label": STRATEGY_LABELS["common_h5_combination"],
            "config": load_config(_resolve("config/default.yaml")),
            "factor_source_config": str(search_root),
            "factors": search_factors,
            "factor_set_id": "common_h5_combination",
            "directions": {name: h5_directions[name] for name in search_factors},
            "direction_source": factor_search_source,
        })

    direction_union: dict[str, int] = {}
    for candidate in candidates:
        for name, direction in (candidate["directions"] or {}).items():
            previous = direction_union.get(name)
            if previous is not None and previous != int(direction):
                raise ValueError(f"factor direction conflict for shared panel: {name}")
            direction_union[name] = int(direction)
    all_factors = sorted(set().union(*(set(item["factors"]) for item in candidates)))
    from research.portfolio_experiment_support import FactorPanelRunner, latest_local_date

    latest = pd.Timestamp(latest_local_date()).normalize()
    panel_started = time.perf_counter()
    panel_runner = FactorPanelRunner(
        all_factors,
        start=pd.Timestamp(COMPARISON_START) - pd.Timedelta(days=LEGACY_PANEL_BUFFER_DAYS),
        end=latest,
        factor_directions=direction_union,
        ic_horizon=ic_horizon,
    )
    panel_seconds = time.perf_counter() - panel_started

    run_id = run_id_override or (
        COMMON_H5_COMPARISON_RUN_ID if ic_horizon == 1
        else COMMON_H5_MATCHED_COMPARISON_RUN_ID
    ) or datetime.now().strftime(
        "%Y%m%d_%H%M%S_common_h5_"
        + ("default_recipe_compare" if ic_horizon == 1 else "matched_ic_compare")
    )
    output = _resolve("runs/portfolio_backtest") / run_id
    output.mkdir(parents=True, exist_ok=False)
    rows: list[dict] = []
    navs: dict[str, pd.Series] = {}
    configs: dict[str, dict] = {}
    strategy_results: list[tuple[object, object, object]] = []
    strategy_timings: list[dict] = []
    cutoff = research_cutoff(default_config)
    for candidate in candidates:
        strategy_started = time.perf_counter()
        strategy_id = candidate["id"]
        config = candidate["config"]
        config.factors = list(candidate["factors"])
        config.date_range.start = COMPARISON_START
        strategy_dir = output / strategy_id
        result, route_meta = _run_production_portfolio(
            config,
            strategy_dir,
            strategy_id=strategy_id,
            route=(
                "common_h5_selection_default_production_recipe"
                if ic_horizon == 1
                else "common_h5_selection_h5_ic_sensitivity"
            ),
            factor_directions=candidate["directions"],
            factor_direction_source=candidate["direction_source"],
            panel_runner_override=panel_runner,
            ic_horizon=ic_horizon,
        )
        if catalog.plot:
            result.plot(save_dir=str(strategy_dir))
        navs[strategy_id] = result.nav / float(result.nav.iloc[0])
        rows.append({
            "strategy": strategy_id,
            "status": "observing",
            "factor_set_id": candidate["factor_set_id"],
            "start": route_meta["comparison_start"],
            "end": route_meta["observation_end"],
            "factor_count": len(candidate["factors"]),
            **_comparison_metrics(result),
        })
        configs[strategy_id] = {
            "config_path": str(_resolve("config/default.yaml")),
            "config_sha256": hashlib.sha256(
                _resolve("config/default.yaml").read_bytes()
            ).hexdigest(),
            "factor_source_config": candidate["factor_source_config"],
            "factors": list(candidate["factors"]),
            "factor_direction_source": candidate["direction_source"],
            "execution": route_meta,
        }
        strategy_results.append((
            SimpleNamespace(
                id=strategy_id,
                status="observing",
                factor_set_id=candidate["factor_set_id"],
            ),
            result,
            config,
        ))
        strategy_timings.append({
            "strategy": strategy_id,
            "seconds": time.perf_counter() - strategy_started,
        })
    comparison = pd.DataFrame(rows).set_index("strategy")
    comparison.to_csv(output / "comparison.csv", encoding="utf-8-sig")
    nav_table = pd.DataFrame(navs)
    nav_table.to_csv(output / "nav_comparison.csv")
    run_contract = {
        "schema_version": 1,
        "run_id": run_id,
        "comparison_start": COMPARISON_START,
        "comparison_policy": {
            "construction_route": "default_production_portfolio",
            "factor_variable": "candidate_factor_set",
            "method": "config/default.yaml::production_portfolio",
            "horizon_note": (
                "H5 selection evidence; production route uses the default daily IC ledger"
                if ic_horizon == 1
                else "all peers use the explicit H5 IC sensitivity route; this is not the default"
            ),
            "ic_horizon": ic_horizon,
        },
        "selection_source": selection_summary,
        "factor_search_source": factor_search_source,
        "strategies": configs,
    }
    (output / "run_contract.json").write_text(
        json.dumps(run_contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if catalog.plot:
        _write_comparison_plot(output, nav_table, rows, cutoff=cutoff)
    _write_segment_report(
        output,
        strategy_results,
        cutoff,
        production_method_compare=True,
        ic_horizon=ic_horizon,
    )
    (output / "performance.json").write_text(
        json.dumps({
            "factor_panel_seconds": panel_seconds,
            "factor_panel_profile": panel_runner.performance_profile(),
            "strategy_seconds": strategy_timings,
            "total_seconds": time.perf_counter() - run_started,
            "process_peak_working_set_mib": _peak_working_set_mib(),
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    contract = json.loads((output / "run_contract.json").read_text(encoding="utf-8"))
    artifact_names = (
        "comparison.csv", "nav_comparison.csv", "segment_comparison.csv",
        "portfolio_report.md", "nav_comparison.png", "performance.json",
    )
    contract["artifacts"] = {
        name: {"sha256": hashlib.sha256((output / name).read_bytes()).hexdigest()}
        for name in artifact_names if (output / name).exists()
    }
    (output / "run_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    if WORKFLOW is PortfolioWorkflow.SEARCH_COMMON_H5_FACTOR_SET:
        from workflows.experiments.historical_portfolio_search import (
            run_combination_aware_factor_search,
        )

        run_id = COMMON_H5_SEARCH_RUN_ID or datetime.now().strftime(
            "%Y%m%d_%H%M%S_common_h5_combination_search"
        )
        output = run_combination_aware_factor_search(
            run_id=run_id,
            config_path="config/default.yaml",
            source_validation_dir=COMMON_H5_VALIDATION_RUN_DIR,
            source_selection_dir=COMMON_H5_SEARCH_BASELINE_DIR,
            common_horizon=5,
        )
        print(f"组合互补性因子搜索产物: {output}")
        return

    if WORKFLOW is PortfolioWorkflow.RUN_AND_COMPARE_COMMON_H5:
        print(f"共同H5因子集默认方法比较结果: {run_common_h5_compare()}")
        return
    if WORKFLOW is PortfolioWorkflow.RUN_AND_COMPARE_COMMON_H5_MATCHED:
        print(
            "共同H5因子集H5-IC敏感性比较结果: "
            f"{run_common_h5_compare(ic_horizon=5)}"
        )
        return

    if WORKFLOW is PortfolioWorkflow.RUN_AND_COMPARE_COMMON_H5_SEARCH:
        print(
            "共同H5组合搜索候选对比结果: "
            f"{run_common_h5_compare(ic_horizon=COMMON_H5_SEARCH_IC_HORIZON, run_id_override=COMMON_H5_SEARCH_COMPARISON_RUN_ID, factor_search_run_dir=COMMON_H5_FACTOR_SEARCH_RUN_DIR)}"
        )
        return
    catalog_path, _, specs = _validated_specs()
    if WORKFLOW is PortfolioWorkflow.VALIDATE_CONFIGURATIONS:
        print(f"strategy_library={catalog_path}")
        for strategy, path, config in specs:
            gate = config.factor_library.enforce_effective_membership
            print(
                f"{strategy.id}: status={strategy.status}, mode={strategy.mode}, "
                f"factor_set={strategy.factor_set_id or '-'}, config={path}, "
                f"data_source={config.data.source}, "
                f"dates={config.date_range.start}~{config.date_range.end}, "
                f"effective_membership_gate={gate}"
            )
        return
    if WORKFLOW in {PortfolioWorkflow.RUN_PREFERRED, PortfolioWorkflow.RUN_AND_COMPARE}:
        print(f"默认生产方法组合回测结果: {run_and_compare()}")
        return
    if WORKFLOW is PortfolioWorkflow.RUN_AND_COMPARE_CONFIGURED:
        print(f"显式配置方法组合回测结果: {run_and_compare()}")
        return
    if WORKFLOW is PortfolioWorkflow.RUN_AND_COMPARE_SNAPSHOT_AUDIT:
        print(f"历史快照当前口径重评结果: {run_and_compare()}")
        return
    if WORKFLOW is PortfolioWorkflow.RUN_AND_COMPARE_ALL:
        print(f"全部备选策略统一组合回测结果: {run_and_compare()}")
        return
    raise ValueError(f"unsupported portfolio workflow: {WORKFLOW!r}")


if __name__ == "__main__":
    main()
