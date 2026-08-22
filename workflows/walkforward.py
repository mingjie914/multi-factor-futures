"""多段 walk-forward 验证 + 蒙特卡洛扰动 + 参数敏感性分析.

验证策略鲁棒性, 不优化参数, 只做诊断:
1. 4 段滚动样本外验证: 每段 1 年样本外, 报告所有段表现一致性
2. 蒙特卡洛扰动测试: 子组合权重 ±20% 随机扰动 1000 次, 看夏普分布
3. 参数敏感性分析: retrain_freq / holding_period ±20%, 看夏普稳定性

Usage:
    python main.py walkforward
    python main.py walkforward --mc-only  # 只跑蒙特卡洛
    python main.py walkforward --sens-only  # 只跑参数敏感性
"""
from __future__ import annotations
import sys
import os
import json
import logging
import time
import argparse
import copy
from pathlib import Path
import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.logger import setup_logger
from pipeline.runner import PipelineRunner
from core.config import load_config
from research.validation import OOS_END, SIMULATED_LIVE_START

setup_logger("multi_factor")


def run_backtest_with_config(config, quiet=True, runner=None):
    """用指定配置对象运行多子组合回测, 返回结果对象."""
    import logging
    if quiet:
        logging.getLogger("multi_factor").setLevel(logging.WARNING)
    runner = runner or PipelineRunner(config=config)
    if runner.config is not config:
        raise ValueError("prevalidated runner must share the supplied config object")
    result = runner.run_multi_portfolio()
    return result


def compute_metrics(nav: pd.Series) -> dict:
    """从净值序列计算核心指标."""
    from backtest.metrics import compute_all_metrics
    metrics = compute_all_metrics(nav)
    return {
        "annual_return": float(metrics.get("annual_return", 0)),
        "sharpe": float(metrics.get("sharpe", 0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0)),
        "volatility": float(metrics.get("volatility", 0)),
    }


def _calendar_coverage_bounds(
    calendar, start, end, grace_bars: int = 5
):
    """Return coverage bounds on the supplied exchange calendar."""
    dates = pd.DatetimeIndex(calendar)
    dates = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    if dates.empty:
        raise ValueError(f"test range contains no exchange bars: {start}~{end}")
    grace = max(int(grace_bars), 0)
    if 2 * grace >= len(dates):
        raise ValueError("coverage grace consumes the complete test range")
    return dates[grace], dates[-grace - 1] if grace else dates[-1]


def _candidate_factor_names() -> list[str]:
    # Importing the module registers both built-in and SPEC factors.
    import workflows.factor_adaptivity  # noqa: F401
    from core.registry import list_registered

    return sorted(list_registered("factor").get("factor", {}).keys())


def _horizon_targets(config, period: float):
    """Return nearest sleeve, plus optional neighbours, using training metadata."""
    ranked = sorted(
        config.sub_portfolios,
        key=lambda sub: (
            abs(np.log(max(float(period), 1.0) / max(float(sub.holding_period), 1.0))),
            float(sub.holding_period),
            sub.name,
        ),
    )
    ensemble = getattr(config, "horizon_ensemble", None)
    if ensemble is None or not ensemble.enabled:
        return ranked[:1]
    limit = 1 + max(int(ensemble.neighbor_count), 0)
    return [
        sub for sub in ranked[:limit]
        if abs(np.log(max(float(period), 1.0) / max(float(sub.holding_period), 1.0)))
        <= float(ensemble.max_log_distance)
    ] or ranked[:1]


def _materialize_exact_horizon_sleeves(config, periods) -> None:
    """Create an exact sleeve for every approved sector-level horizon."""
    requested = sorted({
        int(float(period)) for period in periods
        if pd.notna(period) and float(period) > 0
    })
    used_names = {sub.name for sub in config.sub_portfolios}
    ensemble = getattr(config, "horizon_ensemble", None)
    retrain_map = (
        dict(getattr(ensemble, "retrain_freq_by_horizon", {}) or {})
        if ensemble is not None else {}
    )
    for period in requested:
        existing = [
            sub for sub in config.sub_portfolios
            if int(sub.holding_period) == period
        ]
        if existing:
            configured = retrain_map.get(str(period), retrain_map.get(period))
            if configured is not None:
                for sleeve in existing:
                    sleeve.retrain_freq = int(configured)
            continue
        source = min(
            config.sub_portfolios,
            key=lambda sub: (
                abs(np.log(period / max(float(sub.holding_period), 1.0))),
                float(sub.holding_period),
                sub.name,
            ),
        )
        sleeve = copy.deepcopy(source)
        base_name = f"horizon_{period}"
        sleeve.name = base_name
        suffix = 2
        while sleeve.name in used_names:
            sleeve.name = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(sleeve.name)
        sleeve.holding_period = period
        sleeve.retrain_freq = int(
            retrain_map.get(str(period), retrain_map.get(period, min(period, 20)))
        )
        if sleeve.retrain_freq < 1:
            raise ValueError(f"retrain frequency must be positive for horizon {period}")
        sleeve.factors = []
        config.sub_portfolios.append(sleeve)


def _apply_practical_profile(config) -> None:
    """Apply the four approved trading-oriented settings as one profile."""
    config.asset_selection.enabled = True
    config.asset_selection.mode = "hysteresis_top_n"
    config.asset_selection.top_n_per_side = 2
    config.asset_selection.exit_buffer = 1
    config.asset_selection.min_abs_forecast = 0.0
    config.asset_selection.restrict_to_valid_sectors = True

    config.horizon_ensemble.enabled = True
    config.horizon_ensemble.use_valid_periods = True
    config.horizon_ensemble.neighbor_count = 2
    config.horizon_ensemble.max_log_distance = 1.50
    config.horizon_ensemble.retrain_freq_by_horizon = {"20": 5}


def _sector_row_horizons(config, row) -> list[int]:
    from core.period import approved_horizon_ensemble

    ensemble = getattr(config, "horizon_ensemble", None)
    use_valid = bool(
        ensemble is not None and ensemble.enabled and ensemble.use_valid_periods
    )
    return approved_horizon_ensemble(
        row.get("best_period"),
        row.get("valid_periods", "") if use_valid else "",
        enabled=use_valid,
        neighbor_count=(ensemble.neighbor_count if ensemble is not None else 0),
        max_log_distance=(ensemble.max_log_distance if ensemble is not None else 0.0),
    )


def _assign_fold_factors(
    config,
    bundle,
    *,
    deduplicate_clusters: bool = False,
    drop_empty_sleeves: bool = False,
) -> dict:
    """Assign FDR-approved training factors to the nearest configured horizon."""
    summary = bundle.read_csv("factor_adaptivity_summary", encoding="utf-8-sig")
    required = {"factor", "best_period", "n_valid_sectors", "best_q", "best_t"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"训练产物摘要缺少字段: {sorted(missing)}")
    approved = summary[
        (pd.to_numeric(summary["n_valid_sectors"], errors="coerce").fillna(0) > 0)
        & (pd.to_numeric(summary["best_period"], errors="coerce").fillna(0) > 0)
    ].copy()
    if "sample_sufficient" in approved.columns:
        sample_ok = approved["sample_sufficient"].map(
            lambda value: str(value).strip().lower() in {"1", "true", "yes"}
        )
        approved = approved.loc[sample_ok].copy()
    if approved.empty:
        # Fallback: training window too short for sector adaptivity,
        # use best_period from IC results when adaptivity has none
        if bundle.has("factor_discovery"):
            ic_data = bundle.read_json("factor_discovery")
            ic_best = {
                str(row.get("name", "")): int(row.get("best_period", 0) or 0)
                for row in ic_data.get("all_results", [])
                if str(row.get("name", ""))
            }
            summary["best_period"] = summary["factor"].map(
                lambda name: ic_best.get(str(name), 0)
            )
        fallback = summary[
            pd.to_numeric(summary["best_period"], errors="coerce").fillna(0) > 0
        ].copy()
        if fallback.empty:
            raise RuntimeError("训练期没有通过全局 FDR 和稳定性门槛的因子")
        approved = fallback
    approved["best_q"] = pd.to_numeric(approved["best_q"], errors="coerce").fillna(1.0)
    approved["best_t_abs"] = pd.to_numeric(
        approved["best_t"], errors="coerce"
    ).fillna(0.0).abs()
    approved = approved.sort_values(
        ["best_q", "best_t_abs", "factor"], ascending=[True, False, True]
    )
    if deduplicate_clusters and bundle.has("factor_correlation"):
        correlation = bundle.read_json("factor_correlation")
        keep = set()
        clustered = set()
        approved_by_name = approved.set_index("factor", drop=False)
        for cluster in correlation.get("clusters", []):
            names = [
                item.get("name") for item in cluster.get("factors", [])
                if item.get("name") in approved_by_name.index
            ]
            clustered.update(names)
            if names:
                ranked_names = approved_by_name.loc[names].reset_index(drop=True).sort_values(
                    ["best_q", "best_t_abs", "factor"],
                    ascending=[True, False, True],
                )
                keep.add(str(ranked_names.iloc[0]["factor"]))
        keep.update(set(approved["factor"]) - clustered)
        approved = approved[approved["factor"].isin(keep)].copy()
    governance_cfg = getattr(config, "factor_governance", None)
    if governance_cfg is not None and governance_cfg.enabled:
        from research.governance import select_candidates_by_family

        selected, _ = select_candidates_by_family(
            approved.to_dict("records"),
            default_cap=governance_cfg.default_max_per_family,
            family_caps=governance_cfg.family_caps,
            explicit_map=governance_cfg.explicit_family_map,
        )
        approved = pd.DataFrame(selected)
        if approved.empty:
            raise RuntimeError("经济家族上限应用后没有剩余训练期因子")

    sector_selection = None
    if hasattr(bundle, "has") and bundle.has("factor_sector_selection"):
        sector_selection = bundle.read_csv(
            "factor_sector_selection", encoding="utf-8-sig"
        )
        required_sector = {"factor", "sector", "best_period"}
        missing_sector = required_sector - set(sector_selection.columns)
        if missing_sector:
            raise ValueError(
                f"训练产物板块周期表缺少字段: {sorted(missing_sector)}"
            )
        sector_selection = sector_selection[
            sector_selection["factor"].astype(str).isin(set(approved["factor"]))
        ].copy()
        routing_periods = [
            period
            for _, sector_row in sector_selection.iterrows()
            for period in _sector_row_horizons(config, sector_row)
        ]
        _materialize_exact_horizon_sleeves(config, routing_periods)
    assignments = {sub.name: [] for sub in config.sub_portfolios}
    for _, row in approved.iterrows():
        factor = str(row["factor"])
        periods = []
        if sector_selection is not None:
            matches = sector_selection[
                sector_selection["factor"].astype(str) == factor
            ]
            periods = [
                period
                for _, sector_row in matches.iterrows()
                for period in _sector_row_horizons(config, sector_row)
            ]
        periods = periods or [max(float(row["best_period"]), 1.0)]
        for period in periods:
            exact = [
                sub for sub in config.sub_portfolios
                if int(sub.holding_period) == int(float(period))
            ]
            targets = exact or _horizon_targets(config, max(float(period), 1.0))
            for target in targets:
                assignments[target.name].append(factor)
    original_sub_portfolios = list(config.sub_portfolios)
    for sub in original_sub_portfolios:
        selected = list(dict.fromkeys(assignments[sub.name]))
        if not selected and not drop_empty_sleeves:
            raise RuntimeError(f"训练期子组合 {sub.name!r} 没有可用因子")
        sub.factors = selected
    if drop_empty_sleeves:
        config.sub_portfolios = [
            sub for sub in original_sub_portfolios if sub.factors
        ]
        if not config.sub_portfolios:
            raise RuntimeError("训练期没有任何非空子组合")
        equal_weight = 1.0 / len(config.sub_portfolios)
        for sub in config.sub_portfolios:
            sub.capital_weight = equal_weight
    config.factors = list(
        dict.fromkeys(factor for sub in config.sub_portfolios for factor in sub.factors)
    )
    return {name: len(factors) for name, factors in assignments.items()}


def _build_fold_bundle(
    base_config,
    *,
    name: str,
    train_start: str,
    train_end: str,
    output_dir: Path,
    candidate_factors: list[str],
    build_correlation: bool,
    fdr_method: str,
    frequency: str = "daily",
):
    from workflows.factor_adaptivity import run_adaptivity_analysis
    from workflows.research import _run_multi_period_screening
    from research.artifacts import ResearchArtifactBundle, canonical_config_hash
    from pipeline.runner import PipelineRunner

    if fdr_method != "hierarchical":
        raise ValueError(
            "walk-forward discovery is frozen to validation_policy hierarchical FDR"
        )

    train_config = copy.deepcopy(base_config)
    train_config.research_artifacts.enabled = False
    train_config.research_artifacts.path = ""
    warmup_days = int(
        base_config.validation_policy.warmup_days_by_frequency.get(frequency, 252)
    )
    factor_start = (pd.Timestamp(train_start) - pd.Timedelta(days=warmup_days)).date().isoformat()
    train_runner = PipelineRunner(config=train_config)
    discovery = _run_multi_period_screening(
        train_runner,
        candidate_factors,
        "<in-memory-config>",
        1.96,
        pd.Timestamp(factor_start),
        pd.Timestamp(train_start),
        pd.Timestamp(train_end),
        periods_override=None,
        frequency=frequency,
        output_dir=str(output_dir),
        adaptivity_file=None,
    )
    discovered_names = list(discovery.get("final_factors", []))
    if not discovered_names:
        raise RuntimeError(
            "training fold produced no factor eligible for WF or observation channel"
        )
    discovered_set = set(discovered_names)
    preprocessing_variants = {
        str(row["name"]): str(row.get("best_variant", "neutralized"))
        for row in discovery.get("all_results", [])
        if str(row.get("name", "")) in discovered_set
    }
    candidate_metadata = {
        str(row["name"]): {
            "observation_channel": bool(row.get("observation_channel", False)),
            "observation_reasons": list(row.get("observation_reasons", [])),
            "promotion_status": str(row.get("promotion_status", "observation")),
            "weight_cap": float(row.get("weight_cap", 1.0)),
        }
        for row in discovery.get("significant_factors", [])
        if str(row.get("name", "")) in discovered_set
    }
    # Collect all horizon periods tested across factors
    candidate_periods = sorted({
        int(period)
        for _, row in enumerate(discovery.get("all_results", []))
        for _, values in row.get("all_periods", {}).items()
        if isinstance(values, dict)
        for period in [int(values.get("period", 0))]
        if period > 0
    })
    run_adaptivity_analysis(
        all_factors=discovered_names,
        config_path="<in-memory-config>",
        factor_start=factor_start,
        ic_start=train_start,
        ic_end=train_end,
        periods=candidate_periods,
        output_dir=str(output_dir),
        artifact_id=name,
        build_correlation=build_correlation,
        config=train_config,
        runner=train_runner,
        fdr_method="deployment",
        preprocessing_variants=preprocessing_variants,
        candidate_metadata=candidate_metadata,
    )
    return ResearchArtifactBundle.load(
        output_dir,
        decision_date=pd.Timestamp(train_end) + pd.Timedelta(days=1),
        expected_config_hash=canonical_config_hash(train_config),
    )


def _load_existing_fold_bundle(
    base_config,
    *,
    train_start: str,
    train_end: str,
    output_dir: Path,
):
    """Validate and reuse an immutable fold bundle after an interrupted run."""
    from research.artifacts import ResearchArtifactBundle, canonical_config_hash

    train_config = copy.deepcopy(base_config)
    train_config.research_artifacts.enabled = False
    train_config.research_artifacts.path = ""
    bundle = ResearchArtifactBundle.load(
        output_dir,
        decision_date=pd.Timestamp(train_end) + pd.Timedelta(days=1),
        expected_config_hash=canonical_config_hash(train_config),
    )
    manifest_start = pd.Timestamp(bundle.manifest["train_start"]).date().isoformat()
    manifest_end = pd.Timestamp(bundle.manifest["train_end"]).date().isoformat()
    if manifest_start != train_start or manifest_end != train_end:
        raise ValueError(
            "research artifact training range does not match fold: "
            f"expected={train_start}~{train_end}, "
            f"actual={manifest_start}~{manifest_end}"
        )
    return bundle


def _evaluate_fold_factor_ics(runner, bundle, test_start: str, test_end: str) -> dict:
    """Compute factor-level OOS ICs on one untouched walk-forward test fold."""
    from factors.processor import build_processing_context
    from workflows.research import _joint_ic_ols_statistics

    summary = bundle.read_csv("factor_adaptivity_summary", encoding="utf-8-sig")
    selected = set(runner.config.factors)
    summary = summary[summary["factor"].astype(str).isin(selected)].copy()
    if summary.empty:
        return {}
    factor_start = pd.Timestamp(test_start) - pd.DateOffset(years=1)
    calendar = runner.data_manager.get_calendar(factor_start, test_end)
    calendar = pd.DatetimeIndex(calendar)
    universe = pd.Index(runner.config.universe)
    context = build_processing_context(
        runner.data_manager,
        calendar,
        universe,
        runner.config.universe_selection,
    )
    names = sorted(set(summary["factor"].astype(str)))
    computed = runner.factor_engine.compute_factors(
        names,
        calendar,
        universe,
        parallel=False,
        chunk_size=max(len(names), 1),
    )
    processed = runner.processor.process_batch(computed, context)
    periods = sorted({
        int(value) for value in pd.to_numeric(
            summary["best_period"], errors="coerce"
        ).dropna() if int(value) > 0
    })
    returns = {
        period: runner.data_manager.get_forward_returns(
            calendar, universe, period=period
        )
        for period in periods
    }
    output = {}
    for _, row in summary.iterrows():
        name = str(row["factor"])
        period = int(row["best_period"])
        if name not in computed or period not in returns:
            continue
        if str(row.get("preprocessing_variant", "neutralized")) == "raw":
            matrix = runner.processor.process_excluding(
                computed[name], context, {"neutralize"}
            )
        else:
            matrix = processed[name]
        stats = _joint_ic_ols_statistics(
            matrix.loc[test_start:test_end],
            returns[period].loc[test_start:test_end],
            forward_period=period,
            min_stocks=10,
        )
        train_ic = float(row.get("best_ic", 0.0))
        oos_ic = float(stats["ic"])
        orientation = 1.0 if train_ic >= 0.0 else -1.0
        output[name] = {
            "period": period,
            "preprocessing_variant": str(
                row.get("preprocessing_variant", "neutralized")
            ),
            "train_ic": train_ic,
            "oos_ic": oos_ic,
            "oriented_oos_ic": oos_ic * orientation,
            "same_direction": bool(oos_ic * orientation > 0.0),
            "n_observations": int(stats["ic_n"]),
        }
    runner.factor_engine.clear_cache()
    return output


def summarize_factor_fold_survival(
    folds: list[dict], *, minimum_fold_ratio: float = 0.60
) -> dict:
    """Aggregate fold survival without claiming a new locked OOS.

    A factor must pass both parts of the OOS direction contract: its pooled,
    observation-weighted oriented IC must be positive and at least the
    configured share of folds must have the training direction.  The pooled
    check prevents a few small positive folds from masking one economically
    dominant negative fold.
    """
    histories = {}
    for fold in folds:
        for name, values in fold.get("factor_oos", {}).items():
            histories.setdefault(name, []).append(values)
    summary = {}
    for name, values in histories.items():
        same = [bool(item.get("same_direction", False)) for item in values]
        weighted_ics = []
        weights = []
        for item in values:
            oriented_ic = float(item.get("oriented_oos_ic", float("nan")))
            if not np.isfinite(oriented_ic):
                continue
            weight = float(item.get("n_observations", 1.0))
            if not np.isfinite(weight) or weight <= 0.0:
                continue
            weighted_ics.append(oriented_ic)
            weights.append(weight)
        combined_oriented_oos_ic = (
            float(np.average(weighted_ics, weights=weights))
            if weighted_ics
            else None
        )
        combined_oos_same_direction = bool(
            combined_oriented_oos_ic is not None
            and combined_oriented_oos_ic > 0.0
        )
        fold_sign_ratio = float(np.mean(same)) if same else 0.0
        longest = current = 0
        for survived in same:
            current = current + 1 if survived else 0
            longest = max(longest, current)
        summary[name] = {
            "tested_folds": len(values),
            "same_direction_folds": int(sum(same)),
            "fold_sign_ratio": fold_sign_ratio,
            "combined_oriented_oos_ic": combined_oriented_oos_ic,
            "combined_oos_same_direction": combined_oos_same_direction,
            "longest_consecutive_surviving_folds": int(longest),
            "passes_fold_gate": bool(
                combined_oos_same_direction
                and fold_sign_ratio >= minimum_fold_ratio
            ),
            "passes_two_consecutive_fold_gate": bool(
                combined_oos_same_direction
                and longest >= 2
                and fold_sign_ratio >= minimum_fold_ratio
            ),
            "observation_transition_ready": False,
            "requires_positive_new_locked_oos": True,
            "locked_oos_status": f"frozen_oos_ends_{OOS_END}",
            "simulated_live_status": f"active_since_{SIMULATED_LIVE_START}",
            "production_approved": False,
        }
    return summary


def _build_rolling_folds(
    calendar: pd.DatetimeIndex,
    train_bars: int,
    test_bars: int,
    step_bars: int,
    *,
    frequency: str = "daily",
) -> list[tuple]:
    """Generate rolling WF fold segments from a calendar index and bar counts.

    Each fold is ``(name, train_start, train_end, test_start, test_end)``.
    Test windows are non-overlapping; training windows are fixed-length rolling.

    If fewer than 3 folds can be built, ``step_bars`` is shortened to
    ``test_bars // 2`` as a fallback (with a warning).
    """
    min_len = train_bars + test_bars
    if len(calendar) < min_len:
        return []
    folds: list[tuple] = []
    fold_idx = 1
    cursor = 0
    while cursor + min_len <= len(calendar):
        train_start_dt = pd.Timestamp(calendar[cursor]).date().isoformat()
        train_end_idx = cursor + train_bars - 1
        train_end_dt = pd.Timestamp(calendar[train_end_idx]).date().isoformat()
        test_start_idx = train_end_idx + 1
        test_end_idx = test_start_idx + test_bars - 1
        if test_end_idx >= len(calendar):
            break
        test_start_dt = pd.Timestamp(calendar[test_start_idx]).date().isoformat()
        test_end_dt = pd.Timestamp(calendar[test_end_idx]).date().isoformat()
        folds.append((
            f"折{fold_idx}",
            train_start_dt,
            train_end_dt,
            test_start_dt,
            test_end_dt,
        ))
        cursor += step_bars
        fold_idx += 1
    # Fallback: if too few folds, retry with shorter step
    if len(folds) < 3 and step_bars > test_bars // 2:
        logger = logging.getLogger("multi_factor")
        logger.warning(
            "walk-forward 仅生成 %d 个折叠（最少需要3个），"
            "将 step_bars 从 %d 缩短至 %d 重试",
            len(folds), step_bars, test_bars // 2,
        )
        return _build_rolling_folds(
            calendar, train_bars, test_bars, test_bars // 2, frequency=frequency,
        )
    return folds


def walk_forward_4fold(
    base_config,
    *,
    run_root: str | Path = None,
    candidate_factors: list[str] = None,
    build_correlation: bool = True,
    max_folds: int = None,
    fold_numbers: list[int] = None,
    fdr_method: str = "hierarchical",
    reuse_artifacts: bool = False,
    frequency: str = "daily",
    is_intraday: bool = False,
    calendar=None,
):
    """Nested walk-forward with rolling test folds sized by bar counts.

    Fold segments are generated from the configured calendar + per-frequency
    WF parameters (train/test/step bars).  Daily factors can use separate
    ``daily_intraday`` sizing when ``is_intraday=True``.
    """
    from core.period import PeriodContext

    policy = base_config.validation_policy
    ctx = PeriodContext.from_string(frequency)
    if not ctx.is_daily:
        raise ValueError(
            "portfolio walk-forward accounting is daily-only; non-daily factor "
            "research must remain in the frequency-aware screening workflow"
        )
    _freq_key = (
        "daily_intraday"
        if is_intraday and ctx.unit.value == "daily"
        else ctx.unit.value
    )
    train_bars = int(policy.wf_train_bars_by_frequency.get(_freq_key, 500))
    test_bars = int(policy.wf_test_bars_by_frequency.get(_freq_key, 125))
    step_bars = int(policy.wf_step_bars_by_frequency.get(_freq_key, 125))

    configured_end = pd.Timestamp(base_config.date_range.end)
    configured_start = pd.Timestamp(base_config.date_range.start)
    if calendar is None:
        from data.manager import DataManager

        calendar_manager = DataManager.from_config(base_config)
        calendar = calendar_manager.get_calendar(configured_start, configured_end)
    calendar = pd.DatetimeIndex(calendar)
    if (
        calendar.empty
        or calendar.has_duplicates
        or not calendar.is_monotonic_increasing
    ):
        raise ValueError("walk-forward calendar must be non-empty, unique and sorted")
    calendar = calendar[
        (calendar >= configured_start) & (calendar <= configured_end)
    ]
    segments = _build_rolling_folds(
        calendar, train_bars, test_bars, step_bars,
        frequency=_freq_key,
    )
    if not segments:
        raise RuntimeError(
            "walk-forward 无法生成任何折叠段；"
            f"日历={len(calendar)}天, train={train_bars}, test={test_bars}, step={step_bars}"
        )
    if fold_numbers is not None:
        requested = list(dict.fromkeys(int(value) for value in fold_numbers))
        invalid = [value for value in requested if value < 1 or value > len(segments)]
        if invalid:
            raise ValueError(
                f"fold_numbers outside available range 1..{len(segments)}: {invalid}"
            )
        segments = [segments[value - 1] for value in requested]
    print("\n" + "=" * 70)
    print(f"1. 嵌套 walk-forward 验证 ({len(segments)} 个独立测试折)")
    print("=" * 70)
    if max_folds is not None:
        if max_folds < 1:
            raise ValueError("max_folds must be at least 1")
        segments = segments[:max_folds]
    test_ranges = {(test_start, test_end) for _, _, _, test_start, test_end in segments}
    if len(test_ranges) != len(segments):
        raise ValueError("walk-forward 测试区间存在重复")
    candidate_factors = candidate_factors or _candidate_factor_names()
    if not candidate_factors:
        raise RuntimeError("没有已注册候选因子")
    if run_root is None:
        raise ValueError("run_root is required; automatic run directories are disabled")
    run_root = Path(run_root).resolve()
    if reuse_artifacts:
        if not run_root.is_dir():
            raise FileNotFoundError(f"frozen run directory not found: {run_root}")
    else:
        run_root.mkdir(parents=True, exist_ok=False)

    results = []
    for name, train_start, train_end, test_start, test_end in segments:
        print(f"\n[{name}] 训练 {train_start}~{train_end}, 测试 {test_start}~{test_end}")

        from research.sample_policy import assess_sample_counts

        train_dates = calendar[
            (calendar >= pd.Timestamp(train_start))
            & (calendar <= pd.Timestamp(train_end))
        ]
        test_dates = calendar[
            (calendar >= pd.Timestamp(test_start))
            & (calendar <= pd.Timestamp(test_end))
        ]
        sample_assessment = assess_sample_counts(
            len(train_dates), len(test_dates),
            policy=base_config.validation_policy,
            frequency=_freq_key,
            train_days=len(train_dates), test_days=len(test_dates),
        )
        if not sample_assessment.sufficient:
            results.append({
                "segment": name,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "status": "observation",
                "observation_channel": True,
                "sample_assessment": sample_assessment.to_dict(),
                "production_approved": False,
            })
            continue

        try:
            t0 = time.time()
            fold_dir = run_root / name / "artifacts"
            if reuse_artifacts:
                bundle = _load_existing_fold_bundle(
                    base_config,
                    train_start=train_start,
                    train_end=train_end,
                    output_dir=fold_dir,
                )
            else:
                bundle = _build_fold_bundle(
                    base_config,
                    name=f"{run_root.name}_{name}",
                    train_start=train_start,
                    train_end=train_end,
                    output_dir=fold_dir,
                    candidate_factors=candidate_factors,
                    build_correlation=build_correlation,
                    fdr_method=fdr_method,
                    frequency=_freq_key,
                )
            cfg = copy.deepcopy(base_config)
            cfg.date_range.start = test_start
            cfg.date_range.end = test_end
            cfg.research_artifacts.enabled = True
            cfg.research_artifacts.path = str(fold_dir)
            cfg.research_artifacts.strict_config_hash = False  # WF test config differs from training
            # Validate the frozen research inputs before applying fold-specific
            # portfolio outputs such as selected factors and empty-sleeve removal.
            runner = PipelineRunner(config=cfg)
            selected_counts = _assign_fold_factors(
                cfg,
                bundle,
                deduplicate_clusters=True,
                drop_empty_sleeves=True,
            )
            factor_oos = _evaluate_fold_factor_ics(
                runner, bundle, test_start, test_end
            )
            result = run_backtest_with_config(cfg, quiet=True, runner=runner)
            nav = result.combined_result.nav.dropna()
            if nav.empty:
                raise RuntimeError("测试折没有生成净值")
            expected_start = pd.Timestamp(test_start)
            expected_end = pd.Timestamp(test_end)
            actual_start = pd.Timestamp(nav.index.min())
            actual_end = pd.Timestamp(nav.index.max())
            latest_allowed_start, earliest_allowed_end = _calendar_coverage_bounds(
                calendar, expected_start, expected_end, grace_bars=5
            )
            if actual_start > latest_allowed_start:
                raise RuntimeError(
                    f"测试折起点覆盖不完整: expected={expected_start.date()} "
                    f"actual={actual_start.date()}"
                )
            if actual_end < earliest_allowed_end:
                raise RuntimeError(
                    f"测试折终点覆盖不完整: expected={expected_end.date()} "
                    f"actual={actual_end.date()}"
                )
            m = compute_metrics(nav)
            from research.statistics import deflated_sharpe_ratio

            m["dsr"] = deflated_sharpe_ratio(
                nav.pct_change(fill_method=None).dropna(),
                n_trials=len(candidate_factors),
                risk_free_rate=0.0,
            )
            m["segment"] = name
            m["train_start"] = train_start
            m["train_end"] = train_end
            m["test_start"] = test_start
            m["test_end"] = test_end
            m["sample_assessment"] = sample_assessment.to_dict()
            m["observation_channel"] = False
            m["actual_test_start"] = str(actual_start.date())
            m["actual_test_end"] = str(actual_end.date())
            m["alpha_type"] = base_config.alpha.type
            m["artifact_id"] = bundle.artifact_id
            m["artifact_path"] = str(fold_dir)
            m["selected_factor_counts"] = selected_counts
            m["selected_factors"] = {
                sub.name: list(sub.factors) for sub in cfg.sub_portfolios
            }
            m["factor_oos"] = factor_oos
            m["candidate_factor_count"] = len(candidate_factors)
            m["elapsed"] = time.time() - t0
            portfolio_dir = run_root / name / "portfolio"
            result.save(
                portfolio_dir,
                metadata={
                    "phase": "adaptive_nested_oos",
                    "segment": name,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "alpha_type": base_config.alpha.type,
                    "artifact_id": bundle.artifact_id,
                    "selected_factors": m["selected_factors"],
                    "candidate_factor_count": len(candidate_factors),
                    "fdr_method": fdr_method,
                },
            )
            m["portfolio_path"] = str(portfolio_dir)
            results.append(m)
            print(f"  年化={m['annual_return']:.2%} 夏普={m['sharpe']:.2f} "
                  f"回撤={m['max_drawdown']:.2%} 波动={m['volatility']:.2%} "
                  f"({m['elapsed']:.0f}s)")
        except Exception as e:
            print(f"  失败: {e}")
            results.append({
                "segment": name, "train_start": train_start, "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end, "error": str(e)
            })

    # 汇总
    print("\n" + "-" * 70)
    print(f"{'段':<6} {'测试期':<25} {'年化':>8} {'夏普':>6} {'回撤':>8} {'波动':>8}")
    print("-" * 70)
    valid = [r for r in results if "error" not in r and not r.get("observation_channel")]
    for r in results:
        if "error" in r:
            print(f"{r['segment']:<6} {r['test_start']}~{r['test_end']:<12} 失败: {r['error']}")
        elif r.get("observation_channel"):
            print(f"{r['segment']:<6} {r['test_start']}~{r['test_end']:<12} (观察通道, 样本不足)")
        else:
            print(f"{r['segment']:<6} {r['test_start']}~{r['test_end']:<12} "
                  f"{r.get('annual_return', 0):>8.2%} {r.get('sharpe', 0):>6.2f} "
                  f"{r.get('max_drawdown', 0):>8.2%} {r.get('volatility', 0):>8.2%}")

    if valid:
        sharpes = [r["sharpe"] for r in valid]
        print(f"\n汇总: {len(valid)} 段有效")
        print(f"  夏普均值: {np.mean(sharpes):.2f}")
        print(f"  夏普中位数: {np.median(sharpes):.2f}")
        print(f"  夏普最小值: {np.min(sharpes):.2f}")
        print(f"  夏普最大值: {np.max(sharpes):.2f}")
        print(f"  正夏普段数: {sum(1 for s in sharpes if s > 0)}/{len(sharpes)}")
        print(f"  夏普>0.5段数: {sum(1 for s in sharpes if s > 0.5)}/{len(sharpes)}")

    print(f"\n冻结产物与折叠结果目录: {run_root}")

    return results


def monte_carlo_perturbation(base_config, n_simulations=1000):
    """蒙特卡洛扰动测试: 子组合权重 ±20% 随机扰动.

    验证策略对权重配置的敏感度. 若扰动后夏普分布坍塌, 说明过拟合.
    """
    print("\n" + "=" * 70)
    print(f"2. 蒙特卡洛扰动测试 ({n_simulations} 次模拟)")
    print("=" * 70)

    # 先用基础配置跑一次作为基准
    print("\n[基准] 原始权重配置...")
    try:
        base_result = run_backtest_with_config(copy.deepcopy(base_config), quiet=True)
        base_nav = base_result.combined_result.nav
        base_m = compute_metrics(base_nav)
        print(f"  基准: 年化={base_m['annual_return']:.2%} 夏普={base_m['sharpe']:.2f}")
    except Exception as e:
        print(f"  基准回测失败: {e}")
        return None

    # 获取原始权重
    original_weights = np.array([sp.capital_weight for sp in base_config.sub_portfolios])
    original_weights = original_weights / original_weights.sum()
    print(f"  原始权重: {original_weights}")

    # 蒙特卡洛: 权重 ±20% 扰动
    # 由于完整回测耗时, 这里用简化的扰动方法:
    # 用基准回测的子组合收益序列, 重新加权叠加 (不重跑回测)
    sub_returns = {}
    # MultiPortfolioResult.sub_results 可能是 dict 或 list
    sub_results_obj = base_result.sub_results
    if isinstance(sub_results_obj, dict):
        items = sub_results_obj.items()
    elif isinstance(sub_results_obj, list):
        # 列表元素可能是 dict 或对象
        items = []
        for r in sub_results_obj:
            if isinstance(r, dict):
                items.append((r.get("config", {}).get("name", f"sub_{len(items)}"), r))
            elif hasattr(r, "config") and hasattr(r.config, "name"):
                items.append((r.config.name, r))
            elif hasattr(r, "name"):
                items.append((r.name, r))
            else:
                items.append((f"sub_{len(items)}", r))
    else:
        items = []

    for name, r in items:
        # 尝试多种方式获取 nav
        nav = None
        if isinstance(r, dict):
            result_obj = r.get("result")
            if result_obj is not None and hasattr(result_obj, "nav"):
                nav = result_obj.nav
        elif hasattr(r, "result") and hasattr(r.result, "nav"):
            nav = r.result.nav
        elif hasattr(r, "nav"):
            nav = r.nav

        if nav is not None and len(nav) > 0:
            sub_returns[name] = nav.pct_change(fill_method=None).dropna()

    if not sub_returns:
        print("  无法提取子组合收益序列, 跳过蒙特卡洛")
        return None

    returns_df = pd.DataFrame(sub_returns)
    names = list(sub_returns.keys())
    valid_mask = returns_df[names].notna().any(axis=1)
    returns_df = returns_df.loc[valid_mask]
    returns_arr = returns_df[names].fillna(0.0).values

    rng = np.random.default_rng(42)
    sharpes = []
    annual_returns = []
    max_drawdowns = []

    print(f"\n[扰动] 对权重 ±20% 随机扰动 {n_simulations} 次...")
    t0 = time.time()
    for i in range(n_simulations):
        # 生成扰动权重: 原始权重 × (1 + U(-0.2, 0.2)), 归一化
        perturbation = rng.uniform(-0.2, 0.2, size=len(original_weights))
        perturbed = original_weights * (1 + perturbation)
        perturbed = perturbed / perturbed.sum()

        # 用扰动权重重新计算组合收益
        combined_ret = returns_arr @ perturbed
        nav = (1 + pd.Series(combined_ret, index=returns_df.index)).cumprod()

        m = compute_metrics(nav)
        sharpes.append(m["sharpe"])
        annual_returns.append(m["annual_return"])
        max_drawdowns.append(m["max_drawdown"])

    elapsed = time.time() - t0
    sharpes = np.array(sharpes)
    annual_returns = np.array(annual_returns)
    max_drawdowns = np.array(max_drawdowns)

    print(f"  完成 {n_simulations} 次模拟, 耗时 {elapsed:.1f}s")

    # 统计
    print(f"\n{'指标':<15} {'均值':>10} {'中位数':>10} {'5%分位':>10} {'95%分位':>10} {'最小':>10} {'最大':>10}")
    print("-" * 80)
    for name, vals in [("夏普", sharpes), ("年化收益", annual_returns), ("最大回撤", max_drawdowns)]:
        print(f"{name:<15} {np.mean(vals):>10.3f} {np.median(vals):>10.3f} "
              f"{np.percentile(vals, 5):>10.3f} {np.percentile(vals, 95):>10.3f} "
              f"{np.min(vals):>10.3f} {np.max(vals):>10.3f}")

    print(f"\n基准夏普: {base_m['sharpe']:.3f}")
    print(f"扰动夏普均值: {np.mean(sharpes):.3f} (差 {np.mean(sharpes) - base_m['sharpe']:+.3f})")
    print(f"扰动夏普 > 0.5 占比: {np.mean(sharpes > 0.5):.1%}")
    print(f"扰动夏普 > 0.8 占比: {np.mean(sharpes > 0.8):.1%}")
    print(f"扰动夏普 < 0 占比: {np.mean(sharpes < 0):.1%}")

    # 保存分布图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, vals, name, base_val in [
            (axes[0], sharpes, "Sharpe", base_m["sharpe"]),
            (axes[1], annual_returns, "Annual Return", base_m["annual_return"]),
            (axes[2], max_drawdowns, "Max Drawdown", base_m["max_drawdown"]),
        ]:
            ax.hist(vals, bins=50, alpha=0.7, color="steelblue", edgecolor="white")
            ax.axvline(base_val, color="red", linestyle="--", linewidth=2, label=f"Base={base_val:.3f}")
            ax.axvline(np.mean(vals), color="green", linestyle="-", linewidth=2, label=f"Mean={np.mean(vals):.3f}")
            ax.set_title(name)
            ax.legend()
            ax.grid(alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(_PROJECT_ROOT, "reports", "mc_perturbation.png")
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"\n扰动分布图已保存: {out_path}")
    except Exception as e:
        print(f"绘图失败: {e}")

    return {
        "base_sharpe": base_m["sharpe"],
        "mean_sharpe": float(np.mean(sharpes)),
        "median_sharpe": float(np.median(sharpes)),
        "p5_sharpe": float(np.percentile(sharpes, 5)),
        "p95_sharpe": float(np.percentile(sharpes, 95)),
        "pct_positive": float(np.mean(sharpes > 0)),
        "pct_above_0_5": float(np.mean(sharpes > 0.5)),
    }


def parameter_sensitivity(base_config):
    """参数敏感性分析: retrain_freq / holding_period ±20%.

    验证策略对关键参数的敏感度.
    """
    print("\n" + "=" * 70)
    print("3. 参数敏感性分析")
    print("=" * 70)

    results = []

    # 基准
    print("\n[基准] 原始参数...")
    try:
        base_result = run_backtest_with_config(copy.deepcopy(base_config), quiet=True)
        base_m = compute_metrics(base_result.combined_result.nav)
        base_m["label"] = "基准"
        base_m["retrain_short"] = 5
        base_m["retrain_mid"] = 10
        base_m["retrain_long"] = 20
        results.append(base_m)
        print(f"  年化={base_m['annual_return']:.2%} 夏普={base_m['sharpe']:.2f}")
    except Exception as e:
        print(f"  基准失败: {e}")
        return None

    # 参数变化矩阵: (short_retrain, mid_retrain, long_retrain)
    variations = [
        ("短期+20% (6d)", 6, 10, 20),
        ("短期-20% (4d)", 4, 10, 20),
        ("中期+20% (12d)", 5, 12, 20),
        ("中期-20% (8d)", 5, 8, 20),
        ("长期+20% (24d)", 5, 10, 24),
        ("长期-20% (16d)", 5, 10, 16),
        ("全部+20%", 6, 12, 24),
        ("全部-20%", 4, 8, 16),
    ]

    for label, s_r, m_r, l_r in variations:
        print(f"\n[{label}] retrain=({s_r}/{m_r}/{l_r})...")
        cfg = copy.deepcopy(base_config)
        cfg.sub_portfolios[0].retrain_freq = s_r
        cfg.sub_portfolios[1].retrain_freq = m_r
        cfg.sub_portfolios[2].retrain_freq = l_r

        try:
            t0 = time.time()
            result = run_backtest_with_config(cfg, quiet=True)
            m = compute_metrics(result.combined_result.nav)
            m["label"] = label
            m["retrain_short"] = s_r
            m["retrain_mid"] = m_r
            m["retrain_long"] = l_r
            m["elapsed"] = time.time() - t0
            results.append(m)
            print(f"  年化={m['annual_return']:.2%} 夏普={m['sharpe']:.2f} "
                  f"回撤={m['max_drawdown']:.2%} ({m['elapsed']:.0f}s)")
        except Exception as e:
            print(f"  失败: {e}")

    # 汇总
    print("\n" + "-" * 80)
    print(f"{'配置':<25} {'retrain':>15} {'年化':>8} {'夏普':>6} {'回撤':>8}")
    print("-" * 80)
    for r in results:
        retrain_str = f"{r.get('retrain_short', '?')}/{r.get('retrain_mid', '?')}/{r.get('retrain_long', '?')}"
        print(f"{r['label']:<25} {retrain_str:>15} "
              f"{r['annual_return']:>8.2%} {r['sharpe']:>6.2f} {r['max_drawdown']:>8.2%}")

    if len(results) > 1:
        sharpes = [r["sharpe"] for r in results]
        print(f"\n夏普统计: 均值={np.mean(sharpes):.2f}, 标准差={np.std(sharpes):.2f}, "
              f"范围=[{np.min(sharpes):.2f}, {np.max(sharpes):.2f}]")
        print(f"夏普相对基准变化: {[(r['label'], r['sharpe'] - base_m['sharpe']) for r in results[1:]]}")

    return results


def main():
    parser = argparse.ArgumentParser(description="多段 walk-forward 验证")
    parser.add_argument("--config", default="config/default.yaml", help="框架配置文件")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--wf-only", action="store_true", help="只跑嵌套 walk-forward")
    mode.add_argument("--mc-only", action="store_true", help="只跑蒙特卡洛扰动")
    mode.add_argument("--sens-only", action="store_true", help="只跑参数敏感性")
    parser.add_argument("--n-sim", type=int, default=1000, help="蒙特卡洛模拟次数")
    parser.add_argument("--max-folds", type=int, default=None, help="仅运行前 N 个测试折")
    parser.add_argument(
        "--folds", default=None,
        help="仅运行指定测试折，逗号分隔，例如 '4' 或 '2,4'",
    )
    parser.add_argument(
        "--end", default=None,
        help="覆盖配置的数据结束日；用于追加最新锁定测试折",
    )
    parser.add_argument(
        "--practical-profile", action="store_true",
        help="启用统一板块门控、板块Top-N、短重训和有效周期集成",
    )
    parser.add_argument(
        "--alpha-type",
        choices=["sector_grouped_ols", "sector_grouped_ridge"],
        default=None,
        help="覆盖 Alpha 模型用于候选比较",
    )
    parser.add_argument(
        "--is-intraday", action="store_true",
        help="使用 daily_intraday 参数进行 WF（适用于日内聚合为日频的因子）",
    )
    parser.add_argument(
        "--frequency", default="daily",
        choices=["daily", "1min", "5min", "15min", "30min", "hourly"],
        help="周期单位 (默认 daily). 非日度研究使用数据源的真实 bar 索引",
    )
    parser.add_argument(
        "--candidate-factors", default=None,
        help="候选因子逗号分隔；默认使用全部已注册因子",
    )
    parser.add_argument(
        "--run-root", required=True,
        help="显式冻结运行目录",
    )
    parser.add_argument(
        "--no-correlation", action="store_true",
        help="训练折不生成因子相关性聚类",
    )
    parser.add_argument(
        "--fdr-method", choices=["hierarchical"],
        default="hierarchical",
        help="训练折多重检验口径；默认使用验证策略中的层级FDR q",
    )
    parser.add_argument(
        "--reuse-artifacts", action="store_true",
        help="复用 --run-root 中已冻结的逐折研究 bundle，仅恢复组合回测",
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(_PROJECT_ROOT, config_path)
    base_config = load_config(config_path)
    if args.end:
        base_config.date_range.end = pd.Timestamp(args.end).date().isoformat()
    if args.practical_profile:
        _apply_practical_profile(base_config)
    if args.alpha_type:
        base_config.alpha.type = args.alpha_type
        params = dict(base_config.alpha.params)
        if args.alpha_type == "sector_grouped_ridge":
            params.update({
                "ridge_alphas": [0.01, 0.1, 1.0, 10.0],
                "ridge_cv_folds": 3,
            })
        else:
            params.pop("ridge_alphas", None)
            params.pop("ridge_cv_folds", None)
            params["ridge_alpha"] = 0.0
        base_config.alpha.params = params
    if base_config.alpha.type in {"sector_grouped_ols", "sector_grouped_ridge"}:
        params = dict(base_config.alpha.params)
        params["unmapped_sector_policy"] = "zero"
        base_config.alpha.params = params
    candidate_factors = None
    if args.candidate_factors:
        candidate_factors = [
            item.strip() for item in args.candidate_factors.split(",") if item.strip()
        ]
    fold_numbers = None
    if args.folds:
        fold_numbers = [
            int(item.strip()) for item in args.folds.split(",") if item.strip()
        ]
    run_root = Path(args.run_root).resolve()

    all_results = {}

    if not args.sens_only and not args.mc_only:
        wf_results = walk_forward_4fold(
            base_config,
            run_root=run_root,
            candidate_factors=candidate_factors,
            build_correlation=not args.no_correlation,
            max_folds=args.max_folds,
            fold_numbers=fold_numbers,
            fdr_method=args.fdr_method,
            reuse_artifacts=args.reuse_artifacts,
            frequency=args.frequency,
            is_intraday=args.is_intraday,
        )
        all_results["walk_forward"] = wf_results
        all_results["factor_fold_survival"] = summarize_factor_fold_survival(
            wf_results,
            minimum_fold_ratio=base_config.validation_policy.oos_fold_sign_ratio,
        )
    else:
        run_root.mkdir(parents=True, exist_ok=False)

    if not args.sens_only and not args.wf_only:
        mc_results = monte_carlo_perturbation(base_config, n_simulations=args.n_sim)
        all_results["monte_carlo"] = mc_results

    if not args.mc_only and not args.wf_only:
        sens_results = parameter_sensitivity(base_config)
        all_results["sensitivity"] = sens_results

    # 保存结果
    out_path = run_root / "walkforward_validation.json"

    def serialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [serialize(v) for v in obj]
        return obj

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(serialize(all_results), f, ensure_ascii=False, indent=2, default=str)
    print(f"\n所有验证结果已保存: {out_path}")

    print("\n" + "=" * 70)
    print("验证完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
