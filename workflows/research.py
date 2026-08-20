"""因子研究脚本 — 独立运行, 无需配置 PyCharm Run 参数.

Usage:
    # 研究配置文件中的因子 (默认)
    python main.py research

    # 研究指定因子
    python main.py research --factors momentum_20d,skewness_20d

    # 对指定因子执行完整多持有期筛选
    python main.py research --factors momentum_20d,skewness_20d --multi-period

    # 筛选所有已注册因子, 按 t 值排序 (单持有期, 用配置的 holding_period)
    python main.py research --all

    # 多持有期窗口匹配筛选 (推荐): 5d因子测3/5/10周期, 10d因子测5/10/20周期, 20d因子测10/20/40周期
    python main.py research --all --multi-period

    # 显式周期仅用于契约完全相同的冻结因子批次（例如同一 GP horizon）
    python main.py research --factors mined_gp_h4_a,mined_gp_h4_b --multi-period --periods 4

    # 指定周期单位；非日频使用真实 bar 索引
    python main.py research --all --multi-period --frequency daily

    # 单持有期探索性筛选可覆盖展示阈值；正式多周期筛选固定读取验证策略
    # 中的层级 FDR 与经济量级门槛。
    python main.py research --all --multi-period --t-threshold 1.74

    # 自定义日期范围 (IC检验区间; 因子计算自动提前1年预热)
    python main.py research --all --multi-period --start 2021-01-01 --end 2025-06-30

    # 因子相关性分析 + 聚类 (需先运行 --all --multi-period)
    python main.py research --correlation
    python main.py research --correlation --corr-method hierarchical
    python main.py research --correlation --corr-rolling 252
    python main.py research --correlation --corr-auto-threshold

周期数语义说明:
    - 因子 slug 中的 "5d" 后缀表示 "5个周期" (bar数), 不是 "5个日历天"
    - 持有期 (holding_period) 同样是 "周期数" 语义
    - 当 --frequency=daily (默认) 时, 1个周期 = 1个交易日
    - 当 --frequency=15min 时, 1个周期 = 1个15分钟bar
"""
from __future__ import annotations
import sys
import os
import re
import json
import math
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from factors.library import *  # noqa: F401,F403 注册所有因子
from core.period import (
    PeriodContext,
    parse_slug_window,
    parse_holding_periods,
)


def _infer_window(factor_name: str) -> str:
    """从因子名推断窗口标签 (5d/10d/20d/other).

    委托给 core.period.parse_slug_window, 保持向后兼容.
    """
    return parse_slug_window(factor_name) or "other"


def _safe_int(val, default: int = 0) -> int:
    """安全转 int, 处理 NaN/None/空字符串."""
    try:
        if val is None:
            return default
        # NaN 检测: NaN != NaN (无需依赖 pandas)
        if isinstance(val, float) and val != val:
            return default
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_str(val, default: str = "") -> str:
    """安全转 str, 处理 NaN/None."""
    try:
        if val is None:
            return default
        if isinstance(val, float) and val != val:
            return default
        return str(val)
    except (ValueError, TypeError):
        return default


def _joint_ic_ols_statistics(
    factor,
    forward_returns,
    *,
    forward_period: int,
    min_stocks: int = 10,
) -> dict:
    """Compute Pearson IC and raw univariate OLS from one matrix pass."""
    import numpy as np
    import pandas as pd
    from testing.ic_test import _newey_west_ir
    from testing.regression import _newey_west_t_stat

    common_dates = factor.index.intersection(forward_returns.index)
    common_cols = factor.columns.intersection(forward_returns.columns)
    if len(common_dates) == 0 or len(common_cols) < min_stocks:
        return {
            "ic": 0.0, "ic_hac_t": 0.0, "ir_nw": 0.0,
            "ic_pos_ratio": 0.0, "ic_n": 0, "ols_beta": 0.0,
            "ols_hac_t": 0.0, "ols_n": 0, "ols_days": 0,
        }

    x = factor.loc[common_dates, common_cols].to_numpy(
        dtype=float, copy=False
    )
    y = forward_returns.loc[common_dates, common_cols].to_numpy(
        dtype=float, copy=False
    )
    valid = np.isfinite(x) & np.isfinite(y)
    counts = valid.sum(axis=1)
    safe_counts = np.maximum(counts, 1)
    x_masked = np.where(valid, x, 0.0)
    y_masked = np.where(valid, y, 0.0)
    x_mean = x_masked.sum(axis=1) / safe_counts
    y_mean = y_masked.sum(axis=1) / safe_counts
    x_centered = np.where(valid, x - x_mean[:, None], 0.0)
    y_centered = np.where(valid, y - y_mean[:, None], 0.0)
    numerator = np.einsum("ij,ij->i", x_centered, y_centered)
    x_ss = np.einsum("ij,ij->i", x_centered, x_centered)
    y_ss = np.einsum("ij,ij->i", y_centered, y_centered)

    ic_denominator = np.sqrt(x_ss * y_ss)
    ic_usable = (
        (counts >= min_stocks)
        & np.isfinite(numerator)
        & np.isfinite(ic_denominator)
        & (ic_denominator > 0.0)
    )
    ic_values = np.divide(
        numerator,
        ic_denominator,
        out=np.full(len(common_dates), np.nan, dtype=float),
        where=ic_usable,
    )[ic_usable]
    ic_series = pd.Series(
        ic_values, index=pd.DatetimeIndex(common_dates)[ic_usable], dtype=float
    )
    ir_nw, ic_hac_t = _newey_west_ir(ic_series, forward_period)

    ols_usable = (
        (counts >= min_stocks)
        & np.isfinite(numerator)
        & np.isfinite(x_ss)
        & (x_ss > np.finfo(float).eps)
    )
    slope_values = np.divide(
        numerator,
        x_ss,
        out=np.full(len(common_dates), np.nan, dtype=float),
        where=ols_usable,
    )[ols_usable]
    slope_series = pd.Series(
        slope_values, index=pd.DatetimeIndex(common_dates)[ols_usable], dtype=float
    )
    ols_hac_t = _newey_west_t_stat(slope_series, forward_period)

    return {
        "ic": float(ic_values.mean()) if len(ic_values) else 0.0,
        "ic_hac_t": float(ic_hac_t),
        "ir_nw": float(ir_nw),
        "ic_pos_ratio": (
            float((ic_values > 0.0).mean()) if len(ic_values) else 0.0
        ),
        "ic_n": int(len(ic_values)),
        "ols_beta": (
            float(slope_values.mean()) if len(slope_values) else 0.0
        ),
        "ols_hac_t": float(ols_hac_t),
        "ols_n": int(len(slope_values)),
        "ols_days": int(slope_series.index.normalize().nunique()),
    }


def _parse_requested_factors(raw: str | None) -> list[str]:
    """Parse a stable, de-duplicated comma-separated factor list."""
    if raw is None:
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in raw.split(","):
        name = item.strip()
        if name and name not in seen:
            result.append(name)
            seen.add(name)
    if not result:
        raise ValueError("--factors must contain at least one factor name")
    return result


def _validate_requested_factors(
    requested: list[str], available: set[str]
) -> list[str]:
    """Reject unknown factor names before starting a research run."""
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"unregistered factors: {', '.join(missing)}")
    return requested


def _apply_global_bonferroni(
    results: list[dict], family_alpha: float = 0.05
) -> tuple[int, float]:
    """Apply the global gate and select a horizon only from approved tests.

    Each factor-period pair is one hypothesis. Rejected factors deliberately
    retain ``best_period == 0`` so diagnostics cannot be mistaken for an
    approved holding-period choice.
    """
    total_hypotheses = sum(
        len(factor_result.get("all_periods", {})) for factor_result in results
    )
    cutoff = family_alpha / max(total_hypotheses, 1)

    for factor_result in results:
        factor_result.update({
            "best_period": 0,
            "best_t": 0.0,
            "best_ic_t": 0.0,
            "best_p_value": 1.0,
            "best_ic": 0.0,
            "best_ir": 0.0,
            "best_ic_pos_ratio": 0.0,
            "best_ols_n": 0,
            "best_ols_days": 0,
        })
        approved_rows = []
        for label, values in factor_result.get("all_periods", {}).items():
            period = int(label.replace("period_", ""))
            raw_p = float(values.get("ols_p_value", 1.0))
            valid_p = bool(
                math.isfinite(raw_p) and int(values.get("ols_n", 0)) >= 2
            )
            is_approved = bool(valid_p and raw_p <= cutoff)
            values["bonferroni_adjusted_p"] = float(
                min(max(raw_p, 0.0) * total_hypotheses, 1.0)
                if valid_p else 1.0
            )
            values["bonferroni_significant"] = is_approved
            if is_approved:
                approved_rows.append((period, values))

        if approved_rows:
            best_period, best_values = min(
                approved_rows,
                key=lambda item: (
                    float(item[1].get("ols_p_value", 1.0)),
                    -abs(float(item[1].get("ols_hac_t", 0.0))),
                    item[0],
                ),
            )
            factor_result.update({
                "best_period": int(best_period),
                "best_t": float(best_values.get("ols_hac_t", 0.0)),
                "best_ic_t": float(best_values.get("ic_hac_t", 0.0)),
                "best_p_value": float(best_values.get("ols_p_value", 1.0)),
                "best_ic": float(best_values.get("ic", 0.0)),
                "best_ir": float(best_values.get("ir_nw", 0.0)),
                "best_ic_pos_ratio": float(
                    best_values.get("ic_pos_ratio", 0.0)
                ),
                "best_ols_n": int(best_values.get("ols_n", 0)),
                "best_ols_days": int(best_values.get("ols_days", 0)),
            })
        factor_result["bonferroni_significant"] = bool(approved_rows)

    return total_hypotheses, cutoff


def _apply_hierarchical_discovery(results: list[dict], policy) -> dict:
    """Apply the shared factor-level discovery policy and select a horizon."""
    from research.validation import apply_hierarchical_fdr

    entries_by_factor = {}
    for result in results:
        entries = list(result.get("all_periods", {}).values())
        for entry in entries:
            entry["estimable"] = int(entry.get("ols_n", 0) or 0) >= 2
        entries_by_factor[str(result["name"])] = entries

    audit = apply_hierarchical_fdr(
        entries_by_factor,
        q=float(policy.discovery_q),
        fwer_alpha=float(policy.fwer_report_alpha),
        p_key="ols_p_value",
        estimable_key="estimable",
    )
    for result in results:
        result.update({
            "best_period": 0,
            "best_variant": "",
            "best_t": 0.0,
            "best_ic_t": 0.0,
            "best_p_value": 1.0,
            "best_q_value": 1.0,
            "best_ic": 0.0,
            "best_ir": 0.0,
            "best_ic_pos_ratio": 0.0,
            "best_ols_n": 0,
            "best_ols_days": 0,
        })
        entries = entries_by_factor[str(result["name"])]
        approved = [
            entry for entry in entries
            if entry.get("hierarchical_fdr_significant", False)
        ]
        result["factor_fdr_significant"] = bool(
            entries and entries[0].get("factor_fdr_significant", False)
        )
        result["hierarchical_fdr_significant"] = bool(approved)
        result["bonferroni_significant"] = any(
            entry.get("fwer_significant", False) for entry in entries
        )
        if not approved:
            continue
        best = min(
            approved,
            key=lambda entry: (
                float(entry.get("local_q_value", 1.0)),
                float(entry.get("ols_p_value", 1.0)),
                -abs(float(entry.get("ols_hac_t", 0.0))),
                int(entry.get("period", 0)),
                str(entry.get("preprocessing_variant", "neutralized")),
            ),
        )
        result.update({
            "best_period": int(best.get("period", 0)),
            "best_variant": str(best.get("preprocessing_variant", "neutralized")),
            "best_t": float(best.get("ols_hac_t", 0.0)),
            "best_ic_t": float(best.get("ic_hac_t", 0.0)),
            "best_p_value": float(best.get("ols_p_value", 1.0)),
            "best_q_value": max(
                float(best.get("factor_q_value", 1.0)),
                float(best.get("local_q_value", 1.0)),
            ),
            "best_ic": float(best.get("ic", 0.0)),
            "best_ir": float(best.get("ir_nw", 0.0)),
            "best_ic_pos_ratio": float(best.get("ic_pos_ratio", 0.0)),
            "best_ols_n": int(best.get("ols_n", 0)),
            "best_ols_days": int(best.get("ols_days", 0)),
        })
    return audit


def _build_threshold_sensitivity(results: list[dict], policy) -> dict:
    """Report ±20% policy sensitivity without changing baseline decisions."""
    from research.validation import apply_hierarchical_fdr

    scenarios = {}
    labels = (("minus_20pct", 0.8), ("baseline", 1.0), ("plus_20pct", 1.2))
    for label, multiplier in labels:
        factor_entries = {
            str(result["name"]): [
                {
                    "p_value": float(entry.get("ols_p_value", 1.0)),
                    "estimable": bool(entry.get("estimable", False)),
                }
                for entry in result.get("all_periods", {}).values()
            ]
            for result in results
        }
        q_value = min(float(policy.discovery_q) * multiplier, 0.999999)
        discovery = apply_hierarchical_fdr(
            factor_entries,
            q=q_value,
            fwer_alpha=float(policy.fwer_report_alpha),
            estimable_key="estimable",
        )
        selected_names = {
            name
            for name, entries in factor_entries.items()
            if any(
                entry.get("hierarchical_fdr_significant", False)
                for entry in entries
            )
        }
        economically_qualified = []
        expected_directions = dict(policy.expected_directions or {})
        for result in results:
            name = str(result["name"])
            if name not in selected_names:
                continue
            original_entries = list(result.get("all_periods", {}).values())
            scenario_entries = factor_entries[name]
            approved = [
                (original, scenario)
                for original, scenario in zip(original_entries, scenario_entries)
                if scenario.get("hierarchical_fdr_significant", False)
            ]
            if not approved:
                continue
            best, _ = min(
                approved,
                key=lambda pair: (
                    float(pair[1].get("local_q_value", 1.0)),
                    float(pair[0].get("ols_p_value", 1.0)),
                    -abs(float(pair[0].get("ols_hac_t", 0.0))),
                    int(pair[0].get("period", 0)),
                    str(pair[0].get("preprocessing_variant", "neutralized")),
                ),
            )
            ic_value = float(best.get("ic", 0.0))
            t_value = float(best.get("ols_hac_t", 0.0))
            expected = expected_directions.get(name)
            observed = 1 if ic_value >= 0.0 else -1
            direction_ok = expected not in {-1, 1} or int(expected) == observed
            if (
                abs(ic_value) >= float(policy.min_abs_ic) * multiplier
                and abs(t_value) >= float(policy.min_abs_t) * multiplier
                and direction_ok
            ):
                economically_qualified.append(result)
        post_available = [
            result
            for result in economically_qualified
            if "turnover_below_reference" in result
        ]
        scenarios[label] = {
            "threshold_multiplier": multiplier,
            "discovery_q": q_value,
            "factors_with_local_fdr_discovery": len(selected_names),
            "local_fdr_factor_names": sorted(selected_names),
            "factors_passing_scaled_ic_t_direction": len(
                economically_qualified
            ),
            "economic_factor_names": sorted(
                str(result["name"]) for result in economically_qualified
            ),
            "post_metrics_available": len(post_available),
            "below_scaled_turnover_reference": sum(
                float(result.get("monthly_turnover", float("inf")))
                < float(policy.monthly_turnover_reference) * multiplier
                for result in post_available
            ),
            "passing_scaled_annual_ratios": sum(
                float(
                    result.get("calendar_year_robustness", {}).get(
                        "ic_sign_consistency", 0.0
                    )
                ) >= min(float(policy.annual_direction_ratio) * multiplier, 1.0)
                and float(
                    result.get("calendar_year_robustness", {}).get(
                        "effect_year_ratio", 0.0
                    )
                ) >= min(float(policy.annual_effect_ratio) * multiplier, 1.0)
                for result in post_available
            ),
            "passing_scaled_complete_cost_margin": sum(
                bool(result.get("cost_coverage", {}).get("complete", False))
                and float(
                    result.get("cost_coverage", {}).get(
                        "gross_annual_alpha", float("-inf")
                    )
                )
                >= float(policy.cost_safety_margin)
                * multiplier
                * float(
                    result.get("cost_coverage", {}).get(
                        "total_annual_cost", float("inf")
                    )
                )
                for result in post_available
            ),
            "discovery_audit": discovery,
        }
    baseline = scenarios["baseline"]
    for scenario in scenarios.values():
        for prefix in ("local_fdr", "economic"):
            current = set(scenario[f"{prefix}_factor_names"])
            reference = set(baseline[f"{prefix}_factor_names"])
            union = current | reference
            scenario[f"{prefix}_jaccard_vs_baseline"] = (
                float(len(current & reference) / len(union)) if union else 1.0
            )
    return {
        "report_only": True,
        "selection_uses_baseline_only": True,
        "post_metric_scope": "available_for_baseline_post_test_candidates_only",
        "scenarios": scenarios,
    }


FORMAL_IC_THRESHOLD = 0.01
FORMAL_T_THRESHOLD = 2.0
FORMAL_IC_POS_THRESHOLD = 0.52  # compatibility/reporting only


def _passes_post_bonferroni_quality(result: dict, policy=None) -> bool:
    """Apply economic-size gates after multiplicity-controlled discovery.

    HAC IC-IR remains a reported stability diagnostic. A fixed 0.50 cutoff is
    not used because this framework samples a small futures cross-section
    daily with overlapping forward returns, unlike the broad-equity monthly
    IC convention from which that heuristic was inherited.
    """
    best_ic = float(result.get("best_ic", 0.0))
    ic_floor = float(getattr(policy, "min_abs_ic", FORMAL_IC_THRESHOLD))
    t_floor = float(getattr(policy, "min_abs_t", FORMAL_T_THRESHOLD))
    return bool(
        abs(best_ic) >= ic_floor
        and abs(float(result.get("best_t", 0.0))) >= t_floor
    )


def _annotate_expected_direction(result: dict, policy) -> bool:
    expected = dict(getattr(policy, "expected_directions", {}) or {}).get(
        str(result.get("name", ""))
    )
    observed = 1 if float(result.get("best_ic", 0.0)) >= 0.0 else -1
    result["expected_direction"] = int(expected) if expected in {-1, 1} else None
    result["observed_direction"] = observed
    result["direction_predeclared"] = expected in {-1, 1}
    result["direction_matches_prior"] = bool(expected == observed) if expected else None
    result["direction_status"] = (
        "matches_predeclared" if expected == observed
        else "contradicts_predeclared" if expected in {-1, 1}
        else "exploratory_unregistered"
    )
    return bool(expected == observed) if expected in {-1, 1} else False


def result_direction_is_admissible(result: dict) -> bool:
    """Reject contradictions while retaining unregistered exploratory signs."""
    return result.get("direction_status") != "contradicts_predeclared"


def _require_valid_hypothesis_observations(results: list[dict]) -> int:
    """Fail a research batch that contains no estimable hypothesis at all."""

    valid_count = sum(
        int(values.get("ols_n", 0) or 0) >= 2
        for result in results
        for values in result.get("all_periods", {}).values()
        if isinstance(values, dict)
    )
    if valid_count == 0:
        raise RuntimeError(
            "research produced zero valid factor-period tests; check factor "
            "coverage, the point-in-time universe mask, and forward returns"
        )
    return int(valid_count)


def _load_adaptivity_data(csv_path: str | None = None) -> dict:
    """加载因子适配性研究结果（最小集成）.

    仅从显式指定的 CSV 加载, 返回 {factor_name: row_dict}.
    未指定文件时返回空字典，避免历史报告静默污染本次研究.

    适配性研究提供:
    - best_sector: 最佳板块
    - best_period: 最佳持有期 (适配性研究推荐)
    - valid_sectors: 有效板块列表 (|分隔)
    - n_valid_sectors: 有效板块数 (0 表示该因子在所有板块都无效)
    - decay_type: 衰减类型 (increasing/stable/decaying)
    """
    import pandas as pd
    if not csv_path:
        return {}
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"adaptivity CSV does not exist: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty:
        raise ValueError(f"adaptivity CSV is empty: {csv_path}")
    if "factor" not in df.columns:
        raise ValueError(f"adaptivity CSV has no 'factor' column: {csv_path}")
    # 转为 {factor_name: row_dict} 索引
    result = {}
    for _, row in df.iterrows():
        name = row.get("factor", "")
        if not name or not isinstance(name, str):
            continue
        result[name] = row.to_dict()
    return result


def _run_single_research(runner, factor_names, config_path):
    """运行一批因子的研究, 逐个打印结果."""
    print("=" * 60)
    print("因子研究模式")
    print("=" * 60)
    print(f"  配置文件: {config_path}")
    print(f"  因子数量: {len(factor_names)}")
    print(f"  日期范围: {runner.config.date_range.start} ~ {runner.config.date_range.end}")

    runner.config.factors = factor_names
    results = runner.run_factor_research()

    for factor_name, tests in results.items():
        print(f"\n[{factor_name}]")
        for test_name, result in tests.items():
            print(f"  {test_name}: {result.summary()}")

    print("\n研究完成.")


def _run_screening(runner, all_factors, config_path, t_threshold, output_dir):
    """批量筛选所有已注册因子 (单持有期), 按 |t| 排序输出."""
    import pandas as pd

    print("=" * 60)
    print(f"因子批量筛选 ({len(all_factors)} 个, 单持有期)")
    print("=" * 60)
    print(f"  配置文件: {config_path}")
    print(f"  t 值阈值: |t| ≥ {t_threshold}")
    print(f"  日期范围: {runner.config.date_range.start} ~ {runner.config.date_range.end}\n")

    results = []
    for i, factor_name in enumerate(all_factors, 1):
        print(f"[{i}/{len(all_factors)}] {factor_name} ... ", end="", flush=True)

        runner.config.factors = [factor_name]
        try:
            research = runner.run_factor_research()
        except Exception as e:
            print(f"失败: {e}")
            results.append({"factor": factor_name, "t_stat": 0, "status": "FAIL"})
            continue

        if factor_name not in research:
            print("无结果")
            results.append({"factor": factor_name, "t_stat": 0, "status": "NO_RESULT"})
            continue

        tests = research[factor_name]

        ic = tests.get("ic")
        ic_dict = ic.to_dict() if ic else {}
        ic_mean = ic_dict.get("ic_mean", 0)
        ir_val = ic_dict.get("ir", 0)
        t_stat = ic_dict.get("t_stat", 0)
        ic_pos = ic_dict.get("ic_pos_ratio", 0)

        layered = tests.get("layered")
        lay_dict = layered.to_dict() if layered else {}
        ls_return = lay_dict.get("group_metrics", {}).get("long_short", {}).get("annual_return", 0)
        mono = lay_dict.get("monotonicity", 0)

        reg = tests.get("regression")
        reg_dict = reg.to_dict() if reg else {}
        fm_t = list(reg_dict.get("t_stats", {}).values())
        r2 = reg_dict.get("avg_r2", 0)

        print(f"IC={ic_mean:.4f} t={t_stat:.2f} L/S={ls_return:.2%}")

        results.append({
            "factor": factor_name,
            "ic_mean": ic_mean,
            "ir": ir_val,
            "t_stat": t_stat,
            "ic_pos": ic_pos,
            "ls_return": ls_return,
            "mono": mono,
            "fm_t": fm_t[0] if fm_t else 0,
            "r2": r2,
            "status": "OK",
        })

    df = pd.DataFrame(results)
    df["abs_t"] = df["t_stat"].abs()
    df = df.sort_values("abs_t", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 80)
    print("因子筛选结果 (按 |t| 降序)")
    print("=" * 80)
    header = f"{'':>2}{'因子名':<30}{'IC_mean':>8}{'IR':>7}{'t_stat':>7}{'L/S':>8}{'Mono':>7}{'FM_t':>7}{'R²':>7}"
    print(header)
    print("-" * 80)

    significant = []
    for _, row in df.iterrows():
        if row["status"] != "OK":
            continue
        mark = "*" if abs(row["t_stat"]) >= t_threshold else " "
        if abs(row["t_stat"]) >= t_threshold:
            significant.append(row["factor"])
        print(
            f"{mark}{row.name+1:<1} "
            f"{row['factor']:<30}"
            f"{row['ic_mean']:>8.4f}"
            f"{row['ir']:>7.3f}"
            f"{row['t_stat']:>7.2f}"
            f"{row['ls_return']:>8.2%}"
            f"{row['mono']:>7.3f}"
            f"{row['fm_t']:>7.2f}"
            f"{row['r2']:>7.3f}"
        )

    print("-" * 80)
    print(f"\n显著因子 (|t| ≥ {t_threshold}): {len(significant)} 个")
    if significant:
        print("  " + ", ".join(significant))
        print("\n复制到 config/default.yaml 的 factors 列表:")
        print("  factors:")
        for f in significant:
            print(f"    - {f}")

    csv_path = os.path.join(output_dir, "factor_screening.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存: {csv_path}")
    print("筛选完成.")


def _run_multi_period_screening(runner, all_factors, config_path, t_threshold,
                                 factor_start, ic_start, ic_end,
                                 periods_override=None, frequency="daily",
                                 output_dir=None, adaptivity_file=None):
    """多持有期窗口匹配筛选 (推荐模式).

    持有期为"周期数"语义 (非天数); 当 frequency=daily 时, 1个周期=1个交易日.

    两种持有期选取模式:
      1. 窗口匹配模式 (默认, periods_override=None):
         - 5d 因子 → 测 3/5/10 周期持有期
         - 10d 因子 → 测 5/10/20 周期持有期
         - 20d 因子 → 测 10/20/40 周期持有期
         - 其他因子 → 测 1/5/10/20 周期持有期
      2. 显式持有期模式 (periods_override=[1,5,10,20,40]):
         - 仅允许因子注册契约与显式集合完全一致的同质批次
         - 不允许用该参数扩大或缩小冻结的因子内假设家族

    因子计算从 factor_start 开始 (含1年预热), IC 检验从 ic_start 开始.
    使用 Newey-West HAC 调整 t 统计量.

    Args:
        periods_override: 显式持有期列表 (周期数). None 表示用窗口匹配模式.
        frequency: 周期单位 ("daily"/"1min"/"5min"/"15min"/"30min"/"hourly").
                   非日度研究通过 FrequencyDataProvider 使用真实 bar 索引。
    """
    import numpy as np
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    from data.manager import FrequencyDataProvider
    from core.factor_contract import validate_factor_contract
    from core.registry import get as registry_get
    from factors.engine import FactorEngine
    from factors.processor import build_processing_context
    from research.governance import factor_family
    from research.validation import validate_policy, validation_policy_sha256
    from core.sectors import TAXONOMY_VERSION, taxonomy_sha256
    from scipy import stats as _scipy_stats

    policy = runner.config.validation_policy
    from factor_mining.bridge import registered_expected_directions

    mined_directions = registered_expected_directions(tuple(all_factors))
    if mined_directions:
        effective_directions = dict(policy.expected_directions or {})
        conflicts = {
            name: (effective_directions[name], direction)
            for name, direction in mined_directions.items()
            if name in effective_directions
            and int(effective_directions[name]) != int(direction)
        }
        if conflicts:
            raise ValueError(
                "mined snapshot expected directions conflict with validation policy: "
                f"{conflicts}"
            )
        effective_directions.update(mined_directions)
        policy.expected_directions = effective_directions
    validate_policy(policy)
    policy_hash = validation_policy_sha256(policy)
    _freq_lookup = frequency  # preserve "daily_intraday" before normalization
    minimum_test_bars = int(
        dict(policy.minimum_test_bars_by_frequency).get(
            _freq_lookup,
            dict(policy.minimum_test_bars_by_frequency)[
                PeriodContext.from_string(frequency).unit.value
            ]
        )
    )
    minimum_train_bars = int(
        dict(policy.minimum_train_bars_by_frequency).get(
            _freq_lookup,
            dict(policy.minimum_train_bars_by_frequency)[
                PeriodContext.from_string(frequency).unit.value
            ]
        )
    )
    minimum_test_days = int(
        dict(policy.minimum_test_days_by_frequency).get(
            _freq_lookup,
            dict(policy.minimum_test_days_by_frequency)[
                PeriodContext.from_string(frequency).unit.value
            ]
        )
    )
    minimum_train_days = int(
        dict(policy.minimum_train_days_by_frequency).get(
            _freq_lookup,
            dict(policy.minimum_train_days_by_frequency)[
                PeriodContext.from_string(frequency).unit.value
            ]
        )
    )
    explicit_family_map = dict(
        getattr(runner.config.factor_governance, "explicit_family_map", {}) or {}
    )
    dual_track_families = set(policy.dual_track_families)

    def _factor_family(name: str) -> str:
        return factor_family(name, explicit_family_map)
    # 构造周期上下文；分钟数据加载后会用实际 bar 数校准年化因子。
    period_ctx = PeriodContext.from_string(frequency)

    # Freeze the exact factor-frequency-horizon contract before any data read.
    # Formal research must never broaden a factor's hypothesis family through
    # a CLI or family override that disagrees with its registered contract.
    factor_horizons: dict[str, tuple[int, ...]] = {}
    factor_training_bars: dict[str, int] = {}
    factor_training_days: dict[str, int] = {}
    factor_requires_training_contract: dict[str, bool] = {}
    for factor_name in all_factors:
        factor = registry_get("factor", factor_name)()
        requested = (
            tuple(periods_override)
            if periods_override is not None
            else tuple(
                (policy.family_horizons or {}).get(_factor_family(factor_name))
                or getattr(factor, "validation_horizons", ())
            )
        )
        factor_horizons[factor_name] = validate_factor_contract(
            factor,
            provider_frequency=period_ctx.unit.value,
            requested_horizons=requested,
        )
        factor_training_bars[factor_name] = int(
            getattr(factor, "training_bars", 0) or 0
        )
        factor_training_days[factor_name] = int(
            getattr(factor, "training_days", 0) or 0
        )
        factor_requires_training_contract[factor_name] = bool(
            getattr(factor, "requires_training_sample_contract", False)
        )

    print("=" * 60)
    print(f"因子多持有期窗口匹配筛选 ({len(all_factors)} 个)")
    print("=" * 60)
    print(f"  配置文件: {config_path}")
    print(f"  因子计算区间: {factor_start.date()} ~ {ic_end.date()} (含预热)")
    print(f"  IC检验区间: {ic_start.date()} ~ {ic_end.date()}")
    print(
        "  discovery gate: factor Simes/BH + selection-adjusted local BH "
        f"(q={policy.discovery_q:.2f}); FWER is report-only"
    )
    print(f"  周期单位: {period_ctx.unit.value} (1个周期 = "
          f"{period_ctx.bars_per_day}个bar/交易日)")

    # 持有期选取模式
    if periods_override is not None:
        print(
            f"  持有期模式: 冻结显式列表 {periods_override} "
            "(必须与每个因子契约完全一致)"
        )
    else:
        print(f"  持有期模式: 窗口匹配 (5d→[3,5,10], 10d→[5,10,20], 20d→[10,20,40])")

    # 适配性输出只作为部署元数据，不能预先缩小发现阶段的假设家族。
    adaptivity_data = _load_adaptivity_data(adaptivity_file)
    n_filtered = 0  # 预初始化, 避免作用域问题
    if adaptivity_data:
        print(
            f"  适配性研究: 已加载 {len(adaptivity_data)} 个因子记录，"
            "仅用于部署标注，发现阶段移除 0 个候选"
        )
        print()
    else:
        print("  适配性研究: 未指定有效输入文件，使用纯窗口匹配")
        print()

    # 按窗口分组 (仍需要窗口信息用于结果展示, 即使持有期被覆盖)
    window_groups: dict[str, list[str]] = {}
    for f in all_factors:
        w = _infer_window(f)
        window_groups.setdefault(w, []).append(f)
    print("  分组: " + ", ".join(f"{k}={len(v)}" for k, v in window_groups.items()))
    print()

    base_data_mgr = runner.data_manager
    if getattr(base_data_mgr.source, "formal_research_safe", True) is False:
        raise RuntimeError(
            f"data source {base_data_mgr.source.__class__.__name__} does not provide "
            "an auditable continuous-contract roll ledger and is not allowed in "
            "formal research"
        )
    universe = pd.Index(runner.config.universe) if runner.config.universe else pd.Index([])

    if period_ctx.is_daily:
        data_mgr = base_data_mgr
        calendar = data_mgr.get_calendar(factor_start, ic_end)
    else:
        data_mgr = FrequencyDataProvider(
            base_data_mgr, frequency, factor_start, ic_end, universe
        )
        calendar = data_mgr.get_calendar()
    if hasattr(calendar, "tz") and calendar.tz is not None:
        calendar = calendar.tz_localize(None)
    calendar = pd.DatetimeIndex(sorted(set(calendar)))

    if len(calendar) == 0:
        raise RuntimeError(
            f"published Parquet returned an empty {frequency} research calendar"
        )

    calendar = pd.DatetimeIndex(sorted(set(calendar)))
    if not period_ctx.is_daily and len(calendar):
        close_for_frequency = data_mgr.get("close", calendar, universe)
        per_instrument_day = (
            close_for_frequency.notna()
            .groupby(close_for_frequency.index.normalize())
            .sum()
            .stack()
        )
        active_counts = per_instrument_day.loc[per_instrument_day.gt(0)]
        if not active_counts.empty:
            observed_bars_per_day = max(int(round(active_counts.median())), 1)
            period_ctx = PeriodContext(
                unit=period_ctx.unit,
                bars_per_day=observed_bars_per_day,
                bars_per_year=observed_bars_per_day * 252,
            )
            print(
                f"  observed annualization: {observed_bars_per_day} bars/instrument-day, "
                f"{period_ctx.bars_per_year} bars/year"
            )
    print(f"交易日历: {len(calendar)} 天, universe: {len(universe)} 品种")

    # 预计算各持有期的 forward returns
    print("\n预计算 forward returns...")
    processing_context = build_processing_context(
        data_mgr,
        calendar,
        universe,
        runner.config.universe_selection,
    )
    fwd_returns_by_period: dict[int, pd.DataFrame] = {}
    # 持有期集合: 显式模式 → periods_override; 窗口匹配模式 → 各窗口并集
    all_periods = sorted({
        period
        for factor_name in all_factors
        for period in factor_horizons[factor_name]
    })
    for p in all_periods:
        forward_returns = data_mgr.get_forward_returns(
            calendar, universe, period=p
        )
        if processing_context.eligibility is not None:
            forward_returns = forward_returns.where(
                processing_context.eligibility
            )
        fwd_returns_by_period[p] = forward_returns
    print(f"  持有期 (周期数): {all_periods}")

    # Stream each factor batch through inference, then release its matrices.
    # This keeps census peak memory bounded by the batch size.
    print("\n=== 开始 IC 检验 (流式因子分块) ===")
    results = []
    total = len(all_factors)
    t0 = time.time()
    default_chunk_size = 64 if period_ctx.is_daily else 16
    research_chunk_size = max(
        int(os.environ.get("MF_RESEARCH_FACTOR_CHUNK_SIZE", default_chunk_size)),
        1,
    )
    analysis_workers = max(
        int(os.environ.get(
            "MF_RESEARCH_ANALYSIS_WORKERS", min(8, os.cpu_count() or 1)
        )),
        1,
    )
    print(f"  因子分块大小: {research_chunk_size}")
    print(f"  因子检验线程: {analysis_workers}")

    for batch_start in range(0, total, research_chunk_size):
        batch_names = all_factors[
            batch_start:batch_start + research_chunk_size
        ]
        # Discovery scans the entire registered library, including optional
        # factors.  Failures are recorded instead of aborting unrelated tests.
        engine = FactorEngine(data_mgr, tolerant=True)
        computed_batch = engine.compute_factors(
            batch_names,
            calendar,
            universe,
            parallel=True,
            chunk_size=research_chunk_size,
        )
        factor_batch = runner.processor.process_batch(
            computed_batch, processing_context
        )
        if engine.failures:
            print(f"  本批不可计算因子/依赖: {len(engine.failures)}（已记入日志）")
        raw_variant_batch = {
            name: runner.processor.process_excluding(
                computed_batch[name], processing_context, {"neutralize"}
            )
            for name in batch_names
            if name in computed_batch
            and _factor_family(name) in dual_track_families
        }

        def _evaluate_factor(fname):
            if fname not in factor_batch:
                return None
            window = _infer_window(fname)
            periods = list(factor_horizons[fname])
            all_period_results = {}
            variants = {"neutralized": factor_batch[fname]}
            if fname in raw_variant_batch:
                variants["raw"] = raw_variant_batch[fname]
            for variant, matrix in variants.items():
                f_ic = matrix.loc[ic_start:ic_end]
                for p in periods:
                    fwd = fwd_returns_by_period[p].loc[ic_start:ic_end]
                    estimation_failure = None
                    try:
                        stats = _joint_ic_ols_statistics(
                            f_ic,
                            fwd,
                            forward_period=p,
                            min_stocks=10,
                        )
                        t_stat = stats["ic_hac_t"]
                        ic_mean = stats["ic"]
                        ir_nw = stats["ir_nw"]
                        ic_pos_ratio = stats["ic_pos_ratio"]
                        ols_t = stats["ols_hac_t"]
                        ols_p = float(
                            2.0 * _scipy_stats.t.sf(
                                abs(ols_t), df=max(stats["ols_n"] - 1, 1)
                            )
                        )
                        ols_beta = stats["ols_beta"]
                        ols_n = stats["ols_n"]
                        ols_days = stats["ols_days"]
                        ic_n = stats["ic_n"]
                    except Exception as exc:
                        estimation_failure = {
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                        t_stat = 0.0
                        ic_mean = 0.0
                        ir_nw = 0.0
                        ic_pos_ratio = 0.0
                        ols_t = 0.0
                        ols_p = 1.0
                        ols_beta = 0.0
                        ols_n = 0
                        ols_days = 0
                        ic_n = 0

                    label = f"{variant}_period_{p}"
                    all_period_results[label] = {
                        "period": int(p),
                        "preprocessing_variant": variant,
                        "ic": float(ic_mean),
                        "ic_hac_t": float(t_stat),
                        "t": float(ols_t),
                        "ols_beta": float(ols_beta),
                        "ols_hac_t": float(ols_t),
                        "ols_p_value": float(ols_p),
                        "ols_n": int(ols_n),
                        "ols_days": int(ols_days),
                        "inference_model": "unpenalized_univariate_fama_macbeth_ols_hac",
                        "ir_nw": float(ir_nw),
                        "ic_pos_ratio": float(ic_pos_ratio),
                        "n": int(ic_n),
                    }
                    if estimation_failure is not None:
                        all_period_results[label]["estimation_failure"] = estimation_failure

            return {
                "name": fname,
                "window": window,
                "best_period": 0,
                "best_t": 0.0,
                "best_ic_t": 0.0,
                "best_p_value": 1.0,
                "best_ic": 0.0,
                "best_ir": 0.0,
                "best_ic_pos_ratio": 0.0,
                "all_periods": all_period_results,
                "n_periods_tested": len(periods),
                "training_bars": factor_training_bars[fname],
                "training_days": factor_training_days[fname],
                "requires_training_sample_contract": (
                    factor_requires_training_contract[fname]
                ),
                "adaptivity_best_sector": _safe_str(adaptivity_data.get(fname, {}).get("best_sector", "")),
                "adaptivity_valid_sectors": _safe_str(adaptivity_data.get(fname, {}).get("valid_sectors", "")),
                "adaptivity_n_valid_sectors": _safe_int(adaptivity_data.get(fname, {}).get("n_valid_sectors", 0)),
                "adaptivity_recommended_period": _safe_int(adaptivity_data.get(fname, {}).get("recommended_period", 0)),
                "adaptivity_decay_type": _safe_str(adaptivity_data.get(fname, {}).get("decay_type", "")),
            }

        if analysis_workers == 1 or len(batch_names) <= 1:
            evaluated = map(_evaluate_factor, batch_names)
            pool = None
        else:
            pool = ThreadPoolExecutor(
                max_workers=min(analysis_workers, len(batch_names))
            )
            evaluated = pool.map(_evaluate_factor, batch_names)
        try:
            for factor_result in evaluated:
                if factor_result is not None:
                    results.append(factor_result)
        finally:
            if pool is not None:
                pool.shutdown(wait=True)

        engine.clear_cache()
        del computed_batch, raw_variant_batch
        del factor_batch
        done = min(batch_start + research_chunk_size, total)
        print(f"  [{done}/{total}] 耗时 {time.time() - t0:.1f}s")

    valid_hypotheses = _require_valid_hypothesis_observations(results)

    # 筛选显著因子 (业界标准筛选流程)
    # 正式门槛由层级 FDR、|IC|/|t|、预声明方向和后置交易属性共同治理。

    # 第一阶段: 对每个预声明因子×持有期×预处理版本的原始单因子
    # OLS/HAC p 值执行层级 FDR。Ridge 尚未参与，也不产生筛选 p 值。
    discovery_audit = _apply_hierarchical_discovery(results, policy)
    total_hypotheses = int(discovery_audit["total_hypotheses"])
    bonferroni_alpha = float(discovery_audit["fwer_cutoff"])
    bonferroni_t_threshold = float(
        _scipy_stats.norm.ppf(1 - bonferroni_alpha / 2)
    )

    print(
        f"\nHierarchical FDR: factors={discovery_audit['factor_family_count']}, "
        f"q={policy.discovery_q:.3f}, selected="
        f"{discovery_audit['selected_factor_count']}, local_alpha="
        f"{discovery_audit['local_alpha']:.6f}"
    )
    print(
        f"  report-only FWER: m={total_hypotheses}, "
        f"alpha={bonferroni_alpha:.6g}, reference |t|="
        f"{bonferroni_t_threshold:.3f}"
    )

    t_significant = [
        r for r in results if r["hierarchical_fdr_significant"]
    ]
    for result in t_significant:
        _annotate_expected_direction(result, policy)
    # 第二阶段: 经济量级与方向稳定性门槛
    significant = [
        r for r in t_significant
        if _passes_post_bonferroni_quality(r, policy)
        and result_direction_is_admissible(r)
    ]
    rejected_by_quality = [
        r for r in t_significant
        if not (
            _passes_post_bonferroni_quality(r, policy)
            and result_direction_is_admissible(r)
        )
    ]
    significant.sort(key=lambda x: abs(x["best_t"]), reverse=True)
    governance_cfg = getattr(runner.config, "factor_governance", None)
    governance_enabled = bool(
        governance_cfg is not None and governance_cfg.enabled
    )
    family_governance_audit = {
        "enabled": governance_enabled,
        "applied": False,
        "deferred_until": "after_correlation_deduplication",
    }

    by_period: dict[int, list[str]] = {}
    for r in significant:
        by_period.setdefault(r["best_period"], []).append(r["name"])

    # 统计校正信息；Bonferroni/FWER 只保留报告标签。
    corr_info = (
        f"Simes/BH q={policy.discovery_q:.2f}; local BH alpha="
        f"{discovery_audit['local_alpha']:.3g}; FWER report p≤{bonferroni_alpha:.3g}"
    )

    print("\n" + "=" * 80)
    print(
        f"OLS/IC 检验完成: 层级FDR局部发现 {len(t_significant)} 个 "
        f"→ 经济量级与预声明方向后 {len(significant)} 个"
    )
    print(
        f"  门槛: |IC|≥{policy.min_abs_ic}, |t|≥{policy.min_abs_t}; "
        "命中率与IR稳定性进入记分卡"
    )
    print(f"  多重检验: {corr_info}")
    if rejected_by_quality:
        print(f"  经济质量门槛淘汰: {len(rejected_by_quality)} 个 (t显著但IC/命中率不达标)")
    print("=" * 80)
    print(f"\n按最优持有期分组:")
    for p in sorted(by_period.keys()):
        print(f"  {p}日: {len(by_period[p])} 个")

    print(f"\n显著因子清单 (按 |t| 降序, 含适配性信息):")
    print(f"{'#':>3} {'因子名':<35} {'窗口':>5} {'持有期':>5} {'t值':>7} {'阈值':>7} {'IC':>8} {'IR':>6} {'命中率':>6} {'最佳板块':>8} {'有效板块数':>8}")
    print("-" * 115)
    for i, r in enumerate(significant, 1):
        best_sec = r.get("adaptivity_best_sector", "") or "-"
        n_valid = r.get("adaptivity_n_valid_sectors", 0)
        print(f"{i:>3} {r['name']:<35} {r['window']:>5} {r['best_period']:>5}d "
              f"{r['best_t']:>7.2f} {bonferroni_t_threshold:>7.2f} {r['best_ic']:>8.4f} "
              f"{r['best_ir']:>6.3f} {r['best_ic_pos_ratio']:>6.1%} {best_sec:>8} {n_valid:>8}")

    # ===================================================================
    # 后置检验: 分层单调性 + 换手率 + 稳健性 (仅对IC+IR显著因子)
    # ===================================================================
    final_factors: list = []  # 预初始化, 避免后续引用未定义变量
    candidate_return_series: dict[str, "pd.Series"] = {}
    selection_diagnostics = {
        "deflated_sharpe": {
            "n_trials": int(total_hypotheses),
            "return_sampling": "non_overlapping_at_selected_holding_period",
        },
        "pbo_cscv": {"pbo": None, "n_splits": 0, "n_candidates": 0},
    }
    if significant:
        from testing.layered import LayeredBacktest
        from testing.turnover import TurnoverTest
        from testing.robustness import CalendarYearRobustnessTest
        from optimization.costs import factor_cost_coverage
        from research.statistics import (
            deflated_sharpe_ratio,
            probability_backtest_overfitting,
        )

        print("\n=== 后置检验: 分层单调性 / 换手率 / 稳健性 ===")
        post_names = [row["name"] for row in significant]
        post_engine = FactorEngine(data_mgr)
        computed_post_matrices = post_engine.compute_factors(
            post_names,
            calendar,
            universe,
            parallel=True,
            chunk_size=max(len(post_names), 1),
        )
        factor_matrices = runner.processor.process_batch(
            computed_post_matrices, processing_context
        )
        layered_test = LayeredBacktest(n_groups=policy.n_return_groups)
        turnover_test = TurnoverTest(
            monthly_threshold=policy.monthly_turnover_reference
        )
        scorecard = policy.scorecard
        robustness_test = CalendarYearRobustnessTest(
            min_ic_abs=policy.min_abs_ic,
            direction_ratio=policy.annual_direction_ratio,
            effect_ratio=policy.annual_effect_ratio,
            minimum_years=(
                policy.intraday_minimum_calendar_years
                if (not period_ctx.is_daily or "intraday" in str(frequency))
                else policy.minimum_calendar_years
            ),
            minimum_days_per_year=policy.minimum_year_observations,
            bootstrap_samples=policy.single_instrument_bootstrap_samples,
            scorecard_weights=dict(scorecard.weights),
            hit_rate_threshold=policy.deployment_hit_rate,
            ir_std_max=scorecard.ir_std_max,
            scorecard_threshold=scorecard.threshold,
            scorecard_enforced=scorecard.enforced,
        )

        for r in significant:
            fname = r["name"]
            best_p = r["best_period"]
            if fname not in factor_matrices:
                continue
            if r.get("best_variant") == "raw":
                selected_matrix = runner.processor.process_excluding(
                    computed_post_matrices[fname],
                    processing_context,
                    {"neutralize"},
                )
            else:
                selected_matrix = factor_matrices[fname]
            f_mat = selected_matrix.loc[ic_start:ic_end]
            fwd = fwd_returns_by_period[best_p].loc[ic_start:ic_end]

            # 分层单调性
            failures = r.setdefault("post_test_failures", [])
            try:
                layered_res = layered_test.run(
                    f_mat,
                    fwd,
                    holding_period=best_p,
                    periods_per_year=period_ctx.bars_per_year,
                )
                r["layered_monotonicity"] = float(layered_res.monotonicity)
                r["layered_ls_return"] = float(
                    layered_res.group_metrics.get("long_short", {}).get("annual_return", 0.0)
                )
                direction = 1.0 if r["best_ic"] >= 0 else -1.0
                oriented_returns = (
                    layered_res.group_returns["long_short"].dropna() * direction
                )
                candidate_return_series[fname] = oriented_returns
                non_overlapping = oriented_returns.iloc[::max(int(best_p), 1)]
                dsr = deflated_sharpe_ratio(
                    non_overlapping,
                    n_trials=total_hypotheses,
                    periods_per_year=max(
                        int(period_ctx.bars_per_year / max(int(best_p), 1)), 1
                    ),
                    risk_free_rate=0.0,
                )
                r["deflated_sharpe"] = dsr
                r["passes_dsr_95"] = bool(dsr["probability"] >= 0.95)
            except Exception as exc:
                r["layered_monotonicity"] = 0.0
                r["layered_ls_return"] = 0.0
                r["deflated_sharpe"] = {
                    "sharpe": 0.0,
                    "expected_max_sharpe": 0.0,
                    "probability": 0.0,
                    "n_obs": 0,
                    "n_trials": int(total_hypotheses),
                    "risk_free_rate": 0.0,
                }
                r["passes_dsr_95"] = False
                failures.append({
                    "stage": "layered_and_dsr",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                })

            # 换手率
            try:
                turnover_res = turnover_test.run(
                    f_mat, rebalance_every=best_p
                )
                r["monthly_turnover"] = float(turnover_res.monthly_turnover)
                r["annual_half_turnover"] = float(turnover_res.annual_turnover)
                r["mean_absolute_weights"] = dict(
                    turnover_res.mean_absolute_weights
                )
                r["turnover_definition"] = (
                    "half_turnover_0.5_sum_abs_delta_weight"
                )
                r["turnover_below_reference"] = bool(
                    turnover_res.monthly_turnover
                    < policy.monthly_turnover_reference
                )
            except Exception as exc:
                r["monthly_turnover"] = 0.0
                r["turnover_below_reference"] = False
                failures.append({
                    "stage": "turnover",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                })

            # 稳健性 (时间分段)
            try:
                orientation = 1 if r["best_ic"] >= 0.0 else -1
                robust_res = robustness_test.run(
                    f_mat, fwd, orientation=orientation
                )
                r["ic_sign_consistency"] = float(robust_res.ic_sign_consistency)
                r["ir_std"] = float(robust_res.ir_std)
                r["calendar_year_robustness"] = robust_res.to_dict()
                r["passes_robustness"] = bool(robust_res.passes_scorecard)
                r["observation_channel"] = bool(robust_res.observation_channel)
            except Exception as exc:
                r["ic_sign_consistency"] = 0.0
                r["ir_std"] = 0.0
                r["passes_robustness"] = False
                failures.append({
                    "stage": "robustness",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                })

            cost_coverage = factor_cost_coverage(
                gross_annual_alpha=float(r.get("layered_ls_return", 0.0))
                * (1.0 if r["best_ic"] >= 0.0 else -1.0),
                annual_half_turnover=float(r.get("annual_half_turnover", 0.0)),
                annual_roll_cost=getattr(
                    runner.cost_model, "annual_roll_cost", None
                ),
                annual_fee=float(
                    getattr(runner.cost_model, "annual_fee", 0.0)
                ),
                safety_margin=policy.cost_safety_margin,
                turnover_cost_rate=float(
                    getattr(
                        runner.cost_model,
                        "turnover_cost_rate",
                        0.0002,
                    )
                ),
                include_roll_cost=False,
                cost_stage="factor_validation",
            )
            r["cost_coverage"] = cost_coverage
            observation_reasons = []
            r["sample_test_bars"] = int(r.get("best_ols_n", 0))
            r["minimum_test_bars"] = minimum_test_bars
            r["sample_test_days"] = int(r.get("best_ols_days", 0))
            r["minimum_test_days"] = minimum_test_days
            r["sample_sufficient"] = bool(
                r["sample_test_bars"] >= minimum_test_bars
                and r["sample_test_days"] >= minimum_test_days
            )
            if not r["sample_sufficient"]:
                if r["sample_test_bars"] < minimum_test_bars:
                    observation_reasons.append("insufficient_test_bars")
                if r["sample_test_days"] < minimum_test_days:
                    observation_reasons.append("insufficient_test_days")
            r["minimum_train_bars"] = minimum_train_bars
            r["minimum_train_days"] = minimum_train_days
            if (
                r.get("requires_training_sample_contract", False)
                and (
                    int(r.get("training_bars", 0)) <= 0
                    or int(r.get("training_days", 0)) <= 0
                )
            ):
                r["sample_sufficient"] = False
                observation_reasons.append("missing_frozen_training_sample")
            elif int(r.get("training_bars", 0)) > 0:
                if int(r["training_bars"]) < minimum_train_bars:
                    r["sample_sufficient"] = False
                    observation_reasons.append("insufficient_train_bars")
                if int(r.get("training_days", 0)) < minimum_train_days:
                    r["sample_sufficient"] = False
                    observation_reasons.append("insufficient_train_days")
                ratio = int(r["training_bars"]) / max(r["sample_test_bars"], 1)
                day_ratio = int(r["training_days"]) / max(r["sample_test_days"], 1)
                r["train_test_ratio"] = float(ratio)
                r["train_test_day_ratio"] = float(day_ratio)
                if ratio < float(policy.minimum_train_test_ratio):
                    r["sample_sufficient"] = False
                    observation_reasons.append("insufficient_train_test_ratio")
                if day_ratio < float(policy.minimum_train_test_ratio):
                    r["sample_sufficient"] = False
                    observation_reasons.append(
                        "insufficient_train_test_day_ratio"
                    )
            if r.get("observation_channel", False):
                observation_reasons.append("fewer_than_minimum_calendar_years")
            if not cost_coverage["complete"]:
                observation_reasons.append(cost_coverage["observation_reason"])
            if (
                policy.require_predeclared_direction_for_promotion
                and not r.get("direction_predeclared", False)
            ):
                observation_reasons.append("economic_direction_not_predeclared")
            if scorecard.enabled and not scorecard.calibrated:
                observation_reasons.append("scorecard_not_pilot_calibrated")
            observation_reasons.append("locked_oos_pending")
            r["observation_reasons"] = sorted(set(observation_reasons))
            r["observation_channel"] = bool(observation_reasons)
            r["promotion_status"] = (
                "observation" if observation_reasons else "wf_candidate"
            )
            r["weight_cap"] = (
                policy.observation_weight_cap
                if observation_reasons else 1.0
            )

        if len(candidate_return_series) >= 2:
            candidate_frame = pd.concat(
                candidate_return_series, axis=1, join="inner"
            ).dropna(how="any")
            pbo = probability_backtest_overfitting(
                candidate_frame, n_partitions=8
            )
            pbo_value = pbo.get("pbo")
            if pbo_value is not None and not np.isfinite(pbo_value):
                pbo_value = None
            selection_diagnostics["pbo_cscv"] = {
                "pbo": pbo_value,
                "n_splits": int(pbo.get("n_splits", 0)),
                "n_candidates": int(candidate_frame.shape[1]),
                "n_observations": int(candidate_frame.shape[0]),
                "time_partitioning": "8_contiguous_blocks",
                "logits": pbo.get("logits", []),
            }

        # 打印后置检验摘要
        print(f"\n{'#':>3} {'因子名':<35} {'单调性':>6} {'多空':>7} {'月换手':>7} {'IC一致':>7} {'稳健':>4}")
        print("-" * 85)
        for i, r in enumerate(significant, 1):
            robust_str = "PASS" if r.get("passes_robustness") else "FAIL"
            print(f"{i:>3} {r['name']:<35} {r.get('layered_monotonicity',0):>6.3f} "
                  f"{r.get('layered_ls_return',0):>7.2%} {r.get('monthly_turnover',0):>7.2%} "
                  f"{r.get('ic_sign_consistency',0):>7.1%} {robust_str:>4}")

        # Training candidates may proceed through the observation channel.
        # Monotonicity is diagnostic; three-group spread and complete costs
        # carry the economic meaning.
        economically_eligible = [
            r for r in significant
            if r.get("passes_robustness", False)
            and (
                not r.get("cost_coverage", {}).get("complete", False)
                or r.get("cost_coverage", {}).get("passes", False)
            )
        ]
        # Insufficient samples remain visible as observations but cannot enter
        # the walk-forward capital path.
        final_factors = [
            r for r in economically_eligible
            if r.get("sample_sufficient", False)
        ]
        sample_observations = [
            r for r in economically_eligible
            if not r.get("sample_sufficient", False)
        ]
        rejected_final = [r for r in significant if r not in economically_eligible]
        observations = [
            r for r in economically_eligible if r.get("observation_channel", False)
        ]
        print(
            f"\n进入WF候选: {len(final_factors)} 个 "
            f"(样本不足观察 {len(sample_observations)} 个，全部观察标记 "
            f"{len(observations)} 个，硬门槛淘汰 "
            f"{len(rejected_final)} 个)"
        )

    # 保存 JSON 结果
    final_factor_names = (
        [r["name"] for r in final_factors] if significant else []
    )
    funnel_audit = {
        "factors_submitted": len(results),
        "hypotheses_declared": total_hypotheses,
        "hypotheses_estimable": valid_hypotheses,
        "factors_selected_by_simes_bh": int(
            discovery_audit["selected_factor_count"]
        ),
        "factors_with_local_fdr_discovery": len(t_significant),
        "factors_passing_ic_t_direction": len(significant),
        "factors_below_turnover_reference": sum(
            bool(row.get("turnover_below_reference", False))
            for row in significant
        ),
        "turnover_policy": "diagnostic_only_not_an_admission_gate",
        "factors_with_complete_cost_coverage": sum(
            bool(row.get("cost_coverage", {}).get("passes", False))
            for row in significant
        ),
        "wf_candidates": len(final_factors),
        "observation_candidates": sum(
            bool(row.get("observation_channel", False))
            for row in final_factors
        ),
        "production_approved": 0,
    }
    threshold_sensitivity = _build_threshold_sensitivity(results, policy)
    out = {
        "config": {
            "factor_start": str(factor_start.date()),
            "ic_start": str(ic_start.date()),
            "ic_end": str(ic_end.date()),
            "t_threshold": bonferroni_t_threshold,
            "bonferroni_correction": False,
            "fwer_role": "report_only_not_a_gate",
            "correction_formula": discovery_audit["method"],
            "validation_policy": validate_policy(policy),
            "validation_policy_sha256": policy_hash,
            "taxonomy_version": TAXONOMY_VERSION,
            "taxonomy_sha256": taxonomy_sha256(),
            "inference_model": "unpenalized_univariate_fama_macbeth_ols_hac",
            "factor_preprocessing": [
                "mad_winsorize_by_date",
                "sector_neutralize_by_date_for_neutralized_variant",
                "zscore_standardize_by_date",
            ],
            "factor_preprocessing_variants": {
                "neutralized": [
                    "mad_winsorize_by_date",
                    "sector_neutralize_by_date",
                    "zscore_standardize_by_date",
                ],
                "raw": [
                    "mad_winsorize_by_date",
                    "zscore_standardize_by_date",
                ],
            },
            "selection_order": [
                "predeclared_exposure_preprocessing",
                "raw_ols_hac_p_values",
                "factor_simes_bh",
                "selection_adjusted_local_bh",
                "economic_and_robustness_gates",
                "layered_turnover_stability_diagnostics",
            ],
            "downstream_required_order": [
                "correlation_deduplication",
                "family_governance",
                "training_only_ridge",
            ],
            "total_hypotheses": total_hypotheses,
            "valid_hypotheses": valid_hypotheses,
            "bonferroni_alpha": bonferroni_alpha,
            "ic_threshold": policy.min_abs_ic,
            # 周期架构信息 (新增字段, 向后兼容)
            "frequency": period_ctx.unit.value,
            "periods_mode": "explicit" if periods_override is not None else "window_match",
            "periods_override": periods_override,
            "periods_semantics": "周期数 (非天数); daily频率下1周期=1交易日",
            "bars_per_year": period_ctx.bars_per_year,
            "ir_threshold": None,
            "ir_role": "reported_diagnostic_and_downstream_stability_input",
            "ic_pos_threshold": None,
            "hit_rate_role": "scorecard_and_deployment_constraint",
            "return_groups": policy.n_return_groups,
            "mono_threshold": None,
            "n_factors_total": len(all_factors),
            "n_factors_t_significant": len(t_significant),
            "n_factors_significant": len(significant),
            # 适配性研究集成信息
            "adaptivity_enabled": bool(adaptivity_data),
            "adaptivity_source": os.path.abspath(adaptivity_file) if adaptivity_data else None,
        },
        "significant_factors": significant,
        "all_results": results,
        "final_factors": final_factor_names,
        "selection_diagnostics": selection_diagnostics,
        "discovery_audit": discovery_audit,
        "funnel_audit": funnel_audit,
        "threshold_sensitivity": threshold_sensitivity,
        "family_governance": family_governance_audit,
        "summary": {
            "by_window": {k: len(v) for k, v in window_groups.items()},
            "by_best_period": {str(k): len(v) for k, v in by_period.items()},
            "total_significant": len(significant),
            "total_passed_all_tests": len(final_factors) if significant else 0,
            # 适配性摘要
            "adaptivity_filtered_invalid": n_filtered if adaptivity_data else 0,
        },
    }
    out_path = os.path.join(output_dir, "ic_by_window_period.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    funnel_path = os.path.join(output_dir, "validation_funnel.json")
    with open(funnel_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "validation_policy_sha256": policy_hash,
                "taxonomy_sha256": taxonomy_sha256(),
                "funnel_audit": funnel_audit,
                "threshold_sensitivity": threshold_sensitivity,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n结果已保存: {out_path}")

    # 输出 YAML 配置片段 (优先用 final_factors, 若后置检验未运行则回退 significant)
    yaml_source = final_factors if significant and final_factors else significant
    print("\n=== config/default.yaml 子组合配置片段 ===")
    print(f"# (基于 {'final_factors' if final_factors else 'significant_factors'})")
    print("# 含适配性研究标注 (best_sector / valid_sectors)")
    short_factors = [r for r in yaml_source if r["best_period"] in (3, 5)]
    mid_factors = [r for r in yaml_source if r["best_period"] in (10,)]
    long_factors = [r for r in yaml_source if r["best_period"] in (20, 40)]

    def _format_factor_line(r: dict) -> str:
        """格式化因子行, 含适配性标注 (如有)."""
        name = r["name"]
        best_sec = r.get("adaptivity_best_sector", "")
        valid_secs = r.get("adaptivity_valid_sectors", "")
        if best_sec and valid_secs:
            return f"  - {name}  # best_sector={best_sec}, valid={valid_secs}"
        return f"  - {name}"

    print(f"\n# 短期子组合 ({len(short_factors)}个, 3-5日持有期):")
    for r in short_factors:
        print(_format_factor_line(r))
    print(f"\n# 中期子组合 ({len(mid_factors)}个, 10日持有期):")
    for r in mid_factors:
        print(_format_factor_line(r))
    print(f"\n# 长期子组合 ({len(long_factors)}个, 20-40日持有期):")
    for r in long_factors:
        print(_format_factor_line(r))

    print("\n筛选完成.")
    return out


def _run_correlation_analysis(
    runner, config_path, factor_start, ic_start, ic_end,
    method="greedy", threshold=0.6, rolling_window=None, auto_threshold=False,
    output_dir=None, screening_file=None,
):
    """对显著因子做相关性分析 + 聚类.

    流程:
    1. 从显式研究目录或筛选文件加载显著因子
    2. 计算因子矩阵 (含预热期)
    3. 调用 factors.correlation_analysis.analyze_and_save 做分析
    4. 输出 factor_correlation.json + .png 到本次研究目录

    Usage:
        python main.py research --correlation
        python main.py research --correlation --corr-method hierarchical
        python main.py research --correlation --corr-rolling 252
        python main.py research --correlation --corr-auto-threshold
    """
    import pandas as pd
    from factors.engine import FactorEngine
    from factors.correlation_analysis import analyze_and_save, _load_significant_factors

    print("=" * 60)
    print("因子相关性分析 + 聚类")
    print("=" * 60)
    print(f"  配置文件: {config_path}")
    print(f"  因子计算区间: {factor_start.date()} ~ {ic_end.date()} (含预热)")
    print(f"  相关性计算区间: {ic_start.date()} ~ {ic_end.date()}")
    print(f"  聚类方法: {method}, 阈值: {threshold}")
    print(f"  滚动窗口: {rolling_window or '全样本'}")
    print(f"  自动选阈值: {auto_threshold}")

    # 加载显著因子
    ic_json_path = screening_file or os.path.join(output_dir, "ic_by_window_period.json")
    if not os.path.exists(ic_json_path):
        print(f"\nIC 检验结果不存在: {ic_json_path}")
        print("请先运行: python main.py research --all --multi-period --t-threshold 1.96")
        return

    significant_factors = _load_significant_factors(ic_json_path)
    if not significant_factors:
        print("无显著因子, 退出")
        return

    factor_names = [f["name"] for f in significant_factors]
    print(f"\n显著因子数: {len(factor_names)}")

    # 计算因子矩阵
    data_mgr = runner.data_manager
    calendar = data_mgr.get_calendar(factor_start, ic_end)
    if hasattr(calendar, "tz") and calendar.tz is not None:
        calendar = calendar.tz_localize(None)
    calendar = pd.DatetimeIndex(sorted(set(calendar)))
    universe = pd.Index(runner.config.universe) if runner.config.universe else pd.Index([])

    print(f"\n计算 {len(factor_names)} 个因子矩阵...")
    t0 = time.time()
    from factors.processor import build_processing_context

    processing_context = build_processing_context(
        data_mgr,
        calendar,
        universe,
        runner.config.universe_selection,
    )
    engine = FactorEngine(data_mgr)
    factor_matrices = engine.compute_factors(
        factor_names, calendar, universe, parallel=True, chunk_size=100
    )
    factor_matrices = runner.processor.process_batch(
        factor_matrices, processing_context
    )
    print(f"  完成: {len(factor_matrices)} 个因子, 耗时 {time.time()-t0:.1f}s")

    # 只保留 IC 检验区间的数据用于相关性计算
    for name in factor_matrices:
        factor_matrices[name] = factor_matrices[name].loc[ic_start:ic_end]

    # 运行分析
    analyze_and_save(
        factor_matrices=factor_matrices,
        significant_factors=significant_factors,
        output_dir=output_dir,
        threshold=threshold,
        method=method,
        rolling_window=rolling_window,
        auto_threshold=auto_threshold,
        high_corr_threshold=0.7,
    )


def main():
    import argparse
    import pandas as pd

    try:
        from core.logger import setup_logger
        from pipeline.runner import PipelineRunner
        from core.registry import list_registered
    except ImportError as e:
        print(f"框架模块导入失败: {e}")
        print(f"   请安装依赖: python -m pip install -r requirements-minimal.txt")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="多因子研究 — 因子 IC/分层/回归检验")
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="配置文件路径 (默认: config/default.yaml)")
    parser.add_argument(
        "--factors", default=None,
        help="指定因子, 逗号分隔 (如: momentum_20d,skewness_20d)")
    parser.add_argument(
        "--all", action="store_true",
        help="批量筛选所有已注册因子, 按 |t| 排序")
    parser.add_argument(
        "--multi-period", action="store_true",
        help="多持有期窗口匹配筛选 (推荐): 5d因子测3/5/10周期, 10d因子测5/10/20周期, "
             "20d因子测10/20/40周期 (持有期为周期数, 非天数)")
    parser.add_argument(
        "--periods", default=None,
        help="显式指定持有期列表 (逗号分隔, 周期数语义), 如 '1,5,10,20,40'. "
             "仅用于注册 horizon 契约完全相同的冻结因子批次；与任一因子契约不一致即拒绝. "
             "需配合 --multi-period 使用")
    parser.add_argument(
        "--frequency", default="daily",
        choices=["daily", "1min", "5min", "15min", "30min", "hourly"],
        help="周期单位 (默认 daily). 非日度研究使用数据源的真实 bar 索引")
    parser.add_argument(
        "--t-threshold", type=float, default=1.96,
        help="筛选时 t 值绝对值阈值 (默认 1.96, 即 95%% 置信度)")
    parser.add_argument(
        "--start", default=None,
        help="IC检验起始日期 (默认: 配置文件的 date_range.start)")
    parser.add_argument(
        "--end", default=None,
        help="IC检验结束日期 (默认: 配置文件的 date_range.end)")
    parser.add_argument(
        "--factor-start", default=None,
        help="因子计算起始日 (含预热期, 默认: IC检验起始日往前1年)")
    parser.add_argument(
        "--correlation", action="store_true",
        help="对显著因子做相关性分析 + 聚类, 生成 factor_correlation.json "
             "(需先运行 --all --multi-period 生成 ic_by_window_period.json)")
    parser.add_argument(
        "--corr-method", choices=["greedy", "hierarchical"], default="greedy",
        help="聚类方法 (默认 greedy, 配合 --correlation 使用)")
    parser.add_argument(
        "--corr-threshold", type=float, default=0.6,
        help="聚类阈值 |corr| > threshold (默认 0.6, 配合 --correlation)")
    parser.add_argument(
        "--corr-rolling", type=int, default=None,
        help="滚动相关性窗口天数 (默认全样本, 配合 --correlation)")
    parser.add_argument(
        "--corr-auto-threshold", action="store_true",
        help="自动选最优聚类阈值 (用轮廓系数, 配合 --correlation)")
    parser.add_argument(
        "--run-id", default=None,
        help="研究运行标识；输出到 runs/<run_id>")
    parser.add_argument(
        "--output-dir", default=None,
        help="显式输出目录；指定后覆盖 runs/<run_id>")
    parser.add_argument(
        "--refuse-existing-output", action="store_true",
        help="若输出目录已存在则拒绝运行；用于不可覆盖的规范化研究")
    parser.add_argument(
        "--adaptivity-file", default=None,
        help="显式适配性 CSV。默认不加载历史 reports 产物")
    parser.add_argument(
        "--screening-file", default=None,
        help="相关性分析使用的 ic_by_window_period.json；默认读取本次输出目录")
    args = parser.parse_args()

    if args.all and args.factors:
        parser.error("--all 与 --factors 不能同时使用")
    if args.periods and not args.multi_period:
        parser.error("--periods 必须配合 --multi-period 使用")

    if args.run_id and not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_id):
        parser.error("--run-id 仅允许字母、数字、点、下划线和连字符")
    if not args.run_id and not args.output_dir:
        parser.error("必须显式指定 --run-id 或 --output-dir")
    run_id = args.run_id
    output_dir = args.output_dir or os.path.join(_PROJECT_ROOT, "runs", run_id)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(_PROJECT_ROOT, output_dir)
    output_dir = os.path.normpath(output_dir)
    if args.refuse_existing_output and os.path.exists(output_dir):
        parser.error(f"输出目录已存在，拒绝覆盖: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    adaptivity_file = args.adaptivity_file
    if adaptivity_file and not os.path.isabs(adaptivity_file):
        adaptivity_file = os.path.join(_PROJECT_ROOT, adaptivity_file)
    if adaptivity_file and not os.path.isfile(adaptivity_file):
        parser.error(f"--adaptivity-file 不存在: {adaptivity_file}")

    screening_file = args.screening_file
    if screening_file and not os.path.isabs(screening_file):
        screening_file = os.path.join(_PROJECT_ROOT, screening_file)
    if screening_file and not os.path.isfile(screening_file):
        parser.error(f"--screening-file 不存在: {screening_file}")

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(_PROJECT_ROOT, config_path)
    config_path = os.path.normpath(config_path)

    setup_logger("multi_factor")

    try:
        from core.config import load_config

        config = load_config(config_path)
        runner = PipelineRunner(config=config)
    except Exception as e:
        print(f"框架初始化失败: {e}")
        print(f"   请检查配置文件: {config_path}")
        sys.exit(1)
    runner.config.backtest.report_dir = output_dir
    print(f"  研究输出目录: {output_dir}")

    # 日期范围
    ic_start = pd.Timestamp(args.start) if args.start else pd.Timestamp(runner.config.date_range.start)
    ic_end = pd.Timestamp(args.end) if args.end else pd.Timestamp(runner.config.date_range.end)
    warmup_days = int(
        runner.config.validation_policy.warmup_days_by_frequency.get(
            args.frequency, 252
        )
    )
    factor_start = pd.Timestamp(args.factor_start) if args.factor_start else (
        ic_start - pd.Timedelta(days=warmup_days)
    )
    if ic_start > ic_end:
        parser.error("--start 必须早于或等于 --end")
    runner.config.date_range.start = str(ic_start.date())
    runner.config.date_range.end = str(ic_end.date())

    if args.correlation:
        # 因子相关性分析 + 聚类 (需先运行 --all --multi-period)
        _run_correlation_analysis(
            runner, config_path, factor_start, ic_start, ic_end,
            method=args.corr_method,
            threshold=args.corr_threshold,
            rolling_window=args.corr_rolling,
            auto_threshold=args.corr_auto_threshold,
            output_dir=output_dir,
            screening_file=screening_file,
        )
    elif args.all or args.multi_period:
        available_factors = set(
            list_registered("factor").get("factor", {}).keys()
        )
        if args.all:
            selected_factors = sorted(available_factors)
        else:
            try:
                requested = _parse_requested_factors(args.factors)
                selected_factors = _validate_requested_factors(
                    requested, available_factors
                )
            except ValueError as exc:
                parser.error(str(exc))
        if args.multi_period:
            # 解析 --periods (显式持有期列表, 可选)
            periods_override = parse_holding_periods(args.periods)
            _run_multi_period_screening(
                runner, selected_factors, config_path, args.t_threshold,
                factor_start, ic_start, ic_end,
                periods_override=periods_override,
                frequency=args.frequency,
                output_dir=output_dir,
                adaptivity_file=adaptivity_file,
            )
        else:
            # 单持有期模式: 同步配置日期
            runner.config.date_range.start = str(ic_start.date())
            runner.config.date_range.end = str(ic_end.date())
            _run_screening(
                runner, selected_factors, config_path, args.t_threshold, output_dir
            )
    else:
        if args.factors:
            available_factors = set(
                list_registered("factor").get("factor", {}).keys()
            )
            try:
                factor_names = _validate_requested_factors(
                    _parse_requested_factors(args.factors), available_factors
                )
            except ValueError as exc:
                parser.error(str(exc))
        else:
            factor_names = runner.config.factors
        _run_single_research(runner, factor_names, config_path)


if __name__ == "__main__":
    main()
