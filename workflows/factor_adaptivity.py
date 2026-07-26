"""因子适配性研究 — 寻找每个因子的最佳板块/品种与持有期.

核心思想:
    不同品种具有不同的经济学意义与交易行为, 同一因子在不同板块/品种上的
    表现应有差异. 本脚本对每个因子在 8 板块 × 多持有期上计算 IC/IR/t/稳定性,
    全量记录每个因子的最佳 (板块, 持有期) 组合, 并做样本外验证防止过拟合.

输出:
    runs/adaptivity_<timestamp>/factor_adaptivity.json — 全量适配性记录
    runs/adaptivity_<timestamp>/factor_adaptivity_summary.csv — 精简摘要 (每因子一行)
    runs/adaptivity_<timestamp>/factor_sector_selection.csv — 每个因子×板块的最优持有期

JSON 结构:
    {
      "metadata": {...},
      "factors": {
        "ts_rank_close_5d_smooth": {
          "best_sector": "ferrous",
          "best_period": 5,
          "best_ic": 0.034,
          "best_ir": 0.42,
          "best_t": 2.87,
          "valid_sectors": ["ferrous", "nonferrous"],
          "sectors": {
            "ferrous": {
              "periods": {
                "3": {"ic": 0.031, "ir": 0.38, "t": 2.51, "hit_rate": 0.54, "n_obs": 1200, "oos_ic": 0.028},
                "5": {"ic": 0.034, ...},
                ...
              }
            },
            ...
          }
        }
      }
    }

Usage:
    python main.py adaptivity
    python main.py adaptivity --periods 1,5,10,20,40
    python main.py adaptivity --factors ts_rank_close_5d_smooth,macd_diff_10d_z
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from core.logger import setup_logger
from core.registry import list_registered
from factors.library import *  # noqa: F401,F403 注册所有因子
from factors.specs.technicals import *  # noqa: F401,F403
from factors.specs.directional import *  # noqa: F401,F403
from factors.specs.volume_stat import *  # noqa: F401,F403
from core.config import load_config
from core.sectors import SECTOR_MAP
from data.manager import DataManager
from data.cache import Cache
from factors.engine import FactorEngine
from core.interfaces import ProcessingContext
from testing.ic_test import ICTest, _vectorized_pearson_ic, _newey_west_ir
from testing.regression import _newey_west_t_stat, _vectorized_univariate_ols

# 默认持有期列表 (周期数, daily频率下=交易日)
DEFAULT_PERIODS = [1, 3, 5, 10, 20, 40]

# 样本外验证分割比例 (前60%训练, 后40%测试)
OOS_SPLIT_RATIO = 0.6

# 板块IC有效性门槛
SECTOR_IC_THRESHOLD = 0.015      # 板块IC绝对值下限
SECTOR_T_THRESHOLD = 1.96        # 板块t值下限 (95%置信)
SECTOR_HIT_RATE_THRESHOLD = 0.52 # IC命中率下限
OOS_CONSISTENCY_THRESHOLD = 0.5  # 样本外IC方向一致比例下限


def _load_data_from_cache(data_mgr, universe, factor_start, ic_end):
    """若 MySQL 连接失败, 从缓存 parquet 加载数据.

    复用 workflows.research 的缓存回退逻辑.
    返回: (success: bool, calendar: pd.DatetimeIndex)

    修复: 增加 get_contract_pair 的缓存回退, 使 roll_yield 等多合约因子
    在缓存模式下也能正常计算 (而非返回空 DataFrame).
    """
    cache_dir = os.path.join(_PROJECT_ROOT, "cache")
    if not os.path.exists(cache_dir):
        return False, pd.DatetimeIndex([])

    close_files = [f for f in os.listdir(cache_dir)
                   if f.startswith("futures_MySQLSource_close_") and f.endswith(".parquet")]
    if not close_files:
        return False, pd.DatetimeIndex([])

    # 选择列覆盖最好的缓存文件
    best_dates = pd.DatetimeIndex([])
    best_score = -1
    for f in close_files:
        try:
            df = pd.read_parquet(os.path.join(cache_dir, f))
            col_overlap = len(set(df.columns) & set(universe))
            date_overlap = ((df.index >= pd.Timestamp(factor_start)) &
                            (df.index <= pd.Timestamp(ic_end))).sum()
            score = col_overlap * 1000 + date_overlap
            if score > best_score:
                best_score = score
                best_dates = pd.DatetimeIndex(df.index)
        except Exception:
            continue

    if len(best_dates) == 0:
        return False, pd.DatetimeIndex([])

    # 加载所有字段
    ohlcv_cache = {}
    for field in ["close", "open", "high", "low", "volume", "oi", "settle", "amount"]:
        for f in os.listdir(cache_dir):
            if f.startswith(f"futures_MySQLSource_{field}_") and f.endswith(".parquet"):
                try:
                    df = pd.read_parquet(os.path.join(cache_dir, f))
                    df = df.reindex(index=best_dates, columns=universe)
                    ohlcv_cache[field] = df
                except Exception:
                    pass
                break

    def _patched_get(field, dates, uni):
        if field in ohlcv_cache:
            return ohlcv_cache[field].reindex(index=dates, columns=uni)
        return pd.DataFrame(index=dates, columns=uni)

    data_mgr.get = _patched_get
    # 同时 patch get_calendar 使其从缓存返回日历
    def _patched_get_calendar(start, end):
        mask = (best_dates >= pd.Timestamp(start)) & (best_dates <= pd.Timestamp(end))
        return best_dates[mask]

    data_mgr.get_calendar = _patched_get_calendar

    # Do not fabricate a far contract when the cache only contains the main
    # series. Missing term-structure data must fail the research gate.
    def _patched_get_contract_pair(field, dates, uni):
        empty = pd.DataFrame(np.nan, index=dates, columns=uni)
        return {
            "near": empty.copy(),
            "far": empty.copy(),
        }

    data_mgr.get_contract_pair = _patched_get_contract_pair
    print(
        f"  (从缓存加载数据: {len(ohlcv_cache)} 个字段, {len(best_dates)} 天; "
        "近远月合约对不可用，carry 类因子保持缺失)"
    )
    return True, best_dates


def _compute_ic_by_sector(
    factor_mat: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    sector_map: dict,
    min_stocks: int = 3,
    forward_period: int = 1,
) -> dict:
    """计算因子在各板块上的 IC.

    Args:
        factor_mat: 因子矩阵 (日期 × 品种)
        fwd_returns: 前向收益矩阵 (日期 × 品种)
        sector_map: {ticker: sector} 映射
        min_stocks: 板块内最小品种数 (低于此值跳过该板块)

    Returns:
        {sector: {"ic_series": pd.Series, "ic_mean": float, "ic_std": float,
                   "ir": float, "t_stat": float, "n_obs": int, "hit_rate": float}}
    """
    common_dates = factor_mat.index.intersection(fwd_returns.index)
    common_cols = factor_mat.columns.intersection(fwd_returns.columns)
    if len(common_cols) < 2 or len(common_dates) < 10:
        return {}

    f = factor_mat.loc[common_dates, common_cols]
    r = fwd_returns.loc[common_dates, common_cols]

    # 按板块分组品种
    sector_tickers = {}
    for ticker in common_cols:
        sec = sector_map.get(str(ticker), "other")
        sector_tickers.setdefault(sec, []).append(ticker)

    results = {}
    for sector, tickers in sector_tickers.items():
        if len(tickers) < min_stocks:
            continue
        f_sec = f[tickers]
        r_sec = r[tickers]
        # 板块内截面IC, min_stocks降低到3 (板块品种数少)
        ic_series, valid_mask = _vectorized_pearson_ic(f_sec, r_sec, min_stocks=min_stocks)
        if len(ic_series) < 10:
            continue
        ic_list = ic_series.tolist()
        ic_mean = float(np.mean(ic_list))
        ic_std = float(np.std(ic_list))
        ir = ic_mean / ic_std if ic_std > 0 else 0.0
        # IC remains the economic-effect diagnostic. Statistical admission is
        # based on the raw, unpenalized univariate Fama-MacBeth OLS slope.
        _, ic_t_stat = _newey_west_ir(ic_series, forward_period=forward_period)
        ols_slopes = _vectorized_univariate_ols(
            f_sec, r_sec, min_stocks=min_stocks
        )
        ols_t_stat = _newey_west_t_stat(
            ols_slopes, forward_period=forward_period
        )
        p_value = float(
            2.0 * scipy_stats.t.sf(
                abs(ols_t_stat), df=max(len(ols_slopes) - 1, 1)
            )
        )
        hit_rate = float((ic_series > 0).mean()) if ic_mean >= 0 else float((ic_series < 0).mean())

        results[sector] = {
            "ic_series": ic_series,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ir": ir,
            "t_stat": float(ols_t_stat),
            "ic_t_stat": float(ic_t_stat),
            "ols_beta": float(ols_slopes.mean()) if len(ols_slopes) else 0.0,
            "ols_hac_t": float(ols_t_stat),
            "ols_p_value": p_value,
            "ols_n": int(len(ols_slopes)),
            "inference_model": "unpenalized_univariate_fama_macbeth_ols_hac",
            "p_value": p_value,
            "n_obs": len(ic_list),
            "hit_rate": hit_rate,
        }
    return results


def _compute_oos_consistency(ic_series: pd.Series, oos_ratio: float = OOS_SPLIT_RATIO) -> dict:
    """样本外一致性验证: 前oos_ratio找IC方向, 后段验证方向一致性.

    Returns:
        {"oos_ic": float, "oos_hit_rate": float, "is_consistent": bool}
    """
    if len(ic_series) < 20:
        return {"oos_ic": 0.0, "oos_hit_rate": 0.0, "is_consistent": False}
    split_idx = int(len(ic_series) * oos_ratio)
    train = ic_series.iloc[:split_idx]
    test = ic_series.iloc[split_idx:]
    if len(train) < 10 or len(test) < 10:
        return {"oos_ic": 0.0, "oos_hit_rate": 0.0, "is_consistent": False}
    train_direction = 1 if train.mean() >= 0 else -1
    oos_ic = float(test.mean())
    oos_hit_rate = float((test * train_direction > 0).mean())
    is_consistent = oos_hit_rate >= OOS_CONSISTENCY_THRESHOLD
    return {
        "oos_ic": oos_ic,
        "oos_hit_rate": oos_hit_rate,
        "is_consistent": bool(is_consistent),
    }


def _is_sector_valid(sector_result: dict) -> bool:
    """判断因子在某板块上是否有效 (通过门槛检验)."""
    return (
        abs(sector_result["ic_mean"]) >= SECTOR_IC_THRESHOLD
        and abs(sector_result["t_stat"]) >= SECTOR_T_THRESHOLD
        and sector_result["hit_rate"] >= SECTOR_HIT_RATE_THRESHOLD
        and sector_result["n_obs"] >= 50
    )


def _analyze_ic_decay(period_results: dict) -> dict:
    """分析IC随持有期的衰减曲线, 自动推荐最优持有期.

    Args:
        period_results: {period_str: {ic, ir, t, ...}} 某板块各持有期的IC结果

    Returns:
        {
            "decay_type": "increasing" / "decreasing" / "hump" / "flat",
            "recommended_period": int,    # 推荐持有期
            "recommendation_reason": str, # 推荐理由
            "peak_period": int,           # IC峰值持有期
            "peak_ic": float,
            "half_life_period": int or None, # IC衰减到峰值一半时的持有期 (仅decreasing/hump)
        }
    """
    if not period_results:
        return {
            "decay_type": "flat", "recommended_period": 0,
            "recommendation_reason": "无数据", "peak_period": 0,
            "peak_ic": 0.0, "half_life_period": None,
        }

    # 按持有期排序
    sorted_periods = sorted(int(p) for p in period_results.keys())
    ics = [abs(period_results[str(p)]["ic"]) for p in sorted_periods]
    ts = [abs(period_results[str(p)]["t"]) for p in sorted_periods]
    irs = [abs(period_results[str(p)]["ir"]) for p in sorted_periods]

    if len(ics) < 2:
        p = sorted_periods[0]
        return {
            "decay_type": "flat", "recommended_period": p,
            "recommendation_reason": "仅单一持有期数据",
            "peak_period": p, "peak_ic": ics[0] if ics else 0.0,
            "half_life_period": None,
        }

    # 找IC峰值
    peak_idx = int(np.argmax(ics))
    peak_period = sorted_periods[peak_idx]
    peak_ic = ics[peak_idx]

    # 判断衰减类型
    if peak_idx == 0 and ics[-1] < ics[0] * 0.5:
        decay_type = "decreasing"  # IC随持有期递减 (短期因子)
    elif peak_idx == len(ics) - 1:
        decay_type = "increasing"  # IC随持有期递增 (长期因子)
    elif 0 < peak_idx < len(ics) - 1:
        decay_type = "hump"        # IC先增后减 (中期因子)
    else:
        decay_type = "flat"        # IC基本不随持有期变化

    # 计算半衰期 (仅对decreasing和hump)
    half_life = None
    if decay_type in ("decreasing", "hump") and peak_ic > 0:
        half_threshold = peak_ic * 0.5
        for i in range(peak_idx + 1, len(ics)):
            if ics[i] < half_threshold:
                half_life = sorted_periods[i]
                break

    # 推荐持有期: 综合考虑IC峰值和t值显著性
    # 策略: 选择 IC/t 比值最大的持有期 (兼顾效应大小和统计显著性)
    # 但如果衰减型是increasing, 选择最后一个有效持有期
    # 如果是hump, 选择峰值附近
    # 如果是decreasing, 选择峰值(第一个)
    best_score = -1.0
    recommended = peak_period
    reason = f"IC峰值@{peak_period}周期"

    for i, p in enumerate(sorted_periods):
        ic = ics[i]
        t = ts[i]
        # 评分 = IC × √(t) (兼顾效应量和显著性)
        score = ic * (t ** 0.5) if t > 0 else 0
        if score > best_score:
            best_score = score
            recommended = p
            if p == peak_period:
                reason = f"IC峰值@{peak_period}周期 (IC={peak_ic:.4f}, t={ts[i]:.2f})"
            else:
                reason = f"IC/t综合最优@{p}周期 (IC={ic:.4f}, t={t:.2f})"

    return {
        "decay_type": decay_type,
        "recommended_period": recommended,
        "recommendation_reason": reason,
        "peak_period": peak_period,
        "peak_ic": float(peak_ic),
        "half_life_period": half_life,
    }


def _analyze_factor_across_periods(
    factor_mat: pd.DataFrame,
    fwd_returns_by_period: dict,
    periods: list,
    sector_map: dict,
) -> dict:
    """Evaluate one factor across horizons using read-only shared inputs."""
    result = {
        "best_sector": "",
        "best_period": 0,
        "best_ic": 0.0,
        "best_ir": 0.0,
        "best_t": 0.0,
        "valid_sectors": [],
        "sectors": {},
    }
    best_t_abs = 0.0
    sector_results = {}

    for period in periods:
        sector_ics = _compute_ic_by_sector(
            factor_mat,
            fwd_returns_by_period[period],
            sector_map,
            min_stocks=3,
            forward_period=period,
        )
        for sector, values in sector_ics.items():
            oos = _compute_oos_consistency(values["ic_series"])
            entry = {
                "ic": values["ic_mean"],
                "ir": values["ir"],
                "t": values["t_stat"],
                "p_value": values["p_value"],
                "hit_rate": values["hit_rate"],
                "n_obs": values["n_obs"],
                "oos_ic": oos["oos_ic"],
                "oos_hit_rate": oos["oos_hit_rate"],
                "oos_consistent": oos["is_consistent"],
                "passes_thresholds": (
                    _is_sector_valid(values) and oos["is_consistent"]
                ),
                "is_valid": False,
            }
            sector_results.setdefault(sector, {})[str(period)] = entry

            if abs(values["t_stat"]) > best_t_abs:
                best_t_abs = abs(values["t_stat"])
                result.update({
                    "best_sector": sector,
                    "best_period": period,
                    "best_ic": values["ic_mean"],
                    "best_ir": values["ir"],
                    "best_t": values["t_stat"],
                })

    result["sectors"] = sector_results
    result["ic_decay_analysis"] = {
        sector: _analyze_ic_decay(period_values)
        for sector, period_values in sector_results.items()
    }
    return result


def _apply_multiple_testing(
    all_results: dict,
    *,
    method: str = "bonferroni",
    alpha: float = 0.05,
) -> int:
    """Apply the predeclared multiple-testing gate to raw OLS/HAC p-values."""
    from research.statistics import benjamini_hochberg, simes_p_value

    if method not in {"bonferroni", "global", "hierarchical"}:
        raise ValueError(f"unsupported FDR method: {method}")

    factor_entries = {}
    hypothesis_entries = []
    for name, result in all_results.items():
        entries = [
            entry
            for period_results in result["sectors"].values()
            for entry in period_results.values()
        ]
        factor_entries[name] = entries
        hypothesis_entries.extend(entries)

    if method == "bonferroni":
        adjusted_alpha = alpha / max(len(hypothesis_entries), 1)
        for entry in hypothesis_entries:
            raw_p = float(entry.get("p_value", 1.0))
            if not np.isfinite(raw_p):
                raw_p = 1.0
            is_significant = bool(raw_p <= adjusted_alpha)
            entry["bonferroni_alpha"] = float(adjusted_alpha)
            entry["bonferroni_adjusted_p"] = float(
                min(max(raw_p, 0.0) * len(hypothesis_entries), 1.0)
            )
            entry["q_value"] = entry["bonferroni_adjusted_p"]
            entry["fdr_significant"] = is_significant
            entry["is_valid"] = bool(
                entry["passes_thresholds"] and is_significant
            )
        for name, result in all_results.items():
            factor_significant = any(
                entry.get("fdr_significant", False)
                for entry in factor_entries[name]
            )
            result["factor_p_value"] = min(
                (float(entry.get("p_value", 1.0))
                 for entry in factor_entries[name]),
                default=1.0,
            )
            result["factor_q_value"] = min(
                (float(entry.get("bonferroni_adjusted_p", 1.0))
                 for entry in factor_entries[name]),
                default=1.0,
            )
            result["factor_fdr_significant"] = bool(factor_significant)
    elif method == "global":
        q_values, rejected = benjamini_hochberg(
            [entry.get("p_value", 1.0) for entry in hypothesis_entries],
            alpha=alpha,
        )
        for entry, q_value, is_significant in zip(
            hypothesis_entries, q_values, rejected
        ):
            entry["q_value"] = float(q_value)
            entry["fdr_significant"] = bool(is_significant)
            entry["is_valid"] = bool(
                entry["passes_thresholds"] and is_significant
            )
    else:
        factor_names = list(all_results)
        factor_p_values = []
        local_decisions = {}
        for name in factor_names:
            entries = factor_entries[name]
            p_values = [entry.get("p_value", 1.0) for entry in entries]
            local_q, local_rejected = benjamini_hochberg(p_values, alpha=alpha)
            local_decisions[name] = (local_q, local_rejected)
            factor_p_values.append(simes_p_value(p_values))

        factor_q_values, factor_rejected = benjamini_hochberg(
            factor_p_values, alpha=alpha
        )
        for index, name in enumerate(factor_names):
            result = all_results[name]
            factor_q = float(factor_q_values[index])
            factor_significant = bool(factor_rejected[index])
            result["factor_p_value"] = float(factor_p_values[index])
            result["factor_q_value"] = factor_q
            result["factor_fdr_significant"] = factor_significant
            local_q, local_rejected = local_decisions[name]
            for entry, local_value, local_significant in zip(
                factor_entries[name], local_q, local_rejected
            ):
                entry["local_q_value"] = float(local_value)
                entry["factor_q_value"] = factor_q
                entry["q_value"] = max(factor_q, float(local_value))
                entry["fdr_significant"] = bool(
                    factor_significant and local_significant
                )
                entry["is_valid"] = bool(
                    entry["passes_thresholds"]
                    and factor_significant
                    and local_significant
                )

    for result in all_results.values():
        result["valid_sectors"] = sorted({
            sector
            for sector, period_results in result["sectors"].items()
            if any(entry["is_valid"] for entry in period_results.values())
        })
        _select_approved_optima(result)
    return len(hypothesis_entries)


def _entry_rank(period: int, entry: dict) -> tuple:
    """Deterministic preference among already approved local hypotheses."""
    return (
        float(entry.get("q_value", 1.0)),
        -abs(float(entry.get("t", 0.0))),
        -abs(float(entry.get("ic", 0.0))),
        int(period),
    )


def _select_approved_optima(result: dict) -> list[dict]:
    """Select one approved horizon per sector and the factor-level optimum.

    Selection happens only after threshold and FDR decisions. This prevents an
    unapproved high-t local hypothesis from determining the sleeve assignment.
    """
    sector_optima = []
    for sector, period_results in sorted(result.get("sectors", {}).items()):
        candidates = [
            (int(period), entry)
            for period, entry in period_results.items()
            if bool(entry.get("is_valid", False))
        ]
        if not candidates:
            continue
        period, entry = min(candidates, key=lambda item: _entry_rank(*item))
        sector_optima.append({
            "sector": sector,
            "best_period": period,
            "best_ic": float(entry.get("ic", 0.0)),
            "best_ir": float(entry.get("ir", 0.0)),
            "best_t": float(entry.get("t", 0.0)),
            "best_q": float(entry.get("q_value", 1.0)),
            "n_obs": int(entry.get("n_obs", 0)),
            "valid_period_count": len(candidates),
            "valid_periods": "|".join(str(item[0]) for item in sorted(candidates)),
        })

    result["sector_optima"] = sector_optima
    if sector_optima:
        best = min(
            sector_optima,
            key=lambda item: (
                item["best_q"], -abs(item["best_t"]),
                -abs(item["best_ic"]), item["best_period"], item["sector"],
            ),
        )
        result.update({
            "best_sector": best["sector"],
            "best_period": best["best_period"],
            "best_ic": best["best_ic"],
            "best_ir": best["best_ir"],
            "best_t": best["best_t"],
            "best_q": best["best_q"],
        })
    else:
        result.update({
            "best_sector": "", "best_period": 0, "best_ic": 0.0,
            "best_ir": 0.0, "best_t": 0.0, "best_q": 1.0,
        })
    return sector_optima


def run_adaptivity_analysis(
    all_factors: list,
    config_path: str,
    factor_start: str,
    ic_start: str,
    ic_end: str,
    periods: list = None,
    sectors_filter: list = None,
    output_dir: str = None,
    artifact_id: str = None,
    build_correlation: bool = True,
    workers: int = None,
    config=None,
    fdr_method: str = "bonferroni",
    fdr_alpha: float = 0.05,
):
    """运行因子适配性分析.

    Args:
        all_factors: 因子名列表
        config_path: 配置文件路径
        factor_start: 因子计算起始日 (含预热)
        ic_start: IC检验起始日
        ic_end: IC检验结束日
        periods: 持有期列表 (周期数)
        sectors_filter: 仅分析指定板块 (None=全部)
    """
    periods = periods or DEFAULT_PERIODS
    periods = sorted(set(periods))
    workers = max(1, int(workers or min(4, os.cpu_count() or 1)))

    print("=" * 70)
    print(f"因子适配性研究 ({len(all_factors)} 个因子 × {len(SECTOR_MAP)} 品种 × {len(periods)} 持有期)")
    print("=" * 70)
    print(f"  配置文件: {config_path}")
    print(f"  因子计算区间: {factor_start} ~ {ic_end} (含预热)")
    print(f"  IC检验区间: {ic_start} ~ {ic_end}")
    print(f"  持有期 (周期数): {periods}")
    print(f"  因子分析线程: {workers}")
    print(f"  板块数: {len(set(SECTOR_MAP.values()))} "
          f"({', '.join(sorted(set(SECTOR_MAP.values())))})")
    print(f"  样本外验证: 前{int(OOS_SPLIT_RATIO*100)}%训练 / 后{int((1-OOS_SPLIT_RATIO)*100)}%测试")
    print(f"  有效性门槛: |IC|≥{SECTOR_IC_THRESHOLD}, |t|≥{SECTOR_T_THRESHOLD}, "
          f"命中率≥{SECTOR_HIT_RATE_THRESHOLD}, OOS一致≥{OOS_CONSISTENCY_THRESHOLD}")
    if sectors_filter:
        print(f"  仅分析板块: {sectors_filter}")
    print()

    # 复用 PipelineRunner 初始化数据层 (成熟逻辑, 正确处理 pydantic 配置转换)
    from pipeline.runner import PipelineRunner
    runner = PipelineRunner(config=config) if config is not None else PipelineRunner(config_path)
    config = runner.config
    data_mgr = runner.data_manager

    universe = pd.Index(config.universe) if config.universe else pd.Index([])
    calendar = data_mgr.get_calendar(factor_start, ic_end)
    if hasattr(calendar, "tz") and calendar.tz is not None:
        calendar = calendar.tz_localize(None)
    calendar = pd.DatetimeIndex(sorted(set(calendar)))

    # 若 MySQL 连接失败导致日历为空, 从缓存加载
    if len(calendar) == 0:
        cache_ok, cache_dates = _load_data_from_cache(
            data_mgr, universe, factor_start, ic_end,
        )
        if cache_ok and len(cache_dates) > 0:
            calendar = cache_dates
            if hasattr(calendar, "tz") and calendar.tz is not None:
                calendar = calendar.tz_localize(None)
            calendar = pd.DatetimeIndex(sorted(set(calendar)))

    print(f"交易日历: {len(calendar)} 天, universe: {len(universe)} 品种")

    # 过滤有效品种 (在universe中且有板块映射的)
    valid_universe = [t for t in universe if str(t) in SECTOR_MAP]
    if sectors_filter:
        valid_universe = [t for t in valid_universe if SECTOR_MAP[str(t)] in sectors_filter]
    print(f"有效品种 (有板块映射): {len(valid_universe)}")

    # 计算所有因子矩阵 (一次性)
    print(f"\n计算 {len(all_factors)} 个因子矩阵 (含预热期)...")
    t0 = time.time()
    engine = FactorEngine(data_mgr)
    factor_matrices = engine.compute_factors(
        all_factors, calendar, pd.Index(valid_universe),
        parallel=False, chunk_size=100,
    )
    processing_context = ProcessingContext(
        data=data_mgr, dates=calendar, universe=pd.Index(valid_universe)
    )
    factor_matrices = runner.processor.process_batch(
        factor_matrices, processing_context
    )
    print(f"  完成: {len(factor_matrices)} 个因子, 耗时 {time.time()-t0:.1f}s")

    # 预计算各持有期的 forward returns
    print("\n预计算 forward returns...")
    fwd_returns_by_period = {}
    for p in periods:
        fwd_returns_by_period[p] = data_mgr.get_forward_returns(
            calendar, pd.Index(valid_universe), period=p,
        )
    print(f"  持有期 (周期数): {periods}")

    # 批量适配性分析
    print(f"\n=== 开始适配性 IC 检验 ({len(all_factors)} 因子 × {len(periods)} 持有期 × 多板块) ===")
    all_results = {}
    t0 = time.time()
    available_factors = [name for name in all_factors if name in factor_matrices]
    analysis_returns = {
        period: frame.loc[ic_start:ic_end]
        for period, frame in fwd_returns_by_period.items()
    }

    def _evaluate(name):
        return name, _analyze_factor_across_periods(
            factor_matrices[name].loc[ic_start:ic_end],
            analysis_returns,
            periods,
            SECTOR_MAP,
        )

    if workers == 1:
        evaluated = map(_evaluate, available_factors)
        pool = None
    else:
        pool = ThreadPoolExecutor(max_workers=workers)
        evaluated = pool.map(_evaluate, available_factors)

    try:
        for idx, (fname, factor_result) in enumerate(evaluated, 1):
            all_results[fname] = factor_result
            if idx % 20 == 0 or idx == len(available_factors):
                elapsed = time.time() - t0
                print(f"  [{idx}/{len(available_factors)}] 耗时 {elapsed:.1f}s")
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    n_hypotheses = _apply_multiple_testing(
        all_results, method=fdr_method, alpha=fdr_alpha
    )

    # === 汇总输出 ===
    print("\n" + "=" * 70)
    print("因子适配性研究完成")
    print("=" * 70)

    n_with_valid_sector = sum(1 for r in all_results.values() if r["valid_sectors"])
    n_total = len(all_results)
    print(f"  有至少1个有效板块的因子: {n_with_valid_sector}/{n_total}")

    # 按板块统计
    sector_stats = {}
    for fname, res in all_results.items():
        for sec in res["valid_sectors"]:
            sector_stats.setdefault(sec, 0)
            sector_stats[sec] += 1
    if sector_stats:
        print("\n  有效因子最多的板块:")
        for sec, cnt in sorted(sector_stats.items(), key=lambda x: -x[1]):
            print(f"    {sec}: {cnt} 个因子")

    # 保存JSON
    output_dir = output_dir or os.path.join(
        _PROJECT_ROOT, "runs", time.strftime("adaptivity_%Y%m%d_%H%M%S")
    )
    output_dir = os.path.abspath(output_dir)
    out_path = os.path.join(output_dir, "factor_adaptivity.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out = {
        "metadata": {
            "generated_at": pd.Timestamp.now().isoformat(),
            "factor_start": str(factor_start),
            "ic_start": str(ic_start),
            "ic_end": str(ic_end),
            "periods": periods,
            "sectors": sorted(set(SECTOR_MAP.values())),
            "n_factors": n_total,
            "n_factors_with_valid_sector": n_with_valid_sector,
            "n_hypotheses": n_hypotheses,
            "analysis_workers": workers,
            "fdr_method": fdr_method,
            "fdr_alpha": fdr_alpha,
            "inference_model": "unpenalized_univariate_fama_macbeth_ols_hac",
            "factor_preprocessing": [
                "mad_winsorize_by_date",
                "zscore_standardize_by_date",
            ],
            "selection_order": [
                "predeclared_exposure_preprocessing",
                "raw_ols_hac_p_values",
                "multiple_testing_gate",
                "economic_and_robustness_gates",
            ],
            "downstream_required_order": [
                "correlation_deduplication",
                "family_governance",
                "training_only_ridge",
            ],
            "thresholds": {
                "ic": SECTOR_IC_THRESHOLD,
                "t": SECTOR_T_THRESHOLD,
                "hit_rate": SECTOR_HIT_RATE_THRESHOLD,
                "oos_consistency": OOS_CONSISTENCY_THRESHOLD,
            },
            "oos_split_ratio": OOS_SPLIT_RATIO,
            "sector_map": SECTOR_MAP,
        },
        "factors": all_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n全量记录已保存: {out_path}")

    # 保存逐板块最优周期。组合路由以此文件为准，摘要仅用于因子级治理。
    sector_csv_path = os.path.join(output_dir, "factor_sector_selection.csv")
    sector_rows = []
    for fname, res in all_results.items():
        for optimum in res.get("sector_optima", []):
            sector_rows.append({"factor": fname, **optimum})
    sector_columns = [
        "factor", "sector", "best_period", "best_ic", "best_ir", "best_t",
        "best_q", "n_obs", "valid_period_count", "valid_periods",
    ]
    df_sector = pd.DataFrame(sector_rows, columns=sector_columns)
    if not df_sector.empty:
        df_sector = df_sector.sort_values(
            ["best_q", "best_t", "factor", "sector"],
            key=lambda column: column.abs() if column.name == "best_t" else column,
            ascending=[True, False, True, True],
        )
    df_sector.to_csv(sector_csv_path, index=False, encoding="utf-8-sig")
    print(f"板块×周期选择已保存: {sector_csv_path}")

    # 保存CSV摘要 (每因子一行, 含IC衰减分析)
    csv_path = os.path.join(output_dir, "factor_adaptivity_summary.csv")
    summary_rows = []
    for fname, res in all_results.items():
        # 最佳板块的IC衰减分析
        best_sec = res["best_sector"]
        decay_info = res.get("ic_decay_analysis", {}).get(best_sec, {})
        summary_rows.append({
            "factor": fname,
            "best_sector": res["best_sector"],
            "best_period": res["best_period"],
            "best_ic": round(res["best_ic"], 4),
            "best_ir": round(res["best_ir"], 3),
            "best_t": round(res["best_t"], 2),
            "best_q": res.get("best_q", 1.0),
            "n_valid_sectors": len(res["valid_sectors"]),
            "valid_sectors": "|".join(res["valid_sectors"]),
            "sector_best_periods": "|".join(
                f"{item['sector']}:{item['best_period']}"
                for item in res.get("sector_optima", [])
            ),
            "decay_type": decay_info.get("decay_type", ""),
            "recommended_period": decay_info.get("recommended_period", ""),
            "peak_period": decay_info.get("peak_period", ""),
            "half_life_period": decay_info.get("half_life_period", ""),
        })
    df_summary = pd.DataFrame(summary_rows)
    if not df_summary.empty:
        df_summary = df_summary.sort_values("best_t", key=lambda x: x.abs(), ascending=False)
    df_summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"摘要已保存: {csv_path}")

    artifact_files = {
        "factor_adaptivity": out_path,
        "factor_adaptivity_summary": csv_path,
        "factor_sector_selection": sector_csv_path,
    }
    significant = [
        {
            "name": name,
            "best_t": result["best_t"],
            "best_ic": result["best_ic"],
            "best_period": result["best_period"],
        }
        for name, result in all_results.items()
        if result["valid_sectors"]
    ]
    if build_correlation and len(significant) >= 2:
        from factors.correlation_analysis import analyze_and_save

        selected_matrices = {
            item["name"]: factor_matrices[item["name"]].loc[ic_start:ic_end]
            for item in significant
            if item["name"] in factor_matrices
        }
        correlation = analyze_and_save(
            factor_matrices=selected_matrices,
            significant_factors=significant,
            output_dir=output_dir,
            threshold=0.6,
            method="hierarchical",
            rolling_window=None,
            auto_threshold=False,
            high_corr_threshold=0.7,
        )
        correlation_path = os.path.join(output_dir, "factor_correlation.json")
        if correlation and os.path.isfile(correlation_path):
            artifact_files["factor_correlation"] = correlation_path

    if artifact_id:
        from pathlib import Path
        from research.artifacts import (
            ResearchArtifactBundle,
            canonical_config_hash,
            dataframe_collection_sha256,
            source_tree_hash,
        )

        data_frames = {
            **{f"factor:{name}": frame.loc[ic_start:ic_end] for name, frame in factor_matrices.items()},
            **{f"returns:{period}": frame.loc[ic_start:ic_end] for period, frame in fwd_returns_by_period.items()},
        }
        bundle = ResearchArtifactBundle.create(
            output_dir,
            artifact_id=artifact_id,
            train_start=ic_start,
            train_end=ic_end,
            data_sha256=dataframe_collection_sha256(data_frames),
            config_sha256=canonical_config_hash(config),
            code_sha256=source_tree_hash(Path(_PROJECT_ROOT)),
            files=artifact_files,
            metadata={
                "candidate_factors": len(all_factors),
                "selected_factors": len(significant),
                "periods": periods,
                "fdr_method": fdr_method,
                "fdr_alpha": fdr_alpha,
            },
        )
        print(f"研究 bundle 已冻结: {bundle.root}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="因子适配性研究 — 板块×持有期最优匹配")
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="配置文件路径 (默认: config/default.yaml)")
    parser.add_argument(
        "--periods", default=None,
        help=f"持有期列表, 逗号分隔 (默认: {','.join(map(str, DEFAULT_PERIODS))})")
    parser.add_argument(
        "--factors", default=None,
        help="指定因子名, 逗号分隔 (默认: 全部已注册因子)")
    parser.add_argument(
        "--sectors", default=None,
        help="仅分析指定板块, 逗号分隔 (如 ferrous,agri)")
    parser.add_argument(
        "--factor-start", default="2018-12-10",
        help="因子计算起始日 (含1年预热, 默认: 2018-12-10)")
    parser.add_argument(
        "--ic-start", default="2021-01-01",
        help="IC检验起始日 (默认: 2021-01-01)")
    parser.add_argument(
        "--ic-end", default="2025-06-30",
        help="IC检验结束日 (默认: 2025-06-30)")
    parser.add_argument(
        "--output-dir", default=None,
        help="输出目录 (默认: runs/adaptivity_<timestamp>)")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="因子级分析线程数 (默认: min(4, CPU数); 设为1禁用并行)")
    parser.add_argument(
        "--artifact-id", default=None,
        help="非空时在输出完成后创建不可覆盖的研究 bundle manifest")
    parser.add_argument(
        "--no-correlation", action="store_true",
        help="不生成训练期因子相关性聚类产物")
    parser.add_argument(
        "--fdr-method", choices=["bonferroni", "global", "hierarchical"],
        default="bonferroni",
        help=("多重检验: bonferroni=全部原始OLS/HAC p值全局校正; "
              "global=全部局部假设统一BH; hierarchical=先因子后局部BH"))
    parser.add_argument(
        "--cache-only", action="store_true",
        help="严格只使用本地缓存；缓存未命中时禁止访问数据库")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(_PROJECT_ROOT, config_path)
    config_path = os.path.normpath(config_path)
    config = load_config(config_path)
    if args.cache_only:
        config.data.cache["only"] = True

    setup_logger("multi_factor")

    # 确定因子列表
    if args.factors:
        all_factors = [f.strip() for f in args.factors.split(",") if f.strip()]
    else:
        all_factors = sorted(list_registered("factor").get("factor", {}).keys())
    print(f"已注册因子: {len(all_factors)} 个")

    # 解析持有期
    if args.periods:
        periods = [int(x.strip()) for x in args.periods.split(",") if x.strip()]
        periods = sorted(set(periods))
    else:
        periods = DEFAULT_PERIODS

    # 解析板块过滤
    sectors_filter = None
    if args.sectors:
        sectors_filter = [s.strip() for s in args.sectors.split(",") if s.strip()]

    run_adaptivity_analysis(
        all_factors=all_factors,
        config_path=config_path,
        factor_start=args.factor_start,
        ic_start=args.ic_start,
        ic_end=args.ic_end,
        periods=periods,
        sectors_filter=sectors_filter,
        output_dir=args.output_dir,
        artifact_id=args.artifact_id,
        build_correlation=not args.no_correlation,
        workers=args.workers,
        config=config,
        fdr_method=args.fdr_method,
    )


if __name__ == "__main__":
    main()
