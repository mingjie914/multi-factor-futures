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
import gc
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
from core.period import PeriodContext
from core.sectors import (
    SECTOR_MAP,
    TAXONOMY_VERSION,
    taxonomy_sha256,
)
from data.manager import DataManager, FrequencyDataProvider
from data.cache import Cache
from factors.engine import FactorEngine
from factors.processor import build_processing_context
from testing.ic_test import _newey_west_ir
from testing.regression import _newey_west_t_stat

# 默认持有期列表 (周期数, daily频率下=交易日)
DEFAULT_PERIODS = [1, 3, 5, 10, 20, 40]

# 样本外验证分割比例 (前60%训练, 后40%测试)
OOS_SPLIT_RATIO = 0.75

# 板块IC有效性门槛
SECTOR_IC_THRESHOLD = 0.01       # 全市场统一经济量级底线
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


def _cross_section_ic_and_slopes(
    factor_values: np.ndarray,
    return_values: np.ndarray,
    index: pd.Index,
    *,
    min_stocks: int,
) -> tuple[pd.Series, pd.Series]:
    """Compute Pearson IC and raw OLS slopes from one centered array pass."""
    # Pearson historically treats infinities as present and rejects the row
    # only after the centered products become non-finite. Raw OLS excludes
    # infinities before centering. Reuse the Pearson pass for ordinary rows
    # and recompute only the exceptional rows to preserve that contract.
    valid = ~np.isnan(factor_values) & ~np.isnan(return_values)
    counts = valid.sum(axis=1)
    safe_counts = np.maximum(counts, 1)
    factor_masked = np.where(valid, factor_values, 0.0)
    returns_masked = np.where(valid, return_values, 0.0)
    factor_mean = factor_masked.sum(axis=1) / safe_counts
    returns_mean = returns_masked.sum(axis=1) / safe_counts
    factor_centered = np.where(
        valid, factor_values - factor_mean[:, None], 0.0
    )
    returns_centered = np.where(
        valid, return_values - returns_mean[:, None], 0.0
    )

    cross_product = np.einsum(
        "ij,ij->i", factor_centered, returns_centered
    )
    factor_ss = np.einsum(
        "ij,ij->i", factor_centered, factor_centered
    )
    returns_ss = np.einsum(
        "ij,ij->i", returns_centered, returns_centered
    )

    ic_denominator = np.sqrt(factor_ss * returns_ss)
    ic_usable = (
        (counts >= min_stocks)
        & np.isfinite(cross_product)
        & np.isfinite(ic_denominator)
        & (ic_denominator > 0.0)
    )
    ic_values = np.divide(
        cross_product,
        ic_denominator,
        out=np.full(len(index), np.nan, dtype=float),
        where=ic_usable,
    )

    slope_valid = np.isfinite(factor_values) & np.isfinite(return_values)
    slope_counts = counts.copy()
    slope_cross_product = cross_product.copy()
    slope_factor_ss = factor_ss.copy()
    exceptional_rows = np.any(slope_valid != valid, axis=1)
    if exceptional_rows.any():
        exceptional_factor = factor_values[exceptional_rows]
        exceptional_returns = return_values[exceptional_rows]
        exceptional_valid = slope_valid[exceptional_rows]
        exceptional_counts = exceptional_valid.sum(axis=1)
        exceptional_safe_counts = np.maximum(exceptional_counts, 1)
        exceptional_factor_masked = np.where(
            exceptional_valid, exceptional_factor, 0.0
        )
        exceptional_returns_masked = np.where(
            exceptional_valid, exceptional_returns, 0.0
        )
        exceptional_factor_mean = (
            exceptional_factor_masked.sum(axis=1) / exceptional_safe_counts
        )
        exceptional_return_mean = (
            exceptional_returns_masked.sum(axis=1) / exceptional_safe_counts
        )
        exceptional_factor_centered = np.where(
            exceptional_valid,
            exceptional_factor - exceptional_factor_mean[:, None],
            0.0,
        )
        exceptional_returns_centered = np.where(
            exceptional_valid,
            exceptional_returns - exceptional_return_mean[:, None],
            0.0,
        )
        slope_counts[exceptional_rows] = exceptional_counts
        slope_cross_product[exceptional_rows] = np.einsum(
            "ij,ij->i",
            exceptional_factor_centered,
            exceptional_returns_centered,
        )
        slope_factor_ss[exceptional_rows] = np.einsum(
            "ij,ij->i",
            exceptional_factor_centered,
            exceptional_factor_centered,
        )

    slope_usable = (
        (slope_counts >= min_stocks)
        & np.isfinite(slope_cross_product)
        & np.isfinite(slope_factor_ss)
        & (slope_factor_ss > np.finfo(float).eps)
    )
    slopes = np.divide(
        slope_cross_product,
        slope_factor_ss,
        out=np.full(len(index), np.nan, dtype=float),
        where=slope_usable,
    )
    return (
        pd.Series(ic_values[ic_usable], index=index[ic_usable], dtype=float),
        pd.Series(slopes[slope_usable], index=index[slope_usable], dtype=float),
    )


def _compute_ic_by_sector(
    factor_mat: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    sector_map: dict,
    min_stocks: int = 3,
    forward_period: int = 1,
    single_min_trading_days: int = 750,
    single_bootstrap_samples: int = 399,
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
    if len(common_cols) < 1 or len(common_dates) < 10:
        return {}

    if factor_mat.index.equals(common_dates) and factor_mat.columns.equals(common_cols):
        f = factor_mat
    else:
        f = factor_mat.loc[common_dates, common_cols]
    if fwd_returns.index.equals(common_dates) and fwd_returns.columns.equals(common_cols):
        r = fwd_returns
    else:
        r = fwd_returns.loc[common_dates, common_cols]
    factor_values = f.to_numpy(dtype=float, copy=False)
    return_values = r.to_numpy(dtype=float, copy=False)

    # 按板块分组品种
    sector_positions = {}
    for position, ticker in enumerate(common_cols):
        sec = sector_map.get(str(ticker), "other")
        sector_positions.setdefault(sec, []).append(position)

    results = {}
    for sector, positions in sector_positions.items():
        if len(positions) == 1:
            single = _compute_single_instrument_ts_arrays(
                np.ascontiguousarray(factor_values[:, positions[0]]),
                np.ascontiguousarray(return_values[:, positions[0]]),
                common_dates,
                forward_period=forward_period,
                min_trading_days=single_min_trading_days,
                bootstrap_samples=single_bootstrap_samples,
            )
            if single:
                results[sector] = single
            continue
        # DataFrame column selection historically yielded C-contiguous arrays.
        # Preserve that reduction order so near-constant rows retain identical
        # zero-variance decisions after the numpy fast path.
        factor_sector = np.ascontiguousarray(factor_values[:, positions])
        returns_sector = np.ascontiguousarray(return_values[:, positions])
        if len(positions) < min_stocks:
            pooled = _compute_pooled_ts_fixed_effects_arrays(
                factor_sector,
                returns_sector,
                common_dates,
                forward_period=forward_period,
            )
            if pooled:
                results[sector] = pooled
            continue

        # Pearson IC and raw OLS share alignment, masking and centering.
        ic_series, ols_slopes = _cross_section_ic_and_slopes(
            factor_sector,
            returns_sector,
            common_dates,
            min_stocks=min_stocks,
        )
        if len(ic_series) < 10:
            continue
        ic_list = ic_series.tolist()
        ic_mean = float(np.mean(ic_list))
        ic_std = float(np.std(ic_list))
        ir = ic_mean / ic_std if ic_std > 0 else 0.0
        # IC remains the economic-effect diagnostic. Statistical admission is
        # based on the raw, unpenalized univariate Fama-MacBeth OLS slope.
        _, ic_t_stat = _newey_west_ir(ic_series, forward_period=forward_period)
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
            "inference_models": {
                "cross_section": "unpenalized_univariate_fama_macbeth_ols_hac",
                "small_sector": "unpenalized_pooled_ols_instrument_fe_time_hac",
            },
            "test_type": "cross_section_fama_macbeth",
            "p_value": p_value,
            "n_obs": len(ic_list),
            "hit_rate": hit_rate,
        }
    return results


def _compute_single_instrument_ts_arrays(
    factor: np.ndarray,
    returns: np.ndarray,
    index: pd.Index,
    *,
    forward_period: int,
    min_trading_days: int = 750,
    bootstrap_samples: int = 399,
) -> dict:
    """Single-instrument predictive regression with HAC and wild score test.

    The observation requirement is measured in unique trading dates, never in
    intraday bars.  Insufficient histories are retained as observation-channel
    hypotheses with p=1 so they remain visible in the same FDR family.
    """
    valid = np.isfinite(factor) & np.isfinite(returns)
    x = np.asarray(factor[valid], dtype=float)
    y = np.asarray(returns[valid], dtype=float)
    valid_index = pd.DatetimeIndex(index[valid])
    if len(x) < 20:
        return {}

    trading_dates = valid_index.normalize()
    n_trading_days = int(trading_dates.nunique())
    sufficient_history = n_trading_days >= int(min_trading_days)
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    x_centered = x - x_mean
    y_centered = y - y_mean
    denominator = float(np.dot(x_centered, x_centered))
    if denominator <= np.finfo(float).eps:
        return {}
    beta = float(np.dot(x_centered, y_centered) / denominator)
    residual = y_centered - beta * x_centered
    score = x_centered * residual

    max_lag = min(max(int(forward_period) - 1, 0), len(score) - 1)
    long_run = float(np.dot(score, score))
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        long_run += 2.0 * weight * float(np.dot(score[lag:], score[:-lag]))
    variance = max(long_run, 0.0) / denominator**2
    standard_error = float(np.sqrt(variance))
    hac_t = beta / standard_error if standard_error > 0.0 else 0.0
    hac_p = float(
        2.0 * scipy_stats.t.sf(abs(hac_t), df=max(n_trading_days - 1, 1))
    )

    day_codes, _ = pd.factorize(trading_dates, sort=True)
    # Wild cluster score test under H0: beta=0.  Using unrestricted OLS
    # residual scores here would make their total mechanically zero.
    null_score = x_centered * y_centered
    day_scores = np.bincount(day_codes, weights=null_score)
    cluster_scale = float(np.sqrt(np.dot(day_scores, day_scores)))
    if sufficient_history and cluster_scale > 0.0 and bootstrap_samples > 0:
        observed_score_t = float(day_scores.sum() / cluster_scale)
        rng = np.random.default_rng(20260728)
        extreme = 0
        remaining = int(bootstrap_samples)
        # Bound temporary memory for long histories and large bootstrap counts.
        while remaining:
            batch = min(remaining, 256)
            signs = rng.integers(0, 2, size=(batch, len(day_scores)), dtype=np.int8)
            signs = signs.astype(float) * 2.0 - 1.0
            boot_t = signs @ day_scores / cluster_scale
            extreme += int(
                np.count_nonzero(np.abs(boot_t) >= abs(observed_score_t))
            )
            remaining -= batch
        wild_p = float((extreme + 1) / (bootstrap_samples + 1))
    else:
        wild_p = 1.0
    conservative_p = max(hac_p, wild_p)

    x_std = float(x_centered.std(ddof=0))
    y_std = float(y_centered.std(ddof=0))
    predictive_corr = (
        float(np.dot(x_centered, y_centered) / (len(x) * x_std * y_std))
        if x_std > 0.0 and y_std > 0.0 else 0.0
    )
    oriented_product = pd.Series(
        x_centered * y_centered / max(x_std * y_std, np.finfo(float).eps),
        index=valid_index,
        dtype=float,
    )
    return {
        "ic_series": oriented_product,
        "ic_mean": predictive_corr,
        "ic_std": float(oriented_product.std(ddof=0)),
        "ir": (
            float(oriented_product.mean() / oriented_product.std(ddof=0))
            if float(oriented_product.std(ddof=0)) > 0.0 else 0.0
        ),
        "t_stat": float(hac_t),
        "ic_t_stat": float(hac_t),
        "ols_beta": beta,
        "ols_hac_t": float(hac_t),
        "ols_p_value": float(conservative_p if sufficient_history else 1.0),
        "hac_p_value": hac_p,
        "wild_bootstrap_p_value": wild_p,
        "ols_n": int(len(x)),
        "n_obs": int(len(x)),
        "n_trading_days": n_trading_days,
        "minimum_trading_days": int(min_trading_days),
        "sufficient_history": sufficient_history,
        "observation_channel": not sufficient_history,
        "inference_model": "single_instrument_ts_ols_hac_wild_score_bootstrap",
        "test_type": "single_instrument_time_series",
        "effect_metric": "predictive_correlation",
        "p_value": float(conservative_p if sufficient_history else 1.0),
        "hit_rate": float((oriented_product > 0.0).mean()),
    }


def _compute_pooled_ts_fixed_effects(
    factor_mat: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    *,
    forward_period: int,
) -> dict:
    """Pooled predictive regression for sectors with only two instruments."""
    common_dates = factor_mat.index.intersection(fwd_returns.index)
    common_cols = factor_mat.columns.intersection(fwd_returns.columns)
    if len(common_cols) < 2 or len(common_dates) < 20:
        return {}

    factor = factor_mat.loc[common_dates, common_cols].to_numpy(
        dtype=float, copy=False
    )
    returns = fwd_returns.loc[common_dates, common_cols].to_numpy(
        dtype=float, copy=False
    )
    return _compute_pooled_ts_fixed_effects_arrays(
        factor,
        returns,
        common_dates,
        forward_period=forward_period,
    )


def _compute_pooled_ts_fixed_effects_arrays(
    factor: np.ndarray,
    returns: np.ndarray,
    index: pd.Index,
    *,
    forward_period: int,
) -> dict:
    """Array implementation of the two-instrument fixed-effects test."""
    if factor.shape[1] < 2 or factor.shape[0] < 20:
        return {}
    available_values = ~np.isnan(factor) & ~np.isnan(returns)
    available_counts = available_values.sum(axis=0)
    factor_sums = np.where(available_values, factor, 0.0).sum(axis=0)
    return_sums = np.where(available_values, returns, 0.0).sum(axis=0)
    factor_means = np.divide(
        factor_sums,
        available_counts,
        out=np.full(factor.shape[1], np.nan, dtype=float),
        where=available_counts > 0,
    )
    return_means = np.divide(
        return_sums,
        available_counts,
        out=np.full(returns.shape[1], np.nan, dtype=float),
        where=available_counts > 0,
    )

    # Instrument fixed effects are removed over the declared evaluation sample.
    x_raw = np.where(available_values, factor - factor_means, np.nan)
    y_raw = np.where(available_values, returns - return_means, np.nan)
    valid_values = np.isfinite(x_raw) & np.isfinite(y_raw)
    x_values = np.where(valid_values, x_raw, 0.0)
    y_values = np.where(valid_values, y_raw, 0.0)
    if int(valid_values.sum()) < 50:
        return {}
    denominator = float(np.square(x_values).sum())
    if not np.isfinite(denominator) or denominator <= 0.0:
        return {}

    beta = float((x_values * y_values).sum() / denominator)
    residual = np.where(valid_values, y_values - beta * x_values, 0.0)
    score = (x_values * residual).sum(axis=1)
    active = valid_values.sum(axis=1) >= 2
    score = score[active]
    if len(score) < 20:
        return {}

    score = score - score.mean()
    n_times = len(score)
    gamma0 = float(np.dot(score, score) / n_times)
    max_lag = min(max(int(forward_period) - 1, 0), n_times - 1)
    long_run_variance = gamma0
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        covariance = float(np.dot(score[lag:], score[:-lag]) / n_times)
        long_run_variance += 2.0 * weight * covariance
    variance = max(n_times * long_run_variance, 0.0) / denominator ** 2
    standard_error = float(np.sqrt(variance))
    t_stat = beta / standard_error if standard_error > 0.0 else 0.0
    p_value = float(
        2.0 * scipy_stats.t.sf(abs(t_stat), df=max(n_times - 1, 1))
    )

    column_counts = valid_values.sum(axis=0)
    factor_std = np.sqrt(
        np.divide(
            np.square(x_values).sum(axis=0),
            column_counts,
            out=np.full(factor.shape[1], np.nan, dtype=float),
            where=column_counts > 0,
        )
    )
    return_std = np.sqrt(
        np.divide(
            np.square(y_values).sum(axis=0),
            column_counts,
            out=np.full(returns.shape[1], np.nan, dtype=float),
            where=column_counts > 0,
        )
    )
    standardized_valid = (
        valid_values
        & np.isfinite(factor_std)[None, :]
        & np.isfinite(return_std)[None, :]
        & (factor_std[None, :] > 0.0)
        & (return_std[None, :] > 0.0)
    )
    products = np.divide(
        x_values * y_values,
        factor_std[None, :] * return_std[None, :],
        out=np.zeros_like(x_values, dtype=float),
        where=standardized_valid,
    )
    row_counts = standardized_valid.sum(axis=1)
    predictive_values = np.divide(
        products.sum(axis=1),
        row_counts,
        out=np.full(factor.shape[0], np.nan, dtype=float),
        where=row_counts > 0,
    )
    predictive_mask = (row_counts >= 2) & np.isfinite(predictive_values)
    predictive_score = pd.Series(
        predictive_values[predictive_mask],
        index=index[predictive_mask],
        dtype=float,
    )
    if len(predictive_score) < 20:
        return {}
    ic_mean = float(predictive_score.mean())
    ic_std = float(predictive_score.std(ddof=0))
    ir = ic_mean / ic_std if ic_std > 0.0 else 0.0
    _, ic_t_stat = _newey_west_ir(
        predictive_score, forward_period=forward_period
    )
    hit_rate = float(
        (predictive_score > 0.0).mean()
        if ic_mean >= 0.0
        else (predictive_score < 0.0).mean()
    )

    return {
        "ic_series": predictive_score,
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ir": ir,
        "t_stat": float(t_stat),
        "ic_t_stat": float(ic_t_stat),
        "ols_beta": beta,
        "ols_hac_t": float(t_stat),
        "ols_p_value": p_value,
        "ols_n": int(n_times),
        "inference_model": "unpenalized_pooled_ols_instrument_fe_time_hac",
        "test_type": "pooled_time_series_fixed_effects",
        "p_value": p_value,
        "n_obs": int(len(predictive_score)),
        "hit_rate": hit_rate,
    }


def _compute_oos_consistency(
    ic_series: pd.Series,
    oos_ratio: float = OOS_SPLIT_RATIO,
    *,
    policy=None,
    frequency: str = "daily",
) -> dict:
    """样本外一致性验证，执行分频率样本下限和至少 3:1 切分.

    Returns:
        {"oos_ic": float, "oos_hit_rate": float, "is_consistent": bool}
    """
    from core.config import ValidationPolicyConfig
    from research.sample_policy import chronological_split

    policy = policy or ValidationPolicyConfig()
    required_fraction = float(policy.minimum_train_test_ratio) / (
        float(policy.minimum_train_test_ratio) + 1.0
    )
    if abs(float(oos_ratio) - required_fraction) > 1e-12:
        raise ValueError(
            f"OOS split ratio {oos_ratio} violates the frozen minimum-ratio "
            f"split {required_fraction}"
        )
    train, test, assessment = chronological_split(
        ic_series, policy=policy, frequency=frequency
    )
    base = {
        "sample_assessment": assessment.to_dict(),
        "observation_channel": not assessment.sufficient,
        "observation_reasons": list(assessment.reasons),
        "train_bars": len(train),
        "test_bars": len(test),
    }
    if not assessment.sufficient:
        return {
            "oos_ic": 0.0, "oos_hit_rate": 0.0,
            "is_consistent": False, **base,
        }
    train_direction = 1 if train.mean() >= 0 else -1
    oos_ic = float(test.mean())
    oos_hit_rate = float((test * train_direction > 0).mean())
    is_consistent = bool(
        oos_ic * train_direction > 0.0
        and oos_hit_rate >= OOS_CONSISTENCY_THRESHOLD
    )
    return {
        "oos_ic": oos_ic,
        "oos_hit_rate": oos_hit_rate,
        "is_consistent": bool(is_consistent),
        **base,
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
    single_min_trading_days: int = 750,
    single_bootstrap_samples: int = 399,
    validation_policy=None,
    frequency: str = "daily",
) -> dict:
    """Evaluate one factor across horizons using read-only shared inputs."""
    if validation_policy is not None and frequency != "daily":
        from research.sample_policy import minimum_training_days

        single_min_trading_days = minimum_training_days(
            validation_policy, frequency
        )
    result = {
        "best_sector": "",
        "best_period": 0,
        "best_ic": 0.0,
        "best_ir": 0.0,
        "best_t": 0.0,
        "valid_sectors": [],
        "sectors": {},
        "sample_sufficient": False,
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
            single_min_trading_days=single_min_trading_days,
            single_bootstrap_samples=single_bootstrap_samples,
        )
        for sector, values in sector_ics.items():
            oos = _compute_oos_consistency(
                values["ic_series"], policy=validation_policy,
                frequency=frequency,
            )
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
                "test_type": values.get(
                    "test_type", "cross_section_fama_macbeth"
                ),
                "inference_model": values.get("inference_model", ""),
                "observation_channel": bool(
                    values.get("observation_channel", False)
                    or oos.get("observation_channel", False)
                ),
                "observation_reasons": sorted(set(
                    list(oos.get("observation_reasons", []))
                    + (["insufficient_single_instrument_history"] if values.get(
                        "observation_channel", False
                    ) else [])
                )),
                "sample_assessment": oos.get("sample_assessment", {}),
                "sufficient_history": bool(
                    values.get("sufficient_history", True)
                ),
                "n_trading_days": int(values.get("n_trading_days", 0)),
                "passes_thresholds": (
                    _is_sector_valid(values)
                    and oos["is_consistent"]
                    and not oos.get("observation_channel", False)
                ),
                "is_valid": False,
            }
            sector_results.setdefault(sector, {})[str(period)] = entry
            result["sample_sufficient"] = bool(
                result["sample_sufficient"]
                or not oos.get("observation_channel", False)
            )

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
    from research.statistics import benjamini_hochberg

    if method not in {"bonferroni", "global", "hierarchical", "deployment"}:
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

    if method == "deployment":
        for entry in hypothesis_entries:
            entry["q_value"] = 1.0
            entry["fdr_significant"] = True
            entry["multiplicity_role"] = "frozen_deployment_parameter_selection"
            entry["is_valid"] = bool(entry["passes_thresholds"])
        for name, result in all_results.items():
            result["factor_p_value"] = min(
                (float(entry.get("p_value", 1.0)) for entry in factor_entries[name]),
                default=1.0,
            )
            result["factor_q_value"] = 1.0
            result["factor_fdr_significant"] = bool(factor_entries[name])
    elif method == "bonferroni":
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
        from research.validation import apply_hierarchical_fdr

        audit = apply_hierarchical_fdr(
            factor_entries,
            q=alpha,
            fwer_alpha=0.05,
            p_key="p_value",
        )
        for name in all_results:
            result = all_results[name]
            entries = factor_entries[name]
            first = entries[0] if entries else {}
            result["factor_p_value"] = float(first.get("factor_simes_p_value", 1.0))
            result["factor_q_value"] = float(first.get("factor_q_value", 1.0))
            result["factor_fdr_significant"] = bool(
                first.get("factor_fdr_significant", False)
            )
            result["hierarchical_fdr_audit"] = audit
            for entry in entries:
                entry["q_value"] = max(
                    float(entry.get("factor_q_value", 1.0)),
                    float(entry.get("local_q_value", 1.0)),
                )
                entry["fdr_significant"] = bool(
                    entry.get("hierarchical_fdr_significant", False)
                )
                entry["is_valid"] = bool(
                    entry["passes_thresholds"]
                    and entry["fdr_significant"]
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
            "observation_channel": bool(
                entry.get("observation_channel", False)
            ),
            "n_trading_days": int(entry.get("n_trading_days", 0)),
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
            "observation_channel": bool(best["observation_channel"]),
        })
    else:
        result.update({
            "best_sector": "", "best_period": 0, "best_ic": 0.0,
            "best_ir": 0.0, "best_t": 0.0, "best_q": 1.0,
            "observation_channel": False,
        })
    return sector_optima


def load_discovery_contract(
    path: str,
    *,
    expected_policy_sha256: str,
    expected_taxonomy_sha256: str,
) -> tuple[list[str], dict, dict]:
    """Load the frozen discovery set used by deployment-only adaptivity."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config = dict(payload.get("config", {}) or {})
    if config.get("validation_policy_sha256") != expected_policy_sha256:
        raise ValueError("discovery validation-policy hash mismatch; rerun P0")
    if config.get("taxonomy_sha256") != expected_taxonomy_sha256:
        raise ValueError("discovery taxonomy hash mismatch; rerun P0")
    names = [str(name) for name in payload.get("final_factors", [])]
    if not names or len(names) != len(set(names)):
        raise ValueError("discovery contract has no unique final_factors")
    name_set = set(names)
    rows = {
        str(row.get("name", "")): row
        for row in payload.get("significant_factors", [])
        if str(row.get("name", "")) in name_set
    }
    missing = sorted(name_set - set(rows))
    if missing:
        raise ValueError(f"discovery contract missing final-factor metadata: {missing}")
    variants = {
        name: str(rows[name].get("best_variant", "neutralized"))
        for name in names
    }
    metadata = {
        name: {
            "observation_channel": bool(
                rows[name].get("observation_channel", False)
            ),
            "observation_reasons": list(
                rows[name].get("observation_reasons", [])
            ),
            "promotion_status": str(
                rows[name].get("promotion_status", "observation")
            ),
            "weight_cap": float(rows[name].get("weight_cap", 1.0)),
        }
        for name in names
    }
    return names, variants, metadata


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
    factor_workers: int = None,
    config=None,
    runner=None,
    preprocessing_variants: dict = None,
    candidate_metadata: dict = None,
    fdr_method: str = None,
    fdr_alpha: float = None,
    frequency: str = "daily",
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
    period_ctx = PeriodContext.from_string(frequency)
    factor_workers = max(
        1,
        int(
            factor_workers
            or os.environ.get(
                "MF_ADAPTIVITY_FACTOR_WORKERS",
                1 if period_ctx.is_daily else min(2, workers),
            )
        ),
    )

    print("=" * 70)
    print(f"因子适配性研究 ({len(all_factors)} 个因子 × {len(SECTOR_MAP)} 品种 × {len(periods)} 持有期)")
    print("=" * 70)
    print(f"  配置文件: {config_path}")
    print(f"  因子计算区间: {factor_start} ~ {ic_end} (含预热)")
    print(f"  IC检验区间: {ic_start} ~ {ic_end}")
    print(f"  持有期 (周期数): {periods}")
    print(f"  周期单位: {period_ctx.unit.value}")
    print(f"  因子分析线程: {workers}")
    print(f"  非 SPEC 因子计算线程: {factor_workers}")
    print(f"  板块数: {len(set(SECTOR_MAP.values()))} "
          f"({', '.join(sorted(set(SECTOR_MAP.values())))})")
    print(f"  样本外验证: 前{int(OOS_SPLIT_RATIO*100)}%训练 / 后{int((1-OOS_SPLIT_RATIO)*100)}%测试")
    print(f"  有效性门槛: |IC|≥{SECTOR_IC_THRESHOLD}, |t|≥{SECTOR_T_THRESHOLD}, "
          f"命中率≥{SECTOR_HIT_RATE_THRESHOLD}, OOS一致≥{OOS_CONSISTENCY_THRESHOLD}")
    if sectors_filter:
        print(f"  仅分析板块: {sectors_filter}")
    print()

    from pipeline.runner import PipelineRunner

    runner = runner or (
        PipelineRunner(config=config)
        if config is not None
        else PipelineRunner(config_path)
    )
    config = runner.config
    policy = config.validation_policy
    from research.validation import validate_policy, validation_policy_sha256

    validate_policy(policy)
    from core.factor_contract import validate_factor_contract
    from core.registry import get as registry_get

    factor_periods: dict[str, tuple[int, ...]] = {}
    for factor_name in all_factors:
        factor = registry_get("factor", factor_name)()
        factor_periods[factor_name] = validate_factor_contract(
            factor, provider_frequency=period_ctx.unit.value,
            requested_horizons=getattr(factor, "validation_horizons", ()),
        )
    declared_periods = sorted({
        period for values in factor_periods.values() for period in values
    })
    if sorted(set(map(int, periods))) != declared_periods:
        raise ValueError(
            f"adaptivity periods {sorted(set(map(int, periods)))} do not match "
            f"the frozen factor-contract union {declared_periods}"
        )
    fdr_method = fdr_method or "hierarchical"
    fdr_alpha = (
        float(policy.discovery_q) if fdr_alpha is None else float(fdr_alpha)
    )
    policy_hash = validation_policy_sha256(policy)
    base_data_mgr = runner.data_manager
    universe = pd.Index(config.universe) if config.universe else pd.Index([])

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

    if len(calendar) == 0 and period_ctx.is_daily:
        cache_ok, cache_dates = _load_data_from_cache(
            data_mgr, universe, factor_start, ic_end,
        )
        if cache_ok and len(cache_dates) > 0:
            calendar = cache_dates
            if hasattr(calendar, "tz") and calendar.tz is not None:
                calendar = calendar.tz_localize(None)
            calendar = pd.DatetimeIndex(sorted(set(calendar)))

    valid_universe = [t for t in universe if str(t) in SECTOR_MAP]
    if sectors_filter:
        valid_universe = [
            t for t in valid_universe
            if SECTOR_MAP[str(t)] in sectors_filter
        ]
    valid_universe = pd.Index(valid_universe)
    print(
        f"研究日历: {len(calendar)} bars, universe: "
        f"{len(valid_universe)} mapped instruments"
    )

    print("\n预计算 forward returns...")
    processing_context = build_processing_context(
        data_mgr,
        calendar,
        valid_universe,
        config.universe_selection,
    )
    fwd_returns_by_period = {}
    for period in periods:
        forward_returns = data_mgr.get_forward_returns(
            calendar, valid_universe, period=period
        )
        if processing_context.eligibility is not None:
            forward_returns = forward_returns.where(
                processing_context.eligibility
            )
        fwd_returns_by_period[period] = forward_returns
    analysis_returns = {
        period: frame.loc[ic_start:ic_end]
        for period, frame in fwd_returns_by_period.items()
    }
    artifact_data_hashes = {}
    dataframe_sha256 = None
    if artifact_id:
        from research.artifacts import dataframe_sha256 as _dataframe_sha256

        dataframe_sha256 = _dataframe_sha256
        artifact_data_hashes.update({
            f"returns:{period}": dataframe_sha256(frame)
            for period, frame in analysis_returns.items()
        })

    print(
        f"\n=== 开始流式适配性 IC 检验 "
        f"({len(all_factors)} 因子 × {len(periods)} 持有期 × 多板块) ==="
    )
    all_results = {}
    t0 = time.time()
    default_chunk_size = 64 if period_ctx.is_daily else 16
    chunk_size = max(
        int(os.environ.get("MF_ADAPTIVITY_FACTOR_CHUNK_SIZE", default_chunk_size)),
        1,
    )
    print(f"  因子分块大小: {chunk_size}")
    ta_cn_inputs_ready = False

    for batch_start in range(0, len(all_factors), chunk_size):
        batch_names = all_factors[batch_start:batch_start + chunk_size]
        is_ta_cn_batch = any(
            name.startswith(("gtja191_alpha", "wq101_alpha"))
            for name in batch_names
        )
        if (
            not ta_cn_inputs_ready
            and is_ta_cn_batch
        ):
            from factors.user.ta_cn_formula_library import _formula_inputs

            _formula_inputs(data_mgr, calendar, valid_universe)
            ta_cn_inputs_ready = True
        engine = FactorEngine(data_mgr)
        computed_batch = engine.compute_factors(
            batch_names,
            calendar,
            valid_universe,
            parallel=factor_workers > 1,
            max_workers=factor_workers,
            chunk_size=chunk_size,
        )
        factor_batch = runner.processor.process_batch(
            computed_batch, processing_context
        )
        for name in batch_names:
            if (
                name in computed_batch
                and (preprocessing_variants or {}).get(name) == "raw"
            ):
                factor_batch[name] = runner.processor.process_excluding(
                    computed_batch[name], processing_context, {"neutralize"}
                )
        available = [name for name in batch_names if name in factor_batch]

        def _evaluate(name):
            analysis_factor = factor_batch[name].loc[ic_start:ic_end]
            factor_result = _analyze_factor_across_periods(
                analysis_factor,
                analysis_returns,
                list(factor_periods[name]),
                SECTOR_MAP,
                single_min_trading_days=policy.single_instrument_min_trading_days,
                single_bootstrap_samples=policy.single_instrument_bootstrap_samples,
                validation_policy=policy,
                frequency=period_ctx.unit.value,
            )
            factor_result["preprocessing_variant"] = (
                (preprocessing_variants or {}).get(name, "neutralized")
            )
            factor_result["candidate_metadata"] = dict(
                (candidate_metadata or {}).get(name, {})
            )
            frame_hash = (
                dataframe_sha256(analysis_factor)
                if dataframe_sha256 is not None
                else None
            )
            return name, factor_result, frame_hash

        if workers == 1:
            evaluated = map(_evaluate, available)
            pool = None
        else:
            pool = ThreadPoolExecutor(max_workers=workers)
            evaluated = pool.map(_evaluate, available)
        try:
            for fname, factor_result, frame_hash in evaluated:
                all_results[fname] = factor_result
                if frame_hash is not None:
                    artifact_data_hashes[f"factor:{fname}"] = frame_hash
        finally:
            if pool is not None:
                pool.shutdown(wait=True)

        engine.clear_cache()
        del computed_batch
        del factor_batch
        if is_ta_cn_batch:
            gc.collect()
        done = min(batch_start + chunk_size, len(all_factors))
        print(f"  [{done}/{len(all_factors)}] 耗时 {time.time() - t0:.1f}s")

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
            "frequency": period_ctx.unit.value,
            "bar_count": len(calendar),
            "periods": periods,
            "sectors": sorted(set(SECTOR_MAP.values())),
            "n_factors": n_total,
            "n_factors_with_valid_sector": n_with_valid_sector,
            "n_hypotheses": n_hypotheses,
            "analysis_workers": workers,
            "factor_compute_workers": factor_workers,
            "factor_chunk_size": chunk_size,
            "fdr_method": fdr_method,
            "fdr_alpha": fdr_alpha,
            "validation_policy_sha256": policy_hash,
            "taxonomy_version": TAXONOMY_VERSION,
            "taxonomy_sha256": taxonomy_sha256(),
            "multiplicity_role": (
                "deployment_parameter_selection"
                if fdr_method == "deployment" else "factor_discovery"
            ),
            "inference_model": "unpenalized_univariate_fama_macbeth_ols_hac",
            "factor_preprocessing": [
                "mad_winsorize_by_date",
                "candidate_selected_sector_neutralization",
                "zscore_standardize_by_date",
            ],
            "preprocessing_variant_source": (
                "frozen_discovery_contract"
                if preprocessing_variants else "neutralized_default"
            ),
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
        "observation_channel", "n_trading_days",
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
        candidate_observation = bool(
            res.get("candidate_metadata", {}).get("observation_channel", False)
        )
        local_observation = bool(res.get("observation_channel", False))
        observation_channel = candidate_observation or local_observation
        candidate_cap = float(
            res.get("candidate_metadata", {}).get("weight_cap", 1.0)
        )
        weight_cap = min(
            candidate_cap,
            float(policy.observation_weight_cap) if local_observation else 1.0,
        )
        summary_rows.append({
            "factor": fname,
            "preprocessing_variant": res.get(
                "preprocessing_variant", "neutralized"
            ),
            "observation_channel": observation_channel,
            "sample_sufficient": bool(res.get("sample_sufficient", False)),
            "weight_cap": weight_cap,
            "promotion_status": (
                "observation" if observation_channel
                else str(
                    res.get("candidate_metadata", {}).get(
                        "promotion_status", "wf_candidate"
                    )
                )
            ),
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
    discovery_path = os.path.join(output_dir, "ic_by_window_period.json")
    if os.path.isfile(discovery_path):
        artifact_files["factor_discovery"] = discovery_path
    funnel_path = os.path.join(output_dir, "validation_funnel.json")
    if os.path.isfile(funnel_path):
        artifact_files["validation_funnel"] = funnel_path
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

    def _compute_processed_matrices(names: list[str]) -> dict:
        if not names:
            return {}
        recompute_engine = FactorEngine(data_mgr)
        computed = recompute_engine.compute_factors(
            names,
            calendar,
            valid_universe,
            parallel=False,
            chunk_size=max(len(names), 1),
        )
        matrices = runner.processor.process_batch(computed, processing_context)
        for name in names:
            if (
                name in computed
                and (preprocessing_variants or {}).get(name) == "raw"
            ):
                matrices[name] = runner.processor.process_excluding(
                    computed[name], processing_context, {"neutralize"}
                )
        recompute_engine.clear_cache()
        return matrices

    if build_correlation and len(significant) >= 2:
        from factors.correlation_analysis import analyze_and_save

        significant_names = [item["name"] for item in significant]
        factor_matrices = _compute_processed_matrices(significant_names)
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
            dataframe_hash_collection_sha256,
            source_tree_hash,
        )

        bundle = ResearchArtifactBundle.create(
            output_dir,
            artifact_id=artifact_id,
            train_start=ic_start,
            train_end=ic_end,
            data_sha256=dataframe_hash_collection_sha256(
                artifact_data_hashes
            ),
            config_sha256=canonical_config_hash(config),
            code_sha256=source_tree_hash(Path(_PROJECT_ROOT)),
            files=artifact_files,
            metadata={
                "candidate_factors": len(all_factors),
                "selected_factors": len(significant),
                "periods": periods,
                "frequency": period_ctx.unit.value,
                "fdr_method": fdr_method,
                "fdr_alpha": fdr_alpha,
                "validation_policy_sha256": policy_hash,
                "taxonomy_sha256": taxonomy_sha256(),
                "data_hash_mode": "streamed_dataframe_collection_v1",
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
        "--frequency",
        choices=["daily", "1min", "5min", "15min", "30min", "hourly"],
        default="daily",
        help="bar frequency used for factor windows and holding periods",
    )
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
        "--factor-workers", type=int, default=None,
        help="非 SPEC 因子计算线程数 (日频默认1, 日内默认min(2, --workers))")
    parser.add_argument(
        "--artifact-id", default=None,
        help="非空时在输出完成后创建不可覆盖的研究 bundle manifest")
    parser.add_argument(
        "--no-correlation", action="store_true",
        help="不生成训练期因子相关性聚类产物")
    parser.add_argument(
        "--fdr-method", choices=["bonferroni", "global", "hierarchical", "deployment"],
        default="hierarchical",
        help=("多重检验: bonferroni=全部原始OLS/HAC p值全局校正; "
              "global=全部局部假设统一BH; hierarchical=层级FDR; "
              "deployment=仅对已冻结发现集做局部门槛适配"))
    parser.add_argument(
        "--fdr-alpha", type=float, default=None,
        help="覆盖验证策略中的 discovery_q；默认读取 validation_policy",
    )
    parser.add_argument(
        "--discovery-file", default=None,
        help=("deployment 模式必填：P0 输出的 ic_by_window_period.json；"
              "自动加载冻结因子、预处理版本和观察期元数据"),
    )
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

    preprocessing_variants = None
    candidate_metadata = None
    if args.fdr_method == "deployment":
        if not args.discovery_file:
            parser.error("--fdr-method deployment requires --discovery-file")
        from research.validation import validation_policy_sha256

        discovery_path = args.discovery_file
        if not os.path.isabs(discovery_path):
            discovery_path = os.path.join(_PROJECT_ROOT, discovery_path)
        frozen_names, preprocessing_variants, candidate_metadata = (
            load_discovery_contract(
                os.path.normpath(discovery_path),
                expected_policy_sha256=validation_policy_sha256(
                    config.validation_policy
                ),
                expected_taxonomy_sha256=taxonomy_sha256(),
            )
        )
        if args.factors:
            requested = [
                value.strip() for value in args.factors.split(",")
                if value.strip()
            ]
            unknown = sorted(set(requested) - set(frozen_names))
            if unknown:
                parser.error(
                    f"--factors contains names outside frozen discovery: {unknown}"
                )
            all_factors = list(dict.fromkeys(requested))
            preprocessing_variants = {
                name: preprocessing_variants[name] for name in all_factors
            }
            candidate_metadata = {
                name: candidate_metadata[name] for name in all_factors
            }
        else:
            all_factors = frozen_names
    else:
        if args.discovery_file:
            parser.error("--discovery-file is only valid with deployment mode")
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
        factor_workers=args.factor_workers,
        config=config,
        preprocessing_variants=preprocessing_variants,
        candidate_metadata=candidate_metadata,
        fdr_method=args.fdr_method,
        fdr_alpha=args.fdr_alpha,
        frequency=args.frequency,
    )


if __name__ == "__main__":
    main()
