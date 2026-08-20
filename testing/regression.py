"""Fama-MacBeth 截面回归检验.

用 numpy.linalg.lstsq 实现, 不需要 sklearn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.registry import register
from core.types import FactorMatrix, ReturnMatrix, UniverseSchedule
from testing.base import FactorTest, TestResult


def _vectorized_univariate_ols(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    min_stocks: int = 10,
) -> pd.Series:
    """Return daily raw OLS slopes for ``return ~ 1 + factor``.

    This is the unpenalized univariate Fama-MacBeth first pass.  The closed
    form is equivalent to fitting ``numpy.linalg.lstsq`` on every date, while
    avoiding a Python loop over thousands of cross-sections.
    """
    common_dates = factor.index.intersection(forward_returns.index)
    common_cols = factor.columns.intersection(forward_returns.columns)
    if len(common_cols) < min_stocks or len(common_dates) == 0:
        return pd.Series(dtype=float)

    x = factor.loc[common_dates, common_cols].to_numpy(dtype=float, copy=False)
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
    denominator = np.einsum("ij,ij->i", x_centered, x_centered)
    numerator = np.einsum("ij,ij->i", x_centered, y_centered)
    usable = (
        (counts >= min_stocks)
        & np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator > np.finfo(float).eps)
    )
    slopes = np.divide(
        numerator,
        denominator,
        out=np.full(len(common_dates), np.nan, dtype=float),
        where=usable,
    )
    return pd.Series(slopes[usable], index=common_dates[usable], dtype=float)


class RegressionResult(TestResult):
    """Result container for Fama-MacBeth style cross-sectional regression."""

    def __init__(
        self,
        factor_returns: pd.DataFrame,
        t_stats: pd.Series,
        r_squared: pd.Series,
        avg_r2: float,
    ):
        self.factor_returns = factor_returns
        self.t_stats = t_stats
        self.r_squared = r_squared
        self.avg_r2 = avg_r2

    def to_dict(self) -> dict:
        return {
            "factor_returns": self.factor_returns.mean().to_dict(),
            "t_stats": self.t_stats.to_dict(),
            "avg_r2": self.avg_r2,
        }

    def summary(self) -> str:
        if self.factor_returns.empty:
            return "FM regression: 无有效截面 (all dates skipped)"
        fr = self.factor_returns.mean()
        parts = [f"{k}: {fr[k]:.4f}" for k in fr.index[:3]]
        return f"FM regression: {' | '.join(parts)} | avg R\u00b2={self.avg_r2:.3f}"


@register("factor_test", "regression")
class RegressionTest(FactorTest):
    """Fama-MacBeth style cross-sectional regression test.

    Runs a univariate regression of forward returns on factor exposures
    for each date, then aggregates the time series of factor returns.
    Uses numpy.linalg.lstsq (no sklearn dependency).
    """

    name = "regression"

    def __init__(self, weighted: bool = False, forward_period: int = 1):
        self.weighted = weighted
        # CR-027: 前向收益天数, 用于 Newey-West HAC 滞后阶数
        if isinstance(forward_period, bool) or int(forward_period) != forward_period or forward_period < 1:
            raise ValueError("forward_period must be a positive integer")
        self.forward_period = int(forward_period)

    def run(
        self,
        factor: FactorMatrix,
        forward_returns: ReturnMatrix,
        universe: UniverseSchedule = None,
        **params,
    ) -> RegressionResult:
        common_dates = factor.index.intersection(forward_returns.index)
        fr_list: list[float] = []
        r2_list: list[float] = []
        # CR-027: 记录实际成功的日期 (跳过日期后不再用最前面的日期标记)
        valid_dates: list = []

        # CR-027: forward_period 可通过 params 覆盖
        forward_period = params.get("forward_period", self.forward_period)
        if isinstance(forward_period, bool) or int(forward_period) != forward_period or forward_period < 1:
            raise ValueError("forward_period must be a positive integer")
        forward_period = int(forward_period)
        # WLS is only valid with weights known before the tested return.  Examples
        # are lagged liquidity or inverse volatility.  Response-derived residual
        # weights are intentionally unsupported because they make a few near-zero
        # residual observations dominate the same cross-section being tested.
        sample_weights = params.get("sample_weights")
        if self.weighted and sample_weights is None:
            raise ValueError("sample_weights are required when weighted=True")

        for dt in common_dates:
            f_row = factor.loc[dt].dropna()
            r_row = forward_returns.loc[dt].dropna()
            common = f_row.index.intersection(r_row.index)
            if len(common) < 10:
                continue
            x = f_row[common].values.reshape(-1, 1)
            y_val = r_row[common].values
            if np.isnan(x).any() or np.isnan(y_val).any():
                continue

            # 加截距列
            X_with_intercept = np.column_stack([np.ones(x.shape[0]), x])

            weights = None
            if self.weighted and sample_weights is not None:
                weights = self._weights_for_date(sample_weights, dt, common)
            if weights is not None:
                sqrt_weights = np.sqrt(weights)
                coef, _, _, _ = np.linalg.lstsq(
                    X_with_intercept * sqrt_weights[:, None],
                    y_val * sqrt_weights,
                    rcond=None,
                )
            else:
                coef, _, _, _ = np.linalg.lstsq(X_with_intercept, y_val, rcond=None)

            beta = coef[1]  # 因子收益 (斜率)

            # R²
            y_pred = X_with_intercept @ coef
            if weights is None:
                ss_res = np.sum((y_val - y_pred) ** 2)
                ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
            else:
                weighted_mean = float(np.average(y_val, weights=weights))
                ss_res = np.sum(weights * (y_val - y_pred) ** 2)
                ss_tot = np.sum(weights * (y_val - weighted_mean) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            fr_list.append(float(beta))
            r2_list.append(float(r2))
            valid_dates.append(dt)

        # CR-027: 使用实际成功的日期作为索引 (不再用 common_dates 的前 N 个位置)
        idx = pd.DatetimeIndex(valid_dates)
        fr_df = pd.DataFrame({"factor_return": fr_list}, index=idx)
        r2_series = pd.Series(r2_list, index=idx)

        if len(fr_list) == 0:
            avg_r2 = np.nan
            t_stat = 0.0
        else:
            avg_r2 = float(np.mean(r2_list))
            # CR-027: 时间序列均值显著性使用 Newey-West HAC
            # (重叠前向收益的 t 值需做 HAC 调整)
            t_stat = _newey_west_t_stat(fr_df["factor_return"], forward_period)
        t_stats = pd.Series({"factor": t_stat})

        return RegressionResult(fr_df, t_stats, r2_series, avg_r2)

    @staticmethod
    def _weights_for_date(sample_weights, date, assets: pd.Index) -> np.ndarray:
        """Align, normalize and clip ex-ante cross-sectional WLS weights."""
        if isinstance(sample_weights, pd.DataFrame):
            if date not in sample_weights.index:
                raise ValueError(f"sample_weights has no row for {date}")
            row = sample_weights.loc[date].reindex(assets)
        elif isinstance(sample_weights, pd.Series):
            row = sample_weights.reindex(assets)
        else:
            values = np.asarray(sample_weights, dtype=float).reshape(-1)
            if values.size != len(assets):
                raise ValueError("sample_weights length does not match the cross-section")
            row = pd.Series(values, index=assets)

        values = row.to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("sample_weights must be finite and strictly positive")
        median = float(np.median(values))
        if median <= 0:
            raise ValueError("sample_weights median must be positive")
        values = values / median
        lower, upper = np.quantile(values, [0.05, 0.95])
        values = np.clip(values, max(lower, 0.1), min(max(upper, 0.1), 10.0))
        return values / float(values.mean())


def _newey_west_t_stat(series: pd.Series, forward_period: int) -> float:
    """计算 Newey-West HAC 调整的 t 统计量 (CR-027).

    重叠前向收益 (forward_period > 1) 导致因子收益序列存在自相关,
    普通 t 统计量会低估标准误. Newey-West HAC 修正这种自相关.

    Args:
        series: 因子收益时间序列
        forward_period: 前向收益天数 (用于确定滞后阶数)

    Returns:
        Newey-West 调整后的 t 统计量
    """
    values = pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(values)
    if n < 2:
        return 0.0

    mean = float(values.mean())
    centered = values.to_numpy(dtype=float) - mean
    gamma_0 = float(centered @ centered / n)
    if gamma_0 <= np.finfo(float).eps:
        return 0.0

    # 滞后阶数: 取 forward_period (重叠期长度)
    lag = min(forward_period, n - 1)
    if lag < 1:
        lag = 1

    nw_var = gamma_0
    for l in range(1, lag + 1):
        gamma_l = float(centered[l:] @ centered[:-l] / n)
        # Bartlett 核权重
        weight = 1 - l / (lag + 1)
        nw_var += 2 * weight * gamma_l

    # 与 IC 检验保持一致：重叠收益的 HAC 修正不能比 iid 推断更激进。
    nw_var = max(float(nw_var), gamma_0)
    se_nw = np.sqrt(nw_var / n)
    return float(mean / se_nw) if se_nw > 0 else 0.0
