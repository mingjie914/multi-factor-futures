from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from core.registry import register
from core.types import FactorMatrix, ReturnMatrix, UniverseSchedule
from testing.base import FactorTest, TestResult


class ICTestResult(TestResult):
    """Result container for IC-based factor tests."""

    def __init__(
        self,
        ic_series: pd.Series,
        rank_ic_series: pd.Series,
        ic_decay: Dict[int, float],
        ic_mean: float,
        ic_std: float,
        ir: float,
        ir_newey_west: float = 0.0,
        t_stat: float = 0.0,
        forward_period: int = 1,
        n_obs: int = 0,
    ):
        self.ic_series = ic_series
        self.rank_ic_series = rank_ic_series
        self.ic_decay = ic_decay
        self.ic_mean = ic_mean
        self.ic_std = ic_std
        self.ir = ir
        self.ir_newey_west = ir_newey_west
        self.t_stat = t_stat
        self.forward_period = forward_period
        self.n_obs = n_obs

    def to_dict(self) -> dict:
        return {
            "ic_mean": self.ic_mean,
            "ic_std": self.ic_std,
            "ir": self.ir,
            "ir_nw": self.ir_newey_west,
            "t_stat": self.t_stat,
            "n_obs": self.n_obs,
            "forward_period": self.forward_period,
            "ic_pos_ratio": (self.ic_series > 0).mean() if len(self.ic_series) > 0 else 0.0,
            "ic_decay": self.ic_decay,
        }

    def summary(self) -> str:
        d = self.to_dict()
        return (
            f"IC={d['ic_mean']:.4f}(±{d['ic_std']:.4f}) "
            f"IR={d['ir']:.3f} IR_NW={d['ir_nw']:.3f} "
            f"t={d['t_stat']:.2f} n={d['n_obs']} fwd={d['forward_period']}d "
            f"pos={d['ic_pos_ratio']:.1%}"
        )


@register("factor_test", "ic")
class ICTest(FactorTest):
    """Information Coefficient (IC) test for factor efficacy.

    支持全交易日截面 + N日持有期收益.
    当 forward_period > 1 时, 收益有重叠, 使用 Newey-West HAC 调整 IR.
    """

    name = "ic"

    def __init__(
        self,
        methods: list = None,
        decay_periods: list = None,
        forward_period: int = 1,
    ):
        self._methods = list(methods) if methods is not None else ["pearson", "spearman"]
        unsupported = set(self._methods) - {"pearson", "spearman"}
        if unsupported:
            raise ValueError(f"unsupported IC methods: {sorted(unsupported)}")
        if not self._methods:
            raise ValueError("at least one IC method is required")
        self._decay_periods = (
            list(decay_periods) if decay_periods is not None else [1, 5, 10, 20]
        )
        self._forward_period = forward_period

    def run(
        self,
        factor: FactorMatrix,
        forward_returns: ReturnMatrix,
        universe: UniverseSchedule = None,** params,
    ) -> ICTestResult:
        common_dates = factor.index.intersection(forward_returns.index)
        f = factor.loc[common_dates]
        r = forward_returns.loc[common_dates]

        # === 向量化 IC 计算 (替代逐日循环) ===
        # 对齐列 (品种), 仅保留两者都有的列
        common_cols = f.columns.intersection(r.columns)
        if len(common_cols) < 10:
            return ICTestResult(
                pd.Series(dtype=float), pd.Series(dtype=float), {},
                0.0, 0.0, 0.0, forward_period=self._forward_period, n_obs=0,
            )
        f = f[common_cols]
        r = r[common_cols]

        if "pearson" in self._methods:
            ic_series, _ = _vectorized_pearson_ic(f, r, min_stocks=10)
        else:
            ic_series = pd.Series(dtype=float)
        if "spearman" in self._methods:
            rank_ic_series, _ = _vectorized_spearman_ic(f, r, min_stocks=10)
        else:
            rank_ic_series = pd.Series(dtype=float)

        # Pearson remains the primary statistic when requested. A
        # Spearman-only test uses rank IC as its primary series.
        primary_series = ic_series if "pearson" in self._methods else rank_ic_series

        ic_list = primary_series.tolist()
        ic_mean = float(np.mean(ic_list)) if ic_list else 0.0
        ic_std = float(np.std(ic_list)) if ic_list else 0.0
        ir = ic_mean / ic_std if ic_std > 0 else 0.0

        # Newey-West 调整 IR (处理重叠收益导致的自相关)
        ir_nw, t_stat = _newey_west_ir(primary_series, self._forward_period)

        # IC decay (autocorrelation at various lags)
        decay: Dict[int, float] = {}
        for lag in self._decay_periods:
            if 0 < lag < len(primary_series):
                left = primary_series.iloc[:-lag].to_numpy(dtype=float)
                right = primary_series.iloc[lag:].to_numpy(dtype=float)
                finite = np.isfinite(left) & np.isfinite(right)
                left = left[finite]
                right = right[finite]
                if (
                    len(left) < 2
                    or np.std(left) <= 1e-12
                    or np.std(right) <= 1e-12
                ):
                    decay[lag] = 0.0
                else:
                    decay[lag] = float(np.corrcoef(left, right)[0, 1])
            elif lag == 0:
                decay[lag] = 1.0
            else:
                decay[lag] = 0.0

        return ICTestResult(
            primary_series, rank_ic_series, decay,
            ic_mean, ic_std, ir,
            ir_newey_west=ir_nw,
            t_stat=t_stat,
            forward_period=self._forward_period,
            n_obs=len(ic_list),
        )


def _vectorized_pearson_ic(
    factor: pd.DataFrame, returns: pd.DataFrame, min_stocks: int = 10
) -> tuple[pd.Series, pd.Series]:
    """向量化计算逐日 Pearson IC.

    利用相关系数公式: corr(F,R) = mean(F_std * R_std)
    其中 F_std, R_std 是按行 (日期) 去均值并除以标准差的标准化值.
    NaN 处理: 逐行仅对两者都非 NaN 的列参与计算.

    Returns:
        (ic_series, valid_mask): IC 时间序列 + 有效日期布尔掩码
    """
    factor_values = factor.to_numpy(dtype=float, copy=False)
    return_values = returns.to_numpy(dtype=float, copy=False)
    valid = ~np.isnan(factor_values) & ~np.isnan(return_values)
    n_valid = valid.sum(axis=1)
    date_mask_values = n_valid >= min_stocks
    date_mask = pd.Series(date_mask_values, index=factor.index, dtype=bool)
    if not date_mask_values.any():
        return pd.Series(dtype=float), pd.Series(dtype=bool)

    counts = np.maximum(n_valid, 1)
    factor_masked = np.where(valid, factor_values, 0.0)
    returns_masked = np.where(valid, return_values, 0.0)
    factor_mean = factor_masked.sum(axis=1) / counts
    returns_mean = returns_masked.sum(axis=1) / counts
    factor_centered = np.where(
        valid, factor_values - factor_mean[:, None], 0.0
    )
    returns_centered = np.where(
        valid, return_values - returns_mean[:, None], 0.0
    )

    dot = np.einsum("ij,ij->i", factor_centered, returns_centered)
    factor_ss = np.einsum("ij,ij->i", factor_centered, factor_centered)
    returns_ss = np.einsum("ij,ij->i", returns_centered, returns_centered)
    denom = np.sqrt(factor_ss * returns_ss)
    usable = (
        date_mask_values
        & np.isfinite(dot)
        & np.isfinite(denom)
        & (denom > 0.0)
    )
    values = np.divide(
        dot,
        denom,
        out=np.full(len(dot), np.nan, dtype=float),
        where=usable,
    )
    ic = pd.Series(values[usable], index=factor.index[usable], dtype=float)
    return ic, date_mask


def _vectorized_spearman_ic(
    factor: pd.DataFrame, returns: pd.DataFrame, min_stocks: int = 10
) -> tuple[pd.Series, pd.Series]:
    """向量化计算逐日 Spearman IC (rank IC).

    对每行的因子值和收益值分别求 rank, 然后计算 Pearson 相关.
    """
    valid = factor.notna() & returns.notna()
    n_valid = valid.sum(axis=1)
    date_mask = n_valid >= min_stocks
    if not date_mask.any():
        return pd.Series(dtype=float), pd.Series(dtype=bool)

    # 逐行 rank (仅对有效值排名, NaN 保持 NaN)
    f_rank = factor.where(valid).rank(axis=1)
    r_rank = returns.where(valid).rank(axis=1)

    # rank 后复用 Pearson IC 计算
    return _vectorized_pearson_ic(f_rank, r_rank, min_stocks=min_stocks)


def _newey_west_ir(ic_series: pd.Series, forward_period: int) -> tuple:
    """计算 Newey-West HAC 调整后的 IR 和 t 统计量.

    当使用 N 日持有期收益 + 全交易日截面时, IC 序列存在自相关
    (相邻截面的收益有 N-1 天重叠). Newey-West 调整修正了这种自相关
    对标准误的低估, 使 t 检验更可靠.

    Args:
        ic_series: IC 时间序列
        forward_period: 持有期天数 (用于确定滞后阶数)

    Returns:
        (ir_newey_west, t_stat)
    """
    values = np.asarray(ic_series, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return 0.0, 0.0

    ic_mean = float(values.mean())
    centered = values - ic_mean
    gamma_0 = float(centered @ centered / n)
    if gamma_0 <= np.finfo(float).eps:
        return 0.0, 0.0

    # 滞后阶数: 取 forward_period (重叠期长度)
    lag = min(forward_period, n - 1)
    if lag < 1:
        lag = 1

    # Newey-West HAC long-run variance. Use one denominator throughout;
    # mixing pandas autocorrelation with sample variance can over-cancel the
    # variance and create implausibly large t statistics.
    nw_var = gamma_0

    for l in range(1, lag + 1):
        gamma_l = float(centered[l:] @ centered[:-l] / n)
        # Bartlett 核权重
        weight = 1 - l / (lag + 1)
        nw_var += 2 * weight * gamma_l

    # Forward returns overlap by construction, so HAC is used here as a
    # conservative correction and must not make inference more significant
    # than the iid estimate when finite-sample autocovariances turn negative.
    nw_var = max(float(nw_var), gamma_0)
    se_nw = np.sqrt(nw_var / n)
    t_stat = ic_mean / se_nw if se_nw > 0 else 0.0
    ir_nw = ic_mean / np.sqrt(nw_var) if nw_var > 0 else 0.0

    return float(ir_nw), float(t_stat)
