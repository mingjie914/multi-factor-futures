from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from backtest.metrics import RISK_FREE_RATE

from core.registry import register
from core.types import FactorMatrix, ReturnMatrix, UniverseSchedule
from testing.base import FactorTest, TestResult


class LayeredResult(TestResult):
    """Result container for layered (grouped) backtest."""

    def __init__(
        self,
        group_returns: Dict[str, pd.Series],
        long_short_nav: pd.Series,
        group_metrics: Dict[str, dict],
        monotonicity: float,
    ):
        self.group_returns = group_returns
        self.long_short_nav = long_short_nav
        self.group_metrics = group_metrics
        self.monotonicity = monotonicity

    def to_dict(self) -> dict:
        return {
            "group_metrics": self.group_metrics,
            "monotonicity": self.monotonicity,
        }

    def summary(self) -> str:
        # CR-026: Q1 = 最低因子值组 (Bottom), Q5 = 最高因子值组 (Top)
        top = self.group_metrics.get("Q5", {}).get("annual_return", 0)
        bot = self.group_metrics.get("Q1", {}).get("annual_return", 0)
        ls = self.group_metrics.get("long_short", {}).get("annual_return", 0)
        return (
            f"Top(Q5)={top:.2%} Bottom(Q1)={bot:.2%} "
            f"L/S={ls:.2%} mono={self.monotonicity:.3f}"
        )


@register("factor_test", "layered")
class LayeredBacktest(FactorTest):
    """Layered / grouped backtest that sorts stocks into quantiles by factor value
    and tracks the forward return of each group.
    """

    name = "layered"

    def __init__(self, n_groups: int = 5, holding_period: int = 1):
        self.n_groups = n_groups
        # CR-026: 持有期 (前向收益天数), 用于非重叠调仓频率年化
        self.holding_period = holding_period

    def run(
        self,
        factor: FactorMatrix,
        forward_returns: ReturnMatrix,
        universe: UniverseSchedule = None,
        **params,
    ) -> LayeredResult:
        common_dates = factor.index.intersection(forward_returns.index)
        common_columns = factor.columns.intersection(forward_returns.columns)
        group_labels = [f"Q{i+1}" for i in range(self.n_groups)]

        # CR-026: holding_period 可通过 params 覆盖
        holding_period = params.get("holding_period", self.holding_period)

        aligned_factor = factor.loc[common_dates, common_columns]
        aligned_returns = forward_returns.loc[common_dates, common_columns]
        ranks = aligned_factor.rank(axis=1, method="first").to_numpy(
            dtype=float, copy=False
        )
        returns = aligned_returns.to_numpy(dtype=float, copy=False)
        counts = np.isfinite(ranks).sum(axis=1)
        valid_rows = counts >= self.n_groups * 2
        valid_index = pd.DatetimeIndex(common_dates[valid_rows])
        group_ids = np.zeros_like(ranks, dtype=np.int16)
        for boundary_index in range(1, self.n_groups):
            boundary = (
                1.0
                + boundary_index * (np.maximum(counts, 1) - 1.0)
                / self.n_groups
            )
            group_ids += ranks > boundary[:, None]

        gs: Dict[str, pd.Series] = {}
        for group_index, label in enumerate(group_labels):
            selected = (
                valid_rows[:, None]
                & np.isfinite(ranks)
                & (group_ids == group_index)
            )
            finite = selected & np.isfinite(returns)
            denominator = finite.sum(axis=1)
            numerator = np.where(finite, returns, 0.0).sum(axis=1)
            means = np.divide(
                numerator,
                denominator,
                out=np.zeros(len(common_dates), dtype=float),
                where=denominator > 0,
            )
            gs[label] = pd.Series(means[valid_rows], index=valid_index)

        # CR-026: 统一 spread 方向: Q5(高因子) - Q1(低因子), 标准多空方向
        ls = (gs[group_labels[-1]] - gs[group_labels[0]]).fillna(0)
        ls_cum = (1 + ls).cumprod()
        gs["long_short"] = ls

        # CR-026: 年化按非重叠调仓频率 (252/holding_period)
        ann_factor = 252.0 / holding_period if holding_period > 0 else 252.0

        # Per-group metrics
        def _metrics(s: pd.Series) -> dict:
            s = s.dropna()
            ann_ret = float(s.mean() * ann_factor) if len(s) > 0 else 0.0
            ann_vol = float(s.std() * np.sqrt(ann_factor)) if len(s) > 0 else 0.0
            sharpe = (
                (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0.0
            )
            return {"annual_return": ann_ret, "annual_vol": ann_vol, "sharpe": sharpe}

        gm: Dict[str, dict] = {g: _metrics(gs[g]) for g in group_labels}
        gm["long_short"] = _metrics(ls)

        # Monotonicity: rank correlation of Q1~Q5 annual returns
        ann_rets = np.asarray(
            [gm[g]["annual_return"] for g in group_labels], dtype=float
        )
        mono = (
            float(np.corrcoef(np.arange(len(ann_rets)), ann_rets)[0, 1])
            if len(ann_rets) > 1 and np.nanstd(ann_rets) > 1e-12
            else 0.0
        )

        return LayeredResult(gs, ls_cum, gm, mono)
