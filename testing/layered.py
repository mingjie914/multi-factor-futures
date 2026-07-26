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
        group_labels = [f"Q{i+1}" for i in range(self.n_groups)]
        group_rets: Dict[str, list] = {g: [] for g in group_labels}
        # CR-026: 记录每个有效观测的真实日期 (跳过无效日期后不再用原始位置)
        valid_dates: list = []

        # CR-026: holding_period 可通过 params 覆盖
        holding_period = params.get("holding_period", self.holding_period)

        for dt in common_dates:
            f_row = factor.loc[dt].dropna()
            r_row = forward_returns.loc[dt]
            common = f_row.index.intersection(r_row.index)
            if len(common) < self.n_groups * 2:
                continue

            # 用 qcut 替代手动排序分组，向量化计算组内均值
            # CR-026: Q1 = 最低因子值组, Q5 = 最高因子值组
            f_vals = f_row[common]
            r_vals = r_row[common]
            labels = pd.qcut(f_vals.rank(method="first"), self.n_groups,
                             labels=group_labels)
            grouped = r_vals.groupby(labels).mean()
            for g in group_labels:
                group_rets[g].append(float(grouped.get(g, 0.0)))
            valid_dates.append(dt)

        # CR-026: 使用真实有效日期作为索引 (不再用 common_dates 的前 N 个位置)
        valid_index = pd.DatetimeIndex(valid_dates)
        gs: Dict[str, pd.Series] = {}
        for g in group_labels:
            gs[g] = pd.Series(group_rets[g], index=valid_index)

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
