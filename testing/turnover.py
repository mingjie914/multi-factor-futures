"""因子换手率检验.

把因子截面排序转换为可交易的市场中性多空权重, 再衡量目标权重变化.
高换手率因子在扣除交易成本后实际收益可能大幅下降.

换手率定义:
    rank_score_t = rank_pct(factor_t) - cross_section_mean
    weight_t = rank_score_t / Σ_i |rank_score_i,t|
    turnover_t = 0.5 × Σ_i |weight_i,t - weight_i,t-1|

月换手率 = 日均换手率 × 21 (交易日)
业界阈值: 月换手 < 50% (即日均 < 2.4%)
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from core.registry import register
from core.types import FactorMatrix, ReturnMatrix, UniverseSchedule
from testing.base import FactorTest, TestResult


class TurnoverResult(TestResult):
    """换手率检验结果."""

    def __init__(
        self,
        daily_turnover: float,
        monthly_turnover: float,
        annual_turnover: float,
        turnover_series: pd.Series,
    ):
        self.daily_turnover = daily_turnover
        self.monthly_turnover = monthly_turnover
        self.annual_turnover = annual_turnover
        self.turnover_series = turnover_series

    def to_dict(self) -> dict:
        return {
            "daily_turnover": float(self.daily_turnover),
            "monthly_turnover": float(self.monthly_turnover),
            "annual_turnover": float(self.annual_turnover),
            "passes_threshold": self.monthly_turnover < 0.50,
        }

    def summary(self) -> str:
        d = self.to_dict()
        status = "✓" if d["passes_threshold"] else "✗"
        return (
            f"换手率: 日均={d['daily_turnover']:.4f} "
            f"月={d['monthly_turnover']:.2%} "
            f"年={d['annual_turnover']:.2%} {status}"
        )


@register("factor_test", "turnover")
class TurnoverTest(FactorTest):
    """因子换手率检验.

    该定义对因子平移和正比例缩放不敏感, 且与真实目标持仓换手一致.
    """

    name = "turnover"

    def __init__(self, annualization: int = 252):
        self.annualization = annualization

    def run(
        self,
        factor: FactorMatrix,
        forward_returns: ReturnMatrix = None,
        universe: UniverseSchedule = None,
        **params,
    ) -> TurnoverResult:
        numeric = factor.replace([np.inf, -np.inf], np.nan).astype(float)
        ranks = numeric.rank(axis=1, method="average", pct=True)
        centered = ranks.sub(ranks.mean(axis=1), axis=0)
        gross = centered.abs().sum(axis=1).replace(0.0, np.nan)
        target_weights = centered.div(gross, axis=0).fillna(0.0)
        daily_series = (0.5 * target_weights.diff().abs().sum(axis=1)).iloc[1:]
        daily_series = daily_series.replace([np.inf, -np.inf], np.nan).dropna()

        if len(daily_series) == 0:
            return TurnoverResult(0.0, 0.0, 0.0, pd.Series(dtype=float))

        daily_turnover = float(daily_series.mean())
        monthly_turnover = daily_turnover * 21  # 21交易日/月
        annual_turnover = daily_turnover * self.annualization

        return TurnoverResult(
            daily_turnover=daily_turnover,
            monthly_turnover=monthly_turnover,
            annual_turnover=annual_turnover,
            turnover_series=daily_series,
        )
