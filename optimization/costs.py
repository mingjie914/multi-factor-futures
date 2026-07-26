from __future__ import annotations

from typing import Optional

import numpy as np  # noqa: F401

from core.types import WeightVector, Date
from core.interfaces import CostModel


def marginal_turnover_cost_rate(
    cost_model: Optional[CostModel],
    universe,
    date: Date,
) -> float:
    """Estimate the cost of one unit of turnover in return units."""
    if cost_model is None or len(universe) == 0:
        return 0.0
    probe_target = WeightVector(0.0, index=universe)
    probe_target.iloc[0] = 1.0
    probe_current = WeightVector(0.0, index=universe)
    rate = float(cost_model.estimate_cost(probe_target, probe_current, date))
    if not np.isfinite(rate) or rate < 0.0:
        raise ValueError(f"invalid marginal turnover cost rate: {rate}")
    return rate
from core.registry import register


@register("cost_model", "simple_a_share")
class SimpleAShareCost(CostModel):
    def __init__(
        self,
        commission: float = 0.0003,
        stamp_duty_sell: float = 0.001,
        slippage: float = 0.001,
    ):
        self.commission = commission
        self.stamp_duty_sell = stamp_duty_sell
        self.slippage = slippage

    def estimate_cost(
        self, target: WeightVector, current: WeightVector, date: Date
    ) -> float:
        turnover = (target - current.reindex(target.index).fillna(0)).abs().sum()
        return (
            turnover * (self.commission + self.slippage)
            + turnover * self.stamp_duty_sell * 0.5
        )


@register("cost_model", "simple_futures")
class SimpleFuturesCost(CostModel):
    def __init__(
        self,
        commission_rate: float = 0.0001,
        slippage: float = 0.001,
        margin_rate: float = 0.12,
        annual_fee: float = 0.0,
    ):
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.margin_rate = margin_rate
        self.annual_fee = float(annual_fee)

    def estimate_cost(
        self, target: WeightVector, current: WeightVector, date: Date
    ) -> float:
        turnover = (target - current.reindex(target.index).fillna(0)).abs().sum()
        return turnover * (self.commission_rate + self.slippage)

    def estimate_holding_cost(
        self, weights: WeightVector, date: Date
    ) -> float:
        del weights, date
        return self.annual_fee / 252.0

    def estimate_per_trade(
        self,
        ticker: str,
        side: str,
        contracts: int,
        price: float,
        date: Date,
    ) -> float:
        return abs(contracts) * price * (self.commission_rate + self.slippage)
