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
        margin_rate: float = 0.12,
        annual_transaction_cost: float = 0.0002,
        annual_fee: float = 0.0,
        annual_roll_cost: float = 0.00105,
        periods_per_year: float = 252.0,
        cost_stage: str = "post_screen_backtest",
    ):
        numeric_rates = {
            "margin_rate": margin_rate,
            "annual_transaction_cost": annual_transaction_cost,
            "annual_fee": annual_fee,
        }
        if any(not np.isfinite(float(value)) or float(value) < 0.0 for value in numeric_rates.values()):
            raise ValueError("futures cost rates must be finite and non-negative")
        if not np.isfinite(float(periods_per_year)) or float(periods_per_year) <= 0.0:
            raise ValueError("periods_per_year must be finite and positive")
        self.margin_rate = float(margin_rate)
        self.annual_transaction_cost = float(annual_transaction_cost)
        self.annual_fee = float(annual_fee)
        self.annual_roll_cost = float(annual_roll_cost)
        if (
            not np.isfinite(self.annual_roll_cost)
            or self.annual_roll_cost < 0.0
        ):
            raise ValueError("annual_roll_cost must be finite and non-negative")
        self.periods_per_year = float(periods_per_year)
        self.cost_stage = str(cost_stage)

    def estimate_cost(
        self, target: WeightVector, current: WeightVector, date: Date
    ) -> float:
        # Futures turnover is diagnostic only under the current research
        # policy. Costs accrue through estimate_holding_cost instead.
        del target, current, date
        return 0.0

    def estimate_holding_cost(
        self, weights: WeightVector, date: Date
    ) -> float:
        del date
        gross_exposure = float(weights.abs().sum())
        transaction_cost = gross_exposure * self.annual_transaction_cost
        roll_cost = gross_exposure * self.annual_roll_cost
        return (
            self.annual_fee + transaction_cost + roll_cost
        ) / self.periods_per_year

def factor_cost_coverage(
    *,
    gross_annual_alpha: float,
    annual_half_turnover: float,
    annual_roll_cost: float = 0.00105,
    annual_fee: float = 0.0,
    safety_margin: float = 1.5,
    annual_transaction_cost: Optional[float] = None,
    include_roll_cost: bool = True,
    cost_stage: str = "formal_validation",
) -> dict:
    """Compare gross long-short alpha with fully declared annual costs.

    The current validation policy declares one exposure-scaled annual
    transaction-cost rate. Turnover remains in the audit output but never
    multiplies the cost. Omitting the annual rate uses the governed 0.02%
    default.

    Roll cost is explicitly deferred from validation to the post-screen
    backtest unless ``include_roll_cost`` is requested.
    """
    trading_cost = max(
        float(
            0.0002
            if annual_transaction_cost is None
            else annual_transaction_cost
        ),
        0.0,
    )
    transaction_cost_mode = "fixed_annual_exposure_rate"
    roll_cost = max(float(annual_roll_cost), 0.0) if include_roll_cost else 0.0
    complete = True
    total_cost = trading_cost + roll_cost + max(float(annual_fee), 0.0)
    required_alpha = float(safety_margin) * total_cost
    return {
        "gross_annual_alpha": float(gross_annual_alpha),
        "annual_half_turnover": float(annual_half_turnover),
        "annual_trading_cost": trading_cost,
        "annual_transaction_cost": (
            float(annual_transaction_cost)
            if annual_transaction_cost is not None else None
        ),
        "transaction_cost_mode": transaction_cost_mode,
        "annual_roll_cost": float(annual_roll_cost),
        "annual_roll_cost_charged": float(roll_cost),
        "annual_fee": float(annual_fee),
        "total_annual_cost": total_cost,
        "safety_margin": float(safety_margin),
        "required_gross_alpha": required_alpha,
        "include_roll_cost": bool(include_roll_cost),
        "cost_stage": str(cost_stage),
        "complete": complete,
        "passes": bool(complete and float(gross_annual_alpha) >= required_alpha),
        "observation_reason": "",
    }
