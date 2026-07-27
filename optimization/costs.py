from __future__ import annotations

from typing import Mapping, Optional

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
        annual_roll_cost: Optional[float] = None,
        roll_cost_by_instrument: Optional[Mapping[str, float]] = None,
        roll_cost_source: str = "",
    ):
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.margin_rate = margin_rate
        self.annual_fee = float(annual_fee)
        self.annual_roll_cost = (
            None if annual_roll_cost is None else float(annual_roll_cost)
        )
        self.roll_cost_by_instrument = {
            str(key): float(value)
            for key, value in (roll_cost_by_instrument or {}).items()
        }
        self.roll_cost_source = str(roll_cost_source)

    def estimate_cost(
        self, target: WeightVector, current: WeightVector, date: Date
    ) -> float:
        turnover = (target - current.reindex(target.index).fillna(0)).abs().sum()
        return turnover * (self.commission_rate + self.slippage)

    def estimate_holding_cost(
        self, weights: WeightVector, date: Date
    ) -> float:
        del date
        roll_cost = 0.0
        if self.roll_cost_by_instrument:
            roll_cost = sum(
                abs(float(weight)) * self.roll_cost_by_instrument.get(str(ticker), 0.0)
                for ticker, weight in weights.items()
            )
        elif self.annual_roll_cost is not None:
            roll_cost = float(weights.abs().sum()) * self.annual_roll_cost
        return (self.annual_fee + roll_cost) / 252.0

    @property
    def has_complete_roll_cost(self) -> bool:
        return bool(
            self.roll_cost_source
            and (
                self.annual_roll_cost is not None
                or self.roll_cost_by_instrument
            )
        )

    def estimate_per_trade(
        self,
        ticker: str,
        side: str,
        contracts: int,
        price: float,
        date: Date,
    ) -> float:
        return abs(contracts) * price * (self.commission_rate + self.slippage)


def factor_cost_coverage(
    *,
    gross_annual_alpha: float,
    annual_half_turnover: float,
    one_way_cost_rate: float,
    annual_roll_cost: Optional[float],
    roll_cost_by_instrument: Optional[Mapping[str, float]] = None,
    mean_absolute_weights: Optional[Mapping[str, float]] = None,
    annual_fee: float = 0.0,
    safety_margin: float = 1.5,
    roll_cost_source: str = "",
) -> dict:
    """Compare gross long-short alpha with fully declared annual costs.

    ``annual_half_turnover`` follows ``0.5 * sum(abs(delta_weight))``;
    multiplying by two therefore converts it to one-way traded notional.
    Missing roll data fails closed into the observation channel.
    """
    trading_cost = (
        max(float(annual_half_turnover), 0.0)
        * max(float(one_way_cost_rate), 0.0)
        * 2.0
    )
    declared_roll_costs = {
        str(key): max(float(value), 0.0)
        for key, value in (roll_cost_by_instrument or {}).items()
    }
    active_weights = {
        str(key): max(float(value), 0.0)
        for key, value in (mean_absolute_weights or {}).items()
        if float(value) > 0.0
    }
    missing_roll_instruments: list[str] = []
    if annual_roll_cost is not None:
        roll_cost = max(float(annual_roll_cost), 0.0)
        complete = bool(str(roll_cost_source))
        roll_cost_mode = "aggregate_annual_rate"
    elif declared_roll_costs and active_weights:
        missing_roll_instruments = sorted(set(active_weights) - set(declared_roll_costs))
        roll_cost = sum(
            weight * declared_roll_costs.get(instrument, 0.0)
            for instrument, weight in active_weights.items()
        )
        complete = bool(str(roll_cost_source)) and not missing_roll_instruments
        roll_cost_mode = "instrument_weighted_annual_rates"
    else:
        roll_cost = 0.0
        complete = False
        roll_cost_mode = "missing"
    total_cost = trading_cost + roll_cost + max(float(annual_fee), 0.0)
    required_alpha = float(safety_margin) * total_cost
    return {
        "gross_annual_alpha": float(gross_annual_alpha),
        "annual_half_turnover": float(annual_half_turnover),
        "one_way_cost_rate": float(one_way_cost_rate),
        "annual_trading_cost": trading_cost,
        "annual_roll_cost": (float(annual_roll_cost) if annual_roll_cost is not None else None),
        "estimated_weighted_roll_cost": float(roll_cost),
        "roll_cost_mode": roll_cost_mode,
        "missing_roll_instruments": missing_roll_instruments,
        "annual_fee": float(annual_fee),
        "total_annual_cost": total_cost,
        "safety_margin": float(safety_margin),
        "required_gross_alpha": required_alpha,
        "roll_cost_source": str(roll_cost_source),
        "complete": complete,
        "passes": bool(complete and float(gross_annual_alpha) >= required_alpha),
        "observation_reason": (
            "" if complete
            else "incomplete_realized_roll_cost_ledger"
            if missing_roll_instruments
            else "missing_realized_roll_cost_ledger"
        ),
    }
