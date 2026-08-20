from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from core.interfaces import CostModel
from core.registry import register
from core.types import Date, WeightVector


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


@register("cost_model", "simple_a_share")
class SimpleAShareCost(CostModel):
    def __init__(
        self,
        commission: float = 0.0003,
        stamp_duty_sell: float = 0.001,
        slippage: float = 0.001,
    ):
        values = np.asarray([commission, stamp_duty_sell, slippage], dtype=float)
        if not np.isfinite(values).all() or bool((values < 0.0).any()):
            raise ValueError("A-share cost rates must be finite and non-negative")
        self.commission = float(commission)
        self.stamp_duty_sell = float(stamp_duty_sell)
        self.slippage = float(slippage)

    def estimate_cost(
        self, target: WeightVector, current: WeightVector, date: Date
    ) -> float:
        del date
        assets = pd.Index(target.index).union(pd.Index(current.index), sort=False)
        target_values = pd.Series(target, dtype=float)
        current_values = pd.Series(current, dtype=float)
        if (
            not np.isfinite(target_values.to_numpy(dtype=float)).all()
            or not np.isfinite(current_values.to_numpy(dtype=float)).all()
        ):
            raise ValueError("A-share weights must be finite")
        target_aligned = target_values.reindex(assets).fillna(0.0)
        current_aligned = current_values.reindex(assets).fillna(0.0)
        change = target_aligned - current_aligned
        turnover = float(change.abs().sum())
        sell_turnover = float((-change).clip(lower=0.0).sum())
        return (
            turnover * (self.commission + self.slippage)
            + sell_turnover * self.stamp_duty_sell
        )


@register("cost_model", "simple_futures")
class SimpleFuturesCost(CostModel):
    def __init__(
        self,
        turnover_cost_rate: float = 0.0002,
        annual_fee: float = 0.0,
        annual_roll_cost: float = 0.00105,
        periods_per_year: float = 252.0,
        cost_stage: str = "post_screen_backtest",
    ):
        numeric_rates = {
            "turnover_cost_rate": turnover_cost_rate,
            "annual_fee": annual_fee,
        }
        if any(not np.isfinite(float(value)) or float(value) < 0.0 for value in numeric_rates.values()):
            raise ValueError("futures cost rates must be finite and non-negative")
        if not np.isfinite(float(periods_per_year)) or float(periods_per_year) <= 0.0:
            raise ValueError("periods_per_year must be finite and positive")
        self.turnover_cost_rate = float(turnover_cost_rate)
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
        del date
        assets = pd.Index(target.index).union(pd.Index(current.index), sort=False)
        target_values = pd.Series(target, dtype=float)
        current_values = pd.Series(current, dtype=float)
        if (
            not np.isfinite(target_values.to_numpy(dtype=float)).all()
            or not np.isfinite(current_values.to_numpy(dtype=float)).all()
        ):
            raise ValueError("futures weights must be finite")
        target_aligned = target_values.reindex(assets).fillna(0.0)
        current_aligned = current_values.reindex(assets).fillna(0.0)
        traded_notional = float((target_aligned - current_aligned).abs().sum())
        return traded_notional * self.turnover_cost_rate

    def estimate_holding_cost(
        self, weights: WeightVector, date: Date
    ) -> float:
        del date
        weights = pd.Series(weights, dtype=float)
        if not np.isfinite(weights.to_numpy(dtype=float)).all():
            raise ValueError("futures weights must be finite")
        gross_exposure = float(weights.abs().sum())
        roll_cost = gross_exposure * self.annual_roll_cost
        return (self.annual_fee + roll_cost) / self.periods_per_year

    def ledger_parameters(self) -> dict[str, float]:
        """Return the exact parameters consumed by the shared research ledger."""
        return {
            "trade_cost_rate": self.turnover_cost_rate,
            "annual_fee": self.annual_fee,
            "annual_roll_cost": self.annual_roll_cost,
            "periods_per_year": self.periods_per_year,
        }

def factor_cost_coverage(
    *,
    gross_annual_alpha: float,
    annual_half_turnover: float,
    annual_roll_cost: float = 0.00105,
    annual_fee: float = 0.0,
    safety_margin: float = 1.5,
    turnover_cost_rate: Optional[float] = None,
    include_roll_cost: bool = True,
    cost_stage: str = "formal_validation",
) -> dict:
    """Compare gross long-short alpha with fully declared annual costs.

    Trading cost is full traded notional times the declared per-unit rate.
    ``annual_half_turnover`` is converted back to full traded notional by
    multiplying by two. Omitting the rate uses the governed 0.02% default.

    Roll cost is explicitly deferred from validation to the post-screen
    backtest unless ``include_roll_cost`` is requested.
    """
    half_turnover = float(annual_half_turnover)
    gross_alpha = float(gross_annual_alpha)
    rate = float(0.0002 if turnover_cost_rate is None else turnover_cost_rate)
    roll_rate = float(annual_roll_cost)
    fee = float(annual_fee)
    if (
        not np.isfinite(
            [gross_alpha, half_turnover, rate, roll_rate, fee, safety_margin]
        ).all()
        or min(half_turnover, rate, roll_rate, fee, float(safety_margin)) < 0.0
    ):
        raise ValueError("factor cost inputs must be finite and non-negative")
    trading_cost = 2.0 * half_turnover * rate
    transaction_cost_mode = "full_traded_notional_times_rate"
    roll_cost = roll_rate if include_roll_cost else 0.0
    complete = True
    total_cost = trading_cost + roll_cost + fee
    required_alpha = float(safety_margin) * total_cost
    return {
        "gross_annual_alpha": gross_alpha,
        "annual_half_turnover": half_turnover,
        "annual_trading_cost": trading_cost,
        "turnover_cost_rate": rate,
        "transaction_cost_mode": transaction_cost_mode,
        "annual_roll_cost": roll_rate,
        "annual_roll_cost_charged": float(roll_cost),
        "annual_fee": fee,
        "total_annual_cost": total_cost,
        "safety_margin": float(safety_margin),
        "required_gross_alpha": required_alpha,
        "include_roll_cost": bool(include_roll_cost),
        "cost_stage": str(cost_stage),
        "complete": complete,
        "passes": bool(complete and gross_alpha >= required_alpha),
        "observation_reason": "",
    }
