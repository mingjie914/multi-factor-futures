from __future__ import annotations

import numpy as np

from core.interfaces import PositionSizer
from core.registry import register


@register("position_sizer", "fixed_fraction")
class FixedFractionSizer(PositionSizer):
    def __init__(self, fraction: float = 0.1):
        self.fraction = fraction

    def size(
        self,
        target_weight,
        price,
        account_value,
        ticker,
        point_value,
    ):
        if price <= 0 or point_value <= 0:
            return 0
        return int(
            account_value * self.fraction * abs(target_weight)
            / (price * point_value)
        )


@register("position_sizer", "atr_based")
class ATRBasedSizer(PositionSizer):
    def __init__(self, target_volatility: float = 0.15, atr_period: int = 20):
        self.target_vol = target_volatility
        self.atr_period = atr_period

    def size(
        self,
        target_weight,
        price,
        account_value,
        ticker,
        point_value,
        atr=0.01,
    ):
        contract_vol = price * atr * point_value
        if contract_vol <= 0:
            return 0
        target_risk = account_value * self.target_vol / np.sqrt(252)
        contracts = int(target_risk / contract_vol)
        return max(1, contracts) if target_weight != 0 else 0


@register("position_sizer", "volatility_inverse")
class VolatilityInverseSizer(PositionSizer):
    def __init__(self, target_vol: float = 0.15):
        self.target_vol = target_vol

    def size(
        self,
        target_weight,
        price,
        account_value,
        ticker,
        point_value,
        vol=0.2,
    ):
        if price <= 0 or vol <= 0:
            return 0
        risk_per_unit = price * vol * point_value
        target_risk = account_value * self.target_vol / np.sqrt(252)
        contracts = int(target_risk / risk_per_unit)
        return max(1, contracts) if target_weight != 0 else 0
