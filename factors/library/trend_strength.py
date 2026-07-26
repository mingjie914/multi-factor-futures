from __future__ import annotations

import numpy as np
import pandas as pd

from core.interfaces import Factor
from core.registry import register_factor


@register_factor("trend_strength_20d", category="momentum")
class TrendStrength20D(Factor):
    name = "trend_strength_20d"
    category = "momentum"
    frequency = "daily"
    description = "趋势强度因子 (日内位移/路程比), 刻画趋势连贯性 (中信期货专题五)"

    def dependencies(self) -> list:
        return ["close", "high", "low"]

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        high = data.get("high", dates, universe)
        low = data.get("low", dates, universe)
        if close.empty or high.empty or low.empty:
            return pd.DataFrame(index=dates, columns=universe)

        # 趋势强度 = |收盘价_t - 收盘价_{t-J}| / Σ|日内振幅|
        # 位移 = close.shift(J) 到 close 的净变化 (方向性)
        # 路程 = 每日 (high - low) 的累计总和 (总波动)
        # 比值越接近1 → 趋势越连贯 (单边行情)
        # 比值越接近0 → 震荡行情 (来回波动但没有方向)
        displacement = (close - close.shift(20)).abs()
        daily_range = (high - low).abs()
        path = daily_range.rolling(20).sum()

        return displacement / path
