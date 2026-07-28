from __future__ import annotations

import numpy as np
import pandas as pd

from core.interfaces import Factor
from core.registry import register_factor


@register_factor("liquidity_amihud_20d", category="liquidity")
class AmihudIlliquidity(Factor):
    name = "liquidity_amihud_20d"
    category = "liquidity"
    frequency = "daily"
    description = "Amihud 非流动性指标 (过去 20 日平均 |ret|/volume)"

    def dependencies(self) -> list:
        return ["close", "volume"]

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        volume = data.get("volume", dates, universe)
        if close.empty or volume.empty:
            return pd.DataFrame(index=dates, columns=universe)
        ret = close.pct_change(fill_method=None).abs()
        illiq = ret / (volume + 1e-8)
        return illiq.rolling(20).mean()
