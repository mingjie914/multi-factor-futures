from __future__ import annotations

import numpy as np
import pandas as pd

from core.interfaces import Factor
from core.registry import register_factor


@register_factor("volatility_60d_realized", category="volatility")
class VolatilityRealized(Factor):
    name = "volatility_60d_realized"
    category = "volatility"
    frequency = "daily"
    description = "过去 60 日已实现波动率 (年化)"

    def dependencies(self) -> list:
        return ["close"]

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        if close.empty:
            return pd.DataFrame(index=dates, columns=universe)
        ret = close.pct_change()
        return ret.rolling(60).std() * np.sqrt(252)
