from __future__ import annotations

import numpy as np
import pandas as pd

from core.interfaces import Factor
from core.registry import register_factor


@register_factor("momentum_20d", category="momentum")
class Momentum20D(Factor):
    name = "momentum_20d"
    category = "momentum"
    frequency = "daily"
    description = "过去 20 日 (跳过最近 1 天) 收益率"

    def dependencies(self) -> list:
        return ["close"]

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        if close.empty:
            return pd.DataFrame(index=dates, columns=universe)
        return close.shift(1).pct_change(20)


@register_factor("momentum_60d_skip5", category="momentum")
class Momentum60D(Factor):
    name = "momentum_60d_skip5"
    category = "momentum"
    frequency = "daily"
    description = "过去 60 日 (跳过最近 5 天) 收益率"

    def dependencies(self) -> list:
        return ["close"]

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        if close.empty:
            return pd.DataFrame(index=dates, columns=universe)
        return close.shift(5).pct_change(60)
