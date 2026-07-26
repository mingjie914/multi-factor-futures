from __future__ import annotations

import numpy as np
import pandas as pd

from core.interfaces import Factor
from core.registry import register_factor


@register_factor("skewness_20d", category="skewness")
class Skewness20D(Factor):
    name = "skewness_20d"
    category = "skewness"
    frequency = "daily"
    description = "过去 20 日收益率偏度 (负偏度品种有 crash risk premium)"

    def dependencies(self) -> list:
        return ["close"]

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        if close.empty:
            return pd.DataFrame(index=dates, columns=universe)
        ret = close.pct_change()
        return ret.rolling(20).skew()
