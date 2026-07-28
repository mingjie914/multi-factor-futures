from __future__ import annotations

import numpy as np
import pandas as pd

from core.interfaces import Factor
from core.registry import register_factor


@register_factor("skewness_150d", category="skewness")
class Skewness150D(Factor):
    name = "skewness_150d"
    category = "skewness"
    frequency = "daily"
    description = "过去 150 日收益率偏度 (长周期, 研报推荐140-180日区间)"

    def dependencies(self) -> list:
        return ["close"]

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        if close.empty:
            return pd.DataFrame(index=dates, columns=universe)
        ret = close.pct_change(fill_method=None)
        # 研报: 偏度必须窗口期放到很长(140-180日)才能体现效果
        # Miffre(2013): 做空高偏度品种, 做多低偏度品种, 年化8.01%
        return ret.rolling(150).skew()
