from __future__ import annotations

import numpy as np
import pandas as pd

from core.interfaces import Factor
from core.registry import register_factor


@register_factor("intraday_range_20d", category="volatility")
class IntradayRange20D(Factor):
    name = "intraday_range_20d"
    category = "volatility"
    frequency = "daily"
    description = "过去 20 日平均日内振幅 (high-low)/close, 衡量市场分歧度"

    def dependencies(self) -> list:
        return ["high", "low", "close"]

    def compute(self, data, dates, universe):
        high = data.get("high", dates, universe)
        low = data.get("low", dates, universe)
        close = data.get("close", dates, universe)
        if high.empty or low.empty or close.empty:
            return pd.DataFrame(index=dates, columns=universe)
        # 日内振幅 = (最高价 - 最低价) / 收盘价
        range_pct = (high - low) / close
        return range_pct.rolling(20).mean()
