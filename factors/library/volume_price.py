from __future__ import annotations

import numpy as np
import pandas as pd

from core.interfaces import Factor
from core.registry import register_factor


@register_factor("volume_price_corr_20d", category="volume_price")
class VolumePriceCorr20D(Factor):
    name = "volume_price_corr_20d"
    category = "volume_price"
    frequency = "daily"
    description = "过去 20 日成交量与收益率绝对值的相关系数 (量价关系因子)"

    def dependencies(self) -> list:
        return ["close", "volume"]

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        volume = data.get("volume", dates, universe)
        if close.empty or volume.empty:
            return pd.DataFrame(index=dates, columns=universe)
        ret_abs = close.pct_change().abs()
        # 量价同向 (正相关): 放量上涨/下跌, 趋势确认
        # 量价背离 (负相关): 放量但价格不动, 可能反转
        return ret_abs.rolling(20).corr(volume)
