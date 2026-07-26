from __future__ import annotations

import numpy as np
import pandas as pd

from core.interfaces import Factor
from core.registry import register_factor


@register_factor("vstd_normalized_20d", category="volume_price")
class VSTDNormalized20D(Factor):
    name = "vstd_normalized_20d"
    category = "volume_price"
    frequency = "daily"
    description = "归一化成交量标准差 (20日 volume std / volume mean), 负向因子"

    def dependencies(self) -> list:
        return ["volume"]

    def compute(self, data, dates, universe):
        vol = data.get("volume", dates, universe)
        if vol.empty:
            return pd.DataFrame(index=dates, columns=universe)
        # 归一化VSTD = std(volume) / mean(volume)
        # 衡量成交量波动剧烈程度, 值越大说明成交量忽高忽低 → 交易不稳定 → 负向因子
        std = vol.rolling(20).std()
        mean = vol.rolling(20).mean()
        return std / mean
