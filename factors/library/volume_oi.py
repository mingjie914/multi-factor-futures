from __future__ import annotations

import numpy as np
import pandas as pd

from core.interfaces import Factor
from core.registry import register_factor


@register_factor("oi_change_20d", category="volume_oi")
class OIChange20D(Factor):
    name = "oi_change_20d"
    category = "volume_oi"
    frequency = "daily"
    description = "过去 20 日持仓量变化率"

    def dependencies(self) -> list:
        return ["oi"]

    def compute(self, data, dates, universe):
        oi = data.get("oi", dates, universe)
        if oi.empty:
            return pd.DataFrame(index=dates, columns=universe)
        return oi.pct_change(20)


@register_factor("volume_change_20d", category="volume_oi")
class VolumeChange20D(Factor):
    name = "volume_change_20d"
    category = "volume_oi"
    frequency = "daily"
    description = "过去 20 日成交量变化率"

    def dependencies(self) -> list:
        return ["volume"]

    def compute(self, data, dates, universe):
        vol = data.get("volume", dates, universe)
        if vol.empty:
            return pd.DataFrame(index=dates, columns=universe)
        return vol.pct_change(20)
