from __future__ import annotations

import numpy as np
import pandas as pd

from core.interfaces import Factor
from core.registry import register_factor


@register_factor("oi_momentum_20d", category="volume_oi")
class OIMomentum20D(Factor):
    name = "oi_momentum_20d"
    category = "volume_oi"
    frequency = "daily"
    description = "持仓量相对20日均值的偏离度 (资金持续流入/流出趋势)"

    def dependencies(self) -> list:
        return ["oi"]

    def compute(self, data, dates, universe):
        oi = data.get("oi", dates, universe)
        if oi.empty:
            return pd.DataFrame(index=dates, columns=universe)
        # 偏离度 = (当前OI - 20日均值) / 20日均值
        # 与 oi_change_20d (pct_change) 不同: 用均值平滑, 捕捉持续趋势而非单点跳变
        ma20 = oi.rolling(20).mean()
        return (oi - ma20) / ma20
