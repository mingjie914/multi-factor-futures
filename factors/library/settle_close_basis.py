from __future__ import annotations

import numpy as np
import pandas as pd

from core.interfaces import Factor
from core.registry import register_factor


@register_factor("settle_close_basis_20d", category="basis")
class SettleCloseBasis20D(Factor):
    name = "settle_close_basis_20d"
    category = "basis"
    frequency = "daily"
    description = "过去 20 日平均 (结算价 - 收盘价) / 收盘价, 反映多空博弈力量"

    def dependencies(self) -> list:
        return ["settle", "close"]

    def compute(self, data, dates, universe):
        settle = data.get("settle", dates, universe)
        close = data.get("close", dates, universe)
        if settle.empty or close.empty:
            return pd.DataFrame(index=dates, columns=universe)
        # 结算价是当日成交量的加权平均价, 收盘价是最后成交价
        # 结算价 > 收盘价 → 多头在日内占优 (尾盘被空压)
        # 结算价 < 收盘价 → 空头在日内占优 (尾盘拉升)
        basis = (settle - close) / close
        return basis.rolling(20).mean()
