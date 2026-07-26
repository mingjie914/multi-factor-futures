from __future__ import annotations

import numpy as np
import pandas as pd

from core.interfaces import Factor
from core.registry import register_factor


@register_factor("overnight_return_5d", category="sentiment")
class OvernightReturn5D(Factor):
    name = "overnight_return_5d"
    category = "sentiment"
    frequency = "daily"
    description = "过去 5 日平均隔夜收益率 (open/pre_settle - 1), 反映隔夜信息冲击"

    def dependencies(self) -> list:
        return ["open", "pre_settle"]

    def compute(self, data, dates, universe):
        opn = data.get("open", dates, universe)
        pre_settle = data.get("pre_settle", dates, universe)
        if opn.empty or pre_settle.empty:
            return pd.DataFrame(index=dates, columns=universe)
        # 隔夜收益 = 开盘价 / 前结算价 - 1
        # 前结算价是上一交易日的结算价, 反映隔夜信息冲击
        overnight = opn / pre_settle - 1
        return overnight.rolling(5).mean()
