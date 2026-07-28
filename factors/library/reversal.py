from __future__ import annotations

import numpy as np
import pandas as pd

from core.interfaces import Factor
from core.registry import register_factor


@register_factor("reversal_5d", category="reversal")
class Reversal5D(Factor):
    name = "reversal_5d"
    category = "reversal"
    frequency = "daily"
    description = "过去 5 日反转 (短周期反转因子)"

    def dependencies(self) -> list:
        return ["close"]

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        if close.empty:
            return pd.DataFrame(index=dates, columns=universe)
        return -close.pct_change(5, fill_method=None)  # 负号: 过去跌的多 → 预期涨 (反转)
