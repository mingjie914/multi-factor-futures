"""短周期因子 (2d-10d), 适应快速变化的市场环境."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.interfaces import Factor
from core.registry import register_factor


@register_factor("momentum_5d", category="momentum")
class Momentum5D(Factor):
    name = "momentum_5d"
    category = "momentum"
    frequency = "daily"
    description = "过去 5 日 (跳过最近 1 天) 收益率, 捕捉短期趋势"

    def dependencies(self) -> list:
        return ["close"]

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        if close.empty:
            return pd.DataFrame(index=dates, columns=universe)
        return close.shift(1).pct_change(5)


@register_factor("momentum_10d", category="momentum")
class Momentum10D(Factor):
    name = "momentum_10d"
    category = "momentum"
    frequency = "daily"
    description = "过去 10 日 (跳过最近 1 天) 收益率, 中短期趋势"

    def dependencies(self) -> list:
        return ["close"]

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        if close.empty:
            return pd.DataFrame(index=dates, columns=universe)
        return close.shift(1).pct_change(10)


@register_factor("reversal_2d", category="reversal")
class Reversal2D(Factor):
    name = "reversal_2d"
    category = "reversal"
    frequency = "daily"
    description = "过去 2 日反转 (超短周期反转), 捕捉隔夜过度反应"

    def dependencies(self) -> list:
        return ["close"]

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        if close.empty:
            return pd.DataFrame(index=dates, columns=universe)
        return -close.pct_change(2)


@register_factor("volatility_10d", category="volatility")
class Volatility10D(Factor):
    name = "volatility_10d"
    category = "volatility"
    frequency = "daily"
    description = "过去 10 日已实现波动率 (年化), 短期风险状态"

    def dependencies(self) -> list:
        return ["close"]

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        if close.empty:
            return pd.DataFrame(index=dates, columns=universe)
        ret = close.pct_change()
        return ret.rolling(10).std() * np.sqrt(252)


@register_factor("oi_change_5d", category="volume_oi")
class OIChange5D(Factor):
    name = "oi_change_5d"
    category = "volume_oi"
    frequency = "daily"
    description = "过去 5 日持仓量变化率, 短期资金流向"

    def dependencies(self) -> list:
        return ["oi"]

    def compute(self, data, dates, universe):
        oi = data.get("oi", dates, universe)
        if oi.empty:
            return pd.DataFrame(index=dates, columns=universe)
        return oi.pct_change(5)


@register_factor("intraday_range_5d", category="volatility")
class IntradayRange5D(Factor):
    name = "intraday_range_5d"
    category = "volatility"
    frequency = "daily"
    description = "过去 5 日平均日内振幅, 短期市场分歧度"

    def dependencies(self) -> list:
        return ["high", "low", "close"]

    def compute(self, data, dates, universe):
        high = data.get("high", dates, universe)
        low = data.get("low", dates, universe)
        close = data.get("close", dates, universe)
        if high.empty or low.empty or close.empty:
            return pd.DataFrame(index=dates, columns=universe)
        range_pct = (high - low) / close
        return range_pct.rolling(5).mean()


@register_factor("volume_price_corr_5d", category="volume_price")
class VolumePriceCorr5D(Factor):
    name = "volume_price_corr_5d"
    category = "volume_price"
    frequency = "daily"
    description = "过去 5 日量价相关性, 短期量价关系"

    def dependencies(self) -> list:
        return ["close", "volume"]

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        volume = data.get("volume", dates, universe)
        if close.empty or volume.empty:
            return pd.DataFrame(index=dates, columns=universe)
        ret_abs = close.pct_change().abs()
        return ret_abs.rolling(5).corr(volume)


@register_factor("overnight_return_1d", category="sentiment")
class OvernightReturn1D(Factor):
    name = "overnight_return_1d"
    category = "sentiment"
    frequency = "daily"
    description = "当日隔夜收益率 (open/pre_settle - 1), 最即时情绪信号"

    def dependencies(self) -> list:
        return ["open", "pre_settle"]

    def compute(self, data, dates, universe):
        opn = data.get("open", dates, universe)
        pre_settle = data.get("pre_settle", dates, universe)
        if opn.empty or pre_settle.empty:
            return pd.DataFrame(index=dates, columns=universe)
        return opn / pre_settle - 1
