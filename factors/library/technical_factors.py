"""日度技术面因子 (动量/反转/偏度/趋势强度/波动率).

合并自早期逐主题小文件, 保持注册名与 category 不变.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.interfaces import Factor
from core.registry import register_factor


@register_factor("momentum_20d", category="momentum")
class Momentum20D(Factor):
    name = "momentum_20d"
    category = "momentum"
    frequency = "daily"
    description = "过去 20 日 (跳过最近 1 天) 收益率"

    def dependencies(self) -> list:
        return ["close"]

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        if close.empty:
            return pd.DataFrame(index=dates, columns=universe)
        return close.shift(1).pct_change(20, fill_method=None)


@register_factor("momentum_60d_skip5", category="momentum")
class Momentum60D(Factor):
    name = "momentum_60d_skip5"
    category = "momentum"
    frequency = "daily"
    description = "过去 60 日 (跳过最近 5 天) 收益率"

    def dependencies(self) -> list:
        return ["close"]

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        if close.empty:
            return pd.DataFrame(index=dates, columns=universe)
        return close.shift(5).pct_change(60, fill_method=None)

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
        return close.shift(1).pct_change(5, fill_method=None)


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
        return close.shift(1).pct_change(10, fill_method=None)


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
        return -close.pct_change(2, fill_method=None)


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
        ret = close.pct_change(fill_method=None)
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
        return oi.pct_change(5, fill_method=None)


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
        ret_abs = close.pct_change(fill_method=None).abs()
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

@register_factor("trend_strength_20d", category="momentum")
class TrendStrength20D(Factor):
    name = "trend_strength_20d"
    category = "momentum"
    frequency = "daily"
    description = "趋势强度因子 (日内位移/路程比), 刻画趋势连贯性 (中信期货专题五)"

    def dependencies(self) -> list:
        return ["close", "high", "low"]

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        high = data.get("high", dates, universe)
        low = data.get("low", dates, universe)
        if close.empty or high.empty or low.empty:
            return pd.DataFrame(index=dates, columns=universe)

        # 趋势强度 = |收盘价_t - 收盘价_{t-J}| / Σ|日内振幅|
        # 位移 = close.shift(J) 到 close 的净变化 (方向性)
        # 路程 = 每日 (high - low) 的累计总和 (总波动)
        # 比值越接近1 → 趋势越连贯 (单边行情)
        # 比值越接近0 → 震荡行情 (来回波动但没有方向)
        displacement = (close - close.shift(20)).abs()
        daily_range = (high - low).abs()
        path = daily_range.rolling(20).sum()

        return displacement / path

@register_factor("skewness_20d", category="skewness")
class Skewness20D(Factor):
    name = "skewness_20d"
    category = "skewness"
    frequency = "daily"
    description = "过去 20 日收益率偏度 (负偏度品种有 crash risk premium)"

    def dependencies(self) -> list:
        return ["close"]

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        if close.empty:
            return pd.DataFrame(index=dates, columns=universe)
        ret = close.pct_change(fill_method=None)
        return ret.rolling(20).skew()

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

@register_factor("volatility_60d_realized", category="volatility")
class VolatilityRealized(Factor):
    name = "volatility_60d_realized"
    category = "volatility"
    frequency = "daily"
    description = "过去 60 日已实现波动率 (年化)"

    def dependencies(self) -> list:
        return ["close"]

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        if close.empty:
            return pd.DataFrame(index=dates, columns=universe)
        ret = close.pct_change(fill_method=None)
        return ret.rolling(60).std() * np.sqrt(252)

