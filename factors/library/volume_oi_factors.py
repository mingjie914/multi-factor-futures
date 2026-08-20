"""量价与持仓因子 (成交/持仓/流动性/隔夜/日内振幅).

合并自早期逐主题小文件, 保持注册名与 category 不变.
"""
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
        return oi.pct_change(20, fill_method=None)


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
        return vol.pct_change(20, fill_method=None)

@register_factor("volume_oi_ratio_20d", category="volume_oi")
class VolumeOIRatio20D(Factor):
    """成交量/持仓量比因子 (20日均值).

    计算逻辑:
        1. ratio = volume / oi
        2. 对原始 ratio 做 20 日窗口均值平滑

    含义:
        - 衡量交易活跃度相对于持仓规模:
            高值 = 短线交易活跃 (日内换手率高, 投机资金主导)
            低值 = 长线持仓主导 (持仓稳定, 日内交易清淡)
        - 与 volume_change_20d / oi_change_20d 的区别:
            本因子刻画量仓的相对水平而非变化率,
            能识别品种的交易属性 (短线品种 vs 长线品种)

    注意:
        - 除零保护: oi <= 0 时 ratio 置为 NaN
        - 持仓量为 0 通常出现在新合约上市初期或合约退市前
    """

    name = "volume_oi_ratio_20d"
    category = "volume_oi"
    frequency = "daily"
    description = "成交量/持仓量比 (20日均值, 衡量交易活跃度)"

    # 滚动窗口大小
    ROLLING_WINDOW = 20

    def dependencies(self) -> list:
        return ["volume", "oi"]

    def compute(self, data, dates, universe):
        vol = data.get("volume", dates, universe)
        oi = data.get("oi", dates, universe)

        if vol.empty or oi.empty:
            return pd.DataFrame(index=dates, columns=universe)

        # 对齐到 dates 和 universe
        vol = vol.reindex(index=dates, columns=universe)
        oi = oi.reindex(index=dates, columns=universe)

        # 计算成交量/持仓量比
        # 避免除零: oi <= 0 时置为 NaN (新合约上市或合约退市可能出现持仓量为 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = vol / oi
        ratio = ratio.where(oi > 0, np.nan)

        # 20 日窗口均值平滑 (最小周期 10, 避免窗口初期全 NaN)
        return ratio.rolling(self.ROLLING_WINDOW, min_periods=10).mean()


@register_factor("oi_change_rate_5d", category="volume_oi")
class OIChangeRate5D(Factor):
    """持仓量5日变化率因子.

    计算逻辑:
        1. change_rate = oi.pct_change(5, fill_method=None)
        2. 返回最近 5 个交易日的持仓量变化率

    含义:
        - 反映短期资金流入流出:
            正值 = 持仓量增加, 资金流入 (多空分歧加大或新趋势启动)
            负值 = 持仓量减少, 资金流出 (趋势结束或仓位平仓)
        - 与 oi_change_20d 的区别:
            本因子使用 5 日窗口, 捕捉短期资金动向,
            20 日版本刻画中期持仓趋势
        - 5 日窗口对突发事件 (如政策、消息) 更敏感

    注意:
        - pct_change 对首日及前 4 日返回 NaN (无历史可比)
        - 持仓量为 0 的合约 (新上市/退市) 计算结果可能产生 inf,
          已通过 replace(inf, NaN) 处理, 避免后续计算污染
    """

    name = "oi_change_rate_5d"
    category = "volume_oi"
    frequency = "daily"
    description = "持仓量5日变化率 (短期资金流入流出)"

    # 变化率窗口大小
    WINDOW = 5

    def dependencies(self) -> list:
        return ["oi"]

    def compute(self, data, dates, universe):
        oi = data.get("oi", dates, universe)

        if oi.empty:
            return pd.DataFrame(index=dates, columns=universe)

        # 对齐到 dates 和 universe
        oi = oi.reindex(index=dates, columns=universe)

        # 计算 5 日持仓量变化率
        # pct_change(5) = (oi_t - oi_{t-5}) / oi_{t-5}
        # 前置 5 日无历史数据, 返回 NaN
        # 除零保护: oi_{t-5}=0 时产生 inf/-inf, 替换为 NaN
        with np.errstate(divide="ignore", invalid="ignore"):
            change_rate = oi.pct_change(self.WINDOW, fill_method=None)
        # 替换 inf 为 NaN (新合约上市首日 oi 从 0 跳到正值会产生 inf)
        change_rate = change_rate.replace([np.inf, -np.inf], np.nan)
        return change_rate

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
        ret_abs = close.pct_change(fill_method=None).abs()
        # 量价同向 (正相关): 放量上涨/下跌, 趋势确认
        # 量价背离 (负相关): 放量但价格不动, 可能反转
        return ret_abs.rolling(20).corr(volume)

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

@register_factor("liquidity_amihud_20d", category="liquidity")
class AmihudIlliquidity(Factor):
    name = "liquidity_amihud_20d"
    category = "liquidity"
    frequency = "daily"
    description = "Amihud 非流动性指标 (过去 20 日平均 |ret|/volume)"

    def dependencies(self) -> list:
        return ["close", "volume"]

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        volume = data.get("volume", dates, universe)
        if close.empty or volume.empty:
            return pd.DataFrame(index=dates, columns=universe)
        ret = close.pct_change(fill_method=None).abs()
        illiq = ret / (volume + 1e-8)
        return illiq.rolling(20).mean()

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

