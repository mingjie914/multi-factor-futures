"""期限结构与基差因子.

合并自早期逐主题小文件, 保持注册名与 category 不变.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.interfaces import Factor
from core.registry import register_factor


@register_factor("roll_yield_20d", category="term_structure")
class RollYield(Factor):
    """展期收益率因子 (近月-远月价差 / 近月价格).

    计算逻辑:
        1. 每个交易日取近月合约和远月合约的收盘价
        2. roll_yield = (far_close - near_close) / near_close
           正值 = 远月升水 (contango), 负值 = 近月升水 (backwardation)
        3. 对原始 roll_yield 做 20 日窗口均值平滑

    近月/远月确定 (避免未来函数):
        - 按合约 Wind 代码字符串排序 (等同于到期日排序)
        - 每个交易日取该日有数据的前两个合约
        - 退市日当天若仍有数据则计入, 次日自动切换

    换月跳空处理:
        - roll_yield 是比值, 换月时因对比合约变化会产生跳空
        - 这是真实市场信息 (反映了换月时的展期成本)
        - 不对价格做复权 (我们要的是真实的价差关系)
        - 20 日窗口均值会平滑跳空的影响

    注意: 不使用 cfuturescontractmapping 主力合约表, 因为:
        1. 主力合约表只给一个合约, 无法区分近月/远月
        2. 主力合约切换规则基于成交量, 可能含未来信息
    """

    name = "roll_yield_20d"
    category = "term_structure"
    frequency = "daily"
    description = "展期收益率 (近月-远月价差/近月价格, 20日均值平滑)"

    # 窗口大小
    ROLLING_WINDOW = 20

    def dependencies(self) -> list:
        # 通过 get_contract_pair 获取近月/远月价格, 不依赖标准 get(field) 接口
        return ["close"]

    def compute(self, data, dates, universe):
        # 获取近月和远月合约的收盘价
        pair = data.get_contract_pair("close", dates, universe)
        near_close = pair.get("near", pd.DataFrame())
        far_close = pair.get("far", pd.DataFrame())

        if near_close.empty or far_close.empty:
            return pd.DataFrame(np.nan, index=dates, columns=universe)

        # 对齐到 dates 和 universe
        near_close = near_close.reindex(index=dates, columns=universe)
        far_close = far_close.reindex(index=dates, columns=universe)

        # 计算展期收益率: (远月 - 近月) / 近月
        # 正值 = 远月升水 (contango), 负值 = 近月升水 (backwardation)
        # 避免除零: near_close <= 0 时置为 NaN
        with np.errstate(divide="ignore", invalid="ignore"):
            roll_yield = (far_close - near_close) / near_close
        roll_yield = roll_yield.where(near_close > 0, np.nan)

        # 20 日窗口均值平滑 (最小周期 10, 避免窗口初期全 NaN)
        result = roll_yield.rolling(self.ROLLING_WINDOW, min_periods=10).mean()

        return result

@register_factor("term_structure_slope_20d", category="term_structure")
class TermStructureSlope(Factor):
    """期限曲线斜率因子 (结算价版, 20日均值).

    计算逻辑:
        1. 通过 get_contract_pair 取近月/远月合约的结算价 (settle)
        2. slope = (far_settle - near_settle) / near_settle
           正值 = 远月升水 (contango), 负值 = 近月升水 (backwardation)
        3. 对原始 slope 做 20 日窗口均值平滑

    与 roll_yield_20d 的区别:
        - roll_yield_20d 使用 close 价格计算
        - 本因子使用 settle 价格计算
        - 结算价由交易所按成交量加权计算, 反映当日撮合的均衡价格,
          能更稳定地刻画期限结构
    """

    name = "term_structure_slope_20d"
    category = "term_structure"
    frequency = "daily"
    description = "期限曲线斜率 (远月-近月结算价差/近月结算价, 20日均值)"

    # 滚动窗口大小
    ROLLING_WINDOW = 20

    def dependencies(self) -> list:
        # 通过 get_contract_pair 获取近月/远月结算价
        return ["settle"]

    def compute(self, data, dates, universe):
        # 获取近月和远月合约的结算价
        pair = data.get_contract_pair("settle", dates, universe)
        near = pair.get("near", pd.DataFrame())
        far = pair.get("far", pd.DataFrame())

        if near.empty or far.empty:
            return pd.DataFrame(np.nan, index=dates, columns=universe)

        # 对齐到 dates 和 universe
        near = near.reindex(index=dates, columns=universe)
        far = far.reindex(index=dates, columns=universe)

        # 计算期限曲线斜率: (远月 - 近月) / 近月
        # 正值 = 远月升水 (contango), 负值 = 近月升水 (backwardation)
        # 避免除零: near <= 0 时置为 NaN
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = (far - near) / near
        slope = slope.where(near > 0, np.nan)

        # 20 日窗口均值平滑 (最小周期 10, 避免窗口初期全 NaN)
        return slope.rolling(self.ROLLING_WINDOW, min_periods=10).mean()


@register_factor("basis_ratio_20d", category="basis")
class BasisRatio20D(Factor):
    """基差率因子 (20日均值).

    计算逻辑:
        1. basis_ratio = (settle - close) / close
        2. 对原始 basis_ratio 做 20 日窗口均值平滑

    含义:
        - 结算价 (settle) 由交易所按成交量加权计算, 收盘价 (close) 为最后成交价
        - 二者之差反映结算机制带来的价格信息:
            正值 = 结算价高于收盘价 (当日盘中偏强, 尾盘回落)
            负值 = 结算价低于收盘价 (当日盘中偏弱, 尾盘拉升)
        - 与 basis_momentum 的区别: 本因子计算比值而非差值, 并做 20 日均值平滑,
          消除日间噪声, 刻画基差率的中期水平
    """

    name = "basis_ratio_20d"
    category = "basis"
    frequency = "daily"
    description = "基差率 (settle-close)/close, 20日均值平滑"

    # 滚动窗口大小
    ROLLING_WINDOW = 20

    def dependencies(self) -> list:
        return ["close", "settle"]

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        settle = data.get("settle", dates, universe)

        if close.empty or settle.empty:
            return pd.DataFrame(np.nan, index=dates, columns=universe)

        # 对齐到 dates 和 universe
        close = close.reindex(index=dates, columns=universe)
        settle = settle.reindex(index=dates, columns=universe)

        # 计算基差率: (settle - close) / close
        # 避免除零: close <= 0 时置为 NaN
        with np.errstate(divide="ignore", invalid="ignore"):
            basis_ratio = (settle - close) / close
        basis_ratio = basis_ratio.where(close > 0, np.nan)

        # 20 日窗口均值平滑 (最小周期 10, 避免窗口初期全 NaN)
        return basis_ratio.rolling(self.ROLLING_WINDOW, min_periods=10).mean()


@register_factor("roll_yield_skew_20d", category="term_structure")
class RollYieldSkew20D(Factor):
    """展期收益偏度因子 (20日滚动偏度).

    计算逻辑:
        1. 通过 get_contract_pair 取近月/远月合约的结算价 (settle)
        2. roll_yield = (far_settle - near_settle) / near_settle
        3. 对 roll_yield 做 20 日窗口滚动偏度 (skew)

    含义:
        - 偏度 (skew) 反映分布的非对称性:
            skew > 0 = 右偏, 展期收益有正向尾部 (远月升水偶发扩大)
            skew < 0 = 左偏, 展期收益有负向尾部 (近月升水偶发扩大)
            skew ≈ 0 = 近似对称
        - 相比均值 (term_structure_slope_20d), 偏度刻画分布形态,
          能识别期限结构的尾部风险特征
    """

    name = "roll_yield_skew_20d"
    category = "term_structure"
    frequency = "daily"
    description = "展期收益偏度 ((远月-近月)/近月 的20日滚动偏度)"

    # 滚动窗口大小
    ROLLING_WINDOW = 20

    def dependencies(self) -> list:
        # 通过 get_contract_pair 获取近月/远月结算价
        return ["settle"]

    def compute(self, data, dates, universe):
        # 获取近月和远月合约的结算价
        pair = data.get_contract_pair("settle", dates, universe)
        near = pair.get("near", pd.DataFrame())
        far = pair.get("far", pd.DataFrame())

        if near.empty or far.empty:
            return pd.DataFrame(np.nan, index=dates, columns=universe)

        # 对齐到 dates 和 universe
        near = near.reindex(index=dates, columns=universe)
        far = far.reindex(index=dates, columns=universe)

        # 计算展期收益: (远月 - 近月) / 近月
        # 避免除零: near <= 0 时置为 NaN
        with np.errstate(divide="ignore", invalid="ignore"):
            roll_yield = (far - near) / near
        roll_yield = roll_yield.where(near > 0, np.nan)

        # 20 日窗口滚动偏度 (最小周期 10, 避免窗口初期全 NaN)
        # pandas rolling.skew 默认使用无偏样本偏度 (调整后的偏度)
        return roll_yield.rolling(self.ROLLING_WINDOW, min_periods=10).skew()

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

@register_factor("basis_momentum_5d", category="basis")
class BasisMomentum5d(Factor):
    """基差动量因子 (5日).

    基差 = settle - close, 动量 = 基差的 5 日变化率.
    正值 = 基差扩大 (现货走强), 负值 = 基差缩小 (现货走弱).
    """
    name = "basis_momentum_5d"
    category = "basis"
    frequency = "daily"
    description = "基差动量 (settle-close 差的5日变化率)"
    WINDOW = 5

    def dependencies(self) -> list:
        return ["close", "settle"]

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        settle = data.get("settle", dates, universe)
        basis = settle - close
        # 基差的 5 日变化 (绝对变化, 不用变化率因为基差可能接近0)
        result = basis.diff(self.WINDOW)
        # 用收盘价标准化 (消除量纲)
        result = result / close.replace(0, np.nan)
        return result


@register_factor("basis_momentum_20d", category="basis")
class BasisMomentum20d(BasisMomentum5d):
    """基差动量因子 (20日)."""
    name = "basis_momentum_20d"
    description = "基差动量 (settle-close 差的20日变化率)"
    WINDOW = 20


@register_factor("basis_zscore_20d", category="basis")
class BasisZscore20d(Factor):
    """基差 z-score 因子 (20日).

    基差在 20 日窗口内的 z-score 标准化值.
    高 z-score = 基差异常偏高, 低 z-score = 基差异常偏低.
    """
    name = "basis_zscore_20d"
    category = "basis"
    frequency = "daily"
    description = "基差z-score (settle-close 在20日窗口的标准化值)"
    WINDOW = 20

    def dependencies(self) -> list:
        return ["close", "settle"]

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        settle = data.get("settle", dates, universe)
        basis = settle - close
        # 用收盘价标准化
        basis_norm = basis / close.replace(0, np.nan)
        # 20日 z-score
        mean = basis_norm.rolling(self.WINDOW, min_periods=10).mean()
        std = basis_norm.rolling(self.WINDOW, min_periods=10).std()
        result = (basis_norm - mean) / std.replace(0, np.nan)
        return result

