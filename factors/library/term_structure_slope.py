"""期限曲线斜率及分布特征因子.

本模块基于近月/远月合约价格计算期限结构相关因子:
- term_structure_slope_20d: 期限曲线斜率 (结算价版), 与 roll_yield_20d 公式相同
  但使用 settle 价格而非 close, 20 日均值平滑
- basis_ratio_20d: 基差率因子 (settle-close)/close, 20 日均值平滑
- roll_yield_skew_20d: 展期收益的 20 日滚动偏度, 反映期限结构分布的非对称性
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.interfaces import Factor
from core.registry import register_factor


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
