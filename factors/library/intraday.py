"""日内因子 (分钟数据聚合为日度值).

依赖 DDB 分钟K线数据, 通过 DDBSource.fetch_intraday_features() 聚合为日度字段:
- vwap: 成交量加权均价
- intraday_return: 日内收益 (close-open)/open
- overnight_gap: 隔夜跳空 (open-prev_close)/prev_close
- intraday_volatility: 日内振幅 (high-low)/open
- close_to_vwap: 收盘相对VWAP偏离
- volume_concentration: 成交量集中度
- amihud_illiquidity: Amihud非流动性
- tail_momentum: 尾盘动量

当数据源不支持日内字段时 (如 MySQLSource), 因子返回 NaN, 不影响回测.
因子自动降级: 无日内数据时被 IC 检验过滤掉.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.interfaces import Factor
from core.registry import register_factor


class IntradayFactorBase(Factor):
    """日内因子基类: 封装通用逻辑.

    子类只需定义:
    - name, description, WINDOW
    - dependencies() → 返回日内字段名
    - _transform(df) → 对原始日内字段做变换
    """

    category = "intraday"
    frequency = "daily"
    WINDOW = 5

    def compute(self, data, dates, universe):
        field = self.dependencies()[0]
        raw = data.get(field, dates, universe)
        if raw is None or raw.empty or raw.isna().all().all():
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        return self._transform(raw)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """子类实现: 对原始日内字段做变换."""
        raise NotImplementedError


# ======================================================================
# 日内动量因子: 日内收益的滚动均值
# ======================================================================

@register_factor("intraday_momentum_5d", category="intraday")
class IntradayMomentum5d(IntradayFactorBase):
    """日内动量因子 (5日).

    日内收益 (close-open)/open 的 5 日滚动均值.
    正值 = 日内趋势向上, 负值 = 日内趋势向下.
    """
    name = "intraday_momentum_5d"
    description = "日内动量 (日内收益5日均值)"
    WINDOW = 5

    def dependencies(self) -> list:
        return ["intraday_return"]

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rolling(self.WINDOW, min_periods=3).mean()


@register_factor("intraday_momentum_20d", category="intraday")
class IntradayMomentum20d(IntradayMomentum5d):
    """日内动量因子 (20日)."""
    name = "intraday_momentum_20d"
    description = "日内动量 (日内收益20日均值)"
    WINDOW = 20


# ======================================================================
# 隔夜跳空因子: 隔夜跳空的滚动均值
# ======================================================================

@register_factor("overnight_gap_5d", category="intraday")
class OvernightGap5d(IntradayFactorBase):
    """隔夜跳空因子 (5日).

    隔夜跳空 (open-prev_close)/prev_close 的 5 日滚动均值.
    正值 = 持续高开, 负值 = 持续低开.
    逻辑: 隔夜跳空反映夜间信息冲击, 持续同方向跳空预示趋势.
    """
    name = "overnight_gap_5d"
    description = "隔夜跳空 (隔夜收益5日均值)"
    WINDOW = 5

    def dependencies(self) -> list:
        return ["overnight_gap"]

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rolling(self.WINDOW, min_periods=3).mean()


@register_factor("overnight_gap_20d", category="intraday")
class OvernightGap20d(OvernightGap5d):
    """隔夜跳空因子 (20日)."""
    name = "overnight_gap_20d"
    description = "隔夜跳空 (隔夜收益20日均值)"
    WINDOW = 20


# ======================================================================
# VWAP偏离因子: 收盘相对VWAP偏离的滚动均值
# ======================================================================

@register_factor("vwap_deviation_5d", category="intraday")
class VWAPDeviation5d(IntradayFactorBase):
    """VWAP偏离因子 (5日).

    (close-vwap)/vwap 的 5 日滚动均值.
    正值 = 收盘价持续高于VWAP (买盘强), 负值 = 持续低于VWAP (卖盘强).
    """
    name = "vwap_deviation_5d"
    description = "VWAP偏离 (收盘-VWAP的5日均值)"
    WINDOW = 5

    def dependencies(self) -> list:
        return ["close_to_vwap"]

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rolling(self.WINDOW, min_periods=3).mean()


@register_factor("vwap_deviation_20d", category="intraday")
class VWAPDeviation20d(VWAPDeviation5d):
    """VWAP偏离因子 (20日)."""
    name = "vwap_deviation_20d"
    description = "VWAP偏离 (收盘-VWAP的20日均值)"
    WINDOW = 20


# ======================================================================
# 尾盘动量因子: 尾盘动量的滚动均值
# ======================================================================

@register_factor("tail_momentum_5d", category="intraday")
class TailMomentum5d(IntradayFactorBase):
    """尾盘动量因子 (5日).

    尾盘30分钟收益的 5 日滚动均值.
    正值 = 尾盘持续走强 (机构买入), 负值 = 尾盘持续走弱 (机构卖出).
    逻辑: 尾盘交易反映机构意图, 尾盘强→次日大概率延续.
    """
    name = "tail_momentum_5d"
    description = "尾盘动量 (尾盘收益5日均值)"
    WINDOW = 5

    def dependencies(self) -> list:
        return ["tail_momentum"]

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rolling(self.WINDOW, min_periods=3).mean()


@register_factor("tail_momentum_20d", category="intraday")
class TailMomentum20d(TailMomentum5d):
    """尾盘动量因子 (20日)."""
    name = "tail_momentum_20d"
    description = "尾盘动量 (尾盘收益20日均值)"
    WINDOW = 20


# ======================================================================
# 日内波动率结构因子: 日内振幅 / 日度收益波动率
# ======================================================================

@register_factor("intraday_vol_ratio_5d", category="intraday")
class IntradayVolRatio5d(IntradayFactorBase):
    """日内波动率结构因子 (5日).

    日内振幅 (high-low)/open 的 5 日均值.
    高值 = 日内波动剧烈 (趋势中), 低值 = 日内平淡 (盘整中).
    """
    name = "intraday_vol_ratio_5d"
    description = "日内波动率结构 (日内振幅5日均值)"
    WINDOW = 5

    def dependencies(self) -> list:
        return ["intraday_volatility"]

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rolling(self.WINDOW, min_periods=3).mean()


@register_factor("intraday_vol_ratio_20d", category="intraday")
class IntradayVolRatio20d(IntradayVolRatio5d):
    """日内波动率结构因子 (20日)."""
    name = "intraday_vol_ratio_20d"
    description = "日内波动率结构 (日内振幅20日均值)"
    WINDOW = 20


# ======================================================================
# 成交量集中度因子: 成交量集中度的滚动均值
# ======================================================================

@register_factor("volume_concentration_5d", category="intraday")
class VolumeConcentration5d(IntradayFactorBase):
    """成交量集中度因子 (5日).

    成交量集中度 (峰值/均值) 的 5 日滚动均值.
    高值 = 成交量集中在少数时段 (信息驱动), 低值 = 成交均匀 (流动性驱动).
    """
    name = "volume_concentration_5d"
    description = "成交量集中度 (5日均值)"
    WINDOW = 5

    def dependencies(self) -> list:
        return ["volume_concentration"]

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rolling(self.WINDOW, min_periods=3).mean()


# ======================================================================
# Amihud非流动性因子: Amihud指标 的滚动均值
# ======================================================================

@register_factor("amihud_illiquidity_5d", category="intraday")
class AmihudIlliquidity5d(IntradayFactorBase):
    """Amihud非流动性因子 (5日).

    Amihud指标 (|收益|/金额) 的 5 日滚动均值.
    高值 = 流动性差 (大额交易冲击大), 低值 = 流动性好.
    逻辑: 流动性差的品种预期有流动性溢价.
    """
    name = "amihud_illiquidity_5d"
    description = "Amihud非流动性 (5日均值)"
    WINDOW = 5

    def dependencies(self) -> list:
        return ["amihud_illiquidity"]

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rolling(self.WINDOW, min_periods=3).mean()


@register_factor("amihud_illiquidity_20d", category="intraday")
class AmihudIlliquidity20d(AmihudIlliquidity5d):
    """Amihud非流动性因子 (20日)."""
    name = "amihud_illiquidity_20d"
    description = "Amihud非流动性 (20日均值)"
    WINDOW = 20


# ======================================================================
# 日内反转因子: 负的日内动量 (短期反转)
# ======================================================================

@register_factor("intraday_reversal_5d", category="intraday")
class IntradayReversal5d(IntradayFactorBase):
    """日内反转因子 (5日).

    负的日内动量: 日内收益5日均值的反面.
    逻辑: 日内过度上涨的品种次日有回落倾向 (短期反转).
    """
    name = "intraday_reversal_5d"
    description = "日内反转 (负的日内动量5日均值)"
    WINDOW = 5

    def dependencies(self) -> list:
        return ["intraday_return"]

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return -df.rolling(self.WINDOW, min_periods=3).mean()
