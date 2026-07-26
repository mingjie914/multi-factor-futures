"""跨品种动量与板块轮动因子.

核心思想: 单品种动量只看自身历史收益, 跨品种动量看品种间的相对强弱.
- 截面动量排序: 品种收益在全部品种中的排名百分位
- 板块动量: 品种所属板块的平均动量
- 相对强弱: 品种收益 vs 板块平均收益的差

这是 CTA 策略的核心 alpha 源之一, 当前 697 因子全部为单品种,
本模块补充跨品种结构性 alpha.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.interfaces import Factor
from core.registry import register_factor
from core.sectors import SECTOR_MAP


def _get_sector_series(universe: pd.Index) -> pd.Series:
    """获取品种->板块映射 Series."""
    return pd.Series([SECTOR_MAP.get(s, "other") for s in universe], index=universe)


@register_factor("cross_section_momentum_5d", category="cross_commodity")
class CrossSectionMomentum5d(Factor):
    """截面动量排序因子 (5日).

    品种 5 日收益率在全部品种中的排名百分位 (0~1).
    高百分位 = 近期强势, 低百分位 = 近期弱势.

    与单品种 momentum_5d 的区别:
    - 单品种: 看自身 5 日收益的绝对值
    - 截面: 看自身 5 日收益在所有品种中的相对位置
    """
    name = "cross_section_momentum_5d"
    category = "cross_commodity"
    frequency = "daily"
    description = "截面动量排序 (5日收益在全部品种中的排名百分位)"
    WINDOW = 5

    def dependencies(self) -> list:
        return ["close"]

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        ret = close.pct_change(self.WINDOW)
        # 截面排名百分位 (每行跨所有品种)
        result = ret.rank(pct=True, axis=1) - 0.5
        return result


@register_factor("cross_section_momentum_20d", category="cross_commodity")
class CrossSectionMomentum20d(CrossSectionMomentum5d):
    """截面动量排序因子 (20日)."""
    name = "cross_section_momentum_20d"
    description = "截面动量排序 (20日收益在全部品种中的排名百分位)"
    WINDOW = 20


@register_factor("sector_momentum_5d", category="cross_commodity")
class SectorMomentum5d(Factor):
    """板块动量因子 (5日).

    品种所属板块的平均 5 日收益率.
    正值 = 板块整体强势, 负值 = 板块整体弱势.

    逻辑: 板块动量是 CTA 策略的重要 alpha 源,
    同板块品种有共动性, 板块趋势比单品种趋势更稳定.
    """
    name = "sector_momentum_5d"
    category = "cross_commodity"
    frequency = "daily"
    description = "板块动量 (品种所属板块的平均5日收益率)"
    WINDOW = 5

    def dependencies(self) -> list:
        return ["close"]

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        ret = close.pct_change(self.WINDOW)
        sectors = _get_sector_series(universe)

        # 计算每日每个板块的平均收益
        result = pd.DataFrame(np.nan, index=dates, columns=universe)
        for sector in sectors.unique():
            sector_symbols = sectors[sectors == sector].index
            sector_symbols = [s for s in sector_symbols if s in ret.columns]
            if not sector_symbols:
                continue
            # 板块平均收益
            sector_avg = ret[sector_symbols].mean(axis=1)
            # 广播到该板块的所有品种
            for sym in sector_symbols:
                result[sym] = sector_avg

        return result


@register_factor("sector_momentum_20d", category="cross_commodity")
class SectorMomentum20d(SectorMomentum5d):
    """板块动量因子 (20日)."""
    name = "sector_momentum_20d"
    description = "板块动量 (品种所属板块的平均20日收益率)"
    WINDOW = 20


@register_factor("relative_strength_sector_20d", category="cross_commodity")
class RelativeStrengthSector20d(Factor):
    """板块相对强弱因子 (20日).

    品种收益 vs 所属板块平均收益的差.
    正值 = 品种跑赢板块, 负值 = 品种跑输板块.

    逻辑: 板块内强者恒强, 跑赢板块的品种可能继续跑赢.
    """
    name = "relative_strength_sector_20d"
    category = "cross_commodity"
    frequency = "daily"
    description = "板块相对强弱 (品种收益 - 板块平均收益, 20日)"
    WINDOW = 20

    def dependencies(self) -> list:
        return ["close"]

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        ret = close.pct_change(self.WINDOW)
        sectors = _get_sector_series(universe)

        result = pd.DataFrame(np.nan, index=dates, columns=universe)
        for sector in sectors.unique():
            sector_symbols = sectors[sectors == sector].index
            sector_symbols = [s for s in sector_symbols if s in ret.columns]
            if not sector_symbols:
                continue
            sector_avg = ret[sector_symbols].mean(axis=1)
            for sym in sector_symbols:
                result[sym] = ret[sym] - sector_avg

        return result


@register_factor("sector_rotation_5d", category="cross_commodity")
class SectorRotation5d(Factor):
    """板块轮动因子 (5日).

    品种所属板块的动量在所有板块中的排名百分位.
    高百分位 = 板块是当前强势板块, 低百分位 = 弱势板块.

    逻辑: 板块轮动是宏观资金流向的体现,
    资金从弱势板块流向强势板块, 强势板块品种有超额收益.
    """
    name = "sector_rotation_5d"
    category = "cross_commodity"
    frequency = "daily"
    description = "板块轮动 (板块动量在所有板块中的排名百分位, 5日)"
    WINDOW = 5

    def dependencies(self) -> list:
        return ["close"]

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        ret = close.pct_change(self.WINDOW)
        sectors = _get_sector_series(universe)

        # 计算每日每个板块的平均收益
        sector_names = sorted(sectors.unique())
        sector_ret = pd.DataFrame(np.nan, index=dates, columns=sector_names)
        for sector in sector_names:
            sector_symbols = sectors[sectors == sector].index
            sector_symbols = [s for s in sector_symbols if s in ret.columns]
            if not sector_symbols:
                continue
            sector_ret[sector] = ret[sector_symbols].mean(axis=1)

        # 板块收益排名百分位
        sector_rank = sector_ret.rank(pct=True, axis=1) - 0.5

        # 广播到每个品种
        result = pd.DataFrame(np.nan, index=dates, columns=universe)
        for sym in universe:
            sector = sectors.get(sym, "other")
            if sector in sector_rank.columns:
                result[sym] = sector_rank[sector]

        return result


@register_factor("sector_rotation_20d", category="cross_commodity")
class SectorRotation20d(SectorRotation5d):
    """板块轮动因子 (20日)."""
    name = "sector_rotation_20d"
    description = "板块轮动 (板块动量在所有板块中的排名百分位, 20日)"
    WINDOW = 20


@register_factor("cross_section_reversal_5d", category="cross_commodity")
class CrossSectionReversal5d(Factor):
    """截面反转因子 (5日).

    负的截面动量: 5日收益排名百分位的反面.
    低百分位品种 (近期弱势) 预期反转上涨.

    逻辑: 短期截面反转是期货市场的常见效应,
    超跌品种有均值回归倾向.
    """
    name = "cross_section_reversal_5d"
    category = "cross_commodity"
    frequency = "daily"
    description = "截面反转 (负的5日截面动量, 超跌品种预期反弹)"
    WINDOW = 5

    def dependencies(self) -> list:
        return ["close"]

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        ret = close.pct_change(self.WINDOW)
        # 截面排名百分位, 取反 (反转)
        result = 0.5 - ret.rank(pct=True, axis=1)
        return result
