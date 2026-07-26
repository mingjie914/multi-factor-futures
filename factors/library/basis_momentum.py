"""基差动量因子.

基差 = 结算价 - 收盘价 (或 近月-远月, 此处用 settle-close).
基差动量 = 基差的变化率, 反映期现结构的变化趋势.

逻辑:
- 基差扩大 (正值变大): 多头逼仓/现货紧张, 利多
- 基差缩小 (正值变小): 现货宽松/仓单增加, 利空
- 基差动量为正: 基差在扩大, 预期价格上行
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.interfaces import Factor
from core.registry import register_factor


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
