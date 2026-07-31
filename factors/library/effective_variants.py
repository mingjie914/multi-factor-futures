"""有效因子变体批次 — 基于已验证的 11 个有效因子做变换生成 10 个新候选.

思路: 复用原因子 compute() 得到日度暴露, 再做二次变换 (rank/zscore/delta/smooth/
vol_scaled/stability/组合), 全部保持 shift(1) 防未来. 因子方向继承原因子方向.

10 个变体:
  V1  jump_intensity_rank_20d        跳跃强度截面rank
  V2  peak_count_zscore_20d          价峰计数截面zscore
  V3  skewness_delta_10d             已实现偏度10日差分
  V4  dtws_smooth_3d                 跌幅时间重心3日平滑
  V5  roll_spread_vol_scaled_20d     Roll价差波动率缩放
  V6  kyle_lambda_stability_20d      Kyle冲击稳定性
  V7  open_close_vol_rank_20d        开盘尾盘量比截面rank
  V8  parkinson_over_rv_20d          Parkinson/已实现波动比
  V9  jump_times_skew_20d            跳跃×偏度交互
  V10 peak_count_delta_20d           价峰计数10日差分
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.interfaces import Factor
from core.registry import register_factor
from factors.library.intraday import (
    IntradayJumpIntensity20d,
    IntradayPricePeakCount20d,
    IntradayRealisedSkewness20d,
    IntradayDTWS20d,
    IntradayRollSpread20d,
    IntradayKyleLambda20d,
    IntradayOpenCloseVolumeRatio20d,
    IntradayParkinsonVolRatio20d,
)


class _VariantBase(Factor):
    """变体基类: 调用基因子 compute, 再做截面/时序变换."""

    category = "intraday_advanced"
    frequency = "daily"
    validation_horizons = (5, 10, 20)
    BASE = None  # 子类设置

    def dependencies(self) -> list:
        return []

    def _transform(self, base: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def compute(self, data, dates, universe):
        base = self.BASE().compute(data, dates, universe)
        transformed = self._transform(base)
        return transformed.reindex(index=dates, columns=universe)


def _cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """截面 rank (0~1)."""
    return df.rank(axis=1, pct=True)


def _cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """截面 z-score."""
    mean = df.mean(axis=1)
    std = df.std(axis=1, ddof=0).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)


# ═══════════════════════════════════════════════════════════════════════════
# V1. jump_intensity_rank_20d — 跳跃强度截面rank (方向: 负向不变)
# ═══════════════════════════════════════════════════════════════════════════
@register_factor("jump_intensity_rank_20d", category="intraday_advanced")
class JumpIntensityRank20d(_VariantBase):
    name = "jump_intensity_rank_20d"
    description = "跳跃强度截面排名"
    BASE = IntradayJumpIntensity20d

    def _transform(self, base):
        return _cs_rank(base)


# ═══════════════════════════════════════════════════════════════════════════
# V2. peak_count_zscore_20d — 价峰计数截面zscore (方向: 正向不变)
# ═══════════════════════════════════════════════════════════════════════════
@register_factor("peak_count_zscore_20d", category="intraday_advanced")
class PeakCountZscore20d(_VariantBase):
    name = "peak_count_zscore_20d"
    description = "价峰计数截面标准化"
    BASE = IntradayPricePeakCount20d

    def _transform(self, base):
        return _cs_zscore(base)


# ═══════════════════════════════════════════════════════════════════════════
# V3. skewness_delta_10d — 已实现偏度10日差分 (方向: 正向)
#     捕捉偏度水平的短期变化
# ═══════════════════════════════════════════════════════════════════════════
@register_factor("skewness_delta_10d", category="intraday_advanced")
class SkewnessDelta10d(_VariantBase):
    name = "skewness_delta_10d"
    description = "已实现偏度10日差分"
    BASE = IntradayRealisedSkewness20d

    def _transform(self, base):
        return base.diff(10)


# ═══════════════════════════════════════════════════════════════════════════
# V4. dtws_smooth_3d — 跌幅时间重心3日平滑 (方向: 正向)
# ═══════════════════════════════════════════════════════════════════════════
@register_factor("dtws_smooth_3d", category="intraday_advanced")
class DTWSSmooth3d(_VariantBase):
    name = "dtws_smooth_3d"
    description = "跌幅时间重心3日平滑"
    BASE = IntradayDTWS20d

    def _transform(self, base):
        return base.rolling(3, min_periods=2).mean()


# ═══════════════════════════════════════════════════════════════════════════
# V5. roll_spread_vol_scaled_20d — Roll价差波动率缩放 (方向: 负向)
#     价差/波动率: 剔除波动影响的真实价差
# ═══════════════════════════════════════════════════════════════════════════
@register_factor("roll_spread_vol_scaled_20d", category="intraday_advanced")
class RollSpreadVolScaled20d(_VariantBase):
    name = "roll_spread_vol_scaled_20d"
    description = "Roll价差波动率缩放"
    BASE = IntradayRollSpread20d

    def _transform(self, base):
        rv = base.rolling(20, min_periods=5).std(ddof=0).replace(0, np.nan)
        return base / rv


# ═══════════════════════════════════════════════════════════════════════════
# V6. kyle_lambda_stability_20d — Kyle冲击稳定性 (方向: 负向)
#     均值/标准差: 高=冲击行为稳定可预测
# ═══════════════════════════════════════════════════════════════════════════
@register_factor("kyle_lambda_stability_20d", category="intraday_advanced")
class KyleLambdaStability20d(_VariantBase):
    name = "kyle_lambda_stability_20d"
    description = "Kyle冲击稳定性 (均值/标准差)"
    BASE = IntradayKyleLambda20d

    def _transform(self, base):
        rm = base.rolling(20, min_periods=5).mean()
        rs = base.rolling(20, min_periods=5).std(ddof=0).replace(0, np.nan)
        return rm / rs


# ═══════════════════════════════════════════════════════════════════════════
# V7. open_close_vol_rank_20d — 开盘尾盘量比截面rank (方向: 负向不变)
# ═══════════════════════════════════════════════════════════════════════════
@register_factor("open_close_vol_rank_20d", category="intraday_advanced")
class OpenCloseVolRank20d(_VariantBase):
    name = "open_close_vol_rank_20d"
    description = "开盘尾盘量比截面排名"
    BASE = IntradayOpenCloseVolumeRatio20d

    def _transform(self, base):
        return _cs_rank(base)


# ═══════════════════════════════════════════════════════════════════════════
# V8. parkinson_over_rv_20d — Parkinson/已实现波动比 (方向: 负向)
#     高=日内震荡但收盘不动→噪声主导
# ═══════════════════════════════════════════════════════════════════════════
@register_factor("parkinson_over_rv_20d", category="intraday_advanced")
class ParkinsonOverRV20d(_VariantBase):
    name = "parkinson_over_rv_20d"
    description = "Parkinson/已实现波动比"
    BASE = IntradayParkinsonVolRatio20d

    def _transform(self, base):
        # 用基因子自身rolling std作为已实现波动代理
        rv = base.rolling(20, min_periods=5).std(ddof=0).replace(0, np.nan)
        return base / rv


# ═══════════════════════════════════════════════════════════════════════════
# V9. jump_times_skew_20d — 跳跃×偏度交互 (方向: 负向)
#     负偏度伴随高跳跃→恐慌抛售信号增强
# ═══════════════════════════════════════════════════════════════════════════
@register_factor("jump_times_skew_20d", category="intraday_advanced")
class JumpTimesSkew20d(Factor):
    name = "jump_times_skew_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "跳跃×偏度交互"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        jump = IntradayJumpIntensity20d().compute(data, dates, universe)
        skew = IntradayRealisedSkewness20d().compute(data, dates, universe)
        # 跳跃(负向) × 偏度(正向): 高跳跃低偏度→负向信号更强
        return (jump * skew).reindex(index=dates, columns=universe)


# ═══════════════════════════════════════════════════════════════════════════
# V10. peak_count_delta_20d — 价峰计数10日差分 (方向: 正向)
#      跳跃活动上升→信息加速
# ═══════════════════════════════════════════════════════════════════════════
@register_factor("peak_count_delta_20d", category="intraday_advanced")
class PeakCountDelta20d(_VariantBase):
    name = "peak_count_delta_20d"
    description = "价峰计数10日差分"
    BASE = IntradayPricePeakCount20d

    def _transform(self, base):
        return base.diff(10)
