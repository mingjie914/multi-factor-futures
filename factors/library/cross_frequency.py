"""跨频率因子 — 日度因子 × 日内特征的组合确认信号.

核心思想:
    日度因子提供趋势方向, 日内特征 (从分钟数据聚合) 提供微观确认.
    两者相乘 (z-score 标准化后) 形成跨频率确认信号, 比单一频率因子更稳健.

因子列表 (8 组合 × 2 窗口 = 16 个因子):
    1. daily_mom_x_intraday_vol:     日度动量 × 日内波动率 (趋势+波动确认)
    2. daily_mom_x_tail_momentum:    日度动量 × 尾盘动量 (趋势+尾盘确认)
    3. daily_mom_x_overnight_gap:    日度动量 × 隔夜跳空 (趋势+隔夜确认)
    4. daily_mom_x_vwap_dev:         日度动量 × VWAP偏离 (趋势+VWAP确认)
    5. overnight_gap_x_intraday_vol: 隔夜跳空 × 日内波动率 (跳空+波动确认)
    6. vwap_dev_x_intraday_vol:      VWAP偏离 × 日内波动率 (VWAP+波动确认)
    7. tail_mom_x_volume_conc:       尾盘动量 × 成交量集中度 (尾盘+量能确认)
    8. daily_mom_x_amihud:           日度动量 × Amihud非流动性 (趋势+流动性确认)

数据依赖:
    - 日度 OHLCV: 通过 DataManager 从本地日线 Parquet 获取
    - 日内特征: 通过 DataManager 从本地分钟 Parquet 聚合获取
    - 发现扫描允许缺失因子保持 NaN；正式选中因子必须通过严格完整性检查

计算逻辑:
    1. 获取日度因子值 (如 5日收益率)
    2. 获取日内特征值 (如 日内波动率5日均值)
    3. 对两者分别做截面 z-score 标准化 (by_date=True)
    4. 相乘得到跨频率确认信号
    5. 公式: cross_signal = z(daily_factor) × z(intraday_feature)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.interfaces import Factor
from core.registry import register_factor


def _zscore_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """截面 z-score 标准化 (每日横截面去均值/除标准差)."""
    mean = df.mean(axis=1)
    std = df.std(axis=1, ddof=0)
    return df.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)


def _rolling_mean(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """滚动均值 (最小3期)."""
    return df.rolling(window, min_periods=3).mean()


class CrossFrequencyFactorBase(Factor):
    """跨频率因子基类.

    子类需定义:
    - name, description, WINDOW
    - daily_field: 日度字段名 (如 "close")
    - intraday_field: 日内字段名 (如 "intraday_volatility")
    - _compute_daily(df): 从日度字段计算日度因子值
    """

    category = "cross_frequency"
    frequency = "daily"
    WINDOW = 5
    daily_field = "close"
    intraday_field = ""

    def dependencies(self) -> list:
        """返回因子依赖的数据字段 (日度 + 日内)."""
        deps = [self.daily_field]
        if self.intraday_field:
            deps.append(self.intraday_field)
        return deps

    def compute(self, data, dates, universe):
        # 1. 获取日度数据, 计算日度因子值
        daily_raw = data.get(self.daily_field, dates, universe)
        if daily_raw is None or daily_raw.empty or daily_raw.isna().all().all():
            return pd.DataFrame(np.nan, index=dates, columns=universe)

        daily_signal = self._compute_daily(daily_raw)

        # 2. 获取日内特征数据
        if not self.intraday_field:
            return pd.DataFrame(np.nan, index=dates, columns=universe)

        intraday_raw = data.get(self.intraday_field, dates, universe)
        if intraday_raw is None or intraday_raw.empty or intraday_raw.isna().all().all():
            # 发现扫描保持缺失；正式选中因子由严格计算链拒绝。
            return pd.DataFrame(np.nan, index=dates, columns=universe)

        # 3. 日内特征滚动均值
        intraday_signal = _rolling_mean(intraday_raw, self.WINDOW)

        # 4. 截面 z-score 标准化后相乘
        daily_z = _zscore_by_date(daily_signal)
        intraday_z = _zscore_by_date(intraday_signal)
        cross_signal = daily_z * intraday_z

        return cross_signal.reindex(index=dates, columns=universe)

    def _compute_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """子类实现: 从日度字段计算日度因子值."""
        raise NotImplementedError


# ======================================================================
# 1. 日度动量 × 日内波动率 (趋势+波动确认)
# ======================================================================

@register_factor("daily_mom_x_intraday_vol_5d", category="cross_frequency")
class DailyMomXIntradayVol5d(CrossFrequencyFactorBase):
    """日度动量 × 日内波动率 (5日).

    逻辑: 趋势方向 (日度动量) 与 波动强度 (日内波动率) 的确认.
    高波动+正动量 = 强趋势信号; 低波动+正动量 = 弱趋势信号.
    """
    name = "daily_mom_x_intraday_vol_5d"
    description = "日度5日动量 × 日内波动率5日均值 (趋势+波动确认)"
    WINDOW = 5
    daily_field = "close"
    intraday_field = "intraday_volatility"

    def _compute_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.pct_change(self.WINDOW, fill_method=None)


@register_factor("daily_mom_x_intraday_vol_20d", category="cross_frequency")
class DailyMomXIntradayVol20d(DailyMomXIntradayVol5d):
    """日度动量 × 日内波动率 (20日)."""
    name = "daily_mom_x_intraday_vol_20d"
    description = "日度20日动量 × 日内波动率20日均值 (趋势+波动确认)"
    WINDOW = 20


# ======================================================================
# 2. 日度动量 × 尾盘动量 (趋势+尾盘确认)
# ======================================================================

@register_factor("daily_mom_x_tail_momentum_5d", category="cross_frequency")
class DailyMomXTailMomentum5d(CrossFrequencyFactorBase):
    """日度动量 × 尾盘动量 (5日).

    逻辑: 日度趋势方向与尾盘动量的确认.
    正动量+正尾盘 = 机构尾盘买入, 次日延续概率高.
    """
    name = "daily_mom_x_tail_momentum_5d"
    description = "日度5日动量 × 尾盘动量5日均值 (趋势+尾盘确认)"
    WINDOW = 5
    daily_field = "close"
    intraday_field = "tail_momentum"

    def _compute_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.pct_change(self.WINDOW, fill_method=None)


@register_factor("daily_mom_x_tail_momentum_20d", category="cross_frequency")
class DailyMomXTailMomentum20d(DailyMomXTailMomentum5d):
    """日度动量 × 尾盘动量 (20日)."""
    name = "daily_mom_x_tail_momentum_20d"
    description = "日度20日动量 × 尾盘动量20日均值 (趋势+尾盘确认)"
    WINDOW = 20


# ======================================================================
# 3. 日度动量 × 隔夜跳空 (趋势+隔夜确认)
# ======================================================================

@register_factor("daily_mom_x_overnight_gap_5d", category="cross_frequency")
class DailyMomXOvernightGap5d(CrossFrequencyFactorBase):
    """日度动量 × 隔夜跳空 (5日).

    逻辑: 日度趋势方向与隔夜跳空的确认.
    正动量+正跳空 = 持续高开, 趋势强化; 负动量+负跳空 = 持续低开, 跌势强化.
    """
    name = "daily_mom_x_overnight_gap_5d"
    description = "日度5日动量 × 隔夜跳空5日均值 (趋势+隔夜确认)"
    WINDOW = 5
    daily_field = "close"
    intraday_field = "overnight_gap"

    def _compute_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.pct_change(self.WINDOW, fill_method=None)


@register_factor("daily_mom_x_overnight_gap_20d", category="cross_frequency")
class DailyMomXOvernightGap20d(DailyMomXOvernightGap5d):
    """日度动量 × 隔夜跳空 (20日)."""
    name = "daily_mom_x_overnight_gap_20d"
    description = "日度20日动量 × 隔夜跳空20日均值 (趋势+隔夜确认)"
    WINDOW = 20


# ======================================================================
# 4. 日度动量 × VWAP偏离 (趋势+VWAP确认)
# ======================================================================

@register_factor("daily_mom_x_vwap_dev_5d", category="cross_frequency")
class DailyMomXVwapDev5d(CrossFrequencyFactorBase):
    """日度动量 × VWAP偏离 (5日).

    逻辑: 日度趋势方向与收盘相对VWAP偏离的确认.
    正动量+正偏离 = 收盘持续高于VWAP (买盘强), 趋势可信.
    """
    name = "daily_mom_x_vwap_dev_5d"
    description = "日度5日动量 × VWAP偏离5日均值 (趋势+VWAP确认)"
    WINDOW = 5
    daily_field = "close"
    intraday_field = "close_to_vwap"

    def _compute_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.pct_change(self.WINDOW, fill_method=None)


@register_factor("daily_mom_x_vwap_dev_20d", category="cross_frequency")
class DailyMomXVwapDev20d(DailyMomXVwapDev5d):
    """日度动量 × VWAP偏离 (20日)."""
    name = "daily_mom_x_vwap_dev_20d"
    description = "日度20日动量 × VWAP偏离20日均值 (趋势+VWAP确认)"
    WINDOW = 20


# ======================================================================
# 5. 隔夜跳空 × 日内波动率 (跳空+波动确认)
# ======================================================================

@register_factor("overnight_gap_x_intraday_vol_5d", category="cross_frequency")
class OvernightGapXIntradayVol5d(CrossFrequencyFactorBase):
    """隔夜跳空 × 日内波动率 (5日).

    逻辑: 隔夜信息冲击 (跳空) 与 日内波动确认.
    大跳空+高日内波动 = 信息驱动行情, 趋势可能延续.
    """
    name = "overnight_gap_x_intraday_vol_5d"
    description = "隔夜跳空5日均值 × 日内波动率5日均值 (跳空+波动确认)"
    WINDOW = 5
    daily_field = "close"  # 仅用于确定日期范围, 实际不参与计算
    intraday_field = "intraday_volatility"

    def compute(self, data, dates, universe):
        # 隔夜跳空本身就是日内特征
        overnight = data.get("overnight_gap", dates, universe)
        intraday_vol = data.get("intraday_volatility", dates, universe)

        if (overnight is None or overnight.empty or overnight.isna().all().all()
                or intraday_vol is None or intraday_vol.empty
                or intraday_vol.isna().all().all()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)

        overnight_signal = _rolling_mean(overnight, self.WINDOW)
        vol_signal = _rolling_mean(intraday_vol, self.WINDOW)

        overnight_z = _zscore_by_date(overnight_signal)
        vol_z = _zscore_by_date(vol_signal)
        return (overnight_z * vol_z).reindex(index=dates, columns=universe)

    def _compute_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        return df  # 不使用


@register_factor("overnight_gap_x_intraday_vol_20d", category="cross_frequency")
class OvernightGapXIntradayVol20d(OvernightGapXIntradayVol5d):
    """隔夜跳空 × 日内波动率 (20日)."""
    name = "overnight_gap_x_intraday_vol_20d"
    description = "隔夜跳空20日均值 × 日内波动率20日均值 (跳空+波动确认)"
    WINDOW = 20


# ======================================================================
# 6. VWAP偏离 × 日内波动率 (VWAP+波动确认)
# ======================================================================

@register_factor("vwap_dev_x_intraday_vol_5d", category="cross_frequency")
class VwapDevXIntradayVol5d(CrossFrequencyFactorBase):
    """VWAP偏离 × 日内波动率 (5日).

    逻辑: 收盘相对VWAP偏离 与 日内波动率的确认.
    正偏离+高波动 = 强势收盘+信息驱动, 趋势可信.
    """
    name = "vwap_dev_x_intraday_vol_5d"
    description = "VWAP偏离5日均值 × 日内波动率5日均值 (VWAP+波动确认)"
    WINDOW = 5
    daily_field = "close"
    intraday_field = "intraday_volatility"

    def compute(self, data, dates, universe):
        vwap_dev = data.get("close_to_vwap", dates, universe)
        intraday_vol = data.get("intraday_volatility", dates, universe)

        if (vwap_dev is None or vwap_dev.empty or vwap_dev.isna().all().all()
                or intraday_vol is None or intraday_vol.empty
                or intraday_vol.isna().all().all()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)

        vwap_signal = _rolling_mean(vwap_dev, self.WINDOW)
        vol_signal = _rolling_mean(intraday_vol, self.WINDOW)

        vwap_z = _zscore_by_date(vwap_signal)
        vol_z = _zscore_by_date(vol_signal)
        return (vwap_z * vol_z).reindex(index=dates, columns=universe)

    def _compute_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        return df  # 不使用


@register_factor("vwap_dev_x_intraday_vol_20d", category="cross_frequency")
class VwapDevXIntradayVol20d(VwapDevXIntradayVol5d):
    """VWAP偏离 × 日内波动率 (20日)."""
    name = "vwap_dev_x_intraday_vol_20d"
    description = "VWAP偏离20日均值 × 日内波动率20日均值 (VWAP+波动确认)"
    WINDOW = 20


# ======================================================================
# 7. 尾盘动量 × 成交量集中度 (尾盘+量能确认)
# ======================================================================

@register_factor("tail_mom_x_volume_conc_5d", category="cross_frequency")
class TailMomXVolumeConc5d(CrossFrequencyFactorBase):
    """尾盘动量 × 成交量集中度 (5日).

    逻辑: 尾盘动量与成交量集中度的确认.
    正尾盘+高集中 = 尾盘放量买入, 机构意图明显.
    """
    name = "tail_mom_x_volume_conc_5d"
    description = "尾盘动量5日均值 × 成交量集中度5日均值 (尾盘+量能确认)"
    WINDOW = 5
    daily_field = "close"
    intraday_field = "volume_concentration"

    def compute(self, data, dates, universe):
        tail_mom = data.get("tail_momentum", dates, universe)
        vol_conc = data.get("volume_concentration", dates, universe)

        if (tail_mom is None or tail_mom.empty or tail_mom.isna().all().all()
                or vol_conc is None or vol_conc.empty
                or vol_conc.isna().all().all()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)

        tail_signal = _rolling_mean(tail_mom, self.WINDOW)
        conc_signal = _rolling_mean(vol_conc, self.WINDOW)

        tail_z = _zscore_by_date(tail_signal)
        conc_z = _zscore_by_date(conc_signal)
        return (tail_z * conc_z).reindex(index=dates, columns=universe)

    def _compute_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        return df  # 不使用


@register_factor("tail_mom_x_volume_conc_20d", category="cross_frequency")
class TailMomXVolumeConc20d(TailMomXVolumeConc5d):
    """尾盘动量 × 成交量集中度 (20日)."""
    name = "tail_mom_x_volume_conc_20d"
    description = "尾盘动量20日均值 × 成交量集中度20日均值 (尾盘+量能确认)"
    WINDOW = 20


# ======================================================================
# 8. 日度动量 × Amihud非流动性 (趋势+流动性确认)
# ======================================================================

@register_factor("daily_mom_x_amihud_5d", category="cross_frequency")
class DailyMomXAmihud5d(CrossFrequencyFactorBase):
    """日度动量 × Amihud非流动性 (5日).

    逻辑: 趋势方向与流动性溢价的确认.
    正动量+高Amihud (低流动性) = 流动性溢价驱动的趋势, 可能更持久.
    """
    name = "daily_mom_x_amihud_5d"
    description = "日度5日动量 × Amihud非流动性5日均值 (趋势+流动性确认)"
    WINDOW = 5
    daily_field = "close"
    intraday_field = "amihud_illiquidity"

    def _compute_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.pct_change(self.WINDOW, fill_method=None)


@register_factor("daily_mom_x_amihud_20d", category="cross_frequency")
class DailyMomXAmihud20d(DailyMomXAmihud5d):
    """日度动量 × Amihud非流动性 (20日)."""
    name = "daily_mom_x_amihud_20d"
    description = "日度20日动量 × Amihud非流动性20日均值 (趋势+流动性确认)"
    WINDOW = 20
