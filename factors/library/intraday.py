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

import os

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

_MINUTE_FIELDS = ["open", "high", "low", "close", "volume", "amount"]

_LOCAL_MINUTE_ROOT = r"E:\程明杰公司内容\期货行情数据\本地表"

_FREQ_DIR_MAP = {
    "1min": "futureshistoryprices1m",
    "5min": "futureshistoryprices1m",
    "15min": "futureshistoryprices15m",
    "30min": "futureshistoryprices15m",
    "daily": "futureshistoryprices1d",
    "1d": "futureshistoryprices1d",
}


def _read_local_minute(dates, universe, freq="1min"):
    """从本地 Parquet 读取分钟/日度 OHLCV 面板, 按根代码聚合."""
    import logging
    log = logging.getLogger("multi_factor")
    dates = pd.DatetimeIndex(dates)
    subdir = _FREQ_DIR_MAP.get(freq)
    if subdir is None:
        return {}
    base = os.path.join(_LOCAL_MINUTE_ROOT, subdir)
    if not os.path.isdir(base):
        return {}
    start = pd.Timestamp(dates.min())
    end = pd.Timestamp(dates.max())
    months = pd.date_range(start.replace(day=1), end.replace(day=1), freq="MS")
    frames = []
    for month in months:
        partition = "year_month=" + month.strftime("%Y-%m")
        parquet_path = os.path.join(base, partition, "data_0.parquet")
        if not os.path.exists(parquet_path):
            continue
        try:
            df = pd.read_parquet(parquet_path)
            ts = pd.to_datetime(df["trade_datetime"])
            df = df.loc[(ts >= start) & (ts <= end)]
            if not df.empty:
                frames.append(df)
        except Exception:
            log.debug("读取 %s 失败", parquet_path, exc_info=True)
    if not frames:
        return {}
    all_data = pd.concat(frames, ignore_index=True)
    if all_data.empty:
        return {}
    universe_set = set(str(u).upper() for u in universe)
    # 为每个合约分配根代码 (前缀匹配)，统一转大写并优先匹配长代码避免前缀碰撞
    universe_sorted = sorted(universe_set, key=len, reverse=True)
    symbol_root: dict[str, str] = {}
    for sym in all_data["symbol"].unique():
        sym_upper = str(sym).upper()
        for ut in universe_sorted:
            if sym_upper.startswith(ut):
                symbol_root[sym] = ut
                break
    if not symbol_root:
        return {}
    all_data = all_data[all_data["symbol"].isin(symbol_root)]
    all_data["root"] = all_data["symbol"].map(symbol_root)
    if all_data.empty:
        return {}
    # 按 (trade_datetime, root) 聚合: 价格取成交量加权, 量/额求和
    ts = pd.to_datetime(all_data["trade_datetime"])
    all_data["_ts"] = ts
    grouped = all_data.groupby(["_ts", "root"])
    panel = {}
    if "close" in all_data.columns and "volume" in all_data.columns:
        vol = all_data["volume"].replace(0, np.nan)
        vwap_close = (all_data["close"] * vol).groupby([ts, all_data["root"]]).sum() / vol.groupby([ts, all_data["root"]]).sum()
        panel["close"] = vwap_close.unstack(level="root")
        panel["close"].index = pd.DatetimeIndex(panel["close"].index)
    for field in ["open", "high", "low"]:
        if field in all_data.columns:
            s = all_data[field].groupby([ts, all_data["root"]]).mean()
            s = s.unstack(level="root")
            s.index = pd.DatetimeIndex(s.index)
            panel[field] = s
    for field in ["volume", "amount"]:
        if field in all_data.columns:
            s = all_data[field].groupby([ts, all_data["root"]]).sum()
            s = s.unstack(level="root")
            s.index = pd.DatetimeIndex(s.index)
            panel[field] = s
    return panel


def _get_minute_panel(data, dates, universe, freq="1min"):
    """获取分钟级 OHLCV 面板.

    优先级: 本地 Parquet > data.get_at_frequency() > DDBSource.
    """
    import logging
    log = logging.getLogger("multi_factor")
    dates = pd.DatetimeIndex(dates)
    # 1) 本地 Parquet
    try:
        panel = _read_local_minute(dates, universe, freq=freq)
        if panel:
            return panel
    except Exception:
        log.debug("本地分钟数据读取失败, 回退", exc_info=True)
    # 2) data.get_at_frequency
    try:
        panel = {}
        for field in _MINUTE_FIELDS:
            frame = data.get_at_frequency(field, dates, universe, frequency=freq)
            if frame is not None and not frame.empty:
                panel[field] = frame
        if panel:
            return panel
    except Exception:
        pass
    # 3) DDBSource
    source = getattr(data, "source", None)
    if source is not None and hasattr(source, "fetch_price_at_frequency"):
        try:
            return source.fetch_price_at_frequency(
                list(universe), dates.min(), dates.max(), _MINUTE_FIELDS, frequency=freq,
            ) or {}
        except Exception:
            log.debug("DDB 获取失败", exc_info=True)
    return {}

def _safe_div(a, b):
    import numpy as np
    return a / b.where(np.abs(b) > 1e-12, np.nan)

def _cs_zscore(frame):
    mean = frame.mean(axis=1)
    std = frame.std(axis=1, ddof=0)
    if (std < 1e-12).all():
        return pd.DataFrame(0.0, index=frame.index, columns=frame.columns)
    z = frame.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)
    return z.fillna(0.0)

def _mean_distance(frame):
    return _cs_zscore(frame).abs()

def _roll20_mean_std(frame):
    rm = frame.rolling(20, min_periods=5).mean()
    rs = frame.rolling(20, min_periods=5).std(ddof=0)
    return rm.add(rs, fill_value=0)
    return {}


# 1. vp_corr_intraday — 分钟收益×成交量日内相关
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("vp_corr_intraday", category="intraday_advanced")
class VpCorrIntraday(Factor):
    """分钟收益与成交量的日内相关系数.

    corr(ret_1m, volume_1m)，20日滚动均值平滑.
    IC 方向: 负向 (五年累计 IC=-40).
    """
    name = "vp_corr_intraday"
    category = "intraday_advanced"
    frequency = "daily"
    description = "分钟收益与成交量日内相关系数 (20日平滑)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel or "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close_1m, vol_1m = panel["close"], panel["volume"]
        day = close_1m.index.normalize()
        results: dict = {}
        for dt, grp_close in close_1m.groupby(day):
            if dt not in vol_1m.index.normalize():
                continue
            grp_vol = vol_1m.loc[vol_1m.index.normalize() == dt]
            common = grp_close.columns.intersection(grp_vol.columns)
            if len(common) < 1:
                continue
            ret = grp_close[common].pct_change().iloc[1:]
            vol = grp_vol[common].iloc[1:]
            if len(ret) < 5:
                continue
            results[dt] = ret.corrwith(vol, method="pearson")
        if not results:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(results).T
        daily.index = pd.DatetimeIndex(daily.index)
        smoothed = daily.rolling(20, min_periods=5).mean()
        return smoothed.reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. vp_corr_intraday_eod — 尾盘量价相关快照
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("vp_corr_intraday_eod", category="intraday_advanced")
class VpCorrIntradayEod(Factor):
    """分钟收益-量滚动相关性的尾盘快照.

    日内分钟频率 10 窗口滚动 corr(ret, vol)，取收盘前最后一个有效值.

    IC 方向: 负向.
    """
    name = "vp_corr_intraday_eod"
    category = "intraday_advanced"
    frequency = "daily"
    description = "分钟收益成交量尾盘相关性 (滚动10min取eod)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel or "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close_1m, vol_1m = panel["close"], panel["volume"]
        ret_1m = close_1m.pct_change()
        roll_window = 10
        results: dict = {}
        for col in close_1m.columns.intersection(vol_1m.columns):
            r = ret_1m[col].dropna()
            v = vol_1m[col].dropna()
            common_idx = r.index.intersection(v.index)
            if len(common_idx) < roll_window + 5:
                continue
            r, v = r.loc[common_idx], v.loc[common_idx]
            roll_corr = r.rolling(roll_window, min_periods=5).corr(v)
            eod = roll_corr.groupby(roll_corr.index.normalize()).last()
            eod.name = col
            results[col] = eod
        if not results:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(results)
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. intraday_trend_efficiency — 日内价格运动效率（趋势占比）
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_trend_efficiency_20d", category="intraday_advanced")
class IntradayTrendEfficiency20d(Factor):
    """日内价格运动效率比 (趋势占比因子).

    = |P_close - P_open| / Σ|ΔP_i|
    衡量价格朝着单一方向运动的"效率".
    高效下跌→未来正收益 (恐慌反弹)；高效上涨→未来负收益 (情绪透支).
    IC 方向: 负向.
    """
    name = "intraday_trend_efficiency_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "日内价格运动效率比 (|位移|/总路程)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel or "open" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close_1m, open_1m = panel["close"], panel["open"]
        day = close_1m.index.normalize()
        results: dict = {}
        for dt, grp_close in close_1m.groupby(day):
            if dt not in open_1m.index.normalize():
                continue
            grp_open = open_1m.loc[open_1m.index.normalize() == dt]
            common = grp_close.columns.intersection(grp_open.columns)
            if len(common) == 0:
                continue
            displacement = (grp_close[common].iloc[-1] - grp_open[common].iloc[0]).abs()
            path_length = grp_close[common].diff().abs().sum()
            results[dt] = displacement / path_length.replace(0, np.nan)
        if not results:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(results).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. intraday_realised_skewness — 修正高频已实现偏度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_realised_skewness_20d", category="intraday_advanced")
class IntradayRealisedSkewness20d(Factor):
    """修正的高频已实现偏度.

    日度: 分钟收益偏度 → |截面 z-score| → roll20 均值+std.
    无分钟数据时回退到日度收益偏度.
    方向: 正向 (修正后).
    """
    name = "intraday_realised_skewness_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "修正高频已实现偏度 (|z-score(skew)| + roll20均值+std)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            daily_close = data.get("close", dates, universe)
            if daily_close is None or daily_close.empty:
                return pd.DataFrame(np.nan, index=dates, columns=universe)
            ret = daily_close.pct_change()
            skew = ret.rolling(20, min_periods=10).skew()
            return _roll20_mean_std(_mean_distance(skew)).reindex(index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        results: dict = {}
        for dt, grp in ret_1m.groupby(day):
            if len(grp) >= 10:
                results[dt] = grp.skew()
        if not results:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(results).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll20_mean_std(_mean_distance(daily)).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. intraday_cvar — 日内条件在险价值 (成交量加权)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_cvar_20d", category="intraday_advanced")
class IntradayCVaR20d(Factor):
    """日内成交量加权 CVaR.

    成交量加权分钟收益 → 均值距离化 → roll20 均值+std.
    低波动股票的 CVaR 更小, 预期未来收益更高 (低波动异象).
    方向: 负向.
    """
    name = "intraday_cvar_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "日内成交量加权条件在险价值 (CVaR)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel or "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        vol_1m = panel["volume"]
        day = ret_1m.index.normalize()
        cvar_results: dict = {}
        for dt in sorted(set(day)):
            grp_ret = ret_1m.loc[day == dt]
            grp_vol = vol_1m.loc[vol_1m.index.normalize() == dt]
            common = grp_ret.columns.intersection(grp_vol.columns)
            if len(common) == 0 or len(grp_ret) < 20:
                continue
            vals = {}
            for col in common:
                r = grp_ret[col].dropna()
                v = grp_vol[col].dropna()
                idx = r.index.intersection(v.index)
                if len(idx) < 10:
                    continue
                r, v = r.loc[idx], v.loc[idx]
                vwar = (r * v).sum() / v.sum()
                vals[col] = vwar
            if vals:
                cvar_results[dt] = pd.Series(vals)
        if not cvar_results:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(cvar_results).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll20_mean_std(_mean_distance(daily)).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. intraday_overconfidence — 过度自信因子 (CP_Intraday)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_overconfidence_20d", category="intraday_advanced")
class IntradayOverconfidence20d(Factor):
    """过度自信因子.

    CP_Intraday = 下跌极端分钟序号中位数 - 上涨极端分钟序号中位数.
    极值阈值为 μ±σ.
    roll20 均值升序排名 + roll20 std 降序排名 → 相加.
    方向: 负向.
    """
    name = "intraday_overconfidence_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "过度自信因子 (CP_Intraday): 日内极端涨跌时序错位"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        cp_results: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 100:
                continue
            mu, sigma = grp.mean(), grp.std(ddof=0)
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 100:
                    continue
                up_idx = np.where(r.values > (mu[col] + sigma[col]))[0]
                dn_idx = np.where(r.values < (mu[col] - sigma[col]))[0]
                up_med = np.median(up_idx) if len(up_idx) else np.nan
                dn_med = np.median(dn_idx) if len(dn_idx) else np.nan
                if not np.isnan(up_med) and not np.isnan(dn_med):
                    vals[col] = dn_med - up_med
            if vals:
                cp_results[dt] = pd.Series(vals)
        if not cp_results:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(cp_results).T
        daily.index = pd.DatetimeIndex(daily.index)
        roll_mean = daily.rolling(20, min_periods=5).mean()
        roll_std = daily.rolling(20, min_periods=5).std(ddof=0)
        rank_mean = roll_mean.rank(axis=1, ascending=True, pct=True)
        rank_std = roll_std.rank(axis=1, ascending=False, pct=True)
        return (rank_mean + rank_std).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. intraday_herding — 羊群效应因子
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_herding_20d", category="intraday_advanced")
class IntradayHerding20d(Factor):
    """羊群效应因子.

    过去5日分钟成交量90%分位→趋势资金；之后5分钟最大量→极端跟随.
    因子 = mean(极端跟随量 / 趋势资金量), roll20 平滑.
    方向: 负向 (羊群越强, 未来收益越低).
    """
    name = "intraday_herding_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "羊群效应因子 (极端跟随量/趋势资金量)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        vol_1m = panel["volume"]
        day = vol_1m.index.normalize()
        herding: dict = {}
        for dt in sorted(set(day)):
            grp = vol_1m.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                v = grp[col].dropna()
                if len(v) < 30:
                    continue
                hist = v.iloc[-240 * 5:] if len(v) > 240 * 5 else v
                threshold = np.nanquantile(hist, 0.90)
                trend_idx = np.where(v.values > threshold)[0]
                if len(trend_idx) == 0:
                    continue
                ratios = [
                    v.iloc[min(ti + 5, len(v) - 1)] / v.iloc[ti]
                    for ti in trend_idx if ti + 5 < len(v)
                ]
                vals[col] = np.mean(ratios) if ratios else np.nan
            if vals:
                herding[dt] = pd.Series(vals)
        if not herding:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(herding).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. intraday_jump_intensity — 日内跳跃度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_jump_intensity_20d", category="intraday_advanced")
class IntradayJumpIntensity20d(Factor):
    """日内跳跃度: 简单收益 vs 对数收益的绝对差值.

    jump = |ret_simple - ret_log|, 差值越大→价格跳跃越剧烈→博彩型特征.
    方向: 负向.
    """
    name = "intraday_jump_intensity_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "日内跳跃强度 (|简单收益 - 对数收益|)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close_1m = panel["close"]
        ret_simple = close_1m.pct_change()
        ret_log = np.log(close_1m / close_1m.shift(1))
        jump = (ret_simple - ret_log).abs()
        day = jump.index.normalize()
        daily = pd.DataFrame({
            dt: jump.loc[day == dt].mean()
            for dt in sorted(set(day)) if len(jump.loc[day == dt]) > 10
        }).T
        if daily.empty:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. intraday_dtws — 跌幅时间重心偏移 (DTWS)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_dtws_20d", category="intraday_advanced")
class IntradayDTWS20d(Factor):
    """跌幅时间重心偏移因子.

    分钟负收益的时间加权均值 → 截面 |z-score| → roll20 均值+std.
    跌幅偏尾盘 → 卖出压力持续 → 负向信号.
    方向: 负向 (五年累计 IC=-50).
    """
    name = "intraday_dtws_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "跌幅时间重心偏移 (负收益时间加权位置)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        dtws: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 30:
                    continue
                tw = np.arange(1, len(r) + 1) / len(r)
                neg_mask = r < 0
                vals[col] = float(np.average(r[neg_mask], weights=tw[neg_mask])) if neg_mask.any() else 0.0
            if vals:
                dtws[dt] = pd.Series(vals)
        if not dtws:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(dtws).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll20_mean_std(_mean_distance(daily)).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. intraday_peak_ridge_ratio — 峰岭成交额比率
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_peak_ridge_ratio_20d", category="intraday_advanced")
class IntradayPeakRidgeRatio20d(Factor):
    """峰岭成交额比率因子.

    "峰": 孤立成交脉冲 (当前分钟 > μ+σ, 前后 < μ+σ).
    "岭": 持续放量区间 (连续 > μ).
    比率 = 滚动窗口峰成交额合计 / 岭成交额合计.
    方向: 正向 (峰值脉冲多→信息驱动强).
    """
    name = "intraday_peak_ridge_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "峰岭成交额比率 (孤立脉冲/持续放量)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "amount" in panel:
            amt = panel["amount"]
        elif "volume" in panel and "close" in panel:
            amt = panel["volume"] * panel["close"]
        else:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        day = amt.index.normalize()
        ratios: dict = {}
        for dt in sorted(set(day)):
            grp = amt.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                a = grp[col].dropna()
                if len(a) < 20:
                    continue
                mu, sigma = a.mean(), a.std(ddof=0)
                is_peak = (a > mu + sigma) & (a.shift(1) < mu + sigma) & (a.shift(-1) < mu + sigma)
                is_ridge = (a > mu) & ((a.shift(1) > mu) | (a.shift(-1) > mu))
                peak_sum = a[is_peak].sum()
                ridge_sum = a[is_ridge].sum()
                vals[col] = peak_sum / ridge_sum if ridge_sum > 0 else 0.0
            if vals:
                ratios[dt] = pd.Series(vals)
        if not ratios:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(ratios).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. intraday_blowup_position — 高低位放量因子
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_blowup_position_20d", category="intraday_advanced")
class IntradayBlowupPosition20d(Factor):
    """高低位放量因子.

    放量阈值 = μ_vol + 3σ_vol, 计算放量时均价在日内价格区间的相对位置.
    高位放量→出货, 低位放量→吸筹.
    方向: 负向 (五年累计 IC=-40).
    """
    name = "intraday_blowup_position_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "高低位放量 (异常放量的相对价格位置)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel or "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close_1m, vol_1m = panel["close"], panel["volume"]
        day = close_1m.index.normalize()
        positions: dict = {}
        for dt in sorted(set(day)):
            grp_c = close_1m.loc[day == dt]
            grp_v = vol_1m.loc[vol_1m.index.normalize() == dt]
            common = grp_c.columns.intersection(grp_v.columns)
            if len(common) == 0:
                continue
            vals = {}
            for col in common:
                c = grp_c[col].dropna()
                v = grp_v[col].dropna()
                idx = c.index.intersection(v.index)
                if len(idx) < 10:
                    continue
                c, v = c.loc[idx], v.loc[idx]
                threshold = v.mean() + 3 * v.std(ddof=0)
                blowup_idx = v > threshold
                if not blowup_idx.any():
                    continue
                blowup_close = c.loc[blowup_idx].mean()
                c_high, c_low = c.max(), c.min()
                vals[col] = ((blowup_close - c_low) / (c_high - c_low)) if c_high > c_low else 0.5
            if vals:
                positions[dt] = pd.Series(vals)
        if not positions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(positions).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll20_mean_std(_mean_distance(daily)).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. intraday_volume_vol — 高频成交量波动因子
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_volume_vol_20d", category="intraday_advanced")
class IntradayVolumeVol20d(Factor):
    """高频成交量波动因子.

    日成交量波动 = std(分钟成交量).
    因子 = std(日成交量波动_roll20) / mean(日成交量波动_roll20).
    即"波动的波动"/"波动的均值" (变异系数).
    方向: 负向.
    """
    name = "intraday_volume_vol_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "高频成交量波动 (日内量波动的变异系数)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        vol_1m = panel["volume"]
        day = vol_1m.index.normalize()
        daily_vol = pd.DataFrame({
            dt: vol_1m.loc[day == dt].std(ddof=0)
            for dt in sorted(set(day)) if len(vol_1m.loc[day == dt]) > 10
        }).T
        if daily_vol.empty:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily_vol.index = pd.DatetimeIndex(daily_vol.index)
        roll_std = daily_vol.rolling(20, min_periods=5).std(ddof=0)
        roll_mean = daily_vol.rolling(20, min_periods=5).mean()
        return (roll_std / roll_mean.replace(0, np.nan)).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 13. intraday_price_peak_count — 价峰分钟数因子
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_price_peak_count_20d", category="intraday_advanced")
class IntradayPricePeakCount20d(Factor):
    """价峰分钟数因子.

    识别孤立且无缺口的价格跳跃 (振幅 > μ+σ、前后非跳跃、前后区间重叠).
    统计每日满足条件的分钟数, roll20 求和.
    方向: 正向 (五年累计 IC=60).
    """
    name = "intraday_price_peak_count_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "价峰分钟数 (孤立无缺口价格跳跃计数)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        needed = {"high", "low", "close"}
        if not needed.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, low, close = panel["high"], panel["low"], panel["close"]
        amplitude = (high - low) / close.replace(0, np.nan)
        day = close.index.normalize()
        counts: dict = {}
        for dt in sorted(set(day)):
            grp_amp = amplitude.loc[day == dt]
            if len(grp_amp) < 10:
                continue
            vals = {}
            for col in grp_amp.columns:
                amp = grp_amp[col].dropna()
                if len(amp) < 10:
                    continue
                mu_a, sigma_a = amp.mean(), amp.std(ddof=0)
                is_jump = amp > (mu_a + sigma_a)
                cnt = 0
                for i in range(1, len(amp) - 1):
                    if not is_jump.iloc[i]:
                        continue
                    if is_jump.iloc[i - 1] and is_jump.iloc[i + 1]:
                        continue  # 连续跳跃排除
                    prev_high = high.loc[amp.index[i - 1], col] if col in high.columns else np.nan
                    prev_low = low.loc[amp.index[i - 1], col] if col in low.columns else np.nan
                    next_high = high.loc[amp.index[i + 1], col] if col in high.columns else np.nan
                    next_low = low.loc[amp.index[i + 1], col] if col in low.columns else np.nan
                    arr = np.array([prev_high, prev_low, next_high, next_low])
                    if not np.isnan(arr).any() and max(prev_low, next_low) <= min(prev_high, next_high):
                        cnt += 1
                vals[col] = cnt
            if vals:
                counts[dt] = pd.Series(vals)
        if not counts:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(counts).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=3).sum().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 14. intraday_torrent — 激流勇进因子
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_torrent_20d", category="intraday_advanced")
class IntradayTorrent20d(Factor):
    """激流勇进因子.

    5分钟邻域成交量判断放缩量 + 5分钟收益趋势判断涨跌.
    放量下跌时刻: 成交额占比 - 成交量占比 → 衡量买入强度.
    均值距离化 → roll20 平滑.
    方向: 负向 (五年累计 IC=-70).
    """
    name = "intraday_torrent_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "激流勇进 (放量下跌中的买入强度)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        available = set(panel.keys())
        if "close" not in available or "volume" not in available:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close_1m, vol_1m = panel["close"], panel["volume"]
        amt_1m = panel.get("amount", vol_1m * close_1m)
        day = close_1m.index.normalize()
        torrent: dict = {}
        for dt in sorted(set(day)):
            grp_c = close_1m.loc[day == dt]
            grp_v = vol_1m.loc[vol_1m.index.normalize() == dt]
            grp_a = amt_1m.loc[amt_1m.index.normalize() == dt]
            common_cols = grp_c.columns.intersection(grp_v.columns)
            if len(common_cols) == 0:
                continue
            vals = {}
            for col in common_cols:
                v = grp_v[col].dropna()
                c = grp_c[col].dropna()
                a = grp_a[col].dropna() if col in grp_a.columns else (v * c)
                idx = v.index.intersection(c.index)
                if len(idx) < 30:
                    continue
                v, c = v.loc[idx], c.loc[idx]
                a = a.loc[idx] if isinstance(a, pd.Series) else pd.Series(a.values[:len(idx)], index=idx)
                n5_vol = v.rolling(5).sum()
                is_fangliang = n5_vol > n5_vol.shift(1)
                ret_trend = c.pct_change(5).fillna(0)
                fangliang_diedie = is_fangliang & (ret_trend <= 0)
                if not fangliang_diedie.any():
                    continue
                amt_share = a.loc[fangliang_diedie].sum() / a.sum()
                vol_share = v.loc[fangliang_diedie].sum() / v.sum()
                vals[col] = amt_share - vol_share
            if vals:
                torrent[dt] = pd.Series(vals)
        if not torrent:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(torrent).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _mean_distance(daily).rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 15. intraday_drip_stone — 滴水穿石因子 (频谱分析)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_drip_stone_20d", category="intraday_advanced")
class IntradayDripStone20d(Factor):
    """滴水穿石因子 (成交量频谱分析).

    日内成交量 → IQR限幅 → Hann窗 → rFFT → 功率谱.
    统计 2-5 分钟频带能量占比. 高占比 = 机构分批吸筹.
    方向: 正向 (五年累计 IC>100).
    """
    name = "intraday_drip_stone_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "滴水穿石 (FFT频谱: 2-5分钟成交量能量占比)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        vol_1m = panel["volume"]
        day = vol_1m.index.normalize()
        band_ratios: dict = {}
        for dt in sorted(set(day)):
            grp = vol_1m.loc[day == dt]
            if len(grp) < 60:
                continue
            vals = {}
            for col in grp.columns:
                v = grp[col].dropna().values
                if len(v) < 60:
                    continue
                q25, q75 = np.nanpercentile(v, [25, 75])
                upper = q75 + 1.5 * (q75 - q25)
                v_clipped = np.clip(v, 0, upper)
                v_centered = v_clipped - np.mean(v_clipped)
                v_windowed = v_centered * np.hanning(len(v_centered))
                fft = np.fft.rfft(v_windowed)
                power = np.abs(fft) ** 2
                freqs = np.fft.rfftfreq(len(v_windowed), d=1.0)
                period = 1.0 / (freqs + 1e-12)
                mask_2_5 = (period >= 2) & (period <= 5)
                band_power = power[mask_2_5].sum()
                vals[col] = band_power / power.sum() if power.sum() > 0 else 0.0
            if vals:
                band_ratios[dt] = pd.Series(vals)
        if not band_ratios:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(band_ratios).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 16. intraday_volume_time_centroid — 成交量时间重心
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_volume_time_centroid_20d", category="intraday_advanced")
class IntradayVolumeTimeCentroid20d(Factor):
    """成交量时间重心因子.

    日内成交额的时间加权平均位置 (0=早盘, 1=尾盘).
    重心偏早盘 → 资金有计划介入 → 正向信号.
    方向: 正向 (取负重心使早盘集中=高值).
    """
    name = "intraday_volume_time_centroid_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "成交量时间重心 (成交额的时间加权位置, 早盘集中为正)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "amount" in panel:
            amt = panel["amount"]
        elif "volume" in panel and "close" in panel:
            amt = panel["volume"] * panel["close"]
        else:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        day = amt.index.normalize()
        centroids: dict = {}
        for dt in sorted(set(day)):
            grp = amt.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                a = grp[col].dropna().values
                if len(a) < 20:
                    continue
                tw = np.arange(1, len(a) + 1) / len(a)
                numerator = np.sum(tw * a)
                denominator = np.sum(a)
                vals[col] = float(numerator / denominator) if denominator > 0 else 0.5
            if vals:
                centroids[dt] = pd.Series(vals)
        if not centroids:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(centroids).T
        daily.index = pd.DatetimeIndex(daily.index)
        # 重心小=早盘放量=正向, 取负号使高值=正向
        return (-daily).rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 17. intraday_close_position — 收盘位置因子
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_close_position_20d", category="intraday_advanced")
class IntradayClosePosition20d(Factor):
    """收盘位置因子.

    收盘价在日内最高/最低区间中的相对位置 (0=最低, 1=最高).
    持续高位收盘 → 买方控盘能力稳定 → 正向信号.
    方向: 正向.
    """
    name = "intraday_close_position_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "收盘位置 (收盘在日内高低区间的相对位置)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"high", "low", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, low, close = panel["high"], panel["low"], panel["close"]
        day = close.index.normalize()
        positions: dict = {}
        for dt in sorted(set(day)):
            grp_high = high.loc[day == dt]
            grp_low = low.loc[day == dt]
            grp_close = close.loc[day == dt]
            if len(grp_close) < 20:
                continue
            vals = {}
            for col in grp_close.columns:
                h = grp_high[col].dropna()
                l = grp_low[col].dropna()
                c = grp_close[col].dropna()
                if len(c) < 20:
                    continue
                h_day, l_day = h.max(), l.min()
                c_end = c.iloc[-1]
                denom = h_day - l_day
                vals[col] = float((c_end - l_day) / denom) if denom > 1e-12 else 0.5
            if vals:
                positions[dt] = pd.Series(vals)
        if not positions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(positions).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll20_mean_std(_mean_distance(daily)).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 18. intraday_reversal_intensity — 日内反转强度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_reversal_intensity_20d", category="intraday_advanced")
class IntradayReversalIntensity20d(Factor):
    """日内反转强度因子.

    每分钟收益方向切换的频率 (0=全同向, 1=每分钟切换).
    频繁反转 → 多空分歧大、共识不足 → 负向信号.
    方向: 负向.
    """
    name = "intraday_reversal_intensity_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "日内反转强度 (分钟收益方向切换频率)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        intensities: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 20:
                    continue
                signs = np.sign(r)
                switches = int(np.sum(np.diff(signs) != 0))
                vals[col] = float(switches / (len(r) - 1)) if len(r) > 1 else 0.0
            if vals:
                intensities[dt] = pd.Series(vals)
        if not intensities:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(intensities).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 19. intraday_upper_lower_volume_ratio — 高位低位量比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_upper_lower_volume_ratio_20d", category="intraday_advanced")
class IntradayUpperLowerVolumeRatio20d(Factor):
    """高位/低位成交量比率因子.

    价格运行在日内中位价以上时的成交额 vs 中位价以下的成交额.
    高位放量 → 资金愿以较高成本跟进 → 正向信号.
    方向: 正向.
    """
    name = "intraday_upper_lower_volume_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "高位低位量比 (高位成交额/低位成交额)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        if "amount" in panel:
            amt = panel["amount"]
        elif "volume" in panel:
            amt = panel["volume"] * panel["close"]
        else:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        ratios: dict = {}
        for dt in sorted(set(day)):
            grp_close = close.loc[day == dt]
            grp_amt = amt.loc[day == dt]
            if len(grp_close) < 20:
                continue
            vals = {}
            for col in grp_close.columns:
                c = grp_close[col].dropna()
                a = grp_amt[col].dropna()
                common_idx = c.index.intersection(a.index)
                if len(common_idx) < 20:
                    continue
                c, a = c.loc[common_idx], a.loc[common_idx]
                mid = (c.max() + c.min()) / 2.0
                upper_mask = c > mid
                lower_mask = c < mid
                upper_vol = a[upper_mask].sum()
                lower_vol = a[lower_mask].sum()
                vals[col] = float(upper_vol / lower_vol) if lower_vol > 1e-12 else 1.0
            if vals:
                ratios[dt] = pd.Series(vals)
        if not ratios:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(ratios).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 20. intraday_early_late_divergence — 早晚盘背离度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_early_late_divergence_20d", category="intraday_advanced")
class IntradayEarlyLateDivergence20d(Factor):
    """早晚盘背离度因子.

    前半段与后半段价格走势的背离程度.
    divergence = -sign(早盘收益) × 尾盘收益.
    早盘涨尾盘跌 (冲高回落) → 背离大 → 主力诱多出货 → 负向信号.
    方向: 负向.
    """
    name = "intraday_early_late_divergence_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "早晚盘背离度 (早盘与尾盘走势一致性, 背离=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        divergences: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 20:
                continue
            mid = len(grp) // 2
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < 20:
                    continue
                early = c.iloc[:mid]
                late = c.iloc[mid:]
                if len(early) < 2 or len(late) < 2:
                    continue
                ret_early = early.iloc[-1] / early.iloc[0] - 1.0
                ret_late = late.iloc[-1] / late.iloc[0] - 1.0
                # 早涨尾跌 → 正背离值 → 负向信号
                vals[col] = float(-np.sign(ret_early) * ret_late)
            if vals:
                divergences[dt] = pd.Series(vals)
        if not divergences:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(divergences).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 21. intraday_volume_dispersion — 成交量离散度 (基尼系数)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_volume_dispersion_20d", category="intraday_advanced")
class IntradayVolumeDispersion20d(Factor):
    """成交量基尼系数因子.

    日内成交量在各分钟之间的分布不均匀程度 (基尼系数).
    高基尼 → 成交量高度集中在特定时段 → 大资金择时入场 → 正向信号.
    低基尼 → 成交量均匀分布 → 散户随机交易主导.
    方向: 正向.
    """
    name = "intraday_volume_dispersion_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "成交量基尼系数 (日内量分布不均匀度)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        vol = panel["volume"]
        day = vol.index.normalize()
        ginis: dict = {}
        for dt in sorted(set(day)):
            grp = vol.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                v = grp[col].dropna().values
                if len(v) < 20:
                    continue
                v_clean = v[v > 0]
                if len(v_clean) < 10:
                    continue
                v_sorted = np.sort(v_clean)
                n = len(v_sorted)
                v_sum = np.sum(v_sorted)
                if v_sum < 1e-12:
                    continue
                gini = float(2.0 * np.sum(np.arange(1, n + 1) * v_sorted) / (n * v_sum) - (n + 1.0) / n)
                vals[col] = max(0.0, min(1.0, gini))
            if vals:
                ginis[dt] = pd.Series(vals)
        if not ginis:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(ginis).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 22. intraday_open_vp_corr — 开盘量价相关性
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_open_vp_corr_20d", category="intraday_advanced")
class IntradayOpenVpCorr20d(Factor):
    """开盘阶段量价相关性因子.

    前30分钟内, 每分钟成交量与绝对收益率的 Spearman 相关.
    开盘量价正相关 → 资金通过放量推动价格 → 知情交易 → 正向信号.
    方向: 正向.
    """
    name = "intraday_open_vp_corr_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "开盘量价相关 (前30分钟量×|收益|相关性)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, volume = panel["close"], panel["volume"]
        ret_1m = close.pct_change().abs()
        day = ret_1m.index.normalize()
        corrs: dict = {}
        for dt in sorted(set(day)):
            grp_ret = ret_1m.loc[day == dt]
            grp_vol = volume.loc[day == dt]
            if len(grp_ret) < 30:
                continue
            n_open = max(10, min(30, len(grp_ret) // 3))
            vals = {}
            for col in grp_ret.columns:
                r = grp_ret[col].dropna().iloc[:n_open]
                v = grp_vol[col].dropna()
                common = r.index.intersection(v.index)
                if len(common) < 10:
                    continue
                r_open = r.loc[common]
                v_open = v.loc[common]
                if r_open.std() < 1e-12 or v_open.std() < 1e-12:
                    vals[col] = 0.0
                else:
                    corr = r_open.corr(v_open, method="spearman")
                    vals[col] = float(corr) if not np.isnan(corr) else 0.0
            if vals:
                corrs[dt] = pd.Series(vals)
        if not corrs:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(corrs).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 23. intraday_open_close_volume_ratio — 开盘/尾盘成交量比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_open_close_volume_ratio_20d", category="intraday_advanced")
class IntradayOpenCloseVolumeRatio20d(Factor):
    """开盘/尾盘成交量比率因子.

    前30分钟成交额 / 最后30分钟成交额.
    开盘放量大于尾盘放量 → 机构隔夜研究后集中执行 → 正向信号.
    方向: 正向.
    """
    name = "intraday_open_close_volume_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "开盘尾盘量比 (前30分/后30分成交额)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "amount" in panel:
            amt = panel["amount"]
        elif "volume" in panel and "close" in panel:
            amt = panel["volume"] * panel["close"]
        else:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        day = amt.index.normalize()
        ratios: dict = {}
        for dt in sorted(set(day)):
            grp = amt.loc[day == dt]
            n = len(grp)
            if n < 30:
                continue
            n_window = max(5, min(30, n // 4))
            vals = {}
            for col in grp.columns:
                a = grp[col].dropna()
                if len(a) < 30:
                    continue
                open_vol = a.iloc[:n_window].sum()
                close_vol = a.iloc[-n_window:].sum()
                vals[col] = float(open_vol / close_vol) if close_vol > 1e-12 else 1.0
            if vals:
                ratios[dt] = pd.Series(vals)
        if not ratios:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(ratios).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 24. intraday_amplitude_volume_corr — 振幅-成交量分钟级相关
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_amplitude_volume_corr_20d", category="intraday_advanced")
class IntradayAmplitudeVolumeCorr20d(Factor):
    """振幅-成交量相关性因子.

    每分钟振幅 (high-low)/close 与成交量的 Spearman 相关.
    振幅与量正相关 → 价格波动由真实资金驱动 (非噪声) → 正向信号.
    无资金支撑的波动 → 噪声主导 → 相关性弱.
    方向: 正向.
    """
    name = "intraday_amplitude_volume_corr_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "振幅量相关 (分钟振幅×成交量 Spearman r)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"high", "low", "close", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, low, close, volume = panel["high"], panel["low"], panel["close"], panel["volume"]
        amplitude = (high - low) / close.replace(0, np.nan)
        day = amplitude.index.normalize()
        corrs: dict = {}
        for dt in sorted(set(day)):
            grp_amp = amplitude.loc[day == dt]
            grp_vol = volume.loc[day == dt]
            if len(grp_amp) < 20:
                continue
            vals = {}
            for col in grp_amp.columns:
                amp = grp_amp[col].dropna()
                vol = grp_vol[col].dropna()
                common = amp.index.intersection(vol.index)
                if len(common) < 20:
                    continue
                a = amp.loc[common]
                v = vol.loc[common]
                if a.std() < 1e-12 or v.std() < 1e-12:
                    vals[col] = 0.0
                else:
                    corr = a.corr(v, method="spearman")
                    vals[col] = float(corr) if not np.isnan(corr) else 0.0
            if vals:
                corrs[dt] = pd.Series(vals)
        if not corrs:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(corrs).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 25. intraday_ret_vol_coupling — 收益-成交量耦合度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_ret_vol_coupling_20d", category="intraday_advanced")
class IntradayRetVolCoupling20d(Factor):
    """收益-成交量耦合度因子.

    放量分钟 (量 > 日均量) 的收益方向一致性.
    计算放量分钟中同向收益占比:  max(涨占比, 跌占比).
    耦合度高 → 放量时有明确方向 → 主力控盘有序 → 正向信号.
    方向: 正向.
    """
    name = "intraday_ret_vol_coupling_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "收益量耦合度 (放量分钟方向一致性)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, volume = panel["close"], panel["volume"]
        ret_1m = close.pct_change()
        day = ret_1m.index.normalize()
        couplings: dict = {}
        for dt in sorted(set(day)):
            grp_ret = ret_1m.loc[day == dt]
            grp_vol = volume.loc[day == dt]
            if len(grp_ret) < 20:
                continue
            vals = {}
            for col in grp_ret.columns:
                r = grp_ret[col].dropna()
                v = grp_vol[col].dropna()
                common = r.index.intersection(v.index)
                if len(common) < 20:
                    continue
                r_c = r.loc[common]
                v_c = v.loc[common]
                v_mean = v_c.mean()
                if v_mean < 1e-12:
                    continue
                high_vol_mask = v_c > v_mean
                if high_vol_mask.sum() < 3:
                    continue
                r_high = r_c[high_vol_mask]
                up_ratio = (r_high > 0).sum() / len(r_high)
                down_ratio = (r_high < 0).sum() / len(r_high)
                vals[col] = float(max(up_ratio, down_ratio))
            if vals:
                couplings[dt] = pd.Series(vals)
        if not couplings:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(couplings).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 26. intraday_price_volume_elasticity — 价量弹性
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_price_volume_elasticity_20d", category="intraday_advanced")
class IntradayPriceVolumeElasticity20d(Factor):
    """价量弹性因子.

    |分钟收益| / 分钟成交量 (归一化后) 的日内均值.
    弹性高 → 少量成交即推动价格大幅变化 → 流动性差/卖盘枯竭 → 负向信号.
    弹性低 → 价格稳定、流动性充裕 → 正向信号.
    方向: 负向.
    """
    name = "intraday_price_volume_elasticity_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "价量弹性 (|收益|/成交量, 弹性高=流动性差=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, volume = panel["close"], panel["volume"]
        ret_1m = close.pct_change().abs()
        day = ret_1m.index.normalize()
        elasticities: dict = {}
        for dt in sorted(set(day)):
            grp_ret = ret_1m.loc[day == dt]
            grp_vol = volume.loc[day == dt]
            if len(grp_ret) < 20:
                continue
            vals = {}
            for col in grp_ret.columns:
                r = grp_ret[col].dropna()
                v = grp_vol[col].dropna()
                common = r.index.intersection(v.index)
                if len(common) < 20:
                    continue
                r_abs = r.loc[common]
                v_c = v.loc[common]
                v_mean = v_c.mean()
                if v_mean < 1e-12:
                    continue
                v_norm = v_c / v_mean
                elasticity = r_abs / (v_norm + 1e-12)
                vals[col] = float(elasticity.mean())
            if vals:
                elasticities[dt] = pd.Series(vals)
        if not elasticities:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(elasticities).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 27. intraday_open_gap_persistence — 开盘缺口持续性
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_open_gap_persistence_20d", category="intraday_advanced")
class IntradayOpenGapPersistence20d(Factor):
    """开盘缺口持续性因子.

    开盘跳空 (今开 vs 昨收) 方向与日内收益方向的一致性.
    同向 → 跳空方向被日内确认 → 共识强 → 正向信号.
    反向 (跳空被回补) → 开盘定价错误 → 负向信号.
    方向: 正向.
    """
    name = "intraday_open_gap_persistence_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "开盘缺口持续性 (跳空方向与日内方向一致性)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"open", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        open_px, close_px = panel["open"], panel["close"]
        day = close_px.index.normalize()
        persistence: dict = {}
        prev_close: dict = {}
        for dt in sorted(set(day)):
            grp_open = open_px.loc[day == dt]
            grp_close = close_px.loc[day == dt]
            if len(grp_close) < 20:
                continue
            vals = {}
            for col in grp_close.columns:
                o = grp_open[col].dropna()
                c = grp_close[col].dropna()
                if len(o) < 1 or len(c) < 5:
                    continue
                o_first = o.iloc[0]
                c_last = c.iloc[-1]
                prev_c = prev_close.get(col)
                if prev_c is None or prev_c < 1e-12:
                    prev_close[col] = c_last
                    continue
                gap_dir = np.sign(o_first - prev_c)
                intraday_dir = np.sign(c_last - o_first)
                if abs(gap_dir) < 1e-12 or abs(intraday_dir) < 1e-12:
                    vals[col] = 0.5
                else:
                    vals[col] = 1.0 if gap_dir == intraday_dir else 0.0
                prev_close[col] = c_last
            if vals:
                persistence[dt] = pd.Series(vals)
        if not persistence:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(persistence).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 28. intraday_overnight_absorption — 隔夜信息吸收效率
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_overnight_absorption_20d", category="intraday_advanced")
class IntradayOvernightAbsorption20d(Factor):
    """隔夜信息吸收效率因子.

    日内振幅 / |隔夜跳空| (跳空=今开/昨收-1).
    比值大 → 跳空虽大但日内充分消化 → 信息效率高 → 正向信号.
    比值小 → 隔夜跳空大但日内波动小 → 信息未充分吸收 → 负向信号.
    方向: 正向.
    """
    name = "intraday_overnight_absorption_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "隔夜信息吸收率 (日内振幅/|隔夜跳空|)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"open", "high", "low", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        open_px, high, low, close = panel["open"], panel["high"], panel["low"], panel["close"]
        day = close.index.normalize()
        absorptions: dict = {}
        prev_close: dict = {}
        for dt in sorted(set(day)):
            grp_open = open_px.loc[day == dt]
            grp_high = high.loc[day == dt]
            grp_low = low.loc[day == dt]
            grp_close = close.loc[day == dt]
            if len(grp_close) < 20:
                continue
            vals = {}
            for col in grp_close.columns:
                o = grp_open[col].dropna()
                h = grp_high[col].dropna()
                l = grp_low[col].dropna()
                c = grp_close[col].dropna()
                if len(o) < 1 or len(h) < 5:
                    continue
                o_first = o.iloc[0]
                h_day, l_day = h.max(), l.min()
                intraday_range = h_day - l_day
                prev_c = prev_close.get(col)
                if prev_c is None or prev_c < 1e-12:
                    prev_close[col] = c.iloc[-1]
                    continue
                overnight_gap = abs(o_first / prev_c - 1.0)
                if overnight_gap < 1e-8:
                    vals[col] = 1.0  # 无跳空=无信息需消化
                else:
                    vals[col] = float(intraday_range / (o_first * overnight_gap + 1e-12))
                prev_close[col] = c.iloc[-1]
            if vals:
                absorptions[dt] = pd.Series(vals)
        if not absorptions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(absorptions).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 29. intraday_up_down_volume_asymmetry — 涨跌量不对称
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_up_down_volume_asymmetry_20d", category="intraday_advanced")
class IntradayUpDownVolumeAsymmetry20d(Factor):
    """涨跌量不对称因子.

    上涨分钟平均成交量 / 下跌分钟平均成交量.
    不对称 > 1 → 涨时放量 > 跌时放量 → 买方主导 → 正向信号.
    不对称 < 1 → 跌时放量 > 涨时放量 → 卖方主导 → 负向信号.
    方向: 正向.
    """
    name = "intraday_up_down_volume_asymmetry_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "涨跌量不对称 (涨分钟均量/跌分钟均量)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, volume = panel["close"], panel["volume"]
        ret_1m = close.pct_change()
        day = ret_1m.index.normalize()
        asymmetries: dict = {}
        for dt in sorted(set(day)):
            grp_ret = ret_1m.loc[day == dt]
            grp_vol = volume.loc[day == dt]
            if len(grp_ret) < 20:
                continue
            vals = {}
            for col in grp_ret.columns:
                r = grp_ret[col].dropna()
                v = grp_vol[col].dropna()
                common = r.index.intersection(v.index)
                if len(common) < 20:
                    continue
                r_c = r.loc[common]
                v_c = v.loc[common]
                up_mask = r_c > 0
                down_mask = r_c < 0
                up_vol = v_c[up_mask].mean() if up_mask.any() else 0.0
                down_vol = v_c[down_mask].mean() if down_mask.any() else 0.0
                vals[col] = float(up_vol / down_vol) if down_vol > 1e-12 else (2.0 if up_vol > 1e-12 else 1.0)
            if vals:
                asymmetries[dt] = pd.Series(vals)
        if not asymmetries:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(asymmetries).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 30. intraday_vwap_deviation — VWAP偏离度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_vwap_deviation_20d", category="intraday_advanced")
class IntradayVwapDeviation20d(Factor):
    """VWAP偏离度因子.

    每分钟价格与日内VWAP的标准差.
    VWAP = Σ(close × volume) / Σ(volume).
    偏离大 → 价格围绕公允均价大幅摆动 → 情绪化交易 → 负向信号.
    偏离小 → 价格稳定在VWAP附近 → 交易有序 → 正向信号.
    方向: 负向.
    """
    name = "intraday_vwap_deviation_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "VWAP偏离度 (价格与成交量加权均价的标准差)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, volume = panel["close"], panel["volume"]
        day = close.index.normalize()
        deviations: dict = {}
        for dt in sorted(set(day)):
            grp_close = close.loc[day == dt]
            grp_vol = volume.loc[day == dt]
            if len(grp_close) < 20:
                continue
            vals = {}
            for col in grp_close.columns:
                c = grp_close[col].dropna()
                v = grp_vol[col].dropna()
                common = c.index.intersection(v.index)
                if len(common) < 20:
                    continue
                c_c = c.loc[common]
                v_c = v.loc[common]
                v_sum = v_c.sum()
                if v_sum < 1e-12:
                    continue
                vwap = float((c_c * v_c).sum() / v_sum)
                deviations_from_vwap = c_c - vwap
                vals[col] = float(deviations_from_vwap.std(ddof=0))
            if vals:
                deviations[dt] = pd.Series(vals)
        if not deviations:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(deviations).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll20_mean_std(_mean_distance(daily)).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 31. intraday_hurst — 日内赫斯特指数
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_hurst_20d", category="intraday_advanced")
class IntradayHurst20d(Factor):
    """日内赫斯特指数因子.

    用重标极差 (R/S) 法估计日内价格序列的 Hurst 指数.
    H > 0.5 → 趋势持续性强 → 动量效应 → 正向信号.
    H < 0.5 → 均值回归主导 → 反向信号.
    方向: 正向.
    """
    name = "intraday_hurst_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "赫斯特指数 (R/S法, 趋势持续性)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    @staticmethod
    def _hurst_rs(series: np.ndarray) -> float:
        """R/S Hurst estimator on a 1-D series (simplified)."""
        n = len(series)
        if n < 30:
            return 0.5
        # use 3 lag scales
        lags = [max(5, n // 8), max(8, n // 4), max(15, n // 2)]
        rs_values = []
        for lag in lags:
            if lag < 5 or lag > n - 1:
                continue
            segments = n // lag
            if segments < 2:
                continue
            rs_seg = []
            for i in range(segments):
                seg = series[i * lag:(i + 1) * lag]
                mean = seg.mean()
                deviate = seg - mean
                cum_dev = np.cumsum(deviate)
                r_val = cum_dev.max() - cum_dev.min()
                s_val = seg.std(ddof=0)
                if s_val > 1e-12:
                    rs_seg.append(r_val / s_val)
            if rs_seg:
                rs_values.append(np.mean(rs_seg))
        if len(rs_values) < 3:
            return 0.5
        log_lags = np.log(lags[:len(rs_values)])
        log_rs = np.log(rs_values)
        slope = float(np.polyfit(log_lags, log_rs, 1)[0])
        return max(0.0, min(1.0, slope))

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        hursts: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna().values
                if len(c) < 30:
                    continue
                vals[col] = self._hurst_rs(c)
            if vals:
                hursts[dt] = pd.Series(vals)
        if not hursts:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(hursts).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 32. intraday_cs_spread — Corwin-Schultz 价差代理
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_cs_spread_20d", category="intraday_advanced")
class IntradayCSSpread20d(Factor):
    """Corwin-Schultz 价差代理因子.

    基于日内 high-low 的买卖价差估计 (Corwin & Schultz 2012 简化版).
    高低波动与真实价差的关系推导出有效spread.
    高 spread → 流动性差、交易成本高 → 负向信号.
    方向: 负向.
    """
    name = "intraday_cs_spread_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "CS价差 (日内HL买卖价差估计)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"high", "low"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, low = panel["high"], panel["low"]
        day = high.index.normalize()
        spreads: dict = {}
        for dt in sorted(set(day)):
            grp_high = high.loc[day == dt]
            grp_low = low.loc[day == dt]
            if len(grp_high) < 30:
                continue
            vals = {}
            for col in grp_high.columns:
                h = grp_high[col].dropna()
                l = grp_low[col].dropna()
                common = h.index.intersection(l.index)
                if len(common) < 30:
                    continue
                h_c = h.loc[common]
                l_c = l.loc[common]
                log_hl = np.log(h_c / l_c.replace(0, np.nan))
                log_hl = log_hl.replace([np.inf, -np.inf], np.nan).dropna()
                if len(log_hl) < 20:
                    continue
                beta = float((log_hl ** 2).mean())
                # two-day combined HL: split into first/second half
                mid = len(log_hl) // 2
                if mid < 5:
                    continue
                half1 = log_hl.iloc[:mid]
                half2 = log_hl.iloc[mid:mid * 2]
                if len(half1) < 5 or len(half2) < 5:
                    continue
                gamma = float(((half1.sum() + half2.sum()) / 2) ** 2)
                if beta < 1e-12:
                    continue
                alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / (3.0 - 2.0 * np.sqrt(2.0))
                alpha -= np.sqrt(gamma / (3.0 - 2.0 * np.sqrt(2.0)))
                alpha = max(0.0, alpha)
                spread_val = float(2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha)))
                vals[col] = spread_val
            if vals:
                spreads[dt] = pd.Series(vals)
        if not spreads:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(spreads).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 33. intraday_tail_acceleration — 尾盘加速度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_tail_acceleration_20d", category="intraday_advanced")
class IntradayTailAcceleration20d(Factor):
    """尾盘加速度因子.

    尾盘最后N分钟的收益二阶变化 (加速度).
    ret_early_tail = close[-N:-N/2] 收益, ret_late_tail = close[-N/2:] 收益.
    acceleration = ret_late_tail - ret_early_tail.
    正向加速度 → 尾盘加速上涨 → 资金抢筹 → 正向信号.
    负向加速度 → 尾盘加速下跌 → 恐慌抛售 → 负向信号.
    方向: 正向.
    """
    name = "intraday_tail_acceleration_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "尾盘加速度 (最后N分钟收益的二阶变化)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        accelerations: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            n_total = len(grp)
            if n_total < 30:
                continue
            n_tail = max(10, n_total // 5)  # 最后20%的分钟数
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < 30:
                    continue
                tail = c.iloc[-n_tail:]
                mid_tail = len(tail) // 2
                if mid_tail < 3:
                    continue
                ret_early_tail = tail.iloc[mid_tail] / tail.iloc[0] - 1.0
                ret_late_tail = tail.iloc[-1] / tail.iloc[mid_tail] - 1.0
                vals[col] = float(ret_late_tail - ret_early_tail)
            if vals:
                accelerations[dt] = pd.Series(vals)
        if not accelerations:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(accelerations).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 34. intraday_volatility_smile — 波动率微笑
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_volatility_smile_20d", category="intraday_advanced")
class IntradayVolatilitySmile20d(Factor):
    """波动率微笑因子.

    日内波动率的 U 型结构强度: (早盘vol + 尾盘vol) / (2 × 午盘vol).
    U型明显 → 开盘与尾盘博弈激烈、午盘相对冷静 → 正向信号.
    平坦 → 日内波动均匀 → 缺乏定价博弈焦点.
    方向: 正向.
    """
    name = "intraday_volatility_smile_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "波动率微笑 (U型结构: (早+尾)/(2×午))"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        smiles: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            n = len(grp)
            if n < 30:
                continue
            third = max(8, n // 3)
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 30:
                    continue
                early = r.iloc[:third]
                midday = r.iloc[third:2 * third]
                late = r.iloc[2 * third:]
                if len(early) < 5 or len(midday) < 5 or len(late) < 5:
                    continue
                vol_early = early.std(ddof=0)
                vol_mid = midday.std(ddof=0)
                vol_late = late.std(ddof=0)
                if vol_mid < 1e-12:
                    vals[col] = 1.0
                else:
                    vals[col] = float((vol_early + vol_late) / (2.0 * vol_mid))
            if vals:
                smiles[dt] = pd.Series(vals)
        if not smiles:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(smiles).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 35. intraday_opening_range_breakout — 开盘区间突破
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_opening_range_breakout_20d", category="intraday_advanced")
class IntradayOpeningRangeBreakout20d(Factor):
    """开盘区间突破因子.

    前N分钟形成开盘区间 [OR_low, OR_high].
    日内是否突破该区间: 突破上轨=+1, 突破下轨=-1, 未突破=0.
    向上突破 → 开盘区间上破 → 强势信号 → 正向.
    方向: 正向.
    """
    name = "intraday_opening_range_breakout_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "开盘区间突破 (前N分钟区间突破方向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"high", "low", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, low, close = panel["high"], panel["low"], panel["close"]
        day = close.index.normalize()
        breakouts: dict = {}
        for dt in sorted(set(day)):
            grp_high = high.loc[day == dt]
            grp_low = low.loc[day == dt]
            grp_close = close.loc[day == dt]
            n = len(grp_close)
            if n < 30:
                continue
            n_or = max(10, n // 4)  # 前25%为开盘区间
            vals = {}
            for col in grp_close.columns:
                h = grp_high[col].dropna()
                l = grp_low[col].dropna()
                c = grp_close[col].dropna()
                if len(h) < n_or or len(c) < 20:
                    continue
                or_high = h.iloc[:n_or].max()
                or_low = l.iloc[:n_or].min()
                c_after = c.iloc[n_or:]
                if len(c_after) < 5:
                    continue
                if or_high - or_low < 1e-12:
                    vals[col] = 0.0
                else:
                    # 收盘在区间上方=+1, 下方=-1, 中间=0
                    c_end = c.iloc[-1]
                    if c_end > or_high:
                        vals[col] = 1.0
                    elif c_end < or_low:
                        vals[col] = -1.0
                    else:
                        vals[col] = 0.0
            if vals:
                breakouts[dt] = pd.Series(vals)
        if not breakouts:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(breakouts).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 36. intraday_large_order_impact — 大单冲击方向
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_large_order_impact_20d", category="intraday_advanced")
class IntradayLargeOrderImpact20d(Factor):
    """大单冲击方向因子.

    检测成交量突变分钟 (vol > μ+2σ), 计算突变后5分钟的价格变化方向.
    量突变后价格上涨 → 大买单冲击 → 正向信号.
    量突变后价格下跌 → 大卖单冲击 → 负向信号.
    方向: 正向 (大买单净冲击).
    """
    name = "intraday_large_order_impact_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "大单冲击 (成交量突变后价格走向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, volume = panel["close"], panel["volume"]
        ret_1m = close.pct_change()
        day = ret_1m.index.normalize()
        impacts: dict = {}
        for dt in sorted(set(day)):
            grp_ret = ret_1m.loc[day == dt]
            grp_vol = volume.loc[day == dt]
            if len(grp_ret) < 30:
                continue
            vals = {}
            for col in grp_ret.columns:
                r = grp_ret[col].dropna()
                v = grp_vol[col].dropna()
                common = r.index.intersection(v.index)
                if len(common) < 30:
                    continue
                r_c = r.loc[common]
                v_c = v.loc[common]
                mu_v = v_c.mean()
                sigma_v = v_c.std(ddof=0)
                if sigma_v < 1e-12:
                    continue
                spike_mask = v_c > (mu_v + 2.0 * sigma_v)
                n_spikes = spike_mask.sum()
                if n_spikes < 1:
                    vals[col] = 0.0
                    continue
                spike_indices = np.where(spike_mask.values)[0]
                impacts_list = []
                for idx in spike_indices:
                    look_forward = min(5, len(r_c) - idx - 1)
                    if look_forward < 1:
                        continue
                    cum_ret = r_c.iloc[idx + 1:idx + 1 + look_forward].sum()
                    impacts_list.append(np.sign(cum_ret))
                vals[col] = float(np.mean(impacts_list)) if impacts_list else 0.0
            if vals:
                impacts[dt] = pd.Series(vals)
        if not impacts:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(impacts).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 37. intraday_volume_price_entropy — 量价联合信息熵
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_volume_price_entropy_20d", category="intraday_advanced")
class IntradayVolumePriceEntropy20d(Factor):
    """量价联合信息熵因子.

    将(分钟收益, 分钟成交量)离散化为2D直方图, 计算信息熵.
    高熵 → 量价关系复杂/无规律 → 噪声主导 → 负向信号.
    低熵 → 量价关系有序/可预测 → 信息主导 → 正向信号.
    方向: 正向 (取负熵使低熵=高值).
    """
    name = "intraday_volume_price_entropy_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "量价联合熵 (二维直方图信息熵, 负号使低熵=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, volume = panel["close"], panel["volume"]
        ret_1m = close.pct_change()
        day = ret_1m.index.normalize()
        entropies: dict = {}
        for dt in sorted(set(day)):
            grp_ret = ret_1m.loc[day == dt]
            grp_vol = volume.loc[day == dt]
            if len(grp_ret) < 30:
                continue
            vals = {}
            for col in grp_ret.columns:
                r = grp_ret[col].dropna()
                v = grp_vol[col].dropna()
                common = r.index.intersection(v.index)
                if len(common) < 30:
                    continue
                r_c = r.loc[common]
                v_c = v.loc[common]
                # discretize into 5×5 bins
                try:
                    r_disc = pd.cut(r_c, bins=5, labels=False)
                    v_disc = pd.cut(v_c, bins=5, labels=False)
                except (ValueError, TypeError):
                    continue
                valid = r_disc.notna() & v_disc.notna()
                if valid.sum() < 20:
                    continue
                hist2d, _, _ = np.histogram2d(
                    r_disc[valid].values, v_disc[valid].values, bins=5, range=[[0, 4], [0, 4]])
                prob = hist2d / hist2d.sum()
                prob = prob[prob > 0]
                entropy = float(-np.sum(prob * np.log(prob)))
                vals[col] = -entropy  # 低熵=高值=正向
            if vals:
                entropies[dt] = pd.Series(vals)
        if not entropies:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(entropies).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 38. intraday_signed_volume_ratio — 净主动买卖量比例
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_signed_volume_ratio_20d", category="intraday_advanced")
class IntradaySignedVolumeRatio20d(Factor):
    """净主动买卖量比例因子.

    使用 tick test (涨=主动买, 跌=主动卖) 推断每笔分钟量的方向.
    ratio = (买入量 - 卖出量) / 总成交量 ∈ [-1, +1].
    正ratio → 主动买入主导 → 正向信号.
    方向: 正向.
    """
    name = "intraday_signed_volume_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "净主动买卖比 ((买量-卖量)/总量, tick test推断)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, volume = panel["close"], panel["volume"]
        ret_1m = close.pct_change()
        day = ret_1m.index.normalize()
        ratios: dict = {}
        for dt in sorted(set(day)):
            grp_ret = ret_1m.loc[day == dt]
            grp_vol = volume.loc[day == dt]
            if len(grp_ret) < 20:
                continue
            vals = {}
            for col in grp_ret.columns:
                r = grp_ret[col].dropna()
                v = grp_vol[col].dropna()
                common = r.index.intersection(v.index)
                if len(common) < 20:
                    continue
                r_c = r.loc[common]
                v_c = v.loc[common]
                up_mask = r_c > 0
                dn_mask = r_c < 0
                buy_vol = v_c[up_mask].sum()
                sell_vol = v_c[dn_mask].sum()
                total = buy_vol + sell_vol
                vals[col] = float((buy_vol - sell_vol) / total) if total > 1e-12 else 0.0
            if vals:
                ratios[dt] = pd.Series(vals)
        if not ratios:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(ratios).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 39. intraday_semivariance_ratio — 上下行已实现方差比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_semivariance_ratio_20d", category="intraday_advanced")
class IntradaySemivarianceRatio20d(Factor):
    """上下行已实现方差比因子.

    上行方差 = Σ(ret²[ret>0]), 下行方差 = Σ(ret²[ret<0]).
    比值 > 1 → 上涨波动主导 (好波动) → 正向信号.
    比值 < 1 → 下跌波动主导 (坏波动) → 负向信号.
    方向: 正向.
    """
    name = "intraday_semivariance_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "上下行方差比 (上行已实现方差/下行已实现方差)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        semivars: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 20:
                    continue
                up_var = (r[r > 0] ** 2).sum()
                dn_var = (r[r < 0] ** 2).sum()
                vals[col] = float(up_var / dn_var) if dn_var > 1e-12 else (2.0 if up_var > 1e-12 else 1.0)
            if vals:
                semivars[dt] = pd.Series(vals)
        if not semivars:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(semivars).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 40. intraday_micro_leverage — 微观杠杆效应
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_micro_leverage_20d", category="intraday_advanced")
class IntradayMicroLeverage20d(Factor):
    """微观杠杆效应因子.

    日内收益与未来波动率的负相关: corr(ret_t, amplitude_{t+1}).
    强杠杆 (大负值) → 下跌后波动急升 → 风险高 → 负向.
    弱杠杆 (近零/正) → 价格与波动脱钩 → 稳定 → 正向.
    方向: 正向 (取 -corr).
    """
    name = "intraday_micro_leverage_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "微观杠杆效应 (ret_t与amp_{t+1}相关, 取负值=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"high", "low", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, low, close = panel["high"], panel["low"], panel["close"]
        ret_1m = close.pct_change()
        amplitude = (high - low) / close.replace(0, np.nan)
        day = ret_1m.index.normalize()
        leverages: dict = {}
        for dt in sorted(set(day)):
            grp_ret = ret_1m.loc[day == dt]
            grp_amp = amplitude.loc[day == dt]
            if len(grp_ret) < 30:
                continue
            vals = {}
            for col in grp_ret.columns:
                r = grp_ret[col].dropna()
                a = grp_amp[col].dropna()
                common = r.index.intersection(a.index)
                if len(common) < 30:
                    continue
                r_c = r.loc[common]
                a_c = a.loc[common]
                if r_c.std() < 1e-12 or a_c.std() < 1e-12:
                    vals[col] = 0.0
                else:
                    corr_val = r_c.iloc[:-1].corr(a_c.iloc[1:], method="pearson")
                    vals[col] = float(-corr_val) if not np.isnan(corr_val) else 0.0
            if vals:
                leverages[dt] = pd.Series(vals)
        if not leverages:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(leverages).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 41. intraday_price_run_duration — 价格连续运行持久性
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_price_run_duration_20d", category="intraday_advanced")
class IntradayPriceRunDuration20d(Factor):
    """价格连续运行持久性因子.

    日内同向连续涨/跌的最长分钟数 / 总分钟数.
    run占比大 → 趋势持久性极强 → 动量特征 → 正向.
    run占比小 → 频繁切换 → 无方向性 → 负向.
    方向: 正向.
    """
    name = "intraday_price_run_duration_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "价格连续运行 (最长同向run/总分钟数)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        durations: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 20:
                    continue
                signs = np.sign(r.values)
                max_run = 1
                cur_run = 1
                for i in range(1, len(signs)):
                    if signs[i] != 0 and signs[i] == signs[i - 1]:
                        cur_run += 1
                    else:
                        max_run = max(max_run, cur_run)
                        cur_run = 1
                max_run = max(max_run, cur_run)
                vals[col] = float(max_run / len(signs))
            if vals:
                durations[dt] = pd.Series(vals)
        if not durations:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(durations).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 42. intraday_parkinson_vol_ratio — Parkinson / Close-Close 波动比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_parkinson_vol_ratio_20d", category="intraday_advanced")
class IntradayParkinsonVolRatio20d(Factor):
    """Parkinson/Close-Close 波动率比因子.

    Parkinson vol = sqrt(Σ(log(H/L)²) / (4·n·log2)) 基于日内HL.
    Close-Close vol = std(close pct change).
    ratio = Parkinson / Close-Close.
    高ratio → 日内震荡大但收盘变动小 → 噪声多/方向不确定 → 负向.
    方向: 正向 (取 -ratio).
    """
    name = "intraday_parkinson_vol_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "Parkinson/Close波动比 (HL波动/收盘波动, 高比=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"high", "low", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, low, close = panel["high"], panel["low"], panel["close"]
        day = close.index.normalize()
        ratios: dict = {}
        for dt in sorted(set(day)):
            grp_high = high.loc[day == dt]
            grp_low = low.loc[day == dt]
            grp_close = close.loc[day == dt]
            if len(grp_close) < 20:
                continue
            vals = {}
            for col in grp_close.columns:
                h = grp_high[col].dropna()
                l = grp_low[col].dropna()
                c = grp_close[col].dropna()
                common_hl = h.index.intersection(l.index)
                if len(common_hl) < 20:
                    continue
                h_c = h.loc[common_hl]
                l_c = l.loc[common_hl]
                log_hl = np.log(h_c / l_c.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
                if len(log_hl) < 10:
                    continue
                parkinson = float(np.sqrt((log_hl ** 2).sum() / (4.0 * len(log_hl) * np.log(2.0))))
                cc_vol = c.pct_change().dropna().std(ddof=0)
                if cc_vol < 1e-12:
                    vals[col] = 0.0
                else:
                    vals[col] = float(-parkinson / cc_vol)
            if vals:
                ratios[dt] = pd.Series(vals)
        if not ratios:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(ratios).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 43. intraday_level_clustering — 价格整数位聚集度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_level_clustering_20d", category="intraday_advanced")
class IntradayLevelClustering20d(Factor):
    """价格整数位聚集度因子.

    收盘价落在整数位 (±0.05) 的分钟占比.
    高聚集 → 交易在整数关口集中 → 散户心理价位主导 → 负向.
    低聚集 → 价格连续分布 → 机构定价主导 → 正向.
    方向: 正向 (取负值, 低聚集=高值).
    """
    name = "intraday_level_clustering_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "价格整数位聚集 (收盘在整数比例, 取负=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        clusterings: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < 20:
                    continue
                rounded = np.round(c.values)
                dist = np.abs(c.values - rounded)
                clustered = float((dist < 0.05).sum())
                vals[col] = -clustered / len(c)
            if vals:
                clusterings[dt] = pd.Series(vals)
        if not clusterings:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(clusterings).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 44. intraday_kyle_lambda — 简化 Kyle's Lambda (价格冲击)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_kyle_lambda_20d", category="intraday_advanced")
class IntradayKyleLambda20d(Factor):
    """简化 Kyle's Lambda 因子.

    λ = |ret| / signed_vol, signed_vol = sign(ret) × sqrt(vol).
    高λ → 单位成交量推动大价格变化 → 流动性差/信息不对称 → 负向.
    方向: 负向.
    """
    name = "intraday_kyle_lambda_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "Kyle Lambda (|ret|/signed_vol, 价格冲击系数)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, volume = panel["close"], panel["volume"]
        ret_1m = close.pct_change()
        day = ret_1m.index.normalize()
        lambdas: dict = {}
        for dt in sorted(set(day)):
            grp_ret = ret_1m.loc[day == dt]
            grp_vol = volume.loc[day == dt]
            if len(grp_ret) < 20:
                continue
            vals = {}
            for col in grp_ret.columns:
                r = grp_ret[col].dropna()
                v = grp_vol[col].dropna()
                common = r.index.intersection(v.index)
                if len(common) < 20:
                    continue
                r_c = r.loc[common].values
                v_c = v.loc[common].values
                signed_vol = np.sign(r_c) * np.sqrt(v_c)
                numer = np.sum(np.abs(r_c) * np.abs(signed_vol))
                denom = np.sum(signed_vol ** 2)
                vals[col] = float(numer / denom) if denom > 1e-12 else 0.0
            if vals:
                lambdas[dt] = pd.Series(vals)
        if not lambdas:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(lambdas).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 45. intraday_roll_spread — Roll (1984) 买卖价差
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_roll_spread_20d", category="intraday_advanced")
class IntradayRollSpread20d(Factor):
    """Roll (1984) 买卖价差因子.

    spread = 2 × sqrt(-cov(Δp_t, Δp_{t-1})) 当协方差为负.
    利用价格变动一阶自协方差推断有效价差.
    高spread → 交易成本高 → 负向.
    方向: 负向.
    """
    name = "intraday_roll_spread_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "Roll价差 (价格变动自协方差估计价差)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        spreads: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 30:
                    continue
                cov_val = np.cov(r[1:], r[:-1], ddof=0)[0, 1]
                vals[col] = float(2.0 * np.sqrt(-cov_val)) if cov_val < 0 else 0.0
            if vals:
                spreads[dt] = pd.Series(vals)
        if not spreads:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(spreads).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 46. intraday_realized_covariance — 收益-成交量已实现协方差
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_realized_covariance_20d", category="intraday_advanced")
class IntradayRealizedCovariance20d(Factor):
    """收益-成交量已实现协方差因子.

    Σ( (ret_t - μ_ret) × (log_vol_t - μ_logvol) ) / n.
    正协方差 → 涨时放量跌时缩量 → 健康量价关系 → 正向.
    负协方差 → 涨时缩量跌时放量 → 不健康 → 负向.
    方向: 正向.
    """
    name = "intraday_realized_covariance_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "收益量已实现协方差 (Σ(ret_dev×logvol_dev)/n)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, volume = panel["close"], panel["volume"]
        ret_1m = close.pct_change()
        day = ret_1m.index.normalize()
        covariances: dict = {}
        for dt in sorted(set(day)):
            grp_ret = ret_1m.loc[day == dt]
            grp_vol = volume.loc[day == dt]
            if len(grp_ret) < 20:
                continue
            vals = {}
            for col in grp_ret.columns:
                r = grp_ret[col].dropna()
                v = grp_vol[col].dropna()
                common = r.index.intersection(v.index)
                if len(common) < 20:
                    continue
                r_c = r.loc[common].values
                v_c = v.loc[common].values
                log_v = np.log(v_c + 1.0)
                r_dev = r_c - r_c.mean()
                v_dev = log_v - log_v.mean()
                vals[col] = float(np.mean(r_dev * v_dev))
            if vals:
                covariances[dt] = pd.Series(vals)
        if not covariances:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(covariances).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 47. intraday_seasonality_residual — 日内分时模式偏离度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_seasonality_residual_20d", category="intraday_advanced")
class IntradaySeasonalityResidual20d(Factor):
    """日内分时模式偏离度因子.

    当日成交量分时分布与近10日平均模式的标准化偏差.
    偏离大 → 今日成交节奏异于常态 → 有增量事件 → 正向.
    方向: 正向.
    """
    name = "intraday_seasonality_residual_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "日内模式偏离 (成交量分时vs近10日均值偏差)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        volume = panel["volume"]
        day = volume.index.normalize()
        sorted_days = sorted(set(day))
        residuals: dict = {}
        lookback_days = 10
        for idx, dt in enumerate(sorted_days):
            grp = volume.loc[day == dt]
            n_bars = len(grp)
            if n_bars < 20 or idx < lookback_days:
                continue
            recent_days = sorted_days[idx - lookback_days:idx]
            vals = {}
            for col in grp.columns:
                v_today = grp[col].dropna()
                if len(v_today) < 20:
                    continue
                profiles = []
                for rd in recent_days:
                    rg = volume.loc[day == rd, col].dropna()
                    if len(rg) >= n_bars:
                        profiles.append(rg.values[:n_bars])
                if len(profiles) < 3:
                    continue
                # Align to shortest day to avoid broadcast errors
                min_bars = min(len(v_today), min(len(p) for p in profiles))
                avg_profile = np.mean(np.array([p[:min_bars] for p in profiles]), axis=0)
                std_profile = np.std(np.array([p[:min_bars] for p in profiles]), axis=0, ddof=0)
                std_profile[std_profile < 1e-12] = 1.0
                v_arr = v_today.values[:min_bars]
                deviation = np.mean(np.abs(v_arr - avg_profile) / std_profile)
                vals[col] = float(deviation)
            if vals:
                residuals[dt] = pd.Series(vals)
        if not residuals:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(residuals).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)
