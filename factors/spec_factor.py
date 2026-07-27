"""SPEC 驱动的因子框架 (向量化版本).

借鉴 QuantSkills 的 base + transform 组合模式, 通过 SPEC 字典批量生成因子.
每个因子 = base 计算 (原始信号) + transform 变换 (标准化/平滑/排名等).

优势:
- base × transform × window 由 SPEC 批量定义, 避免公式代码重复
- 向量化计算: 一次性处理所有品种, 性能比逐品种循环快 10-50 倍
- SPEC 字典同时作为因子元数据, 便于索引和文档生成

兼容性:
- SpecFactor 继承 Factor 基类, 通过 @register_factor 注册
- compute() 接口与原有一致, 返回 FactorMatrix (日期×品种)
- 原有因子不受影响, 与新因子共存
"""
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from core.interfaces import Factor
from core.registry import register_factor
from factors.practical_bases import PRACTICAL_BASES, compute_practical_base


# ---------------------------------------------------------------------------
# 辅助函数 (向量化, 处理 rolling 窗口初期数据不足)
# ---------------------------------------------------------------------------

def _minp(window: int, floor: int = 5) -> int:
    """计算 rolling 最小有效周期, 避免窗口初期全 NaN."""
    return min(window, max(2, min(floor, window // 2 if window > 2 else 2)))


_DEFAULT_SPEC_FIELDS = ["open", "high", "low", "close", "volume"]


def _normalise_frequency(value: object) -> str:
    aliases = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "60m": "hourly",
        "60min": "hourly",
        "1h": "hourly",
    }
    raw = str(value).lower()
    return aliases.get(raw, raw)


def _spec_dependencies(spec: dict) -> List[str]:
    """Return stable, de-duplicated source fields declared by a SPEC."""
    fields = spec.get("dependencies") or _DEFAULT_SPEC_FIELDS
    return list(dict.fromkeys(str(field) for field in fields))


# ---------------------------------------------------------------------------
# 分钟级数据获取与按日重采样 (frequency != "daily" 时使用)
# ---------------------------------------------------------------------------

def _fetch_intraday_ohlcv(
    data, dates, universe, freq: str = "15min",
) -> Dict[str, pd.DataFrame]:
    """从 DataManager 的底层 DataSource 获取分钟级 OHLCV.

    当 DataSource 为 DDBSource 时, 调用 fetch_price_at_frequency().
    其他数据源或 DDB 不可用时返回空字典 (优雅降级).

    Args:
        data: DataManager 实例 (需暴露 .source 属性)
        dates: 日度日期索引 (用于确定日期范围)
        universe: 品种池
        freq: 分钟频率 ("15min" / "30min" / "60min")

    Returns:
        {field: DataFrame(index=分钟时间戳, columns=tickers)} 或 {} (不可用)
    """
    try:
        requested_frequency = _normalise_frequency(freq)
        provider_frequency = getattr(data, "frequency", None)
        if provider_frequency is not None:
            if _normalise_frequency(provider_frequency) != requested_frequency:
                return {}
            return {
                field: data.get(field, dates, universe)
                for field in _DEFAULT_SPEC_FIELDS
            }

        source = getattr(data, "source", None)
        if source is None:
            return {}
        # 检查是否为 DDBSource (避免硬依赖, 用 duck-typing)
        if not hasattr(source, "fetch_price_at_frequency"):
            return {}
        start = dates.min()
        end = dates.max()
        fields = ["open", "high", "low", "close", "volume"]
        panel = source.fetch_price_at_frequency(
            list(universe), start, end, fields, frequency=freq,
        )
        return panel if panel else {}
    except Exception:
        logging.getLogger("multi_factor").debug(
            "分钟级 OHLCV 获取失败 (DDB 可能不可用)", exc_info=True,
        )
        return {}


def _resample_to_daily(
    minute_df: pd.DataFrame, dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Aggregate intraday values by exchange trading day and lag one day.

    分钟因子的输出最终需要与日度因子对齐 (日度日期索引),
    以便后续 IC 检验/组合优化/回测统一处理.

    Args:
        minute_df: 分钟级 DataFrame (index=分钟时间戳, columns=tickers)
        dates: 目标日度日期索引

    Returns:
        日度 DataFrame (index=dates, columns=minute_df.columns)
    """
    if minute_df is None or minute_df.empty:
        return pd.DataFrame()
    # 确保 index 是 DatetimeIndex
    if not isinstance(minute_df.index, pd.DatetimeIndex):
        minute_df.index = pd.DatetimeIndex(minute_df.index)
    trading_calendar = pd.DatetimeIndex(dates).normalize().unique().sort_values()
    natural_dates = minute_df.index.normalize()
    targets = natural_dates.where(
        minute_df.index.hour < 18,
        natural_dates + pd.Timedelta(days=1),
    )
    locations = trading_calendar.searchsorted(targets, side="left")
    trading_dates = np.full(
        len(targets), np.datetime64("NaT"), dtype="datetime64[ns]"
    )
    valid = locations < len(trading_calendar)
    trading_dates[valid] = trading_calendar.to_numpy()[locations[valid]]
    usable = ~pd.isna(trading_dates)
    daily = minute_df.loc[usable].groupby(trading_dates[usable]).last()
    daily.index = pd.DatetimeIndex(daily.index)
    # The trading-day close is only available after that close. Shift the
    # exposure so research never assumes execution at the same close.
    return daily.reindex(dates).shift(1)


def _apply_spec_decision_lag(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Apply the point-in-time lag declared by a SPEC (one bar by default)."""
    lag = max(int(spec.get("decision_lag_bars", 1)), 0)
    return frame.shift(lag) if lag else frame


def _finalize_intraday_result(
    frame: pd.DataFrame,
    data,
    dates,
    universe,
    spec: dict,
) -> pd.DataFrame:
    """Align an intraday SPEC to either real bars or daily decisions."""
    provider_frequency = getattr(data, "frequency", None)
    spec_frequency = _normalise_frequency(spec.get("frequency", "daily"))
    if provider_frequency is not None:
        if _normalise_frequency(provider_frequency) != spec_frequency:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        return _apply_spec_decision_lag(frame, spec).reindex(
            index=dates, columns=universe
        )

    daily = _resample_to_daily(frame, pd.DatetimeIndex(dates))
    if daily.empty:
        return pd.DataFrame(np.nan, index=dates, columns=universe)
    return daily.reindex(index=dates, columns=universe)


def _zscore_df(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """向量化 z-score (每列独立)."""
    mean = df.rolling(window, min_periods=_minp(window)).mean()
    std = df.rolling(window, min_periods=_minp(window)).std(ddof=0)
    return (df - mean) / std.replace(0, np.nan)


def _ema_df(df: pd.DataFrame, span: int) -> pd.DataFrame:
    """向量化 EMA."""
    return df.ewm(span=span, adjust=False, min_periods=_minp(span, 3)).mean()


def _atr_df(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int) -> pd.DataFrame:
    """向量化 ATR (Average True Range)."""
    prev = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev).abs()
    tr3 = (low - prev).abs()
    # 逐元素取三者最大值
    tr = tr1.where(tr1 >= tr2, tr2)
    tr = tr.where(tr >= tr3, tr3)
    return tr.rolling(window, min_periods=_minp(window)).mean()


def _rsi_df(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """向量化 RSI 相对强弱指标."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(
        alpha=1.0 / window, adjust=False, min_periods=_minp(window)
    ).mean()
    avg_loss = loss.ewm(
        alpha=1.0 / window, adjust=False, min_periods=_minp(window)
    ).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _ts_rank_df(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """向量化时序百分位排名 (每列独立).

    使用 pandas 内置 rolling().rank(pct=True) 实现,
    比逐窗口 Python apply 快 50-100 倍, 且行为一致.

    CR-018: 输出范围为 [0, 1], 不在此处中心化. 中心化统一在合成阶段
    通过截面 z-score 标准化完成, 避免重复中心化.
    """
    minp = _minp(window)
    # pandas 1.4+ 支持 rolling().rank(pct=True), 旧版本回退到 rank().rolling()
    try:
        return df.rolling(window, min_periods=minp).rank(pct=True)
    except (AttributeError, TypeError):
        # 回退: 逐列用 rank(pct=True).rolling(window).mean() 近似
        return df.rank(pct=True).rolling(window, min_periods=minp).mean()


def _slope_df(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """向量化滚动线性回归斜率 (x 为等间距时间索引 1..n).

    对每个滚动窗口做 OLS: y ~ a + b*x, 返回斜率 b.
    使用 rolling.apply(raw=True) 逐窗口计算.
    在批量分组中每组仅调用一次, 性能可接受.
    """
    minp = _minp(window)

    def _ols_slope(v):
        n = len(v)
        if n < 2:
            return np.nan
        x = np.arange(1, n + 1, dtype=float)
        xm = x.mean()
        ym = v.mean()
        denom = np.sum((x - xm) ** 2)
        if denom == 0:
            return np.nan
        return np.dot(x - xm, v - ym) / denom

    return df.rolling(window, min_periods=minp).apply(_ols_slope, raw=True)


# ---------------------------------------------------------------------------
# Base 因子计算 (向量化, 一次性处理所有品种)
# ---------------------------------------------------------------------------

def compute_base_df(
    base: str, params: dict, ohlcv: Dict[str, pd.DataFrame],
    period_ctx=None,
) -> pd.DataFrame:
    """根据 base 类型计算原始因子信号 (向量化, 所有品种同时计算).

    Args:
        base: base 因子名称 (如 'rsi', 'atr_ratio').
        params: 参数字典, 至少包含 'window'.
            **window 为"周期数" (bar数) 语义**, 不是天数.
            当 period_ctx.unit == DAILY 时, 1 个周期 = 1 个交易日;
            当 period_ctx.unit == MINUTE_15 时, 1 个周期 = 1 个 15分钟 bar.
        ohlcv: {'open':DataFrame, 'high':..., 'low':..., 'close':..., 'volume':...}.
               每个 value 是 dates × tickers 的 DataFrame.
        period_ctx: 周期上下文 (core.period.PeriodContext, 可选).
            当前未参与计算 (预留接口, 未来用于年化波动率等需要周期单位换算的场景).
            None 时行为与现有完全一致.

    Returns:
        dates × tickers 的因子值 DataFrame.
    """
    close = ohlcv["close"].astype(float, copy=False)
    open_ = ohlcv["open"].astype(float, copy=False)
    high = ohlcv["high"].astype(float, copy=False)
    low = ohlcv["low"].astype(float, copy=False)
    volume = ohlcv["volume"].astype(float, copy=False)
    ret = ohlcv.get("_return_1d")
    if ret is None:
        ret = close.pct_change()
    w = int(params.get("window", 20))

    # --- 震荡类 (Oscillator) ---
    if base == "rsi":
        return _rsi_df(close, w) / 100 - 0.5
    if base == "rsi_reversal":
        return 0.5 - _rsi_df(close, w) / 100
    if base == "stoch":
        lo = low.rolling(w, min_periods=_minp(w)).min()
        hi = high.rolling(w, min_periods=_minp(w)).max()
        return (close - lo) / (hi - lo).replace(0, np.nan) - 0.5

    # --- 波动率类 (Volatility) ---
    if base == "atr_ratio":
        return _atr_df(high, low, close, w) / close.replace(0, np.nan)
    if base == "range_ratio":
        return ((high - low) / close.replace(0, np.nan)).rolling(
            w, min_periods=_minp(w)
        ).mean()
    if base == "realized_vol":
        return ret.rolling(w, min_periods=_minp(w)).std(ddof=0)
    if base == "downside_vol":
        return ret.where(ret < 0, 0).rolling(w, min_periods=_minp(w)).std(ddof=0)

    # --- 形态类 (Pattern) ---
    if base == "gap_sum":
        return (open_ / close.shift(1) - 1).rolling(w, min_periods=_minp(w)).sum()
    if base == "intraday":
        return (close / open_.replace(0, np.nan) - 1).rolling(
            w, min_periods=_minp(w)
        ).mean()
    if base == "upper_wick":
        top = open_.where(open_ >= close, close)
        return ((high - top) / (high - low).replace(0, np.nan)).rolling(
            w, min_periods=_minp(w)
        ).mean()
    if base == "lower_wick":
        bottom = open_.where(open_ <= close, close)
        return ((bottom - low) / (high - low).replace(0, np.nan)).rolling(
            w, min_periods=_minp(w)
        ).mean()

    # --- 回撤类 (Drawdown) ---
    if base == "drawdown":
        return close / close.rolling(w, min_periods=_minp(w)).max().replace(
            0, np.nan
        ) - 1

    # --- 技术指标类 (Technical Indicators, 来自 DolphinDB alpha_db 推导) ---
    # fast/slow 参数从 params 读取 (已在 SPEC 模板中定义)
    fast = int(params.get("fast", max(3, w // 3)))
    slow = int(params.get("slow", max(5, w)))

    if base == "macd_diff":
        # MACD 柱状图: DIF - DEA
        # DIF = EMA(fast) - EMA(slow), DEA = EMA(DIF, signal)
        ema_fast = _ema_df(close, fast)
        ema_slow = _ema_df(close, slow)
        dif = ema_fast - ema_slow
        signal_period = max(3, (fast + slow) // 3)
        dea = _ema_df(dif, signal_period)
        return dif - dea

    if base == "kdj_j":
        # KDJ 的 J 值: 3K - 2D, K=D 的 EMA, D=RSV 的 EMA
        # RSV = (close - low_min) / (high_max - low_min)
        lo = low.rolling(w, min_periods=_minp(w)).min()
        hi = high.rolling(w, min_periods=_minp(w)).max()
        rsv = (close - lo) / (hi - lo).replace(0, np.nan)
        # K = EMA(RSV, 3), D = EMA(K, 3) (传统 KDJ 用 3 周期平滑)
        k = _ema_df(rsv, 3)
        d = _ema_df(k, 3)
        return 3 * k - 2 * d - 0.5  # 居中到 0

    if base == "boll_position":
        # 布林带位置: (close - MA) / (k * std)
        ma = close.rolling(w, min_periods=_minp(w)).mean()
        std = close.rolling(w, min_periods=_minp(w)).std(ddof=0)
        # 位置在 [-1, 1] 之间 (close 在下轨=-1, 上轨=+1)
        return (close - ma) / (2 * std.replace(0, np.nan))

    if base == "boll_width":
        # 布林带宽度: (upper - lower) / MA = 2*k*std / MA
        ma = close.rolling(w, min_periods=_minp(w)).mean()
        std = close.rolling(w, min_periods=_minp(w)).std(ddof=0)
        return (4 * std) / ma.replace(0, np.nan)

    if base == "obv_slope":
        # OBV 斜率: OBV 的滚动变化率
        # OBV = cumsum(sign(ret) * volume)
        sign_ret = np.sign(ret)
        obv = (sign_ret * volume).cumsum()
        # 用滚动差分作为斜率 (避免 cumsum 的非平稳性)
        return obv.diff(w) / volume.rolling(w, min_periods=_minp(w)).mean().replace(0, np.nan)

    if base == "bias":
        # 乖离率: (close - MA) / MA
        ma = close.rolling(w, min_periods=_minp(w)).mean()
        return (close - ma) / ma.replace(0, np.nan)

    if base == "dmi_adx":
        # DMI 的 ADX: DI 差值的 EMA (简化版)
        # +DM = high - prev_high (若 > 0 且 > -DM), -DM = prev_low - low (同理)
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        up_move = high - prev_high
        down_move = prev_low - low
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        atr = _atr_df(high, low, close, w)
        plus_di = 100 * plus_dm.rolling(w, min_periods=_minp(w)).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.rolling(w, min_periods=_minp(w)).mean() / atr.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        # ADX = DX 的 EMA
        adx = _ema_df(dx.fillna(0), w)
        # 返回 ADX - 50, 居中到 0 (ADX > 50 表示强趋势)
        return adx - 50

    if base == "vr_ratio":
        # VR 成交量比率: 上涨日成交量 / 下跌日成交量
        up_vol = volume.where(ret > 0, 0).rolling(w, min_periods=_minp(w)).sum()
        down_vol = volume.where(ret < 0, 0).rolling(w, min_periods=_minp(w)).sum()
        return up_vol / down_vol.replace(0, np.nan) - 1

    if base == "wr_ratio":
        # WR 威廉指标: (H_max - close) / (H_max - L_min)
        # 与 stoch 互补: WR = -stoch (符号相反)
        lo = low.rolling(w, min_periods=_minp(w)).min()
        hi = high.rolling(w, min_periods=_minp(w)).max()
        return (hi - close) / (hi - lo).replace(0, np.nan) - 0.5

    if base == "trix":
        # TRIX: 三重平滑收益率的差分
        t1 = _ema_df(close, w)
        t2 = _ema_df(t1, w)
        t3 = _ema_df(t2, w)
        return t3.pct_change()

    if base == "dpo":
        # DPO 去趋势价格: close - MA(shifted)
        # DPO 消除长期趋势, 突出周期性
        shift_periods = w // 2 + 1
        ma = close.rolling(w, min_periods=_minp(w)).mean()
        return close - ma.shift(shift_periods)

    if base == "mavol_ratio":
        # MAVOL 量比: 当前成交量 / 滚动平均成交量
        vol_ma = volume.rolling(w, min_periods=_minp(w)).mean()
        return volume / vol_ma.replace(0, np.nan) - 1

    if base == "arbr_sentiment":
        # ARBR 情绪指标: AR = sum(high-open) / sum(open-prev_close)
        # 简化为 AR 信号 (买卖气势)
        prev_close = close.shift(1)
        ar_num = (high - open_).rolling(w, min_periods=_minp(w)).sum()
        ar_den = (open_ - prev_close).rolling(w, min_periods=_minp(w)).sum()
        return ar_num / ar_den.replace(0, np.nan) - 1

    if base == "asi_slope":
        # ASI 累计摆动指标斜率 (简化版)
        # ASI 基于 close 相对 prev_close 的真实波动
        prev_close = close.shift(1)
        # 简化 ASI: 累计 (close - prev_close) / ATR
        atr = _atr_df(high, low, close, w)
        asi_daily = (close - prev_close) / atr.replace(0, np.nan)
        asi = asi_daily.cumsum()
        return asi.diff(w)

    if base == "bbi_position":
        # BBI 多空指标位置: close 相对 4 条 MA 均值的平均偏离
        ma1 = close.rolling(w, min_periods=_minp(w)).mean()
        ma2 = close.rolling(w * 2, min_periods=_minp(w * 2)).mean()
        ma3 = close.rolling(w * 3, min_periods=_minp(w * 3)).mean()
        ma4 = close.rolling(w * 4, min_periods=_minp(w * 4)).mean()
        bbi = (ma1 + ma2 + ma3 + ma4) / 4
        return (close - bbi) / bbi.replace(0, np.nan)

    if base == "newdma_trend":
        # NEWDMA 动态均线趋势: close 相对 DMA(快) - DMA(慢) 的位置
        dma_fast = close.rolling(fast, min_periods=_minp(fast)).mean()
        dma_slow = close.rolling(slow, min_periods=_minp(slow)).mean()
        # 当 close > DMA_fast > DMA_slow 时为正趋势
        return (dma_fast - dma_slow) / dma_slow.replace(0, np.nan)

    # --- 日内代理特征 (Intraday Proxy, 从日度OHLC推导日内行为) ---
    rng = (high - low).replace(0, np.nan)

    if base == "intraday_strength":
        # 日内强度: (close-open)/(high-low), >0.5强势收盘
        return ((close - open_) / rng).rolling(w, min_periods=_minp(w)).mean()

    if base == "close_position":
        # 收盘位置: (close-low)/(high-low), 接近1=收盘在最高附近
        return ((close - low) / rng).rolling(w, min_periods=_minp(w)).mean()

    if base == "body_ratio":
        # 实体占比: |close-open|/(high-low), 高=趋势日
        return ((close - open_).abs() / rng).rolling(w, min_periods=_minp(w)).mean()

    if base == "upper_wick_ratio":
        # 上影线占比: (high-max(open,close))/(high-low)
        top = open_.where(open_ >= close, close)
        return ((high - top) / rng).rolling(w, min_periods=_minp(w)).mean()

    if base == "lower_wick_ratio":
        # 下影线占比: (min(open,close)-low)/(high-low)
        bottom = open_.where(open_ <= close, close)
        return ((bottom - low) / rng).rolling(w, min_periods=_minp(w)).mean()

    if base == "overnight_intraday_split":
        # 隔夜-日内收益分解: 日内收益 / |总收益|
        # intraday_ret = close/open - 1, overnight_ret = open/prev_close - 1
        # 比值接近1=信息日内释放(趋势可持续), 接近0=隔夜跳空
        intraday_ret = close / open_.replace(0, np.nan) - 1
        overnight_ret = open_ / close.shift(1).replace(0, np.nan) - 1
        total_abs = (overnight_ret + intraday_ret).abs()
        split = intraday_ret / total_abs.replace(0, np.nan)
        # CR-020: 缺失行情保留 NaN, 不填 0.5 (避免制造虚假中性信号).
        # clip 防极端值; NaN 保留后, 截面排序/标准化时该品种不会被纳入组合.
        split = split.clip(-2, 2)
        return split.rolling(w, min_periods=_minp(w)).mean()

    # --- 方向类 (Directional, 借鉴 QuantSkills directional-alpha) ---

    if base == "sma_gap":
        # SMA 偏离: 收盘价相对简单均线偏离率
        sma = close.rolling(w, min_periods=_minp(w)).mean()
        return close / sma.replace(0, np.nan) - 1

    if base == "ema_gap":
        # EMA 偏离: 收盘价相对指数均线偏离率
        ema = _ema_df(close, w)
        return close / ema.replace(0, np.nan) - 1

    if base == "dual_ema_gap":
        # 双 EMA 差: (EMA_fast - EMA_slow) / close, fast=w//2, slow=w
        fast_w = max(2, w // 2)
        slow_w = w
        ema_fast = _ema_df(close, fast_w)
        ema_slow = _ema_df(close, slow_w)
        return (ema_fast - ema_slow) / close.replace(0, np.nan)

    if base == "sma_slope":
        # SMA 斜率: 简单均线的线性回归斜率 / close
        sma = close.rolling(w, min_periods=_minp(w)).mean()
        slope = _slope_df(sma, w)
        return slope / close.replace(0, np.nan)

    if base == "trend_strength":
        # 夏普式趋势强度: close.pct_change(w) / std(ret, w)
        ret_w = close.pct_change(w)
        vol = ret.rolling(w, min_periods=_minp(w)).std(ddof=0)
        return ret_w / vol.replace(0, np.nan)

    if base == "efficiency":
        # Kaufman 趋势效率: |净位移| / 路径总和
        displacement = (close - close.shift(w)).abs()
        path = close.diff().abs().rolling(w, min_periods=_minp(w)).sum()
        return displacement / path.replace(0, np.nan)

    if base == "return":
        # 收益率: close.pct_change(w)
        return close.pct_change(w)

    if base == "skip_return":
        # 跳期动量: close.shift(w//2).pct_change(w), 规避短期反转
        skip = max(1, w // 2)
        return close.shift(skip).pct_change(w)

    if base == "reversal":
        # 反转: 负收益率
        return -close.pct_change(w)

    if base == "breakout":
        # 上轨突破: close / max(high.shift(1).rolling(w).max()) - 1
        hh = high.shift(1).rolling(w, min_periods=_minp(w)).max()
        return close / hh.replace(0, np.nan) - 1

    if base == "breakdown":
        # 下轨跌破: -(close / min(low.shift(1).rolling(w).min()) - 1)
        ll = low.shift(1).rolling(w, min_periods=_minp(w)).min()
        return -(close / ll.replace(0, np.nan) - 1)

    if base == "range_position":
        # 通道位置: 收盘价在高低通道中的位置, 居中到 0
        lo = low.rolling(w, min_periods=_minp(w)).min()
        hi = high.rolling(w, min_periods=_minp(w)).max()
        return (close - lo) / (hi - lo).replace(0, np.nan) - 0.5

    # --- 持仓参与类 (Positioning Participation) ---

    if base == "low_churn_trend":
        oi = ohlcv.get("oi")
        if oi is None or oi.empty:
            raise ValueError("low_churn_trend requires open interest field 'oi'")
        lag_close = close.where(close > 0).shift(1)
        lag_volume = volume.where(volume > 0).shift(1)
        lag_oi = oi.astype(float).where(oi.astype(float) > 0).shift(1)
        momentum = np.log(lag_close).diff(w)
        turnover = (lag_volume / lag_oi).clip(lower=1e-4, upper=10.0)
        average_turnover = turnover.rolling(w, min_periods=w).mean()
        return momentum / np.sqrt(average_turnover)

    # --- 量价统计类 (Volume Stat, 借鉴 QuantSkills volume-stat-alpha) ---
    # 注意: obv_slope 已在 technicals 中实现, 此处不重复

    if base == "volume_ratio":
        # 量比: 当前成交量 / 滚动均量 - 1
        vol_ma = volume.rolling(w, min_periods=_minp(w)).mean()
        return volume / vol_ma.replace(0, np.nan) - 1

    if base == "volume_z":
        # 量能 Z 值: (volume - mean) / std
        vol_mean = volume.rolling(w, min_periods=_minp(w)).mean()
        vol_std = volume.rolling(w, min_periods=_minp(w)).std(ddof=0)
        return (volume - vol_mean) / vol_std.replace(0, np.nan)

    if base == "dollar_volume":
        # 成交额比: (close*volume) / rolling_mean(close*volume) - 1
        dv = close * volume
        dv_ma = dv.rolling(w, min_periods=_minp(w)).mean()
        return dv / dv_ma.replace(0, np.nan) - 1

    if base == "price_volume_corr":
        # 量价相关: ret.rolling(w).corr(volume.pct_change())
        vol_ret = volume.pct_change()
        return ret.rolling(w, min_periods=_minp(w)).corr(vol_ret)

    if base == "ts_rank_close":
        # 收盘价时序百分位排名, 居中到 0 (CR-018: _ts_rank_df 输出 [0,1])
        return _ts_rank_df(close, w) - 0.5

    if base == "ts_rank_volume":
        # 成交量时序百分位排名, 居中到 0
        return _ts_rank_df(volume, w) - 0.5

    if base == "ret_skew":
        # 收益偏度: 负偏度品种有 crash risk premium
        return ret.rolling(w, min_periods=_minp(w)).skew()

    if base == "ret_kurt":
        # 收益峰度: 高峰度有尾部风险溢价
        return ret.rolling(w, min_periods=_minp(w)).kurt()

    if base in PRACTICAL_BASES:
        return compute_practical_base(base, params, ohlcv)

    raise ValueError(f"unknown base: {base}")


# ---------------------------------------------------------------------------
# Transform 变换 (向量化, 一次性处理所有品种)
# ---------------------------------------------------------------------------

def apply_transform_df(
    signal: pd.DataFrame, transform: str, params: dict, ohlcv: Dict[str, pd.DataFrame],
    period_ctx=None,
) -> pd.DataFrame:
    """对原始信号应用变换 (向量化).

    Args:
        signal: base 计算的原始因子信号 DataFrame (dates × tickers).
        transform: 变换类型 (如 'z', 'delta', 'smooth').
        params: 参数字典. window/lag/smooth 等均为"周期数" (bar数) 语义.
        ohlcv: OHLCV 数据 (部分变换需要 close/volume).
        period_ctx: 周期上下文 (可选, 预留接口). None 时行为与现有完全一致.

    Returns:
        变换后的因子值 DataFrame (dates × tickers).
    """
    w = int(params.get("window", 20))
    norm = int(params.get("norm", max(20, w)))
    if transform == "raw":
        return signal
    if transform == "neg":
        return -signal
    if transform == "z":
        return _zscore_df(signal, norm)
    if transform == "delta":
        lag = int(params.get("lag", max(1, w // 4)))
        return signal - signal.shift(lag)
    if transform == "smooth":
        span = int(params.get("smooth", max(3, w // 3)))
        return _ema_df(signal, span)
    if transform == "rank":
        # CR-018: _ts_rank_df 已输出 [0, 1], 中心化在合成阶段 z-score 完成
        return _ts_rank_df(signal, norm)
    if transform == "vol_scaled":
        ret = ohlcv.get("_return_1d")
        if ret is None:
            ret = ohlcv["close"].pct_change()
        vol = ret.rolling(norm, min_periods=_minp(norm)).std(ddof=0)
        return signal / vol.replace(0, np.nan)
    if transform == "stability":
        vol = signal.rolling(norm, min_periods=_minp(norm)).std(ddof=0)
        return signal / vol.replace(0, np.nan)
    if transform == "confirm_volume":
        volume = ohlcv["volume"].astype(float)
        vol_ma = volume.rolling(norm, min_periods=_minp(norm)).mean()
        return signal * (volume / vol_ma.replace(0, np.nan))
    if transform == "compress":
        return np.tanh(signal)

    raise ValueError(f"unknown transform: {transform}")


# ---------------------------------------------------------------------------
# SpecFactor 适配器
# ---------------------------------------------------------------------------

class SpecFactor(Factor):
    """SPEC 驱动的因子, 通过 base + transform 组合定义.

    用法:
        spec = {
            "slug": "rsi_5d",
            "name_cn": "5日RSI强度",
            "base": "rsi",
            "transform": "raw",
            "params": {"window": 5, "norm": 20},
            "category": "oscillator",
            "description": "RSI 相对强弱指标",
            # 可选字段 (周期架构):
            "frequency": "daily",  # 周期单位, 默认 "daily", 未来可 "15min" 等
        }
        SpecFactor(spec).register()

    周期架构说明:
        - `params["window"]` 是"周期数" (bar数), 不是"天数"
        - 当 frequency="daily" (默认) 时, window=5 表示 5 个交易日
        - 当 frequency="15min" 时, window=5 表示 5 个 15分钟 bar
        - slug 中的 "5d" 后缀在 daily 频率下与 5 个周期等价; 非日度场景
          应使用更明确的命名 (如 "rsi_5p_15min")
    """

    def __init__(self, spec: dict):
        self.spec = spec
        self.name = spec["slug"]
        self.category = spec.get("category", "")
        # 周期单位: 从 SPEC 字典读取, 默认 "daily" (向后兼容)
        # 支持值: "daily" / "15min" / "30min" / "hourly" (见 core.period.PeriodUnit)
        self.frequency = spec.get("frequency", "daily")
        self.description = spec.get("description", spec.get("name_cn", ""))
        self.research_tier = spec.get("research_tier", "core")
        self.expected_direction = spec.get("expected_direction", "unspecified")
        self.source = spec.get("source", "")

    def dependencies(self) -> List[str]:
        """返回 SPEC 声明的数据字段，旧 SPEC 默认依赖 OHLCV。"""
        return _spec_dependencies(self.spec)

    def compute(self, data, dates, universe) -> pd.DataFrame:
        """计算因子矩阵 (日期 × 品种) — 向量化版本.

        一次性获取所有品种的 OHLCV 数据, 向量化计算 base+transform.
        当 frequency != "daily" 时, 走分钟级数据路径.
        """
        params = self.spec.get("params", {})
        base = self.spec["base"]
        transform = self.spec.get("transform", "raw")
        freq = self.spec.get("frequency", "daily")

        # 分钟级因子: 从 DDB 获取分钟 OHLCV, 计算后按日重采样
        if freq != "daily":
            return self._compute_intraday(
                data, dates, universe, base, transform, params, freq,
            )

        # 日度因子: 原有路径
        fields = _spec_dependencies(self.spec)
        ohlcv_data = {}
        for f in fields:
            df = data.get(f, dates, universe)
            if df.empty:
                return pd.DataFrame(np.nan, index=dates, columns=universe)
            ohlcv_data[f] = df

        # 向量化计算 base + transform (所有品种同时)
        try:
            signal = compute_base_df(base, params, ohlcv_data)
            transformed = apply_transform_df(signal, transform, params, ohlcv_data)
            result = _apply_spec_decision_lag(
                transformed, self.spec
            ).reindex(index=dates, columns=universe)
            return result
        except Exception:
            logging.getLogger("multi_factor").exception(f"SpecFactor '{self.name}' 计算失败")
            return pd.DataFrame(np.nan, index=dates, columns=universe)

    def _compute_intraday(
        self, data, dates, universe, base, transform, params, freq,
    ) -> pd.DataFrame:
        """分钟级因子计算: 获取分钟数据 → base+transform → 按日重采样.

        当 DDB 不可用时返回 NaN (优雅降级).
        """
        # 1. 获取分钟级 OHLCV
        minute_ohlcv = _fetch_intraday_ohlcv(data, dates, universe, freq=freq)
        if not minute_ohlcv:
            return pd.DataFrame(np.nan, index=dates, columns=universe)

        # 2. 向量化计算 base + transform (在分钟级数据上)
        try:
            signal = compute_base_df(base, params, minute_ohlcv)
            transformed = apply_transform_df(
                signal, transform, params, minute_ohlcv,
            )
        except Exception:
            logging.getLogger("multi_factor").exception(
                f"SpecFactor '{self.name}' 分钟级计算失败"
            )
            return pd.DataFrame(np.nan, index=dates, columns=universe)

        return _finalize_intraday_result(
            transformed, data, dates, universe, self.spec
        )


# ---------------------------------------------------------------------------
# 批量注册辅助函数
# ---------------------------------------------------------------------------

def register_spec_factor(spec: dict) -> type:
    """根据 SPEC 字典动态创建并注册一个 Factor 子类.

    Args:
        spec: 因子规格字典, 必须包含 'slug', 'base', 'transform'.

    Returns:
        注册后的 Factor 子类.
    """
    slug = spec["slug"]
    category = spec.get("category", "")
    description = spec.get("description", spec.get("name_cn", ""))

    # 动态创建类
    cls = type(
        slug.title().replace("_", ""),
        (Factor,),
        {
            "name": slug,
            "category": category,
            # 周期单位: 从 SPEC 字典读取, 默认 "daily" (向后兼容)
            "frequency": spec.get("frequency", "daily"),
            "description": description,
            "research_tier": spec.get("research_tier", "core"),
            "expected_direction": spec.get("expected_direction", "unspecified"),
            "source": spec.get("source", ""),
            "dependencies": lambda self: _spec_dependencies(spec),
            "compute": _create_compute_method(spec),
        },
    )

    register_factor(slug, category=category)(cls)
    return cls


def _create_compute_method(spec: dict):
    """为动态类生成 compute 方法."""

    def compute(self, data, dates, universe) -> pd.DataFrame:
        params = spec.get("params", {})
        base = spec["base"]
        transform = spec.get("transform", "raw")
        freq = spec.get("frequency", "daily")

        # 分钟级因子: 走分钟数据路径
        if freq != "daily":
            minute_ohlcv = _fetch_intraday_ohlcv(data, dates, universe, freq=freq)
            if not minute_ohlcv:
                return pd.DataFrame(np.nan, index=dates, columns=universe)
            try:
                signal = compute_base_df(base, params, minute_ohlcv)
                transformed = apply_transform_df(
                    signal, transform, params, minute_ohlcv,
                )
            except Exception:
                logging.getLogger("multi_factor").exception(
                    f"动态 SpecFactor '{spec['slug']}' 分钟级计算失败"
                )
                return pd.DataFrame(np.nan, index=dates, columns=universe)
            return _finalize_intraday_result(
                transformed, data, dates, universe, spec
            )

        # 日度因子: 原有路径
        fields = _spec_dependencies(spec)
        ohlcv_data = {}
        for f in fields:
            df = data.get(f, dates, universe)
            if df.empty:
                return pd.DataFrame(np.nan, index=dates, columns=universe)
            ohlcv_data[f] = df

        try:
            signal = compute_base_df(base, params, ohlcv_data)
            transformed = apply_transform_df(signal, transform, params, ohlcv_data)
            result = _apply_spec_decision_lag(
                transformed, spec
            ).reindex(index=dates, columns=universe)
            return result
        except Exception:
            logging.getLogger("multi_factor").exception(f"动态 SpecFactor 计算失败")
            return pd.DataFrame(np.nan, index=dates, columns=universe)

    return compute


# ---------------------------------------------------------------------------
# 批量计算 (性能优化: 按 base+window 分组, 避免重复计算 base)
# ---------------------------------------------------------------------------

def _spec_group_key(spec: dict) -> tuple:
    """生成 SPEC 因子的分组键 (base, window, fast, slow).

    同一分组键的因子共享 base 计算, 仅 transform 不同.
    fast/slow 参数纳入键以处理 MACD/NEWDMA 等需要快慢线的因子.
    """
    params = spec.get("params", {})
    return (
        spec["base"],
        int(params.get("window", 20)),
        int(params.get("fast", 0)),
        int(params.get("slow", 0)),
    )


def _compute_intraday_spec_factors_batch(
    specs: list,
    data,
    dates,
    universe,
    period_ctx=None,
) -> Dict[str, pd.DataFrame]:
    """Compute non-daily SPEC groups without falling back to daily OHLCV."""
    log = logging.getLogger("multi_factor")
    result: Dict[str, pd.DataFrame] = {}
    by_frequency: Dict[str, list] = {}
    for spec in specs:
        by_frequency.setdefault(spec.get("frequency", "daily"), []).append(spec)

    for frequency, frequency_specs in by_frequency.items():
        provider_frequency = getattr(data, "frequency", None)
        if (
            provider_frequency is not None
            and _normalise_frequency(provider_frequency)
            != _normalise_frequency(frequency)
        ):
            log.warning(
                "SPEC batch: provider frequency %s is incompatible with %s; "
                "%d factors remain invalid",
                provider_frequency,
                frequency,
                len(frequency_specs),
            )
            for spec in frequency_specs:
                result[spec["slug"]] = pd.DataFrame(
                    np.nan, index=dates, columns=universe
                )
            continue

        minute_ohlcv = _fetch_intraday_ohlcv(
            data, dates, universe, freq=frequency
        )
        if not minute_ohlcv:
            log.warning(
                "SPEC batch: %s data unavailable; %d factors remain invalid",
                frequency,
                len(frequency_specs),
            )
            for spec in frequency_specs:
                result[spec["slug"]] = pd.DataFrame(
                    np.nan, index=dates, columns=universe
                )
            continue

        minute_ohlcv["_return_1d"] = minute_ohlcv["close"].astype(
            float, copy=False
        ).pct_change(fill_method=None)

        groups: Dict[tuple, list] = {}
        for spec in frequency_specs:
            groups.setdefault(_spec_group_key(spec), []).append(spec)

        for key, group_specs in groups.items():
            try:
                signal = compute_base_df(
                    key[0], group_specs[0].get("params", {}), minute_ohlcv,
                    period_ctx,
                )
            except Exception:
                log.warning(
                    "Intraday SPEC base '%s' failed", key[0], exc_info=True
                )
                for spec in group_specs:
                    result[spec["slug"]] = pd.DataFrame(
                        np.nan, index=dates, columns=universe
                    )
                continue

            for spec in group_specs:
                try:
                    transformed = apply_transform_df(
                        signal,
                        spec.get("transform", "raw"),
                        spec.get("params", {}),
                        minute_ohlcv,
                        period_ctx,
                    )
                    result[spec["slug"]] = _finalize_intraday_result(
                        transformed, data, dates, universe, spec
                    )
                except Exception:
                    log.warning(
                        "Intraday SPEC '%s' failed", spec["slug"], exc_info=True
                    )
                    result[spec["slug"]] = pd.DataFrame(
                        np.nan, index=dates, columns=universe
                    )
    return result


def compute_spec_factors_batch(
    specs: list,
    data,
    dates,
    universe,
    period_ctx=None,
) -> Dict[str, pd.DataFrame]:
    """批量计算 SPEC 因子 (性能优化版本).

    按 (base, window, fast, slow) 分组, 每组只计算一次 base 信号,
    然后对组内所有 transform 复用该 base 信号. 相比逐因子独立计算,
    可将 base 计算次数从因子数降到独立参数组数.

    设计考虑:
    - 异常隔离: 单个分组失败不影响其他分组, 返回 NaN 矩阵
    - OHLCV 复用: 所有分组共享同一份 OHLCV 数据, 一次获取
    - 通用性: 支持任意 base/transform/params 组合, 未来扩展无需改动
    - 向后兼容: SpecFactor.compute() 单因子接口不变

    Args:
        specs: SPEC 字典列表, 每个包含 slug/base/transform/params.
        data: DataManager 实例.
        dates: 日期索引.
        universe: 品种池.
        period_ctx: 周期上下文 (core.period.PeriodContext, 可选).
            透传给 compute_base_df / apply_transform_df.
            当前未参与计算 (预留接口). None 时行为与现有完全一致.

    Returns:
        {slug: FactorMatrix} 字典.
    """
    log = logging.getLogger("multi_factor")
    result: Dict[str, pd.DataFrame] = {}
    if not specs:
        return result

    intraday_specs = [
        spec for spec in specs if spec.get("frequency", "daily") != "daily"
    ]
    daily_specs = [
        spec for spec in specs if spec.get("frequency", "daily") == "daily"
    ]
    if intraday_specs:
        result.update(
            _compute_intraday_spec_factors_batch(
                intraday_specs, data, dates, universe, period_ctx
            )
        )
    specs = daily_specs
    if not specs:
        return result

    # 1. 一次性获取所有已声明字段 (所有分组共享)
    fields = list(
        dict.fromkeys(
            field for spec in specs for field in _spec_dependencies(spec)
        )
    )
    ohlcv_data: Dict[str, pd.DataFrame] = {}
    unavailable_fields = set()
    for f in fields:
        try:
            df = data.get(f, dates, universe)
        except Exception:
            log.warning(
                "SPEC batch: source field '%s' failed to load", f,
                exc_info=True,
            )
            unavailable_fields.add(f)
            continue
        if df.empty:
            log.warning("SPEC batch: source field '%s' is unavailable", f)
            unavailable_fields.add(f)
            continue
        ohlcv_data[f] = df

    if "close" in ohlcv_data:
        ohlcv_data["_return_1d"] = ohlcv_data["close"].astype(
            float, copy=False
        ).pct_change(fill_method=None)

    # 2. 按 (base, window, fast, slow) 分组
    groups: Dict[tuple, list] = {}
    for spec in specs:
        key = _spec_group_key(spec)
        groups.setdefault(key, []).append(spec)

    log.info(
        f"SPEC 批量计算: {len(specs)} 个因子分为 {len(groups)} 组 "
        f"(平均 {len(specs) / max(len(groups), 1):.1f} 因子/组)"
    )
    target_index = pd.Index(dates)
    target_columns = pd.Index(universe)

    # 3. 逐组计算: 每组算一次 base, 批量应用 transform
    for key, group_specs in groups.items():
        base_name = key[0]
        required_fields = set(_spec_dependencies(group_specs[0]))
        missing_fields = sorted(required_fields & unavailable_fields)
        if missing_fields:
            log.warning(
                "SPEC batch: base '%s' skipped; missing fields: %s",
                base_name,
                ", ".join(missing_fields),
            )
            for spec in group_specs:
                result[spec["slug"]] = pd.DataFrame(
                    np.nan, index=dates, columns=universe
                )
            continue
        # 同组内 params 可能略有差异 (如 norm/lag/smooth), 取首个的 base 相关参数
        ref_params = group_specs[0].get("params", {})

        # 计算 base 信号 (整组共享)
        try:
            signal = compute_base_df(base_name, ref_params, ohlcv_data, period_ctx)
        except Exception:
            log.warning(
                f"SPEC 批量计算: base '{base_name}' 计算失败, 组内 "
                f"{len(group_specs)} 个因子返回 NaN",
                exc_info=True,
            )
            for spec in group_specs:
                result[spec["slug"]] = pd.DataFrame(
                    np.nan, index=dates, columns=universe
                )
            continue

        # 对组内每个 transform 应用变换 (复用同一份 signal)
        for spec in group_specs:
            slug = spec["slug"]
            transform = spec.get("transform", "raw")
            params = spec.get("params", {})
            try:
                transformed = apply_transform_df(
                    signal, transform, params, ohlcv_data, period_ctx
                )
                if (
                    transformed.index.equals(target_index)
                    and transformed.columns.equals(target_columns)
                ):
                    result[slug] = _apply_spec_decision_lag(
                        transformed, spec
                    )
                else:
                    result[slug] = _apply_spec_decision_lag(
                        transformed, spec
                    ).reindex(
                        index=dates, columns=universe
                    )
            except Exception:
                log.warning(
                    f"SPEC 批量计算: 因子 '{slug}' transform '{transform}' 失败, 返回 NaN",
                    exc_info=True,
                )
                result[slug] = pd.DataFrame(
                    np.nan, index=dates, columns=universe
                )

    return result


def is_spec_factor(name: str) -> bool:
    """检查因子名是否为 SPEC 因子 (用于引擎路由).

    SPEC 因子命名约定: {base}_{window}d_{transform}
    其中 transform ∈ {z, delta, smooth, rank, vol_scaled, stability, confirm_volume, compress, raw, neg}
    """
    _SPEC_TRANSFORMS = {
        "z", "delta", "smooth", "rank", "vol_scaled",
        "stability", "confirm_volume", "compress", "raw", "neg",
    }
    return any(name.endswith(f"_{transform}") for transform in _SPEC_TRANSFORMS)
