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

═══════════════════════════════════════════════════════════════════════════════
跨日因子总览 (依赖跨交易日数据, 非纯日内):
- 依赖昨收 (prev_close 追踪, 需≥2日):  #27 open_gap_persistence, #28 overnight_absorption,
  #64 overnight_return, #66 overnight_share, #91 anchoring
- 依赖前10日 (rolling(10, min_periods=3), 需≥4日): #116 volatility_breakout, #142 vol_compression
- 依赖昨日量 (prev_total 追踪, 需≥2日): #145 volume_momentum
- 依赖前日高低 (需≥2日): #153 vs_prev_high, #154 vs_prev_low
- 补充批次 (K/V 系列): #K2 overnight_intraday_vol (相邻日收盘, 需≥2日);
  V 系列 rolling/diff 变体基于日度因子值变换, 预热期由基因子覆盖, 无需额外跨日标注

框架支持: compute 一次性接收整个研究期 (含预热期), rolling/跨日 dict 为标准模式;
滚动窗口前的观测为 NaN, 有效值从预热期 (ic_start) 后开始. 各因子注释处均有 ⚠ 标注.
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import re

from core.interfaces import Factor
from core.registry import register_factor

# ────────────────────────────────────────────────────────────────────────────
# Bottleneck 加速的 rolling 薄封装 (可选加速, 数值等价已验证).
# 仅用于最热的模式: rolling(20, min_periods=X).mean()/.std(ddof=0)/.sum().
# bottleneck 缺失或 window > len 时自动回退 pandas, 保证任何环境数值一致.
# 验证: bn.move_mean/move_std(ddof=0)/move_sum 与 pandas 对应操作
#       在含 NaN/全NaN/短序列场景下逐元素一致 (差异 <= 1e-15).
# ────────────────────────────────────────────────────────────────────────────
try:
    import bottleneck as _bn
except ImportError:  # pragma: no cover
    _bn = None


def _roll_mean(df: pd.DataFrame, window: int = 20, min_periods: int = 5) -> pd.DataFrame:
    if _bn is None or df.size == 0 or len(df) < window:
        return df.rolling(window, min_periods=min_periods).mean()
    return pd.DataFrame(
        _bn.move_mean(df.values, window=window, min_count=min_periods, axis=0),
        index=df.index, columns=df.columns,
    )


def _roll_std(df: pd.DataFrame, window: int = 20, min_periods: int = 5) -> pd.DataFrame:
    if _bn is None or df.size == 0 or len(df) < window:
        return df.rolling(window, min_periods=min_periods).std(ddof=0)
    return pd.DataFrame(
        _bn.move_std(df.values, window=window, min_count=min_periods, axis=0),
        index=df.index, columns=df.columns,
    )


def _roll_sum(df: pd.DataFrame, window: int = 20, min_periods: int = 3) -> pd.DataFrame:
    if _bn is None or df.size == 0 or len(df) < window:
        return df.rolling(window, min_periods=min_periods).sum()
    return pd.DataFrame(
        _bn.move_sum(df.values, window=window, min_count=min_periods, axis=0),
        index=df.index, columns=df.columns,
    )


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

_MINUTE_FIELDS = ["open", "high", "low", "close", "volume", "amount", "position"]

_LOCAL_MINUTE_ROOT = r"E:\程明杰公司内容\期货行情数据\本地表"

_FREQ_DIR_MAP = {
    "1min": "futureshistoryprices1m",
    "5min": "futureshistoryprices5m",
    "15min": "futureshistoryprices15m",
    "30min": "futureshistoryprices15m",
    "daily": "futureshistoryprices1d",
    "1d": "futureshistoryprices1d",
}


_RAW_CACHE: dict = {}
_MAX_RAW_CACHE_ENTRIES = 4


def _read_local_raw(dates, universe, freq="1min"):
    """读取本地 Parquet 并按根代码映射, 返回含 _ts/root 列的原始数据 (共享读取).

    供 _read_local_minute (OHLCV聚合) 与 _read_local_term (期限结构) 复用.
    读取失败或无匹配合约时返回 None.
    带内存缓存 (按 freq + 日期范围), 消除多因子重复读 parquet 的瓶颈.
    """
    import logging
    log = logging.getLogger("multi_factor")
    dates = pd.DatetimeIndex(dates)
    # 缓存 key: freq + 日期范围 (同一范围复用, 避免重复读 parquet)
    _dmin = pd.Timestamp(dates.min())
    _dmax = pd.Timestamp(dates.max())
    cache_key = (freq, str(_dmin.date()), str(_dmax.date()))
    if cache_key in _RAW_CACHE:
        return _RAW_CACHE[cache_key]
    subdir = _FREQ_DIR_MAP.get(freq)
    if subdir is None:
        return None
    base = os.path.join(_LOCAL_MINUTE_ROOT, subdir)
    if not os.path.isdir(base):
        return None
    start = pd.Timestamp(dates.min())
    end = pd.Timestamp(dates.max())
    months = pd.date_range(start.replace(day=1), end.replace(day=1), freq="MS")
    frames = []
    for month in months:
        partition = "year_month=" + month.strftime("%Y-%m")
        pdir = os.path.join(base, partition)
        if not os.path.isdir(pdir):
            continue
        # 读取分区内全部 parquet 分片: data_0~data_N.parquet (1m/15m/1d 多分片)
        # 或 part.parquet (5m)。仅读 data_0 会漏掉多分片月份 (如 2024-11 的 data_0~data_3)。
        import glob as _glob
        parquet_files = sorted(_glob.glob(os.path.join(pdir, "data_*.parquet")))
        if not parquet_files:
            parquet_files = sorted(_glob.glob(os.path.join(pdir, "part.parquet")))
        for parquet_path in parquet_files:
            try:
                # 列裁剪: 只读因子计算实际需要的列 (省 exchange/type/trade_date/year_month 4列, ~30% IO)
                cols = ["trade_datetime", "symbol", "open", "high", "low", "close",
                        "volume", "amount", "position"]
                df = pd.read_parquet(parquet_path, columns=cols)
                ts = pd.to_datetime(df["trade_datetime"])
                df = df.loc[(ts >= start) & (ts <= end)]
                if not df.empty:
                    frames.append(df)
            except Exception:
                log.debug("读取 %s 失败", parquet_path, exc_info=True)
    if not frames:
        return None
    all_data = pd.concat(frames, ignore_index=True)
    if all_data.empty:
        return None
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
        return None
    all_data = all_data[all_data["symbol"].isin(symbol_root)]
    all_data["root"] = all_data["symbol"].map(symbol_root)
    ts = pd.to_datetime(all_data["trade_datetime"])
    # 交易日归属: 仅凌晨段 (00:00-07:59, 即前一夜盘尾) 归入前一交易日
    # 例: 周五夜盘 21:00-周六 02:30 的 bar (自然日周六 00:00-02:30) 归入周五
    # 日盘 (08:00 后) 保持当天; 与 1d 表 trade_date 对齐
    all_data["_ts"] = ts.where(ts.dt.hour >= 8, ts - pd.Timedelta(hours=8))
    # 存入缓存 (LRU: 超限淘汰最旧)
    if len(_RAW_CACHE) >= _MAX_RAW_CACHE_ENTRIES:
        _RAW_CACHE.pop(next(iter(_RAW_CACHE)))
    _RAW_CACHE[cache_key] = all_data
    return all_data


def _read_local_minute(dates, universe, freq="1min"):
    """从本地 Parquet 读取分钟/日度 OHLCV 面板, 按根代码聚合."""
    all_data = _read_local_raw(dates, universe, freq=freq)
    if all_data is None:
        return {}
    ts = all_data["_ts"]
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
    if "position" in all_data.columns:
        # 持仓量(OI)是存量: 每个 (ts, root, symbol) 取该分钟最后一笔,
        # 再在 (ts, root) 内取持仓最大的合约(主力合约)的 position 作为该 root 的代表值.
        pos = (all_data.sort_values("_ts")
               .groupby(["_ts", "root", "symbol"])["position"].last().reset_index())
        idx = pos.groupby(["_ts", "root"])["position"].idxmax()
        main_pos = pos.loc[idx].set_index(["_ts", "root"])["position"]
        s = main_pos.unstack(level="root")
        s.index = pd.DatetimeIndex(s.index)
        panel["position"] = s
    return panel


_PANEL_CACHE: dict = {}
_MAX_PANEL_CACHE_ENTRIES = 8


def _panel_cache_key(dates, universe, freq):
    d0 = pd.Timestamp(dates.min())
    d1 = pd.Timestamp(dates.max())
    return (d0, d1, tuple(sorted(str(u) for u in universe)), freq)


# 日内数据频率开关: "1min" (默认) 或 "5min" (加速, 已验证因子值/回测 100% 一致)
# 2026-08-06 准入验证 (scripts/verify_5min_switch.py) 全部 PASS:
#   覆盖一致 / 6因子值相对差 0.00% 方向一致 100% / 回测夏普 2.15 完全相同
#   且 5min 回测 67s vs 1min 555s (8倍加速)
# 切换: 设 _INTRADAY_FREQ = "5min" 后, 所有 _get_minute_panel(freq="1min") 自动用 5min 数据
_INTRADAY_FREQ = "5min"

# 需强制真 1min 的因子 (声明集): 即使全局切 5min 也强制用 1min 数据
# 规则: 因子 compute 里调用 _get_minute_panel(..., force_1min=True) 或
#       将因子名加入 _REQUIRE_1MIN_FACTORS 集合
# 当前: 全量 53 因子已验证 5min 与 1min 值 100% 一致 (verify_53_consistency.py),
#       无一需要真 1min. 未来新增真 1min 因子 (微结构/订单流类) 时在此声明.
_REQUIRE_1MIN_FACTORS: set = set()


def _get_minute_panel(data, dates, universe, freq="1min", force_1min=False):
    """获取分钟级 OHLCV 面板.

    优先级: 本地 Parquet > data.get_at_frequency() > DDBSource.
    带模块级缓存: 同一 (日期区间, 品种池, 频率) 组合只读一次 Parquet,
    多个因子共享面板, 避免 N 因子重复读取 N 次.

    频率准入 (防 1min/5min 错配):
    - 因子请求 freq="1min" 时, 若全局 _INTRADAY_FREQ != "1min" 且
      未声明 force_1min, 则改用 _INTRADAY_FREQ (默认降采样到 5min)
    - force_1min=True: 强制用真 1min (不降采样)
    - 因子请求 freq="5min"/其他: 直接用请求频率 (不映射)
    """
    import logging
    log = logging.getLogger("multi_factor")
    dates = pd.DatetimeIndex(dates)
    # 准入映射: 请求 1min -> 若开关开启且非强制真1min, 用开关频率
    if freq == "1min" and _INTRADAY_FREQ != "1min" and not force_1min:
        freq = _INTRADAY_FREQ
    cache_key = _panel_cache_key(dates, universe, freq)
    if cache_key in _PANEL_CACHE:
        return _PANEL_CACHE[cache_key]
    # 1) 本地 Parquet
    try:
        panel = _read_local_minute(dates, universe, freq=freq)
        if panel:
            if len(_PANEL_CACHE) >= _MAX_PANEL_CACHE_ENTRIES:
                _PANEL_CACHE.pop(next(iter(_PANEL_CACHE)))
            _PANEL_CACHE[cache_key] = panel
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


# ═══════════════════════════════════════════════════════════════════════════════
# 期限结构管道 (主连 + 次连): 用于跨期价差类因子
# ═══════════════════════════════════════════════════════════════════════════════

_TERM_CACHE: dict = {}
_MAX_TERM_CACHE_ENTRIES = 8


from functools import lru_cache


@lru_cache(maxsize=65536)
def _expiry_ym(sym):
    # symbol 形如 RB2610 / IC2609 → (年, 月) 或 None; 月份须 1-12
    # lru_cache: 每个唯一 symbol 只正则一次 (此前 6400 万次调用, 回测瓶颈)
    m = re.match(r"^[A-Za-z]+(\d{2})(\d{2})$", str(sym))
    if not m:
        return None
    yy, mm = 2000 + int(m.group(1)), int(m.group(2))
    if not 1 <= mm <= 12:
        return None
    return yy, mm


def _read_local_term(dates, universe, freq="1min"):
    """从本地 Parquet 读取主连/次连合约面板 (期限结构).

    对每个 (datetime, root):
      - 排除合成合约 (8888/9998/9999, 其持仓/价格语义与真实合约不同);
      - 从 symbol 解析到期月 (YYMM), 按到期日最近排序取 near(最近)/far(次近),
        替代旧版"持仓 top2" (旧版会选到远季或倒挂, 且可能命中合成合约).
      输出面板(与旧版字段名一致, 旧因子输入零改动):
      {near_close, far_close, near_position, far_position, near_volume, far_volume}
      仅当某时刻存在≥2个活跃合约时 far 才非空, 否则 far 为 NaN (因子自动降级).
      新增字段(供新因子使用):
      {near_expiry, far_expiry, remaining_days, annualized_basis,
       roll_yield_annualized, rollover_flag}
      - near_expiry/far_expiry: 到期年月 (YYYYMM 整数);
      - remaining_days: near 距到期月的剩余自然日 (到期日=该月最后一天, 最小 1);
      - annualized_basis: (far-near)/near * 365/remaining_days (正=contango);
      - roll_yield_annualized: (near-far)/far * 365/remaining_days (正=近月升水/backwardation);
      - rollover_flag: near 到期月相对上一根 bar 变化(换月点)±1 根 K 线标记为 1, 其余 0.
    """
    _SYN_SUFFIXES = ("8888", "9998", "9999")

    all_data = _read_local_raw(dates, universe, freq=freq)
    if all_data is None or "position" not in all_data.columns:
        return {}
    # 排除合成合约
    concrete = all_data[~all_data["symbol"].astype(str).str.endswith(_SYN_SUFFIXES)].copy()
    if concrete.empty:
        return {}
    # 每个 (ts, root, symbol) 取最后一笔 position (持仓是存量)
    per_symbol = (concrete.sort_values("_ts")
                  .groupby(["_ts", "root", "symbol"], as_index=False)
                  .agg({"close": "last", "position": "last", "volume": "sum"}))
    # 解析到期月并取最近两月
    per_symbol["_expiry"] = per_symbol["symbol"].map(_expiry_ym)
    per_symbol = per_symbol.dropna(subset=["_expiry"])
    if per_symbol.empty:
        return {}
    per_symbol = per_symbol.sort_values(["_ts", "root", "_expiry"])
    # 每个 (ts, root) 到期月排名: 最近=near(1), 次近=far(2)
    per_symbol["_rank"] = per_symbol.groupby(["_ts", "root"]).cumcount() + 1
    near = per_symbol[per_symbol["_rank"] == 1].set_index(["_ts", "root"])
    far = per_symbol[per_symbol["_rank"] == 2].set_index(["_ts", "root"])
    panel = {}
    for field, src in [("close", near), ("position", near), ("volume", near)]:
        s = src[field].unstack(level="root")
        s.index = pd.DatetimeIndex(s.index)
        panel[f"near_{field}"] = s
    for field, src in [("close", far), ("position", far), ("volume", far)]:
        s = src[field].unstack(level="root")
        s.index = pd.DatetimeIndex(s.index)
        panel[f"far_{field}"] = s
    # 兼容旧因子: 旧代码假设 near/far 行数一致 (far.loc[day==dt] 布尔索引).
    # 修复面板后 far 可能只在有≥2活跃合约的时刻存在, 这里把 far 面板
    # reindex 到 near 的行索引 (缺失置 NaN), 使旧因子代码零改动仍可运行.
    near_idx = panel["near_close"].index
    for key in list(panel):
        if key.startswith("far_"):
            panel[key] = panel[key].reindex(near_idx)
    # 剩余天数: near 到期月最后一天 - 当前日期 (自然日, 最小1, 上界365防过期放大); 显式同索引 Series 相减
    near_exp = near["_expiry"].apply(lambda ym: (ym[0], ym[1]))

    def _expiry_date(ym):
        y, m = ym
        return pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)

    exp_dates = near_exp.apply(_expiry_date)
    ts_series = pd.Series(pd.DatetimeIndex(near_exp.index.get_level_values("_ts")),
                          index=near_exp.index)
    rem = ((exp_dates - ts_series) / pd.Timedelta(days=1)).clip(lower=1, upper=365)
    s_rem = rem.unstack(level="root")
    s_rem.index = pd.DatetimeIndex(s_rem.index)
    panel["remaining_days"] = s_rem
    # near/far 到期月 (YYYYMM 整数, far 与 far_close 同索引)
    for key, src in [("near_expiry", near), ("far_expiry", far)]:
        s = src["_expiry"].apply(lambda ym: ym[0] * 100 + ym[1]).unstack(level="root")
        s.index = pd.DatetimeIndex(s.index)
        if key.startswith("far_"):
            s = s.reindex(near_idx)
        panel[key] = s
    # 年化基差率 (正=contango 远高近)
    near_c, far_c = panel["near_close"], panel["far_close"]
    rem_p = panel["remaining_days"].replace(0, np.nan)
    ann_basis = ((far_c - near_c) / near_c.replace(0, np.nan)) * (365.0 / rem_p)
    roll_ann = ((near_c - far_c) / far_c.replace(0, np.nan)) * (365.0 / rem_p)
    panel["annualized_basis"] = ann_basis
    panel["roll_yield_annualized"] = roll_ann
    # 换月标记: near 到期月相对上一根 bar 变化 (换月点) ±1 根 K 线
    # 注: 基于相邻 bar 的 near_expiry 比较; near 临时缺失时由 notna 保护避免误标
    exp_s = panel["near_expiry"].copy()
    prev = exp_s.shift(1)
    roll_day = exp_s.notna() & prev.notna() & (exp_s != prev)
    flag = (roll_day | roll_day.shift(1, fill_value=False) | roll_day.shift(-1, fill_value=False)).astype(float)
    panel["rollover_flag"] = flag
    return panel


def _get_term_structure_panel(data, dates, universe, freq="1min"):
    """获取期限结构面板 (主连+次连), 优先本地 Parquet, 带缓存."""
    import logging
    log = logging.getLogger("multi_factor")
    dates = pd.DatetimeIndex(dates)
    cache_key = _panel_cache_key(dates, universe, freq) + ("term",)
    if cache_key in _TERM_CACHE:
        return _TERM_CACHE[cache_key]
    try:
        panel = _read_local_term(dates, universe, freq=freq)
        if panel:
            if len(_TERM_CACHE) >= _MAX_TERM_CACHE_ENTRIES:
                _TERM_CACHE.pop(next(iter(_TERM_CACHE)))
            _TERM_CACHE[cache_key] = panel
            return panel
    except Exception:
        log.debug("本地期限结构读取失败", exc_info=True)
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# 日度结算/持仓数据: 从 futureshistoryprices1d 读取 (ths_data_daily.db 已弃用)
# settle → 1d 的 settle_price 字段; oi → 1d 的 position 字段
# ═══════════════════════════════════════════════════════════════════════════════

_FIELD_TO_1D_COL = {"settle": "settle_price", "oi": "position"}


def _read_local_daily(data, dates, universe, field="settle"):
    """获取日度结算/持仓面板, 从 futureshistoryprices1d 读取.

    settle 对应 1d 的 settle_price (结算价, 日度概念), oi 对应 position.
    对每个 (date, root) 取持仓(oi)最大的主力合约的字段值.
    数据不可得时返回全 NaN (因子自动降级).
    """
    import logging
    log = logging.getLogger("multi_factor")
    dates = pd.DatetimeIndex(dates)
    col = _FIELD_TO_1D_COL.get(field)
    if col is None:
        return pd.DataFrame(np.nan, index=dates, columns=universe)
    # 1) 数据源优先 (未来数据源支持 settle_price/position 时自动生效)
    #    注: 该分支无法获取 symbol 级信息, 不执行 B 阶段的合成排除/换月日 NaN/日期对齐;
    #    换月清洗仅作用于本地 1d Parquet 分支. 当前无数据源实现, 无实际影响.
    try:
        frame = data.get(col, dates, universe)
        if frame is not None and not frame.empty and frame.notna().any().any():
            return frame.reindex(index=dates, columns=universe)
    except Exception:
        pass
    # 2) 本地 1d Parquet (futureshistoryprices1d)
    try:
        raw = _read_local_raw(dates, universe, freq="daily")
        if raw is None or raw.empty or col not in raw.columns:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        _cols = ["_ts", "root", "position", "symbol"]
        if "trade_date" in raw.columns:
            _cols.insert(1, "trade_date")
        if col not in _cols:  # oi 时 col=position 已在列中, 避免重复列名
            _cols.append(col)
        if "position" not in raw.columns:  # dropna(subset=["position"]) 的前提
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = raw[_cols].copy()
        # bug 修正(2026-08, B 阶段): 1d 数据的 _ts 为"前一自然日 16:00"(bar 时间),
        # 直接 normalize 会把交易日标签错位一天(前视/滞后); 改用 trade_date 交易日对齐
        if "trade_date" in raw.columns:
            df["_ts"] = pd.to_datetime(df["trade_date"]).dt.normalize()
        else:
            df["_ts"] = pd.to_datetime(df["_ts"]).dt.normalize()
        df = df.dropna(subset=["position"])
        if df.empty:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        # bug 修正(2026-08, B 阶段): 排除合成/连续合约(00/01M/8888/纯根名),
        # 只保留可解析出合法到期月(YYMM, 月份 1-12)的具体合约, 避免主力取到合成价
        df = df[df["symbol"].map(_expiry_ym).notna()]
        if df.empty:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        # 取每 (date, root) 持仓最大的主力合约的字段值
        idx = df.groupby(["_ts", "root"])["position"].idxmax()
        main = df.loc[idx]
        # bug 修正(2026-08, B 阶段): 主力合约切换日(换月日)该字段置 NaN,
        # 消除跨合约结算价/持仓跳变对日度因子的伪信号污染.
        # 注意: ①不能用 shift(1) 按索引对齐(main 索引不连续), 用 numpy 位置对比;
        #       ②必须按 (日期, 品种) 单元格置 NaN, 不能整行置 NaN(否则误伤当日
        #       未换月的其他品种, 多品种 universe 下会大面积全 NaN).
        roll_cells = set()
        mm = main.sort_values("_ts")
        for _root, _g in mm.groupby("root"):
            syms = _g["symbol"].astype(str).to_numpy()
            tss = pd.to_datetime(_g["_ts"].to_numpy())
            for i in range(1, len(syms)):
                if syms[i] and syms[i - 1] and syms[i] != syms[i - 1]:
                    roll_cells.add((pd.Timestamp(tss[i]).normalize(), _root))
        pivot = main.pivot(index="_ts", columns="root", values=col)
        pivot.index = pd.DatetimeIndex(pivot.index)
        pivot = pivot.reindex(index=dates, columns=universe)
        for _d, _r in roll_cells:
            if _d in pivot.index and _r in pivot.columns:
                pivot.loc[_d, _r] = np.nan
        return pivot
    except Exception:
        log.debug("1d %s 读取失败", col, exc_info=True)
        return pd.DataFrame(np.nan, index=dates, columns=universe)


def _get_rollover_calendar(dates, universe):
    """换月日历: 每品种主力合约切换日(换月日)列表. B 阶段新增, 独立数据产物.

    判定口径与 _read_local_daily 修复后一致: 排除合成/连续合约, 每交易日取
    持仓最大的具体合约(symbol 可解析 YYMM), 主力 symbol 变化日即换月日.
    Returns: dict[root] -> pd.DatetimeIndex(换月交易日, 升序)
    """
    empty = {u: pd.DatetimeIndex([]) for u in universe}
    try:
        raw = _read_local_raw(dates, universe, freq="daily")
        if raw is None or "position" not in raw.columns or "trade_date" not in raw.columns:
            return empty
        df = raw[["trade_date", "root", "position", "symbol"]].copy()
        df["td"] = pd.to_datetime(df["trade_date"]).dt.normalize()
        df = df.dropna(subset=["position"])
        df = df[df["symbol"].map(_expiry_ym).notna()]
        if df.empty:
            return empty
        for root, g in df.groupby("root"):
            top = (g.sort_values(["td", "position"], ascending=[True, False])
                    .groupby("td").head(1).sort_values("td"))
            syms = top["symbol"].astype(str).to_numpy()
            tds = pd.to_datetime(top["td"].to_numpy())
            days = []
            for i in range(1, len(syms)):
                if syms[i] and syms[i - 1] and syms[i] != syms[i - 1]:
                    days.append(pd.Timestamp(tds[i]).normalize())
            empty[root] = pd.DatetimeIndex(sorted(days))
        return empty
    except Exception:
        return empty


# ═══════════════════════════════════════════════════════════════════════════════
# 席位数据管道 (日度, futuresseatdata/derive_product_daily): 品种级多空持仓汇总
# ═══════════════════════════════════════════════════════════════════════════════

_SEAT_ROOT = os.path.join(_LOCAL_MINUTE_ROOT, "futuresseatdata", "derive_product_daily")
_SEAT_FIELDS = ["total_long", "total_short", "net_position", "long_change", "short_change", "seat_count"]
_SEAT_CACHE: dict = {}


def _read_local_seat(dates, universe):
    """读取席位日度面板 (品种级多空汇总), 返回 {field: DataFrame(dates×universe)}.

    席位数据仅日度频率, 每交易日每条 (2020-03 起). 数据不可得时字段缺失.
    """
    import logging
    log = logging.getLogger("multi_factor")
    dates = pd.DatetimeIndex(dates)
    if not os.path.isdir(_SEAT_ROOT):
        return {}
    start = pd.Timestamp(dates.min())
    end = pd.Timestamp(dates.max())
    months = pd.date_range(start.replace(day=1), end.replace(day=1), freq="MS")
    frames = []
    for month in months:
        partition = "year_month=" + month.strftime("%Y-%m")
        parquet_path = os.path.join(_SEAT_ROOT, partition, "data_0.parquet")
        if not os.path.exists(parquet_path):
            continue
        try:
            df = pd.read_parquet(parquet_path)
            td = pd.to_datetime(df["trade_date"])
            df = df.loc[(td >= start) & (td <= end)]
            if not df.empty:
                frames.append(df)
        except Exception:
            log.debug("席位读取失败 %s", parquet_path, exc_info=True)
    if not frames:
        return {}
    all_data = pd.concat(frames, ignore_index=True)
    if all_data.empty:
        return {}
    universe_sorted = sorted({str(u).upper() for u in universe}, key=len, reverse=True)
    root_map: dict[str, str] = {}
    for r in all_data["root"].unique():
        r_up = str(r).upper()
        for ut in universe_sorted:
            if r_up.startswith(ut):
                root_map[r] = ut
                break
    all_data["_root"] = all_data["root"].map(root_map)
    all_data = all_data[all_data["_root"].notna()]
    if all_data.empty:
        return {}
    panel = {}
    for field in _SEAT_FIELDS:
        if field not in all_data.columns:
            continue
        s = all_data.pivot_table(index="trade_date", columns="_root", values=field, aggfunc="last")
        s.index = pd.DatetimeIndex(s.index)
        panel[field] = s.reindex(index=dates, columns=universe)
    return panel


def _get_seat_panel(data, dates, universe):
    """获取席位日度面板 (带缓存). 返回 {field: DataFrame}."""
    import logging
    log = logging.getLogger("multi_factor")
    dates = pd.DatetimeIndex(dates)
    cache_key = _panel_cache_key(dates, universe, "daily") + ("seat",)
    if cache_key in _SEAT_CACHE:
        return _SEAT_CACHE[cache_key]
    try:
        panel = _read_local_seat(dates, universe)
        if panel:
            if len(_SEAT_CACHE) >= _MAX_PANEL_CACHE_ENTRIES:
                _SEAT_CACHE.pop(next(iter(_SEAT_CACHE)))
            _SEAT_CACHE[cache_key] = panel
            return panel
    except Exception:
        log.debug("本地席位读取失败", exc_info=True)
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# 通用席位表管道: 支持全部 6 张表 (product_daily/product_seat/main_contract_seat/
#   raw_seat_position/delivery_summary/delivery_seat)
# ═══════════════════════════════════════════════════════════════════════════════

_SEAT_TABLE_DIRS = {
    "product_daily": "derive_product_daily",
    "product_seat": "derive_product_seat",
    "main_contract_seat": "derive_main_contract_seat",
    "raw_seat": "raw_seat_position",
    "delivery_summary": "delivery_summary",
    "delivery_seat": "delivery_seat",
}
_SEAT_DETAIL_CACHE: dict = {}


def _read_local_seat_table(dates, universe, table):
    """通用读取席位表, 返回过滤+root映射后的原始 DataFrame (含 _ts/_root).

    delivery 表日期列用 delivery_date, 其余用 trade_date.
    """
    import logging
    log = logging.getLogger("multi_factor")
    subdir = _SEAT_TABLE_DIRS.get(table)
    if subdir is None:
        return None
    base = os.path.join(_LOCAL_MINUTE_ROOT, "futuresseatdata", subdir)
    if not os.path.isdir(base):
        return None
    dates = pd.DatetimeIndex(dates)
    date_col = "delivery_date" if "delivery" in table else "trade_date"
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
            td = pd.to_datetime(df[date_col])
            df = df.loc[(td >= start) & (td <= end)]
            if not df.empty:
                frames.append(df)
        except Exception:
            log.debug("席位表 %s 读取失败 %s", table, parquet_path, exc_info=True)
    if not frames:
        return None
    all_data = pd.concat(frames, ignore_index=True)
    if all_data.empty:
        return None
    universe_sorted = sorted({str(u).upper() for u in universe}, key=len, reverse=True)
    root_map: dict[str, str] = {}
    for r in all_data["root"].unique():
        r_up = str(r).upper()
        for ut in universe_sorted:
            if r_up.startswith(ut):
                root_map[r] = ut
                break
    all_data["_root"] = all_data["root"].map(root_map)
    all_data = all_data[all_data["_root"].notna()]
    if all_data.empty:
        return None
    all_data["_ts"] = pd.to_datetime(all_data[date_col])
    return all_data


def _get_seat_table(data, dates, universe, table):
    """获取席位表原始数据 (带缓存). 返回 DataFrame 或 None."""
    import logging
    log = logging.getLogger("multi_factor")
    dates = pd.DatetimeIndex(dates)
    cache_key = _panel_cache_key(dates, universe, "daily") + ("seat", table)
    if cache_key in _SEAT_DETAIL_CACHE:
        return _SEAT_DETAIL_CACHE[cache_key]
    try:
        df = _read_local_seat_table(dates, universe, table)
        if df is not None:
            if len(_SEAT_DETAIL_CACHE) >= _MAX_PANEL_CACHE_ENTRIES * 2:
                _SEAT_DETAIL_CACHE.pop(next(iter(_SEAT_DETAIL_CACHE)))
            _SEAT_DETAIL_CACHE[cache_key] = df
            return df
    except Exception:
        log.debug("席位表 %s 读取失败", table, exc_info=True)
    return None


def _seat_topN_ratio(all_data, field, n=5):
    """按 (date, root) 计算指定多空字段的 topN 集中度: Σ topN / Σ 全部."""
    data = all_data[["_ts", "_root", field]].dropna()
    total = data.groupby(["_ts", "_root"])[field].sum()
    top = (data.sort_values(["_ts", "_root", field], ascending=[True, True, False])
           .groupby(["_ts", "_root"]).head(n).groupby(["_ts", "_root"])[field].sum())
    ratio = (top / total.replace(0, np.nan)).reset_index()
    pivot = ratio.pivot(index="_ts", columns="_root", values=field)
    pivot.index = pd.DatetimeIndex(pivot.index)
    return pivot


def _seat_to_panel(all_data, field, agg="last"):
    """按 (date, root) 聚合字段为面板 (agg: last/sum/mean/std)."""
    data = all_data[["_ts", "_root", field]].dropna()
    if agg == "sum":
        s = data.groupby(["_ts", "_root"])[field].sum()
    elif agg == "mean":
        s = data.groupby(["_ts", "_root"])[field].mean()
    elif agg == "std":
        s = data.groupby(["_ts", "_root"])[field].std()
    else:
        s = data.groupby(["_ts", "_root"])[field].last()
    pivot = s.reset_index().pivot(index="_ts", columns="_root", values=field)
    pivot.index = pd.DatetimeIndex(pivot.index)
    return pivot


# ═══════════════════════════════════════════════════════════════════════════════
# 跨品种/板块辅助: 板块映射 + 板块内截面 z-score
# ═══════════════════════════════════════════════════════════════════════════════

_SECTOR_MAP = {
    # 黑色系
    "RB": "black", "HC": "black", "I": "black", "J": "black", "JM": "black",
    "SF": "black", "SM": "black", "FG": "black",
    # 能化
    "MA": "chem", "TA": "chem", "PP": "chem", "L": "chem", "V": "chem",
    "EG": "chem", "FU": "chem", "BU": "chem", "SC": "chem", "RU": "chem",
    "EB": "chem", "PG": "chem", "UR": "chem",
    # 有色
    "CU": "metal", "AL": "metal", "ZN": "metal", "NI": "metal",
    "SN": "metal", "PB": "metal", "SS": "metal",
    # 贵金属
    "AU": "precious", "AG": "precious",
    # 农产品
    "A": "agri", "M": "agri", "Y": "agri", "P": "agri", "C": "agri",
    "CS": "agri", "CF": "agri", "SR": "agri", "OI": "agri", "RM": "agri",
    "JD": "agri", "AP": "agri",
    # 股指
    "IF": "index", "IC": "index", "IH": "index", "IM": "index",
    # 国债
    "T": "bond", "TF": "bond", "TL": "bond", "TS": "bond",
    # 生猪
    "LH": "agri",
}


def _sector_zscore(df):
    """按板块分组的截面 z-score (板块内≥2个品种才计算, 否则全 NaN)."""
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    sectors: dict = {}
    for col in df.columns:
        sec = _SECTOR_MAP.get(str(col).upper(), "other")
        sectors.setdefault(sec, []).append(col)
    for sec, cols in sectors.items():
        if len(cols) < 2:
            continue
        sub = df[cols]
        std = sub.std(axis=1, ddof=0).replace(0, np.nan)
        z = sub.sub(sub.mean(axis=1), axis=0).div(std, axis=0)
        out[cols] = z
    return out


def _sector_mean(df):
    """板块内其他品种的均值 (排除自身, 板块内≥2个品种)."""
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    sectors: dict = {}
    for col in df.columns:
        sec = _SECTOR_MAP.get(str(col).upper(), "other")
        sectors.setdefault(sec, []).append(col)
    for sec, cols in sectors.items():
        if len(cols) < 2:
            continue
        for col in cols:
            others = [c for c in cols if c != col]
            out[col] = df[others].mean(axis=1)
    return out


def _sector_rank(row):
    """按板块内 percentile rank (单日截面, 板块内≥2个品种)."""
    out = pd.Series(index=row.index, dtype=float)
    sectors: dict = {}
    for col in row.index:
        sec = _SECTOR_MAP.get(str(col).upper(), "other")
        sectors.setdefault(sec, []).append(col)
    for sec, cols in sectors.items():
        if len(cols) < 2:
            continue
        vals = row[cols]
        out[cols] = vals.rank(pct=True)
    return out


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
    rm = _roll_mean(frame, 20, 5)
    rs = _roll_std(frame, 20, 5)
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
        smoothed = _roll_mean(daily, 20, 5)
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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


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
        roll_mean = _roll_mean(daily, 20, 5)
        roll_std = _roll_std(daily, 20, 5)
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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


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
        roll_std = _roll_std(daily_vol, 20, 5)
        roll_mean = _roll_mean(daily_vol, 20, 5)
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
        return _roll_sum(daily, 20, 3).reindex(dates).shift(1).reindex(columns=universe)


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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 27. intraday_open_gap_persistence — 开盘缺口持续性
# ⚠ 跨日因子: 依赖昨日收盘 (prev_close 跨日追踪), 需≥2个交易日历史; 首日为NaN
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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 28. intraday_overnight_absorption — 隔夜信息吸收效率
# ⚠ 跨日因子: 依赖昨日收盘 (prev_close 跨日追踪), 需≥2个交易日历史; 首日为NaN
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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


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
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 48. intraday_jump_ratio — 跳跃方差占比 (Bipower Variation 分解)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_jump_ratio_20d", category="intraday_advanced")
class IntradayJumpRatio20d(Factor):
    """跳跃方差占比因子.

    借鉴 #8 jump_intensity 的跳跃思想, 用 Bipower Variation (BNS 2004/2006) 分解波动:
    BV = (π/2)·Σ|r_t|·|r_{t-1}|,  jump_ratio = (RV - BV) / RV.
    jump_ratio 高 → 当日价格由离散跳跃主导 → 博彩型/信息冲击 → 负向.
    方向: 负向.
    """
    name = "intraday_jump_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "跳跃方差占比 ((RV-BV)/RV, Bipower分解)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        ratios: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 30:
                    continue
                rv = float(np.sum(r ** 2))
                # staggered bipower: |r_t|·|r_{t-2}| 对微观结构噪声更稳健
                bv = float((np.pi / 2.0) * np.sum(np.abs(r[2:]) * np.abs(r[:-2])))
                if rv > 1e-12:
                    vals[col] = (rv - bv) / rv
                else:
                    vals[col] = 0.0
            if vals:
                ratios[dt] = pd.Series(vals)
        if not ratios:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(ratios).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 49. intraday_continuous_vol — 连续波动成分 (sqrt BV)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_continuous_vol_20d", category="intraday_advanced")
class IntradayContinuousVol20d(Factor):
    """连续波动成分因子.

    BV = (π/2)·Σ|r_t|·|r_{t-1}| 是剔除跳跃后的纯连续积分方差.
    连续波动低 → 低波动异象 (低风险溢价) → 正向.
    方向: 正向 (取 -sqrt(BV)).
    """
    name = "intraday_continuous_vol_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "连续波动成分 (-sqrt(BV), 低波动异象)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        vols: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 30:
                    continue
                bv = float((np.pi / 2.0) * np.sum(np.abs(r[2:]) * np.abs(r[:-2])))
                vals[col] = -np.sqrt(max(0.0, bv))
            if vals:
                vols[dt] = pd.Series(vals)
        if not vols:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(vols).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 50. intraday_realized_quarticity — 已实现四次矩 (尾部风险)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_realized_quarticity_20d", category="intraday_advanced")
class IntradayRealizedQuarticity20d(Factor):
    """已实现四次矩因子.

    RQ = (M/3)·Σr_t⁴ (Barndorff-Nielsen & Shephard 2002).
    度量收益分布的尾部厚度; RQ 高 → 大波动集中出现 → 尾部风险 → 负向.
    方向: 负向.
    """
    name = "intraday_realized_quarticity_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "已实现四次矩 ((M/3)Σr^4, 尾部风险)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        quartics: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 30:
                    continue
                m = len(r)
                vals[col] = float((m / 3.0) * np.sum(r ** 4))
            if vals:
                quartics[dt] = pd.Series(vals)
        if not quartics:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(quartics).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 51. intraday_downside_semivariance — 下行半方差 (坏波动)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_downside_semivariance_20d", category="intraday_advanced")
class IntradayDownsideSemivariance20d(Factor):
    """下行半方差因子.

    RS⁻ = Σr_t²·1{r_t<0} (BNKS 2010).
    下行波动是"坏波动" (Segal et al. 2015), 杠杆效应下预测未来波动更强 → 负向.
    方向: 负向.
    """
    name = "intraday_downside_semivariance_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "下行半方差 (Σr²·1[r<0], 坏波动)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        downside: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 20:
                    continue
                vals[col] = float((r[r < 0] ** 2).sum())
            if vals:
                downside[dt] = pd.Series(vals)
        if not downside:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(downside).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 52. intraday_signed_jump — 带符号跳跃 (上下半方差之差)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_signed_jump_20d", category="intraday_advanced")
class IntradaySignedJump20d(Factor):
    """带符号跳跃因子.

    (RS⁻ - RS⁺)/RV 收敛于 (负跳方差 - 正跳方差)/总方差 (BNKS 2010).
    正值 → 下行跳跃主导 → 负面冲击 → 负向.
    方向: 负向.
    """
    name = "intraday_signed_jump_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "带符号跳跃 ((RS⁻-RS⁺)/RV, 下行跳跃主导)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        signed_jumps: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 20:
                    continue
                rs_minus = float((r[r < 0] ** 2).sum())
                rs_plus = float((r[r > 0] ** 2).sum())
                rv = rs_minus + rs_plus
                vals[col] = (rs_minus - rs_plus) / rv if rv > 1e-12 else 0.0
            if vals:
                signed_jumps[dt] = pd.Series(vals)
        if not signed_jumps:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(signed_jumps).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 53. intraday_realized_kurtosis — 已实现峰度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_realized_kurtosis_20d", category="intraday_advanced")
class IntradayRealizedKurtosis20d(Factor):
    """已实现峰度因子.

    kurt = M·Σr_t⁴ / RV² (标准化的四次矩).
    高峰度 → 收益分布厚尾 → 极端事件频发 → 博彩型特征 → 负向.
    方向: 负向.
    """
    name = "intraday_realized_kurtosis_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "已实现峰度 (M·Σr^4/RV², 厚尾风险)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        kurtosis: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 30:
                    continue
                rv = float(np.sum(r ** 2))
                if rv > 1e-12:
                    vals[col] = float(len(r) * np.sum(r ** 4) / (rv ** 2))
                else:
                    vals[col] = 0.0
            if vals:
                kurtosis[dt] = pd.Series(vals)
        if not kurtosis:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(kurtosis).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 54. intraday_autocorr_ret — 分钟收益一阶自相关
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_autocorr_ret_20d", category="intraday_advanced")
class IntradayAutocorrRet20d(Factor):
    """分钟收益一阶自相关因子.

    ρ = corr(ret_t, ret_{t-1}) 日内分钟级.
    正自相关 → 分钟动量 → 趋势延续 → 正向.
    负自相关 → 分钟反转 (买卖反弹) → 噪声 → 负向.
    方向: 正向.
    """
    name = "intraday_autocorr_ret_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "分钟收益一阶自相关 (动量vs反转)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        autocorrs: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 30:
                    continue
                r_t, r_tm1 = r[1:], r[:-1]
                if r_t.std(ddof=0) < 1e-12 or r_tm1.std(ddof=0) < 1e-12:
                    vals[col] = 0.0
                else:
                    vals[col] = float(np.corrcoef(r_t, r_tm1)[0, 1])
            if vals:
                autocorrs[dt] = pd.Series(vals)
        if not autocorrs:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(autocorrs).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 55. intraday_volatility_clustering — 波动聚集度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_volatility_clustering_20d", category="intraday_advanced")
class IntradayVolatilityClustering20d(Factor):
    """波动聚集度因子.

    corr(|ret_t|, |ret_{t-1}|) 绝对收益自相关.
    高聚集 → 大波动扎堆出现 → 风险不均衡 → 负向.
    低聚集 → 波动均匀 → 稳定 → 正向.
    方向: 负向.
    """
    name = "intraday_volatility_clustering_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "波动聚集 (|ret|一阶自相关, 高=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        clusters: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 30:
                    continue
                ar = np.abs(r)
                a_t, a_tm1 = ar[1:], ar[:-1]
                if a_t.std(ddof=0) < 1e-12 or a_tm1.std(ddof=0) < 1e-12:
                    vals[col] = 0.0
                else:
                    vals[col] = float(np.corrcoef(a_t, a_tm1)[0, 1])
            if vals:
                clusters[dt] = pd.Series(vals)
        if not clusters:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(clusters).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 56. intraday_variance_ratio_5m — 5分钟方差比 (Lo-MacKinlay)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_variance_ratio_5m_20d", category="intraday_advanced")
class IntradayVarianceRatio5m20d(Factor):
    """5分钟方差比因子 (Lo-MacKinlay 1988).

    VR(q) = Var(5期收益)/(5·Var(1期收益)), 用自相关加权形式:
    VR = 1 + 2Σ(1-k/q)·ρ(k).
    VR>1 → 正自相关 → 动量;  VR<1 → 反转. 因子值 = VR-1.
    方向: 正向.
    """
    name = "intraday_variance_ratio_5m_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "5分钟方差比 (Lo-MacKinlay VR(5)-1)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    @staticmethod
    def _vr(series, q):
        n = len(series)
        if n < 5 * q:
            return 0.0
        mu = series.mean()
        var1 = np.sum((series[1:] - mu) ** 2) / (n - 1)
        if var1 < 1e-12:
            return 0.0
        # autocorrelation form
        vr = 1.0
        for k in range(1, q):
            r_t = series[k:]
            r_tm = series[:-k]
            rho_k = float(np.corrcoef(r_t, r_tm)[0, 1]) if r_t.std() > 1e-12 and r_tm.std() > 1e-12 else 0.0
            vr += 2.0 * (1.0 - k / q) * rho_k
        return float(vr - 1.0)

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        vrs: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 30:
                    continue
                vals[col] = self._vr(r, q=5)
            if vals:
                vrs[dt] = pd.Series(vals)
        if not vrs:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(vrs).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 57. intraday_variance_ratio_30m — 30分钟方差比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_variance_ratio_30m_20d", category="intraday_advanced")
class IntradayVarianceRatio30m20d(Factor):
    """30分钟方差比因子.

    与 #56 同构, 但 q=30, 捕捉更长时间尺度的动量/反转.
    VR(30)>1 → 长尺度趋势持续 → 正向.
    方向: 正向.
    """
    name = "intraday_variance_ratio_30m_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "30分钟方差比 (Lo-MacKinlay VR(30)-1)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    @staticmethod
    def _vr(series, q):
        n = len(series)
        if n < 5 * q:
            return 0.0
        mu = series.mean()
        var1 = np.sum((series[1:] - mu) ** 2) / (n - 1)
        if var1 < 1e-12:
            return 0.0
        vr = 1.0
        for k in range(1, q):
            r_t = series[k:]
            r_tm = series[:-k]
            rho_k = float(np.corrcoef(r_t, r_tm)[0, 1]) if r_t.std() > 1e-12 and r_tm.std() > 1e-12 else 0.0
            vr += 2.0 * (1.0 - k / q) * rho_k
        return float(vr - 1.0)

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        vrs: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 60:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 60:
                    continue
                vals[col] = self._vr(r, q=30)
            if vals:
                vrs[dt] = pd.Series(vals)
        if not vrs:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(vrs).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 58. intraday_direction_persistence — 方向持续性 (sign自相关)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_direction_persistence_20d", category="intraday_advanced")
class IntradayDirectionPersistence20d(Factor):
    """方向持续性因子.

    corr(sign(ret_t), sign(ret_{t-1})) 与零的偏离度.
    持续正 → 同向延续 (与 #41 run 互补, 这里是相关强度而非长度) → 正向.
    方向: 正向.
    """
    name = "intraday_direction_persistence_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "方向持续性 (sign(ret)自相关)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        persist: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 30:
                    continue
                s = np.sign(r)
                s_t, s_tm1 = s[1:], s[:-1]
                if s_t.std(ddof=0) < 1e-12 or s_tm1.std(ddof=0) < 1e-12:
                    vals[col] = 0.0
                else:
                    vals[col] = float(np.corrcoef(s_t, s_tm1)[0, 1])
            if vals:
                persist[dt] = pd.Series(vals)
        if not persist:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(persist).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 59. intraday_vpin — 毒性订单流 VPIN 代理
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_vpin_20d", category="intraday_advanced")
class IntradayVPIN20d(Factor):
    """VPIN 毒性订单流因子 (Easley, Lopez de Prado & O'Hara 2012).

    分钟OHLCV无法获得真实逐笔, 按 BVC (Bulk Volume Classification) 近似:
    每5分钟桶, buy = vol·Φ(ΔP/σ), 不平衡 |buy-sell|/桶量, 日内均值.
    高VPIN → 毒性订单流 → 知情交易活跃 → 信息不对称 → 负向.
    方向: 负向.
    """
    name = "intraday_vpin_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "VPIN代理 (BVC成交量桶不平衡, 高=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        from scipy.stats import norm as _norm
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, volume = panel["close"], panel["volume"]
        day = close.index.normalize()
        vpins: dict = {}
        bucket = 5  # 5分钟桶
        for dt in sorted(set(day)):
            grp_close = close.loc[day == dt]
            grp_vol = volume.loc[day == dt]
            n = len(grp_close)
            if n < 60:
                continue
            vals = {}
            for col in grp_close.columns:
                c = grp_close[col].dropna()
                v = grp_vol[col].dropna()
                common = c.index.intersection(v.index)
                if len(common) < 60:
                    continue
                c_c = c.loc[common].values
                v_c = v.loc[common].values
                n_b = len(c_c) // bucket
                if n_b < 3:
                    continue
                ois = []
                for b in range(n_b):
                    seg_c = c_c[b * bucket:(b + 1) * bucket]
                    seg_v = v_c[b * bucket:(b + 1) * bucket]
                    d_price = seg_c[-1] - seg_c[0]
                    sigma = seg_c.std(ddof=0)
                    if sigma < 1e-12:
                        buy_frac = 0.5 if d_price >= 0 else 0.5
                    else:
                        buy_frac = float(_norm.cdf(d_price / sigma))
                    buy = seg_v.sum() * buy_frac
                    sell = seg_v.sum() * (1.0 - buy_frac)
                    total = buy + sell
                    ois.append(abs(buy - sell) / total if total > 1e-12 else 0.0)
                vals[col] = float(np.mean(ois))
            if vals:
                vpins[dt] = pd.Series(vals)
        if not vpins:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(vpins).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 60. intraday_order_flow_imbalance — 订单流不平衡 (净方向)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_order_flow_imbalance_20d", category="intraday_advanced")
class IntradayOrderFlowImbalance20d(Factor):
    """订单流不平衡因子.

    (Σbuy - Σsell)/Σvol, 用分钟收益方向近似买卖方向.
    持续净买入 → 订单流不平衡为正 → 看涨压力 → 正向.
    方向: 正向.
    """
    name = "intraday_order_flow_imbalance_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "订单流不平衡 ((买量-卖量)/总量, 净方向)"
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
        imbalances: dict = {}
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
                buy = v_c[r_c > 0].sum()
                sell = v_c[r_c < 0].sum()
                total = buy + sell
                vals[col] = float((buy - sell) / total) if total > 1e-12 else 0.0
            if vals:
                imbalances[dt] = pd.Series(vals)
        if not imbalances:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(imbalances).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 61. intraday_informed_trading — 知情交易持续性 (不平衡自相关)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_informed_trading_20d", category="intraday_advanced")
class IntradayInformedTrading20d(Factor):
    """知情交易持续性因子.

    分钟买卖不平衡序列 (sign(ret)·vol) 的一阶自相关.
    持续性高 → 知情交易者分批持续同向进场 (GM 1985 信息不对称) → 负向.
    方向: 负向.
    """
    name = "intraday_informed_trading_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "知情交易持续 (买卖不平衡自相关, 高=负向)"
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
        informed: dict = {}
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
                imbalance = np.sign(r.loc[common].values) * v.loc[common].values
                imb_t, imb_tm1 = imbalance[1:], imbalance[:-1]
                if imb_t.std(ddof=0) < 1e-12 or imb_tm1.std(ddof=0) < 1e-12:
                    vals[col] = 0.0
                else:
                    vals[col] = float(np.corrcoef(imb_t, imb_tm1)[0, 1])
            if vals:
                informed[dt] = pd.Series(vals)
        if not informed:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(informed).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 62. intraday_amihud_trend — 非流动性趋势 (Amihud 斜率)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_amihud_trend_20d", category="intraday_advanced")
class IntradayAmihudTrend20d(Factor):
    """非流动性趋势因子 (Amihud 2002 日内版).

    分钟 Amihud = |ret|/amount, 计算其日内均值随时间的变化方向
    (前半段均值 - 后半段均值).
    非流动性改善 (后半段更流动) → 流动性注入 → 正向.
    方向: 正向.
    """
    name = "intraday_amihud_trend_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "非流动性趋势 (日内Amihud前半-后半, 改善=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "amount"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, amount = panel["close"], panel["amount"]
        ret_1m = close.pct_change().abs()
        day = ret_1m.index.normalize()
        trends: dict = {}
        for dt in sorted(set(day)):
            grp_ret = ret_1m.loc[day == dt]
            grp_amt = amount.loc[day == dt]
            if len(grp_ret) < 30:
                continue
            mid = len(grp_ret) // 2
            vals = {}
            for col in grp_ret.columns:
                r = grp_ret[col].dropna()
                a = grp_amt[col].dropna()
                common = r.index.intersection(a.index)
                if len(common) < 30:
                    continue
                r_c = r.loc[common]
                a_c = a.loc[common]
                amihud = r_c / (a_c + 1e-12)
                amihud = amihud[amihud < 10.0]  # 剔除极端值
                if len(amihud) < 20:
                    continue
                first_half = amihud.iloc[:len(amihud) // 2].mean()
                second_half = amihud.iloc[len(amihud) // 2:].mean()
                vals[col] = float(first_half - second_half)
            if vals:
                trends[dt] = pd.Series(vals)
        if not trends:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(trends).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 63. intraday_impact_asymmetry — 涨跌冲击不对称
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_impact_asymmetry_20d", category="intraday_advanced")
class IntradayImpactAsymmetry20d(Factor):
    """涨跌冲击不对称因子.

    上涨分钟的 |ret|/vol 均值 vs 下跌分钟 (买入冲击 vs 卖出冲击).
    正差 → 上涨需要更多量推动 (卖出压力大) → 上行受阻 → 负向.
    方向: 负向.
    """
    name = "intraday_impact_asymmetry_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "涨跌冲击不对称 (涨/跌的|ret|/vol差)"
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
        asym: dict = {}
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
                impact = (r_c.abs() / (v_c + 1e-12))
                up_impact = impact[r_c > 0]
                dn_impact = impact[r_c < 0]
                if len(up_impact) < 3 or len(dn_impact) < 3:
                    vals[col] = 0.0
                else:
                    vals[col] = float(up_impact.mean() - dn_impact.mean())
            if vals:
                asym[dt] = pd.Series(vals)
        if not asym:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(asym).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 64. intraday_overnight_return — 隔夜收益
# ⚠ 跨日因子: 依赖昨日收盘 (prev_close 跨日追踪), 需≥2个交易日历史; 首日为NaN
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_overnight_return_20d", category="intraday_advanced")
class IntradayOvernightReturn20d(Factor):
    """隔夜收益因子.

    (今日开盘 - 昨日收盘) / 昨日收盘.
    隔夜收益捕捉盘后信息 (隔夜新闻/外盘) 的定价.
    隔夜动量持续 → 正向.
    方向: 正向.
    """
    name = "intraday_overnight_return_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "隔夜收益 ((今开-昨收)/昨收)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"open", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        open_px, close_px = panel["open"], panel["close"]
        day = close_px.index.normalize()
        overnight: dict = {}
        prev_close: dict = {}
        for dt in sorted(set(day)):
            grp_open = open_px.loc[day == dt]
            grp_close = close_px.loc[day == dt]
            if len(grp_close) < 5:
                continue
            vals = {}
            for col in grp_close.columns:
                o = grp_open[col].dropna()
                c = grp_close[col].dropna()
                if len(o) < 1 or len(c) < 5:
                    continue
                o_first = o.iloc[0]
                prev_c = prev_close.get(col)
                if prev_c is None or prev_c < 1e-12:
                    prev_close[col] = c.iloc[-1]
                    continue
                vals[col] = float(o_first / prev_c - 1.0)
                prev_close[col] = c.iloc[-1]
            if vals:
                overnight[dt] = pd.Series(vals)
        if not overnight:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(overnight).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 65. intraday_intraday_return — 日内收益 (扣除隔夜)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_intraday_return_20d", category="intraday_advanced")
class IntradayIntradayReturn20d(Factor):
    """日内收益因子 (排除隔夜跳空).

    (收盘 - 开盘) / 开盘 — 只含开盘后到收盘的定价.
    日内动量持续 → 正向.
    方向: 正向.
    """
    name = "intraday_intraday_return_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "日内收益 ((收盘-开盘)/开盘)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"open", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        open_px, close_px = panel["open"], panel["close"]
        day = close_px.index.normalize()
        intraday: dict = {}
        for dt in sorted(set(day)):
            grp_open = open_px.loc[day == dt]
            grp_close = close_px.loc[day == dt]
            if len(grp_close) < 5:
                continue
            vals = {}
            for col in grp_close.columns:
                o = grp_open[col].dropna()
                c = grp_close[col].dropna()
                if len(o) < 1 or len(c) < 5:
                    continue
                o_first = o.iloc[0]
                if o_first < 1e-12:
                    continue
                vals[col] = float(c.iloc[-1] / o_first - 1.0)
            if vals:
                intraday[dt] = pd.Series(vals)
        if not intraday:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(intraday).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 66. intraday_overnight_share — 隔夜收益占比
# ⚠ 跨日因子: 依赖昨日收盘 (prev_close 跨日追踪), 需≥2个交易日历史; 首日为NaN
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_overnight_share_20d", category="intraday_advanced")
class IntradayOvernightShare20d(Factor):
    """隔夜收益占比因子.

    |隔夜收益| / (|隔夜收益| + |日内收益|).
    隔夜主导 → 定价依赖隔夜信息 → 盘中缺乏信息增量 → 负向.
    方向: 负向.
    """
    name = "intraday_overnight_share_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "隔夜收益占比 (|隔夜|/(|隔夜|+|日内|))"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"open", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        open_px, close_px = panel["open"], panel["close"]
        day = close_px.index.normalize()
        shares: dict = {}
        prev_close: dict = {}
        for dt in sorted(set(day)):
            grp_open = open_px.loc[day == dt]
            grp_close = close_px.loc[day == dt]
            if len(grp_close) < 5:
                continue
            vals = {}
            for col in grp_close.columns:
                o = grp_open[col].dropna()
                c = grp_close[col].dropna()
                if len(o) < 1 or len(c) < 5:
                    continue
                o_first = o.iloc[0]
                prev_c = prev_close.get(col)
                if prev_c is None or prev_c < 1e-12:
                    prev_close[col] = c.iloc[-1]
                    continue
                ret_overnight = abs(o_first / prev_c - 1.0)
                ret_intraday = abs(c.iloc[-1] / o_first - 1.0)
                denom = ret_overnight + ret_intraday
                vals[col] = float(ret_overnight / denom) if denom > 1e-12 else 0.5
                prev_close[col] = c.iloc[-1]
            if vals:
                shares[dt] = pd.Series(vals)
        if not shares:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(shares).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 67. intraday_first_hour_volume — 首小时成交量占比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_first_hour_volume_20d", category="intraday_advanced")
class IntradayFirstHourVolume20d(Factor):
    """首小时成交量占比因子.

    开盘60分钟成交量 / 全天成交量.
    首小时集中放量 → 隔夜信息被快速定价 → 机构盘前研究充分 → 正向.
    方向: 正向.
    """
    name = "intraday_first_hour_volume_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "首小时量占比 (开盘60分/全天量)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        volume = panel["volume"]
        day = volume.index.normalize()
        first_hour: dict = {}
        for dt in sorted(set(day)):
            grp = volume.loc[day == dt]
            n = len(grp)
            if n < 60:
                continue
            n_first = max(20, min(60, n // 4))
            vals = {}
            for col in grp.columns:
                v = grp[col].dropna()
                if len(v) < 60:
                    continue
                total = v.sum()
                if total < 1e-12:
                    continue
                vals[col] = float(v.iloc[:n_first].sum() / total)
            if vals:
                first_hour[dt] = pd.Series(vals)
        if not first_hour:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(first_hour).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 68. intraday_noon_lull — 午间量能萎缩度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_noon_lull_20d", category="intraday_advanced")
class IntradayNoonLull20d(Factor):
    """午间量能萎缩度因子.

    午间(中间1/3)成交量占比 / 全天均匀值(1/3).
    午间异常活跃 (占比>1/3) → 反常的量能节奏 → 事件驱动 → 负向.
    午间缩量 → 正常U型节奏 → 正向.
    方向: 正向 (取 -午间占比).
    """
    name = "intraday_noon_lull_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "午间量能萎缩 (取负午间量占比, 正常午间缩量=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        volume = panel["volume"]
        day = volume.index.normalize()
        lulls: dict = {}
        for dt in sorted(set(day)):
            grp = volume.loc[day == dt]
            n = len(grp)
            if n < 30:
                continue
            third = max(8, n // 3)
            vals = {}
            for col in grp.columns:
                v = grp[col].dropna()
                if len(v) < 30:
                    continue
                total = v.sum()
                if total < 1e-12:
                    continue
                midday_share = float(v.iloc[third:2 * third].sum() / total)
                vals[col] = -midday_share
            if vals:
                lulls[dt] = pd.Series(vals)
        if not lulls:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(lulls).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 69. intraday_mfdfa_width — 多重分形宽度 Δh
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_mfdfa_width_20d", category="intraday_advanced")
class IntradayMfdfaWidth20d(Factor):
    """多重分形宽度因子 (MF-DFA, Kantelhardt et al. 2002).

    广义Hurst指数谱 h(q) 的宽度 Δh = h(q_min) - h(q_max).
    宽谱 → 大小波动标度行为差异大 → 多尺度复杂度高 → 噪声/混战 → 负向.
    方向: 负向.
    """
    name = "intraday_mfdfa_width_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "多重分形宽度 (MF-DFA的Δh, 宽=复杂=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    @staticmethod
    def _mfdfa_width(series, qs=(-2.0, 2.0)):
        """简化MF-DFA: 对两个q阶计算广义Hurst之差."""
        n = len(series)
        if n < 40:
            return 0.0
        x = series - series.mean()
        y = np.cumsum(x)
        scales = [max(6, n // 8), max(8, n // 4), max(10, n // 2)]
        scales = [s for s in scales if s < n - 2]
        if len(scales) < 2:
            return 0.0

        def _h(q):
            log_s, log_f = [], []
            for s in scales:
                n_s = n // s
                if n_s < 2:
                    continue
                f2s = []
                for v in range(n_s):
                    seg = y[v * s:(v + 1) * s]
                    t = np.arange(len(seg))
                    coeff = np.polyfit(t, seg, 1)
                    trend = np.polyval(coeff, t)
                    f2s.append(np.mean((seg - trend) ** 2))
                f2s = np.array(f2s)
                if q == 0:
                    fq = np.exp(0.5 * np.mean(np.log(f2s)))
                else:
                    fq = (np.mean(f2s ** (q / 2.0))) ** (1.0 / q)
                if fq > 1e-12:
                    log_s.append(np.log(s))
                    log_f.append(np.log(fq))
            if len(log_s) < 2:
                return 0.5
            return float(np.polyfit(log_s, log_f, 1)[0])

        h_lo = _h(qs[0])
        h_hi = _h(qs[1])
        return max(0.0, h_lo - h_hi)

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        widths: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 60:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna().values
                if len(c) < 60:
                    continue
                r = np.diff(np.log(c[c > 0]))
                if len(r) < 40:
                    continue
                vals[col] = self._mfdfa_width(r)
            if vals:
                widths[dt] = pd.Series(vals)
        if not widths:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(widths).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 70. intraday_mfdfa_h2 — DFA 广义 Hurst h(2)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_mfdfa_h2_20d", category="intraday_advanced")
class IntradayMfdfaH220d(Factor):
    """DFA 广义 Hurst h(2) 因子.

    q=2 阶的 DFA 标度指数, 等价于去趋势的 Hurst (比 #31 R/S 对非平稳更稳健).
    h(2)>0.5 → 趋势长记忆 → 动量;  h(2)<0.5 → 反持续 → 反转.
    方向: 正向.
    """
    name = "intraday_mfdfa_h2_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "DFA Hurst h(2) (去趋势标度指数)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    @staticmethod
    def _dfa_h2(series):
        n = len(series)
        if n < 40:
            return 0.5
        x = series - series.mean()
        y = np.cumsum(x)
        scales = [max(6, n // 8), max(8, n // 4), max(10, n // 2)]
        scales = [s for s in scales if s < n - 2]
        log_s, log_f = [], []
        for s in scales:
            n_s = n // s
            if n_s < 2:
                continue
            f2s = []
            for v in range(n_s):
                seg = y[v * s:(v + 1) * s]
                t = np.arange(len(seg))
                trend = np.polyval(np.polyfit(t, seg, 1), t)
                f2s.append(np.mean((seg - trend) ** 2))
            fq = np.sqrt(np.mean(f2s))
            if fq > 1e-12:
                log_s.append(np.log(s))
                log_f.append(np.log(fq))
        if len(log_s) < 2:
            return 0.5
        return float(np.polyfit(log_s, log_f, 1)[0])

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        h2s: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 60:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna().values
                if len(c) < 60:
                    continue
                r = np.diff(np.log(c[c > 0]))
                if len(r) < 40:
                    continue
                vals[col] = self._dfa_h2(r)
            if vals:
                h2s[dt] = pd.Series(vals)
        if not h2s:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(h2s).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 71. intraday_spectral_slope — 功率谱斜率 (1/f 特征)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_spectral_slope_20d", category="intraday_advanced")
class IntradaySpectralSlope20d(Factor):
    """功率谱斜率因子.

    日内收益序列 rFFT 功率谱的 log-log 斜率.
    斜率陡负 → 低频主导 → 平滑趋势; 斜率平 → 白噪声.
    借鉴 #15 drip_stone 的 FFT 思路扩展到全谱.
    方向: 正向 (平滑趋势=可预测).
    """
    name = "intraday_spectral_slope_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "功率谱斜率 (rFFT log-log斜率, 陡=平滑=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        slopes: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 60:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna().values
                if len(c) < 60:
                    continue
                r = np.diff(np.log(c[c > 0]))
                r = r - r.mean()
                r = r * np.hanning(len(r))
                fft = np.fft.rfft(r)
                power = np.abs(fft) ** 2
                freqs = np.fft.rfftfreq(len(r), d=1.0)
                mask = (freqs > 0.01) & (freqs < 0.45)
                if mask.sum() < 5:
                    continue
                log_f = np.log(freqs[mask])
                log_p = np.log(power[mask])
                slopes[col] = float(np.polyfit(log_f, log_p, 1)[0])
            if vals:
                slopes[dt] = pd.Series(vals)
        if not slopes:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(slopes).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 72. intraday_permutation_entropy — 排列熵
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_permutation_entropy_20d", category="intraday_advanced")
class IntradayPermutationEntropy20d(Factor):
    """排列熵因子 (Permutation Entropy, Bandt & Pompe 2002).

    将收益序列按3连序模式编码, 计算模式分布的香农熵 (归一化).
    高熵 → 序列高度不可预测 → 随机游走 → 负向.
    低熵 → 序列有模式 → 可预测 → 正向 (取负).
    方向: 正向 (取负熵).
    """
    name = "intraday_permutation_entropy_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "排列熵 (3连序模式熵, 取负=低熵=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    @staticmethod
    def _perm_entropy(series, order=3):
        n = len(series)
        if n < 20:
            return 0.5
        patterns = {}
        for i in range(n - order + 1):
            window = series[i:i + order]
            perm = tuple(np.argsort(window))
            patterns[perm] = patterns.get(perm, 0) + 1
        total = float(sum(patterns.values()))
        if total <= 0:
            return 0.5
        entropy = -sum((c / total) * np.log(c / total) for c in patterns.values())
        import math as _math
        max_entropy = np.log(_math.factorial(order))
        return float(entropy / max_entropy) if max_entropy > 0 else 0.5

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        entropies: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 30:
                    continue
                vals[col] = -self._perm_entropy(r)
            if vals:
                entropies[dt] = pd.Series(vals)
        if not entropies:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(entropies).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 73. intraday_fractal_dimension — 盒维数 (分形维度)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_fractal_dimension_20d", category="intraday_advanced")
class IntradayFractalDimension20d(Factor):
    """盒维数因子 (分形维度).

    用 Higuchi 法估计日内价格序列的分形维度 D (1≤D≤2).
    D 接近 1 → 平滑低维; D 接近 2 → 高维噪声.
    高维 → 随机/噪声主导 → 负向.
    方向: 负向.
    """
    name = "intraday_fractal_dimension_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "分形维度 (Higuchi法, 高维=噪声=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    @staticmethod
    def _higuchi_dim(series):
        n = len(series)
        if n < 30:
            return 1.5
        k_max = max(4, min(12, n // 5))
        if k_max < 4:
            return 1.5
        log_k, log_l = [], []
        for k in range(1, k_max + 1):
            lengths = []
            for m in range(k):
                idx = np.arange(m, n, k)
                if len(idx) < 2:
                    continue
                seg = series[idx]
                length = np.sum(np.abs(np.diff(seg)))
                length *= (n - 1) / (k * max(1, (n - m) // k))
                lengths.append(length)
            if lengths:
                l_k = np.mean(lengths)
                if l_k > 0:
                    log_k.append(np.log(k))
                    log_l.append(np.log(l_k))
        if len(log_k) < 3:
            return 1.5
        slope = float(np.polyfit(log_k, log_l, 1)[0])
        return max(1.0, min(2.0, 2.0 - slope))

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        dims: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna().values
                if len(c) < 30:
                    continue
                vals[col] = self._higuchi_dim(c)
            if vals:
                dims[dt] = pd.Series(vals)
        if not dims:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(dims).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 74. intraday_body_ratio — K线实体占比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_body_ratio_20d", category="intraday_advanced")
class IntradayBodyRatio20d(Factor):
    """K线实体占比因子.

    分钟bar的 |close-open| / (high-low) 均值.
    实体大 → 干脆的趋势日 → 方向明确 → 正向.
    实体小 → 十字星纠结日 → 分歧 → 负向.
    方向: 正向.
    """
    name = "intraday_body_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "K线实体占比 (|close-open|/(high-low)均值)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"open", "high", "low", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        open_px, high, low, close = panel["open"], panel["high"], panel["low"], panel["close"]
        day = close.index.normalize()
        bodies: dict = {}
        for dt in sorted(set(day)):
            grp_o = open_px.loc[day == dt]
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 20:
                continue
            vals = {}
            for col in grp_c.columns:
                o = grp_o[col].dropna()
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                c = grp_c[col].dropna()
                common = o.index.intersection(h.index).intersection(l.index).intersection(c.index)
                if len(common) < 20:
                    continue
                o_c = o.loc[common]
                h_c = h.loc[common]
                l_c = l.loc[common]
                c_c = c.loc[common]
                rng = (h_c - l_c).replace(0, np.nan)
                ratio = ((c_c - o_c).abs() / rng).dropna()
                if len(ratio) < 10:
                    continue
                vals[col] = float(ratio.mean())
            if vals:
                bodies[dt] = pd.Series(vals)
        if not bodies:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(bodies).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 75. intraday_upper_wick_ratio — 上影线占比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_upper_wick_ratio_20d", category="intraday_advanced")
class IntradayUpperWickRatio20d(Factor):
    """上影线占比因子.

    上影线 = (high - max(open, close)) / (high - low) 均值.
    长上影 → 上方抛压 (冲高被拒绝) → 负向.
    方向: 负向.
    """
    name = "intraday_upper_wick_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "上影线占比 ((high-max(o,c))/(h-l)均值)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"open", "high", "low", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        open_px, high, low, close = panel["open"], panel["high"], panel["low"], panel["close"]
        day = close.index.normalize()
        wicks: dict = {}
        for dt in sorted(set(day)):
            grp_o = open_px.loc[day == dt]
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 20:
                continue
            vals = {}
            for col in grp_c.columns:
                o = grp_o[col].dropna()
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                c = grp_c[col].dropna()
                common = o.index.intersection(h.index).intersection(l.index).intersection(c.index)
                if len(common) < 20:
                    continue
                o_c, h_c, l_c, c_c = (x.loc[common] for x in (o, h, l, c))
                rng = (h_c - l_c).replace(0, np.nan)
                upper = (h_c - np.maximum(o_c, c_c)) / rng
                ratio = upper.dropna()
                if len(ratio) < 10:
                    continue
                vals[col] = float(ratio.mean())
            if vals:
                wicks[dt] = pd.Series(vals)
        if not wicks:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(wicks).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 76. intraday_lower_wick_ratio — 下影线占比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_lower_wick_ratio_20d", category="intraday_advanced")
class IntradayLowerWickRatio20d(Factor):
    """下影线占比因子.

    下影线 = (min(open, close) - low) / (high - low) 均值.
    长下影 → 下方承接 (探底被买回) → 正向.
    方向: 正向.
    """
    name = "intraday_lower_wick_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "下影线占比 ((min(o,c)-low)/(h-l)均值)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"open", "high", "low", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        open_px, high, low, close = panel["open"], panel["high"], panel["low"], panel["close"]
        day = close.index.normalize()
        wicks: dict = {}
        for dt in sorted(set(day)):
            grp_o = open_px.loc[day == dt]
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 20:
                continue
            vals = {}
            for col in grp_c.columns:
                o = grp_o[col].dropna()
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                c = grp_c[col].dropna()
                common = o.index.intersection(h.index).intersection(l.index).intersection(c.index)
                if len(common) < 20:
                    continue
                o_c, h_c, l_c, c_c = (x.loc[common] for x in (o, h, l, c))
                rng = (h_c - l_c).replace(0, np.nan)
                lower = (np.minimum(o_c, c_c) - l_c) / rng
                ratio = lower.dropna()
                if len(ratio) < 10:
                    continue
                vals[col] = float(ratio.mean())
            if vals:
                wicks[dt] = pd.Series(vals)
        if not wicks:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(wicks).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 77. intraday_wick_symmetry — 影线对称性
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_wick_symmetry_20d", category="intraday_advanced")
class IntradayWickSymmetry20d(Factor):
    """影线对称性因子.

    (上影线 - 下影线) / (high - low) 均值.
    正值 → 上影主导 → 抛压 → 负向.
    负值 → 下影主导 → 承接 → 正向.
    方向: 负向.
    """
    name = "intraday_wick_symmetry_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "影线对称性 ((上影-下影)/(h-l)均值)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"open", "high", "low", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        open_px, high, low, close = panel["open"], panel["high"], panel["low"], panel["close"]
        day = close.index.normalize()
        sym: dict = {}
        for dt in sorted(set(day)):
            grp_o = open_px.loc[day == dt]
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 20:
                continue
            vals = {}
            for col in grp_c.columns:
                o = grp_o[col].dropna()
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                c = grp_c[col].dropna()
                common = o.index.intersection(h.index).intersection(l.index).intersection(c.index)
                if len(common) < 20:
                    continue
                o_c, h_c, l_c, c_c = (x.loc[common] for x in (o, h, l, c))
                rng = (h_c - l_c).replace(0, np.nan)
                upper = (h_c - np.maximum(o_c, c_c)) / rng
                lower = (np.minimum(o_c, c_c) - l_c) / rng
                ratio = (upper - lower).dropna()
                if len(ratio) < 10:
                    continue
                vals[col] = float(ratio.mean())
            if vals:
                sym[dt] = pd.Series(vals)
        if not sym:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(sym).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 78. intraday_hammer_freq — 锤子线频率
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_hammer_freq_20d", category="intraday_advanced")
class IntradayHammerFreq20d(Factor):
    """锤子线频率因子.

    锤子线: 下影线 ≥ 2×实体 且 上影线很短 (下方承接强烈).
    频率高 → 反复探底被买回 → 承接资金活跃 → 正向.
    方向: 正向.
    """
    name = "intraday_hammer_freq_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "锤子线频率 (下影≥2×实体且上影小)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"open", "high", "low", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        open_px, high, low, close = panel["open"], panel["high"], panel["low"], panel["close"]
        day = close.index.normalize()
        hammers: dict = {}
        for dt in sorted(set(day)):
            grp_o = open_px.loc[day == dt]
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 20:
                continue
            vals = {}
            for col in grp_c.columns:
                o = grp_o[col].dropna()
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                c = grp_c[col].dropna()
                common = o.index.intersection(h.index).intersection(l.index).intersection(c.index)
                if len(common) < 20:
                    continue
                o_c, h_c, l_c, c_c = (x.loc[common] for x in (o, h, l, c))
                rng = (h_c - l_c).values
                body = np.abs(c_c.values - o_c.values)
                lower = np.minimum(o_c.values, c_c.values) - l_c.values
                upper = h_c.values - np.maximum(o_c.values, c_c.values)
                valid = rng > 1e-12
                if valid.sum() < 10:
                    continue
                body_n = np.where(body / rng > 0.1, 1, 0)  # 实体需有意义
                hammer = (lower >= 2.0 * body) & (upper <= 0.3 * body + 1e-12) & (body_n == 1)
                vals[col] = float(hammer.sum() / valid.sum())
            if vals:
                hammers[dt] = pd.Series(vals)
        if not hammers:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(hammers).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 79. intraday_shooting_star_freq — 射击之星频率
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_shooting_star_freq_20d", category="intraday_advanced")
class IntradayShootingStarFreq20d(Factor):
    """射击之星频率因子.

    射击之星: 上影线 ≥ 2×实体 且 下影线很短 (上方抛压强烈).
    频率高 → 反复冲高被砸回 → 抛压持续 → 负向.
    方向: 负向.
    """
    name = "intraday_shooting_star_freq_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "射击之星频率 (上影≥2×实体且下影小)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"open", "high", "low", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        open_px, high, low, close = panel["open"], panel["high"], panel["low"], panel["close"]
        day = close.index.normalize()
        stars: dict = {}
        for dt in sorted(set(day)):
            grp_o = open_px.loc[day == dt]
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 20:
                continue
            vals = {}
            for col in grp_c.columns:
                o = grp_o[col].dropna()
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                c = grp_c[col].dropna()
                common = o.index.intersection(h.index).intersection(l.index).intersection(c.index)
                if len(common) < 20:
                    continue
                o_c, h_c, l_c, c_c = (x.loc[common] for x in (o, h, l, c))
                rng = (h_c - l_c).values
                body = np.abs(c_c.values - o_c.values)
                lower = np.minimum(o_c.values, c_c.values) - l_c.values
                upper = h_c.values - np.maximum(o_c.values, c_c.values)
                valid = rng > 1e-12
                if valid.sum() < 10:
                    continue
                body_n = np.where(body / rng > 0.1, 1, 0)
                star = (upper >= 2.0 * body) & (lower <= 0.3 * body + 1e-12) & (body_n == 1)
                vals[col] = float(star.sum() / valid.sum())
            if vals:
                stars[dt] = pd.Series(vals)
        if not stars:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(stars).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 80. intraday_vol_ratio_5_30 — 短期/中期波动比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_vol_ratio_5_30_20d", category="intraday_advanced")
class IntradayVolRatio53020d(Factor):
    """短期/中期波动比因子.

    5分钟段波动率 / 30分钟段波动率 (用分段收益std).
    高比值 → 微观结构噪声 (短期剧烈) → 非信息波动 → 负向.
    方向: 负向.
    """
    name = "intraday_vol_ratio_5_30_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "短期/中期波动比 (5分钟/30分钟段波动)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        ratios: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            n = len(grp)
            if n < 60:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 60:
                    continue
                # 5-min segments
                vol5 = []
                for i in range(0, len(r) - 4, 5):
                    seg = r[i:i + 5]
                    vol5.append(seg.std(ddof=0))
                vol5 = np.array([v for v in vol5 if not np.isnan(v)])
                # 30-min segments
                vol30 = []
                for i in range(0, len(r) - 29, 30):
                    seg = r[i:i + 30]
                    vol30.append(seg.std(ddof=0))
                vol30 = np.array([v for v in vol30 if not np.isnan(v)])
                m5 = vol5.mean() if len(vol5) else 0.0
                m30 = vol30.mean() if len(vol30) else 0.0
                vals[col] = float(m5 / m30) if m30 > 1e-12 else 1.0
            if vals:
                ratios[dt] = pd.Series(vals)
        if not ratios:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(ratios).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 81. intraday_vol_segment_consistency — 波动分段一致性
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_vol_segment_consistency_20d", category="intraday_advanced")
class IntradayVolSegmentConsistency20d(Factor):
    """波动分段一致性因子.

    前半段 |ret| 均值与后半段 |ret| 均值的相关 (跨分钟).
    一致性高 → 波动水平稳定 → 有序市场 → 正向.
    不一致 → 波动突然切换 → 不稳定 → 负向.
    方向: 正向.
    """
    name = "intraday_vol_segment_consistency_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "波动分段一致性 (前半|ret|与后半相关)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        consistencies: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            n = len(grp)
            if n < 40:
                continue
            mid = n // 2
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 40:
                    continue
                abs_r = r.abs()
                first = abs_r.iloc[:mid]
                second = abs_r.iloc[mid:]
                # 将两半段按分钟位置对齐比较, 用滚动均值的相关
                win = max(3, min(10, len(first) // 5))
                f_ma = first.rolling(win).mean().dropna().values
                s_ma = second.rolling(win).mean().dropna().values
                common_len = min(len(f_ma), len(s_ma))
                if common_len < 5:
                    vals[col] = 0.0
                else:
                    corr_val = np.corrcoef(f_ma[:common_len], s_ma[:common_len])[0, 1]
                    vals[col] = float(corr_val) if not np.isnan(corr_val) else 0.0
            if vals:
                consistencies[dt] = pd.Series(vals)
        if not consistencies:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(consistencies).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 82. intraday_range_asymmetry — 上下区间不对称
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_range_asymmetry_20d", category="intraday_advanced")
class IntradayRangeAsymmetry20d(Factor):
    """上下区间不对称因子.

    (high - close) / (close - low) 均值 (上区间 vs 下区间).
    比值 > 1 → 收盘远离高点 → 上方压力 → 负向.
    比值 < 1 → 收盘贴近高点 → 强势 → 正向.
    方向: 正向 (取负值).
    """
    name = "intraday_range_asymmetry_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "上下区间不对称 (-(high-close)/(close-low)均值)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"high", "low", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, low, close = panel["high"], panel["low"], panel["close"]
        day = close.index.normalize()
        asym: dict = {}
        for dt in sorted(set(day)):
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 20:
                continue
            vals = {}
            for col in grp_c.columns:
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                c = grp_c[col].dropna()
                common = h.index.intersection(l.index).intersection(c.index)
                if len(common) < 20:
                    continue
                h_c, l_c, c_c = (x.loc[common] for x in (h, l, c))
                upper = (h_c - c_c)
                lower = (c_c - l_c).replace(0, np.nan)
                ratio = (upper / lower).dropna()
                if len(ratio) < 10:
                    continue
                vals[col] = float(-ratio.mean())
            if vals:
                asym[dt] = pd.Series(vals)
        if not asym:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(asym).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 83. intraday_vol_persistence — 波动持续性 (AR1 系数)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_vol_persistence_20d", category="intraday_advanced")
class IntradayVolPersistence20d(Factor):
    """波动持续性因子 (分钟方差 AR(1) 系数).

    回归 |ret_t| = a + b·|ret_{t-1}| 的 b.
    高持续性 → 波动一旦放大就持续 → 风险积聚 → 负向.
    方向: 负向.
    """
    name = "intraday_vol_persistence_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "波动持续性 (|ret|的AR(1)系数, 高=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        persist: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 30:
                    continue
                ar = np.abs(r)
                y = ar[1:]
                x = ar[:-1]
                if x.std(ddof=0) < 1e-12:
                    vals[col] = 0.0
                else:
                    b = float(np.polyfit(x, y, 1)[0])
                    vals[col] = b
            if vals:
                persist[dt] = pd.Series(vals)
        if not persist:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(persist).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 84. intraday_vol_of_vol — 波动的波动 (二级波动率)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_vol_of_vol_20d", category="intraday_advanced")
class IntradayVolOfVol20d(Factor):
    """波动的波动因子 (Vol of Vol).

    std(|ret|滚动均值) / mean(|ret|滚动均值) — 分钟波动率的变异系数.
    高 VoV → 波动水平不稳定 → 环境剧变 → 负向.
    方向: 负向.
    """
    name = "intraday_vol_of_vol_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "波动的波动 (分钟波动率变异系数)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        vov: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 30:
                    continue
                abs_r = pd.Series(np.abs(r))
                win = max(3, min(10, len(abs_r) // 5))
                vol_series = abs_r.rolling(win).mean().dropna().values
                if len(vol_series) < 10:
                    continue
                m = vol_series.mean()
                if m < 1e-12:
                    vals[col] = 0.0
                else:
                    vals[col] = float(vol_series.std(ddof=0) / m)
            if vals:
                vov[dt] = pd.Series(vals)
        if not vov:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(vov).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 85. intraday_liquidity_dryup — 流动性枯竭 (零量分钟占比)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_liquidity_dryup_20d", category="intraday_advanced")
class IntradayLiquidityDryup20d(Factor):
    """流动性枯竭因子.

    日内成交量接近0 (≤1%分位) 的分钟占比.
    枯竭占比高 → 流动性断层 → 交易中断风险 → 负向.
    方向: 负向.
    """
    name = "intraday_liquidity_dryup_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "流动性枯竭 (极低量分钟占比)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        volume = panel["volume"]
        day = volume.index.normalize()
        dryups: dict = {}
        for dt in sorted(set(day)):
            grp = volume.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                v = grp[col].dropna().values
                if len(v) < 20:
                    continue
                q01 = np.percentile(v, 1)
                dry_ratio = float((v <= q01).sum() / len(v))
                vals[col] = dry_ratio
            if vals:
                dryups[dt] = pd.Series(vals)
        if not dryups:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(dryups).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 86. intraday_volume_skew — 成交量分布偏度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_volume_skew_20d", category="intraday_advanced")
class IntradayVolumeSkew20d(Factor):
    """成交量分布偏度因子.

    skew(分钟成交量). 正偏 → 少数大单尾 → 大资金集中 → 正向.
    负偏 → 多数中等单 + 少数小单 → 散户 → 负向.
    方向: 正向.
    """
    name = "intraday_volume_skew_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "量分布偏度 (正偏=大单尾=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        volume = panel["volume"]
        day = volume.index.normalize()
        skews: dict = {}
        for dt in sorted(set(day)):
            grp = volume.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                v = grp[col].dropna().values
                if len(v) < 30:
                    continue
                v_clean = v[v > 0]
                if len(v_clean) < 20:
                    continue
                vals[col] = float(pd.Series(v_clean).skew())
            if vals:
                skews[dt] = pd.Series(vals)
        if not skews:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(skews).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 87. intraday_depth_proxy — 市场深度代理
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_depth_proxy_20d", category="intraday_advanced")
class IntradayDepthProxy20d(Factor):
    """市场深度代理因子.

    平均分钟成交额 / 分钟价格振幅 (量能承载价格冲击的能力).
    深度高 → 吸收冲击能力强 → 流动性好 → 正向.
    方向: 正向.
    """
    name = "intraday_depth_proxy_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "市场深度代理 (平均成交额/平均振幅)"
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
        if not {"high", "low"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, low = panel["high"], panel["low"]
        day = amt.index.normalize()
        depths: dict = {}
        for dt in sorted(set(day)):
            grp_amt = amt.loc[day == dt]
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            if len(grp_amt) < 20:
                continue
            vals = {}
            for col in grp_amt.columns:
                a = grp_amt[col].dropna()
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                common = a.index.intersection(h.index).intersection(l.index)
                if len(common) < 20:
                    continue
                a_c, h_c, l_c = (x.loc[common] for x in (a, h, l))
                rng = (h_c - l_c).replace(0, np.nan)
                rng_clean = rng.dropna()
                if len(rng_clean) < 10:
                    continue
                mean_amt = a_c.mean()
                mean_rng = rng_clean.mean()
                vals[col] = float(mean_amt / mean_rng) if mean_rng > 1e-12 else 0.0
            if vals:
                depths[dt] = pd.Series(vals)
        if not depths:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(depths).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 88. intraday_turnover_velocity — 换手速度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_turnover_velocity_20d", category="intraday_advanced")
class IntradayTurnoverVelocity20d(Factor):
    """换手速度因子.

    平均分钟成交量 / 成交量标准差 (量的"流动速度").
    高速度 → 量能平稳充裕 → 活跃流动性 → 正向.
    低速度 → 量忽大忽小 → 不稳定 → 负向.
    方向: 正向.
    """
    name = "intraday_turnover_velocity_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "换手速度 (均量/量std, 平稳充裕=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        volume = panel["volume"]
        day = volume.index.normalize()
        velocities: dict = {}
        for dt in sorted(set(day)):
            grp = volume.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                v = grp[col].dropna().values
                if len(v) < 20:
                    continue
                m, s = v.mean(), v.std(ddof=0)
                vals[col] = float(m / s) if s > 1e-12 else 0.0
            if vals:
                velocities[dt] = pd.Series(vals)
        if not velocities:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(velocities).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 89. intraday_liquidity_spike_freq — 流动性尖峰频率
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_liquidity_spike_freq_20d", category="intraday_advanced")
class IntradayLiquiditySpikeFreq20d(Factor):
    """流动性尖峰频率因子.

    成交量 > μ+3σ 的分钟占比 (极端放量频率).
    高频尖峰 → 频繁异常事件/对倒 → 不稳定 → 负向.
    方向: 负向.
    """
    name = "intraday_liquidity_spike_freq_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "流动性尖峰频率 (量>μ+3σ占比)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        volume = panel["volume"]
        day = volume.index.normalize()
        spikes: dict = {}
        for dt in sorted(set(day)):
            grp = volume.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                v = grp[col].dropna().values
                if len(v) < 30:
                    continue
                mu, sigma = v.mean(), v.std(ddof=0)
                if sigma < 1e-12:
                    vals[col] = 0.0
                else:
                    vals[col] = float((v > mu + 3.0 * sigma).sum() / len(v))
            if vals:
                spikes[dt] = pd.Series(vals)
        if not spikes:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(spikes).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 90. intraday_disposition_proxy — 处置效应代理 (回本点放量)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_disposition_proxy_20d", category="intraday_advanced")
class IntradayDispositionProxy20d(Factor):
    """处置效应代理因子 (Shefrin & Statman 1985).

    价格贴近当日 VWAP (参考点) 时的分钟成交量 / 基准量.
    贴近参考点放量 → 投资者解套离场 → 处置效应 → 抛压 → 负向.
    方向: 负向.
    """
    name = "intraday_disposition_proxy_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "处置效应代理 (贴近VWAP放量, 解套抛压=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, volume = panel["close"], panel["volume"]
        day = close.index.normalize()
        proxies: dict = {}
        for dt in sorted(set(day)):
            grp_close = close.loc[day == dt]
            grp_vol = volume.loc[day == dt]
            if len(grp_close) < 30:
                continue
            vals = {}
            for col in grp_close.columns:
                c = grp_close[col].dropna()
                v = grp_vol[col].dropna()
                common = c.index.intersection(v.index)
                if len(common) < 30:
                    continue
                c_c = c.loc[common]
                v_c = v.loc[common]
                # VWAP as reference point
                vwap = float((c_c * v_c).sum() / v_c.sum()) if v_c.sum() > 1e-12 else c_c.mean()
                if vwap < 1e-12:
                    continue
                dist = (c_c - vwap).abs() / vwap
                # 贴近参考点: 距离在10%分位内
                threshold = np.percentile(dist, 10)
                near_mask = dist <= max(threshold, 1e-6)
                base_vol = v_c.mean()
                if base_vol < 1e-12:
                    continue
                near_vol = v_c[near_mask].mean()
                vals[col] = float(near_vol / base_vol)
            if vals:
                proxies[dt] = pd.Series(vals)
        if not proxies:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(proxies).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 91. intraday_anchoring — 锚定效应 (昨收锚)
# ⚠ 跨日因子: 依赖昨日收盘 (prev_close 跨日追踪), 需≥2个交易日历史; 首日为NaN
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_anchoring_20d", category="intraday_advanced")
class IntradayAnchoring20d(Factor):
    """锚定效应因子.

    价格长期贴近昨收锚点的比例: mean(1 - |close - prev_close| / (high - low)).
    高锚定 → 价格被昨收牵制 → 缺乏独立定价 → 负向.
    方向: 负向.
    """
    name = "intraday_anchoring_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "锚定效应 (价格贴近昨收比例, 高锚定=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"high", "low", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, low, close = panel["high"], panel["low"], panel["close"]
        day = close.index.normalize()
        anchors: dict = {}
        prev_close: dict = {}
        for dt in sorted(set(day)):
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 20:
                continue
            vals = {}
            for col in grp_c.columns:
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                c = grp_c[col].dropna()
                common = h.index.intersection(l.index).intersection(c.index)
                if len(common) < 20:
                    continue
                h_c, l_c, c_c = (x.loc[common] for x in (h, l, c))
                prev_c = prev_close.get(col)
                if prev_c is None or prev_c < 1e-12:
                    prev_close[col] = c_c.iloc[-1]
                    continue
                rng = (h_c - l_c).replace(0, np.nan)
                anch = 1.0 - (c_c - prev_c).abs() / rng
                anch_clean = anch.dropna()
                if len(anch_clean) < 10:
                    continue
                vals[col] = float(anch_clean.mean())
                prev_close[col] = c_c.iloc[-1]
            if vals:
                anchors[dt] = pd.Series(vals)
        if not anchors:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(anchors).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 92. intraday_attention — 注意力强度 (量突变频率)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_attention_20d", category="intraday_advanced")
class IntradayAttention20d(Factor):
    """注意力强度因子.

    成交量 > μ+2σ 的分钟频率 (放量吸引注意的事件频率).
    高频注意事件 → 情绪化交易活跃 → 追涨杀跌 → 负向.
    方向: 负向.
    """
    name = "intraday_attention_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "注意力强度 (量>μ+2σ频率, 情绪化=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        volume = panel["volume"]
        day = volume.index.normalize()
        attention: dict = {}
        for dt in sorted(set(day)):
            grp = volume.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                v = grp[col].dropna().values
                if len(v) < 30:
                    continue
                mu, sigma = v.mean(), v.std(ddof=0)
                if sigma < 1e-12:
                    vals[col] = 0.0
                else:
                    vals[col] = float((v > mu + 2.0 * sigma).sum() / len(v))
            if vals:
                attention[dt] = pd.Series(vals)
        if not attention:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(attention).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 93. intraday_gambling — 博彩偏好 (高波动低成交)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_gambling_20d", category="intraday_advanced")
class IntradayGambling20d(Factor):
    """博彩偏好因子.

    波动率 / 成交量 比值 (高波动但量小 = 彩票型投机).
    博彩型特征 → 投机溢价 → 预期收益低 → 负向.
    方向: 负向.
    """
    name = "intraday_gambling_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "博彩偏好 (波动/成交量, 高=彩票型=负向)"
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
        gamblings: dict = {}
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
                vol = r_c.std(ddof=0)
                vol_sum = v_c.sum()
                if vol_sum < 1e-12:
                    continue
                vals[col] = float(vol / vol_sum)
            if vals:
                gamblings[dt] = pd.Series(vals)
        if not gamblings:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(gamblings).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 94. intraday_lottery_max — 彩票效应 (最大分钟收益)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_lottery_max_20d", category="intraday_advanced")
class IntradayLotteryMax20d(Factor):
    """彩票效应因子 (MAX效应, Bali et al. 2011).

    日内最大单分钟收益 |max(ret)| 或最大正收益.
    大 MAX → 彩票型资产 → 被高估 → 预期收益低 → 负向.
    方向: 负向.
    """
    name = "intraday_lottery_max_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "彩票效应 (日内最大分钟收益, 大MAX=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        maxs: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 20:
                    continue
                vals[col] = float(-r.max())  # 大MAX=低值=负向
            if vals:
                maxs[dt] = pd.Series(vals)
        if not maxs:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(maxs).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 95. intraday_price_delay — 价格延迟 (昨日收益滞后反应)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_price_delay_20d", category="intraday_advanced")
class IntradayPriceDelay20d(Factor):
    """价格延迟因子 (开盘价格发现份额).

    价格对隔夜信息的反应速度: 开盘前30分钟收益的绝对占比.
    share = |ret_first30| / (|ret_first30| + |ret_rest|).
    share 高 → 信息在开盘快速定价 → 延迟低 → 效率高 → 正向.
    方向: 正向.
    """
    name = "intraday_price_delay_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "价格延迟 (开盘30分收益占比, 高=快速定价=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        shares: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            n = len(grp)
            if n < 40:
                continue
            n_first = max(10, min(30, n // 4))
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < 40:
                    continue
                first30 = c.iloc[:n_first]
                rest = c.iloc[n_first:]
                if len(rest) < 5:
                    continue
                ret_first = abs(first30.iloc[-1] / first30.iloc[0] - 1.0)
                ret_rest = abs(rest.iloc[-1] / rest.iloc[0] - 1.0)
                denom = ret_first + ret_rest
                vals[col] = float(ret_first / denom) if denom > 1e-12 else 0.5
            if vals:
                shares[dt] = pd.Series(vals)
        if not shares:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(shares).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 96. intraday_market_efficiency — 市场效率偏离度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_market_efficiency_20d", category="intraday_advanced")
class IntradayMarketEfficiency20d(Factor):
    """市场效率偏离度因子.

    |Hurst - 0.5| — 价格偏离随机游走的程度 (借用 #31 的R/S估计).
    高偏离 → 价格有可预测结构 → 效率低 → 有趋势可抓 → 正向.
    方向: 正向.
    """
    name = "intraday_market_efficiency_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "市场效率偏离 (|Hurst-0.5|, 偏离随机游走=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        efficiencies: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna().values
                if len(c) < 30:
                    continue
                r = np.diff(np.log(c[c > 0]))
                if len(r) < 30:
                    continue
                h = IntradayHurst20d._hurst_rs(r)
                vals[col] = float(abs(h - 0.5))
            if vals:
                efficiencies[dt] = pd.Series(vals)
        if not efficiencies:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(efficiencies).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 97. intraday_trend_continuation — 趋势延续性 (半日动量)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_trend_continuation_20d", category="intraday_advanced")
class IntradayTrendContinuation20d(Factor):
    """趋势延续性因子.

    前半段收益方向与后半段收益方向一致的比例:
    sign(ret_early) == sign(ret_late) 时记 +1, 否则 -1.
    持续同向 → 趋势延续 → 正向. 反转 → 负向.
    与 #20 early_late_divergence 互补 (那里测背离幅度, 这里测方向命中率).
    方向: 正向.
    """
    name = "intraday_trend_continuation_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "趋势延续 (半日方向一致比例, +1/-1)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        continuations: dict = {}
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
                if abs(ret_early) < 1e-12 or abs(ret_late) < 1e-12:
                    vals[col] = 0.0
                else:
                    vals[col] = 1.0 if np.sign(ret_early) == np.sign(ret_late) else -1.0
            if vals:
                continuations[dt] = pd.Series(vals)
        if not continuations:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(continuations).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 98. intraday_highest_time — 最高点时间位置
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_highest_time_20d", category="intraday_advanced")
class IntradayHighestTime20d(Factor):
    """最高点时间位置因子.

    日内最高价出现在第几分 (归一化到[0,1], 0=开盘, 1=收盘).
    高点偏尾盘 → 渐进走强 → 正向. 高点偏早盘 → 冲高回落 → 负向.
    方向: 正向.
    """
    name = "intraday_highest_time_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "最高点时间位置 (高点偏尾盘=渐进走强=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "high" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high = panel["high"]
        day = high.index.normalize()
        times: dict = {}
        for dt in sorted(set(day)):
            grp = high.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                h = grp[col].dropna()
                if len(h) < 20:
                    continue
                idx = int(np.argmax(h.values))
                vals[col] = float(idx / max(1, len(h) - 1))
            if vals:
                times[dt] = pd.Series(vals)
        if not times:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(times).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 99. intraday_lowest_time — 最低点时间位置
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_lowest_time_20d", category="intraday_advanced")
class IntradayLowestTime20d(Factor):
    """最低点时间位置因子.

    日内最低价出现时间 (归一化). 低点偏早盘 → 探底回升 → 正向.
    低点偏尾盘 → 阴跌不止 → 负向. 取负使早盘见底=高值=正向.
    方向: 正向.
    """
    name = "intraday_lowest_time_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "最低点时间位置 (取负, 低点偏早=探底回升=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "low" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        low = panel["low"]
        day = low.index.normalize()
        times: dict = {}
        for dt in sorted(set(day)):
            grp = low.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                l = grp[col].dropna()
                if len(l) < 20:
                    continue
                idx = int(np.argmin(l.values))
                vals[col] = -float(idx / max(1, len(l) - 1))
            if vals:
                times[dt] = pd.Series(vals)
        if not times:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(times).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 100. intraday_max_drawdown — 最大回撤深度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_max_drawdown_20d", category="intraday_advanced")
class IntradayMaxDrawdown20d(Factor):
    """最大回撤深度因子.

    日内累计收益序列的最大峰谷回撤 (从峰值到谷底的跌幅).
    回撤深 → 抛压重/趋势恶化 → 负向.
    方向: 负向.
    """
    name = "intraday_max_drawdown_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "最大回撤深度 (日内累计收益峰谷跌幅)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        dd: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < 20:
                    continue
                cum = c.values / c.values[0] - 1.0
                peak = np.maximum.accumulate(cum)
                drawdown = peak - cum
                vals[col] = -float(drawdown.max())
            if vals:
                dd[dt] = pd.Series(vals)
        if not dd:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(dd).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 101. intraday_zigzag_ratio — 锯齿度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_zigzag_ratio_20d", category="intraday_advanced")
class IntradayZigzagRatio20d(Factor):
    """锯齿度因子.

    用摆动转折点检测 (局部高低点, 需左右相邻都更低/更高) 统计显著转折.
    与 #18 逐分钟切换不同: 这里过滤微小噪声, 只计显著摆动点.
    高锯齿 → 频繁显著折返 → 无方向共识 → 负向.
    方向: 负向.
    """
    name = "intraday_zigzag_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "锯齿度 (摆动转折点数/分钟数)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    @staticmethod
    def _swing_turns(values):
        """检测局部极值转折点 (摆动高低点交替)."""
        n = len(values)
        if n < 5:
            return 0
        turns = 0
        last_dir = 0
        for i in range(1, n - 1):
            # 严格局部极值: 左右都更低(高点)或都更高(低点)
            is_high = values[i] > values[i - 1] and values[i] > values[i + 1]
            is_low = values[i] < values[i - 1] and values[i] < values[i + 1]
            if is_high or is_low:
                direction = 1 if is_high else -1
                if last_dir != 0 and direction != last_dir:
                    turns += 1
                last_dir = direction
        return turns

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        zigzags: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna().values
                if len(c) < 20:
                    continue
                turns = self._swing_turns(c)
                vals[col] = float(turns / len(c))
            if vals:
                zigzags[dt] = pd.Series(vals)
        if not zigzags:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(zigzags).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 102. intraday_trend_occupancy — 趋势段时间占比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_trend_occupancy_20d", category="intraday_advanced")
class IntradayTrendOccupancy20d(Factor):
    """趋势段时间占比因子.

    方向明确 (|ret|超过中位) 且与全天主导方向一致的分钟占比.
    高占比 → 大部分时间处于有效趋势中 → 方向性强 → 正向.
    方向: 正向.
    """
    name = "intraday_trend_occupancy_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "趋势段时间占比 (与主导方向一致的大波动分钟占比)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        occupancies: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 20:
                    continue
                dominant = np.sign(r.sum())
                if abs(dominant) < 1e-12:
                    vals[col] = 0.0
                    continue
                med_abs = r.abs().median()
                aligned = ((r.abs() > med_abs) & (np.sign(r) == dominant)).sum()
                vals[col] = float(aligned / len(r))
            if vals:
                occupancies[dt] = pd.Series(vals)
        if not occupancies:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(occupancies).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 103. intraday_shock_continuation — 大冲击后延续度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_shock_continuation_20d", category="intraday_advanced")
class IntradayShockContinuation20d(Factor):
    """大冲击后延续度因子.

    检测 |ret|>3σ 的冲击分钟, 其后5分钟方向与冲击方向一致的比例.
    高延续 → 冲击信息被持续定价 → 趋势确认 → 正向.
    方向: 正向.
    """
    name = "intraday_shock_continuation_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "冲击延续 (大冲击后5分钟同向比例)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        continuations: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 30:
                    continue
                sigma = r.std(ddof=0)
                if sigma < 1e-12:
                    vals[col] = 0.0
                    continue
                shock_idx = np.where(np.abs(r) > 3.0 * sigma)[0]
                scores = []
                for i in shock_idx:
                    look = min(5, len(r) - i - 1)
                    if look < 1:
                        continue
                    fwd = np.sign(np.sum(r[i + 1:i + 1 + look]))
                    scores.append(1.0 if fwd == np.sign(r[i]) else 0.0)
                vals[col] = float(np.mean(scores)) if scores else 0.0
            if vals:
                continuations[dt] = pd.Series(vals)
        if not continuations:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(continuations).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 104. intraday_volatility_cooling — 冲击后冷却速度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_volatility_cooling_20d", category="intraday_advanced")
class IntradayVolatilityCooling20d(Factor):
    """冲击后冷却速度因子.

    大冲击 (|ret|>3σ) 后波动回落到常态所需的分钟数 (取负).
    冷却快 → 冲击一次性消化 → 市场健康 → 正向.
    冷却慢 → 波动持续发酵 → 不稳定 → 负向.
    方向: 正向 (取负冷却时长).
    """
    name = "intraday_volatility_cooling_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "冲击冷却 (大冲击后波动回落时长, 取负=快速冷却=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        coolings: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 30:
                    continue
                sigma = r.std(ddof=0)
                if sigma < 1e-12:
                    vals[col] = 0.0
                    continue
                threshold = 2.0 * sigma
                shock_idx = np.where(np.abs(r) > 3.0 * sigma)[0]
                cooling_times = []
                for i in shock_idx:
                    j = i + 1
                    while j < len(r) and np.abs(r[j]) > threshold:
                        j += 1
                    cooling_times.append(j - i)
                mean_cooling = float(np.mean(cooling_times)) if cooling_times else 0.0
                vals[col] = -mean_cooling
            if vals:
                coolings[dt] = pd.Series(vals)
        if not coolings:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(coolings).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 105. intraday_drawdown_speed — 回撤速度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_drawdown_speed_20d", category="intraday_advanced")
class IntradayDrawdownSpeed20d(Factor):
    """回撤速度因子.

    最大回撤深度 / 回撤持续分钟数 (单位时间跌幅).
    回撤快 → 恐慌性抛售 → 负向. 回撤慢 → 渐进调整 → 相对健康.
    方向: 负向.
    """
    name = "intraday_drawdown_speed_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "回撤速度 (最大回撤/持续分钟, 急跌=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        speeds: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < 20:
                    continue
                cum = c.values / c.values[0] - 1.0
                peak = np.maximum.accumulate(cum)
                dd = peak - cum
                max_dd = dd.max()
                if max_dd < 1e-12:
                    vals[col] = 0.0
                    continue
                trough_idx = int(np.argmax(dd))
                peak_idx = int(np.argmax(cum[:trough_idx + 1]))
                duration = max(1, trough_idx - peak_idx)
                vals[col] = -float(max_dd / duration)
            if vals:
                speeds[dt] = pd.Series(vals)
        if not speeds:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(speeds).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 106. intraday_recovery_speed — 回撤恢复速度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_recovery_speed_20d", category="intraday_advanced")
class IntradayRecoverySpeed20d(Factor):
    """回撤恢复速度因子.

    从最大回撤谷底恢复到收盘的幅度 / 恢复时间.
    快速收复 → 承接有力 → 正向.
    方向: 正向.
    """
    name = "intraday_recovery_speed_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "回撤恢复 (谷底到收盘的回升速度)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        recoveries: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < 20:
                    continue
                cum = c.values / c.values[0] - 1.0
                peak = np.maximum.accumulate(cum)
                dd = peak - cum
                trough_idx = int(np.argmax(dd))
                if trough_idx >= len(cum) - 1:
                    vals[col] = 0.0
                    continue
                trough_val = cum[trough_idx]
                recover_amt = cum[-1] - trough_val
                duration = len(cum) - 1 - trough_idx
                vals[col] = float(recover_amt / max(1, duration))
            if vals:
                recoveries[dt] = pd.Series(vals)
        if not recoveries:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(recoveries).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 107. intraday_shock_asymmetry — 冲击方向不对称
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_shock_asymmetry_20d", category="intraday_advanced")
class IntradayShockAsymmetry20d(Factor):
    """冲击方向不对称因子.

    正冲击均值 (|ret|>2σ 且 ret>0) - 负冲击均值 (取正数).
    正冲击占优 → 向上的爆发力更强 → 正向.
    方向: 正向.
    """
    name = "intraday_shock_asymmetry_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "冲击不对称 (正冲击幅度-负冲击幅度)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        asym: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 30:
                    continue
                sigma = r.std(ddof=0)
                if sigma < 1e-12:
                    vals[col] = 0.0
                    continue
                pos = r[(r > 0) & (np.abs(r) > 2.0 * sigma)]
                neg = r[(r < 0) & (np.abs(r) > 2.0 * sigma)]
                pos_m = pos.mean() if len(pos) else 0.0
                neg_m = neg.mean() if len(neg) else 0.0
                vals[col] = float(pos_m - neg_m)
            if vals:
                asym[dt] = pd.Series(vals)
        if not asym:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(asym).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 108. intraday_morning_afternoon_ratio — 上午/下午收益比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_morning_afternoon_ratio_20d", category="intraday_advanced")
class IntradayMorningAfternoonRatio20d(Factor):
    """上午/下午收益比因子.

    上午收益 / (上午收益+下午收益), 衡量收益的时间分配.
    上午主导 → 开盘定价充分 → 早盘动能强 → 正向.
    方向: 正向.
    """
    name = "intraday_morning_afternoon_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "上午/下午收益比 (上午主导=开盘定价充分=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        ratios: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            n = len(grp)
            if n < 20:
                continue
            mid = n // 2
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < 20:
                    continue
                am_ret = c.iloc[mid] / c.iloc[0] - 1.0
                pm_ret = c.iloc[-1] / c.iloc[mid] - 1.0
                denom = abs(am_ret) + abs(pm_ret)
                vals[col] = float(am_ret / denom) if denom > 1e-12 else 0.0
            if vals:
                ratios[dt] = pd.Series(vals)
        if not ratios:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(ratios).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 109. intraday_close_drift — 尾盘漂移
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_close_drift_20d", category="intraday_advanced")
class IntradayCloseDrift20d(Factor):
    """尾盘漂移因子.

    最后30分钟收益 (收盘价相对30分钟前).
    尾盘上涨 → 收盘动能强 → 正向. 与 #33 加速度互补 (这里是一阶漂移).
    方向: 正向.
    """
    name = "intraday_close_drift_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "尾盘漂移 (最后30分钟收益)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        drifts: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            n = len(grp)
            if n < 30:
                continue
            n_tail = max(10, min(30, n // 4))
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < 30:
                    continue
                tail_start = c.iloc[-n_tail]
                vals[col] = float(c.iloc[-1] / tail_start - 1.0)
            if vals:
                drifts[dt] = pd.Series(vals)
        if not drifts:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(drifts).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 110. intraday_open_surge — 开盘脉冲强度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_open_surge_20d", category="intraday_advanced")
class IntradayOpenSurge20d(Factor):
    """开盘脉冲强度因子.

    开盘30分钟的平均|ret| / 全天平均|ret|.
    开盘异常剧烈 → 隔夜信息集中释放但可能过度反应 → 负向.
    方向: 负向.
    """
    name = "intraday_open_surge_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "开盘脉冲 (开盘30分波动/全天, 剧烈=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change().abs()
        day = ret_1m.index.normalize()
        surges: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            n = len(grp)
            if n < 30:
                continue
            n_open = max(10, min(30, n // 4))
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 30:
                    continue
                day_mean = r.mean()
                open_mean = r.iloc[:n_open].mean()
                if day_mean < 1e-12:
                    vals[col] = 1.0
                else:
                    vals[col] = float(open_mean / day_mean)
            if vals:
                surges[dt] = pd.Series(vals)
        if not surges:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(surges).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 111. intraday_last15min_volume — 尾盘15分钟量占比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_last15min_volume_20d", category="intraday_advanced")
class IntradayLast15minVolume20d(Factor):
    """尾盘15分钟量占比因子.

    最后15分钟成交量 / 全天成交量 (与 #67 首小时占比对称).
    尾盘集中放量 → 收盘抢跑/恐慌 → 负向.
    方向: 负向.
    """
    name = "intraday_last15min_volume_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "尾盘15分量占比 (集中放量=收盘抢跑=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        volume = panel["volume"]
        day = volume.index.normalize()
        shares: dict = {}
        for dt in sorted(set(day)):
            grp = volume.loc[day == dt]
            n = len(grp)
            if n < 30:
                continue
            n_last = max(5, min(15, n // 6))
            vals = {}
            for col in grp.columns:
                v = grp[col].dropna()
                if len(v) < 30:
                    continue
                total = v.sum()
                if total < 1e-12:
                    continue
                vals[col] = float(v.iloc[-n_last:].sum() / total)
            if vals:
                shares[dt] = pd.Series(vals)
        if not shares:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(shares).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 112. intraday_third_share — 尾盘1/3收益占比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_third_share_20d", category="intraday_advanced")
class IntradayThirdShare20d(Factor):
    """尾盘1/3收益占比因子.

    最后1/3段收益 / (|早1/3|+|中1/3|+|晚1/3|) 绝对和.
    尾盘贡献大部分收益 → 尾盘定价动能强 → 正向.
    方向: 正向.
    """
    name = "intraday_third_share_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "尾盘1/3收益占比 (尾盘定价动能=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        shares: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            n = len(grp)
            if n < 30:
                continue
            third = max(8, n // 3)
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < 30:
                    continue
                r1 = c.iloc[third] / c.iloc[0] - 1.0
                r2 = c.iloc[2 * third] / c.iloc[third] - 1.0
                r3 = c.iloc[-1] / c.iloc[2 * third] - 1.0
                denom = abs(r1) + abs(r2) + abs(r3)
                vals[col] = float(r3 / denom) if denom > 1e-12 else 0.0
            if vals:
                shares[dt] = pd.Series(vals)
        if not shares:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(shares).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 113. intraday_regression_curvature — 路径凸度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_regression_curvature_20d", category="intraday_advanced")
class IntradayRegressionCurvature20d(Factor):
    """路径凸度因子.

    对日内价格路径做时间二次回归 y = a·t² + b·t + c, 取二次项 a 标准化.
    凸 (a>0) → 加速上行 → 正向. 凹 (a<0) → 减速/冲高回落 → 负向.
    方向: 正向.
    """
    name = "intraday_regression_curvature_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "路径凸度 (时间二次回归系数, 凸=加速上行=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        curvatures: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < 30:
                    continue
                y = c.values / c.values[0] - 1.0  # 归一化收益路径
                t = np.arange(len(y)) / max(1, len(y) - 1)
                coeff = np.polyfit(t, y, 2)
                vals[col] = float(coeff[0])
            if vals:
                curvatures[dt] = pd.Series(vals)
        if not curvatures:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(curvatures).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 114. intraday_ols_rsquared — 趋势拟合优度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_ols_rsquared_20d", category="intraday_advanced")
class IntradayOlsRsquared20d(Factor):
    """趋势拟合优度因子.

    日内价格路径对时间线性回归的 R².
    R² 高 → 路径近似直线 → 单边趋势清晰 → 正向.
    R² 低 → 路径来回震荡 → 无清晰趋势 → 负向.
    方向: 正向.
    """
    name = "intraday_ols_rsquared_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "趋势拟合优度 (路径线性R², 清晰单边=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        r2s: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < 30:
                    continue
                y = c.values / c.values[0] - 1.0
                t = np.arange(len(y))
                slope, intercept = np.polyfit(t, y, 1)
                fitted = slope * t + intercept
                ss_res = np.sum((y - fitted) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
                vals[col] = float(max(0.0, r2))
            if vals:
                r2s[dt] = pd.Series(vals)
        if not r2s:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(r2s).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 115. intraday_residual_skew — 回归残差偏度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_residual_skew_20d", category="intraday_advanced")
class IntradayResidualSkew20d(Factor):
    """回归残差偏度因子.

    价格路径去趋势后的残差分布偏度.
    残差正偏 → 上行脉冲多于下行 → 向上的噪声有利 → 正向.
    方向: 正向.
    """
    name = "intraday_residual_skew_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "回归残差偏度 (去趋势后残差分布)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        skews: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < 30:
                    continue
                y = c.values / c.values[0] - 1.0
                t = np.arange(len(y))
                slope, intercept = np.polyfit(t, y, 1)
                resid = y - (slope * t + intercept)
                vals[col] = float(pd.Series(resid).skew())
            if vals:
                skews[dt] = pd.Series(vals)
        if not skews:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(skews).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 116. intraday_volatility_breakout — 跨日波动突破
# ⚠ 跨日因子: 依赖前10日波动率 (rolling(10, min_periods=3)), 需≥4个交易日历史; 滚动窗口前为NaN
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_volatility_breakout_20d", category="intraday_advanced")
class IntradayVolatilityBreakout20d(Factor):
    """跨日波动突破因子.

    今日日内波动率 / 近10日平均日内波动率.
    今日异常放大 → 波动突破 → 事件日/不稳定 → 负向.
    方向: 负向.
    """
    name = "intraday_volatility_breakout_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "跨日波动突破 (今日波动/近10日均值)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        daily_vol: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 30:
                    continue
                vals[col] = float(r.std(ddof=0))
            if vals:
                daily_vol[dt] = pd.Series(vals)
        if not daily_vol:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        vol_df = pd.DataFrame(daily_vol).T
        vol_df.index = pd.DatetimeIndex(vol_df.index)
        vol_df = vol_df.reindex(dates)
        vol_ma = vol_df.rolling(10, min_periods=3).mean()
        ratios = vol_df / vol_ma.replace(0, np.nan)
        return ratios.reindex(columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 117. intraday_ma_cross — 分钟均线交叉
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_ma_cross_20d", category="intraday_advanced")
class IntradayMaCross20d(Factor):
    """分钟均线交叉因子.

    收盘时 MA5(分钟) 相对 MA20(分钟) 的位置: (MA5 - MA20)/MA20.
    快线上穿慢线 → 短期动能向上 → 正向.
    方向: 正向.
    """
    name = "intraday_ma_cross_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "分钟均线交叉 ((MA5-MA20)/MA20 收盘值)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        crosses: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < 30:
                    continue
                ma5 = c.rolling(5).mean().iloc[-1]
                ma20 = c.rolling(20).mean().iloc[-1]
                if ma20 is None or abs(ma20) < 1e-12:
                    vals[col] = 0.0
                else:
                    vals[col] = float((ma5 - ma20) / abs(ma20))
            if vals:
                crosses[dt] = pd.Series(vals)
        if not crosses:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(crosses).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 118. intraday_tail_ratio — 极端收益占比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_tail_ratio_20d", category="intraday_advanced")
class IntradayTailRatio20d(Factor):
    """极端收益占比因子.

    |ret|>3σ 的分钟占比 (极端事件频率).
    高频极端 → 不稳定/操纵 → 负向. 与 #92 attention (2σ) 阈值不同.
    方向: 负向.
    """
    name = "intraday_tail_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "极端收益占比 (|ret|>3σ频率)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        tails: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 30:
                    continue
                sigma = r.std(ddof=0)
                if sigma < 1e-12:
                    vals[col] = 0.0
                else:
                    vals[col] = float((np.abs(r) > 3.0 * sigma).sum() / len(r))
            if vals:
                tails[dt] = pd.Series(vals)
        if not tails:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(tails).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 119. intraday_abs_ret_ratio — 均值/中位绝对收益比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_abs_ret_ratio_20d", category="intraday_advanced")
class IntradayAbsRetRatio20d(Factor):
    """均值/中位绝对收益比因子.

    mean(|ret|) / median(|ret|). 比值>1 表示少数大波动抬高均值.
    高比值 → 厚尾/依赖少数大波动 → 不稳定 → 负向.
    方向: 负向.
    """
    name = "intraday_abs_ret_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "均值/中位绝对收益比 (厚尾特征, 高=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        ratios: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 30:
                    continue
                ar = r.abs()
                med = ar.median()
                if med < 1e-12:
                    vals[col] = 1.0
                else:
                    vals[col] = float(ar.mean() / med)
            if vals:
                ratios[dt] = pd.Series(vals)
        if not ratios:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(ratios).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 120. intraday_extreme_conc — 单分钟波动集中度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_extreme_conc_20d", category="intraday_advanced")
class IntradayExtremeConc20d(Factor):
    """单分钟波动集中度因子.

    最大单分钟|ret| / 全部|ret|之和 (波动是否集中在单一分钟).
    高集中 → 当日波动由单点驱动 → 不可持续 → 负向.
    方向: 负向.
    """
    name = "intraday_extreme_conc_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "单分钟波动集中 (max|ret|/Σ|ret|)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        concs: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 30:
                    continue
                ar = r.abs()
                total = ar.sum()
                vals[col] = float(ar.max() / total) if total > 1e-12 else 0.0
            if vals:
                concs[dt] = pd.Series(vals)
        if not concs:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(concs).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 121. intraday_profit_loss_ratio — 盈亏幅度比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_profit_loss_ratio_20d", category="intraday_advanced")
class IntradayProfitLossRatio20d(Factor):
    """盈亏幅度比因子.

    盈利分钟平均收益 / |亏损分钟平均收益|.
    比值>1 → 涨多跌少 → 上行效率高 → 正向.
    方向: 正向.
    """
    name = "intraday_profit_loss_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "盈亏幅度比 (涨分钟均值/跌分钟均值)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        ratios: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 20:
                    continue
                up = r[r > 0]
                dn = r[r < 0]
                up_m = up.mean() if len(up) else 0.0
                dn_m = dn.mean() if len(dn) else 0.0
                vals[col] = float(up_m / abs(dn_m)) if abs(dn_m) > 1e-12 else (2.0 if up_m > 1e-12 else 1.0)
            if vals:
                ratios[dt] = pd.Series(vals)
        if not ratios:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(ratios).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 122. intraday_kurtosis_tail — 分位数尾部厚度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_kurtosis_tail_20d", category="intraday_advanced")
class IntradayKurtosisTail20d(Factor):
    """分位数尾部厚度因子.

    99分位|ret| / 90分位|ret|. 稳健的分位数版尾部厚度 (互补于 #53 矩估计).
    高比值 → 极端尾部突出 → 尾部风险 → 负向.
    方向: 负向.
    """
    name = "intraday_kurtosis_tail_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "分位数尾部厚度 (p99|ret|/p90|ret|)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        tails: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 30:
                    continue
                ar = r.abs().values
                p90, p99 = np.percentile(ar, [90, 99])
                vals[col] = float(p99 / p90) if p90 > 1e-12 else 1.0
            if vals:
                tails[dt] = pd.Series(vals)
        if not tails:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(tails).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 123. intraday_volume_trend — 量的时间趋势
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_volume_trend_20d", category="intraday_advanced")
class IntradayVolumeTrend20d(Factor):
    """量的时间趋势因子.

    分钟成交量对时间线性回归的斜率 (归一化).
    量递增 → 参与度上升/资金流入 → 正向.
    方向: 正向.
    """
    name = "intraday_volume_trend_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "量的时间趋势 (成交量线性斜率, 递增=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        volume = panel["volume"]
        day = volume.index.normalize()
        trends: dict = {}
        for dt in sorted(set(day)):
            grp = volume.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                v = grp[col].dropna().values
                if len(v) < 30:
                    continue
                t = np.arange(len(v)) / max(1, len(v) - 1)
                slope = np.polyfit(t, v, 1)[0]
                v_mean = v.mean()
                vals[col] = float(slope / v_mean) if v_mean > 1e-12 else 0.0
            if vals:
                trends[dt] = pd.Series(vals)
        if not trends:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(trends).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 124. intraday_new_high_volume — 创新高放量确认
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_new_high_volume_20d", category="intraday_advanced")
class IntradayNewHighVolume20d(Factor):
    """创新高放量确认因子.

    价格创日内新高的分钟平均成交量 / 全日均量.
    创新高伴随放量 → 突破真实 → 正向. 缩量新高 → 假突破 → 负向.
    方向: 正向.
    """
    name = "intraday_new_high_volume_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "创新高放量确认 (新高分钟均量/全日均量)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"high", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, volume = panel["high"], panel["volume"]
        day = high.index.normalize()
        confirms: dict = {}
        for dt in sorted(set(day)):
            grp_h = high.loc[day == dt]
            grp_v = volume.loc[day == dt]
            if len(grp_h) < 20:
                continue
            vals = {}
            for col in grp_h.columns:
                h = grp_h[col].dropna()
                v = grp_v[col].dropna()
                common = h.index.intersection(v.index)
                if len(common) < 20:
                    continue
                h_c = h.loc[common]
                v_c = v.loc[common]
                cum_max = h_c.cummax()
                new_high_mask = (h_c == cum_max) & (cum_max > cum_max.shift(1).fillna(cum_max.iloc[0] - 1.0))
                base_vol = v_c.mean()
                if base_vol < 1e-12:
                    continue
                if new_high_mask.sum() >= 2:
                    vals[col] = float(v_c[new_high_mask].mean() / base_vol)
                else:
                    vals[col] = 1.0
            if vals:
                confirms[dt] = pd.Series(vals)
        if not confirms:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(confirms).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 125. intraday_obv_slope — OBV 斜率
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_obv_slope_20d", category="intraday_advanced")
class IntradayObvSlope20d(Factor):
    """OBV 斜率因子.

    日内 OBV = Σ sign(ret)·volume 的线性趋势斜率 (归一化).
    OBV 上升 → 量价同向积累 → 资金净流入 → 正向.
    方向: 正向.
    """
    name = "intraday_obv_slope_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "OBV斜率 (Σsign(ret)×vol的趋势)"
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
        slopes: dict = {}
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
                r_c = r.loc[common].values
                v_c = v.loc[common].values
                obv = np.cumsum(np.sign(r_c) * v_c)
                t = np.arange(len(obv)) / max(1, len(obv) - 1)
                slope = np.polyfit(t, obv, 1)[0]
                base = (v_c.mean() * len(v_c))
                vals[col] = float(slope / base) if base > 1e-12 else 0.0
            if vals:
                slopes[dt] = pd.Series(vals)
        if not slopes:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(slopes).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 126. intraday_path_above_vwap — VWAP 上方时间占比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_path_above_vwap_20d", category="intraday_advanced")
class IntradayPathAboveVwap20d(Factor):
    """VWAP 上方时间占比因子.

    价格在 VWAP 上方的分钟占比 (时间维度, 互补于 #30 偏离度的幅度维度).
    长时间在上方 → 买方主导全天 → 正向.
    方向: 正向.
    """
    name = "intraday_path_above_vwap_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "VWAP上方时间占比 (买方主导时长)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, volume = panel["close"], panel["volume"]
        day = close.index.normalize()
        above: dict = {}
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
                vwap = float((c_c * v_c).sum() / v_c.sum()) if v_c.sum() > 1e-12 else c_c.mean()
                vals[col] = float((c_c > vwap).sum() / len(c_c))
            if vals:
                above[dt] = pd.Series(vals)
        if not above:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(above).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 127. intraday_rally_volume_ratio — 浪级涨跌量比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_rally_volume_ratio_20d", category="intraday_advanced")
class IntradayRallyVolumeRatio20d(Factor):
    """浪级涨跌量比因子.

    用摆动转折点把日内分成上涨浪/下跌浪 (浪级, 不同于 #29 分钟级),
    上涨浪平均分钟量 / 下跌浪平均分钟量.
    上涨浪放量 → 多头浪更实 → 正向.
    方向: 正向.
    """
    name = "intraday_rally_volume_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "浪级涨跌量比 (摆动分浪, 上涨浪均量/下跌浪均量)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    @staticmethod
    def _swing_points(values):
        """返回摆动转折点索引 (局部极值)."""
        n = len(values)
        pts = []
        for i in range(1, n - 1):
            if values[i] > values[i - 1] and values[i] > values[i + 1]:
                pts.append((i, 1))  # 高点
            elif values[i] < values[i - 1] and values[i] < values[i + 1]:
                pts.append((i, -1))  # 低点
        return pts

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, volume = panel["close"], panel["volume"]
        day = close.index.normalize()
        ratios: dict = {}
        for dt in sorted(set(day)):
            grp_c = close.loc[day == dt]
            grp_v = volume.loc[day == dt]
            if len(grp_c) < 30:
                continue
            vals = {}
            for col in grp_c.columns:
                c = grp_c[col].dropna()
                v = grp_v[col].dropna()
                common = c.index.intersection(v.index)
                if len(common) < 30:
                    continue
                c_c = c.loc[common].values
                v_c = v.loc[common].values
                pts = self._swing_points(c_c)
                if len(pts) < 2:
                    vals[col] = 1.0
                    continue
                # 转折点间的段: 判断上涨浪/下跌浪并累计分钟量
                up_vols, dn_vols = [], []
                bounds = [0] + [p[0] for p in pts] + [len(c_c) - 1]
                for k in range(len(bounds) - 1):
                    s, e = bounds[k], bounds[k + 1]
                    if e <= s:
                        continue
                    seg_ret = c_c[e] - c_c[s]
                    seg_vol = float(v_c[s:e + 1].mean())
                    if seg_ret > 0:
                        up_vols.append(seg_vol)
                    elif seg_ret < 0:
                        dn_vols.append(seg_vol)
                up_m = np.mean(up_vols) if up_vols else 0.0
                dn_m = np.mean(dn_vols) if dn_vols else 0.0
                vals[col] = float(up_m / dn_m) if dn_m > 1e-12 else (2.0 if up_m > 1e-12 else 1.0)
            if vals:
                ratios[dt] = pd.Series(vals)
        if not ratios:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(ratios).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 128. intraday_choppiness — 市场纠结度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_choppiness_20d", category="intraday_advanced")
class IntradayChoppiness20d(Factor):
    """市场纠结度因子 (Choppiness Index).

    CHOP = log10(Σ|ret| / (high-low)) / log10(n). 接近1=强趋势, 接近0=震荡.
    取 -CHOP 使震荡(低CHOP)=高值=负向信号; 实际输出 -CHOP 高值=纠结=负向.
    方向: 负向.
    """
    name = "intraday_choppiness_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "市场纠结度 (-CHOP指数, 震荡纠结=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"high", "low", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, low, close = panel["high"], panel["low"], panel["close"]
        ret_1m = close.pct_change()
        day = ret_1m.index.normalize()
        chops: dict = {}
        for dt in sorted(set(day)):
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            grp_ret = ret_1m.loc[day == dt]
            if len(grp_ret) < 20:
                continue
            vals = {}
            for col in grp_ret.columns:
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                r = grp_ret[col].dropna()
                common = h.index.intersection(l.index).intersection(r.index)
                if len(common) < 20:
                    continue
                h_c, l_c, r_c = (x.loc[common] for x in (h, l, r))
                rng = h_c.max() - l_c.min()
                if rng < 1e-12:
                    vals[col] = 0.0
                    continue
                sum_ret = r_c.abs().sum()
                n = len(r_c)
                chop = np.log10(sum_ret / rng) / np.log10(n) if n > 1 and sum_ret > 1e-12 else 0.0
                vals[col] = -float(chop)
            if vals:
                chops[dt] = pd.Series(vals)
        if not chops:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(chops).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 129. intraday_adx — 日内趋势强度 (ADX)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_adx_20d", category="intraday_advanced")
class IntradayADX20d(Factor):
    """日内趋势强度因子 (ADX 简化版).

    ADX = 100·|DM+ - DM-|/(DM+ + DM-), DM基于分钟high/low扩张.
    高ADX → 强趋势 → 方向明确 → 正向.
    方向: 正向.
    """
    name = "intraday_adx_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "日内趋势强度 (简化ADX, 强趋势=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"high", "low"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, low = panel["high"], panel["low"]
        day = high.index.normalize()
        adxs: dict = {}
        for dt in sorted(set(day)):
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            if len(grp_h) < 30:
                continue
            vals = {}
            for col in grp_h.columns:
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                common = h.index.intersection(l.index)
                if len(common) < 30:
                    continue
                h_c, l_c = (x.loc[common] for x in (h, l))
                dh = h_c.diff()
                dl = -l_c.diff()
                up_move = dh.clip(lower=0)
                dn_move = dl.clip(lower=0)
                dm_plus = ((up_move > dn_move) & (up_move > 0)) * up_move
                dm_minus = ((dn_move > up_move) & (dn_move > 0)) * dn_move
                sum_plus = dm_plus.sum()
                sum_minus = dm_minus.sum()
                denom = sum_plus + sum_minus
                vals[col] = float(100.0 * abs(sum_plus - sum_minus) / denom) if denom > 1e-12 else 0.0
            if vals:
                adxs[dt] = pd.Series(vals)
        if not adxs:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(adxs).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 130. intraday_liquidity_vol_ratio — 流动性调整波动
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_liquidity_vol_ratio_20d", category="intraday_advanced")
class IntradayLiquidityVolRatio20d(Factor):
    """流动性调整波动因子.

    成交额 / 日内波动率 (单位波动的流动性支撑).
    高比值 → 波动由大资金流动驱动 (健康) → 正向.
    方向: 正向.
    """
    name = "intraday_liquidity_vol_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "流动性调整波动 (成交额/波动率, 资金支撑=正向)"
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
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        ret_1m = close.pct_change()
        day = ret_1m.index.normalize()
        ratios: dict = {}
        for dt in sorted(set(day)):
            grp_amt = amt.loc[day == dt]
            grp_ret = ret_1m.loc[day == dt]
            if len(grp_ret) < 20:
                continue
            vals = {}
            for col in grp_ret.columns:
                a = grp_amt[col].dropna()
                r = grp_ret[col].dropna()
                common = a.index.intersection(r.index)
                if len(common) < 20:
                    continue
                a_c = a.loc[common]
                r_c = r.loc[common]
                vol = r_c.std(ddof=0)
                total_amt = a_c.sum()
                vals[col] = float(total_amt / vol) if vol > 1e-12 else 0.0
            if vals:
                ratios[dt] = pd.Series(vals)
        if not ratios:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(ratios).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 131. intraday_amihud_stability — 非流动性稳定性
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_amihud_stability_20d", category="intraday_advanced")
class IntradayAmihudStability20d(Factor):
    """非流动性稳定性因子.

    Amihud 分钟序列的 均值/标准差 (稳定性, 互补于 #62 的前后变化).
    稳定的低冲击 → 流动性可预期 → 正向.
    方向: 正向.
    """
    name = "intraday_amihud_stability_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "非流动性稳定 (Amihud均值/标准差)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "amount"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, amount = panel["close"], panel["amount"]
        ret_1m = close.pct_change().abs()
        day = ret_1m.index.normalize()
        stabilities: dict = {}
        for dt in sorted(set(day)):
            grp_ret = ret_1m.loc[day == dt]
            grp_amt = amount.loc[day == dt]
            if len(grp_ret) < 20:
                continue
            vals = {}
            for col in grp_ret.columns:
                r = grp_ret[col].dropna()
                a = grp_amt[col].dropna()
                common = r.index.intersection(a.index)
                if len(common) < 20:
                    continue
                amihud = r.loc[common] / (a.loc[common] + 1e-12)
                amihud = amihud[amihud < 10.0]
                if len(amihud) < 10:
                    continue
                m, s = amihud.mean(), amihud.std(ddof=0)
                vals[col] = float(m / s) if s > 1e-12 else 0.0
            if vals:
                stabilities[dt] = pd.Series(vals)
        if not stabilities:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(stabilities).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 132. intraday_depth_trend — 深度变化率
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_depth_trend_20d", category="intraday_advanced")
class IntradayDepthTrend20d(Factor):
    """深度变化率因子.

    后半段平均成交额/振幅 vs 前半段 (深度随时间的变化, 互补于 #87 均值).
    深度递增 → 流动性持续改善 → 正向.
    方向: 正向.
    """
    name = "intraday_depth_trend_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "深度变化率 (后半段深度/前半段, 改善=正向)"
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
        if not {"high", "low"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, low = panel["high"], panel["low"]
        day = amt.index.normalize()
        trends: dict = {}
        for dt in sorted(set(day)):
            grp_amt = amt.loc[day == dt]
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            n = len(grp_amt)
            if n < 30:
                continue
            mid = n // 2
            vals = {}
            for col in grp_amt.columns:
                a = grp_amt[col].dropna()
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                common = a.index.intersection(h.index).intersection(l.index)
                if len(common) < 30:
                    continue
                a_c, h_c, l_c = (x.loc[common] for x in (a, h, l))
                rng = (h_c - l_c).replace(0, np.nan)
                depth = a_c / rng
                first = depth.iloc[:mid].mean()
                second = depth.iloc[mid:].mean()
                vals[col] = float(second / first) if first > 1e-12 else 1.0
            if vals:
                trends[dt] = pd.Series(vals)
        if not trends:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(trends).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 133. intraday_zero_volume_run — 流动性中断持续
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_zero_volume_run_20d", category="intraday_advanced")
class IntradayZeroVolumeRun20d(Factor):
    """流动性中断持续因子.

    连续极低量 (≤1%分位) 分钟的最长连续长度 (互补于 #85 的占比).
    长中断 → 流动性断层持续 → 交易风险 → 负向.
    方向: 负向.
    """
    name = "intraday_zero_volume_run_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "流动性中断 (极低量最长连续分钟, 长=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        volume = panel["volume"]
        day = volume.index.normalize()
        runs: dict = {}
        for dt in sorted(set(day)):
            grp = volume.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                v = grp[col].dropna().values
                if len(v) < 30:
                    continue
                q01 = np.percentile(v, 1)
                low_mask = v <= q01
                max_run = 0
                cur = 0
                for m in low_mask:
                    cur = cur + 1 if m else 0
                    max_run = max(max_run, cur)
                vals[col] = -float(max_run)
            if vals:
                runs[dt] = pd.Series(vals)
        if not runs:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(runs).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 134. intraday_panic_strength — 恐慌强度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_panic_strength_20d", category="intraday_advanced")
class IntradayPanicStrength20d(Factor):
    """恐慌强度因子.

    最长连续下跌run的长度 × 期间累计跌幅 (恐慌抛售的强度).
    与 #41 最长run互补 (这里聚焦下跌且加权跌幅).
    方向: 负向.
    """
    name = "intraday_panic_strength_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "恐慌强度 (最长连跌run×跌幅)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        panics: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 20:
                    continue
                best_len, best_loss = 0, 0.0
                cur_len, cur_loss = 0, 0.0
                for x in r:
                    if x < 0:
                        cur_len += 1
                        cur_loss += x
                        if cur_loss < best_loss:
                            best_loss = cur_loss
                            best_len = cur_len
                    else:
                        cur_len, cur_loss = 0, 0.0
                vals[col] = float(best_len * best_loss)  # 负值
            if vals:
                panics[dt] = pd.Series(vals)
        if not panics:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(panics).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 135. intraday_euphoria_strength — 狂热强度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_euphoria_strength_20d", category="intraday_advanced")
class IntradayEuphoriaStrength20d(Factor):
    """狂热强度因子.

    最长连续上涨run的长度 × 期间累计涨幅.
    连续放量上涨 → 情绪亢奋 → 动量强但易透支 → 短期正向.
    方向: 正向.
    """
    name = "intraday_euphoria_strength_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "狂热强度 (最长连涨run×涨幅)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        euphorias: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 20:
                    continue
                best_len, best_gain = 0, 0.0
                cur_len, cur_gain = 0, 0.0
                for x in r:
                    if x > 0:
                        cur_len += 1
                        cur_gain += x
                        if cur_gain > best_gain:
                            best_gain = cur_gain
                            best_len = cur_len
                    else:
                        cur_len, cur_gain = 0, 0.0
                vals[col] = float(best_len * best_gain)
            if vals:
                euphorias[dt] = pd.Series(vals)
        if not euphorias:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(euphorias).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 136. intraday_wash_trade — 洗盘特征频率
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_wash_trade_20d", category="intraday_advanced")
class IntradayWashTrade20d(Factor):
    """洗盘特征因子.

    快速下跌 (5分钟内累计跌幅>0.5%) 后快速收回 (10分钟内收复) 的频率.
    频繁洗盘 → 主力操纵/假摔 → 不稳定 → 负向.
    方向: 负向.
    """
    name = "intraday_wash_trade_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "洗盘特征 (快跌后快收频率, 操纵=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        washes: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 40:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna().values
                if len(c) < 40:
                    continue
                n = len(c)
                cnt = 0
                for i in range(n - 10):
                    drop_win = c[i:i + 6]
                    if len(drop_win) < 6:
                        continue
                    drop = drop_win[-1] / drop_win[0] - 1.0
                    if drop < -0.005:  # 5分钟跌超0.5%
                        low = drop_win.min()
                        for j in range(i + 6, min(i + 16, n)):
                            if c[j] >= drop_win[0]:
                                cnt += 1
                                break
                vals[col] = -float(cnt)  # 洗盘多=负向
            if vals:
                washes[dt] = pd.Series(vals)
        if not washes:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(washes).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 137. intraday_close_auction_pressure — 收盘竞价压力
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_close_auction_pressure_20d", category="intraday_advanced")
class IntradayCloseAuctionPressure20d(Factor):
    """收盘竞价压力因子.

    最后5分钟收益方向×幅度 (收盘竞价/尾盘抢筹压力).
    尾盘强势买压 → 收盘动能 → 正向.
    方向: 正向.
    """
    name = "intraday_close_auction_pressure_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "收盘竞价压力 (最后5分钟收益)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        pressures: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 15:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < 15:
                    continue
                start = c.iloc[-6] if len(c) >= 6 else c.iloc[0]
                vals[col] = float(c.iloc[-1] / start - 1.0)
            if vals:
                pressures[dt] = pd.Series(vals)
        if not pressures:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(pressures).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 138. intraday_support_test — 支撑测试
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_support_test_20d", category="intraday_advanced")
class IntradaySupportTest20d(Factor):
    """支撑测试因子.

    盘中价格反复触及 (±ε) 已形成低点区域的次数.
    支撑被反复测试 → 卖压不断 → 支撑位承压 → 负向.
    方向: 负向.
    """
    name = "intraday_support_test_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "支撑测试 (反复触及低点区域次数)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"low", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        low, close = panel["low"], panel["close"]
        day = low.index.normalize()
        tests: dict = {}
        for dt in sorted(set(day)):
            grp_l = low.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_l) < 30:
                continue
            vals = {}
            for col in grp_l.columns:
                l = grp_l[col].dropna()
                c = grp_c[col].dropna()
                common = l.index.intersection(c.index)
                if len(common) < 30:
                    continue
                l_c = l.loc[common]
                c_c = c.loc[common]
                day_low = l_c.min()
                if day_low < 1e-12:
                    continue
                eps = (c_c.max() - day_low) * 0.02 + 1e-8  # 2%容差带
                touch = ((l_c - day_low).abs() <= eps).sum()
                vals[col] = -float(touch)  # 测试多=负向
            if vals:
                tests[dt] = pd.Series(vals)
        if not tests:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(tests).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 139. intraday_resistance_test — 阻力测试
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_resistance_test_20d", category="intraday_advanced")
class IntradayResistanceTest20d(Factor):
    """阻力测试因子.

    盘中价格反复触及高点区域的次数.
    阻力反复被测试 → 上攻无力 → 负向.
    方向: 负向.
    """
    name = "intraday_resistance_test_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "阻力测试 (反复触及高点区域次数)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"high", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, close = panel["high"], panel["close"]
        day = high.index.normalize()
        tests: dict = {}
        for dt in sorted(set(day)):
            grp_h = high.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_h) < 30:
                continue
            vals = {}
            for col in grp_h.columns:
                h = grp_h[col].dropna()
                c = grp_c[col].dropna()
                common = h.index.intersection(c.index)
                if len(common) < 30:
                    continue
                h_c = h.loc[common]
                c_c = c.loc[common]
                day_high = h_c.max()
                eps = (day_high - c_c.min()) * 0.02 + 1e-8
                touch = ((h_c - day_high).abs() <= eps).sum()
                vals[col] = -float(touch)
            if vals:
                tests[dt] = pd.Series(vals)
        if not tests:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(tests).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 140. intraday_range_crossing — 区间穿越次数
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_range_crossing_20d", category="intraday_advanced")
class IntradayRangeCrossing20d(Factor):
    """区间穿越次数因子.

    价格穿越日内中位价线的次数 (震荡度, 与 #101 转折点互补: 这是穿越).
    频繁穿越 → 无方向/震荡市 → 负向.
    方向: 负向.
    """
    name = "intraday_range_crossing_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "区间穿越 (穿越中位价线次数, 震荡=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"high", "low", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, low, close = panel["high"], panel["low"], panel["close"]
        day = close.index.normalize()
        crossings: dict = {}
        for dt in sorted(set(day)):
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 30:
                continue
            vals = {}
            for col in grp_c.columns:
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                c = grp_c[col].dropna()
                common = h.index.intersection(l.index).intersection(c.index)
                if len(common) < 30:
                    continue
                h_c, l_c, c_c = (x.loc[common] for x in (h, l, c))
                mid = (h_c.max() + l_c.min()) / 2.0
                above = (c_c > mid).astype(int).values
                crossings_n = int(np.sum(np.diff(above) != 0))
                vals[col] = -float(crossings_n)
            if vals:
                crossings[dt] = pd.Series(vals)
        if not crossings:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(crossings).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 141. intraday_breakout_retest — 突破回踩确认
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_breakout_retest_20d", category="intraday_advanced")
class IntradayBreakoutRetest20d(Factor):
    """突破回踩确认因子.

    价格突破日内前段高点/低点后回踩不破再延续的次数 (有效突破).
    与 #35 互补: #35看收盘位置, 这里看突破后的回踩行为.
    高确认 → 突破真实有效 → 正向.
    方向: 正向.
    """
    name = "intraday_breakout_retest_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "突破回踩确认 (突破后回踩不破次数)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        confirms: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 40:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna().values
                if len(c) < 40:
                    continue
                n = len(c)
                cnt = 0
                running_high = np.maximum.accumulate(c)
                running_low = np.minimum.accumulate(c)
                for i in range(5, n - 5):
                    # 突破前高
                    if c[i] > running_high[i - 1]:
                        retest_ok = False
                        for j in range(i + 1, min(i + 6, n)):
                            if abs(c[j] - running_high[i - 1]) < 0.001 * running_high[i - 1]:
                                retest_ok = c[j + 1] > c[j] if j + 1 < n else False
                                break
                        if retest_ok:
                            cnt += 1
                vals[col] = float(cnt)
            if vals:
                confirms[dt] = pd.Series(vals)
        if not confirms:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(confirms).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 142. intraday_vol_compression — 跨日波动压缩
# ⚠ 跨日因子: 依赖前10日振幅 (rolling(10, min_periods=3)), 需≥4个交易日历史; 滚动窗口前为NaN
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_vol_compression_20d", category="intraday_advanced")
class IntradayVolCompression20d(Factor):
    """跨日波动压缩因子.

    今日日内振幅 / 近10日平均振幅. 低压缩比 → 波动收窄 → 变盘前兆.
    压缩到极致 → 突破在即 → 正向 (压缩=蓄势).
    方向: 正向 (取 -比值).
    """
    name = "intraday_vol_compression_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "跨日波动压缩 (今日振幅/近10日均值, 取负=压缩蓄势=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"high", "low"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, low = panel["high"], panel["low"]
        day = high.index.normalize()
        daily_range: dict = {}
        for dt in sorted(set(day)):
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            if len(grp_h) < 20:
                continue
            vals = {}
            for col in grp_h.columns:
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                common = h.index.intersection(l.index)
                if len(common) < 20:
                    continue
                vals[col] = float(h.loc[common].max() / l.loc[common].min() - 1.0)
            if vals:
                daily_range[dt] = pd.Series(vals)
        if not daily_range:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        rng_df = pd.DataFrame(daily_range).T
        rng_df.index = pd.DatetimeIndex(rng_df.index)
        rng_df = rng_df.reindex(dates)
        rng_ma = rng_df.rolling(10, min_periods=3).mean()
        ratios = rng_df / rng_ma.replace(0, np.nan)
        return (-ratios).reindex(columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 143. intraday_risk_adj_momentum — 风险调整动量
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_risk_adj_momentum_20d", category="intraday_advanced")
class IntradayRiskAdjMomentum20d(Factor):
    """风险调整动量因子.

    日内收益 / 日内波动率 (日内Sharpe比率).
    高风险调整收益 → 涨得稳 → 质量高 → 正向.
    方向: 正向.
    """
    name = "intraday_risk_adj_momentum_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "风险调整动量 (日内收益/波动率, Sharpe-like)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"open", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        open_px, close = panel["open"], panel["close"]
        ret_1m = close.pct_change()
        day = close.index.normalize()
        rarms: dict = {}
        for dt in sorted(set(day)):
            grp_o = open_px.loc[day == dt]
            grp_c = close.loc[day == dt]
            grp_r = ret_1m.loc[day == dt]
            if len(grp_c) < 20:
                continue
            vals = {}
            for col in grp_c.columns:
                o = grp_o[col].dropna()
                c = grp_c[col].dropna()
                r = grp_r[col].dropna()
                common = o.index.intersection(c.index).intersection(r.index)
                if len(common) < 20:
                    continue
                o_c, c_c, r_c = (x.loc[common] for x in (o, c, r))
                o_first = o_c.iloc[0]
                if o_first < 1e-12:
                    continue
                intraday_ret = c_c.iloc[-1] / o_first - 1.0
                vol = r_c.std(ddof=0)
                vals[col] = float(intraday_ret / vol) if vol > 1e-12 else 0.0
            if vals:
                rarms[dt] = pd.Series(vals)
        if not rarms:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(rarms).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 144. intraday_session_consistency — 多时段方向一致性
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_session_consistency_20d", category="intraday_advanced")
class IntradaySessionConsistency20d(Factor):
    """多时段方向一致性因子.

    日内分4段 (开盘/上午/下午/尾盘), 统计与全天方向一致的段数.
    全段同向 → 单边市 → 一致性高 → 正向. 互补于 #97 (2段).
    方向: 正向.
    """
    name = "intraday_session_consistency_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "多时段方向一致 (4段同向段数)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        consistencies: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 40:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < 40:
                    continue
                n = len(c)
                q = n // 4
                segs = [c.iloc[0], c.iloc[q], c.iloc[2 * q], c.iloc[3 * q], c.iloc[-1]]
                seg_rets = [segs[i + 1] / segs[i] - 1.0 for i in range(4)]
                day_dir = np.sign(sum(seg_rets))
                if abs(day_dir) < 1e-12:
                    vals[col] = 0.0
                    continue
                consistent = sum(1 for sr in seg_rets if abs(sr) > 1e-12 and np.sign(sr) == day_dir)
                vals[col] = float(consistent)
            if vals:
                consistencies[dt] = pd.Series(vals)
        if not consistencies:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(consistencies).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 145. intraday_volume_momentum — 跨日量动量
# ⚠ 跨日因子: 依赖昨日成交量 (prev_total 跨日追踪), 需≥2个交易日历史; 首日为NaN
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_volume_momentum_20d", category="intraday_advanced")
class IntradayVolumeMomentum20d(Factor):
    """跨日量动量因子.

    今日成交量 / 昨日成交量 (跨日量能变化, 不同于 #23 日内开尾比).
    量能放大 → 关注度上升 → 动量延续 → 正向.
    方向: 正向.
    """
    name = "intraday_volume_momentum_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "跨日量动量 (今日量/昨日量)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        volume = panel["volume"]
        day = volume.index.normalize()
        moments: dict = {}
        prev_total: dict = {}
        for dt in sorted(set(day)):
            grp = volume.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                v = grp[col].dropna()
                if len(v) < 20:
                    continue
                today = v.sum()
                prev = prev_total.get(col)
                if prev is None or prev < 1e-12:
                    prev_total[col] = today
                    continue
                vals[col] = float(today / prev)
                prev_total[col] = today
            if vals:
                moments[dt] = pd.Series(vals)
        if not moments:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(moments).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 146. intraday_order_flow_variability — 订单流不稳定度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_order_flow_variability_20d", category="intraday_advanced")
class IntradayOrderFlowVariability20d(Factor):
    """订单流不稳定度因子.

    分钟买卖不平衡序列的变异系数 (互补于 #60 的均值: 这里是波动).
    不平衡忽大忽小 → 方向反复 → 情绪化 → 负向.
    方向: 负向.
    """
    name = "intraday_order_flow_variability_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "订单流不稳定 (不平衡序列变异系数)"
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
        variabilities: dict = {}
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
                imbalance = np.sign(r.loc[common].values) * v.loc[common].values
                m, s = imbalance.mean(), imbalance.std(ddof=0)
                vals[col] = float(s / abs(m)) if abs(m) > 1e-12 else float(s / (v.loc[common].mean() + 1e-12))
            if vals:
                variabilities[dt] = pd.Series(vals)
        if not variabilities:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(variabilities).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 147. intraday_vwap_position — 收盘VWAP位置
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_vwap_position_20d", category="intraday_advanced")
class IntradayVwapPosition20d(Factor):
    """收盘VWAP位置因子.

    收盘价相对VWAP的方向性位置: (close - vwap) / std(close).
    互补于 #30 偏离度 (幅度) 与 #126 时间占比: 这里是收盘时点位置.
    收盘高于VWAP → 尾盘定价偏多 → 正向.
    方向: 正向.
    """
    name = "intraday_vwap_position_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "收盘VWAP位置 ((close-vwap)/std, 收盘偏多=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, volume = panel["close"], panel["volume"]
        day = close.index.normalize()
        positions: dict = {}
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
                vwap = float((c_c * v_c).sum() / v_c.sum()) if v_c.sum() > 1e-12 else c_c.mean()
                std_c = c_c.std(ddof=0)
                if std_c < 1e-12:
                    vals[col] = 0.0
                else:
                    vals[col] = float((c_c.iloc[-1] - vwap) / std_c)
            if vals:
                positions[dt] = pd.Series(vals)
        if not positions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(positions).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 148. intraday_up_minute_ratio — 上涨分钟占比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_up_minute_ratio_20d", category="intraday_advanced")
class IntradayUpMinuteRatio20d(Factor):
    """上涨分钟占比因子.

    上涨分钟数 / 总分钟数 (多头时间优势).
    多数时间在涨 → 买方掌控全天 → 正向.
    方向: 正向.
    """
    name = "intraday_up_minute_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "上涨分钟占比 (多头时间优势)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        ratios: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 20:
                    continue
                up = (r > 0).sum()
                dn = (r < 0).sum()
                total = up + dn
                vals[col] = float(up / total) if total > 0 else 0.5
            if vals:
                ratios[dt] = pd.Series(vals)
        if not ratios:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(ratios).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 149. intraday_high_low_break_ratio — 创新高/新低次数比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_high_low_break_ratio_20d", category="intraday_advanced")
class IntradayHighLowBreakRatio20d(Factor):
    """创新高/新低次数比因子.

    日内创阶段新高的次数 / (创新高+创新低次数).
    突破向上为主 → 多方攻势 → 正向.
    方向: 正向.
    """
    name = "intraday_high_low_break_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "创新高/新低次数比 (突破方向平衡)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        ratios: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna().values
                if len(c) < 30:
                    continue
                cum_max = np.maximum.accumulate(c)
                cum_min = np.minimum.accumulate(c)
                new_high = int(np.sum(c[1:] > cum_max[:-1]))
                new_low = int(np.sum(c[1:] < cum_min[:-1]))
                total = new_high + new_low
                vals[col] = float(new_high / total) if total > 0 else 0.5
            if vals:
                ratios[dt] = pd.Series(vals)
        if not ratios:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(ratios).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 150. intraday_zero_ret_freq — 零收益分钟占比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_zero_ret_freq_20d", category="intraday_advanced")
class IntradayZeroRetFreq20d(Factor):
    """零收益分钟占比因子.

    分钟收益=0 的比例 (价格停滞/无成交/流动性缺乏).
    高频零收益 → 交易停滞 → 流动性差 → 负向.
    方向: 负向.
    """
    name = "intraday_zero_ret_freq_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "零收益分钟占比 (价格停滞=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        freqs: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 20:
                    continue
                vals[col] = float((r == 0).sum() / len(r))
            if vals:
                freqs[dt] = pd.Series(vals)
        if not freqs:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(freqs).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 151. intraday_signed_run_balance — 正负run时长平衡
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_signed_run_balance_20d", category="intraday_advanced")
class IntradaySignedRunBalance20d(Factor):
    """正负run时长平衡因子.

    上涨run总分钟数 - 下跌run总分钟数 (方向持续的时间净额).
    与 #134/#135 最大run强度互补: 这里看总时长平衡.
    正值 → 上涨主导时长 → 正向.
    方向: 正向.
    """
    name = "intraday_signed_run_balance_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "正负run时长平衡 (上涨总时长-下跌总时长)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        balances: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 20:
                    continue
                up_total = int(np.sum(r > 0))
                dn_total = int(np.sum(r < 0))
                vals[col] = float(up_total - dn_total)
            if vals:
                balances[dt] = pd.Series(vals)
        if not balances:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(balances).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 152. intraday_extreme_freq_balance — 极端涨跌频率差
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_extreme_freq_balance_20d", category="intraday_advanced")
class IntradayExtremeFreqBalance20d(Factor):
    """极端涨跌频率差因子.

    |ret|>2σ 的分钟中 上涨占比 (极端方向上多空频率).
    与 #118 频率互补: #118只看极端总频率, 这里分方向.
    极端上涨为主 → 多方爆发力 → 正向.
    方向: 正向.
    """
    name = "intraday_extreme_freq_balance_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "极端涨跌频率差 (|ret|>2σ中上涨占比)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        balances: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 30:
                    continue
                sigma = r.std(ddof=0)
                if sigma < 1e-12:
                    vals[col] = 0.5
                    continue
                extreme = r[np.abs(r) > 2.0 * sigma]
                if len(extreme) == 0:
                    vals[col] = 0.5
                else:
                    vals[col] = float((extreme > 0).sum() / len(extreme))
            if vals:
                balances[dt] = pd.Series(vals)
        if not balances:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(balances).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 153. intraday_vs_prev_high — 相对前日高点距离
# ⚠ 跨日因子: 依赖前日高点 (prev_high 跨日追踪), 需≥2个交易日历史; 首日为NaN
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_vs_prev_high_20d", category="intraday_advanced")
class IntradayVsPrevHigh20d(Factor):
    """相对前日高点距离因子 (跨日参考点).

    (收盘 - 前日高点) / 前日高点. 正值=突破前高.
    逼近/突破前高 → 阻力测试 → 上方抛压 → 负向.
    方向: 负向.
    """
    name = "intraday_vs_prev_high_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "相对前日高点 ((close-prev_high)/prev_high, 突破=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"high", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, close = panel["high"], panel["close"]
        day = close.index.normalize()
        distances: dict = {}
        prev_high: dict = {}
        for dt in sorted(set(day)):
            grp_h = high.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 5:
                continue
            vals = {}
            for col in grp_c.columns:
                h = grp_h[col].dropna()
                c = grp_c[col].dropna()
                if len(h) < 5 or len(c) < 5:
                    continue
                ph = prev_high.get(col)
                if ph is None or ph < 1e-12:
                    prev_high[col] = h.max()
                    continue
                vals[col] = float(c.iloc[-1] / ph - 1.0)
                prev_high[col] = h.max()
            if vals:
                distances[dt] = pd.Series(vals)
        if not distances:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(distances).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 154. intraday_vs_prev_low — 相对前日低点距离
# ⚠ 跨日因子: 依赖前日低点 (prev_low 跨日追踪), 需≥2个交易日历史; 首日为NaN
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_vs_prev_low_20d", category="intraday_advanced")
class IntradayVsPrevLow20d(Factor):
    """相对前日低点距离因子 (跨日参考点).

    (收盘 - 前日低点) / 前日低点. 远离前低=支撑有效.
    支撑有效 → 抛压受限 → 正向.
    方向: 正向.
    """
    name = "intraday_vs_prev_low_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "相对前日低点 ((close-prev_low)/prev_low, 远离=支撑有效=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"low", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        low, close = panel["low"], panel["close"]
        day = close.index.normalize()
        distances: dict = {}
        prev_low: dict = {}
        for dt in sorted(set(day)):
            grp_l = low.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 5:
                continue
            vals = {}
            for col in grp_c.columns:
                l = grp_l[col].dropna()
                c = grp_c[col].dropna()
                if len(l) < 5 or len(c) < 5:
                    continue
                pl = prev_low.get(col)
                if pl is None or pl < 1e-12:
                    prev_low[col] = l.min()
                    continue
                vals[col] = float(c.iloc[-1] / pl - 1.0)
                prev_low[col] = l.min()
            if vals:
                distances[dt] = pd.Series(vals)
        if not distances:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(distances).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 155. intraday_low_before_high — 低点先于高点 (V形)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_low_before_high_20d", category="intraday_advanced")
class IntradayLowBeforeHigh20d(Factor):
    """低点先于高点因子 (V形路径).

    日内低点出现时间 < 高点出现时间 → V形探底回升 → 正向.
    高先低后 → 倒V冲高回落 → 负向. 与 #98/#99 时间互补 (这里是顺序关系).
    方向: 正向.
    """
    name = "intraday_low_before_high_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "低点先于高点 (V形探底回升=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"high", "low"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, low = panel["high"], panel["low"]
        day = high.index.normalize()
        shapes: dict = {}
        for dt in sorted(set(day)):
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            if len(grp_h) < 20:
                continue
            vals = {}
            for col in grp_h.columns:
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                common = h.index.intersection(l.index)
                if len(common) < 20:
                    continue
                h_c, l_c = (x.loc[common] for x in (h, l))
                idx_high = int(np.argmax(h_c.values))
                idx_low = int(np.argmin(l_c.values))
                vals[col] = 1.0 if idx_low < idx_high else -1.0
            if vals:
                shapes[dt] = pd.Series(vals)
        if not shapes:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(shapes).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 156. intraday_mid_line_time — 中位线上方时间占比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_mid_line_time_20d", category="intraday_advanced")
class IntradayMidLineTime20d(Factor):
    """中位线上方时间占比因子.

    价格在日内中位价线上方的分钟占比 (与 #126 VWAP线互补: 参考线不同).
    长时间在区间上半部 → 偏强 → 正向.
    方向: 正向.
    """
    name = "intraday_mid_line_time_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "中位线上方时间占比 (区间上半部停留时长)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"high", "low", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, low, close = panel["high"], panel["low"], panel["close"]
        day = close.index.normalize()
        times: dict = {}
        for dt in sorted(set(day)):
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 20:
                continue
            vals = {}
            for col in grp_c.columns:
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                c = grp_c[col].dropna()
                common = h.index.intersection(l.index).intersection(c.index)
                if len(common) < 20:
                    continue
                h_c, l_c, c_c = (x.loc[common] for x in (h, l, c))
                mid = (h_c.max() + l_c.min()) / 2.0
                vals[col] = float((c_c > mid).sum() / len(c_c))
            if vals:
                times[dt] = pd.Series(vals)
        if not times:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(times).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 157. intraday_drawdown_recover_ratio — 回撤收复比例
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_drawdown_recover_ratio_20d", category="intraday_advanced")
class IntradayDrawdownRecoverRatio20d(Factor):
    """回撤收复比例因子.

    (收盘 - 最大回撤谷底) / (回撤起点峰值 - 谷底).
    与 #106 恢复速度互补: 这里看最终收复的比例 (0~1), 那里看速度.
    高收复 → 回撤被充分消化 → 正向.
    方向: 正向.
    """
    name = "intraday_drawdown_recover_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "回撤收复比例 (收盘/峰值-谷底跨度)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        recoveries: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < 20:
                    continue
                cum = c.values / c.values[0] - 1.0
                peak = np.maximum.accumulate(cum)
                dd = peak - cum
                trough_idx = int(np.argmax(dd))
                peak_idx = int(np.argmax(cum[:trough_idx + 1]))
                peak_val = cum[peak_idx]
                trough_val = cum[trough_idx]
                span = peak_val - trough_val
                if span < 1e-12:
                    vals[col] = 1.0
                else:
                    recover = (cum[-1] - trough_val) / span
                    vals[col] = float(max(0.0, min(1.0, recover)))
            if vals:
                recoveries[dt] = pd.Series(vals)
        if not recoveries:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(recoveries).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 158. intraday_vol_ratio_2h — 前2小时/后2小时波动比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_vol_ratio_2h_20d", category="intraday_advanced")
class IntradayVolRatio2h20d(Factor):
    """前2小时/后2小时波动比因子.

    前2小时段波动 / 后2小时段波动 (与 #80 的5/30分钟窗口互补).
    前段波动大 → 开盘定价主导 → 隔夜信息释放 → 负向 (开盘冲动).
    方向: 负向.
    """
    name = "intraday_vol_ratio_2h_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "前2小时/后2小时波动比 (开盘定价主导=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        ratios: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            n = len(grp)
            if n < 60:
                continue
            mid = n // 2
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 60:
                    continue
                first = r.iloc[:mid].std(ddof=0)
                second = r.iloc[mid:].std(ddof=0)
                vals[col] = float(first / second) if second > 1e-12 else 1.0
            if vals:
                ratios[dt] = pd.Series(vals)
        if not ratios:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(ratios).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 159. intraday_vol_quarter_trend — 波动率四段趋势
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_vol_quarter_trend_20d", category="intraday_advanced")
class IntradayVolQuarterTrend20d(Factor):
    """波动率四段趋势因子.

    日内4等分段波动率对时间回归的斜率 (与 #123 量趋势对称).
    波动递增 → 风险随时间积聚 → 负向.
    方向: 负向.
    """
    name = "intraday_vol_quarter_trend_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "波动率四段趋势 (波动递增=风险积聚=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        trends: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            n = len(grp)
            if n < 60:
                continue
            q = n // 4
            if q < 5:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 60:
                    continue
                vols = [r.iloc[i * q:(i + 1) * q].std(ddof=0) for i in range(4)]
                t = np.arange(4)
                slope = np.polyfit(t, vols, 1)[0]
                base = np.mean(vols)
                vals[col] = float(slope / base) if base > 1e-12 else 0.0
            if vals:
                trends[dt] = pd.Series(vals)
        if not trends:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(trends).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 160. intraday_volatility_drift — 波动率凸度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_volatility_drift_20d", category="intraday_advanced")
class IntradayVolatilityDrift20d(Factor):
    """波动率凸度因子.

    分钟|ret|对时间二次回归的二次项 (波动路径的凸性, 与 #113 价格凸度对称).
    凸 (正二次项) → 前段平静后段爆发 → 蓄势后释放 → 负向 (不稳定).
    方向: 负向.
    """
    name = "intraday_volatility_drift_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "波动率凸度 (|ret|二次回归, 后段爆发=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        drifts: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 60:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 60:
                    continue
                ar = np.abs(r)
                t = np.arange(len(ar)) / max(1, len(ar) - 1)
                coeff = np.polyfit(t, ar, 2)
                vals[col] = float(coeff[0])
            if vals:
                drifts[dt] = pd.Series(vals)
        if not drifts:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(drifts).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 161. intraday_rv_half_life — 波动半衰期
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_rv_half_life_20d", category="intraday_advanced")
class IntradayRvHalfLife20d(Factor):
    """波动半衰期因子.

    分钟波动率自相关从1衰减到0.5所需的滞后数 (波动记忆时长).
    与 #55/#83 互补: #55测一阶相关, #83测AR1系数, 这里测衰减速度.
    记忆长 → 波动持续发酵 → 负向.
    方向: 负向.
    """
    name = "intraday_rv_half_life_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "波动半衰期 (波动自相关衰减滞后数, 长=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        half_lives: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 60:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 60:
                    continue
                ar = np.abs(r)
                max_lag = min(20, len(ar) // 4)
                rho0 = 1.0
                hl = max_lag
                for lag in range(1, max_lag + 1):
                    x, y = ar[:-lag], ar[lag:]
                    if x.std(ddof=0) < 1e-12 or y.std(ddof=0) < 1e-12:
                        continue
                    rho = float(np.corrcoef(x, y)[0, 1])
                    if rho <= rho0 * 0.5:
                        hl = lag
                        break
                vals[col] = -float(hl)  # 半衰期长=负向
            if vals:
                half_lives[dt] = pd.Series(vals)
        if not half_lives:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(half_lives).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 162. intraday_vol_upside — 波动率上偏
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_vol_upside_20d", category="intraday_advanced")
class IntradayVolUpside20d(Factor):
    """波动率上偏因子.

    后半段波动 / 前半段波动 (波动方向偏置).
    波动放大 → 后段不稳定 → 负向.
    方向: 负向.
    """
    name = "intraday_vol_upside_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "波动率上偏 (后半段/前半段波动, 放大=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        upsides: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            n = len(grp)
            if n < 40:
                continue
            mid = n // 2
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 40:
                    continue
                first = r.iloc[:mid].std(ddof=0)
                second = r.iloc[mid:].std(ddof=0)
                vals[col] = float(second / first) if first > 1e-12 else 1.0
            if vals:
                upsides[dt] = pd.Series(vals)
        if not upsides:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(upsides).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 163. intraday_tick_activity — 价格活跃度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_tick_activity_20d", category="intraday_advanced")
class IntradayTickActivity20d(Factor):
    """价格活跃度因子.

    价格发生变化 (ret≠0) 的分钟占比 (交易活跃度).
    高活跃 → 持续定价 → 流动性好 → 正向.
    方向: 正向.
    """
    name = "intraday_tick_activity_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "价格活跃度 (ret≠0分钟占比, 持续定价=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        activities: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 20:
                    continue
                vals[col] = float((r != 0).sum() / len(r))
            if vals:
                activities[dt] = pd.Series(vals)
        if not activities:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(activities).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 164. intraday_volume_tail_conc — 顶部量集中度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_volume_tail_conc_20d", category="intraday_advanced")
class IntradayVolumeTailConc20d(Factor):
    """顶部量集中度因子.

    top5%量分钟的量占总量的比例 (与 #21 整体基尼互补: 这里只看头部).
    头部集中 → 依赖少数大单 → 单点驱动 → 负向.
    方向: 负向.
    """
    name = "intraday_volume_tail_conc_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "顶部量集中 (top5%分钟量占比, 大单依赖=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        volume = panel["volume"]
        day = volume.index.normalize()
        concs: dict = {}
        for dt in sorted(set(day)):
            grp = volume.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                v = grp[col].dropna().values
                if len(v) < 30:
                    continue
                total = v.sum()
                if total < 1e-12:
                    continue
                n_top = max(1, int(len(v) * 0.05))
                top_sum = np.sum(np.sort(v)[-n_top:])
                vals[col] = float(top_sum / total)
            if vals:
                concs[dt] = pd.Series(vals)
        if not concs:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(concs).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 165. intraday_price_volume_divergence — 量价背离
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_price_volume_divergence_20d", category="intraday_advanced")
class IntradayPriceVolumeDivergence20d(Factor):
    """量价背离因子.

    价格创新高但成交量小于前一次创新高时的量 (顶背离) 的分钟占比.
    顶背离 → 涨势缺乏量能支撑 → 虚假上涨 → 负向.
    方向: 负向.
    """
    name = "intraday_price_volume_divergence_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "量价背离 (新高但量缩占比, 虚涨=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"high", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, volume = panel["high"], panel["volume"]
        day = high.index.normalize()
        divergences: dict = {}
        for dt in sorted(set(day)):
            grp_h = high.loc[day == dt]
            grp_v = volume.loc[day == dt]
            if len(grp_h) < 30:
                continue
            vals = {}
            for col in grp_h.columns:
                h = grp_h[col].dropna()
                v = grp_v[col].dropna()
                common = h.index.intersection(v.index)
                if len(common) < 30:
                    continue
                h_c = h.loc[common].values
                v_c = v.loc[common].values
                new_high_vols = []
                diverge_cnt = 0
                for i in range(1, len(h_c)):
                    if h_c[i] > h_c[:i].max():
                        # 创新高: 比较量是否小于此前新高时的量
                        if new_high_vols and v_c[i] < new_high_vols[-1]:
                            diverge_cnt += 1
                        new_high_vols.append(v_c[i])
                vals[col] = float(diverge_cnt / max(1, len(new_high_vols)))
            if vals:
                divergences[dt] = pd.Series(vals)
        if not divergences:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(divergences).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 166. intraday_imbalance_acceleration — 订单流不平衡加速度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_imbalance_acceleration_20d", category="intraday_advanced")
class IntradayImbalanceAcceleration20d(Factor):
    """订单流不平衡加速度因子.

    分钟买卖不平衡的二阶差分均值 (不平衡的变化趋势).
    与 #146 波动互补: #146看不平衡的波动, 这里看其加速度方向.
    加速度正 → 不平衡持续扩大 → 情绪化加剧 → 负向.
    方向: 负向.
    """
    name = "intraday_imbalance_acceleration_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "订单流不平衡加速度 (二阶差分, 扩大=情绪化=负向)"
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
        accelerations: dict = {}
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
                imbalance = np.sign(r.loc[common].values) * v.loc[common].values
                if len(imbalance) < 5:
                    continue
                second_diff = np.diff(imbalance, n=2)
                vals[col] = float(np.mean(second_diff))
            if vals:
                accelerations[dt] = pd.Series(vals)
        if not accelerations:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(accelerations).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 167. intraday_ret_distribution_peak — 收益分布中心度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_ret_distribution_peak_20d", category="intraday_advanced")
class IntradayRetDistributionPeak20d(Factor):
    """收益分布中心度因子.

    p50|ret| / p25|ret| — 收益分布在小波动区的聚集程度 (峰部).
    高中心度 → 大量微小波动聚集 → 稳定 → 正向.
    方向: 正向.
    """
    name = "intraday_ret_distribution_peak_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "收益分布中心度 (p50|ret|/p25|ret|, 稳定=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        peaks: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 30:
                    continue
                ar = r.abs().values
                p25, p50 = np.percentile(ar, [25, 50])
                vals[col] = float(p50 / p25) if p25 > 1e-12 else 1.0
            if vals:
                peaks[dt] = pd.Series(vals)
        if not peaks:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(peaks).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 168. intraday_trend_follow_score — 趋势跟随得分
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_trend_follow_score_20d", category="intraday_advanced")
class IntradayTrendFollowScore20d(Factor):
    """趋势跟随得分因子.

    与全天主导方向同向的分钟收益之和 / 所有分钟收益绝对值之和.
    与 #102 计数互补: 这里按收益加权 (顺趋势的"量价贡献").
    高得分 → 收益集中在顺趋势方向 → 有效趋势 → 正向.
    方向: 正向.
    """
    name = "intraday_trend_follow_score_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "趋势跟随得分 (顺趋势收益贡献占比)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        scores: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 20:
                    continue
                day_dir = np.sign(r.sum())
                total_abs = r.abs().sum()
                if abs(day_dir) < 1e-12 or total_abs < 1e-12:
                    vals[col] = 0.0
                    continue
                aligned = (r * day_dir).clip(lower=0).sum()
                vals[col] = float(aligned / total_abs)
            if vals:
                scores[dt] = pd.Series(vals)
        if not scores:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(scores).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 169. intraday_momentum_consistency — 多尺度动量一致性
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_momentum_consistency_20d", category="intraday_advanced")
class IntradayMomentumConsistency20d(Factor):
    """多尺度动量一致性因子.

    5/10/30/60分钟四种尺度的日内动量方向一致的数量.
    多尺度同向 → 趋势在不同时间框架共振 → 强信号 → 正向.
    方向: 正向.
    """
    name = "intraday_momentum_consistency_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "多尺度动量一致 (5/10/30/60分动量同向数)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        consistencies: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            n = len(grp)
            if n < 60:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < 60:
                    continue
                c_end = c.iloc[-1]
                moms = []
                for lag in (5, 10, 30, 60):
                    if len(c) > lag:
                        mom = c_end / c.iloc[-lag - 1] - 1.0
                        moms.append(np.sign(mom))
                if not moms:
                    vals[col] = 0.0
                    continue
                vals[col] = float(abs(sum(moms)))  # 一致数越多|和|越大
            if vals:
                consistencies[dt] = pd.Series(vals)
        if not consistencies:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(consistencies).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 170. intraday_session_symmetry — 早晚盘路径对称度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_session_symmetry_20d", category="intraday_advanced")
class IntradaySessionSymmetry20d(Factor):
    """早晚盘路径对称度因子.

    前半段收益路径与后半段路径的 Pearson 相关 (镜像对齐).
    高对称 → 路径结构有序 → 可预测 → 正向.
    方向: 正向.
    """
    name = "intraday_session_symmetry_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "早晚盘路径对称 (前后半段路径相关)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        symmetries: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            n = len(grp)
            if n < 40:
                continue
            mid = n // 2
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < 40:
                    continue
                first = c.iloc[:mid].values
                second = c.iloc[mid:].values
                k = min(len(first), len(second))
                if k < 10:
                    vals[col] = 0.0
                    continue
                # 将后半段反转对齐 (镜像对称)
                f = first[:k] / first[0] - 1.0
                s = second[:k] / second[0] - 1.0
                if f.std(ddof=0) < 1e-12 or s.std(ddof=0) < 1e-12:
                    vals[col] = 0.0
                else:
                    corr_val = float(np.corrcoef(f, s)[0, 1])
                    vals[col] = corr_val if not np.isnan(corr_val) else 0.0
            if vals:
                symmetries[dt] = pd.Series(vals)
        if not symmetries:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(symmetries).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 171. intraday_vwap_reversion_speed — VWAP 回归速度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_vwap_reversion_speed_20d", category="intraday_advanced")
class IntradayVwapReversionSpeed20d(Factor):
    """VWAP 回归速度因子.

    价格偏离 VWAP 后向其回归的速度: 相邻分钟 |close - vwap| 的平均衰减率.
    与 #30 偏离幅度/#147 收盘位置互补 (这里是回归的动态过程).
    快回归 → 价格锚定公允价 → 市场有序 → 正向.
    方向: 正向.
    """
    name = "intraday_vwap_reversion_speed_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "VWAP回归速度 (偏离后回归速率, 快=有序=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, volume = panel["close"], panel["volume"]
        day = close.index.normalize()
        speeds: dict = {}
        for dt in sorted(set(day)):
            grp_c = close.loc[day == dt]
            grp_v = volume.loc[day == dt]
            if len(grp_c) < 30:
                continue
            vals = {}
            for col in grp_c.columns:
                c = grp_c[col].dropna()
                v = grp_v[col].dropna()
                common = c.index.intersection(v.index)
                if len(common) < 30:
                    continue
                c_c = c.loc[common]
                v_c = v.loc[common]
                vwap = float((c_c * v_c).sum() / v_c.sum()) if v_c.sum() > 1e-12 else c_c.mean()
                dist = (c_c - vwap).abs()
                # 距离序列的自衰减: 相邻距离的均值变化率
                d_t = dist.values[1:]
                d_tm1 = dist.values[:-1]
                mask = d_tm1 > 1e-12
                if mask.sum() < 10:
                    vals[col] = 0.0
                    continue
                decay = float(np.mean(1.0 - d_t[mask] / d_tm1[mask]))
                vals[col] = decay  # 正=距离在缩小=回归快
            if vals:
                speeds[dt] = pd.Series(vals)
        if not speeds:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(speeds).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 172. intraday_vwap_crossings — VWAP 穿越次数
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_vwap_crossings_20d", category="intraday_advanced")
class IntradayVwapCrossings20d(Factor):
    """VWAP 穿越次数因子.

    价格上穿/下穿 VWAP 的次数 (VWAP 附近的震荡频率).
    与 #126 上方时间占比互补: 那里看停留, 这里看穿越.
    频繁穿越 → 多空反复拉锯 → 无方向 → 负向.
    方向: 负向.
    """
    name = "intraday_vwap_crossings_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "VWAP穿越次数 (多空拉锯=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, volume = panel["close"], panel["volume"]
        day = close.index.normalize()
        crossings: dict = {}
        for dt in sorted(set(day)):
            grp_c = close.loc[day == dt]
            grp_v = volume.loc[day == dt]
            if len(grp_c) < 30:
                continue
            vals = {}
            for col in grp_c.columns:
                c = grp_c[col].dropna()
                v = grp_v[col].dropna()
                common = c.index.intersection(v.index)
                if len(common) < 30:
                    continue
                c_c = c.loc[common]
                v_c = v.loc[common]
                vwap = float((c_c * v_c).sum() / v_c.sum()) if v_c.sum() > 1e-12 else c_c.mean()
                above = (c_c > vwap).astype(int).values
                crosses = int(np.sum(np.diff(above) != 0))
                vals[col] = -float(crosses)
            if vals:
                crossings[dt] = pd.Series(vals)
        if not crossings:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(crossings).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 173. intraday_run_length_median — 同向run长度中位数
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_run_length_median_20d", category="intraday_advanced")
class IntradayRunLengthMedian20d(Factor):
    """同向run长度中位数因子.

    同向连续涨/跌 run 长度的中位数 (分布形状, 与 #41 最大run/#151 总时长互补).
    中位run长 → 趋势惯性大 → 正向.
    方向: 正向.
    """
    name = "intraday_run_length_median_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "同向run中位长度 (趋势惯性=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        medians: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 20:
                    continue
                runs = []
                cur = 0
                for x in r:
                    if abs(x) > 1e-12:
                        cur += 1
                    else:
                        if cur > 0:
                            runs.append(cur)
                        cur = 0
                if cur > 0:
                    runs.append(cur)
                vals[col] = float(np.median(runs)) if runs else 1.0
            if vals:
                medians[dt] = pd.Series(vals)
        if not medians:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(medians).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 174. intraday_run_length_skew — 同向run长度偏度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_run_length_skew_20d", category="intraday_advanced")
class IntradayRunLengthSkew20d(Factor):
    """同向run长度偏度因子.

    run 长度分布的偏度 (大多数短run + 少数超长run = 右偏).
    高右偏 → 趋势靠偶发长run驱动 → 不稳定 → 负向.
    方向: 负向.
    """
    name = "intraday_run_length_skew_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "同向run长度偏度 (右偏=偶发长run=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        skews: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 30:
                    continue
                runs = []
                cur = 0
                for x in r:
                    if abs(x) > 1e-12:
                        cur += 1
                    else:
                        if cur > 0:
                            runs.append(cur)
                        cur = 0
                if cur > 0:
                    runs.append(cur)
                if len(runs) >= 5:
                    vals[col] = -float(pd.Series(runs).skew())  # 右偏=负向
                else:
                    vals[col] = 0.0
            if vals:
                skews[dt] = pd.Series(vals)
        if not skews:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(skews).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 175. intraday_volume_half_life — 成交量记忆半衰期
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_volume_half_life_20d", category="intraday_advanced")
class IntradayVolumeHalfLife20d(Factor):
    """成交量记忆半衰期因子.

    分钟成交量自相关从1衰减到0.5的滞后数 (量记忆时长, 与 #161 波动半衰期互补).
    量记忆长 → 放量惯性持续 → 不稳定 → 负向.
    方向: 负向.
    """
    name = "intraday_volume_half_life_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "量记忆半衰期 (量自相关衰减滞后数, 长=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        volume = panel["volume"]
        day = volume.index.normalize()
        half_lives: dict = {}
        for dt in sorted(set(day)):
            grp = volume.loc[day == dt]
            if len(grp) < 60:
                continue
            vals = {}
            for col in grp.columns:
                v = grp[col].dropna().values
                if len(v) < 60:
                    continue
                max_lag = min(20, len(v) // 4)
                hl = max_lag
                for lag in range(1, max_lag + 1):
                    x, y = v[:-lag], v[lag:]
                    if x.std(ddof=0) < 1e-12 or y.std(ddof=0) < 1e-12:
                        continue
                    rho = float(np.corrcoef(x, y)[0, 1])
                    if rho <= 0.5:
                        hl = lag
                        break
                vals[col] = -float(hl)
            if vals:
                half_lives[dt] = pd.Series(vals)
        if not half_lives:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(half_lives).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 176. intraday_vol_regime_switches — 波动状态切换次数
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_vol_regime_switches_20d", category="intraday_advanced")
class IntradayVolRegimeSwitches20d(Factor):
    """波动状态切换次数因子.

    日内分钟波动(滚动窗口|ret|)在高/低两个状态间切换的次数 (regime 动态).
    与 #116 跨日突破/#84 变异系数互补: 这里是状态切换频率.
    切换频繁 → 波动环境不稳 → 负向.
    方向: 负向.
    """
    name = "intraday_vol_regime_switches_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "波动状态切换次数 (高/低波动切换, 频繁=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        switches: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 60:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 60:
                    continue
                ar = pd.Series(np.abs(r))
                win = max(5, min(15, len(ar) // 6))
                vol_series = ar.rolling(win).mean().dropna().values
                if len(vol_series) < 20:
                    continue
                med = np.median(vol_series)
                state = vol_series > med
                sw = int(np.sum(np.diff(state.astype(int)) != 0))
                vals[col] = -float(sw)
            if vals:
                switches[dt] = pd.Series(vals)
        if not switches:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(switches).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 177. intraday_vol_regime_duration_ratio — 高波动状态时长占比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_vol_regime_duration_ratio_20d", category="intraday_advanced")
class IntradayVolRegimeDurationRatio20d(Factor):
    """高波动状态时长占比因子.

    分钟波动处于高状态(>中位)的时长占比 (regime 时长, 与 #176 切换频率互补).
    高波动主导 → 风险环境持续 → 负向.
    方向: 负向.
    """
    name = "intraday_vol_regime_duration_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "高波动状态时长占比 (风险持续=负向)"
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
            if len(grp) < 60:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 60:
                    continue
                ar = pd.Series(np.abs(r))
                win = max(5, min(15, len(ar) // 6))
                vol_series = ar.rolling(win).mean().dropna().values
                if len(vol_series) < 20:
                    continue
                med = np.median(vol_series)
                vals[col] = -float((vol_series > med).mean())  # 高波动多=负向
            if vals:
                durations[dt] = pd.Series(vals)
        if not durations:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(durations).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 178. intraday_skew_stability — 分段偏度一致性
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_skew_stability_20d", category="intraday_advanced")
class IntradaySkewStability20d(Factor):
    """分段偏度一致性因子.

    日内4等分段各自的收益偏度符号一致性 (与 #4 整体偏度互补: 这里看分段间的稳定性).
    全段同向偏 → 偏度结构稳定 → 有序 → 正向.
    方向: 正向.
    """
    name = "intraday_skew_stability_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "分段偏度一致 (4段偏度同向数, 稳定=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        stabilities: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            n = len(grp)
            if n < 60:
                continue
            q = n // 4
            if q < 8:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 60:
                    continue
                segs = [r.iloc[i * q:(i + 1) * q] for i in range(4)]
                signs = []
                for s in segs:
                    if len(s) < 5 or s.std(ddof=0) < 1e-12:
                        continue
                    sk = float(pd.Series(s).skew())
                    if abs(sk) > 0.05:  # 显著的偏度符号
                        signs.append(np.sign(sk))
                if not signs:
                    vals[col] = 0.0
                    continue
                dom = np.sign(sum(signs))
                consistent = sum(1 for sg in signs if sg == dom)
                vals[col] = float(consistent / len(signs))
            if vals:
                stabilities[dt] = pd.Series(vals)
        if not stabilities:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(stabilities).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 179. intraday_quantile_skew — 分位数偏度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_quantile_skew_20d", category="intraday_advanced")
class IntradayQuantileSkew20d(Factor):
    """分位数偏度因子.

    Q3+Q1-2·Q2 / (Q3-Q1) — 稳健的分位数偏度 (互补于 #4 矩偏度).
    稳健右偏 → 上行占优 → 正向.
    方向: 正向.
    """
    name = "intraday_quantile_skew_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "分位数偏度 ((Q3+Q1-2Q2)/(Q3-Q1), 稳健偏度)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        qskews: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 30:
                    continue
                q1, q2, q3 = np.percentile(r, [25, 50, 75])
                denom = q3 - q1
                vals[col] = float((q3 + q1 - 2.0 * q2) / denom) if denom > 1e-12 else 0.0
            if vals:
                qskews[dt] = pd.Series(vals)
        if not qskews:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(qskews).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 180. intraday_temporal_consistency — 时间聚合一致性
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_temporal_consistency_20d", category="intraday_advanced")
class IntradayTemporalConsistency20d(Factor):
    """时间聚合一致性因子.

    5分钟收益与1分钟收益的关系稳定性: std(5min_ret) / sqrt(5·std(1min_ret)).
    比值接近1 → 收益可加性成立 → 无微观结构噪声干扰 → 正向.
    偏离1 → 噪声主导 (买卖反弹) → 负向. 输出 -|ratio-1|.
    方向: 正向.
    """
    name = "intraday_temporal_consistency_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "时间聚合一致 (-|σ5m/(√5·σ1m)-1|, 噪声=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        ret_1m = close.pct_change()
        day = ret_1m.index.normalize()
        consistencies: dict = {}
        for dt in sorted(set(day)):
            grp_c = close.loc[day == dt]
            grp_r = ret_1m.loc[day == dt]
            if len(grp_c) < 60:
                continue
            vals = {}
            for col in grp_c.columns:
                c = grp_c[col].dropna()
                r = grp_r[col].dropna()
                if len(c) < 60 or len(r) < 60:
                    continue
                sigma_1m = r.std(ddof=0)
                if sigma_1m < 1e-12:
                    vals[col] = 0.0
                    continue
                # 5分钟非重叠收益
                c_v = c.values
                n5 = len(c_v) // 5
                if n5 < 10:
                    vals[col] = 0.0
                    continue
                ret5 = []
                for i in range(n5):
                    seg = c_v[i * 5:(i + 1) * 5]
                    if seg[0] > 1e-12:
                        ret5.append(seg[-1] / seg[0] - 1.0)
                if len(ret5) < 5:
                    vals[col] = 0.0
                    continue
                sigma_5m = np.std(ret5, ddof=0)
                ratio = sigma_5m / (np.sqrt(5.0) * sigma_1m)
                vals[col] = -abs(ratio - 1.0)
            if vals:
                consistencies[dt] = pd.Series(vals)
        if not consistencies:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(consistencies).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 181. intraday_morning_star_freq — 早晨之星频率
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_morning_star_freq_20d", category="intraday_advanced")
class IntradayMorningStarFreq20d(Factor):
    """早晨之星形态频率因子 (双bar组合形态).

    阴线 → 小实体星线 → 阳线 的三bar组合 (探底反转信号).
    与 #74-79 单bar形态互补: 这里是多bar组合形态.
    高频率 → 反复探底反转 → 承接活跃 → 正向.
    方向: 正向.
    """
    name = "intraday_morning_star_freq_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "早晨之星频率 (阴-星-阳三bar反转组合)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"open", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        open_px, close = panel["open"], panel["close"]
        day = close.index.normalize()
        freqs: dict = {}
        for dt in sorted(set(day)):
            grp_o = open_px.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 30:
                continue
            vals = {}
            for col in grp_c.columns:
                o = grp_o[col].dropna()
                c = grp_c[col].dropna()
                common = o.index.intersection(c.index)
                if len(common) < 30:
                    continue
                o_c, c_c = (x.loc[common].values for x in (o, c))
                n = len(o_c)
                cnt = 0
                for i in range(1, n - 1):
                    rng1 = max(abs(c_c[i - 1] - o_c[i - 1]), 1e-12)
                    body1 = (c_c[i - 1] - o_c[i - 1]) / rng1  # <0 阴线
                    body2 = abs(c_c[i] - o_c[i]) / max(abs(c_c[i] - o_c[i]) + abs(c_c[i - 1] - o_c[i - 1]) + 1e-12, 1e-12)
                    rng3 = max(abs(c_c[i + 1] - o_c[i + 1]), 1e-12)
                    body3 = (c_c[i + 1] - o_c[i + 1]) / rng3  # >0 阳线
                    # 阴线 + 小实体 + 阳线
                    if body1 < -0.5 and body2 < 0.3 and body3 > 0.5:
                        cnt += 1
                vals[col] = float(cnt / n)
            if vals:
                freqs[dt] = pd.Series(vals)
        if not freqs:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(freqs).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 182. intraday_engulfing_freq — 吞没形态频率
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_engulfing_freq_20d", category="intraday_advanced")
class IntradayEngulfingFreq20d(Factor):
    """吞没形态频率因子 (双bar反转组合).

    阳线实体完全吞没前一根阴线实体 (看涨吞没) 或反向 (看跌吞没).
    频繁吞没 → 多空快速反转 → 方向不稳定 → 负向.
    方向: 负向.
    """
    name = "intraday_engulfing_freq_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "吞没形态频率 (反转组合频繁=方向不稳=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"open", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        open_px, close = panel["open"], panel["close"]
        day = close.index.normalize()
        freqs: dict = {}
        for dt in sorted(set(day)):
            grp_o = open_px.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 30:
                continue
            vals = {}
            for col in grp_c.columns:
                o = grp_o[col].dropna()
                c = grp_c[col].dropna()
                common = o.index.intersection(c.index)
                if len(common) < 30:
                    continue
                o_c, c_c = (x.loc[common].values for x in (o, c))
                n = len(o_c)
                cnt = 0
                for i in range(1, n):
                    prev_body = c_c[i - 1] - o_c[i - 1]
                    cur_body = c_c[i] - o_c[i]
                    if prev_body == 0 or cur_body == 0:
                        continue
                    # 看涨吞没: 前阴后阳, 阳实体包住阴实体
                    if prev_body < 0 < cur_body and o_c[i] <= c_c[i - 1] and c_c[i] >= o_c[i - 1]:
                        cnt += 1
                    # 看跌吞没: 前阳后阴
                    elif prev_body > 0 > cur_body and o_c[i] >= c_c[i - 1] and c_c[i] <= o_c[i - 1]:
                        cnt += 1
                vals[col] = -float(cnt / n)
            if vals:
                freqs[dt] = pd.Series(vals)
        if not freqs:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(freqs).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 183. intraday_lz_complexity — LZ 复杂度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_lz_complexity_20d", category="intraday_advanced")
class IntradayLzComplexity20d(Factor):
    """LZ 复杂度因子.

    将收益符号序列(涨/跌/平)用 LZ78 算法压缩, 复杂度=模式数/长度 (归一化).
    与 #72 排列熵互补 (不同算法度量序列可预测性).
    高复杂度 → 序列随机 → 不可预测 → 负向.
    方向: 负向.
    """
    name = "intraday_lz_complexity_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "LZ复杂度 (符号序列模式数/长度, 高=随机=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    @staticmethod
    def _lz76(seq):
        """LZ76 复杂度 (模式计数)."""
        n = len(seq)
        if n == 0:
            return 0.0
        i, c = 0, 1
        while i < n - 1:
            j = 0
            while j < i + 1 and i + j < n:
                k = 0
                while k < i + 1 and i + j + k < n and seq[k] == seq[i + j + k]:
                    k += 1
                j = max(j + 1, k)
            c += 1
            i += j
        return float(c)

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        complexities: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 40:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 40:
                    continue
                seq = [1 if x > 0 else (0 if x < 0 else 2) for x in r]
                c = self._lz76(seq)
                vals[col] = -float(c / max(1, len(seq)))
            if vals:
                complexities[dt] = pd.Series(vals)
        if not complexities:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(complexities).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 184. intraday_or_retention — 开盘区间维持度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_or_retention_20d", category="intraday_advanced")
class IntradayOrRetention20d(Factor):
    """开盘区间维持度因子.

    收盘仍在开盘区间 [OR_low, OR_high] 内时记1否则0 (与 #35 突破互补: #35看突破, 这里看未突破的维持).
    收盘未突破开盘区间 → 全天被开盘框定 → 缺乏方向性 → 负向.
    方向: 负向.
    """
    name = "intraday_or_retention_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "开盘区间维持 (收盘未突破开盘区间=缺乏方向=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"high", "low", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, low, close = panel["high"], panel["low"], panel["close"]
        day = close.index.normalize()
        retentions: dict = {}
        for dt in sorted(set(day)):
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            grp_c = close.loc[day == dt]
            n = len(grp_c)
            if n < 30:
                continue
            n_or = max(10, n // 4)
            vals = {}
            for col in grp_c.columns:
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                c = grp_c[col].dropna()
                if len(h) < n_or or len(c) < 20:
                    continue
                or_high = h.iloc[:n_or].max()
                or_low = l.iloc[:n_or].min()
                c_end = c.iloc[-1]
                retained = 1.0 if (or_low <= c_end <= or_high) else 0.0
                vals[col] = -retained
            if vals:
                retentions[dt] = pd.Series(vals)
        if not retentions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(retentions).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 185. intraday_vol_volume_regime — 量价状态联合占比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_vol_volume_regime_20d", category="intraday_advanced")
class IntradayVolVolumeRegime20d(Factor):
    """量价状态联合占比因子.

    高波动且放量的分钟占比 (风险与活跃的联合状态).
    高频风险+活跃 → 情绪化交易主导 → 负向.
    方向: 负向.
    """
    name = "intraday_vol_volume_regime_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "量价联合状态 (高波动+放量占比, 情绪化=负向)"
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
        regimes: dict = {}
        for dt in sorted(set(day)):
            grp_r = ret_1m.loc[day == dt]
            grp_v = volume.loc[day == dt]
            if len(grp_r) < 30:
                continue
            vals = {}
            for col in grp_r.columns:
                r = grp_r[col].dropna()
                v = grp_v[col].dropna()
                common = r.index.intersection(v.index)
                if len(common) < 30:
                    continue
                r_c = r.loc[common]
                v_c = v.loc[common]
                high_vol = r_c > r_c.median()
                high_vol_vol = v_c > v_c.median()
                joint = (high_vol & high_vol_vol).sum()
                vals[col] = -float(joint / len(r_c))
            if vals:
                regimes[dt] = pd.Series(vals)
        if not regimes:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(regimes).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 186. intraday_body_consistency — 实体方向一致性
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_body_consistency_20d", category="intraday_advanced")
class IntradayBodyConsistency20d(Factor):
    """实体方向一致性因子.

    相邻分钟 K 线实体方向 (收>开=阳) 的序列相关 (与 #74 实体占比均值互补).
    实体方向持续 → 多头/空头排列整齐 → 正向.
    方向: 正向.
    """
    name = "intraday_body_consistency_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "实体方向一致 (相邻K线实体方向相关)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"open", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        open_px, close = panel["open"], panel["close"]
        day = close.index.normalize()
        consistencies: dict = {}
        for dt in sorted(set(day)):
            grp_o = open_px.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 30:
                continue
            vals = {}
            for col in grp_c.columns:
                o = grp_o[col].dropna()
                c = grp_c[col].dropna()
                common = o.index.intersection(c.index)
                if len(common) < 30:
                    continue
                body = np.sign(c.loc[common].values - o.loc[common].values)
                body = body[body != 0]
                if len(body) < 10:
                    vals[col] = 0.0
                    continue
                s_t, s_tm1 = body[1:], body[:-1]
                corr_val = float(np.corrcoef(s_t, s_tm1)[0, 1]) if s_t.std() > 1e-12 and s_tm1.std() > 1e-12 else 0.0
                vals[col] = corr_val if not np.isnan(corr_val) else 0.0
            if vals:
                consistencies[dt] = pd.Series(vals)
        if not consistencies:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(consistencies).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 187. intraday_tail_cluster — 极端收益时间聚集
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_tail_cluster_20d", category="intraday_advanced")
class IntradayTailCluster20d(Factor):
    """极端收益时间聚集因子.

    |ret|>2σ 的极端分钟之间的平均间隔 (时间聚集度, 与 #118 频率/#103 冲击互补).
    间隔短 → 极端事件扎堆 → 风险集中 → 负向. 输出 -平均间隔.
    方向: 负向.
    """
    name = "intraday_tail_cluster_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "极端收益聚集 (极端分钟间隔, 扎堆=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        clusters: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 30:
                    continue
                sigma = r.std(ddof=0)
                if sigma < 1e-12:
                    vals[col] = 0.0
                    continue
                idx = np.where(np.abs(r) > 2.0 * sigma)[0]
                if len(idx) < 2:
                    vals[col] = 0.0
                    continue
                gaps = np.diff(idx)
                vals[col] = -float(np.mean(gaps))
            if vals:
                clusters[dt] = pd.Series(vals)
        if not clusters:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(clusters).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 188. intraday_extreme_timing — 极端收益时间位置
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_extreme_timing_20d", category="intraday_advanced")
class IntradayExtremeTiming20d(Factor):
    """极端收益时间位置因子.

    |ret| 最大的分钟出现的时间位置 (与 #98/#99 高低点时间互补: 这里用收益幅度).
    极端波动偏尾盘 → 尾盘情绪化异动 → 负向. 输出 -时间位置.
    方向: 正向 (早盘极端=已释放=正向).
    """
    name = "intraday_extreme_timing_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "极端收益时间位置 (-最大|ret|分钟时间, 早=已释放=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        timings: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 20:
                    continue
                idx = int(np.argmax(np.abs(r)))
                vals[col] = -float(idx / max(1, len(r) - 1))
            if vals:
                timings[dt] = pd.Series(vals)
        if not timings:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(timings).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 189. intraday_volume_pareto_tail — 成交量 Pareto 尾部指数
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_volume_pareto_tail_20d", category="intraday_advanced")
class IntradayVolumeParetoTail20d(Factor):
    """成交量 Pareto 尾部指数因子.

    Hill 估计量: 尾部指数的倒数 α = 1/(mean(log(v/top_threshold))).
    高 α → 薄尾 → 量分布温和 → 正向. 低 α → 厚尾 → 大单依赖 → 负向.
    方向: 正向.
    """
    name = "intraday_volume_pareto_tail_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "量Pareto尾部指数 (Hill估计, 厚尾=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        volume = panel["volume"]
        day = volume.index.normalize()
        alphas: dict = {}
        for dt in sorted(set(day)):
            grp = volume.loc[day == dt]
            if len(grp) < 60:
                continue
            vals = {}
            for col in grp.columns:
                v = grp[col].dropna().values
                if len(v) < 60:
                    continue
                v_pos = v[v > 0]
                if len(v_pos) < 20:
                    continue
                k = max(3, len(v_pos) // 10)
                sorted_v = np.sort(v_pos)[::-1]
                top = sorted_v[:k]
                if top[-1] < 1e-12:
                    vals[col] = 0.0
                    continue
                hill = float(np.mean(np.log(top / top[-1])))
                vals[col] = float(1.0 / hill) if hill > 1e-12 else 10.0
            if vals:
                alphas[dt] = pd.Series(vals)
        if not alphas:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(alphas).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 190. intraday_close_slope_r2 — 尾盘趋势清晰度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_close_slope_r2_20d", category="intraday_advanced")
class IntradayCloseSlopeR220d(Factor):
    """尾盘趋势清晰度因子.

    最后30分钟价格路径线性回归的 R² (与 #114 全天R²/#109 尾盘漂移互补).
    尾盘 R² 高 → 收盘段单边清晰 → 方向确认 → 正向.
    方向: 正向.
    """
    name = "intraday_close_slope_r2_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "尾盘趋势清晰度 (最后30分路径R², 单边=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        r2s: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            n = len(grp)
            if n < 40:
                continue
            n_tail = max(10, min(30, n // 4))
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < 40:
                    continue
                tail = c.iloc[-n_tail:]
                y = tail.values
                t = np.arange(len(y))
                slope, intercept = np.polyfit(t, y, 1)
                fitted = slope * t + intercept
                ss_res = np.sum((y - fitted) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
                vals[col] = float(max(0.0, r2))
            if vals:
                r2s[dt] = pd.Series(vals)
        if not r2s:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(r2s).T
        daily.index = pd.DatetimeIndex(daily.index)
        return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════
# 以下为自创因子补充批次 (由 microstructure_batch.py + effective_variants.py 合并而来)
# ═══════════════════════════════════════════════════════════════════════════

def _finalize(daily: pd.DataFrame, dates, universe, window: int = 20) -> pd.DataFrame:
    """统一的输出管线: 对齐日期索引 → 滚动平滑 → shift(1) → 对齐列."""
    if daily.empty:
        return pd.DataFrame(np.nan, index=dates, columns=universe)
    daily.index = pd.DatetimeIndex(daily.index)
    return (
        daily.rolling(window, min_periods=5).mean()
        .reindex(dates).shift(1).reindex(columns=universe)
    )


def _daily_agg(panel_key: str, panel, day, agg):
    """按日聚合面板中某个字段."""
    frame = panel[panel_key]
    return pd.DataFrame({
        dt: agg(frame.loc[day == dt])
        for dt in sorted(set(day)) if len(frame.loc[day == dt]) > 10
    }).T


# ═══════════════════════════════════════════════════════════════════════════
# K1. realized_kurtosis — 已实现峰度
#     = 分钟收益四阶矩 / (二阶矩^2), 刻画收益分布的厚尾程度.
#     高峰度 → 极端行情频繁 → 风险溢价.
# ═══════════════════════════════════════════════════════════════════════════
@register_factor("realized_kurtosis_20d", category="intraday_advanced")
class RealizedKurtosis20d(Factor):
    name = "realized_kurtosis_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "已实现峰度 (分钟收益四阶矩/二阶矩^2)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret = panel["close"].pct_change()
        day = ret.index.normalize()
        daily = {}
        for dt in sorted(set(day)):
            r = ret.loc[day == dt]
            r = r.dropna(how="all")
            if r.shape[0] < 20:
                continue
            m2 = (r ** 2).mean()
            m4 = (r ** 4).mean()
            kurt = m4 / (m2.replace(0, np.nan) ** 2)
            daily[dt] = kurt
        return _finalize(pd.DataFrame(daily).T, dates, universe)


# ═══════════════════════════════════════════════════════════════════════════
# K2. overnight_intraday_vol — 隔夜/日内波动比
#     = 隔夜跳空波动 / 日内实现波动. 高 → 隔夜信息主导.
# ⚠ 跨日因子: 依赖相邻交易日收盘 (daily_close.pct_change), 需≥2个交易日历史; 首日为NaN
# ═══════════════════════════════════════════════════════════════════════════
@register_factor("overnight_intraday_vol_20d", category="intraday_advanced")
class OvernightIntradayVol20d(Factor):
    name = "overnight_intraday_vol_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "隔夜/日内波动比"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        daily_close = close.groupby(close.index.normalize()).last()
        overnight = daily_close.pct_change().abs()  # 相邻日收盘的隔夜变动代理
        intraday_vol = daily_close.pct_change().rolling(5, min_periods=3).std(ddof=0)
        ratio = overnight / intraday_vol.replace(0, np.nan)
        return _finalize(ratio, dates, universe)


# ═══════════════════════════════════════════════════════════════════════════
# K3. jump_persistence — 跳跃持续性
#     = 相邻跳跃方向一致的比例. 持续同向跳跃 → 趋势信息驱动.
# ═══════════════════════════════════════════════════════════════════════════
@register_factor("jump_persistence_20d", category="intraday_advanced")
class JumpPersistence20d(Factor):
    name = "jump_persistence_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "跳跃持续性 (相邻跳跃同向比例)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret = panel["close"].pct_change()
        day = ret.index.normalize()
        daily = {}
        for dt in sorted(set(day)):
            r = ret.loc[day == dt].dropna(how="all")
            if r.shape[0] < 30:
                continue
            jump_sign = np.sign(r)  # -1/0/+1
            vals = {}
            for col in r.columns:
                s = jump_sign[col].replace(0, np.nan).dropna()
                if len(s) < 10:
                    continue
                same = (s == s.shift(1)).sum()
                vals[col] = same / (len(s) - 1)
            if vals:
                daily[dt] = pd.Series(vals)
        return _finalize(pd.DataFrame(daily).T, dates, universe)


# ═══════════════════════════════════════════════════════════════════════════
# K4. intraday_amihud — 日内 Amihud 非流动性
#     = |分钟收益| / 分钟成交额, 日内平均. 高 → 冲击成本大.
# ═══════════════════════════════════════════════════════════════════════════
@register_factor("intraday_amihud_20d", category="intraday_advanced")
class IntradayAmihud20d(Factor):
    name = "intraday_amihud_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "日内Amihud非流动性 (|ret|/amount 日内均值)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel or "amount" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret = panel["close"].pct_change().abs()
        amt = panel["amount"].replace(0, np.nan)
        amihud = ret / amt
        day = amihud.index.normalize()
        daily = pd.DataFrame({
            dt: amihud.loc[day == dt].mean()
            for dt in sorted(set(day)) if len(amihud.loc[day == dt]) > 10
        }).T
        return _finalize(daily, dates, universe)


# ═══════════════════════════════════════════════════════════════════════════
# K5. liquidity_elasticity — 流动性弹性
#     = 冲击后价格恢复比例: |冲击后5分钟累计收益| / |冲击时收益|.
#     高 → 市场吸收冲击快 → 流动性好.
# ═══════════════════════════════════════════════════════════════════════════
@register_factor("liquidity_elasticity_20d", category="intraday_advanced")
class LiquidityElasticity20d(Factor):
    name = "liquidity_elasticity_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "流动性弹性 (冲击后价格恢复速度)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel or "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        vol = panel["volume"]
        ret = close.pct_change()
        day = close.index.normalize()
        daily = {}
        for dt in sorted(set(day)):
            c = close.loc[day == dt]
            v = vol.loc[vol.index.normalize() == dt]
            common = c.columns.intersection(v.columns)
            vals = {}
            for col in common:
                cc, vv = c[col], v[col]
                idx = cc.dropna().index.intersection(vv.dropna().index)
                if len(idx) < 30:
                    continue
                cc, vv = cc.loc[idx], vv.loc[idx]
                rr = cc.pct_change()
                vol_spike = vv > (vv.mean() + 2 * vv.std(ddof=0))
                if not vol_spike.any():
                    continue
                spike_idx = vol_spike[vol_spike].index
                # 冲击时收益绝对值
                shock_abs = rr.loc[spike_idx].abs()
                if shock_abs.empty:
                    continue
                # 冲击后5分钟累计收益
                recover = []
                for si in spike_idx:
                    pos = rr.index.get_loc(si)
                    if pos + 5 < len(rr):
                        recover.append(abs(rr.iloc[pos + 1:pos + 6].sum()))
                if not recover:
                    continue
                vals[col] = float(np.mean(recover) / max(shock_abs.mean(), 1e-9))
            if vals:
                daily[dt] = pd.Series(vals)
        return _finalize(pd.DataFrame(daily).T, dates, universe)


# ═══════════════════════════════════════════════════════════════════════════
# K6. early_late_vol_asym — 早盘/尾盘成交量不对称
#     = 开盘30分钟成交量 / 尾盘30分钟成交量. 高 → 开盘集中交易.
# ═══════════════════════════════════════════════════════════════════════════
@register_factor("early_late_vol_asym_20d", category="intraday_advanced")
class EarlyLateVolAsym20d(Factor):
    name = "early_late_vol_asym_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "早盘/尾盘成交量不对称"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        vol = panel["volume"]
        day = vol.index.normalize()
        daily = {}
        for dt in sorted(set(day)):
            v = vol.loc[day == dt]
            vals = {}
            for col in v.columns:
                vv = v[col].dropna()
                if len(vv) < 60:
                    continue
                early = vv.iloc[:30].sum()
                late = vv.iloc[-30:].sum()
                vals[col] = early / max(late, 1e-9)
            if vals:
                daily[dt] = pd.Series(vals)
        return _finalize(pd.DataFrame(daily).T, dates, universe)


# ═══════════════════════════════════════════════════════════════════════════
# K7. tail_return_ratio — 尾部收益比
#     = 上尾部均值 / 下尾部均值 (按 ±2σ 定义尾部). 高 → 正尾部占优.
# ═══════════════════════════════════════════════════════════════════════════
@register_factor("tail_return_ratio_20d", category="intraday_advanced")
class TailReturnRatio20d(Factor):
    name = "tail_return_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "尾部收益比 (上下尾部均值比)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret = panel["close"].pct_change()
        day = ret.index.normalize()
        daily = {}
        for dt in sorted(set(day)):
            r = ret.loc[day == dt]
            vals = {}
            for col in r.columns:
                rr = r[col].dropna()
                if len(rr) < 20:
                    continue
                sigma = rr.std(ddof=0)
                if sigma < 1e-12:
                    continue
                upper = rr[rr > 2 * sigma].mean()
                lower = rr[rr < -2 * sigma].mean()
                if np.isnan(upper) or np.isnan(lower) or abs(lower) < 1e-9:
                    continue
                vals[col] = upper / abs(lower)
            if vals:
                daily[dt] = pd.Series(vals)
        return _finalize(pd.DataFrame(daily).T, dates, universe)


# ═══════════════════════════════════════════════════════════════════════════
# K8. vol_clustering — 波动率聚集
#     = 日内分钟波动率的一阶自相关. 高 → 波动率持续性强.
# ═══════════════════════════════════════════════════════════════════════════
@register_factor("vol_clustering_20d", category="intraday_advanced")
class VolClustering20d(Factor):
    name = "vol_clustering_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "波动率聚集 (分钟波动率一阶自相关)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret = panel["close"].pct_change()
        day = ret.index.normalize()
        daily = {}
        for dt in sorted(set(day)):
            r = ret.loc[day == dt]
            vals = {}
            for col in r.columns:
                rr = r[col].dropna()
                if len(rr) < 30:
                    continue
                vol = rr.abs()
                ac = vol.autocorr(lag=1)
                if not np.isnan(ac):
                    vals[col] = ac
            if vals:
                daily[dt] = pd.Series(vals)
        return _finalize(pd.DataFrame(daily).T, dates, universe)


# ═══════════════════════════════════════════════════════════════════════════
# K9. micro_price_impact — 微价格冲击
#     = 成交额加权分钟收益的绝对值均值. 高 → 大资金推动价格.
# ═══════════════════════════════════════════════════════════════════════════
@register_factor("micro_price_impact_20d", category="intraday_advanced")
class MicroPriceImpact20d(Factor):
    name = "micro_price_impact_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "微价格冲击 (成交额加权收益)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel or "amount" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret = panel["close"].pct_change().abs()
        amt = panel["amount"].replace(0, np.nan)
        day = ret.index.normalize()
        daily = {}
        for dt in sorted(set(day)):
            r = ret.loc[day == dt]
            a = amt.loc[amt.index.normalize() == dt]
            common = r.columns.intersection(a.columns)
            vals = {}
            for col in common:
                rr, aa = r[col], a[col]
                idx = rr.dropna().index.intersection(aa.dropna().index)
                if len(idx) < 10:
                    continue
                rr, aa = rr.loc[idx], aa.loc[idx]
                vals[col] = (rr * aa).sum() / aa.sum()
            if vals:
                daily[dt] = pd.Series(vals)
        return _finalize(pd.DataFrame(daily).T, dates, universe)


# ═══════════════════════════════════════════════════════════════════════════
# K10. signed_volume_pressure — 买卖压力
#     = 用 tick-test 近似: 价格上升分钟成交量占比 - 价格下降分钟成交量占比.
#     正 → 买方主导.
# ═══════════════════════════════════════════════════════════════════════════
@register_factor("signed_volume_pressure_20d", category="intraday_advanced")
class SignedVolumePressure20d(Factor):
    name = "signed_volume_pressure_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "买卖压力 (signed volume 代理)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel or "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        vol = panel["volume"]
        ret = close.pct_change()
        day = close.index.normalize()
        daily = {}
        for dt in sorted(set(day)):
            c = close.loc[day == dt]
            v = vol.loc[vol.index.normalize() == dt]
            common = c.columns.intersection(v.columns)
            vals = {}
            for col in common:
                cc, vv = c[col], v[col]
                idx = cc.dropna().index.intersection(vv.dropna().index)
                if len(idx) < 20:
                    continue
                cc, vv = cc.loc[idx], vv.loc[idx]
                rr = cc.pct_change().fillna(0)
                buy = vv[rr > 0].sum()
                sell = vv[rr < 0].sum()
                total = buy + sell
                vals[col] = (buy - sell) / max(total, 1e-9)
            if vals:
                daily[dt] = pd.Series(vals)
        return _finalize(pd.DataFrame(daily).T, dates, universe)

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


def _v_v_cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """截面 rank (0~1)."""
    return df.rank(axis=1, pct=True)


def _v_v_cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """截面 z-score."""
    mean = df.mean(axis=1)
    std = df.std(axis=1, ddof=0).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)


# ═══════════════════════════════════════════════════════════════════════════
# V1. jump_intensity_rank_20d — 跳跃强度截面rank (方向: 负向不变)
# ═══════════════════════════════════════════════════════════════════════════
@register_factor("jump_intensity_rank_20d", category="intraday_advanced")
class JumpIntensityRank20d(_VariantBase):
    """跳跃强度截面排名变体 (基于 intraday_jump_intensity_20d)."""
    name = "jump_intensity_rank_20d"
    description = "跳跃强度截面排名"
    BASE = IntradayJumpIntensity20d

    def _transform(self, base):
        return _v_cs_rank(base)


# ═══════════════════════════════════════════════════════════════════════════
# V2. peak_count_zscore_20d — 价峰计数截面zscore (方向: 正向不变)
# ═══════════════════════════════════════════════════════════════════════════
@register_factor("peak_count_zscore_20d", category="intraday_advanced")
class PeakCountZscore20d(_VariantBase):
    """价峰计数截面标准化变体 (基于 intraday_price_peak_count_20d)."""
    name = "peak_count_zscore_20d"
    description = "价峰计数截面标准化"
    BASE = IntradayPricePeakCount20d

    def _transform(self, base):
        return _v_cs_zscore(base)


# ═══════════════════════════════════════════════════════════════════════════
# V3. skewness_delta_10d — 已实现偏度10日差分 (方向: 正向)
#     捕捉偏度水平的短期变化
# ═══════════════════════════════════════════════════════════════════════════
@register_factor("skewness_delta_10d", category="intraday_advanced")
class SkewnessDelta10d(_VariantBase):
    """已实现偏度10日差分变体 (基于 intraday_realised_skewness_20d)."""
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
    """跌幅时间重心3日平滑变体 (基于 intraday_dtws_20d)."""
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
    """Roll价差波动率缩放变体 (基于 intraday_roll_spread_20d)."""
    name = "roll_spread_vol_scaled_20d"
    description = "Roll价差波动率缩放"
    BASE = IntradayRollSpread20d

    def _transform(self, base):
        rv = _roll_std(base, 20, 5).replace(0, np.nan)
        return base / rv


# ═══════════════════════════════════════════════════════════════════════════
# V6. kyle_lambda_stability_20d — Kyle冲击稳定性 (方向: 负向)
#     均值/标准差: 高=冲击行为稳定可预测
# ═══════════════════════════════════════════════════════════════════════════
@register_factor("kyle_lambda_stability_20d", category="intraday_advanced")
class KyleLambdaStability20d(_VariantBase):
    """Kyle冲击稳定性变体 (均值/标准差, 基于 intraday_kyle_lambda_20d)."""
    name = "kyle_lambda_stability_20d"
    description = "Kyle冲击稳定性 (均值/标准差)"
    BASE = IntradayKyleLambda20d

    def _transform(self, base):
        rm = _roll_mean(base, 20, 5)
        rs = _roll_std(base, 20, 5).replace(0, np.nan)
        return rm / rs


# ═══════════════════════════════════════════════════════════════════════════
# V7. open_close_vol_rank_20d — 开盘尾盘量比截面rank (方向: 负向不变)
# ═══════════════════════════════════════════════════════════════════════════
@register_factor("open_close_vol_rank_20d", category="intraday_advanced")
class OpenCloseVolRank20d(_VariantBase):
    """开盘尾盘量比截面排名变体 (基于 intraday_open_close_volume_ratio_20d)."""
    name = "open_close_vol_rank_20d"
    description = "开盘尾盘量比截面排名"
    BASE = IntradayOpenCloseVolumeRatio20d

    def _transform(self, base):
        return _v_cs_rank(base)


# ═══════════════════════════════════════════════════════════════════════════
# V8. parkinson_over_rv_20d — Parkinson/已实现波动比 (方向: 负向)
#     高=日内震荡但收盘不动→噪声主导
# ═══════════════════════════════════════════════════════════════════════════
@register_factor("parkinson_over_rv_20d", category="intraday_advanced")
class ParkinsonOverRV20d(_VariantBase):
    """Parkinson/已实现波动比变体 (基于 intraday_parkinson_vol_ratio_20d)."""
    name = "parkinson_over_rv_20d"
    description = "Parkinson/已实现波动比"
    BASE = IntradayParkinsonVolRatio20d

    def _transform(self, base):
        # 用基因子自身rolling std作为已实现波动代理
        rv = _roll_std(base, 20, 5).replace(0, np.nan)
        return base / rv


# ═══════════════════════════════════════════════════════════════════════════
# V9. jump_times_skew_20d — 跳跃×偏度交互 (方向: 负向)
#     负偏度伴随高跳跃→恐慌抛售信号增强
# ═══════════════════════════════════════════════════════════════════════════
@register_factor("jump_times_skew_20d", category="intraday_advanced")
class JumpTimesSkew20d(Factor):
    """跳跃×偏度交互因子: 高跳跃低偏度→恐慌抛售信号增强."""
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
    """价峰计数10日差分变体 (基于 intraday_price_peak_count_20d)."""
    name = "peak_count_delta_20d"
    description = "价峰计数10日差分"
    BASE = IntradayPricePeakCount20d

    def _transform(self, base):
        return base.diff(10)


# ═══════════════════════════════════════════════════════════════════════════════
# 191. intraday_open_price_crossings — 穿越开盘价次数
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_open_price_crossings_20d", category="intraday_advanced")
class IntradayOpenPriceCrossings20d(Factor):
    """穿越开盘价次数因子.

    价格上穿/下穿当日开盘价的次数 (开盘价作为锚点, 与 #172 VWAP穿越互补).
    频繁穿越 → 多空拉锯 → 无方向 → 负向.
    方向: 负向.
    """
    name = "intraday_open_price_crossings_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "穿越开盘价次数 (开盘锚振荡=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"open", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        open_px, close = panel["open"], panel["close"]
        day = close.index.normalize()
        crossings: dict = {}
        for dt in sorted(set(day)):
            grp_o = open_px.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 30:
                continue
            vals = {}
            for col in grp_c.columns:
                o = grp_o[col].dropna()
                c = grp_c[col].dropna()
                common = o.index.intersection(c.index)
                if len(common) < 30:
                    continue
                o_first = o.loc[common].iloc[0]
                if o_first < 1e-12:
                    continue
                above = (c.loc[common] > o_first).astype(int).values
                crosses = int(np.sum(np.diff(above) != 0))
                vals[col] = -float(crosses)
            if vals:
                crossings[dt] = pd.Series(vals)
        if not crossings:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(crossings).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 192. intraday_prev_close_crossings — 穿越昨收次数
# ⚠ 跨日因子: 依赖昨日收盘 (prev_close 跨日追踪), 需≥2个交易日历史; 首日为NaN
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_prev_close_crossings_20d", category="intraday_advanced")
class IntradayPrevCloseCrossings20d(Factor):
    """穿越昨收次数因子.

    价格上穿/下穿昨日收盘价的次数 (昨收作为心理锚点).
    频繁穿越昨收 → 多空在昨收附近拉锯 → 无明确方向 → 负向.
    方向: 负向.
    """
    name = "intraday_prev_close_crossings_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "穿越昨收次数 (昨收锚振荡=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        crossings: dict = {}
        prev_close: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < 30:
                    continue
                prev_c = prev_close.get(col)
                if prev_c is None or prev_c < 1e-12:
                    prev_close[col] = c.iloc[-1]
                    continue
                above = (c > prev_c).astype(int).values
                crosses = int(np.sum(np.diff(above) != 0))
                vals[col] = -float(crosses)
                prev_close[col] = c.iloc[-1]
            if vals:
                crossings[dt] = pd.Series(vals)
        if not crossings:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(crossings).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 193. intraday_vwap_band_retention — VWAP 带内停留占比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_vwap_band_retention_20d", category="intraday_advanced")
class IntradayVwapBandRetention20d(Factor):
    """VWAP 带内停留占比因子.

    价格停留在 VWAP±1σ(close) 带内的时间占比 (与 #172 穿越/#126 单线上方互补).
    带内停留多 → 价格收敛于公允价 → 有序 → 正向.
    方向: 正向.
    """
    name = "intraday_vwap_band_retention_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "VWAP带内停留占比 (收敛于公允价=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, volume = panel["close"], panel["volume"]
        day = close.index.normalize()
        retentions: dict = {}
        for dt in sorted(set(day)):
            grp_c = close.loc[day == dt]
            grp_v = volume.loc[day == dt]
            if len(grp_c) < 30:
                continue
            vals = {}
            for col in grp_c.columns:
                c = grp_c[col].dropna()
                v = grp_v[col].dropna()
                common = c.index.intersection(v.index)
                if len(common) < 30:
                    continue
                c_c = c.loc[common]
                v_c = v.loc[common]
                vwap = float((c_c * v_c).sum() / v_c.sum()) if v_c.sum() > 1e-12 else c_c.mean()
                sigma = c_c.std(ddof=0)
                if sigma < 1e-12:
                    vals[col] = 0.0
                    continue
                in_band = ((c_c - vwap).abs() <= sigma).sum()
                vals[col] = float(in_band / len(c_c))
            if vals:
                retentions[dt] = pd.Series(vals)
        if not retentions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(retentions).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 194. intraday_range_position_avg — 日内区间平均位置
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_range_position_avg_20d", category="intraday_advanced")
class IntradayRangePositionAvg20d(Factor):
    """日内区间平均位置因子.

    每分钟价格在日内区间(high-low)位置的均值 (时间积分, 与 #17 收盘单点互补).
    平均位置高 → 多数时间在上半区间 → 买方主导 → 正向.
    方向: 正向.
    """
    name = "intraday_range_position_avg_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "日内区间平均位置 (时间积分, 上方主导=正向)"
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
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 30:
                continue
            vals = {}
            for col in grp_c.columns:
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                c = grp_c[col].dropna()
                common = h.index.intersection(l.index).intersection(c.index)
                if len(common) < 30:
                    continue
                h_c, l_c, c_c = (x.loc[common] for x in (h, l, c))
                rng = h_c.max() - l_c.min()
                if rng < 1e-12:
                    vals[col] = 0.5
                    continue
                pos = (c_c - l_c.min()) / rng
                vals[col] = float(pos.mean())
            if vals:
                positions[dt] = pd.Series(vals)
        if not positions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(positions).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 195. intraday_path_bandwidth — 路径带宽集中度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_path_bandwidth_20d", category="intraday_advanced")
class IntradayPathBandwidth20d(Factor):
    """路径带宽集中度因子.

    close 序列的 (p90-p10) / (high-low) — 价格路径实际占据的带宽.
    窄带宽 → 价格集中在小范围 → 收敛稳定 → 正向.
    方向: 正向.
    """
    name = "intraday_path_bandwidth_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "路径带宽 ((p90-p10)/区间, 窄=收敛=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"high", "low", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, low, close = panel["high"], panel["low"], panel["close"]
        day = close.index.normalize()
        bandwidths: dict = {}
        for dt in sorted(set(day)):
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 30:
                continue
            vals = {}
            for col in grp_c.columns:
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                c = grp_c[col].dropna()
                common = h.index.intersection(l.index).intersection(c.index)
                if len(common) < 30:
                    continue
                h_c, l_c, c_c = (x.loc[common] for x in (h, l, c))
                rng = h_c.max() - l_c.min()
                if rng < 1e-12:
                    vals[col] = 0.0
                    continue
                p10, p90 = np.percentile(c_c.values, [10, 90])
                vals[col] = float((p90 - p10) / rng)
            if vals:
                bandwidths[dt] = pd.Series(vals)
        if not bandwidths:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(bandwidths).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 196. intraday_edge_touch_ratio — 触边时间占比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_edge_touch_ratio_20d", category="intraday_advanced")
class IntradayEdgeTouchRatio20d(Factor):
    """触边时间占比因子.

    价格停留在日内区间上下 10% 内的时间占比 (触边行为).
    触边多 → 价格反复冲边界 → 情绪化/拉锯 → 负向.
    方向: 负向.
    """
    name = "intraday_edge_touch_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "触边时间占比 (反复触边界=拉锯=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"high", "low", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, low, close = panel["high"], panel["low"], panel["close"]
        day = close.index.normalize()
        touches: dict = {}
        for dt in sorted(set(day)):
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 30:
                continue
            vals = {}
            for col in grp_c.columns:
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                c = grp_c[col].dropna()
                common = h.index.intersection(l.index).intersection(c.index)
                if len(common) < 30:
                    continue
                h_c, l_c, c_c = (x.loc[common] for x in (h, l, c))
                rng = h_c.max() - l_c.min()
                if rng < 1e-12:
                    vals[col] = 0.0
                    continue
                pos = (c_c - l_c.min()) / rng
                touch = ((pos <= 0.1) | (pos >= 0.9)).sum()
                vals[col] = -float(touch / len(c_c))
            if vals:
                touches[dt] = pd.Series(vals)
        if not touches:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(touches).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 197. intraday_vwap_above_run — VWAP 上方最长停留
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_vwap_above_run_20d", category="intraday_advanced")
class IntradayVwapAboveRun20d(Factor):
    """VWAP 上方最长停留因子.

    连续在 VWAP 上方的最长分钟数 (与 #172 穿越/#126 占比互补: 这里是单次停留时长).
    长停留 → 买方持续主导 → 正向.
    方向: 正向.
    """
    name = "intraday_vwap_above_run_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "VWAP上方最长停留 (买方持续主导=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, volume = panel["close"], panel["volume"]
        day = close.index.normalize()
        runs: dict = {}
        for dt in sorted(set(day)):
            grp_c = close.loc[day == dt]
            grp_v = volume.loc[day == dt]
            if len(grp_c) < 30:
                continue
            vals = {}
            for col in grp_c.columns:
                c = grp_c[col].dropna()
                v = grp_v[col].dropna()
                common = c.index.intersection(v.index)
                if len(common) < 30:
                    continue
                c_c = c.loc[common]
                v_c = v.loc[common]
                vwap = float((c_c * v_c).sum() / v_c.sum()) if v_c.sum() > 1e-12 else c_c.mean()
                above = (c_c > vwap).astype(int).values
                max_run = 0
                cur = 0
                for a in above:
                    cur = cur + 1 if a else 0
                    max_run = max(max_run, cur)
                vals[col] = float(max_run / len(c_c))
            if vals:
                runs[dt] = pd.Series(vals)
        if not runs:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(runs).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 198. intraday_open_side_retention — 开盘价上方时间占比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_open_side_retention_20d", category="intraday_advanced")
class IntradayOpenSideRetention20d(Factor):
    """开盘价上方时间占比因子.

    价格在开盘价上方的时间占比 (开盘锚定, 与 #191 穿越互补).
    上方时间长 → 开盘后买方主导 → 正向.
    方向: 正向.
    """
    name = "intraday_open_side_retention_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "开盘价上方时间占比 (开盘后买方主导=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"open", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        open_px, close = panel["open"], panel["close"]
        day = close.index.normalize()
        retentions: dict = {}
        for dt in sorted(set(day)):
            grp_o = open_px.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 30:
                continue
            vals = {}
            for col in grp_c.columns:
                o = grp_o[col].dropna()
                c = grp_c[col].dropna()
                common = o.index.intersection(c.index)
                if len(common) < 30:
                    continue
                o_first = o.loc[common].iloc[0]
                if o_first < 1e-12:
                    continue
                vals[col] = float((c.loc[common] > o_first).sum() / len(c.loc[common]))
            if vals:
                retentions[dt] = pd.Series(vals)
        if not retentions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(retentions).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 199. intraday_midline_direction — 中位线穿越方向平衡
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_midline_direction_20d", category="intraday_advanced")
class IntradayMidlineDirection20d(Factor):
    """中位线穿越方向平衡因子.

    穿越日内中位线时向上穿越次数 / 总穿越次数 (方向平衡, 与 #140 总次数互补).
    向上穿越占优 → 突破方向偏多 → 正向.
    方向: 正向.
    """
    name = "intraday_midline_direction_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "中位线穿越方向平衡 (向上穿越占比)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"high", "low", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, low, close = panel["high"], panel["low"], panel["close"]
        day = close.index.normalize()
        balances: dict = {}
        for dt in sorted(set(day)):
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 30:
                continue
            vals = {}
            for col in grp_c.columns:
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                c = grp_c[col].dropna()
                common = h.index.intersection(l.index).intersection(c.index)
                if len(common) < 30:
                    continue
                h_c, l_c, c_c = (x.loc[common] for x in (h, l, c))
                mid = (h_c.max() + l_c.min()) / 2.0
                above = (c_c > mid).astype(int).values
                diff = np.diff(above)
                up_cross = int(np.sum(diff == 1))
                dn_cross = int(np.sum(diff == -1))
                total = up_cross + dn_cross
                vals[col] = float(up_cross / total) if total > 0 else 0.5
            if vals:
                balances[dt] = pd.Series(vals)
        if not balances:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(balances).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 200. intraday_anchor_distance — 价格距日内极值平均距离
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_anchor_distance_20d", category="intraday_advanced")
class IntradayAnchorDistance20d(Factor):
    """价格距日内极值平均距离因子.

    每分钟价格距最近极值(高点或低点)的平均距离 (极值锚定, 与 #194 位置互补).
    距离大 → 价格远离两端 → 居中的犹豫 → 负向.
    方向: 负向.
    """
    name = "intraday_anchor_distance_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "距日内极值平均距离 (居中犹豫=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"high", "low", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, low, close = panel["high"], panel["low"], panel["close"]
        day = close.index.normalize()
        distances: dict = {}
        for dt in sorted(set(day)):
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 30:
                continue
            vals = {}
            for col in grp_c.columns:
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                c = grp_c[col].dropna()
                common = h.index.intersection(l.index).intersection(c.index)
                if len(common) < 30:
                    continue
                h_c, l_c, c_c = (x.loc[common] for x in (h, l, c))
                rng = h_c.max() - l_c.min()
                if rng < 1e-12:
                    vals[col] = 0.0
                    continue
                dist_to_high = (h_c.max() - c_c) / rng
                dist_to_low = (c_c - l_c.min()) / rng
                min_dist = np.minimum(dist_to_high, dist_to_low)
                vals[col] = -float(min_dist.mean())
            if vals:
                distances[dt] = pd.Series(vals)
        if not distances:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(distances).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 201. intraday_oi_turnover — 持仓换手率 (Volume/OI)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_turnover_20d", category="intraday_advanced")
class IntradayOiTurnover20d(Factor):
    """持仓换手率因子.

    当日总成交量 / 当日平均持仓量 (Volume/OI).
    高换手 → 短线交易主导 → 价格噪音大 → 负向.
    低换手 → 长线资金主导 → 趋势稳定 → 正向.
    方向: 负向.
    """
    name = "intraday_oi_turnover_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "持仓换手率 (Volume/OI, 高=短线噪音=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "volume" not in panel or "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        volume, position = panel["volume"], panel["position"]
        day = volume.index.normalize()
        turnovers: dict = {}
        for dt in sorted(set(day)):
            grp_v = volume.loc[day == dt]
            grp_p = position.loc[day == dt]
            if len(grp_v) < 20:
                continue
            vals = {}
            for col in grp_v.columns:
                v = grp_v[col].dropna()
                p = grp_p[col].dropna()
                common = v.index.intersection(p.index)
                if len(common) < 20:
                    continue
                vol_sum = v.loc[common].sum()
                oi_mean = p.loc[common].mean()
                vals[col] = float(vol_sum / oi_mean) if oi_mean > 1e-12 else 0.0
            if vals:
                turnovers[dt] = pd.Series(vals)
        if not turnovers:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(turnovers).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 202. intraday_oi_accumulation — 持仓累积强度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_accumulation_20d", category="intraday_advanced")
class IntradayOiAccumulation20d(Factor):
    """持仓累积强度因子.

    (OI_end - OI_start) / OI_start — 日内持仓量净变化率.
    持续增仓 → 资金流入 → 趋势延续性 → 正向.
    减仓 → 资金离场 → 趋势动能减弱 → 负向.
    方向: 正向.
    """
    name = "intraday_oi_accumulation_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "持仓累积强度 ((OI_end-OI_start)/OI_start, 增仓=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        position = panel["position"]
        day = position.index.normalize()
        accumulations: dict = {}
        for dt in sorted(set(day)):
            grp = position.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                p = grp[col].dropna()
                if len(p) < 20:
                    continue
                oi_start = p.iloc[0]
                oi_end = p.iloc[-1]
                vals[col] = float(oi_end / oi_start - 1.0) if oi_start > 1e-12 else 0.0
            if vals:
                accumulations[dt] = pd.Series(vals)
        if not accumulations:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(accumulations).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 203. intraday_oi_price_sensitivity — 持仓-价格敏感度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_price_sensitivity_20d", category="intraday_advanced")
class IntradayOiPriceSensitivity20d(Factor):
    """持仓-价格敏感度因子.

    分钟收益与分钟持仓变化的相关系数 corr(ret_1m, ΔOI_1m).
    正相关 → 涨时增仓 (多头主动加仓) → 趋势健康 → 正向.
    负相关 → 涨时减仓 (空头平仓/多头获利离场) → 趋势存疑 → 负向.
    方向: 正向.
    """
    name = "intraday_oi_price_sensitivity_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "持仓-价格敏感度 (corr(ret, ΔOI), 涨时增仓=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel or "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, position = panel["close"], panel["position"]
        ret_1m = close.pct_change()
        oi_change = position.diff()
        day = ret_1m.index.normalize()
        sensitivities: dict = {}
        for dt in sorted(set(day)):
            grp_r = ret_1m.loc[day == dt]
            grp_o = oi_change.loc[day == dt]
            if len(grp_r) < 30:
                continue
            vals = {}
            for col in grp_r.columns:
                r = grp_r[col].dropna()
                o = grp_o[col].dropna()
                common = r.index.intersection(o.index)
                if len(common) < 30:
                    continue
                r_c = r.loc[common]
                o_c = o.loc[common]
                if r_c.std(ddof=0) < 1e-12 or o_c.std(ddof=0) < 1e-12:
                    vals[col] = 0.0
                else:
                    corr_val = float(np.corrcoef(r_c, o_c)[0, 1])
                    vals[col] = corr_val if not np.isnan(corr_val) else 0.0
            if vals:
                sensitivities[dt] = pd.Series(vals)
        if not sensitivities:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(sensitivities).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 204. intraday_close_high_strength — 尾盘高位维持强度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_close_high_strength_20d", category="intraday_advanced")
class IntradayCloseHighStrength20d(Factor):
    """尾盘高位维持强度因子.

    最后30分钟平均价 / 当日最高价 — 收盘段能否维持在高点附近 (用户 extreme_persistence 改名).
    高位维持强 → 买盘持续 → 正向. 与 #17 收盘位置互补 (高点锚 vs 低点锚).
    方向: 正向.
    """
    name = "intraday_close_high_strength_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "尾盘高位维持 (最后30分均价/日内高点)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"high", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, close = panel["high"], panel["close"]
        day = close.index.normalize()
        strengths: dict = {}
        for dt in sorted(set(day)):
            grp_h = high.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 30:
                continue
            n_tail = max(10, min(30, len(grp_c) // 4))
            vals = {}
            for col in grp_c.columns:
                h = grp_h[col].dropna()
                c = grp_c[col].dropna()
                if len(h) < 20 or len(c) < 30:
                    continue
                h_day = h.max()
                if h_day < 1e-12:
                    continue
                tail_mean = c.iloc[-n_tail:].mean()
                vals[col] = float(tail_mean / h_day)
            if vals:
                strengths[dt] = pd.Series(vals)
        if not strengths:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(strengths).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 205. intraday_open_drive — 开盘方向动能
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_open_drive_20d", category="intraday_advanced")
class IntradayOpenDrive20d(Factor):
    """开盘方向动能因子.

    前30分钟收益 / 日内区间(high-low) — 开盘段方向性动能 (用户 opening_drive 改名).
    开盘强势 → 隔夜信息被快速定价 → 动量延续 → 正向.
    与 #65 全天收益互补: 这里聚焦开盘30分钟且按区间归一化.
    方向: 正向.
    """
    name = "intraday_open_drive_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "开盘方向动能 (前30分收益/日内区间)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"open", "high", "low", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        open_px, high, low, close = panel["open"], panel["high"], panel["low"], panel["close"]
        day = close.index.normalize()
        drives: dict = {}
        for dt in sorted(set(day)):
            grp_o = open_px.loc[day == dt]
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 30:
                continue
            n_open = max(10, min(30, len(grp_c) // 4))
            vals = {}
            for col in grp_c.columns:
                o = grp_o[col].dropna()
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                c = grp_c[col].dropna()
                common = o.index.intersection(h.index).intersection(l.index).intersection(c.index)
                if len(common) < 30:
                    continue
                o_c, h_c, l_c, c_c = (x.loc[common] for x in (o, h, l, c))
                o_first = o_c.iloc[0]
                rng = h_c.max() - l_c.min()
                if o_first < 1e-12 or rng < 1e-12:
                    continue
                open_ret = c_c.iloc[n_open - 1] / o_first - 1.0
                vals[col] = float(open_ret / rng)
            if vals:
                drives[dt] = pd.Series(vals)
        if not drives:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(drives).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 206. intraday_big_bar_ratio — 大单分钟占比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_big_bar_ratio_20d", category="intraday_advanced")
class IntradayBigBarRatio20d(Factor):
    """大单分钟占比因子.

    成交量 > 1.5×当日均量的分钟的量占总量的比例 (用户 big_trade_ratio 改名).
    大单占比高 → 机构参与度高 → 趋势延续 → 正向.
    与 #89 (3σ频率) / #164 (top5%集中) 阈值不同.
    方向: 正向.
    """
    name = "intraday_big_bar_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "大单分钟占比 (>1.5x均量分钟的量占比)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        volume = panel["volume"]
        day = volume.index.normalize()
        ratios: dict = {}
        for dt in sorted(set(day)):
            grp = volume.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                v = grp[col].dropna()
                if len(v) < 20:
                    continue
                v_mean = v.mean()
                total = v.sum()
                if v_mean < 1e-12 or total < 1e-12:
                    continue
                big_mask = v > 1.5 * v_mean
                vals[col] = float(v[big_mask].sum() / total)
            if vals:
                ratios[dt] = pd.Series(vals)
        if not ratios:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(ratios).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 207. intraday_range_vol_ratio — 相对波动水平
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_range_vol_ratio_20d", category="intraday_advanced")
class IntradayRangeVolRatio20d(Factor):
    """相对波动水平因子.

    (high-low) / |prev_close| — 日内振幅相对昨收价的扩张程度 (用户 range_expansion 改名).
    振幅突扩 → 重大信息冲击 → 方向难测 → 负向.
    与 #142 (今vs历史振幅) 互补: 这里是相对昨收绝对水平.
    方向: 负向.
    """
    name = "intraday_range_vol_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "相对波动水平 ((high-low)/|prev_close|, 突扩=负向)"
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
        prev_close: dict = {}
        for dt in sorted(set(day)):
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 20:
                continue
            vals = {}
            for col in grp_c.columns:
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                c = grp_c[col].dropna()
                if len(h) < 10 or len(l) < 10 or len(c) < 10:
                    continue
                prev_c = prev_close.get(col)
                if prev_c is None or prev_c < 1e-12:
                    prev_close[col] = c.iloc[-1]
                    continue
                rng = h.max() - l.min()
                vals[col] = float(rng / abs(prev_c)) if abs(prev_c) > 1e-12 else 0.0
                prev_close[col] = c.iloc[-1]
            if vals:
                ratios[dt] = pd.Series(vals)
        if not ratios:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(ratios).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 208. intraday_oi_trend — 持仓变化时间趋势
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_trend_20d", category="intraday_advanced")
class IntradayOiTrend20d(Factor):
    """持仓变化时间趋势因子.

    分钟 OI 变化对时间回归的斜率 (借鉴 #123 volume_trend 的构造, 迁移到 OI).
    持续增仓趋势 → 资金持续流入 → 正向.
    方向: 正向.
    """
    name = "intraday_oi_trend_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "持仓变化趋势 (OI对时间斜率, 增仓趋势=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        position = panel["position"]
        day = position.index.normalize()
        trends: dict = {}
        for dt in sorted(set(day)):
            grp = position.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                p = grp[col].dropna()
                if len(p) < 30:
                    continue
                t = np.arange(len(p)) / max(1, len(p) - 1)
                slope = np.polyfit(t, p.values, 1)[0]
                base = p.mean()
                vals[col] = float(slope / base) if base > 1e-12 else 0.0
            if vals:
                trends[dt] = pd.Series(vals)
        if not trends:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(trends).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 209. intraday_oi_dispersion — 持仓波动度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_dispersion_20d", category="intraday_advanced")
class IntradayOiDispersion20d(Factor):
    """持仓波动度因子.

    分钟 OI 序列的变异系数 (借鉴 #12 volume_vol 的构造, 迁移到 OI).
    OI 波动大 → 持仓不稳/多空反复 → 负向.
    方向: 负向.
    """
    name = "intraday_oi_dispersion_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "持仓波动度 (OI变异系数, 持仓不稳=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        position = panel["position"]
        day = position.index.normalize()
        dispersions: dict = {}
        for dt in sorted(set(day)):
            grp = position.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                p = grp[col].dropna()
                if len(p) < 20:
                    continue
                m, s = p.mean(), p.std(ddof=0)
                vals[col] = -float(s / m) if m > 1e-12 else 0.0
            if vals:
                dispersions[dt] = pd.Series(vals)
        if not dispersions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(dispersions).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 210. intraday_oi_vol_corr — 持仓-成交量相关
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_vol_corr_20d", category="intraday_advanced")
class IntradayOiVolCorr20d(Factor):
    """持仓-成交量相关因子.

    corr(ΔOI_1m, vol_1m) (借鉴 #1 vp_corr 的构造, 迁移到 OI×量).
    放量伴随增仓 → 真实换手 → 健康; 放量但减仓 → 对倒/离场 → 负向.
    方向: 正向.
    """
    name = "intraday_oi_vol_corr_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "持仓-量相关 (corr(ΔOI,vol), 放量增仓=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "volume" not in panel or "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        volume, position = panel["volume"], panel["position"]
        oi_change = position.diff()
        day = volume.index.normalize()
        corrs: dict = {}
        for dt in sorted(set(day)):
            grp_v = volume.loc[day == dt]
            grp_o = oi_change.loc[day == dt]
            if len(grp_v) < 30:
                continue
            vals = {}
            for col in grp_v.columns:
                v = grp_v[col].dropna()
                o = grp_o[col].dropna()
                common = v.index.intersection(o.index)
                if len(common) < 30:
                    continue
                v_c = v.loc[common]
                o_c = o.loc[common]
                if v_c.std(ddof=0) < 1e-12 or o_c.std(ddof=0) < 1e-12:
                    vals[col] = 0.0
                else:
                    corr_val = float(np.corrcoef(v_c, o_c)[0, 1])
                    vals[col] = corr_val if not np.isnan(corr_val) else 0.0
            if vals:
                corrs[dt] = pd.Series(vals)
        if not corrs:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(corrs).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 211. intraday_oi_half_life — 持仓记忆半衰期
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_half_life_20d", category="intraday_advanced")
class IntradayOiHalfLife20d(Factor):
    """持仓记忆半衰期因子.

    分钟 OI 变化自相关衰减到0.5的滞后数 (借鉴 #161/#175 的构造, 迁移到 OI).
    持仓变动记忆长 → 资金动作持续 → 不稳定 → 负向.
    方向: 负向.
    """
    name = "intraday_oi_half_life_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "持仓记忆半衰期 (OI变化自相关衰减, 长=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        oi_change = panel["position"].diff()
        day = oi_change.index.normalize()
        half_lives: dict = {}
        for dt in sorted(set(day)):
            grp = oi_change.loc[day == dt]
            if len(grp) < 60:
                continue
            vals = {}
            for col in grp.columns:
                d = grp[col].dropna().values
                if len(d) < 60:
                    continue
                max_lag = min(15, len(d) // 4)
                hl = max_lag
                for lag in range(1, max_lag + 1):
                    x, y = d[:-lag], d[lag:]
                    if x.std(ddof=0) < 1e-12 or y.std(ddof=0) < 1e-12:
                        continue
                    rho = float(np.corrcoef(x, y)[0, 1])
                    if rho <= 0.5:
                        hl = lag
                        break
                vals[col] = -float(hl)
            if vals:
                half_lives[dt] = pd.Series(vals)
        if not half_lives:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(half_lives).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 212. intraday_oi_change_skew — 持仓变化偏度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_change_skew_20d", category="intraday_advanced")
class IntradayOiChangeSkew20d(Factor):
    """持仓变化偏度因子.

    分钟 OI 变化分布偏度 (借鉴 #86 volume_skew 的构造, 迁移到 OI).
    正偏 → 少数大笔增仓 → 资金集中突击 → 正向.
    方向: 正向.
    """
    name = "intraday_oi_change_skew_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "持仓变化偏度 (OI变化偏度, 大笔增仓=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        oi_change = panel["position"].diff()
        day = oi_change.index.normalize()
        skews: dict = {}
        for dt in sorted(set(day)):
            grp = oi_change.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                d = grp[col].dropna()
                if len(d) < 30:
                    continue
                d_nz = d[d != 0]
                if len(d_nz) < 15:
                    vals[col] = 0.0
                    continue
                vals[col] = float(pd.Series(d_nz).skew())
            if vals:
                skews[dt] = pd.Series(vals)
        if not skews:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(skews).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 213. intraday_oi_signed_change — 增减仓量平衡
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_signed_change_20d", category="intraday_advanced")
class IntradayOiSignedChange20d(Factor):
    """增减仓量平衡因子.

    (增仓分钟的量 - 减仓分钟的量) / 总成交量 (借鉴 #38 signed_volume_ratio 的构造, 迁移到 OI).
    正 → 增仓主导 → 资金流入 → 正向.
    方向: 正向.
    """
    name = "intraday_oi_signed_change_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "增减仓平衡 (增仓量-减仓量)/总量"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "volume" not in panel or "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        volume, position = panel["volume"], panel["position"]
        oi_change = position.diff()
        day = volume.index.normalize()
        balances: dict = {}
        for dt in sorted(set(day)):
            grp_v = volume.loc[day == dt]
            grp_o = oi_change.loc[day == dt]
            if len(grp_v) < 20:
                continue
            vals = {}
            for col in grp_v.columns:
                v = grp_v[col].dropna()
                o = grp_o[col].dropna()
                common = v.index.intersection(o.index)
                if len(common) < 20:
                    continue
                v_c = v.loc[common]
                o_c = o.loc[common]
                add_vol = v_c[o_c > 0].sum()
                red_vol = v_c[o_c < 0].sum()
                total = add_vol + red_vol
                vals[col] = float((add_vol - red_vol) / total) if total > 1e-12 else 0.0
            if vals:
                balances[dt] = pd.Series(vals)
        if not balances:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(balances).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 214. intraday_oi_accumulation_run — 连续增仓最长run
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_accumulation_run_20d", category="intraday_advanced")
class IntradayOiAccumulationRun20d(Factor):
    """连续增仓最长run因子.

    连续增仓分钟的最长长度 (借鉴 #41 price_run_duration 的构造, 迁移到 OI).
    长增仓run → 资金持续流入 → 正向.
    方向: 正向.
    """
    name = "intraday_oi_accumulation_run_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "连续增仓最长run (资金持续流入=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        oi_change = panel["position"].diff()
        day = oi_change.index.normalize()
        runs: dict = {}
        for dt in sorted(set(day)):
            grp = oi_change.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                d = grp[col].dropna().values
                if len(d) < 20:
                    continue
                max_run = 0
                cur = 0
                for x in d:
                    cur = cur + 1 if x > 0 else 0
                    max_run = max(max_run, cur)
                vals[col] = float(max_run / len(d))
            if vals:
                runs[dt] = pd.Series(vals)
        if not runs:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(runs).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 215. intraday_oi_vol_ratio — 持仓变动效率
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_vol_ratio_20d", category="intraday_advanced")
class IntradayOiVolRatio20d(Factor):
    """持仓变动效率因子.

    |ΣΔOI| / Σvol — 净持仓变动占总成交量的比例 (持仓变动的"效率").
    高 → 大部分成交导致持仓变化 → 真实建仓/平仓 → 信息含量高 → 正向.
    低 → 大量对倒/换手不改变持仓 → 噪音 → 负向.
    方向: 正向.
    """
    name = "intraday_oi_vol_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "持仓变动效率 (|ΣΔOI|/Σvol, 真实建仓=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "volume" not in panel or "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        volume, position = panel["volume"], panel["position"]
        oi_change = position.diff()
        day = volume.index.normalize()
        ratios: dict = {}
        for dt in sorted(set(day)):
            grp_v = volume.loc[day == dt]
            grp_o = oi_change.loc[day == dt]
            if len(grp_v) < 20:
                continue
            vals = {}
            for col in grp_v.columns:
                v = grp_v[col].dropna()
                o = grp_o[col].dropna()
                common = v.index.intersection(o.index)
                if len(common) < 20:
                    continue
                vol_sum = v.loc[common].sum()
                net_oi = abs(o.loc[common].sum())
                vals[col] = float(net_oi / vol_sum) if vol_sum > 1e-12 else 0.0
            if vals:
                ratios[dt] = pd.Series(vals)
        if not ratios:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(ratios).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 216. intraday_settle_drift — 收盘结算价偏离
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_settle_drift_20d", category="intraday_advanced")
class IntradaySettleDrift20d(Factor):
    """收盘结算价偏离因子.

    (close - settle) / settle — 收盘价相对当日结算价的偏离.
    需要日度 settle 字段 (从 futureshistoryprices1d 的 settle_price 读取; 缺失时因子返回 NaN 自动降级).
    收盘强于结算 → 尾盘买方主导 → 正向.
    方向: 正向.
    """
    name = "intraday_settle_drift_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "收盘结算偏离 ((close-settle)/settle, 需settle数据)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        settle = _read_local_daily(data, dates, universe, "settle")
        if close is None or settle is None or close.empty or settle.empty:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        drift = (close - settle) / settle.replace(0, np.nan)
        return drift.reindex(index=dates, columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 217. intraday_settle_position — 结算价日内位置
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_settle_position_20d", category="intraday_advanced")
class IntradaySettlePosition20d(Factor):
    """结算价日内位置因子.

    (settle - low) / (high - low) — 结算价在日内高低区间的位置.
    结算偏高 → 当日定价偏多 → 正向. 需 settle 数据.
    方向: 正向.
    """
    name = "intraday_settle_position_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "结算价日内位置 ((settle-low)/(high-low), 需settle)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        settle = _read_local_daily(data, dates, universe, "settle")
        high = data.get("high", dates, universe)
        low = data.get("low", dates, universe)
        if settle is None or settle.empty or high is None or low is None:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        if settle.empty or high.empty or low.empty:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        rng = (high - low).replace(0, np.nan)
        pos = (settle - low) / rng
        return pos.reindex(index=dates, columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 218. intraday_settle_oi_change — 结算价-持仓协同
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_settle_oi_change_20d", category="intraday_advanced")
class IntradaySettleOiChange20d(Factor):
    """结算价-持仓协同因子.

    跨日: 结算价变动方向与持仓变动方向一致的比例 (借鉴 #203 的日度版).
    结算上涨+增仓 → 多头主动加仓且被结算确认 → 正向. 需 settle 数据.
    方向: 正向.
    """
    name = "intraday_settle_oi_change_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "结算价-持仓协同 (结算涨+增仓=正向, 需settle)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        settle = _read_local_daily(data, dates, universe, "settle")
        oi = _read_local_daily(data, dates, universe, "oi")
        if settle is None or oi is None or settle.empty or oi.empty:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        settle_chg = settle.diff()
        oi_chg = oi.diff()
        agree = (settle_chg > 0) == (oi_chg > 0)
        # 只统计有变动的日子
        valid = (settle_chg.abs() > 1e-12) & (oi_chg.abs() > 1e-12)
        ratio = agree.where(valid).rolling(20, min_periods=5).mean()
        return ratio.reindex(index=dates, columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 219. intraday_settle_gap — 开盘结算跳空
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_settle_gap_20d", category="intraday_advanced")
class IntradaySettleGap20d(Factor):
    """开盘结算跳空因子.

    (open - prev_settle) / prev_settle — 开盘价相对前日结算价的跳空.
    开盘强于结算 → 盘后信息被正向定价 → 正向. 需 settle 数据.
    方向: 正向.
    """
    name = "intraday_settle_gap_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "开盘结算跳空 ((open-prev_settle)/prev_settle, 需settle)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        open_px = data.get("open", dates, universe)
        settle = _read_local_daily(data, dates, universe, "settle")
        if open_px is None or settle is None or open_px.empty or settle.empty:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        prev_settle = settle.shift(1)
        gap = (open_px - prev_settle) / prev_settle.replace(0, np.nan)
        return gap.reindex(index=dates, columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 220. intraday_oi_price_divergence — 持仓-价格背离
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_price_divergence_20d", category="intraday_advanced")
class IntradayOiPriceDivergence20d(Factor):
    """持仓-价格背离因子.

    价格上涨但持仓减少的分钟占比 (借鉴 #165 背离构造, 迁移到 OI×价).
    涨但减仓 → 上涨由空头平仓推动 → 趋势存疑 → 负向.
    方向: 负向.
    """
    name = "intraday_oi_price_divergence_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "持仓-价背离 (涨但减仓占比, 趋势存疑=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel or "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, position = panel["close"], panel["position"]
        ret_1m = close.pct_change()
        oi_change = position.diff()
        day = ret_1m.index.normalize()
        divergences: dict = {}
        for dt in sorted(set(day)):
            grp_r = ret_1m.loc[day == dt]
            grp_o = oi_change.loc[day == dt]
            if len(grp_r) < 20:
                continue
            vals = {}
            for col in grp_r.columns:
                r = grp_r[col].dropna()
                o = grp_o[col].dropna()
                common = r.index.intersection(o.index)
                if len(common) < 20:
                    continue
                r_c = r.loc[common]
                o_c = o.loc[common]
                up = r_c > 0
                red = o_c < 0
                if up.sum() < 3:
                    vals[col] = 0.0
                    continue
                diverge = (up & red).sum()
                vals[col] = -float(diverge / up.sum())
            if vals:
                divergences[dt] = pd.Series(vals)
        if not divergences:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(divergences).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 221. intraday_oi_extreme_conc — 单分钟持仓变动集中度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_extreme_conc_20d", category="intraday_advanced")
class IntradayOiExtremeConc20d(Factor):
    """单分钟持仓变动集中度因子.

    最大单分钟 |ΔOI| / Σ|ΔOI| (借鉴 #120 extreme_conc 的构造, 迁移到 OI).
    高集中 → 持仓变动依赖单点 → 大资金一次性动作 → 不稳定 → 负向.
    方向: 负向.
    """
    name = "intraday_oi_extreme_conc_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "单分钟持仓集中 (max|ΔOI|/Σ|ΔOI|)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        oi_change = panel["position"].diff().abs()
        day = oi_change.index.normalize()
        concs: dict = {}
        for dt in sorted(set(day)):
            grp = oi_change.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                d = grp[col].dropna()
                if len(d) < 30:
                    continue
                total = d.sum()
                vals[col] = float(d.max() / total) if total > 1e-12 else 0.0
            if vals:
                concs[dt] = pd.Series(vals)
        if not concs:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(concs).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 222. intraday_oi_shock_freq — 持仓突变频率
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_shock_freq_20d", category="intraday_advanced")
class IntradayOiShockFreq20d(Factor):
    """持仓突变频率因子.

    |ΔOI| > 3σ 的分钟占比 (借鉴 #89 liquidity_spike_freq 的构造, 迁移到 OI).
    高频突变 → 资金频繁大动作 → 不稳定 → 负向.
    方向: 负向.
    """
    name = "intraday_oi_shock_freq_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "持仓突变频率 (|ΔOI|>3σ占比)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        oi_change = panel["position"].diff().abs()
        day = oi_change.index.normalize()
        freqs: dict = {}
        for dt in sorted(set(day)):
            grp = oi_change.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                d = grp[col].dropna()
                if len(d) < 30:
                    continue
                mu, sigma = d.mean(), d.std(ddof=0)
                if sigma < 1e-12:
                    vals[col] = 0.0
                else:
                    vals[col] = -float((d > mu + 3.0 * sigma).sum() / len(d))
            if vals:
                freqs[dt] = pd.Series(vals)
        if not freqs:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(freqs).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 223. intraday_oi_time_centroid — 增仓时间重心
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_time_centroid_20d", category="intraday_advanced")
class IntradayOiTimeCentroid20d(Factor):
    """增仓时间重心因子.

    增仓分钟的时间加权重心 (借鉴 #16 volume_time_centroid 的构造, 迁移到 OI).
    增仓偏早盘 → 资金有计划入场 → 正向. 取负使早盘集中=高值.
    方向: 正向.
    """
    name = "intraday_oi_time_centroid_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "增仓时间重心 (增仓偏早盘=有计划=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        oi_change = panel["position"].diff()
        day = oi_change.index.normalize()
        centroids: dict = {}
        for dt in sorted(set(day)):
            grp = oi_change.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                d = grp[col].dropna()
                if len(d) < 20:
                    continue
                add_mask = d > 0
                if not add_mask.any():
                    vals[col] = 0.0
                    continue
                tw = np.arange(1, len(d) + 1) / len(d)
                weights = d[add_mask]
                tw_add = tw[add_mask.values]
                denom = weights.sum()
                centroid = float((tw_add * weights.values).sum() / denom) if denom > 1e-12 else 0.5
                vals[col] = -centroid
            if vals:
                centroids[dt] = pd.Series(vals)
        if not centroids:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(centroids).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 224. intraday_term_slope — 期限结构斜率
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_term_slope_20d", category="intraday_advanced")
class IntradayTermSlope20d(Factor):
    """期限结构斜率因子.

    (near_close - far_close) / far_close 的日内均值 — 主连(近月)相对次连(远月)的升贴水.
    正值(backwardation近高远低) → 现货紧张 → 看多 → 正向.
    方向: 正向.
    """
    name = "intraday_term_slope_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "期限结构斜率 ((near-far)/far, backwardation=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_term_structure_panel(data, dates, universe, freq="1min")
        if "near_close" not in panel or "far_close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        near, far = panel["near_close"], panel["far_close"]
        day = near.index.normalize()
        slopes: dict = {}
        for dt in sorted(set(day)):
            grp_n = near.loc[day == dt]
            grp_f = far.loc[day == dt]
            if len(grp_n) < 20:
                continue
            vals = {}
            for col in grp_n.columns:
                n = grp_n[col].dropna()
                f = grp_f[col].dropna()
                common = n.index.intersection(f.index)
                if len(common) < 20:
                    continue
                n_c = n.loc[common]
                f_c = f.loc[common].replace(0, np.nan)
                slope = ((n_c - f_c) / f_c).dropna()
                if len(slope) < 10:
                    continue
                vals[col] = float(slope.mean())
            if vals:
                slopes[dt] = pd.Series(vals)
        if not slopes:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(slopes).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 225. intraday_term_slope_change — 期限结构斜率变化
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_term_slope_change_20d", category="intraday_advanced")
class IntradayTermSlopeChange20d(Factor):
    """期限结构斜率变化因子.

    今日斜率 - 昨日斜率 (期限结构转紧/转松的动态, 借鉴 #116 跨日比较).
    斜率走强 → 现货端趋紧 → 正向.
    方向: 正向.
    """
    name = "intraday_term_slope_change_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "期限结构斜率变化 (今日-昨日, 转紧=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_term_structure_panel(data, dates, universe, freq="1min")
        if "near_close" not in panel or "far_close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        near, far = panel["near_close"], panel["far_close"]
        day = near.index.normalize()
        daily_slope: dict = {}
        for dt in sorted(set(day)):
            grp_n = near.loc[day == dt]
            grp_f = far.loc[day == dt]
            if len(grp_n) < 20:
                continue
            vals = {}
            for col in grp_n.columns:
                n = grp_n[col].dropna()
                f = grp_f[col].dropna()
                common = n.index.intersection(f.index)
                if len(common) < 20:
                    continue
                n_c = n.loc[common]
                f_c = f.loc[common]
                slope = (n_c - f_c) / f_c.replace(0, np.nan)
                slope = slope.dropna()
                if len(slope) < 10:
                    continue
                vals[col] = float(slope.mean())
            if vals:
                daily_slope[dt] = pd.Series(vals)
        if not daily_slope:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily_slope).T
        df.index = pd.DatetimeIndex(df.index)
        chg = df.diff()
        return chg.reindex(index=dates, columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 226. intraday_term_spread_vol — 价差波动率
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_term_spread_vol_20d", category="intraday_advanced")
class IntradayTermSpreadVol20d(Factor):
    """价差波动率因子.

    (far - near) 价差的日内标准差 (价差的稳定性).
    价差波动大 → 期限结构不稳定 → 负向.
    方向: 负向.
    """
    name = "intraday_term_spread_vol_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "价差波动率 (far-near日内std, 不稳定=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_term_structure_panel(data, dates, universe, freq="1min")
        if "near_close" not in panel or "far_close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        near, far = panel["near_close"], panel["far_close"]
        day = near.index.normalize()
        vols: dict = {}
        for dt in sorted(set(day)):
            grp_n = near.loc[day == dt]
            grp_f = far.loc[day == dt]
            if len(grp_n) < 20:
                continue
            vals = {}
            for col in grp_n.columns:
                n = grp_n[col].dropna()
                f = grp_f[col].dropna()
                common = n.index.intersection(f.index)
                if len(common) < 20:
                    continue
                spread = f.loc[common] - n.loc[common]
                vals[col] = -float(spread.std(ddof=0))
            if vals:
                vols[dt] = pd.Series(vals)
        if not vols:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(vols).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 227. intraday_term_oi_ratio — 近月持仓占比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_term_oi_ratio_20d", category="intraday_advanced")
class IntradayTermOiRatio20d(Factor):
    """近月持仓占比因子.

    near_position / (near_position + far_position) 日内均值 — 持仓在期限结构的分布.
    近月持仓集中 → 现货端博弈激烈 → 逼仓/紧缺 → 正向.
    方向: 正向.
    """
    name = "intraday_term_oi_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "近月持仓占比 (near_pos/(near+far)pos, 现货紧=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_term_structure_panel(data, dates, universe, freq="1min")
        if "near_position" not in panel or "far_position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        near_pos, far_pos = panel["near_position"], panel["far_position"]
        day = near_pos.index.normalize()
        ratios: dict = {}
        for dt in sorted(set(day)):
            grp_n = near_pos.loc[day == dt]
            grp_f = far_pos.loc[day == dt]
            if len(grp_n) < 20:
                continue
            vals = {}
            for col in grp_n.columns:
                n = grp_n[col].dropna()
                f = grp_f[col].dropna()
                common = n.index.intersection(f.index)
                if len(common) < 20:
                    continue
                n_c = n.loc[common]
                f_c = f.loc[common]
                total = n_c + f_c
                ratio = (n_c / total.replace(0, np.nan)).dropna()
                if len(ratio) < 10:
                    continue
                vals[col] = float(ratio.mean())
            if vals:
                ratios[dt] = pd.Series(vals)
        if not ratios:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(ratios).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 228. intraday_term_slope_ma_cross — 斜率均线交叉
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_term_slope_ma_cross_20d", category="intraday_advanced")
class IntradayTermSlopeMaCross20d(Factor):
    """斜率均线交叉因子.

    期限结构斜率的 MA5 与 MA20 之差 (借鉴 #117 均线交叉构造, 用于期限结构斜率).
    快线在慢线上方 → 斜率走强 → 正向.
    方向: 正向.
    """
    name = "intraday_term_slope_ma_cross_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "斜率均线交叉 (slope的MA5-MA20)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_term_structure_panel(data, dates, universe, freq="1min")
        if "near_close" not in panel or "far_close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        near, far = panel["near_close"], panel["far_close"]
        day = near.index.normalize()
        daily_slope: dict = {}
        for dt in sorted(set(day)):
            grp_n = near.loc[day == dt]
            grp_f = far.loc[day == dt]
            if len(grp_n) < 20:
                continue
            vals = {}
            for col in grp_n.columns:
                n = grp_n[col].dropna()
                f = grp_f[col].dropna()
                common = n.index.intersection(f.index)
                if len(common) < 20:
                    continue
                slope = (n.loc[common] - f.loc[common]) / f.loc[common].replace(0, np.nan)
                slope = slope.dropna()
                if len(slope) < 10:
                    continue
                vals[col] = float(slope.mean())
            if vals:
                daily_slope[dt] = pd.Series(vals)
        if not daily_slope:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily_slope).T
        df.index = pd.DatetimeIndex(df.index)
        ma5 = df.rolling(5, min_periods=3).mean()
        ma20 = df.rolling(20, min_periods=5).mean()
        cross = (ma5 - ma20) / ma20.replace(0, np.nan)
        return cross.reindex(index=dates, columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 229. intraday_term_vol_spread — 近远月波动差
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_term_vol_spread_20d", category="intraday_advanced")
class IntradayTermVolSpread20d(Factor):
    """近远月波动差因子.

    近月分钟收益波动率 - 远月分钟收益波动率 (借鉴 #80 分段波动比, 用于期限结构).
    近月波动高 → 现货端不确定性大 → 负向.
    方向: 负向.
    """
    name = "intraday_term_vol_spread_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "近远月波动差 (near_vol-far_vol, 近月波动高=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_term_structure_panel(data, dates, universe, freq="1min")
        if "near_close" not in panel or "far_close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        near, far = panel["near_close"], panel["far_close"]
        ret_n, ret_f = near.pct_change(), far.pct_change()
        day = ret_n.index.normalize()
        spreads: dict = {}
        for dt in sorted(set(day)):
            grp_n = ret_n.loc[day == dt]
            grp_f = ret_f.loc[day == dt]
            if len(grp_n) < 30:
                continue
            vals = {}
            for col in grp_n.columns:
                rn = grp_n[col].dropna()
                rf = grp_f[col].dropna()
                common = rn.index.intersection(rf.index)
                if len(common) < 30:
                    continue
                vol_n = rn.loc[common].std(ddof=0)
                vol_f = rf.loc[common].std(ddof=0)
                vals[col] = -float(vol_n - vol_f)
            if vals:
                spreads[dt] = pd.Series(vals)
        if not spreads:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(spreads).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 230. intraday_term_breakout — 价差突破
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_term_breakout_20d", category="intraday_advanced")
class IntradayTermBreakout20d(Factor):
    """价差突破因子.

    今日价差幅度 / 近10日均值 (借鉴 #116 跨日波动突破, 用于期限价差).
    价差异常放大 → 期限结构剧变 → 不稳定 → 负向.
    方向: 负向.
    """
    name = "intraday_term_breakout_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "价差突破 (今日价差/近10日均值)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_term_structure_panel(data, dates, universe, freq="1min")
        if "near_close" not in panel or "far_close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        near, far = panel["near_close"], panel["far_close"]
        day = near.index.normalize()
        daily_spread: dict = {}
        for dt in sorted(set(day)):
            grp_n = near.loc[day == dt]
            grp_f = far.loc[day == dt]
            if len(grp_n) < 20:
                continue
            vals = {}
            for col in grp_n.columns:
                n = grp_n[col].dropna()
                f = grp_f[col].dropna()
                common = n.index.intersection(f.index)
                if len(common) < 20:
                    continue
                spread = (f.loc[common] - n.loc[common]).abs()
                vals[col] = float(spread.mean())
            if vals:
                daily_spread[dt] = pd.Series(vals)
        if not daily_spread:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily_spread).T
        df.index = pd.DatetimeIndex(df.index)
        df = df.reindex(dates)
        ma = df.rolling(10, min_periods=3).mean()
        ratio = df / ma.replace(0, np.nan)
        return (-ratio).reindex(columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 231. intraday_term_reversion — 价差均值回归偏离
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_term_reversion_20d", category="intraday_advanced")
class IntradayTermReversion20d(Factor):
    """价差均值回归偏离因子.

    |今日价差 - 近20日均价差| / 近20日价差std (z-score 偏离, 分位数/标准化构造).
    偏离大 → 价差过度拉伸 → 回归压力 → 负向.
    方向: 负向.
    """
    name = "intraday_term_reversion_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "价差回归偏离 (|价差-20日均|/std, 拉伸=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_term_structure_panel(data, dates, universe, freq="1min")
        if "near_close" not in panel or "far_close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        near, far = panel["near_close"], panel["far_close"]
        day = near.index.normalize()
        daily_spread: dict = {}
        for dt in sorted(set(day)):
            grp_n = near.loc[day == dt]
            grp_f = far.loc[day == dt]
            if len(grp_n) < 20:
                continue
            vals = {}
            for col in grp_n.columns:
                n = grp_n[col].dropna()
                f = grp_f[col].dropna()
                common = n.index.intersection(f.index)
                if len(common) < 20:
                    continue
                spread = f.loc[common] - n.loc[common]
                vals[col] = float(spread.mean())
            if vals:
                daily_spread[dt] = pd.Series(vals)
        if not daily_spread:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily_spread).T
        df.index = pd.DatetimeIndex(df.index)
        df = df.reindex(dates)
        ma20 = df.rolling(20, min_periods=5).mean()
        std20 = df.rolling(20, min_periods=5).std(ddof=0)
        z = (df - ma20) / std20.replace(0, np.nan)
        return (-z.abs()).reindex(columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 232. intraday_oi_log_change_vol — 对数持仓变动波动
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_log_change_vol_20d", category="intraday_advanced")
class IntradayOiLogChangeVol20d(Factor):
    """对数持仓变动波动因子.

    log(OI_t / OI_{t-1}) 的日内标准差 (对数变换处理持仓变动, 去除量纲).
    对数持仓变动波动大 → 资金进出剧烈 → 负向.
    方向: 负向.
    """
    name = "intraday_oi_log_change_vol_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "对数持仓变动波动 (std(logΔOI), 剧烈=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        position = panel["position"]
        log_oi = np.log(position.replace(0, np.nan))
        log_chg = log_oi.diff()
        day = log_chg.index.normalize()
        vols: dict = {}
        for dt in sorted(set(day)):
            grp = log_chg.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                d = grp[col].dropna()
                if len(d) < 30:
                    continue
                vals[col] = -float(d.std(ddof=0))
            if vals:
                vols[dt] = pd.Series(vals)
        if not vols:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(vols).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 233. intraday_oi_ma_cross — 持仓均线交叉
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_ma_cross_20d", category="intraday_advanced")
class IntradayOiMaCross20d(Factor):
    """持仓均线交叉因子.

    (OI_ma5 - OI_ma20) / OI_ma20 — 持仓量的短期/长期均线差 (借鉴 #117 均线交叉, 迁移到 OI).
    快线在慢线上方 → 持仓加速上升 → 资金流入 → 正向.
    方向: 正向.
    """
    name = "intraday_oi_ma_cross_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "持仓均线交叉 ((OI_ma5-OI_ma20)/OI_ma20)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        position = panel["position"]
        day = position.index.normalize()
        crosses: dict = {}
        for dt in sorted(set(day)):
            grp = position.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                p = grp[col].dropna()
                if len(p) < 30:
                    continue
                ma5 = p.rolling(5).mean().iloc[-1]
                ma20 = p.rolling(20).mean().iloc[-1]
                if ma20 is None or abs(ma20) < 1e-12:
                    vals[col] = 0.0
                else:
                    vals[col] = float((ma5 - ma20) / abs(ma20))
            if vals:
                crosses[dt] = pd.Series(vals)
        if not crosses:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(crosses).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 234. intraday_oi_quantile_range — 持仓分位数跨度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_quantile_range_20d", category="intraday_advanced")
class IntradayOiQuantileRange20d(Factor):
    """持仓分位数跨度因子.

    (p90 - p10) / median — 持仓分布的分位数跨度 (分位数构造, 稳健于极端值).
    跨度大 → 持仓水平剧烈摆动 → 不稳定 → 负向.
    方向: 负向.
    """
    name = "intraday_oi_quantile_range_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "持仓分位数跨度 ((p90-p10)/median, 摆动大=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        position = panel["position"]
        day = position.index.normalize()
        ranges: dict = {}
        for dt in sorted(set(day)):
            grp = position.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                p = grp[col].dropna()
                if len(p) < 30:
                    continue
                p10, p50, p90 = np.percentile(p.values, [10, 50, 90])
                vals[col] = -float((p90 - p10) / p50) if p50 > 1e-12 else 0.0
            if vals:
                ranges[dt] = pd.Series(vals)
        if not ranges:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(ranges).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 235. intraday_oi_vol_price_corr — 持仓-波动相关
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_vol_price_corr_20d", category="intraday_advanced")
class IntradayOiVolPriceCorr20d(Factor):
    """持仓-波动相关因子.

    corr(ΔOI_1m, |ret_1m|) — 持仓变化与价格波动的相关 (借鉴 #24 振幅-量相关构造).
    增仓伴随高波动 → 信息驱动的主动建仓 → 正向.
    方向: 正向.
    """
    name = "intraday_oi_vol_price_corr_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "持仓-波动相关 (corr(ΔOI,|ret|), 信息建仓=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel or "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, position = panel["close"], panel["position"]
        ret_abs = close.pct_change().abs()
        oi_change = position.diff()
        day = ret_abs.index.normalize()
        corrs: dict = {}
        for dt in sorted(set(day)):
            grp_r = ret_abs.loc[day == dt]
            grp_o = oi_change.loc[day == dt]
            if len(grp_r) < 30:
                continue
            vals = {}
            for col in grp_r.columns:
                r = grp_r[col].dropna()
                o = grp_o[col].dropna()
                common = r.index.intersection(o.index)
                if len(common) < 30:
                    continue
                r_c = r.loc[common]
                o_c = o.loc[common]
                if r_c.std(ddof=0) < 1e-12 or o_c.std(ddof=0) < 1e-12:
                    vals[col] = 0.0
                else:
                    corr_val = float(np.corrcoef(r_c, o_c)[0, 1])
                    vals[col] = corr_val if not np.isnan(corr_val) else 0.0
            if vals:
                corrs[dt] = pd.Series(vals)
        if not corrs:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(corrs).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 236. intraday_oi_momentum — 持仓跨日动量
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_momentum_20d", category="intraday_advanced")
class IntradayOiMomentum20d(Factor):
    """持仓跨日动量因子.

    今日平均持仓 / 昨日平均持仓 (借鉴 #145 volume_momentum 的跨日构造, 迁移到 OI).
    持仓环比放大 → 资金持续流入 → 正向.
    方向: 正向.
    """
    name = "intraday_oi_momentum_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "持仓跨日动量 (今日OI/昨日OI)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        position = panel["position"]
        day = position.index.normalize()
        daily_oi: dict = {}
        for dt in sorted(set(day)):
            grp = position.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                p = grp[col].dropna()
                if len(p) < 20:
                    continue
                vals[col] = float(p.mean())
            if vals:
                daily_oi[dt] = pd.Series(vals)
        if not daily_oi:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily_oi).T
        df.index = pd.DatetimeIndex(df.index)
        ratio = df / df.shift(1).replace(0, np.nan)
        return ratio.reindex(index=dates, columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 237. intraday_price_oi_lead — 价格-持仓领先滞后
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_price_oi_lead_20d", category="intraday_advanced")
class IntradayPriceOiLead20d(Factor):
    """价格-持仓领先滞后因子.

    corr(ret_t, ΔOI_{t+1}) - corr(ret_{t+1}, ΔOI_t) — 价格领先持仓还是反之 (领先滞后分析).
    正值 → 价格先动、持仓后跟 → 价格领先定价 → 正向.
    方向: 正向.
    """
    name = "intraday_price_oi_lead_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "价格-持仓领先滞后 (价领先=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel or "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, position = panel["close"], panel["position"]
        ret_1m = close.pct_change()
        oi_change = position.diff()
        day = ret_1m.index.normalize()
        leads: dict = {}
        for dt in sorted(set(day)):
            grp_r = ret_1m.loc[day == dt]
            grp_o = oi_change.loc[day == dt]
            if len(grp_r) < 40:
                continue
            vals = {}
            for col in grp_r.columns:
                r = grp_r[col].dropna()
                o = grp_o[col].dropna()
                common = r.index.intersection(o.index)
                if len(common) < 40:
                    continue
                r_c = r.loc[common].values
                o_c = o.loc[common].values
                # 价格领先: corr(ret_t, ΔOI_{t+1}) 用 t=0..n-2
                r_t, o_next = r_c[:-1], o_c[1:]
                # 持仓领先: corr(ret_{t+1}, ΔOI_t)
                r_next, o_t = r_c[1:], o_c[:-1]
                c1 = float(np.corrcoef(r_t, o_next)[0, 1]) if r_t.std() > 1e-12 and o_next.std() > 1e-12 else 0.0
                c2 = float(np.corrcoef(r_next, o_t)[0, 1]) if r_next.std() > 1e-12 and o_t.std() > 1e-12 else 0.0
                vals[col] = c1 - c2 if not np.isnan(c1) and not np.isnan(c2) else 0.0
            if vals:
                leads[dt] = pd.Series(vals)
        if not leads:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(leads).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 238. intraday_vol_oi_corr — 波动-持仓水平相关
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_vol_oi_corr_20d", category="intraday_advanced")
class IntradayVolOiCorr20d(Factor):
    """波动-持仓水平相关因子.

    corr(|ret_1m|, OI_1m) — 波动与持仓水平的日内相关.
    高波动伴随高持仓 → 多空分歧随持仓扩大 → 负向.
    方向: 负向.
    """
    name = "intraday_vol_oi_corr_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "波动-持仓相关 (corr(|ret|,OI), 分歧扩大=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel or "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, position = panel["close"], panel["position"]
        ret_abs = close.pct_change().abs()
        day = ret_abs.index.normalize()
        corrs: dict = {}
        for dt in sorted(set(day)):
            grp_r = ret_abs.loc[day == dt]
            grp_p = position.loc[day == dt]
            if len(grp_r) < 30:
                continue
            vals = {}
            for col in grp_r.columns:
                r = grp_r[col].dropna()
                p = grp_p[col].dropna()
                common = r.index.intersection(p.index)
                if len(common) < 30:
                    continue
                r_c = r.loc[common]
                p_c = p.loc[common]
                if r_c.std(ddof=0) < 1e-12 or p_c.std(ddof=0) < 1e-12:
                    vals[col] = 0.0
                else:
                    corr_val = float(np.corrcoef(r_c, p_c)[0, 1])
                    vals[col] = -corr_val if not np.isnan(corr_val) else 0.0
            if vals:
                corrs[dt] = pd.Series(vals)
        if not corrs:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(corrs).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 239. intraday_oi_volume_trend_ratio — 持仓-量比趋势
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_volume_trend_ratio_20d", category="intraday_advanced")
class IntradayOiVolumeTrendRatio20d(Factor):
    """持仓-量比趋势因子.

    (OI/vol) 的 MA5 与 MA20 之差 — 换手率(vol/OI)的倒数的趋势 (均线构造, 多数据组合).
    比值上升 → 持仓相对成交量增加 → 长线资金沉淀 → 正向.
    方向: 正向.
    """
    name = "intraday_oi_volume_trend_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "持仓-量比趋势 (OI/vol的MA5-MA20)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "volume" not in panel or "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        volume, position = panel["volume"], panel["position"]
        ratio = position / volume.replace(0, np.nan)
        day = ratio.index.normalize()
        daily_ratio: dict = {}
        for dt in sorted(set(day)):
            grp = ratio.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 20:
                    continue
                vals[col] = float(r.mean())
            if vals:
                daily_ratio[dt] = pd.Series(vals)
        if not daily_ratio:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily_ratio).T
        df.index = pd.DatetimeIndex(df.index)
        ma5 = df.rolling(5, min_periods=3).mean()
        ma20 = df.rolling(20, min_periods=5).mean()
        diff = ma5 - ma20
        return diff.reindex(index=dates, columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 240. intraday_settle_vol_ratio — 结算收敛度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_settle_vol_ratio_20d", category="intraday_advanced")
class IntradaySettleVolRatio20d(Factor):
    """结算收敛度因子.

    |close - settle| / (high - low) — 收盘相对结算价的收敛程度 (借鉴 #95 份额构造, 用于结算价).
    收盘贴近结算 → 市场认可当日定价 → 正向. 需 settle 数据.
    方向: 正向.
    """
    name = "intraday_settle_vol_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "结算收敛度 (-|close-settle|/(high-low), 需settle)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        settle = _read_local_daily(data, dates, universe, "settle")
        high = data.get("high", dates, universe)
        low = data.get("low", dates, universe)
        if close is None or settle is None or high is None or low is None:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        if close.empty or settle.empty or high.empty or low.empty:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        rng = (high - low).replace(0, np.nan)
        ratio = (close - settle).abs() / rng
        return (-ratio).reindex(index=dates, columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 251. intraday_oi_jump_intensity — 持仓突变强度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_jump_intensity_20d", category="intraday_advanced")
class IntradayOiJumpIntensity20d(Factor):
    """持仓突变强度因子.

    |ΔOI_t| 相对其历史波动的异常度: mean(|ΔOI|) / std(|ΔOI|) (借鉴 #8 jump_intensity 的异常检测思想).
    资金异常进出 → 信息驱动 → 正向 (区别于 #222 突变频率: 这里看强度).
    方向: 正向.
    """
    name = "intraday_oi_jump_intensity_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "持仓突变强度 (mean|ΔOI|/std|ΔOI|, 资金异常=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        oi_change = panel["position"].diff().abs()
        day = oi_change.index.normalize()
        intensities: dict = {}
        for dt in sorted(set(day)):
            grp = oi_change.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                d = grp[col].dropna()
                if len(d) < 30:
                    continue
                m, s = d.mean(), d.std(ddof=0)
                vals[col] = float(m / s) if s > 1e-12 else 0.0
            if vals:
                intensities[dt] = pd.Series(vals)
        if not intensities:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(intensities).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 252. intraday_oi_peak_ridge_ratio — 增仓峰岭比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_peak_ridge_ratio_20d", category="intraday_advanced")
class IntradayOiPeakRidgeRatio20d(Factor):
    """增仓峰岭比因子.

    孤立增仓脉冲(峰: ΔOI>μ+σ 且前后正常) vs 持续增仓(岭: 连续>μ) 的变化量比 (借鉴 #10 峰岭构造).
    峰多 → 信息脉冲式进场 → 正向.
    方向: 正向.
    """
    name = "intraday_oi_peak_ridge_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "增仓峰岭比 (孤立脉冲/持续增仓, 信息脉冲=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        oi_change = panel["position"].diff().abs()
        day = oi_change.index.normalize()
        ratios: dict = {}
        for dt in sorted(set(day)):
            grp = oi_change.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                d = grp[col].dropna()
                if len(d) < 20:
                    continue
                mu, sigma = d.mean(), d.std(ddof=0)
                if sigma < 1e-12:
                    vals[col] = 0.0
                    continue
                prev = d.shift(1)
                nxt = d.shift(-1)
                is_peak = (d > mu + sigma) & (prev < mu + sigma) & (nxt < mu + sigma)
                is_ridge = (d > mu) & ((prev > mu) | (nxt > mu))
                peak_sum = d[is_peak].sum()
                ridge_sum = d[is_ridge].sum()
                vals[col] = float(peak_sum / ridge_sum) if ridge_sum > 1e-12 else 0.0
            if vals:
                ratios[dt] = pd.Series(vals)
        if not ratios:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(ratios).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 253. intraday_oi_blowup_position — 增仓的价格位置
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_blowup_position_20d", category="intraday_advanced")
class IntradayOiBlowupPosition20d(Factor):
    """增仓的价格位置因子.

    大额增仓分钟(|ΔOI|>2σ)的价格在日内区间的位置 (借鉴 #11 blowup_position 构造).
    高位增仓 → 多方主动追进 → 正向; 低位增仓 → 抄底/空头回补 → 中性.
    方向: 正向.
    """
    name = "intraday_oi_blowup_position_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "增仓价格位置 (大额增仓时的价格区间位置, 高位增仓=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"high", "low", "close", "position"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, low, close, position = panel["high"], panel["low"], panel["close"], panel["position"]
        oi_change = position.diff().abs()
        day = oi_change.index.normalize()
        positions: dict = {}
        for dt in sorted(set(day)):
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            grp_c = close.loc[day == dt]
            grp_o = oi_change.loc[day == dt]
            if len(grp_c) < 30:
                continue
            vals = {}
            for col in grp_c.columns:
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                c = grp_c[col].dropna()
                o = grp_o[col].dropna()
                common = h.index.intersection(l.index).intersection(c.index).intersection(o.index)
                if len(common) < 30:
                    continue
                h_c, l_c, c_c, o_c = (x.loc[common] for x in (h, l, c, o))
                rng = h_c.max() - l_c.min()
                sigma = o_c.std(ddof=0)
                if rng < 1e-12 or sigma < 1e-12:
                    vals[col] = 0.5
                    continue
                blow_mask = o_c > o_c.mean() + 2.0 * sigma
                if blow_mask.sum() < 2:
                    vals[col] = 0.5
                    continue
                pos = (c_c[blow_mask] - l_c.min()) / rng
                vals[col] = float(pos.mean())
            if vals:
                positions[dt] = pd.Series(vals)
        if not positions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(positions).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 254. intraday_oi_torrent — 放量下跌增仓
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_torrent_20d", category="intraday_advanced")
class IntradayOiTorrent20d(Factor):
    """放量下跌增仓因子.

    价格下跌且放量时 OI 增仓的强度 (借鉴 #14 torrent 构造, 用于 OI).
    跌+放量+增仓 → 空头主动加仓 → 趋势延续看空 → 负向.
    方向: 负向.
    """
    name = "intraday_oi_torrent_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "放量下跌增仓 (跌+放量+增仓=空头加仓=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "volume", "position"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, volume, position = panel["close"], panel["volume"], panel["position"]
        ret_1m = close.pct_change()
        oi_change = position.diff()
        day = ret_1m.index.normalize()
        torrents: dict = {}
        for dt in sorted(set(day)):
            grp_r = ret_1m.loc[day == dt]
            grp_v = volume.loc[day == dt]
            grp_o = oi_change.loc[day == dt]
            if len(grp_r) < 30:
                continue
            vals = {}
            for col in grp_r.columns:
                r = grp_r[col].dropna()
                v = grp_v[col].dropna()
                o = grp_o[col].dropna()
                common = r.index.intersection(v.index).intersection(o.index)
                if len(common) < 30:
                    continue
                r_c = r.loc[common]
                v_c = v.loc[common]
                o_c = o.loc[common]
                v_mean = v_c.mean()
                if v_mean < 1e-12:
                    continue
                torrent_mask = (r_c < 0) & (v_c > v_mean) & (o_c > 0)
                if torrent_mask.sum() < 2:
                    vals[col] = 0.0
                    continue
                # 强度: 放量下跌增仓分钟的平均跌幅
                vals[col] = -float(r_c[torrent_mask].mean())
            if vals:
                torrents[dt] = pd.Series(vals)
        if not torrents:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(torrents).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 255. intraday_oi_herding — 持仓跟随强度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_herding_20d", category="intraday_advanced")
class IntradayOiHerding20d(Factor):
    """持仓跟随强度因子.

    价格同向运动时的增仓占比 (借鉴 #7 herding 构造, 用于 OI).
    涨时增仓且跌时减仓 → 顺势持仓 → 健康; 反之逆势 → 分歧. 输出顺势占比.
    方向: 正向.
    """
    name = "intraday_oi_herding_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "持仓跟随 (顺价格方向的增仓占比, 顺势=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "position"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, position = panel["close"], panel["position"]
        ret_1m = close.pct_change()
        oi_change = position.diff()
        day = ret_1m.index.normalize()
        herdings: dict = {}
        for dt in sorted(set(day)):
            grp_r = ret_1m.loc[day == dt]
            grp_o = oi_change.loc[day == dt]
            if len(grp_r) < 20:
                continue
            vals = {}
            for col in grp_r.columns:
                r = grp_r[col].dropna()
                o = grp_o[col].dropna()
                common = r.index.intersection(o.index)
                if len(common) < 20:
                    continue
                r_c = r.loc[common]
                o_c = o.loc[common]
                # 顺势: 涨时增仓 或 跌时减仓
                follow = ((r_c > 0) & (o_c > 0)) | ((r_c < 0) & (o_c < 0))
                valid = (r_c != 0) & (o_c != 0)
                if valid.sum() < 10:
                    vals[col] = 0.0
                    continue
                vals[col] = float(follow[valid].sum() / valid.sum())
            if vals:
                herdings[dt] = pd.Series(vals)
        if not herdings:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(herdings).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 256. intraday_oi_peak_count — 增仓脉冲计数
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_peak_count_20d", category="intraday_advanced")
class IntradayOiPeakCount20d(Factor):
    """增仓脉冲计数因子.

    孤立增仓脉冲(|ΔOI|>μ+σ 且前后非脉冲)的分钟计数 (借鉴 #13 price_peak_count 构造).
    脉冲多 → 资金反复突击 → 信息活跃 → 正向.
    方向: 正向.
    """
    name = "intraday_oi_peak_count_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "增仓脉冲计数 (孤立|ΔOI|>μ+σ分钟数)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        oi_change = panel["position"].diff().abs()
        day = oi_change.index.normalize()
        counts: dict = {}
        for dt in sorted(set(day)):
            grp = oi_change.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                d = grp[col].dropna()
                if len(d) < 30:
                    continue
                mu, sigma = d.mean(), d.std(ddof=0)
                if sigma < 1e-12:
                    vals[col] = 0.0
                    continue
                is_jump = d > mu + sigma
                cnt = 0
                for i in range(1, len(d) - 1):
                    if not is_jump.iloc[i]:
                        continue
                    if is_jump.iloc[i - 1] and is_jump.iloc[i + 1]:
                        continue
                    cnt += 1
                vals[col] = float(cnt)
            if vals:
                counts[dt] = pd.Series(vals)
        if not counts:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(counts).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=3).sum().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 257. intraday_oi_skew_stability — 分段增仓偏度一致性
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_skew_stability_20d", category="intraday_advanced")
class IntradayOiSkewStability20d(Factor):
    """分段增仓偏度一致性因子.

    日内4等分段 ΔOI 偏度的符号一致性 (借鉴 #178 skew_stability 构造, 用于 OI).
    全段同向偏 → 资金行为结构稳定 → 正向.
    方向: 正向.
    """
    name = "intraday_oi_skew_stability_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "分段增仓偏度一致 (4段ΔOI偏度同向数)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        oi_change = panel["position"].diff()
        day = oi_change.index.normalize()
        stabilities: dict = {}
        for dt in sorted(set(day)):
            grp = oi_change.loc[day == dt]
            n = len(grp)
            if n < 60:
                continue
            q = n // 4
            if q < 8:
                continue
            vals = {}
            for col in grp.columns:
                d = grp[col].dropna()
                if len(d) < 60:
                    continue
                segs = [d.iloc[i * q:(i + 1) * q] for i in range(4)]
                signs = []
                for s in segs:
                    if len(s) < 5 or s.std(ddof=0) < 1e-12:
                        continue
                    sk = float(pd.Series(s).skew())
                    if abs(sk) > 0.05:
                        signs.append(np.sign(sk))
                if not signs:
                    vals[col] = 0.0
                    continue
                dom = np.sign(sum(signs))
                consistent = sum(1 for sg in signs if sg == dom)
                vals[col] = float(consistent / len(signs))
            if vals:
                stabilities[dt] = pd.Series(vals)
        if not stabilities:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(stabilities).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 258. intraday_oi_trend_follow — 持仓趋势跟随得分
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_trend_follow_20d", category="intraday_advanced")
class IntradayOiTrendFollow20d(Factor):
    """持仓趋势跟随得分因子.

    与主导增仓方向一致的 ΔOI 之和 / Σ|ΔOI| (借鉴 #168 trend_follow_score 构造, 用于 OI).
    高得分 → 增仓方向高度一致 → 资金共识 → 正向.
    方向: 正向.
    """
    name = "intraday_oi_trend_follow_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "持仓趋势跟随 (顺主导增仓方向的ΔOI占比)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        oi_change = panel["position"].diff()
        day = oi_change.index.normalize()
        scores: dict = {}
        for dt in sorted(set(day)):
            grp = oi_change.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                d = grp[col].dropna()
                if len(d) < 20:
                    continue
                dom = np.sign(d.sum())
                total_abs = d.abs().sum()
                if abs(dom) < 1e-12 or total_abs < 1e-12:
                    vals[col] = 0.0
                    continue
                aligned = (d * dom).clip(lower=0).sum()
                vals[col] = float(aligned / total_abs)
            if vals:
                scores[dt] = pd.Series(vals)
        if not scores:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(scores).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 259. intraday_oi_extreme_timing — 持仓极端时间位置
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_extreme_timing_20d", category="intraday_advanced")
class IntradayOiExtremeTiming20d(Factor):
    """持仓极端时间位置因子.

    最大|ΔOI|分钟出现的时间位置 (借鉴 #188 extreme_timing 构造, 用于 OI).
    资金大动作偏早盘 → 已释放 → 正向. 输出 -时间位置.
    方向: 正向.
    """
    name = "intraday_oi_extreme_timing_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "持仓极端时间 (-最大|ΔOI|时间位置, 早=已释放=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        oi_change = panel["position"].diff().abs()
        day = oi_change.index.normalize()
        timings: dict = {}
        for dt in sorted(set(day)):
            grp = oi_change.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                d = grp[col].dropna()
                if len(d) < 20:
                    continue
                idx = int(np.argmax(d.values))
                vals[col] = -float(idx / max(1, len(d) - 1))
            if vals:
                timings[dt] = pd.Series(vals)
        if not timings:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(timings).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 260. intraday_oi_range_position — 增仓价格区间位置
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_range_position_20d", category="intraday_advanced")
class IntradayOiRangePosition20d(Factor):
    """增仓价格区间位置因子.

    增仓分钟的价格在日内区间的平均位置 (借鉴 #194 range_position_avg 构造, 加权 ΔOI).
    增仓偏高位 → 多方主动 → 正向.
    方向: 正向.
    """
    name = "intraday_oi_range_position_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "增仓价格区间位置 (ΔOI加权区间位置, 高位增仓=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"high", "low", "close", "position"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, low, close, position = panel["high"], panel["low"], panel["close"], panel["position"]
        oi_change = position.diff()
        day = oi_change.index.normalize()
        positions: dict = {}
        for dt in sorted(set(day)):
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            grp_c = close.loc[day == dt]
            grp_o = oi_change.loc[day == dt]
            if len(grp_c) < 30:
                continue
            vals = {}
            for col in grp_c.columns:
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                c = grp_c[col].dropna()
                o = grp_o[col].dropna()
                common = h.index.intersection(l.index).intersection(c.index).intersection(o.index)
                if len(common) < 30:
                    continue
                h_c, l_c, c_c, o_c = (x.loc[common] for x in (h, l, c, o))
                rng = h_c.max() - l_c.min()
                if rng < 1e-12:
                    vals[col] = 0.5
                    continue
                add_mask = o_c > 0
                if add_mask.sum() < 5:
                    vals[col] = 0.5
                    continue
                pos = (c_c[add_mask] - l_c.min()) / rng
                w = o_c[add_mask].abs()
                vals[col] = float((pos * w).sum() / w.sum()) if w.sum() > 1e-12 else 0.5
            if vals:
                positions[dt] = pd.Series(vals)
        if not positions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(positions).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 261. intraday_term_skewness — 价差分布偏度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_term_skewness_20d", category="intraday_advanced")
class IntradayTermSkewness20d(Factor):
    """价差分布偏度因子.

    日内 (far-near) 价差序列的偏度 (借鉴 #4 realised_skewness 构造, 用于期限价差).
    正偏 → 价差偶发大幅走阔(远月急拉) → 升水脉冲 → 负向.
    方向: 负向.
    """
    name = "intraday_term_skewness_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "价差分布偏度 (spread偏度, 升水脉冲=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_term_structure_panel(data, dates, universe, freq="1min")
        if "near_close" not in panel or "far_close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        spread = panel["far_close"] - panel["near_close"]
        day = spread.index.normalize()
        skews: dict = {}
        for dt in sorted(set(day)):
            grp = spread.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                s = grp[col].dropna()
                if len(s) < 30:
                    continue
                vals[col] = -float(pd.Series(s).skew())  # 正偏=负向
            if vals:
                skews[dt] = pd.Series(vals)
        if not skews:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(skews).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 262. intraday_term_jump_intensity — 价差突变强度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_term_jump_intensity_20d", category="intraday_advanced")
class IntradayTermJumpIntensity20d(Factor):
    """价差突变强度因子.

    |Δspread| 相对其历史波动的异常度 (借鉴 #8 jump_intensity 构造, 用于期限价差).
    价差突变 → 期限结构剧变 → 不确定 → 负向.
    方向: 负向.
    """
    name = "intraday_term_jump_intensity_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "价差突变强度 (mean|Δspread|/std, 剧变=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_term_structure_panel(data, dates, universe, freq="1min")
        if "near_close" not in panel or "far_close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        spread = panel["far_close"] - panel["near_close"]
        d_spread = spread.diff().abs()
        day = d_spread.index.normalize()
        intensities: dict = {}
        for dt in sorted(set(day)):
            grp = d_spread.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                d = grp[col].dropna()
                if len(d) < 30:
                    continue
                m, s = d.mean(), d.std(ddof=0)
                vals[col] = -float(m / s) if s > 1e-12 else 0.0
            if vals:
                intensities[dt] = pd.Series(vals)
        if not intensities:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(intensities).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 263. intraday_term_peak_ridge_ratio — 价差峰岭比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_term_peak_ridge_ratio_20d", category="intraday_advanced")
class IntradayTermPeakRidgeRatio20d(Factor):
    """价差峰岭比因子.

    价差孤立脉冲(峰) vs 持续趋势(岭) 的变化量比 (借鉴 #10 peak_ridge 构造, 用于期限价差).
    峰多 → 价差受事件脉冲驱动 → 不稳定 → 负向.
    方向: 负向.
    """
    name = "intraday_term_peak_ridge_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "价差峰岭比 (孤立脉冲/持续趋势, 事件驱动=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_term_structure_panel(data, dates, universe, freq="1min")
        if "near_close" not in panel or "far_close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        spread = panel["far_close"] - panel["near_close"]
        d_spread = spread.diff().abs()
        day = d_spread.index.normalize()
        ratios: dict = {}
        for dt in sorted(set(day)):
            grp = d_spread.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                d = grp[col].dropna()
                if len(d) < 20:
                    continue
                mu, sigma = d.mean(), d.std(ddof=0)
                if sigma < 1e-12:
                    vals[col] = 0.0
                    continue
                prev = d.shift(1)
                nxt = d.shift(-1)
                is_peak = (d > mu + sigma) & (prev < mu + sigma) & (nxt < mu + sigma)
                is_ridge = (d > mu) & ((prev > mu) | (nxt > mu))
                peak_sum = d[is_peak].sum()
                ridge_sum = d[is_ridge].sum()
                vals[col] = -float(peak_sum / ridge_sum) if ridge_sum > 1e-12 else 0.0
            if vals:
                ratios[dt] = pd.Series(vals)
        if not ratios:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(ratios).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 264. intraday_term_trend_efficiency — 价差路径效率
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_term_trend_efficiency_20d", category="intraday_advanced")
class IntradayTermTrendEfficiency20d(Factor):
    """价差路径效率因子.

    |spread_end - spread_start| / Σ|Δspread| (借鉴 #3 trend_efficiency 构造, 用于期限价差).
    高效单边 → 价差趋势明确 → 正向.
    方向: 正向.
    """
    name = "intraday_term_trend_efficiency_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "价差路径效率 (|净位移|/Σ|Δspread|, 单边=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_term_structure_panel(data, dates, universe, freq="1min")
        if "near_close" not in panel or "far_close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        spread = panel["far_close"] - panel["near_close"]
        day = spread.index.normalize()
        efficiencies: dict = {}
        for dt in sorted(set(day)):
            grp = spread.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                s = grp[col].dropna()
                if len(s) < 20:
                    continue
                total_path = s.diff().abs().sum()
                if total_path < 1e-12:
                    vals[col] = 0.0
                    continue
                net = abs(s.iloc[-1] - s.iloc[0])
                vals[col] = float(net / total_path)
            if vals:
                efficiencies[dt] = pd.Series(vals)
        if not efficiencies:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(efficiencies).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 265. intraday_term_dtws — 价差走阔时间重心
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_term_dtws_20d", category="intraday_advanced")
class IntradayTermDtws20d(Factor):
    """价差走阔时间重心因子.

    价差走阔(|far-near|增大)分钟的时间加权重心 (借鉴 #9 dtws 构造, 用于期限价差).
    走阔偏尾盘 → 尾盘期限结构突变 → 负向.
    方向: 负向.
    """
    name = "intraday_term_dtws_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "价差走阔时间重心 (走阔偏尾盘=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_term_structure_panel(data, dates, universe, freq="1min")
        if "near_close" not in panel or "far_close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        spread = (panel["far_close"] - panel["near_close"]).abs()
        day = spread.index.normalize()
        dtws: dict = {}
        for dt in sorted(set(day)):
            grp = spread.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                s = grp[col].dropna()
                if len(s) < 30:
                    continue
                widen = s.diff()
                neg_mask = widen < 0  # 价差缩小视为"不利"? 这里用走阔方向: |spread|增大
                # 实际"走阔"=|spread|增大, 但 spread 本身可正可负; 用变化绝对值的符号
                change = s.diff().abs()
                if change.sum() < 1e-12:
                    vals[col] = 0.0
                    continue
                tw = np.arange(1, len(s) + 1) / len(s)
                vals[col] = -float((tw * change.values).sum() / change.sum())
            if vals:
                dtws[dt] = pd.Series(vals)
        if not dtws:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(dtws).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 266. intraday_term_vp_corr — 价差-成交量相关
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_term_vp_corr_20d", category="intraday_advanced")
class IntradayTermVpCorr20d(Factor):
    """价差-成交量相关因子.

    corr(Δspread, volume) (借鉴 #1 vp_corr 构造, 用于期限价差).
    价差变动伴随放量 → 期限结构由资金推动 → 正向.
    方向: 正向.
    """
    name = "intraday_term_vp_corr_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "价差-量相关 (corr(Δspread,vol), 资金推动=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_term_structure_panel(data, dates, universe, freq="1min")
        if "near_close" not in panel or "far_close" not in panel or "near_volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        spread = panel["far_close"] - panel["near_close"]
        volume = panel["near_volume"]
        d_spread = spread.diff()
        day = d_spread.index.normalize()
        corrs: dict = {}
        for dt in sorted(set(day)):
            grp_s = d_spread.loc[day == dt]
            grp_v = volume.loc[day == dt]
            if len(grp_s) < 30:
                continue
            vals = {}
            for col in grp_s.columns:
                s = grp_s[col].dropna()
                v = grp_v[col].dropna()
                common = s.index.intersection(v.index)
                if len(common) < 30:
                    continue
                s_c = s.loc[common]
                v_c = v.loc[common]
                if s_c.std(ddof=0) < 1e-12 or v_c.std(ddof=0) < 1e-12:
                    vals[col] = 0.0
                else:
                    corr_val = float(np.corrcoef(s_c, v_c)[0, 1])
                    vals[col] = corr_val if not np.isnan(corr_val) else 0.0
            if vals:
                corrs[dt] = pd.Series(vals)
        if not corrs:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(corrs).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 267. intraday_term_herding — 价差-持仓协同
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_term_herding_20d", category="intraday_advanced")
class IntradayTermHerding20d(Factor):
    """价差-持仓协同因子.

    价差走阔(远月走强)时近月持仓增加的比例 (借鉴 #7 herding 构造, 用于期限结构).
    升水伴随近月增仓 → 多空在近月博弈加剧 → 现货端关注 → 正向.
    方向: 正向.
    """
    name = "intraday_term_herding_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "价差-持仓协同 (升水+近月增仓占比)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_term_structure_panel(data, dates, universe, freq="1min")
        if "near_close" not in panel or "far_close" not in panel or "near_position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        spread = panel["far_close"] - panel["near_close"]
        near_pos = panel["near_position"]
        d_spread = spread.diff()
        d_pos = near_pos.diff()
        day = d_spread.index.normalize()
        herdings: dict = {}
        for dt in sorted(set(day)):
            grp_s = d_spread.loc[day == dt]
            grp_p = d_pos.loc[day == dt]
            if len(grp_s) < 20:
                continue
            vals = {}
            for col in grp_s.columns:
                s = grp_s[col].dropna()
                p = grp_p[col].dropna()
                common = s.index.intersection(p.index)
                if len(common) < 20:
                    continue
                s_c = s.loc[common]
                p_c = p.loc[common]
                # 协同: 价差走阔(升水增)且近月增仓
                follow = ((s_c > 0) & (p_c > 0)) | ((s_c < 0) & (p_c < 0))
                valid = (s_c != 0) & (p_c != 0)
                if valid.sum() < 10:
                    vals[col] = 0.0
                    continue
                vals[col] = float(follow[valid].sum() / valid.sum())
            if vals:
                herdings[dt] = pd.Series(vals)
        if not herdings:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(herdings).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 268. intraday_term_peak_count — 价差脉冲计数
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_term_peak_count_20d", category="intraday_advanced")
class IntradayTermPeakCount20d(Factor):
    """价差脉冲计数因子.

    孤立价差脉冲(|Δspread|>μ+σ且前后非脉冲)的分钟计数 (借鉴 #13 price_peak_count 构造).
    脉冲多 → 期限结构频繁受事件冲击 → 不稳定 → 负向.
    方向: 负向.
    """
    name = "intraday_term_peak_count_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "价差脉冲计数 (孤立|Δspread|>μ+σ分钟数, 事件频繁=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_term_structure_panel(data, dates, universe, freq="1min")
        if "near_close" not in panel or "far_close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        spread = panel["far_close"] - panel["near_close"]
        d_spread = spread.diff().abs()
        day = d_spread.index.normalize()
        counts: dict = {}
        for dt in sorted(set(day)):
            grp = d_spread.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                d = grp[col].dropna()
                if len(d) < 30:
                    continue
                mu, sigma = d.mean(), d.std(ddof=0)
                if sigma < 1e-12:
                    vals[col] = 0.0
                    continue
                is_jump = d > mu + sigma
                cnt = 0
                for i in range(1, len(d) - 1):
                    if not is_jump.iloc[i]:
                        continue
                    if is_jump.iloc[i - 1] and is_jump.iloc[i + 1]:
                        continue
                    cnt += 1
                vals[col] = -float(cnt)
            if vals:
                counts[dt] = pd.Series(vals)
        if not counts:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(counts).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=3).sum().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 269. intraday_term_range_position — 价差区间位置
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_term_range_position_20d", category="intraday_advanced")
class IntradayTermRangePosition20d(Factor):
    """价差区间位置因子.

    价差在自身日内区间(min-max)的平均位置 (借鉴 #17/#194 区间位置构造, 用于期限价差).
    价差维持高位 → 升水/贴水方向持续 → 正向.
    方向: 正向.
    """
    name = "intraday_term_range_position_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "价差区间位置 (spread在日内区间平均位置)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_term_structure_panel(data, dates, universe, freq="1min")
        if "near_close" not in panel or "far_close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        spread = panel["far_close"] - panel["near_close"]
        day = spread.index.normalize()
        positions: dict = {}
        for dt in sorted(set(day)):
            grp = spread.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                s = grp[col].dropna()
                if len(s) < 20:
                    continue
                smin, smax = s.min(), s.max()
                rng = smax - smin
                if rng < 1e-12:
                    vals[col] = 0.5
                    continue
                vals[col] = float(((s - smin) / rng).mean())
            if vals:
                positions[dt] = pd.Series(vals)
        if not positions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(positions).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 270. intraday_term_quantile_skew — 价差分位数偏度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_term_quantile_skew_20d", category="intraday_advanced")
class IntradayTermQuantileSkew20d(Factor):
    """价差分位数偏度因子.

    (Q3+Q1-2Q2)/(Q3-Q1) — 价差分布的稳健分位数偏度 (借鉴 #179 构造, 用于期限价差).
    正偏 → 升水/贴水脉冲偶发 → 负向.
    方向: 负向.
    """
    name = "intraday_term_quantile_skew_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "价差分位数偏度 ((Q3+Q1-2Q2)/(Q3-Q1), 脉冲=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_term_structure_panel(data, dates, universe, freq="1min")
        if "near_close" not in panel or "far_close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        spread = panel["far_close"] - panel["near_close"]
        day = spread.index.normalize()
        qskews: dict = {}
        for dt in sorted(set(day)):
            grp = spread.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                s = grp[col].dropna().values
                if len(s) < 30:
                    continue
                q1, q2, q3 = np.percentile(s, [25, 50, 75])
                denom = q3 - q1
                vals[col] = -float((q3 + q1 - 2.0 * q2) / denom) if denom > 1e-12 else 0.0
            if vals:
                qskews[dt] = pd.Series(vals)
        if not qskews:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(qskews).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)

# ═══════════════════════════════════════════════════════════════════════════════
# 241. intraday_oi_price_trend_align — OI-价格趋势同向
#     sign(价格20日收益) × sign(OI 20日变化). 同向=持仓跟随趋势. 方向: 正向.
# ═══════════════════════════════════════════════════════════════════════════════
@register_factor("intraday_oi_price_trend_align_20d", category="intraday_advanced")
class IntradayOiPriceTrendAlign20d(Factor):
    """OI与价格变化方向一致性."""
    name = "intraday_oi_price_trend_align_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "OI与价格变化方向一致性"
    validation_horizons = (5, 10, 20)

    def dependencies(self):
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"].groupby(panel["close"].index.normalize()).last()
        oi = panel["position"].groupby(panel["position"].index.normalize()).last()
        p_ret = close.pct_change(20)
        oi_chg = oi.pct_change(20)
        score = np.sign(p_ret) * np.sign(oi_chg)
        return score.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 242. intraday_oi_surge_reversal_20d — OI 突增反转
#     OI 单日突增(>2σ)后 5 日价格反转. 增仓过热 → 反向. 方向: 负向.
# ═══════════════════════════════════════════════════════════════════════════════
@register_factor("intraday_oi_surge_reversal_20d", category="intraday_advanced")
class IntradayOiSurgeReversal20d(Factor):
    """OI突增后价格反转."""
    name = "intraday_oi_surge_reversal_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "OI突增后反转"
    validation_horizons = (5, 10, 20)

    def dependencies(self):
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        oi = panel["position"].groupby(panel["position"].index.normalize()).last()
        close = panel["close"].groupby(panel["close"].index.normalize()).last()
        oi_ret = oi.pct_change()
        oi_std = oi_ret.rolling(20, min_periods=10).std().replace(0, np.nan)
        surge = (oi_ret - oi_ret.rolling(20, min_periods=10).mean()) / oi_std
        fwd_ret = close.pct_change(5).shift(-5)
        score = (surge * fwd_ret).rolling(20, min_periods=5).mean()
        return score.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 243. intraday_oi_change_vol_ratio_20d — 增仓效率
#     OI 变化量 / 成交量. 高=小量增仓(持仓坚定). 方向: 正向.
# ═══════════════════════════════════════════════════════════════════════════════
@register_factor("intraday_oi_change_vol_ratio_20d", category="intraday_advanced")
class IntradayOiChangeVolRatio20d(Factor):
    """增仓效率: OI增量/成交量."""
    name = "intraday_oi_change_vol_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "增仓效率"
    validation_horizons = (5, 10, 20)

    def dependencies(self):
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "position" not in panel or "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        day = panel["position"].index.normalize()
        daily = {}
        for dt in sorted(set(day)):
            p = panel["position"].loc[day == dt]
            v = panel["volume"].loc[panel["volume"].index.normalize() == dt]
            vals = {}
            for col in p.columns:
                pc = p[col].dropna()
                vc = v[col].dropna() if col in v.columns else pd.Series(dtype=float)
                if len(pc) < 10 or len(vc) < 10:
                    continue
                vals[col] = (pc.iloc[-1] - pc.iloc[0]) / max(vc.sum(), 1e-9)
            if vals:
                daily[dt] = pd.Series(vals)
        if not daily:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily).T
        df.index = pd.DatetimeIndex(df.index)
        return _finalize(df, dates, universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 244. intraday_settle_close_basis_20d — 结算-收盘基差
#     (settle - close)/close. 结算偏离收盘 → 尾盘异常. 方向: 负向.
# ═══════════════════════════════════════════════════════════════════════════════
@register_factor("intraday_settle_close_basis_20d", category="intraday_advanced")
class IntradaySettleCloseBasis20d(Factor):
    """结算价与收盘价基差."""
    name = "intraday_settle_close_basis_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "结算-收盘基差"
    validation_horizons = (5, 10, 20)

    def dependencies(self):
        return []

    def compute(self, data, dates, universe):
        settle = _read_local_daily(data, dates, universe, "settle")
        close = data.get("close", dates, universe)
        if settle is None or settle.empty or close is None or close.empty:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        basis = (settle - close) / close.replace(0, np.nan)
        return basis.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 245. intraday_term_roll_yield_20d — 期限结构展期收益
#     (near - far)/far. 近月升水 → 展期正收益. 方向: 正向.
# ═══════════════════════════════════════════════════════════════════════════════
@register_factor("intraday_term_roll_yield_20d", category="intraday_advanced")
class IntradayTermRollYield20d(Factor):
    """期限结构展期收益."""
    name = "intraday_term_roll_yield_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "期限结构展期收益"
    validation_horizons = (5, 10, 20)

    def dependencies(self):
        return []

    def compute(self, data, dates, universe):
        panel = _get_term_structure_panel(data, dates, universe, freq="1min")
        if not panel or "near_close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        near = panel["near_close"].groupby(panel["near_close"].index.normalize()).last()
        far = panel["far_close"].groupby(panel["far_close"].index.normalize()).last()
        roll = (near - far) / far.replace(0, np.nan)
        return roll.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 246. intraday_term_position_ratio_20d — 主力/次主力持仓比
#     near_position / far_position. 方向: 正向.
# ═══════════════════════════════════════════════════════════════════════════════
@register_factor("intraday_term_position_ratio_20d", category="intraday_advanced")
class IntradayTermPositionRatio20d(Factor):
    """主力/次主力持仓比."""
    name = "intraday_term_position_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "主力/次主力持仓比"
    validation_horizons = (5, 10, 20)

    def dependencies(self):
        return []

    def compute(self, data, dates, universe):
        panel = _get_term_structure_panel(data, dates, universe, freq="1min")
        if not panel or "near_position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        np_ = panel["near_position"].groupby(panel["near_position"].index.normalize()).last()
        fp = panel["far_position"].groupby(panel["far_position"].index.normalize()).last()
        ratio = np_ / fp.replace(0, np.nan)
        return ratio.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 247. intraday_oi_vol_corr_daily_20d — OI-成交量滚动相关
#     corr(oi, volume) 20日. 高相关=增仓伴随放量. 方向: 正向.
# ═══════════════════════════════════════════════════════════════════════════════
@register_factor("intraday_oi_vol_corr_daily_20d", category="intraday_advanced")
class IntradayOiVolCorrDaily20d(Factor):
    """OI与成交量滚动相关."""
    name = "intraday_oi_vol_corr_daily_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "OI-成交量相关"
    validation_horizons = (5, 10, 20)

    def dependencies(self):
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "position" not in panel or "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        oi = panel["position"].groupby(panel["position"].index.normalize()).last()
        vol = panel["volume"].groupby(panel["volume"].index.normalize()).sum()
        corr = oi.rolling(20, min_periods=10).corr(vol)
        return corr.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 248. intraday_oi_trend_momentum_20d — OI 动量
#     OI 20日变化率. 持续增仓 → 持仓趋势. 方向: 正向.
# ═══════════════════════════════════════════════════════════════════════════════
@register_factor("intraday_oi_trend_momentum_20d", category="intraday_advanced")
class IntradayOiTrendMomentum20d(Factor):
    """OI 20日动量."""
    name = "intraday_oi_trend_momentum_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "OI动量"
    validation_horizons = (5, 10, 20)

    def dependencies(self):
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        oi = panel["position"].groupby(panel["position"].index.normalize()).last()
        mom = oi.pct_change(20)
        return mom.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 249. intraday_term_vol_ratio_20d — 期限结构波动比
#     near 波动 / far 波动. 近月高波动=投机过度. 方向: 负向.
# ═══════════════════════════════════════════════════════════════════════════════
@register_factor("intraday_term_vol_ratio_20d", category="intraday_advanced")
class IntradayTermVolRatio20d(Factor):
    """近/远月波动比."""
    name = "intraday_term_vol_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "近/远月波动比"
    validation_horizons = (5, 10, 20)

    def dependencies(self):
        return []

    def compute(self, data, dates, universe):
        panel = _get_term_structure_panel(data, dates, universe, freq="1min")
        if not panel or "near_close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        near = panel["near_close"].groupby(panel["near_close"].index.normalize()).last()
        far = panel["far_close"].groupby(panel["far_close"].index.normalize()).last()
        n_vol = near.pct_change().rolling(20, min_periods=10).std()
        f_vol = far.pct_change().rolling(20, min_periods=10).std()
        ratio = n_vol / f_vol.replace(0, np.nan)
        return ratio.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 250. intraday_oi_mean_reversion_20d — OI 均值回归
#     OI 偏离 20 日均值 (标准化). 高偏离 → 回归压力. 方向: 负向.
# ═══════════════════════════════════════════════════════════════════════════════
@register_factor("intraday_oi_mean_reversion_20d", category="intraday_advanced")
class IntradayOiMeanReversion20d(Factor):
    """OI 均值回归偏离度."""
    name = "intraday_oi_mean_reversion_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "OI偏离均线"
    validation_horizons = (5, 10, 20)

    def dependencies(self):
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        oi = panel["position"].groupby(panel["position"].index.normalize()).last()
        mean = oi.rolling(20, min_periods=10).mean()
        std = oi.rolling(20, min_periods=10).std().replace(0, np.nan)
        dev = (oi - mean) / std
        return dev.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 271. intraday_oi_surge_confirm — OI 突增放量确认
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_surge_confirm_20d", category="intraday_advanced")
class IntradayOiSurgeConfirm20d(Factor):
    """OI 突增放量确认因子.

    借鉴 #242 oi_surge_reversal 的 z-score 突增识别, 但叠加放量确认:
    OI 突增(z>2) 且放量(量>1.5x均量) 的日频信号占比.
    突增+放量 → 真实资金进场 (非对倒) → 正向.
    方向: 正向.
    """
    name = "intraday_oi_surge_confirm_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "OI突增放量确认 (z>2且量>1.5x均量占比)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "position" not in panel or "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        oi = panel["position"].groupby(panel["position"].index.normalize()).last()
        vol = panel["volume"].groupby(panel["volume"].index.normalize()).sum()
        oi_ret = oi.pct_change().replace([np.inf, -np.inf], np.nan)
        oi_std = oi_ret.rolling(20, min_periods=10).std().replace(0, np.nan)
        surge_z = (oi_ret - oi_ret.rolling(20, min_periods=10).mean()) / oi_std
        vol_ratio = vol / vol.rolling(20, min_periods=10).mean().replace(0, np.nan)
        confirm = ((surge_z > 2.0) & (vol_ratio > 1.5)).astype(float)
        score = confirm.rolling(20, min_periods=5).mean()
        return score.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 272. intraday_oi_reversion_vol — 波动加权 OI 均值回归
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_reversion_vol_20d", category="intraday_advanced")
class IntradayOiReversionVol20d(Factor):
    """波动加权 OI 均值回归因子.

    借鉴 #250 oi_mean_reversion 的 z-score 构造, 叠加波动率加权 (高波动时信号更可信).
    OI z-score × (1/波动率) — 偏离越远且环境越稳 → 回归压力越大 → 负向.
    方向: 负向.
    """
    name = "intraday_oi_reversion_vol_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "波动加权OI均值回归 (z×逆波动, 偏离=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "position" not in panel or "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        oi = panel["position"].groupby(panel["position"].index.normalize()).last()
        close = panel["close"].groupby(panel["close"].index.normalize()).last()
        mean = oi.rolling(20, min_periods=10).mean()
        std = oi.rolling(20, min_periods=10).std().replace(0, np.nan)
        z = (oi - mean) / std
        vol = close.pct_change().rolling(20, min_periods=10).std().replace(0, np.nan)
        score = z * (1.0 / vol)
        return score.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 273. intraday_settle_basis_momentum — 结算基差动量
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_settle_basis_momentum_20d", category="intraday_advanced")
class IntradaySettleBasisMomentum20d(Factor):
    """结算基差动量因子.

    借鉴 #244 settle_close_basis 的基差构造, 取 20 日动量 (基差的趋势).
    基差持续走扩 (结算持续强于收盘) → 定价结构变化 → 正向.
    方向: 正向.
    """
    name = "intraday_settle_basis_momentum_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "结算基差动量 (settle-close基差的20日变化)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        settle = _read_local_daily(data, dates, universe, "settle")
        close = data.get("close", dates, universe)
        if settle is None or settle.empty or close is None or close.empty:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        basis = (settle - close) / close.replace(0, np.nan)
        mom = basis - basis.shift(20)
        return mom.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 274. intraday_term_roll_yield_change — 展期收益变化
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_term_roll_yield_change_20d", category="intraday_advanced")
class IntradayTermRollYieldChange20d(Factor):
    """展期收益变化因子.

    借鉴 #245 term_roll_yield 的展期收益构造, 取其跨日差分 (展期收益的动量).
    展期收益走强 → 期限结构转紧 → 正向.
    方向: 正向.
    """
    name = "intraday_term_roll_yield_change_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "展期收益变化 (roll_yield跨日差分)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_term_structure_panel(data, dates, universe, freq="1min")
        if not panel or "near_close" not in panel or "far_close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        near = panel["near_close"].groupby(panel["near_close"].index.normalize()).last()
        far = panel["far_close"].groupby(panel["far_close"].index.normalize()).last()
        roll = (near - far) / far.replace(0, np.nan)
        chg = roll - roll.shift(1)
        return chg.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 275. intraday_term_spread_zscore — 价差 z-score 突破
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_term_spread_zscore_20d", category="intraday_advanced")
class IntradayTermSpreadZscore20d(Factor):
    """价差 z-score 突破因子.

    借鉴 #250 oi_mean_reversion 的 z-score 构造, 用于期限价差:
    (spread - 20日均) / 20日std. 价差大幅偏离 → 回归压力 → 负向.
    方向: 负向.
    """
    name = "intraday_term_spread_zscore_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "价差z-score (价差偏离20日均值)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_term_structure_panel(data, dates, universe, freq="1min")
        if not panel or "near_close" not in panel or "far_close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        near = panel["near_close"].groupby(panel["near_close"].index.normalize()).last()
        far = panel["far_close"].groupby(panel["far_close"].index.normalize()).last()
        spread = far - near
        mean = spread.rolling(20, min_periods=10).mean()
        std = spread.rolling(20, min_periods=10).std().replace(0, np.nan)
        z = (spread - mean) / std
        return z.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 276. intraday_oi_vol_corr_change — OI-量相关变化
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_vol_corr_change_20d", category="intraday_advanced")
class IntradayOiVolCorrChange20d(Factor):
    """OI-量相关变化因子.

    借鉴 #247 oi_vol_corr_daily 的滚动相关构造, 取相关性的跨日差分 (相关稳定性).
    相关上升 → 量价持仓更同步 → 正向.
    方向: 正向.
    """
    name = "intraday_oi_vol_corr_change_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "OI-量相关变化 (corr(oi,vol)跨日差分)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "position" not in panel or "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        oi = panel["position"].groupby(panel["position"].index.normalize()).last()
        vol = panel["volume"].groupby(panel["volume"].index.normalize()).sum()
        corr = oi.rolling(20, min_periods=10).corr(vol)
        chg = corr - corr.shift(1)
        return chg.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 277. intraday_settle_oi_signal — 结算-OI-价格三因素一致
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_settle_oi_signal_20d", category="intraday_advanced")
class IntradaySettleOiSignal20d(Factor):
    """结算-OI-价格三因素一致因子.

    借鉴 #241 oi_price_trend_align 的方向一致性打分, 扩展为三因素:
    结算价方向 × OI 方向 × 价格方向 的一致性占比 (同向数/3).
    三因素同向 → 资金/定价/价格共振 → 正向.
    方向: 正向.
    """
    name = "intraday_settle_oi_signal_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "结算-OI-价三因素一致 (同向数/3)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        settle = _read_local_daily(data, dates, universe, "settle")
        oi = _read_local_daily(data, dates, universe, "oi")
        close = data.get("close", dates, universe)
        if settle is None or oi is None or close is None:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        if settle.empty or oi.empty or close.empty:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        s_dir = np.sign(settle.diff())
        o_dir = np.sign(oi.diff())
        c_dir = np.sign(close.diff())
        agree = (s_dir > 0).astype(float) + (o_dir > 0).astype(float) + (c_dir > 0).astype(float)
        score = (agree - 1.5) / 1.5  # 归一化到 [-1, 1]
        return score.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 278. intraday_oi_herding_z — OI 变化 z-score 羊群
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_herding_z_20d", category="intraday_advanced")
class IntradayOiHerdingZ20d(Factor):
    """OI 变化 z-score 羊群因子.

    借鉴 #242 z-score 识别 + #245 herding 顺向思想: 连续同向增仓(z 同号) 的强度.
    z 持续同号 → 资金方向高度一致 → 正向.
    方向: 正向.
    """
    name = "intraday_oi_herding_z_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "OI变化z羊群 (连续同向增仓强度)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        oi = panel["position"].groupby(panel["position"].index.normalize()).last()
        oi_ret = oi.pct_change().replace([np.inf, -np.inf], np.nan)
        std = oi_ret.rolling(20, min_periods=10).std().replace(0, np.nan)
        z = (oi_ret - oi_ret.rolling(20, min_periods=10).mean()) / std
        # 连续同号 z 的最长 run
        def _max_same_run(s):
            arr = s.dropna().values
            if len(arr) < 5:
                return 0.0
            signs = np.sign(arr)
            max_run = 0
            cur = 0
            for x in signs:
                cur = cur + 1 if x != 0 else 0
                max_run = max(max_run, cur)
            return float(max_run)
        score = z.apply(_max_same_run, axis=0)
        return score.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 279. intraday_term_roll_yield_vol — 展期收益波动率
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_term_roll_yield_vol_20d", category="intraday_advanced")
class IntradayTermRollYieldVol20d(Factor):
    """展期收益波动率因子.

    展期收益的 20 日滚动标准差 (借鉴 #249 term_vol_ratio 的波动率构造, 用于展期收益).
    展期收益波动大 → 期限结构不稳定 → 负向.
    方向: 负向.
    """
    name = "intraday_term_roll_yield_vol_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "展期收益波动率 (roll_yield的20日std)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_term_structure_panel(data, dates, universe, freq="1min")
        if not panel or "near_close" not in panel or "far_close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        near = panel["near_close"].groupby(panel["near_close"].index.normalize()).last()
        far = panel["far_close"].groupby(panel["far_close"].index.normalize()).last()
        roll = (near - far) / far.replace(0, np.nan)
        vol = roll.rolling(20, min_periods=10).std()
        return (-vol).reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 280. intraday_oi_surge_follow — OI 突增后价格跟随
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_surge_follow_20d", category="intraday_advanced")
class IntradayOiSurgeFollow20d(Factor):
    """OI 突增后价格跟随因子.

    借鉴 #242 oi_surge_reversal 的"信号×前向收益滚动相关"打分框架,
    但方向相反: OI 突增后价格是否延续 (而非反转).
    突增后延续 → 资金有效性高 → 正向.
    方向: 正向.
    """
    name = "intraday_oi_surge_follow_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "OI突增后跟随 (surge×fwd_ret滚动相关)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "position" not in panel or "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        oi = panel["position"].groupby(panel["position"].index.normalize()).last()
        close = panel["close"].groupby(panel["close"].index.normalize()).last()
        oi_ret = oi.pct_change().replace([np.inf, -np.inf], np.nan)
        oi_std = oi_ret.rolling(20, min_periods=10).std().replace(0, np.nan)
        surge = (oi_ret - oi_ret.rolling(20, min_periods=10).mean()) / oi_std
        fwd_ret = close.pct_change(5).shift(-5)
        score = (surge * fwd_ret).rolling(20, min_periods=5).mean()
        return score.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 281. intraday_seat_long_short_ratio — 席位多空比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_seat_long_short_ratio_20d", category="intraday_advanced")
class IntradaySeatLongShortRatio20d(Factor):
    """席位多空比因子.

    total_long / total_short — 前20会员多头持仓/空头持仓.
    净多格局 → 机构资金看多 → 正向.
    席位数据为日度频率 (T日收盘后公布, shift(1) 后 T+1 可用), 2020-03 起.
    方向: 正向.
    """
    name = "intraday_seat_long_short_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "席位多空比 (total_long/total_short, 日度席位数据)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_seat_panel(data, dates, universe)
        if "total_long" not in panel or "total_short" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ratio = panel["total_long"] / panel["total_short"].replace(0, np.nan)
        return ratio.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 282. intraday_seat_net_change — 席位净持仓变化
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_seat_net_change_20d", category="intraday_advanced")
class IntradaySeatNetChange20d(Factor):
    """席位净持仓变化因子.

    Δnet_position — 前20会员净多头的日度变化 (多空力量此消彼长).
    净多头增加 → 机构加多/减空 → 正向.
    方向: 正向.
    """
    name = "intraday_seat_net_change_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "席位净持仓变化 (Δnet_position, 机构加多=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_seat_panel(data, dates, universe)
        if "net_position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        chg = panel["net_position"].diff()
        return chg.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 283. intraday_seat_divergence — 席位多空分歧
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_seat_divergence_20d", category="intraday_advanced")
class IntradaySeatDivergence20d(Factor):
    """席位多空分歧因子.

    |long_change - short_change| / (total_long + total_short) — 增减仓不对称度.
    分歧大 → 多空激烈博弈 → 方向不明 → 负向.
    方向: 负向.
    """
    name = "intraday_seat_divergence_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "席位多空分歧 (|ΔL-ΔS|/总持仓, 博弈=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_seat_panel(data, dates, universe)
        if not {"long_change", "short_change", "total_long", "total_short"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        denom = panel["total_long"] + panel["total_short"]
        div = (panel["long_change"] - panel["short_change"]).abs() / denom.replace(0, np.nan)
        return (-div).reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 284. intraday_seat_net_strength — 席位净敞口强度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_seat_net_strength_20d", category="intraday_advanced")
class IntradaySeatNetStrength20d(Factor):
    """席位净敞口强度因子.

    net_position / (total_long + total_short) — 净多空头占总量比例 ([-1,1]).
    净敞口大 → 机构方向性强 → 正向.
    方向: 正向.
    """
    name = "intraday_seat_net_strength_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "席位净敞口 (net/(long+short), 方向性强=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_seat_panel(data, dates, universe)
        if "net_position" not in panel or "total_long" not in panel or "total_short" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        denom = panel["total_long"] + panel["total_short"]
        strength = panel["net_position"] / denom.replace(0, np.nan)
        return strength.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 285. intraday_seat_herding — 席位增减仓同向
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_seat_herding_20d", category="intraday_advanced")
class IntradaySeatHerding20d(Factor):
    """席位增减仓同向因子.

    sign(long_change) == sign(short_change) 时记 1 — 多空同向增减仓.
    输出 1 - 同向占比: 高值=单向博弈(一方主导) → 正向; 同向(双加码/双离场) → 低值.
    方向: 正向.
    """
    name = "intraday_seat_herding_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "席位增减仓同向 (1-同向占比, 单向博弈=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_seat_panel(data, dates, universe)
        if "long_change" not in panel or "short_change" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        lc = panel["long_change"]
        sc = panel["short_change"]
        same = (np.sign(lc) == np.sign(sc)) & (lc != 0) & (sc != 0)
        valid = (lc != 0) & (sc != 0)
        ratio = same.where(valid).rolling(20, min_periods=5).mean()
        score = 1.0 - ratio
        return score.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 286. intraday_df_test — 单位根检验 (Dickey-Fuller)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_df_test_20d", category="intraday_advanced")
class IntradayDfTest20d(Factor):
    """单位根检验因子 (Dickey-Fuller, 手动OLS实现).

    日内 log 价格序列做 DF 回归 Δp_t = α + β·p_{t-1} + ε, 取 β 的 t 统计量绝对值.
    借鉴 quantskills 时间序列分析算子 (平稳性检验, 我们此前未涉及).
    |t| 大 → 偏离随机游走 (有趋势或均值回归结构) → 可预测 → 正向.
    方向: 正向.
    """
    name = "intraday_df_test_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "DF单位根检验 (日内log价格|DF t|, 偏离随机游走=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    @staticmethod
    def _df_tstat(series):
        """Dickey-Fuller 检验 t 统计量绝对值.

        优先 statsmodels.adfuller (E:/Python/Pythonenv 已装 0.14.6);
        缺失时回退手动 OLS 实现 (Python312 兼容).
        """
        try:
            from statsmodels.tsa.stattools import adfuller
            res = adfuller(series, autolag="AIC", regresults=False)
            return float(abs(res[0]))
        except Exception:
            pass
        # 手动 OLS Dickey-Fuller: Δy_t = a + b·y_{t-1} + e
        y = np.asarray(series, dtype=float)
        y = y[np.isfinite(y)]
        if len(y) < 30:
            return 0.0
        dy = np.diff(y)
        y_lag = y[:-1]
        X = np.column_stack([np.ones(len(y_lag)), y_lag])
        try:
            beta, res, rank, sv = np.linalg.lstsq(X, dy, rcond=None)
        except Exception:
            return 0.0
        resid = dy - X @ beta
        n, k = X.shape
        s2 = resid @ resid / (n - k)
        if s2 < 1e-16:
            return 0.0
        xtx_inv = np.linalg.inv(X.T @ X)
        se_b = np.sqrt(s2 * xtx_inv[1, 1])
        if se_b < 1e-16:
            return 0.0
        return float(abs(beta[1] / se_b))

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        tests: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna().values
                c = c[c > 0]
                if len(c) < 30:
                    continue
                lp = np.log(c)
                vals[col] = self._df_tstat(lp)
            if vals:
                tests[dt] = pd.Series(vals)
        if not tests:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(tests).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 287. intraday_vwap_residual_vol — VWAP 回归残差波动
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_vwap_residual_vol_20d", category="intraday_advanced")
class IntradayVwapResidualVol20d(Factor):
    """VWAP 回归残差波动因子.

    分钟价格对累计 VWAP 做线性回归, 取残差的标准差 (借鉴 quantskills 正交化算子: 剥离基准后的不可解释部分).
    与 #30 偏离度互补: 这里去除 VWAP 的线性趋势后看残差噪声.
    残差大 → 价格偏离公允趋势 → 噪声主导 → 负向.
    方向: 负向.
    """
    name = "intraday_vwap_residual_vol_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "VWAP回归残差波动 (正交化残差std, 噪声=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, volume = panel["close"], panel["volume"]
        day = close.index.normalize()
        vols: dict = {}
        for dt in sorted(set(day)):
            grp_c = close.loc[day == dt]
            grp_v = volume.loc[day == dt]
            if len(grp_c) < 30:
                continue
            vals = {}
            for col in grp_c.columns:
                c = grp_c[col].dropna()
                v = grp_v[col].dropna()
                common = c.index.intersection(v.index)
                if len(common) < 30:
                    continue
                c_c = c.loc[common]
                v_c = v.loc[common]
                vwap = (c_c * v_c).cumsum() / v_c.cumsum().replace(0, np.nan)
                x = vwap.values
                y = c_c.values
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < 30:
                    continue
                slope, intercept = np.polyfit(x[mask], y[mask], 1)
                resid = y[mask] - (slope * x[mask] + intercept)
                vals[col] = -float(resid.std(ddof=0))
            if vals:
                vols[dt] = pd.Series(vals)
        if not vols:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(vols).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 288. intraday_trend_resid_kurt — 去趋势残差峰度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_trend_resid_kurt_20d", category="intraday_advanced")
class IntradayTrendResidKurt20d(Factor):
    """去趋势残差峰度因子.

    价格路径去线性趋势后残差的峰度 (借鉴正交化算子的高阶矩, 与 #115 残差偏度互补).
    残差峰度高 → 噪声厚尾 → 偶发大幅偏离 → 负向.
    方向: 负向.
    """
    name = "intraday_trend_resid_kurt_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "去趋势残差峰度 (噪声厚尾=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        kurts: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < 30:
                    continue
                y = c.values / c.values[0] - 1.0
                t = np.arange(len(y))
                slope, intercept = np.polyfit(t, y, 1)
                resid = y - (slope * t + intercept)
                s = resid.std(ddof=0)
                if s < 1e-12:
                    vals[col] = 0.0
                    continue
                kurt = float(np.mean((resid / s) ** 4))
                vals[col] = -kurt
            if vals:
                kurts[dt] = pd.Series(vals)
        if not kurts:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(kurts).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 289. intraday_vol_regime_persist — 跨日波动状态延续
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_vol_regime_persist_20d", category="intraday_advanced")
class IntradayVolRegimePersist20d(Factor):
    """跨日波动状态延续因子.

    今日高波动状态延续昨日高波动状态的频率 (借鉴 quantskills 市场状态算子, 与 #176 日内切换互补).
    高波动持续 → 风险环境稳固 → 负向.
    方向: 负向.
    """
    name = "intraday_vol_regime_persist_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "跨日波动延续 (今日高波延续昨日频率)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        daily_vol: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 20:
                    continue
                vals[col] = float(r.std(ddof=0))
            if vals:
                daily_vol[dt] = pd.Series(vals)
        if not daily_vol:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily_vol).T
        df.index = pd.DatetimeIndex(df.index)
        high_state = df > df.rolling(20, min_periods=5).median()
        persist = (high_state & high_state.shift(1)).astype(float)
        score = persist.rolling(20, min_periods=5).mean()
        return (-score).reindex(index=dates, columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 290. intraday_term_half_life — 价差均值回归半衰期
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_term_half_life_20d", category="intraday_advanced")
class IntradayTermHalfLife20d(Factor):
    """价差均值回归半衰期因子.

    期限价差 (far-near) 日度自相关衰减到0.5的滞后数 (借鉴 quantskills 协整/半衰期算子).
    半衰期短 → 价差快速回归 → 期限结构稳定 → 正向.
    方向: 正向.
    """
    name = "intraday_term_half_life_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "价差半衰期 (价差自相关衰减, 快回归=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_term_structure_panel(data, dates, universe, freq="1min")
        if not panel or "near_close" not in panel or "far_close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        near = panel["near_close"].groupby(panel["near_close"].index.normalize()).last()
        far = panel["far_close"].groupby(panel["far_close"].index.normalize()).last()
        spread = (far - near).reindex(index=pd.DatetimeIndex(dates))
        # 滚动 AR(1) 半衰期: hl = log(0.5)/log(ρ1), 用过去 40 日的 1 阶自相关
        win = 40
        out = pd.DataFrame(index=spread.index, columns=spread.columns)
        for col in spread.columns:
            s = spread[col]
            for i in range(len(s)):
                window = s.iloc[max(0, i - win):i].dropna().values
                if len(window) < 20:
                    continue
                x, y = window[:-1], window[1:]
                if x.std(ddof=0) < 1e-12 or y.std(ddof=0) < 1e-12:
                    continue
                rho = float(np.corrcoef(x, y)[0, 1])
                if rho <= 0.0:
                    out.iloc[i, out.columns.get_loc(col)] = 0.0  # 快速回归
                elif rho >= 1.0:
                    out.iloc[i, out.columns.get_loc(col)] = -10.0  # 不回归
                else:
                    hl = np.log(0.5) / np.log(rho)
                    out.iloc[i, out.columns.get_loc(col)] = -hl  # 半衰期短=高值=正向
        return out.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 291. intraday_vp_corr_stability — 量价相关稳定性
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_vp_corr_stability_20d", category="intraday_advanced")
class IntradayVpCorrStability20d(Factor):
    """量价相关稳定性因子.

    20日量价相关的时间稳定性 (rolling std of 20日滚动量价相关, 借鉴相关性动态算子).
    量价相关波动大 → 量价关系不稳定 → 负向.
    方向: 负向.
    """
    name = "intraday_vp_corr_stability_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "量价相关稳定 (20日量价相关的std, 不稳定=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, volume = panel["close"], panel["volume"]
        day = close.index.normalize()
        daily_corr: dict = {}
        for dt in sorted(set(day)):
            grp_c = close.loc[day == dt]
            grp_v = volume.loc[day == dt]
            if len(grp_c) < 30:
                continue
            vals = {}
            for col in grp_c.columns:
                c = grp_c[col].dropna()
                v = grp_v[col].dropna()
                common = c.index.intersection(v.index)
                if len(common) < 30:
                    continue
                rc = c.loc[common].pct_change()
                vc = v.loc[common]
                if rc.std(ddof=0) < 1e-12 or vc.std(ddof=0) < 1e-12:
                    vals[col] = 0.0
                else:
                    corr_val = float(np.corrcoef(rc, vc)[0, 1])
                    vals[col] = corr_val if not np.isnan(corr_val) else 0.0
            if vals:
                daily_corr[dt] = pd.Series(vals)
        if not daily_corr:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily_corr).T
        df.index = pd.DatetimeIndex(df.index)
        corr20 = df.rolling(20, min_periods=10).mean()
        stability = corr20.rolling(20, min_periods=10).std()
        return (-stability).reindex(index=dates, columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 292. intraday_ma_cross_duration — 均线交叉持续时长
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_ma_cross_duration_20d", category="intraday_advanced")
class IntradayMaCrossDuration20d(Factor):
    """均线交叉持续时长因子.

    收盘价 MA5 持续在 MA20 上方的连续天数 (借鉴方向类算子的趋势持续性, 与 #117 交叉值互补).
    持续上穿 → 趋势稳固 → 正向.
    方向: 正向.
    """
    name = "intraday_ma_cross_duration_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "均线交叉持续 (MA5>MA20连续天数, 趋势稳固=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        if close is None or close.empty:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ma5 = close.rolling(5, min_periods=3).mean()
        ma20 = close.rolling(20, min_periods=5).mean()
        above = (ma5 > ma20).astype(float)
        # 连续为正的最长 run (截至当日)
        def _run_today(s):
            arr = s.dropna().values
            if len(arr) == 0:
                return 0.0
            cur = 0
            for x in arr[::-1]:
                if x > 0:
                    cur += 1
                else:
                    break
            return float(cur)
        duration = above.rolling(40, min_periods=1).apply(_run_today, raw=False)
        return duration.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 293. intraday_volume_rank_ratio — 成交量历史分位
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_volume_rank_ratio_20d", category="intraday_advanced")
class IntradayVolumeRankRatio20d(Factor):
    """成交量历史分位因子.

    当日成交量在自身20日分布中的百分位 (借鉴 quantskills 统计排序算子, 我们此前未系统使用 rank 分位).
    量能创高位 → 关注度上升 → 正向.
    方向: 正向.
    """
    name = "intraday_volume_rank_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "量20日分位 (当日量在自身20日分布的百分位)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        volume = panel["volume"]
        day = volume.index.normalize()
        daily_vol: dict = {}
        for dt in sorted(set(day)):
            grp = volume.loc[day == dt]
            if len(grp) < 10:
                continue
            vals = {}
            for col in grp.columns:
                v = grp[col].dropna()
                if len(v) < 10:
                    continue
                vals[col] = float(v.sum())
            if vals:
                daily_vol[dt] = pd.Series(vals)
        if not daily_vol:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily_vol).T
        df.index = pd.DatetimeIndex(df.index)
        rank = df.rolling(20, min_periods=5).apply(
            lambda x: float((x.iloc[:-1] <= x.iloc[-1]).sum() / max(1, len(x) - 1)), raw=False)
        return rank.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 294. intraday_price_rank_vol — 价格历史区间分位
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_price_rank_vol_20d", category="intraday_advanced")
class IntradayPriceRankVol20d(Factor):
    """价格历史区间分位因子.

    收盘价在自身20日高低区间中的位置 (统计排序算子, 跨日, 与 #17 日内位置互补).
    价格处历史高位 → 强势 → 正向.
    方向: 正向.
    """
    name = "intraday_price_rank_vol_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "价格20日区间分位 (收盘在20日高低位置, 高位=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        high = data.get("high", dates, universe)
        low = data.get("low", dates, universe)
        close = data.get("close", dates, universe)
        if high is None or low is None or close is None:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        if high.empty or low.empty or close.empty:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        hh = high.rolling(20, min_periods=5).max()
        ll = low.rolling(20, min_periods=5).min()
        rng = (hh - ll).replace(0, np.nan)
        pos = (close - ll) / rng
        return pos.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 295. intraday_oi_rank — 持仓历史分位
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_rank_20d", category="intraday_advanced")
class IntradayOiRank20d(Factor):
    """持仓历史分位因子.

    当日持仓量在自身20日分布中的百分位 (统计排序算子用于 OI, 跨日).
    持仓创高位 → 资金关注度持续上升 → 正向.
    方向: 正向.
    """
    name = "intraday_oi_rank_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "持仓20日分位 (当日OI在自身20日分布百分位)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        position = panel["position"]
        day = position.index.normalize()
        daily_oi: dict = {}
        for dt in sorted(set(day)):
            grp = position.loc[day == dt]
            if len(grp) < 10:
                continue
            vals = {}
            for col in grp.columns:
                p = grp[col].dropna()
                if len(p) < 10:
                    continue
                vals[col] = float(p.iloc[-1])
            if vals:
                daily_oi[dt] = pd.Series(vals)
        if not daily_oi:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily_oi).T
        df.index = pd.DatetimeIndex(df.index)
        rank = df.rolling(20, min_periods=5).apply(
            lambda x: float((x.iloc[:-1] <= x.iloc[-1]).sum() / max(1, len(x) - 1)), raw=False)
        return rank.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 296. intraday_seat_lsr_rank — 席位多空比历史分位
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_seat_lsr_rank_20d", category="intraday_advanced")
class IntradaySeatLsrRank20d(Factor):
    """席位多空比历史分位因子.

    席位多空比 (total_long/total_short) 在自身20日分布中的百分位 (统计排序算子×席位).
    多空比处历史高位 → 机构净多格局增强 → 正向.
    方向: 正向.
    """
    name = "intraday_seat_lsr_rank_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "席位多空比20日分位 (机构净多增强=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_seat_panel(data, dates, universe)
        if "total_long" not in panel or "total_short" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        lsr = panel["total_long"] / panel["total_short"].replace(0, np.nan)
        rank = lsr.rolling(20, min_periods=5).apply(
            lambda x: float((x.iloc[:-1] <= x.iloc[-1]).sum() / max(1, len(x) - 1)), raw=False)
        return rank.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 297. intraday_seat_net_change_z — 席位净持仓变化 z-score
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_seat_net_change_z_20d", category="intraday_advanced")
class IntradaySeatNetChangeZ20d(Factor):
    """席位净持仓变化 z-score 因子.

    席位净持仓变化 (Δnet_position) 的20日 z-score (标准化异常资金流).
    净多头异常增加 → 机构集中加多 → 正向.
    方向: 正向.
    """
    name = "intraday_seat_net_change_z_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "席位净持仓变化z (Δnet的20日z-score)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_seat_panel(data, dates, universe)
        if "net_position" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        chg = panel["net_position"].diff()
        mean = chg.rolling(20, min_periods=5).mean()
        std = chg.rolling(20, min_periods=5).std().replace(0, np.nan)
        z = (chg - mean) / std
        return z.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 298. intraday_seat_strength_rank — 席位净敞口历史分位
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_seat_strength_rank_20d", category="intraday_advanced")
class IntradaySeatStrengthRank20d(Factor):
    """席位净敞口历史分位因子.

    席位净敞口 (net/(long+short)) 在自身20日分布中的百分位.
    净敞口处历史高位 → 机构方向性最强 → 正向.
    方向: 正向.
    """
    name = "intraday_seat_strength_rank_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "席位净敞口20日分位 (方向性最强=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_seat_panel(data, dates, universe)
        if "net_position" not in panel or "total_long" not in panel or "total_short" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        denom = panel["total_long"] + panel["total_short"]
        strength = panel["net_position"] / denom.replace(0, np.nan)
        rank = strength.rolling(20, min_periods=5).apply(
            lambda x: float((x.iloc[:-1] <= x.iloc[-1]).sum() / max(1, len(x) - 1)), raw=False)
        return rank.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 299. intraday_seat_divergence_z — 席位分歧 z-score
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_seat_divergence_z_20d", category="intraday_advanced")
class IntradaySeatDivergenceZ20d(Factor):
    """席位分歧 z-score 因子.

    席位多空分歧度 (|ΔL-ΔS|/总持仓) 的20日 z-score (标准化异常分歧).
    分歧异常放大 → 多空冲突加剧 → 负向.
    方向: 负向.
    """
    name = "intraday_seat_divergence_z_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "席位分歧z (分歧度的20日z-score, 冲突=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_seat_panel(data, dates, universe)
        if not {"long_change", "short_change", "total_long", "total_short"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        denom = panel["total_long"] + panel["total_short"]
        div = (panel["long_change"] - panel["short_change"]).abs() / denom.replace(0, np.nan)
        mean = div.rolling(20, min_periods=5).mean()
        std = div.rolling(20, min_periods=5).std().replace(0, np.nan)
        z = (div - mean) / std
        return (-z).reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 300. intraday_seat_herding_rank — 席位博弈方向历史分位
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_seat_herding_rank_20d", category="intraday_advanced")
class IntradaySeatHerdingRank20d(Factor):
    """席位博弈方向历史分位因子.

    席位单向博弈占比 (1-增减仓同向占比) 在自身20日分布中的百分位.
    单向博弈处历史高位 → 一方主导极端 → 正向.
    方向: 正向.
    """
    name = "intraday_seat_herding_rank_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "席位博弈方向20日分位 (单向极端=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_seat_panel(data, dates, universe)
        if "long_change" not in panel or "short_change" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        lc = panel["long_change"]
        sc = panel["short_change"]
        same = (np.sign(lc) == np.sign(sc)) & (lc != 0) & (sc != 0)
        valid = (lc != 0) & (sc != 0)
        ratio = same.where(valid)
        one_side = 1.0 - ratio
        rank = one_side.rolling(20, min_periods=5).apply(
            lambda x: float((x.iloc[:-1] <= x.iloc[-1]).sum() / max(1, len(x) - 1)), raw=False)
        return rank.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 301. intraday_settle_basis_rank — 结算基差历史分位
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_settle_basis_rank_20d", category="intraday_advanced")
class IntradaySettleBasisRank20d(Factor):
    """结算基差历史分位因子.

    结算基差 ((settle-close)/close) 在自身20日分布中的百分位 (统计排序×结算).
    基差处历史高位 → 结算持续强于收盘 → 正向.
    方向: 正向.
    """
    name = "intraday_settle_basis_rank_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "结算基差20日分位 (结算持续强=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        settle = _read_local_daily(data, dates, universe, "settle")
        close = data.get("close", dates, universe)
        if settle is None or settle.empty or close is None or close.empty:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        basis = (settle - close) / close.replace(0, np.nan)
        rank = basis.rolling(20, min_periods=5).apply(
            lambda x: float((x.iloc[:-1] <= x.iloc[-1]).sum() / max(1, len(x) - 1)), raw=False)
        return rank.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 302. intraday_settle_basis_z — 结算基差 z-score
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_settle_basis_z_20d", category="intraday_advanced")
class IntradaySettleBasisZ20d(Factor):
    """结算基差 z-score 因子.

    结算基差的20日 z-score (标准化偏离, 借鉴均值回归构造).
    基差异常偏离 → 回归压力 → 负向.
    方向: 负向.
    """
    name = "intraday_settle_basis_z_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "结算基差z (基差20日z-score, 偏离=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        settle = _read_local_daily(data, dates, universe, "settle")
        close = data.get("close", dates, universe)
        if settle is None or settle.empty or close is None or close.empty:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        basis = (settle - close) / close.replace(0, np.nan)
        mean = basis.rolling(20, min_periods=5).mean()
        std = basis.rolling(20, min_periods=5).std().replace(0, np.nan)
        z = (basis - mean) / std
        return (-z).reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 303. intraday_settle_diff_rank — 结算-收盘差历史分位
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_settle_diff_rank_20d", category="intraday_advanced")
class IntradaySettleDiffRank20d(Factor):
    """结算-收盘差历史分位因子.

    结算-收盘价差在自身20日分布中的百分位 (与基差 rank 互补: 这里用绝对价差).
    价差处历史高位 → 结算与收盘脱节 → 异常 → 负向.
    方向: 负向.
    """
    name = "intraday_settle_diff_rank_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "结算-收盘差20日分位 (脱节=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        settle = _read_local_daily(data, dates, universe, "settle")
        close = data.get("close", dates, universe)
        if settle is None or settle.empty or close is None or close.empty:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        diff = (settle - close).abs()
        rank = diff.rolling(20, min_periods=5).apply(
            lambda x: float((x.iloc[:-1] <= x.iloc[-1]).sum() / max(1, len(x) - 1)), raw=False)
        return (-rank).reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 304. intraday_settle_surge_z — 结算价突变 z-score
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_settle_surge_z_20d", category="intraday_advanced")
class IntradaySettleSurgeZ20d(Factor):
    """结算价突变 z-score 因子.

    结算价日度变化率的20日 z-score (借鉴 #242 突增识别, 用于结算价).
    结算价异常突变 → 定价结构剧变 → 不确定 → 负向.
    方向: 负向.
    """
    name = "intraday_settle_surge_z_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "结算价突变z (settle变化率z-score, 剧变=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        settle = _read_local_daily(data, dates, universe, "settle")
        if settle is None or settle.empty:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret = settle.pct_change().replace([np.inf, -np.inf], np.nan)
        mean = ret.rolling(20, min_periods=5).mean()
        std = ret.rolling(20, min_periods=5).std().replace(0, np.nan)
        z = (ret - mean) / std
        return (-z.abs()).reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 305. intraday_seat_count_rank — 席位参与度历史分位
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_seat_count_rank_20d", category="intraday_advanced")
class IntradaySeatCountRank20d(Factor):
    """席位参与度历史分位因子.

    参与席位数量 (seat_count) 在自身20日分布中的百分位 (统计排序×参与度).
    参与席位处历史高位 → 市场关注度提升 → 活跃 → 正向.
    方向: 正向.
    """
    name = "intraday_seat_count_rank_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "席位参与度20日分位 (参与席位高位=活跃=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_seat_panel(data, dates, universe)
        if "seat_count" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        sc = panel["seat_count"]
        rank = sc.rolling(20, min_periods=5).apply(
            lambda x: float((x.iloc[:-1] <= x.iloc[-1]).sum() / max(1, len(x) - 1)), raw=False)
        return rank.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 306. intraday_ret_transition_bias — 收益转移概率不对称
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_ret_transition_bias_20d", category="intraday_advanced")
class IntradayRetTransitionBias20d(Factor):
    """收益转移概率不对称因子.

    P(涨|前涨) - P(跌|前跌) — 收益马尔可夫转移概率的净偏置 (趋势惯性的概率视角).
    与 #54 自相关互补: 这里用转移概率而非相关系数.
    正偏置 → 涨后易涨/跌后易跌 → 趋势惯性 → 正向.
    方向: 正向.
    """
    name = "intraday_ret_transition_bias_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "收益转移不对称 (P(涨|涨)-P(跌|跌), 趋势惯性=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        biases: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna().values
                if len(r) < 30:
                    continue
                s = np.sign(r)
                s = s[s != 0]
                if len(s) < 10:
                    vals[col] = 0.0
                    continue
                up_up = np.sum((s[:-1] > 0) & (s[1:] > 0))
                up_prev = np.sum(s[:-1] > 0)
                dn_dn = np.sum((s[:-1] < 0) & (s[1:] < 0))
                dn_prev = np.sum(s[:-1] < 0)
                p_up = up_up / max(1, up_prev)
                p_dn = dn_dn / max(1, dn_prev)
                vals[col] = float(p_up - p_dn)
            if vals:
                biases[dt] = pd.Series(vals)
        if not biases:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(biases).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 307. intraday_vol_state_trend — 波动状态条件趋势
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_vol_state_trend_20d", category="intraday_advanced")
class IntradayVolStateTrend20d(Factor):
    """波动状态条件趋势因子.

    低波动时段的趋势方向 (低波时段收益方向占比) — 条件趋势 (波动调节的趋势).
    低波时段方向明确 → 冷静期趋势 → 高质量趋势 → 正向.
    方向: 正向.
    """
    name = "intraday_vol_state_trend_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "波动条件趋势 (低波时段方向占比, 冷静期趋势=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        trends: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 30:
                    continue
                ar = r.abs()
                low_vol = ar < ar.median()
                if low_vol.sum() < 5:
                    vals[col] = 0.0
                    continue
                low_dir = np.sign(r[low_vol].sum())
                up_share = (r[low_vol] > 0).sum() / low_vol.sum()
                vals[col] = float(up_share if low_dir > 0 else (1.0 - up_share))
            if vals:
                trends[dt] = pd.Series(vals)
        if not trends:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(trends).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 308. intraday_path_second_diff — 路径平滑度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_path_second_diff_20d", category="intraday_advanced")
class IntradayPathSecondDiff20d(Factor):
    """路径平滑度因子.

    价格路径二阶差分绝对值的均值 (曲率/锯齿度, 与 #101 转折计数互补: 这里是曲率幅度).
    二阶差分大 → 路径急转弯 → 锯齿噪声 → 负向.
    方向: 负向.
    """
    name = "intraday_path_second_diff_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "路径平滑度 (二阶差分均值, 急转弯=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        smooths: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna().values
                if len(c) < 30:
                    continue
                d1 = np.diff(c)
                d2 = np.diff(d1)
                base = np.mean(np.abs(d1)) + 1e-12
                vals[col] = -float(np.mean(np.abs(d2)) / base)
            if vals:
                smooths[dt] = pd.Series(vals)
        if not smooths:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(smooths).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 309. intraday_recovery_shape — 回撤恢复形状
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_recovery_shape_20d", category="intraday_advanced")
class IntradayRecoveryShape20d(Factor):
    """回撤恢复形状因子.

    最大回撤谷底后的恢复路径凸度: 恢复段二次回归系数 (V型快恢复 vs 圆弧慢爬, 与 #106 速度互补).
    凸 (V型) → 恢复干脆 → 承接强 → 正向.
    方向: 正向.
    """
    name = "intraday_recovery_shape_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "回撤恢复形状 (恢复路径凸度, V型=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        shapes: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 40:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < 40:
                    continue
                cum = c.values / c.values[0] - 1.0
                peak = np.maximum.accumulate(cum)
                dd = peak - cum
                trough_idx = int(np.argmax(dd))
                recover_seg = cum[trough_idx:]
                if len(recover_seg) < 8:
                    vals[col] = 0.0
                    continue
                t = np.arange(len(recover_seg)) / max(1, len(recover_seg) - 1)
                coeff = np.polyfit(t, recover_seg, 2)
                vals[col] = float(coeff[0])  # 二次项: 凸=正值=V型
            if vals:
                shapes[dt] = pd.Series(vals)
        if not shapes:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(shapes).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 310. intraday_break_sustain — 突破维持时间
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_break_sustain_20d", category="intraday_advanced")
class IntradayBreakSustain20d(Factor):
    """突破维持时间因子.

    突破日内前高后, 价格停留在突破位上方的时间占比 (突破有效性, 与 #141 回踩互补).
    突破后维持 → 突破真实 → 正向.
    方向: 正向.
    """
    name = "intraday_break_sustain_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "突破维持时间 (突破前高后停留占比, 真实突破=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        sustains: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna().values
                if len(c) < 30:
                    continue
                cum_max = np.maximum.accumulate(c)
                # 突破: 价格高于此前所有 (cum_max 创新高)
                broke = False
                hold_cnt = 0
                total_after = 0
                for i in range(1, len(c)):
                    if c[i] > cum_max[i - 1]:
                        broke = True
                    if broke:
                        total_after += 1
                        if c[i] >= cum_max[i - 1]:
                            hold_cnt += 1
                vals[col] = float(hold_cnt / max(1, total_after)) if total_after > 0 else 0.0
            if vals:
                sustains[dt] = pd.Series(vals)
        if not sustains:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(sustains).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 311. intraday_volume_trend_share — 趋势段量占比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_volume_trend_share_20d", category="intraday_advanced")
class IntradayVolumeTrendShare20d(Factor):
    """趋势段量占比因子.

    与主导趋势方向一致的大波动分钟的量占总量的比例 (有效放量, 与 #102 计数互补: 这里加权量).
    趋势段放量 → 有效资金推动 → 正向.
    方向: 正向.
    """
    name = "intraday_volume_trend_share_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "趋势段量占比 (顺趋势大波动分钟量占比)"
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
        shares: dict = {}
        for dt in sorted(set(day)):
            grp_r = ret_1m.loc[day == dt]
            grp_v = volume.loc[day == dt]
            if len(grp_r) < 20:
                continue
            vals = {}
            for col in grp_r.columns:
                r = grp_r[col].dropna()
                v = grp_v[col].dropna()
                common = r.index.intersection(v.index)
                if len(common) < 20:
                    continue
                r_c = r.loc[common]
                v_c = v.loc[common]
                dom = np.sign(r_c.sum())
                med_abs = r_c.abs().median()
                total_v = v_c.sum()
                if abs(dom) < 1e-12 or total_v < 1e-12:
                    vals[col] = 0.0
                    continue
                aligned = (r_c.abs() > med_abs) & (np.sign(r_c) == dom)
                vals[col] = float(v_c[aligned].sum() / total_v)
            if vals:
                shares[dt] = pd.Series(vals)
        if not shares:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(shares).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 312. intraday_range_position_drift — 区间位置推进速度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_range_position_drift_20d", category="intraday_advanced")
class IntradayRangePositionDrift20d(Factor):
    """区间位置推进速度因子.

    价格在日内区间位置序列的线性回归斜率 (位置推进速度, 与 #98/#99 极值时间互补).
    快速推进到高位 → 买方持续发力 → 正向.
    方向: 正向.
    """
    name = "intraday_range_position_drift_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "区间位置推进速度 (位置序列斜率, 快速推进=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"high", "low", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, low, close = panel["high"], panel["low"], panel["close"]
        day = close.index.normalize()
        drifts: dict = {}
        for dt in sorted(set(day)):
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 30:
                continue
            vals = {}
            for col in grp_c.columns:
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                c = grp_c[col].dropna()
                common = h.index.intersection(l.index).intersection(c.index)
                if len(common) < 30:
                    continue
                h_c, l_c, c_c = (x.loc[common] for x in (h, l, c))
                rng = h_c.max() - l_c.min()
                if rng < 1e-12:
                    vals[col] = 0.0
                    continue
                pos = (c_c - l_c.min()) / rng
                t = np.arange(len(pos)) / max(1, len(pos) - 1)
                slope = np.polyfit(t, pos.values, 1)[0]
                vals[col] = float(slope)
            if vals:
                drifts[dt] = pd.Series(vals)
        if not drifts:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(drifts).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 313. intraday_skew_path — 偏度日内演化
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_skew_path_20d", category="intraday_advanced")
class IntradaySkewPath20d(Factor):
    """偏度日内演化因子.

    日内后半段偏度 - 前半段偏度 (偏度的时间变化, 与 #4 全天偏度/#178 分段一致互补).
    偏度增强 (后半段更右偏) → 上行脉冲增加 → 正向.
    方向: 正向.
    """
    name = "intraday_skew_path_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "偏度日内演化 (后半-前半偏度, 增强=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        paths: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            n = len(grp)
            if n < 40:
                continue
            mid = n // 2
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 40:
                    continue
                first = r.iloc[:mid]
                second = r.iloc[mid:]
                if len(first) < 10 or len(second) < 10:
                    continue
                sk_first = float(pd.Series(first).skew())
                sk_second = float(pd.Series(second).skew())
                vals[col] = float(sk_second - sk_first)
            if vals:
                paths[dt] = pd.Series(vals)
        if not paths:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(paths).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 314. intraday_momentum_decay — 日内动量衰减
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_momentum_decay_20d", category="intraday_advanced")
class IntradayMomentumDecay20d(Factor):
    """日内动量衰减因子.

    前半段动量 - 后半段动量 (动量的时间衰减, 与 #97 方向一致互补: 这里是幅度衰减).
    动量衰减 → 动能减弱 → 负向.
    方向: 负向.
    """
    name = "intraday_momentum_decay_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "日内动量衰减 (前半-后半动量, 衰减=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        decays: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            n = len(grp)
            if n < 40:
                continue
            mid = n // 2
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < 40:
                    continue
                first = c.iloc[:mid]
                second = c.iloc[mid:]
                if len(first) < 5 or len(second) < 5:
                    continue
                mom_first = first.iloc[-1] / first.iloc[0] - 1.0
                mom_second = second.iloc[-1] / second.iloc[0] - 1.0
                # 动量衰减: 前半动量绝对值 - 后半动量绝对值 (同方向下)
                if np.sign(mom_first) == np.sign(mom_second) and abs(mom_first) > 1e-12:
                    vals[col] = -float(abs(mom_first) - abs(mom_second))
                else:
                    vals[col] = float(-abs(mom_first) - abs(mom_second))
            if vals:
                decays[dt] = pd.Series(vals)
        if not decays:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(decays).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 315. intraday_volume_time_shape — 量能时间形态
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_volume_time_shape_20d", category="intraday_advanced")
class IntradayVolumeTimeShape20d(Factor):
    """量能时间形态因子.

    日内量能 U 型强度 (早+晚)/午 的另一种: 用三段量占比刻画形态 (与 #34 波动U型互补: 这里是量).
    U 型明显 → 开盘收盘机构集中 → 正向.
    方向: 正向.
    """
    name = "intraday_volume_time_shape_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "量能时间形态 ((早量+晚量)/午量, U型=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        volume = panel["volume"]
        day = volume.index.normalize()
        shapes: dict = {}
        for dt in sorted(set(day)):
            grp = volume.loc[day == dt]
            n = len(grp)
            if n < 30:
                continue
            vals = {}
            for col in grp.columns:
                v = grp[col].dropna().reset_index(drop=True)
                if len(v) < 30:
                    continue
                # 三段切片基于品种自身长度 (面板长度可能含其他品种夜盘, 不能作基准)
                third = max(8, len(v) // 3)
                early = v.iloc[:third].mean()
                midday = v.iloc[third:2 * third].mean()
                late = v.iloc[2 * third:].mean()
                if midday < 1e-12 or pd.isna(midday):
                    vals[col] = 0.0
                    continue
                vals[col] = float((early + late) / (2.0 * midday))
            if vals:
                shapes[dt] = pd.Series(vals)
        if not shapes:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(shapes).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 316. intraday_gap_trend_interaction — 跳空-日内交互
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_gap_trend_interaction_20d", category="intraday_advanced")
class IntradayGapTrendInteraction20d(Factor):
    """跳空-日内交互因子.

    隔夜跳空方向与日内趋势强度的交互 (跳空大且日内顺势 = 强共识, 与 #27 方向一致互补).
    跳空顺势且日内趋势强 → 共识确认 → 正向.
    方向: 正向.
    """
    name = "intraday_gap_trend_interaction_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "跳空日内交互 (跳空顺势×趋势强度, 共识=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"open", "high", "low", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        open_px, high, low, close = panel["open"], panel["high"], panel["low"], panel["close"]
        day = close.index.normalize()
        interactions: dict = {}
        prev_close: dict = {}
        for dt in sorted(set(day)):
            grp_o = open_px.loc[day == dt]
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 30:
                continue
            vals = {}
            for col in grp_c.columns:
                o = grp_o[col].dropna()
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                c = grp_c[col].dropna()
                if len(o) < 1 or len(c) < 30:
                    continue
                prev_c = prev_close.get(col)
                if prev_c is None or prev_c < 1e-12:
                    prev_close[col] = c.iloc[-1]
                    continue
                o_first = o.iloc[0]
                gap_dir = np.sign(o_first - prev_c)
                trend_eff = abs(c.iloc[-1] - o_first) / (h.max() - l.min() + 1e-12)
                intraday_dir = np.sign(c.iloc[-1] - o_first)
                aligned = 1.0 if gap_dir == intraday_dir else -1.0
                vals[col] = float(aligned * trend_eff)
                prev_close[col] = c.iloc[-1]
            if vals:
                interactions[dt] = pd.Series(vals)
        if not interactions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(interactions).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 317. intraday_high_low_momentum — 高低点间动量
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_high_low_momentum_20d", category="intraday_advanced")
class IntradayHighLowMomentum20d(Factor):
    """高低点间动量因子.

    从日内最低点到最高点的速度 - 从最高点到最低点的速度 (双向推进对比, 与 #155 顺序互补).
    上行推进快 → 多方爆发力 → 正向.
    方向: 正向.
    """
    name = "intraday_high_low_momentum_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "高低点间动量 (上行速度-下行速度, 多方爆发=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"high", "low", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, low, close = panel["high"], panel["low"], panel["close"]
        day = close.index.normalize()
        momentums: dict = {}
        for dt in sorted(set(day)):
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 30:
                continue
            vals = {}
            for col in grp_c.columns:
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                c = grp_c[col].dropna()
                common = h.index.intersection(l.index).intersection(c.index)
                if len(common) < 30:
                    continue
                h_c, l_c, c_c = (x.loc[common] for x in (h, l, c))
                idx_high = int(np.argmax(h_c.values))
                idx_low = int(np.argmin(l_c.values))
                base = h_c.max() - l_c.min()
                if base < 1e-12 or abs(idx_high - idx_low) < 1:
                    vals[col] = 0.0
                    continue
                speed_up = base / abs(idx_high - idx_low)
                # 下行段: 从先出现的极值到后出现的极值
                first, second = min(idx_high, idx_low), max(idx_high, idx_low)
                if idx_low < idx_high:
                    down_speed = (h_c.max() - l_c.min()) / (second - first)
                    vals[col] = float(speed_up - down_speed)
                else:
                    up_speed = (h_c.max() - l_c.min()) / (second - first)
                    vals[col] = float(up_speed - speed_up)
            if vals:
                momentums[dt] = pd.Series(vals)
        if not momentums:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(momentums).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 318. intraday_vol_volume_lead — 波动-量领先滞后
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_vol_volume_lead_20d", category="intraday_advanced")
class IntradayVolVolumeLead20d(Factor):
    """波动-量领先滞后因子.

    corr(|ret|_t, vol_{t+1}) - corr(|ret|_{t+1}, vol_t) — 波动领先量还是量领先波动.
    量领先波动 → 放量先于波动 → 信息预演 → 正向.
    方向: 正向.
    """
    name = "intraday_vol_volume_lead_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "波动-量领先 (量领先波动=信息预演=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, volume = panel["close"], panel["volume"]
        ret_abs = close.pct_change().abs()
        day = ret_abs.index.normalize()
        leads: dict = {}
        for dt in sorted(set(day)):
            grp_r = ret_abs.loc[day == dt]
            grp_v = volume.loc[day == dt]
            if len(grp_r) < 40:
                continue
            vals = {}
            for col in grp_r.columns:
                r = grp_r[col].dropna()
                v = grp_v[col].dropna()
                common = r.index.intersection(v.index)
                if len(common) < 40:
                    continue
                r_c = r.loc[common].values
                v_c = v.loc[common].values
                # 量领先波动: corr(vol_t, |ret|_{t+1})
                c1 = float(np.corrcoef(v_c[:-1], r_c[1:])[0, 1]) if v_c[:-1].std() > 1e-12 and r_c[1:].std() > 1e-12 else 0.0
                # 波动领先量: corr(|ret|_t, vol_{t+1})
                c2 = float(np.corrcoef(r_c[:-1], v_c[1:])[0, 1]) if r_c[:-1].std() > 1e-12 and v_c[1:].std() > 1e-12 else 0.0
                vals[col] = c1 - c2 if not np.isnan(c1) and not np.isnan(c2) else 0.0
            if vals:
                leads[dt] = pd.Series(vals)
        if not leads:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(leads).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 319. intraday_elasticity_path — 价量弹性日内演化
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_elasticity_path_20d", category="intraday_advanced")
class IntradayElasticityPath20d(Factor):
    """价量弹性日内演化因子.

    后半段价量弹性 - 前半段价量弹性 (弹性=|ret|/vol, 与 #26 全天均值互补).
    弹性恶化 (后半段弹性升) → 流动性变差 → 负向.
    方向: 负向.
    """
    name = "intraday_elasticity_path_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "弹性日内演化 (后半-前半|ret|/vol, 恶化=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, volume = panel["close"], panel["volume"]
        ret_abs = close.pct_change().abs()
        day = ret_abs.index.normalize()
        paths: dict = {}
        for dt in sorted(set(day)):
            grp_r = ret_abs.loc[day == dt]
            grp_v = volume.loc[day == dt]
            n = len(grp_r)
            if n < 40:
                continue
            mid = n // 2
            vals = {}
            for col in grp_r.columns:
                r = grp_r[col].dropna()
                v = grp_v[col].dropna()
                common = r.index.intersection(v.index)
                if len(common) < 40:
                    continue
                r_c = r.loc[common]
                v_c = v.loc[common]
                el = r_c / (v_c + 1e-12)
                first = el.iloc[:mid].mean()
                second = el.iloc[mid:].mean()
                vals[col] = -float(second - first)
            if vals:
                paths[dt] = pd.Series(vals)
        if not paths:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(paths).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 320. intraday_trend_slope_stability — 趋势斜率稳定性
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_trend_slope_stability_20d", category="intraday_advanced")
class IntradayTrendSlopeStability20d(Factor):
    """趋势斜率稳定性因子.

    日内分段线性斜率的符号一致性 (斜率稳定度, 与 #114 R² 互补: 这里看分段斜率方向).
    分段斜率同向 → 趋势稳定推进 → 正向.
    方向: 正向.
    """
    name = "intraday_trend_slope_stability_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "趋势斜率稳定 (分段斜率同向数/4)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        stabilities: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            n = len(grp)
            if n < 60:
                continue
            q = n // 4
            if q < 5:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < 60:
                    continue
                segs = [c.iloc[i * q:(i + 1) * q] for i in range(4)]
                slopes = []
                for s in segs:
                    if len(s) < 5:
                        continue
                    t = np.arange(len(s))
                    sl = np.polyfit(t, s.values, 1)[0]
                    slopes.append(sl)
                if not slopes:
                    vals[col] = 0.0
                    continue
                dom = np.sign(np.mean(slopes))
                if abs(dom) < 1e-12:
                    vals[col] = 0.0
                    continue
                consistent = sum(1 for sl in slopes if np.sign(sl) == dom and abs(sl) > 1e-12)
                vals[col] = float(consistent / len(slopes))
            if vals:
                stabilities[dt] = pd.Series(vals)
        if not stabilities:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(stabilities).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 321. intraday_seat_top5_long_conc — 前5多头集中度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_seat_top5_long_conc_20d", category="intraday_advanced")
class IntradaySeatTop5LongConc20d(Factor):
    """前5多头集中度因子.

    前5席位多头持仓 / 全部席位多头 (derive_product_seat 按席位, 结构视角).
    多头集中 → 主力控盘 → 正向.
    方向: 正向.
    """
    name = "intraday_seat_top5_long_conc_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "前5多头集中度 (top5多头/全部, 主力控盘=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        df = _get_seat_table(data, dates, universe, "product_seat")
        if df is None or "long_position" not in df.columns:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        pivot = _seat_topN_ratio(df, "long_position", n=5)
        return pivot.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 322. intraday_seat_top5_short_conc — 前5空头集中度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_seat_top5_short_conc_20d", category="intraday_advanced")
class IntradaySeatTop5ShortConc20d(Factor):
    """前5空头集中度因子.

    前5席位空头持仓 / 全部席位空头.
    空头集中 → 空头主力控盘 → 负向.
    方向: 负向.
    """
    name = "intraday_seat_top5_short_conc_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "前5空头集中度 (top5空头/全部, 空头控盘=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        df = _get_seat_table(data, dates, universe, "product_seat")
        if df is None or "short_position" not in df.columns:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        pivot = _seat_topN_ratio(df, "short_position", n=5)
        return (-pivot).reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 323. intraday_seat_conc_bias — 多空集中度差
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_seat_conc_bias_20d", category="intraday_advanced")
class IntradaySeatConcBias20d(Factor):
    """多空集中度差因子.

    前5多头集中度 - 前5空头集中度 (哪方更集中).
    多头更集中 → 多方控盘 → 正向.
    方向: 正向.
    """
    name = "intraday_seat_conc_bias_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "多空集中度差 (top5多头-空头集中度)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        df = _get_seat_table(data, dates, universe, "product_seat")
        if df is None or "long_position" not in df.columns or "short_position" not in df.columns:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        long_conc = _seat_topN_ratio(df, "long_position", n=5)
        short_conc = _seat_topN_ratio(df, "short_position", n=5)
        bias = long_conc - short_conc
        return bias.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 324. intraday_seat_leader_net — 龙头席位净多头
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_seat_leader_net_20d", category="intraday_advanced")
class IntradaySeatLeaderNet20d(Factor):
    """龙头席位净多头因子.

    净持仓最大席位(龙头)的 net_position (结构视角: 龙头席位动向).
    龙头净多 → 主导资金看多 → 正向.
    方向: 正向.
    """
    name = "intraday_seat_leader_net_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "龙头席位净多 (net最大席位的net_position)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        df = _get_seat_table(data, dates, universe, "product_seat")
        if df is None or "net_position" not in df.columns:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        data = df[["_ts", "_root", "net_position"]].dropna()
        leader = data.loc[data.groupby(["_ts", "_root"])["net_position"].idxmax()]
        pivot = leader.pivot(index="_ts", columns="_root", values="net_position")
        pivot.index = pd.DatetimeIndex(pivot.index)
        return pivot.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 325. intraday_seat_leader_change — 龙头席位净多变化
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_seat_leader_change_20d", category="intraday_advanced")
class IntradaySeatLeaderChange20d(Factor):
    """龙头席位净多变化因子.

    龙头席位净多头 (net最大席位) 的跨日变化 (龙头资金动向).
    龙头加多 → 主导资金增强 → 正向.
    方向: 正向.
    """
    name = "intraday_seat_leader_change_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "龙头席位净多变化 (龙头加多=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        df = _get_seat_table(data, dates, universe, "product_seat")
        if df is None or "net_position" not in df.columns:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        data = df[["_ts", "_root", "net_position"]].dropna()
        leader = data.loc[data.groupby(["_ts", "_root"])["net_position"].idxmax()]
        pivot = leader.pivot(index="_ts", columns="_root", values="net_position")
        pivot.index = pd.DatetimeIndex(pivot.index)
        chg = pivot.diff()
        return chg.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 326. intraday_seat_long_short_seat_ratio — 净多/净空席位数量比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_seat_long_short_seat_ratio_20d", category="intraday_advanced")
class IntradaySeatLongShortSeatRatio20d(Factor):
    """净多/净空席位数量比因子.

    净多头席位数量 / 净空头席位数量 (多空参与度结构).
    净多席位多 → 多头参与广泛 → 正向.
    方向: 正向.
    """
    name = "intraday_seat_long_short_seat_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "净多/净空席位数量比 (多头参与广=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        df = _get_seat_table(data, dates, universe, "product_seat")
        if df is None or "net_position" not in df.columns:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        data = df[["_ts", "_root", "net_position"]].dropna()
        n_long = data[data["net_position"] > 0].groupby(["_ts", "_root"]).size()
        n_short = data[data["net_position"] < 0].groupby(["_ts", "_root"]).size()
        ratio = (n_long / n_short.replace(0, np.nan)).reset_index(name="v")
        pivot = ratio.pivot(index="_ts", columns="_root", values="v")
        pivot.index = pd.DatetimeIndex(pivot.index)
        return pivot.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 327. intraday_seat_top10_net_share — 前10席位净持仓占比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_seat_top10_net_share_20d", category="intraday_advanced")
class IntradaySeatTop10NetShare20d(Factor):
    """前10席位净持仓占比因子.

    前10席位净持仓绝对值 / 全部席位净持仓绝对值 (净持仓的集中度).
    集中度高 → 少数席位主导方向 → 方向性强 → 正向.
    方向: 正向.
    """
    name = "intraday_seat_top10_net_share_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "前10净持仓占比 (少数席位主导=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        df = _get_seat_table(data, dates, universe, "product_seat")
        if df is None or "net_position" not in df.columns:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        data = df[["_ts", "_root", "net_position"]].dropna()
        total = data["net_position"].abs().groupby([data["_ts"], data["_root"]]).sum()
        top = (data.assign(_abs=data["net_position"].abs())
               .sort_values(["_ts", "_root", "_abs"], ascending=[True, True, False])
               .groupby(["_ts", "_root"]).head(10).groupby(["_ts", "_root"])["_abs"].sum())
        ratio = (top / total.replace(0, np.nan)).reset_index(name="v")
        pivot = ratio.pivot(index="_ts", columns="_root", values="v")
        pivot.index = pd.DatetimeIndex(pivot.index)
        return pivot.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 328. intraday_seat_main_net_oi — 主力合约净头寸/OI
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_seat_main_net_oi_20d", category="intraday_advanced")
class IntradaySeatMainNetOi20d(Factor):
    """主力合约净头寸/OI 因子.

    Σ席位净头寸 / open_interest (derive_main_contract_seat 含 OI).
    净头寸占 OI 比高 → 前20席位主导方向 → 正向.
    方向: 正向.
    """
    name = "intraday_seat_main_net_oi_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "主力净头寸/OI (Σnet/OI, 席位主导=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        df = _get_seat_table(data, dates, universe, "main_contract_seat")
        if df is None or "net_position" not in df.columns or "open_interest" not in df.columns:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        net = _seat_to_panel(df, "net_position", agg="sum")
        oi = _seat_to_panel(df, "open_interest", agg="last")
        ratio = net / oi.replace(0, np.nan)
        return ratio.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 329. intraday_seat_main_top5_oi — 主力前5净敞口/OI
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_seat_main_top5_oi_20d", category="intraday_advanced")
class IntradaySeatMainTop5Oi20d(Factor):
    """主力前5净敞口/OI 因子.

    前5净持仓绝对值 / open_interest (主力合约集中度×OI 归一).
    集中净敞口高 → 少数主力主导 → 方向强 → 正向.
    方向: 正向.
    """
    name = "intraday_seat_main_top5_oi_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "主力前5净敞口/OI (集中主导=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        df = _get_seat_table(data, dates, universe, "main_contract_seat")
        if df is None or "net_position" not in df.columns or "open_interest" not in df.columns:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        data = df[["_ts", "_root", "net_position", "open_interest"]].dropna()
        top5 = (data.assign(_abs=data["net_position"].abs())
                .sort_values(["_ts", "_root", "_abs"], ascending=[True, True, False])
                .groupby(["_ts", "_root"]).head(5).groupby(["_ts", "_root"])["_abs"].sum())
        oi = data.groupby(["_ts", "_root"])["open_interest"].last()
        ratio = (top5 / oi.replace(0, np.nan)).reset_index(name="v")
        pivot = ratio.pivot(index="_ts", columns="_root", values="v")
        pivot.index = pd.DatetimeIndex(pivot.index)
        return pivot.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 330. intraday_seat_main_leader_change — 主力龙头席位变化
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_seat_main_leader_change_20d", category="intraday_advanced")
class IntradaySeatMainLeaderChange20d(Factor):
    """主力龙头席位变化因子.

    主力合约净多最大席位的跨日变化 (主力龙头资金动向, 用 main_contract_seat).
    龙头加多 → 主力资金增强 → 正向.
    方向: 正向.
    """
    name = "intraday_seat_main_leader_change_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "主力龙头席位变化 (主力龙头净多跨日差)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        df = _get_seat_table(data, dates, universe, "main_contract_seat")
        if df is None or "net_position" not in df.columns:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        data = df[["_ts", "_root", "net_position"]].dropna()
        leader = data.loc[data.groupby(["_ts", "_root"])["net_position"].idxmax()]
        pivot = leader.pivot(index="_ts", columns="_root", values="net_position")
        pivot.index = pd.DatetimeIndex(pivot.index)
        chg = pivot.diff()
        return chg.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 331. intraday_seat_delivery_activity — 交割活跃度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_seat_delivery_activity_20d", category="intraday_advanced")
class IntradaySeatDeliveryActivity20d(Factor):
    """交割活跃度因子.

    交割接货+交货总量 (delivery_summary, 交割月现货参与度).
    交割活跃 → 现货端博弈激烈 → 正 (现货紧张) 或负 (交割压力) — 先验正向.
    注意: 交割数据稀疏 (仅到期合约), 多数日期 NaN 自动降级.
    方向: 正向.
    """
    name = "intraday_seat_delivery_activity_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "交割活跃度 (receive+deliver量, 稀疏数据)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        df = _get_seat_table(data, dates, universe, "delivery_summary")
        if df is None or "receive_quantity" not in df.columns or "deliver_quantity" not in df.columns:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        activity = (df["receive_quantity"].fillna(0) + df["deliver_quantity"].fillna(0))
        panel = (pd.DataFrame({"_ts": df["_ts"], "_root": df["_root"], "v": activity})
                 .groupby(["_ts", "_root"])["v"].sum().reset_index()
                 .pivot(index="_ts", columns="_root", values="v"))
        panel.index = pd.DatetimeIndex(panel.index)
        return panel.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 332. intraday_seat_delivery_net — 交割净头寸
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_seat_delivery_net_20d", category="intraday_advanced")
class IntradaySeatDeliveryNet20d(Factor):
    """交割净头寸因子.

    非期货净头寸 (non_futures_net = 接货-交货, delivery_summary).
    接货主导 → 现货需求强 → 正向.
    方向: 正向.
    """
    name = "intraday_seat_delivery_net_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "交割净头寸 (non_futures_net, 接货主导=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        df = _get_seat_table(data, dates, universe, "delivery_summary")
        if df is None or "non_futures_net" not in df.columns:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        panel = (pd.DataFrame({"_ts": df["_ts"], "_root": df["_root"], "v": df["non_futures_net"].fillna(0)})
                 .groupby(["_ts", "_root"])["v"].sum().reset_index()
                 .pivot(index="_ts", columns="_root", values="v"))
        panel.index = pd.DatetimeIndex(panel.index)
        return panel.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 333. intraday_seat_raw_conc_change — 原始席位集中度变化
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_seat_raw_conc_change_20d", category="intraday_advanced")
class IntradaySeatRawConcChange20d(Factor):
    """原始席位集中度变化因子.

    原始席位 (raw_seat_position, 按合约粒度) 前5多头集中度的跨日变化.
    多头集中度上升 → 主力吸筹 → 正向.
    方向: 正向.
    """
    name = "intraday_seat_raw_conc_change_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "原始席位集中度变化 (raw表top5多头集中度跨日差)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        df = _get_seat_table(data, dates, universe, "raw_seat")
        if df is None or "long_position" not in df.columns:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        conc = _seat_topN_ratio(df, "long_position", n=5)
        chg = conc.diff()
        return chg.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 334. intraday_seat_raw_net_dispersion — 原始席位净头寸分化度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_seat_raw_net_dispersion_20d", category="intraday_advanced")
class IntradaySeatRawNetDispersion20d(Factor):
    """原始席位净头寸分化度因子.

    原始席位 net_position 的截面标准差 (席位间多空分化度).
    分化大 → 席位方向不一 → 分歧 → 负向.
    方向: 负向.
    """
    name = "intraday_seat_raw_net_dispersion_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "席位净头寸分化 (raw表net的std, 分歧=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        df = _get_seat_table(data, dates, universe, "raw_seat")
        if df is None or "net_position" not in df.columns:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        pivot = _seat_to_panel(df, "net_position", agg="std")
        return (-pivot).reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 335. intraday_seat_main_conc_price — 主力集中度×价格协同
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_seat_main_conc_price_20d", category="intraday_advanced")
class IntradaySeatMainConcPrice20d(Factor):
    """主力集中度×价格协同因子 (表间共用: main_contract_seat 的集中度 × 价格).

    前5多头集中度变化方向 × 价格变化方向 (集中度上升伴随上涨 = 主力吸筹推动).
    协同 → 集中度与价格共振 → 正向.
    方向: 正向.
    """
    name = "intraday_seat_main_conc_price_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "主力集中度×价格协同 (集中度变化×价格方向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        df = _get_seat_table(data, dates, universe, "main_contract_seat")
        if df is None or "long_position" not in df.columns or "close" not in df.columns:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        conc = _seat_topN_ratio(df, "long_position", n=5)
        conc_chg = np.sign(conc.diff())
        close = _seat_to_panel(df, "close", agg="last")
        price_chg = np.sign(close.diff())
        agree = (conc_chg == price_chg).astype(float)
        # 只用两者都非零的日子
        valid = (conc_chg != 0) & (price_chg != 0)
        score = agree.where(valid).rolling(20, min_periods=5).mean()
        return score.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 336. intraday_vol_liq_interaction — 波动-流动性交互状态
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_vol_liq_interaction_20d", category="intraday_advanced")
class IntradayVolLiqInteraction20d(Factor):
    """波动-流动性交互状态因子.

    高波动(|ret|>中位) 且 高冲击(Amihud>中位) 的分钟占比 (方向A: 波动×流动性交互).
    波动与冲击共振 → 脆弱市场 → 负向.
    方向: 负向.
    """
    name = "intraday_vol_liq_interaction_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "波动×流动性交互 (高波动+高Amihud占比, 脆弱=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "amount"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, amount = panel["close"], panel["amount"]
        ret_abs = close.pct_change().abs()
        amihud = ret_abs / (amount + 1e-12)
        day = ret_abs.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp_r = ret_abs.loc[day == dt]
            grp_a = amihud.loc[day == dt]
            if len(grp_r) < 20:
                continue
            vals = {}
            for col in grp_r.columns:
                r = grp_r[col].dropna()
                a = grp_a[col].dropna()
                common = r.index.intersection(a.index)
                if len(common) < 20:
                    continue
                r_c = r.loc[common]
                a_c = a.loc[common]
                both = (r_c > r_c.median()) & (a_c > a_c.median())
                vals[col] = -float(both.sum() / len(r_c))
            if vals:
                interactions[dt] = pd.Series(vals)
        if not interactions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(interactions).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 337. intraday_amihud_vol_ratio — Amihud/波动率比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_amihud_vol_ratio_20d", category="intraday_advanced")
class IntradayAmihudVolRatio20d(Factor):
    """Amihud/波动率比因子.

    日内 Amihud 均值 / 日内波动率 (单位波动的冲击成本, 方向A交互).
    比值高 → 波动由低流动性放大 → 脆弱 → 负向.
    方向: 负向.
    """
    name = "intraday_amihud_vol_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "Amihud/波动率比 (单位波动的冲击, 高=脆弱=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "amount"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, amount = panel["close"], panel["amount"]
        ret_abs = close.pct_change().abs()
        amihud = ret_abs / (amount + 1e-12)
        day = ret_abs.index.normalize()
        ratios: dict = {}
        for dt in sorted(set(day)):
            grp_r = ret_abs.loc[day == dt]
            grp_a = amihud.loc[day == dt]
            if len(grp_r) < 20:
                continue
            vals = {}
            for col in grp_r.columns:
                r = grp_r[col].dropna()
                a = grp_a[col].dropna()
                common = r.index.intersection(a.index)
                if len(common) < 20:
                    continue
                r_c = r.loc[common]
                a_c = a.loc[common]
                a_c = a_c[a_c < 10.0]  # 剔除极端
                vol = r_c.std(ddof=0)
                if len(a_c) < 10 or vol < 1e-12:
                    continue
                vals[col] = -float(a_c.mean() / vol)
            if vals:
                ratios[dt] = pd.Series(vals)
        if not ratios:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(ratios).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 338. intraday_liq_dry_in_vol — 波动期流动性恶化度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_liq_dry_in_vol_20d", category="intraday_advanced")
class IntradayLiqDryInVol20d(Factor):
    """波动期流动性恶化度因子.

    高波动分钟的 Amihud 均值 / 低波动分钟的 Amihud 均值 (波动时流动性是否恶化).
    恶化比高 → 波动引发流动性枯竭 → 脆弱 → 负向.
    方向: 负向.
    """
    name = "intraday_liq_dry_in_vol_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "波动期流动性恶化 (高波Amihud/低波Amihud, 枯竭=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "amount"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, amount = panel["close"], panel["amount"]
        ret_abs = close.pct_change().abs()
        amihud = ret_abs / (amount + 1e-12)
        day = ret_abs.index.normalize()
        ratios: dict = {}
        for dt in sorted(set(day)):
            grp_r = ret_abs.loc[day == dt]
            grp_a = amihud.loc[day == dt]
            if len(grp_r) < 20:
                continue
            vals = {}
            for col in grp_r.columns:
                r = grp_r[col].dropna()
                a = grp_a[col].dropna()
                common = r.index.intersection(a.index)
                if len(common) < 20:
                    continue
                r_c = r.loc[common]
                a_c = a.loc[common].clip(upper=10.0)
                med = r_c.median()
                a_high = a_c[r_c > med]
                a_low = a_c[r_c <= med]
                if len(a_high) < 3 or len(a_low) < 3:
                    vals[col] = 1.0
                    continue
                low_mean = a_low.mean()
                vals[col] = -float(a_high.mean() / low_mean) if low_mean > 1e-12 else -1.0
            if vals:
                ratios[dt] = pd.Series(vals)
        if not ratios:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(ratios).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 339. intraday_vol_breakout_liq — 波动突破时流动性
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_vol_breakout_liq_20d", category="intraday_advanced")
class IntradayVolBreakoutLiq20d(Factor):
    """波动突破时流动性因子.

    今日波动突破(>10日均)时的 Amihud 水平 (波动突破日的流动性状态, 方向A交互).
    突破日流动性差 → 冲击放大 → 负向.
    方向: 负向.
    """
    name = "intraday_vol_breakout_liq_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "波动突破时流动性 (突破日的Amihud, 差=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "amount"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, amount = panel["close"], panel["amount"]
        ret_abs = close.pct_change().abs()
        amihud = ret_abs / (amount + 1e-12)
        day = ret_abs.index.normalize()
        daily: dict = {}
        for dt in sorted(set(day)):
            grp_r = ret_abs.loc[day == dt]
            grp_a = amihud.loc[day == dt]
            if len(grp_r) < 20:
                continue
            vals = {}
            for col in grp_r.columns:
                r = grp_r[col].dropna()
                a = grp_a[col].dropna()
                common = r.index.intersection(a.index)
                if len(common) < 20:
                    continue
                a_c = a.loc[common].clip(upper=10.0)
                vals[col] = float(a_c.mean())
            if vals:
                daily[dt] = pd.Series(vals)
        if not daily:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily).T
        df.index = pd.DatetimeIndex(df.index)
        return (-df).reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 340. intraday_amihud_vol_corr — Amihud-波动相关
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_amihud_vol_corr_20d", category="intraday_advanced")
class IntradayAmihudVolCorr20d(Factor):
    """Amihud-波动相关因子.

    corr(Amihud_1m, |ret|_1m) (方向A: 冲击与波动的分钟级联动).
    正相关 → 波动越大冲击越大 → 脆弱 → 负向.
    方向: 负向.
    """
    name = "intraday_amihud_vol_corr_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "Amihud-波动相关 (冲击随波动放大=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "amount"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, amount = panel["close"], panel["amount"]
        ret_abs = close.pct_change().abs()
        amihud = ret_abs / (amount + 1e-12)
        day = ret_abs.index.normalize()
        corrs: dict = {}
        for dt in sorted(set(day)):
            grp_r = ret_abs.loc[day == dt]
            grp_a = amihud.loc[day == dt]
            if len(grp_r) < 30:
                continue
            vals = {}
            for col in grp_r.columns:
                r = grp_r[col].dropna()
                a = grp_a[col].dropna()
                common = r.index.intersection(a.index)
                if len(common) < 30:
                    continue
                r_c = r.loc[common]
                a_c = a.loc[common].clip(upper=10.0)
                if r_c.std(ddof=0) < 1e-12 or a_c.std(ddof=0) < 1e-12:
                    vals[col] = 0.0
                else:
                    corr_val = float(np.corrcoef(r_c, a_c)[0, 1])
                    vals[col] = -corr_val if not np.isnan(corr_val) else 0.0
            if vals:
                corrs[dt] = pd.Series(vals)
        if not corrs:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(corrs).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 341. intraday_cross_strength — 截面相对强弱
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_cross_strength_20d", category="intraday_advanced")
class IntradayCrossStrength20d(Factor):
    """截面相对强弱因子 (方向B: 跨品种联动).

    品种日内收益相对当日全部品种均值的截面 z-score.
    相对强势 → 板块内领涨 → 正向.
    方向: 正向.
    """
    name = "intraday_cross_strength_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "截面相对强弱 (日内收益的截面z)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"open", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        open_px, close = panel["open"], panel["close"]
        day = close.index.normalize()
        daily_ret: dict = {}
        for dt in sorted(set(day)):
            grp_o = open_px.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 5:
                continue
            vals = {}
            for col in grp_c.columns:
                o = grp_o[col].dropna()
                c = grp_c[col].dropna()
                if len(o) < 1 or len(c) < 5:
                    continue
                o_first = o.iloc[0]
                if o_first < 1e-12:
                    continue
                vals[col] = float(c.iloc[-1] / o_first - 1.0)
            if vals:
                daily_ret[dt] = pd.Series(vals)
        if not daily_ret:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily_ret).T
        df.index = pd.DatetimeIndex(df.index)
        z = _cs_zscore(df.reindex(index=dates))
        return z.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 342. intraday_cross_vol — 截面相对波动
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_cross_vol_20d", category="intraday_advanced")
class IntradayCrossVol20d(Factor):
    """截面相对波动因子.

    品种日内波动相对当日全部品种的截面 z-score (方向B).
    相对高波动 → 截面内异常波动 → 负向.
    方向: 负向.
    """
    name = "intraday_cross_vol_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "截面相对波动 (日内波动的截面z, 异常=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        daily_vol: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 20:
                    continue
                vals[col] = float(r.std(ddof=0))
            if vals:
                daily_vol[dt] = pd.Series(vals)
        if not daily_vol:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily_vol).T
        df.index = pd.DatetimeIndex(df.index)
        z = _cs_zscore(df.reindex(index=dates))
        return (-z).reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 343. intraday_cross_momentum — 截面相对动量
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_cross_momentum_20d", category="intraday_advanced")
class IntradayCrossMomentum20d(Factor):
    """截面相对动量因子.

    品种5日收益相对全部品种的截面 z-score (方向B, 跨日相对动量).
    相对动量强 → 板块内趋势领先 → 正向.
    方向: 正向.
    """
    name = "intraday_cross_momentum_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "截面相对动量 (5日收益的截面z)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        if close is None or close.empty:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        mom5 = close.pct_change(5)
        z = mom5.apply(lambda row: (row - row.mean()) / (row.std(ddof=0) + 1e-12), axis=1)
        return z.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 344. intraday_cross_position — 截面相对收盘位置
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_cross_position_20d", category="intraday_advanced")
class IntradayCrossPosition20d(Factor):
    """截面相对收盘位置因子.

    品种收盘位置 (收盘在日内区间位置) 相对全部品种的截面 z-score (方向B).
    相对高位收盘 → 板块内最强势 → 正向.
    方向: 正向.
    """
    name = "intraday_cross_position_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "截面相对收盘位置 (收盘位置的截面z)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"high", "low", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, low, close = panel["high"], panel["low"], panel["close"]
        day = close.index.normalize()
        daily_pos: dict = {}
        for dt in sorted(set(day)):
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 20:
                continue
            vals = {}
            for col in grp_c.columns:
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                c = grp_c[col].dropna()
                if len(h) < 10 or len(l) < 10 or len(c) < 10:
                    continue
                rng = h.max() - l.min()
                if rng < 1e-12:
                    vals[col] = 0.5
                    continue
                vals[col] = float((c.iloc[-1] - l.min()) / rng)
            if vals:
                daily_pos[dt] = pd.Series(vals)
        if not daily_pos:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily_pos).T
        df.index = pd.DatetimeIndex(df.index)
        z = _cs_zscore(df.reindex(index=dates))
        return z.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 345. intraday_cross_reversal — 截面相对反转
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_cross_reversal_20d", category="intraday_advanced")
class IntradayCrossReversal20d(Factor):
    """截面相对反转因子.

    昨日截面相对强度取负 (板块内强弱反转, 方向B).
    昨日强势今日回落 → 相对反转 → 负向 (输出 -昨日z).
    方向: 负向.
    """
    name = "intraday_cross_reversal_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "截面相对反转 (昨日截面强度取负)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"open", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        open_px, close = panel["open"], panel["close"]
        day = close.index.normalize()
        daily_ret: dict = {}
        for dt in sorted(set(day)):
            grp_o = open_px.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 5:
                continue
            vals = {}
            for col in grp_c.columns:
                o = grp_o[col].dropna()
                c = grp_c[col].dropna()
                if len(o) < 1 or len(c) < 5:
                    continue
                o_first = o.iloc[0]
                if o_first < 1e-12:
                    continue
                vals[col] = float(c.iloc[-1] / o_first - 1.0)
            if vals:
                daily_ret[dt] = pd.Series(vals)
        if not daily_ret:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily_ret).T
        df.index = pd.DatetimeIndex(df.index)
        z = _cs_zscore(df.reindex(index=dates))
        rev = z.shift(1)
        return (-rev).reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 346. intraday_big_trade_timing — 大单时间分布均匀性
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_big_trade_timing_20d", category="intraday_advanced")
class IntradayBigTradeTiming20d(Factor):
    """大单时间分布均匀性因子 (方向C: 大单占比时序).

    大单分钟(量>1.5x均量)的间隔标准差 (大单在一天中的分布均匀性).
    间隔均匀 → 大单持续规律 → 机构节奏 → 正向; 扎堆 → 脉冲 → 负向.
    方向: 正向 (取负间隔std).
    """
    name = "intraday_big_trade_timing_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "大单时间分布 (大单间隔std, 均匀=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "volume" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        volume = panel["volume"]
        day = volume.index.normalize()
        timings: dict = {}
        for dt in sorted(set(day)):
            grp = volume.loc[day == dt]
            if len(grp) < 30:
                continue
            vals = {}
            for col in grp.columns:
                v = grp[col].dropna()
                if len(v) < 30:
                    continue
                v_mean = v.mean()
                if v_mean < 1e-12:
                    continue
                idx = np.where((v > 1.5 * v_mean).values)[0]
                if len(idx) < 3:
                    vals[col] = 0.0
                    continue
                gaps = np.diff(idx)
                vals[col] = -float(gaps.std(ddof=0))
            if vals:
                timings[dt] = pd.Series(vals)
        if not timings:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(timings).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 347. intraday_signed_imbalance_z — 主动买卖失衡突增
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_signed_imbalance_z_20d", category="intraday_advanced")
class IntradaySignedImbalanceZ20d(Factor):
    """主动买卖失衡突增因子 (方向C: 订单簿失衡代理).

    分钟主动买卖不平衡 (sign(ret)×vol) 的20日 z-score (借鉴 z-score 突增识别).
    不平衡异常放大 → 单边订单流冲击 → 信息驱动 → 正向.
    方向: 正向.
    """
    name = "intraday_signed_imbalance_z_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "主动失衡突增 (不平衡z-score, 订单流冲击=正向)"
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
        daily_imb: dict = {}
        for dt in sorted(set(day)):
            grp_r = ret_1m.loc[day == dt]
            grp_v = volume.loc[day == dt]
            if len(grp_r) < 20:
                continue
            vals = {}
            for col in grp_r.columns:
                r = grp_r[col].dropna()
                v = grp_v[col].dropna()
                common = r.index.intersection(v.index)
                if len(common) < 20:
                    continue
                imb = np.sign(r.loc[common].values) * v.loc[common].values
                vals[col] = float(np.sum(imb))
            if vals:
                daily_imb[dt] = pd.Series(vals)
        if not daily_imb:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily_imb).T
        df.index = pd.DatetimeIndex(df.index)
        mean = df.rolling(20, min_periods=5).mean()
        std = df.rolling(20, min_periods=5).std().replace(0, np.nan)
        z = (df - mean) / std
        return z.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 348. intraday_imbalance_vol — 失衡-波动关系
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_imbalance_vol_20d", category="intraday_advanced")
class IntradayImbalanceVol20d(Factor):
    """失衡-波动关系因子 (方向C).

    主动买卖不平衡绝对值与波动率的关系 (不平衡大时的波动放大程度).
    不平衡伴随高波动 → 单边冲击引发动荡 → 负向.
    方向: 负向.
    """
    name = "intraday_imbalance_vol_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "失衡-波动关系 (|不平衡|大时波动放大=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, volume = panel["close"], panel["volume"]
        ret_1m = close.pct_change()
        ret_abs = ret_1m.abs()
        day = ret_1m.index.normalize()
        relations: dict = {}
        for dt in sorted(set(day)):
            grp_r = ret_1m.loc[day == dt]
            grp_a = ret_abs.loc[day == dt]
            grp_v = volume.loc[day == dt]
            if len(grp_r) < 30:
                continue
            vals = {}
            for col in grp_r.columns:
                r = grp_r[col].dropna()
                a = grp_a[col].dropna()
                v = grp_v[col].dropna()
                common = r.index.intersection(a.index).intersection(v.index)
                if len(common) < 30:
                    continue
                r_c = r.loc[common]
                a_c = a.loc[common]
                v_c = v.loc[common]
                imb = np.sign(r_c.values) * v_c.values
                imb_abs = np.abs(imb)
                med_imb = np.median(imb_abs)
                if med_imb < 1e-12:
                    vals[col] = 0.0
                    continue
                high_imb = a_c[imb_abs > med_imb]
                low_imb = a_c[imb_abs <= med_imb]
                if len(high_imb) < 3 or len(low_imb) < 3:
                    vals[col] = 0.0
                    continue
                vals[col] = -float(high_imb.mean() / low_imb.mean())
            if vals:
                relations[dt] = pd.Series(vals)
        if not relations:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(relations).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 349. intraday_gap_confirm — 跳空开盘量确认
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_gap_confirm_20d", category="intraday_advanced")
class IntradayGapConfirm20d(Factor):
    """跳空开盘量确认因子 (多时点交互: 跳空×开盘量).

    跳空幅度 × 开盘30分钟量能确认 (跳空大且开盘放量 = 有效信息定价).
    跳空+放量 → 隔夜信息被资金确认 → 正向.
    方向: 正向.
    """
    name = "intraday_gap_confirm_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "跳空开盘量确认 (跳空×开盘量, 信息确认=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"open", "close", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        open_px, close, volume = panel["open"], panel["close"], panel["volume"]
        day = close.index.normalize()
        confirms: dict = {}
        prev_close: dict = {}
        for dt in sorted(set(day)):
            grp_o = open_px.loc[day == dt]
            grp_c = close.loc[day == dt]
            grp_v = volume.loc[day == dt]
            if len(grp_c) < 30:
                continue
            n_open = max(10, min(30, len(grp_c) // 4))
            vals = {}
            for col in grp_c.columns:
                o = grp_o[col].dropna()
                c = grp_c[col].dropna()
                v = grp_v[col].dropna()
                common = o.index.intersection(c.index).intersection(v.index)
                if len(common) < 30:
                    continue
                o_c, c_c, v_c = (x.loc[common] for x in (o, c, v))
                prev_c = prev_close.get(col)
                if prev_c is None or prev_c < 1e-12:
                    prev_close[col] = c_c.iloc[-1]
                    continue
                gap = abs(o_c.iloc[0] / prev_c - 1.0)
                open_vol_share = v_c.iloc[:n_open].sum() / v_c.sum()
                vals[col] = float(gap * open_vol_share * 100.0)
                prev_close[col] = c_c.iloc[-1]
            if vals:
                confirms[dt] = pd.Series(vals)
        if not confirms:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(confirms).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 350. intraday_open_state_trend — 开盘状态条件趋势
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_open_state_trend_20d", category="intraday_advanced")
class IntradayOpenStateTrend20d(Factor):
    """开盘状态条件趋势因子 (多时点交互: 开盘波动状态×日内趋势).

    开盘30分钟波动状态 (高/低) 下的日内趋势方向 (开盘安静时趋势质量更高).
    开盘低波动+日内趋势明确 → 有序趋势 → 正向.
    方向: 正向.
    """
    name = "intraday_open_state_trend_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "开盘状态条件趋势 (低波开盘的日内趋势=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        trends: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            n = len(grp)
            if n < 40:
                continue
            n_open = max(10, min(30, n // 4))
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 40:
                    continue
                open_vol = r.iloc[:n_open].std(ddof=0)
                day_vol = r.std(ddof=0)
                if open_vol < 1e-12 or day_vol < 1e-12:
                    continue
                # 开盘低波动: 开盘波动 < 日内中位波动
                low_open = open_vol < np.median(np.abs(r).values)
                day_dir = np.sign(r.sum())
                if abs(day_dir) < 1e-12:
                    vals[col] = 0.0
                    continue
                up_share = (r > 0).sum() / (r != 0).sum()
                aligned = up_share if day_dir > 0 else (1.0 - up_share)
                # 低波开盘时趋势更可信: 用低波状态加权
                vals[col] = float(aligned) if low_open else float(aligned) * 0.5
            if vals:
                trends[dt] = pd.Series(vals)
        if not trends:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(trends).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)

# 351. intraday_liq_shock_freq — 流动性冲击频率
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_liq_shock_freq_20d", category="intraday_advanced")
class IntradayLiqShockFreq20d(Factor):
    """流动性冲击频率因子 (Amihud 突增>μ+σ 的分钟占比). 高频冲击=流动性脆弱, 负向.
    方向: 负向.
    """
    name = "intraday_liq_shock_freq_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "Amihud突增频率 (脆弱=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "amount"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, amount = panel["close"], panel["amount"]
        ret_abs = close.pct_change().abs()
        amihud = ret_abs / (amount + 1e-12)
        day = ret_abs.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp_a = amihud.loc[day == dt]
            if len(grp_a) < 30:
                continue
            vals = {}
            for col in grp_a.columns:
                a = grp_a[col].dropna()
                if len(a) < 30:
                    continue
                mu, sigma = a.mean(), a.std(ddof=0)
                if sigma == 0 or pd.isna(sigma):
                    continue
                vals[col] = float((a > mu + sigma).mean())
            if vals:
                interactions[dt] = pd.Series(vals)
        if not interactions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(interactions).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# 352. intraday_amihud_slope — Amihud 日内斜率
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_amihud_slope_20d", category="intraday_advanced")
class IntradayAmihudSlope20d(Factor):
    """Amihud 日内演化斜率因子 (后半段均值 - 前半段均值). 冲击恶化=流动性恶化, 负向.
    方向: 负向.
    """
    name = "intraday_amihud_slope_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "Amihud日内恶化斜率 (负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "amount"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, amount = panel["close"], panel["amount"]
        ret_abs = close.pct_change().abs()
        amihud = ret_abs / (amount + 1e-12)
        day = ret_abs.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp_a = amihud.loc[day == dt]
            if len(grp_a) < 60:
                continue
            vals = {}
            for col in grp_a.columns:
                a = grp_a[col].dropna()
                if len(a) < 60:
                    continue
                half = len(a) // 2
                vals[col] = float(a.iloc[half:].mean() - a.iloc[:half].mean())
            if vals:
                interactions[dt] = pd.Series(vals)
        if not interactions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(interactions).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# 353. intraday_vol_liq_divergence — 波动-流动性背离
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_vol_liq_divergence_20d", category="intraday_advanced")
class IntradayVolLiqDivergence20d(Factor):
    """波动与流动性背离因子 (波动高位但Amihud低位的占比). 高波动无冲击=有承接, 正向.
    方向: 正向.
    """
    name = "intraday_vol_liq_divergence_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "波动无冲击占比 (健康=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "amount"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, amount = panel["close"], panel["amount"]
        ret_abs = close.pct_change().abs()
        amihud = ret_abs / (amount + 1e-12)
        day = ret_abs.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp_r = ret_abs.loc[day == dt]
            grp_a = amihud.loc[day == dt]
            if len(grp_r) < 30:
                continue
            vals = {}
            for col in grp_r.columns:
                r = grp_r[col].dropna()
                a = grp_a[col].dropna()
                common = r.index.intersection(a.index)
                if len(common) < 30:
                    continue
                r_c, a_c = r.loc[common], a.loc[common]
                rmed, amed = r_c.median(), a_c.median()
                if pd.isna(rmed) or pd.isna(amed):
                    continue
                vals[col] = float(((r_c > rmed) & (a_c < amed)).mean())
            if vals:
                interactions[dt] = pd.Series(vals)
        if not interactions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(interactions).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# 354. intraday_vol_surge_liq_before — 波动突增前流动性
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_vol_surge_liq_before_20d", category="intraday_advanced")
class IntradayVolSurgeLiqBefore20d(Factor):
    """波动突增前流动性水平因子 (波动突增分钟前20分钟平均Amihud). 突增前流动性差=真实冲击, 负向.
    方向: 负向.
    """
    name = "intraday_vol_surge_liq_before_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "突增前流动性 (冲击真实性, 负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "amount"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, amount = panel["close"], panel["amount"]
        ret_abs = close.pct_change().abs()
        amihud = ret_abs / (amount + 1e-12)
        day = ret_abs.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp_r = ret_abs.loc[day == dt]
            grp_a = amihud.loc[day == dt]
            if len(grp_r) < 60:
                continue
            vals = {}
            for col in grp_r.columns:
                r = grp_r[col].dropna()
                a = grp_a[col].dropna()
                common = r.index.intersection(a.index)
                if len(common) < 60:
                    continue
                r_c, a_c = r.loc[common], a.loc[common]
                mu, sigma = r_c.mean(), r_c.std(ddof=0)
                if sigma == 0 or pd.isna(sigma):
                    continue
                surge_idx = np.where(r_c.values > mu + 2 * sigma)[0]
                before = []
                for idx in surge_idx:
                    lo = max(0, idx - 20)
                    win = a_c.iloc[lo:idx].dropna()
                    if len(win) > 5:
                        before.append(win.mean())
                vals[col] = float(np.mean(before)) if before else np.nan
            if vals:
                interactions[dt] = pd.Series(vals)
        if not interactions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(interactions).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# 355. intraday_liq_resilience — 流动性恢复速度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_liq_resilience_20d", category="intraday_advanced")
class IntradayLiqResilience20d(Factor):
    """流动性恢复速度因子 (冲击后Amihud回落半周期, 取负). 恢复快=市场韧性强, 正向.
    方向: 正向.
    """
    name = "intraday_liq_resilience_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "冲击后流动性恢复速度 (韧性=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "amount"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, amount = panel["close"], panel["amount"]
        ret_abs = close.pct_change().abs()
        amihud = ret_abs / (amount + 1e-12)
        day = ret_abs.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp_r = ret_abs.loc[day == dt]
            grp_a = amihud.loc[day == dt]
            if len(grp_r) < 60:
                continue
            vals = {}
            for col in grp_r.columns:
                r = grp_r[col].dropna()
                a = grp_a[col].dropna()
                common = r.index.intersection(a.index)
                if len(common) < 60:
                    continue
                r_c, a_c = r.loc[common], a.loc[common]
                mu, sigma = r_c.mean(), r_c.std(ddof=0)
                if sigma == 0 or pd.isna(sigma):
                    continue
                amed = a_c.median()
                surge_idx = np.where(r_c.values > mu + 2 * sigma)[0]
                half_lives = []
                for idx in surge_idx:
                    if idx + 1 >= len(a_c):
                        continue
                    tail = a_c.iloc[idx + 1:]
                    below = np.where(tail.values < amed)[0]
                    if len(below) > 0:
                        half_lives.append(below[0] + 1)
                vals[col] = -float(np.mean(half_lives)) if half_lives else np.nan
            if vals:
                interactions[dt] = pd.Series(vals)
        if not interactions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(interactions).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# 356. intraday_cross_momentum_slope — 截面动量斜率
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_cross_momentum_slope_20d", category="intraday_advanced")
class IntradayCrossMomentumSlope20d(Factor):
    """截面动量演化因子 (日内前半-后半截面排名的相关). 截面动量稳定=趋势延续, 正向.
    方向: 正向.
    """
    name = "intraday_cross_momentum_slope_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "截面动量稳定性 (延续=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 60:
                continue
            half = len(grp) // 2
            ret_h1 = grp.iloc[half].div(grp.iloc[0].replace(0, np.nan)).dropna()
            ret_h2 = grp.iloc[-1].div(grp.iloc[half].replace(0, np.nan)).dropna()
            common = ret_h1.index.intersection(ret_h2.index)
            if len(common) < 10:
                continue
            corr = ret_h1[common].corr(ret_h2[common])
            if not pd.isna(corr):
                interactions[dt] = float(corr)
        if not interactions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        # 标量 dict -> 广播到全品种 (截面因子按品种无差异, 仅当日水平)
        daily = pd.DataFrame({k: {s: v for s in universe} for k, v in interactions.items()}).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# 357. intraday_cross_divergence — 截面发散度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_cross_divergence_20d", category="intraday_advanced")
class IntradayCrossDivergence20d(Factor):
    """截面收益发散度因子 (日内截面收益的标准差). 发散加剧=信息不对称增加, 负向.
    方向: 负向.
    """
    name = "intraday_cross_divergence_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "截面发散度 (分歧=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 60:
                continue
            ret_cs = grp.iloc[-1].div(grp.iloc[0].replace(0, np.nan)).dropna()
            sd = ret_cs.std(ddof=0)
            if not pd.isna(sd):
                interactions[dt] = float(sd)
        if not interactions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        # 标量 dict -> 广播到全品种 (截面因子按品种无差异, 仅当日水平)
        daily = pd.DataFrame({k: {s: v for s in universe} for k, v in interactions.items()}).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# 358. intraday_cross_concentration — 截面收益集中度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_cross_concentration_20d", category="intraday_advanced")
class IntradayCrossConcentration20d(Factor):
    """截面收益集中度因子 (前3强品种收益占全部正收益比例). 集中=龙头驱动, 正向.
    方向: 正向.
    """
    name = "intraday_cross_concentration_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "截面收益集中度 (龙头驱动=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 60:
                continue
            ret_cs = grp.iloc[-1].div(grp.iloc[0].replace(0, np.nan)).dropna()
            pos = ret_cs[ret_cs > 0]
            if len(pos) < 5 or pos.sum() <= 0:
                continue
            top3 = pos.nlargest(3).sum()
            interactions[dt] = pd.Series({"x": float(top3 / pos.sum())})
        if not interactions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        # 标量 dict -> 广播到全品种 (截面因子按品种无差异, 仅当日水平)
        daily = pd.DataFrame({k: {s: v for s in universe} for k, v in interactions.items()}).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# 359. intraday_cross_rank_stability — 截面排名稳定性
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_cross_rank_stability_20d", category="intraday_advanced")
class IntradayCrossRankStability20d(Factor):
    """截面排名稳定性因子 (日内排名与前日排名的一致性). 排名稳定=趋势持续, 正向.
    方向: 正向.
    """
    name = "intraday_cross_rank_stability_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "截面排名稳定性 (趋势=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        interactions: dict = {}
        prev_rank = None
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 60:
                continue
            ret_cs = grp.iloc[-1].div(grp.iloc[0].replace(0, np.nan)).dropna()
            rank = ret_cs.rank(pct=True)
            if prev_rank is not None:
                common = rank.index.intersection(prev_rank.index)
                if len(common) >= 10:
                    corr = rank[common].corr(prev_rank[common])
                    if not pd.isna(corr):
                        interactions[dt] = float(corr)
            prev_rank = rank
        if not interactions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        # 标量 dict -> 广播到全品种 (截面因子按品种无差异, 仅当日水平)
        daily = pd.DataFrame({k: {s: v for s in universe} for k, v in interactions.items()}).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# 360. intraday_cross_leader_follow — 截面龙头跟随
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_cross_leader_follow_20d", category="intraday_advanced")
class IntradayCrossLeaderFollow20d(Factor):
    """截面龙头跟随因子 (前日龙头今日仍居前的比例). 龙头延续=资金抱团, 正向.
    方向: 正向.
    """
    name = "intraday_cross_leader_follow_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "截面龙头延续 (抱团=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        interactions: dict = {}
        prev_leader = None
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 60:
                continue
            ret_cs = grp.iloc[-1].div(grp.iloc[0].replace(0, np.nan)).dropna()
            leader = set(ret_cs.nlargest(3).index)
            if prev_leader is not None:
                overlap = len(leader & prev_leader) / max(len(prev_leader), 1)
                interactions[dt] = float(overlap)
            prev_leader = leader
        if not interactions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        # 标量 dict -> 广播到全品种 (截面因子按品种无差异, 仅当日水平)
        daily = pd.DataFrame({k: {s: v for s in universe} for k, v in interactions.items()}).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# 361. intraday_buy_aggression — 主动买入强度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_buy_aggression_20d", category="intraday_advanced")
class IntradayBuyAggression20d(Factor):
    """主动买入强度因子 (收盘>开盘分钟计入主动买入, 其成交量占比). 买入攻击性=看涨, 正向.
    方向: 正向.
    """
    name = "intraday_buy_aggression_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "主动买入量占比 (攻击性=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "open", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, open_, volume = panel["close"], panel["open"], panel["volume"]
        day = close.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp_c = close.loc[day == dt]
            grp_o = open_.loc[day == dt]
            grp_v = volume.loc[day == dt]
            vals = {}
            for col in grp_c.columns:
                if col not in grp_o.columns or col not in grp_v.columns:
                    continue
                c, o, v = grp_c[col].dropna(), grp_o[col].dropna(), grp_v[col].dropna()
                common = c.index.intersection(o.index).intersection(v.index)
                if len(common) < 30:
                    continue
                c_c, o_c, v_c = c.loc[common], o.loc[common], v.loc[common]
                buy_vol = v_c[c_c > o_c].sum()
                tot = v_c.sum()
                if tot <= 0 or pd.isna(tot):
                    continue
                vals[col] = float(buy_vol / tot)
            if vals:
                interactions[dt] = pd.Series(vals)
        if not interactions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(interactions).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# 362. intraday_buy_sustain — 主动买入持续性
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_buy_sustain_20d", category="intraday_advanced")
class IntradayBuySustain20d(Factor):
    """主动买入持续性因子 (连续主动买入分钟的最长run). 持续买入=资金坚定, 正向.
    方向: 正向.
    """
    name = "intraday_buy_sustain_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "主动买入最长run (坚定=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "open", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        import itertools
        close, open_, volume = panel["close"], panel["open"], panel["volume"]
        day = close.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp_c = close.loc[day == dt]
            grp_o = open_.loc[day == dt]
            vals = {}
            for col in grp_c.columns:
                if col not in grp_o.columns:
                    continue
                c, o = grp_c[col].dropna(), grp_o[col].dropna()
                common = c.index.intersection(o.index)
                if len(common) < 30:
                    continue
                buy = (c.loc[common] > o.loc[common]).astype(int).values
                runs = [len(list(g)) for k, g in itertools.groupby(buy) if k == 1]
                vals[col] = float(max(runs)) if runs else 0.0
            if vals:
                interactions[dt] = pd.Series(vals)
        if not interactions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(interactions).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# 363. intraday_buy_sell_imbalance_slope — 买卖失衡斜率
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_buy_sell_imbalance_slope_20d", category="intraday_advanced")
class IntradayBuySellImbalanceSlope20d(Factor):
    """主动买卖失衡日内演化因子 (后半段失衡 - 前半段失衡). 失衡转买=尾盘吸筹, 正向.
    方向: 正向.
    """
    name = "intraday_buy_sell_imbalance_slope_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "买卖失衡日内斜率 (尾盘吸筹=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "open", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, open_, volume = panel["close"], panel["open"], panel["volume"]
        day = close.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp_c = close.loc[day == dt]
            grp_o = open_.loc[day == dt]
            grp_v = volume.loc[day == dt]
            vals = {}
            for col in grp_c.columns:
                if col not in grp_o.columns or col not in grp_v.columns:
                    continue
                c, o, v = grp_c[col].dropna(), grp_o[col].dropna(), grp_v[col].dropna()
                common = c.index.intersection(o.index).intersection(v.index)
                if len(common) < 60:
                    continue
                c_c, o_c, v_c = c.loc[common], o.loc[common], v.loc[common]
                buy = (c_c > o_c).astype(float)
                half = len(buy) // 2
                imb_h1 = (buy.iloc[:half] * v_c.iloc[:half]).sum() / v_c.iloc[:half].sum() if v_c.iloc[:half].sum() > 0 else np.nan
                imb_h2 = (buy.iloc[half:] * v_c.iloc[half:]).sum() / v_c.iloc[half:].sum() if v_c.iloc[half:].sum() > 0 else np.nan
                if pd.isna(imb_h1) or pd.isna(imb_h2):
                    continue
                vals[col] = float(imb_h2 - imb_h1)
            if vals:
                interactions[dt] = pd.Series(vals)
        if not interactions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(interactions).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# 364. intraday_big_buy_share — 大单买入占比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_big_buy_share_20d", category="intraday_advanced")
class IntradayBigBuyShare20d(Factor):
    """大单买入占比因子 (主动买入且量>μ+2σ的分钟量占比). 大单买入=机构参与, 正向.
    方向: 正向.
    """
    name = "intraday_big_buy_share_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "大单主动买入占比 (机构=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "open", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, open_, volume = panel["close"], panel["open"], panel["volume"]
        day = close.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp_c = close.loc[day == dt]
            grp_o = open_.loc[day == dt]
            grp_v = volume.loc[day == dt]
            vals = {}
            for col in grp_c.columns:
                if col not in grp_o.columns or col not in grp_v.columns:
                    continue
                c, o, v = grp_c[col].dropna(), grp_o[col].dropna(), grp_v[col].dropna()
                common = c.index.intersection(o.index).intersection(v.index)
                if len(common) < 30:
                    continue
                c_c, o_c, v_c = c.loc[common], o.loc[common], v.loc[common]
                mu, sigma = v_c.mean(), v_c.std(ddof=0)
                if sigma == 0 or pd.isna(sigma):
                    continue
                big_buy = v_c[(c_c > o_c) & (v_c > mu + 2 * sigma)].sum()
                tot = v_c.sum()
                if tot <= 0 or pd.isna(tot):
                    continue
                vals[col] = float(big_buy / tot)
            if vals:
                interactions[dt] = pd.Series(vals)
        if not interactions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(interactions).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# 365. intraday_order_flow_asymmetry — 订单流不对称
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_order_flow_asymmetry_20d", category="intraday_advanced")
class IntradayOrderFlowAsymmetry20d(Factor):
    """订单流不对称因子 (主动买量-主动卖量)/总成交量. 净主动买入=看涨共识, 正向.
    方向: 正向.
    """
    name = "intraday_order_flow_asymmetry_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "订单流净失衡 (共识=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "open", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, open_, volume = panel["close"], panel["open"], panel["volume"]
        day = close.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp_c = close.loc[day == dt]
            grp_o = open_.loc[day == dt]
            grp_v = volume.loc[day == dt]
            vals = {}
            for col in grp_c.columns:
                if col not in grp_o.columns or col not in grp_v.columns:
                    continue
                c, o, v = grp_c[col].dropna(), grp_o[col].dropna(), grp_v[col].dropna()
                common = c.index.intersection(o.index).intersection(v.index)
                if len(common) < 30:
                    continue
                c_c, o_c, v_c = c.loc[common], o.loc[common], v.loc[common]
                buy = (c_c > o_c).astype(float)
                buy_vol = (buy * v_c).sum()
                sell_vol = v_c.sum() - buy_vol
                tot = v_c.sum()
                if tot <= 0 or pd.isna(tot):
                    continue
                vals[col] = float((buy_vol - sell_vol) / tot)
            if vals:
                interactions[dt] = pd.Series(vals)
        if not interactions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(interactions).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# 366. intraday_open_close_drift — 开盘-收盘漂移
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_open_close_drift_20d", category="intraday_advanced")
class IntradayOpenCloseDrift20d(Factor):
    """开盘收盘漂移因子 (日内收益 vs 开盘跳空方向一致性). 跳空被日内确认=共识, 正向.
    方向: 正向.
    """
    name = "intraday_open_close_drift_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "跳空日内确认 (共识=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "open"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, open_ = panel["close"], panel["open"]
        day = close.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp_c = close.loc[day == dt]
            grp_o = open_.loc[day == dt]
            vals = {}
            for col in grp_c.columns:
                if col not in grp_o.columns:
                    continue
                c, o = grp_c[col].dropna(), grp_o[col].dropna()
                common = c.index.intersection(o.index)
                if len(common) < 30:
                    continue
                c_c, o_c = c.loc[common], o.loc[common]
                intraday = c_c.iloc[-1] - o_c.iloc[0]
                gap = o_c.iloc[0] - c_c.iloc[0]
                vals[col] = float(np.sign(intraday) == np.sign(gap))
            if vals:
                interactions[dt] = pd.Series(vals)
        if not interactions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(interactions).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# 367. intraday_session_shape — 交易时段形态
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_session_shape_20d", category="intraday_advanced")
class IntradaySessionShape20d(Factor):
    """交易时段形态因子 (早/中/晚三段收益同向度). 三段同向=趋势全天延续, 正向.
    方向: 正向.
    """
    name = "intraday_session_shape_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "三段收益同向度 (趋势=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 90:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < 90:
                    continue
                third = len(c) // 3
                seg1 = c.iloc[third] / c.iloc[0] - 1
                seg2 = c.iloc[2 * third] / c.iloc[third] - 1
                seg3 = c.iloc[-1] / c.iloc[2 * third] - 1
                signs = np.sign([seg1, seg2, seg3])
                vals[col] = float((signs == signs[0]).mean())
            if vals:
                interactions[dt] = pd.Series(vals)
        if not interactions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(interactions).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# 368. intraday_vol_volume_sync — 量价同步度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_vol_volume_sync_20d", category="intraday_advanced")
class IntradayVolVolumeSync20d(Factor):
    """量价同步度因子 (放量分钟与涨跌方向的一致性). 放量同向=资金驱动, 正向.
    方向: 正向.
    """
    name = "intraday_vol_volume_sync_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "放量方向一致性 (资金驱动=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, volume = panel["close"], panel["volume"]
        ret = close.pct_change()
        day = ret.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp_r = ret.loc[day == dt]
            grp_v = volume.loc[day == dt]
            vals = {}
            for col in grp_r.columns:
                if col not in grp_v.columns:
                    continue
                r, v = grp_r[col].dropna(), grp_v[col].dropna()
                common = r.index.intersection(v.index)
                if len(common) < 30:
                    continue
                r_c, v_c = r.loc[common], v.loc[common]
                vmed = v_c.median()
                if vmed <= 0 or pd.isna(vmed):
                    continue
                high_vol = v_c > vmed
                if high_vol.sum() == 0:
                    continue
                rsign = np.sign(r_c)
                vals[col] = float(np.mean(rsign[high_vol] == np.sign(r_c.median())) if r_c.median() != 0 else np.mean(rsign[high_vol] == 1))
            if vals:
                interactions[dt] = pd.Series(vals)
        if not interactions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(interactions).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# 369. intraday_momentum_vol_adjusted — 波动调整动量
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_momentum_vol_adjusted_20d", category="intraday_advanced")
class IntradayMomentumVolAdjusted20d(Factor):
    """波动调整动量因子 (日内动量 / 日内波动率). 单位风险动量高=效率高, 正向.
    方向: 正向.
    """
    name = "intraday_momentum_vol_adjusted_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "单位风险动量 (效率=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 60:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < 60:
                    continue
                ret_day = c.iloc[-1] / c.iloc[0] - 1
                vol = c.pct_change().std(ddof=0)
                if vol == 0 or pd.isna(vol):
                    continue
                vals[col] = float(ret_day / vol)
            if vals:
                interactions[dt] = pd.Series(vals)
        if not interactions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(interactions).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# 370. intraday_path_smoothness — 路径平滑度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_path_smoothness_20d", category="intraday_advanced")
class IntradayPathSmoothness20d(Factor):
    """路径平滑度因子 (日内收益二阶差分平方和的倒数). 平滑=有序推进, 正向.
    方向: 正向.
    """
    name = "intraday_path_smoothness_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "路径平滑度 (有序=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 60:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < 60:
                    continue
                d2 = c.pct_change().diff()
                rough = d2.pow(2).sum()
                if rough == 0 or pd.isna(rough):
                    continue
                vals[col] = float(1.0 / rough)
            if vals:
                interactions[dt] = pd.Series(vals)
        if not interactions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(interactions).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 371. intraday_sector_strength — 板块内相对强度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_sector_strength_20d", category="intraday_advanced")
class IntradaySectorStrength20d(Factor):
    """板块内相对强度因子 (跨品种: 板块内截面).

    品种日内收益在所属板块内的截面 z-score (比 #341 全市场更精细: 板块内相对强弱).
    板块内领涨 → 品种相对板块最强 → 正向.
    方向: 正向.
    """
    name = "intraday_sector_strength_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "板块内相对强度 (日内收益板块内z)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"open", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        open_px, close = panel["open"], panel["close"]
        day = close.index.normalize()
        daily_ret: dict = {}
        for dt in sorted(set(day)):
            grp_o = open_px.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 5:
                continue
            vals = {}
            for col in grp_c.columns:
                o = grp_o[col].dropna()
                c = grp_c[col].dropna()
                if len(o) < 1 or len(c) < 5:
                    continue
                o_first = o.iloc[0]
                if o_first < 1e-12:
                    continue
                vals[col] = float(c.iloc[-1] / o_first - 1.0)
            if vals:
                daily_ret[dt] = pd.Series(vals)
        if not daily_ret:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily_ret).T
        df.index = pd.DatetimeIndex(df.index)
        z = _sector_zscore(df.reindex(index=dates))
        return z.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 372. intraday_sector_momentum — 板块内相对动量
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_sector_momentum_20d", category="intraday_advanced")
class IntradaySectorMomentum20d(Factor):
    """板块内相对动量因子 (跨品种: 板块内截面).

    品种5日收益在所属板块内的截面 z-score (板块内相对动量).
    板块内动量领先 → 品种趋势领跑板块 → 正向.
    方向: 正向.
    """
    name = "intraday_sector_momentum_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "板块内相对动量 (5日收益板块内z)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        if close is None or close.empty:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        mom5 = close.pct_change(5)
        z = _sector_zscore(mom5.reindex(index=pd.DatetimeIndex(dates)))
        return z.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 373. intraday_sector_sync — 板块同步性
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_sector_sync_20d", category="intraday_advanced")
class IntradaySectorSync20d(Factor):
    """板块同步性因子 (跨品种: 板块联动).

    品种分钟收益与板块内其他品种平均收益的相关 (该品种与板块的同步程度).
    高同步 → 板块联动强 → 系统性驱动 → 正向.
    方向: 正向.
    """
    name = "intraday_sector_sync_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "板块同步性 (品种收益与板块其他均值相关)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        syncs: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 30:
                continue
            sub = grp.dropna(how="all")
            if sub.shape[1] < 2:
                continue
            other_mean = _sector_mean(sub)
            vals = {}
            for col in sub.columns:
                r_i = sub[col]
                r_o = other_mean[col]
                common = r_i.notna() & r_o.notna()
                if common.sum() < 20:
                    continue
                a = r_i[common].values
                b = r_o[common].values
                if a.std(ddof=0) < 1e-12 or b.std(ddof=0) < 1e-12:
                    vals[col] = 0.0
                else:
                    corr_val = float(np.corrcoef(a, b)[0, 1])
                    vals[col] = corr_val if not np.isnan(corr_val) else 0.0
            if vals:
                syncs[dt] = pd.Series(vals)
        if not syncs:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(syncs).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 374. intraday_sector_dispersion — 板块内分化度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_sector_dispersion_20d", category="intraday_advanced")
class IntradaySectorDispersion20d(Factor):
    """板块内分化度因子 (跨品种: 板块结构).

    品种日内收益偏离板块均值的绝对 z-score (板块内分化程度).
    分化大 → 板块无共识 → 负向.
    方向: 负向.
    """
    name = "intraday_sector_dispersion_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "板块内分化 (偏离板块均值绝对z, 无共识=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"open", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        open_px, close = panel["open"], panel["close"]
        day = close.index.normalize()
        daily_ret: dict = {}
        for dt in sorted(set(day)):
            grp_o = open_px.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 5:
                continue
            vals = {}
            for col in grp_c.columns:
                o = grp_o[col].dropna()
                c = grp_c[col].dropna()
                if len(o) < 1 or len(c) < 5:
                    continue
                o_first = o.iloc[0]
                if o_first < 1e-12:
                    continue
                vals[col] = float(c.iloc[-1] / o_first - 1.0)
            if vals:
                daily_ret[dt] = pd.Series(vals)
        if not daily_ret:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily_ret).T
        df.index = pd.DatetimeIndex(df.index)
        z = _sector_zscore(df.reindex(index=dates))
        return (-z.abs()).reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 375. intraday_sector_vol_spill — 板块波动溢出
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_sector_vol_spill_20d", category="intraday_advanced")
class IntradaySectorVolSpill20d(Factor):
    """板块波动溢出因子 (跨品种: 波动传导).

    板块内其他品种的平均日内波动 (波动从板块其他品种溢出到本品种的风险).
    板块其他品种高波动 → 传染风险 → 负向.
    方向: 负向.
    """
    name = "intraday_sector_vol_spill_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "板块波动溢出 (板块其他品种波动, 传染=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        daily_vol: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 20:
                    continue
                vals[col] = float(r.std(ddof=0))
            if vals:
                daily_vol[dt] = pd.Series(vals)
        if not daily_vol:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily_vol).T
        df.index = pd.DatetimeIndex(df.index)
        spill = _sector_mean(df.reindex(index=dates))
        return (-spill).reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 376. intraday_cross_spread — 跨品种价格位置
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_cross_spread_20d", category="intraday_advanced")
class IntradayCrossSpread20d(Factor):
    """跨品种价格位置因子.

    品种 log 价格相对全部品种均值的偏离 (对数价格截面位置, 跨品种价差).
    相对高位 → 品种截面内最贵 → 均值回归压力 → 负向.
    方向: 负向.
    """
    name = "intraday_cross_spread_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "跨品种价格位置 (log价格截面偏离, 高位=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        if close is None or close.empty:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        log_close = np.log(close.replace(0, np.nan))
        spread = log_close.sub(log_close.mean(axis=1), axis=0).div(log_close.std(axis=1, ddof=0).replace(0, np.nan), axis=0)
        return (-spread).reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 377. intraday_sector_rotation — 板块内轮动
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_sector_rotation_20d", category="intraday_advanced")
class IntradaySectorRotation20d(Factor):
    """板块内轮动因子 (跨品种: 板块结构动态).

    品种在板块内的相对强度排名变化 (板块内资金轮动速度).
    排名快速变化 → 板块内轮动剧烈 → 不稳定 → 负向.
    方向: 负向.
    """
    name = "intraday_sector_rotation_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "板块内轮动 (板块内强度排名变化, 剧烈=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"open", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        open_px, close = panel["open"], panel["close"]
        day = close.index.normalize()
        daily_ret: dict = {}
        for dt in sorted(set(day)):
            grp_o = open_px.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 5:
                continue
            vals = {}
            for col in grp_c.columns:
                o = grp_o[col].dropna()
                c = grp_c[col].dropna()
                if len(o) < 1 or len(c) < 5:
                    continue
                o_first = o.iloc[0]
                if o_first < 1e-12:
                    continue
                vals[col] = float(c.iloc[-1] / o_first - 1.0)
            if vals:
                daily_ret[dt] = pd.Series(vals)
        if not daily_ret:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily_ret).T
        df.index = pd.DatetimeIndex(df.index)
        # 板块内排名变化: 先算板块内每日百分位排名, 再跨日差
        rank = df.apply(lambda row: _sector_rank(row), axis=1)
        rank_chg = rank - rank.shift(1)
        return (-rank_chg).reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 378. intraday_cross_lead_lag — 跨品种领先滞后
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_cross_lead_lag_20d", category="intraday_advanced")
class IntradayCrossLeadLag20d(Factor):
    """跨品种领先滞后因子 (跨品种: 信息传导).

    品种分钟收益与板块其他品种平均收益的领先相关差:
    corr(ret_i, other_{t+1}) - corr(other_t, ret_{i,t+1}) (该品种领先还是跟随板块).
    品种领先板块 → 定价者 → 正向.
    方向: 正向.
    """
    name = "intraday_cross_lead_lag_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "跨品种领先滞后 (品种领先板块=定价者=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_1m = panel["close"].pct_change()
        day = ret_1m.index.normalize()
        leads: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 40:
                continue
            sub = grp.dropna(how="all")
            if sub.shape[1] < 2:
                continue
            other_mean = _sector_mean(sub)
            vals = {}
            for col in sub.columns:
                r_i = sub[col].values
                r_o = other_mean[col].values
                mask = np.isfinite(r_i) & np.isfinite(r_o)
                if mask.sum() < 30:
                    continue
                ri, ro = r_i[mask], r_o[mask]
                # 品种领先: corr(ret_i_t, other_{t+1})
                c1 = float(np.corrcoef(ri[:-1], ro[1:])[0, 1]) if ri[:-1].std() > 1e-12 and ro[1:].std() > 1e-12 else 0.0
                # 板块领先: corr(other_t, ret_i_{t+1})
                c2 = float(np.corrcoef(ro[:-1], ri[1:])[0, 1]) if ro[:-1].std() > 1e-12 and ri[1:].std() > 1e-12 else 0.0
                vals[col] = c1 - c2 if not np.isnan(c1) and not np.isnan(c2) else 0.0
            if vals:
                leads[dt] = pd.Series(vals)
        if not leads:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(leads).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 379. intraday_cross_rank_momentum — 全市场排名动量
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_cross_rank_momentum_20d", category="intraday_advanced")
class IntradayCrossRankMomentum20d(Factor):
    """全市场排名动量因子 (跨品种: 排名动态).

    品种在全市场截面强度排名的变化 (排名动量, 5日排名变化).
    排名持续上升 → 品种相对走强 → 正向.
    方向: 正向.
    """
    name = "intraday_cross_rank_momentum_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "全市场排名动量 (截面强度排名5日变化)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        close = data.get("close", dates, universe)
        if close is None or close.empty:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret5 = close.pct_change(5)
        rank = ret5.rank(axis=1, pct=True)
        chg = rank - rank.shift(5)
        return chg.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 380. intraday_cross_reversion — 截面均值回归
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_cross_reversion_20d", category="intraday_advanced")
class IntradayCrossReversion20d(Factor):
    """截面均值回归因子 (跨品种: 截面反转).

    品种日内收益的全市场截面 z-score 的5日滚动均值 (截面强弱的时间均值, 反向).
    持续截面强势 → 均值回归压力 → 负向.
    方向: 负向.
    """
    name = "intraday_cross_reversion_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "截面均值回归 (截面z的5日均, 强势回归=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"open", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        open_px, close = panel["open"], panel["close"]
        day = close.index.normalize()
        daily_ret: dict = {}
        for dt in sorted(set(day)):
            grp_o = open_px.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 5:
                continue
            vals = {}
            for col in grp_c.columns:
                o = grp_o[col].dropna()
                c = grp_c[col].dropna()
                if len(o) < 1 or len(c) < 5:
                    continue
                o_first = o.iloc[0]
                if o_first < 1e-12:
                    continue
                vals[col] = float(c.iloc[-1] / o_first - 1.0)
            if vals:
                daily_ret[dt] = pd.Series(vals)
        if not daily_ret:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily_ret).T
        df.index = pd.DatetimeIndex(df.index)
        z = _cs_zscore(df.reindex(index=dates))
        z5 = z.rolling(5, min_periods=3).mean()
        return (-z5).reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 381. intraday_amihud_cross_z — Amihud 全市场截面 z-score
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_amihud_cross_z_20d", category="intraday_advanced")
class IntradayAmihudCrossZ20d(Factor):
    """Amihud 全市场截面 z-score 因子.

    品种日内 Amihud 在全部品种截面中的 z-score (跨品种比流动性).
    截面 z 高 → 相对全市场流动性差 → 负向.
    方向: 负向.
    """
    name = "intraday_amihud_cross_z_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "Amihud全市场截面z (相对流动性差=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "amount"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, amount = panel["close"], panel["amount"]
        ret_1m = close.pct_change().abs()
        amihud = ret_1m / (amount + 1e-12)
        amihud = amihud.replace([np.inf, -np.inf], np.nan)
        day = amihud.index.normalize()
        daily_amihud: dict = {}
        for dt in sorted(set(day)):
            grp = amihud.loc[day == dt]
            if len(grp) < 10:
                continue
            vals = {}
            for col in grp.columns:
                a = grp[col].dropna()
                a = a[a < 10.0]
                if len(a) < 10:
                    continue
                vals[col] = float(a.mean())
            if vals:
                daily_amihud[dt] = pd.Series(vals)
        if not daily_amihud:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily_amihud).T
        df.index = pd.DatetimeIndex(df.index)
        z = _cs_zscore(df.reindex(index=dates))
        return (-z).reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 382. intraday_amihud_sector_z — Amihud 板块内截面 z-score
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_amihud_sector_z_20d", category="intraday_advanced")
class IntradayAmihudSectorZ20d(Factor):
    """Amihud 板块内截面 z-score 因子.

    品种日内 Amihud 在所属板块内的截面 z-score (板块内相对流动性, 比 #381 更精细).
    板块内 z 高 → 板块中流动性最差 → 负向.
    方向: 负向.
    """
    name = "intraday_amihud_sector_z_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "Amihud板块内截面z (板块内流动性差=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "amount"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, amount = panel["close"], panel["amount"]
        ret_1m = close.pct_change().abs()
        amihud = ret_1m / (amount + 1e-12)
        amihud = amihud.replace([np.inf, -np.inf], np.nan)
        day = amihud.index.normalize()
        daily_amihud: dict = {}
        for dt in sorted(set(day)):
            grp = amihud.loc[day == dt]
            if len(grp) < 10:
                continue
            vals = {}
            for col in grp.columns:
                a = grp[col].dropna()
                a = a[a < 10.0]
                if len(a) < 10:
                    continue
                vals[col] = float(a.mean())
            if vals:
                daily_amihud[dt] = pd.Series(vals)
        if not daily_amihud:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily_amihud).T
        df.index = pd.DatetimeIndex(df.index)
        z = _sector_zscore(df.reindex(index=dates))
        return (-z).reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 383. intraday_amihud_cross_rank — Amihud 全市场截面分位
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_amihud_cross_rank_20d", category="intraday_advanced")
class IntradayAmihudCrossRank20d(Factor):
    """Amihud 全市场截面分位因子.

    品种日内 Amihud 在全部品种截面中的 rank (百分位, 截面排序).
    rank 高 → 相对全市场流动性差 → 负向.
    方向: 负向.
    """
    name = "intraday_amihud_cross_rank_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "Amihud全市场截面分位 (流动性排名靠后=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "amount"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, amount = panel["close"], panel["amount"]
        ret_1m = close.pct_change().abs()
        amihud = ret_1m / (amount + 1e-12)
        amihud = amihud.replace([np.inf, -np.inf], np.nan)
        day = amihud.index.normalize()
        daily_amihud: dict = {}
        for dt in sorted(set(day)):
            grp = amihud.loc[day == dt]
            if len(grp) < 10:
                continue
            vals = {}
            for col in grp.columns:
                a = grp[col].dropna()
                a = a[a < 10.0]
                if len(a) < 10:
                    continue
                vals[col] = float(a.mean())
            if vals:
                daily_amihud[dt] = pd.Series(vals)
        if not daily_amihud:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily_amihud).T
        df.index = pd.DatetimeIndex(df.index)
        rank = df.reindex(index=dates).rank(axis=1, pct=True)
        return (-rank).reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 384. intraday_amihud_sector_relative — Amihud 相对板块均值
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_amihud_sector_relative_20d", category="intraday_advanced")
class IntradayAmihudSectorRelative20d(Factor):
    """Amihud 相对板块均值因子.

    品种日内 Amihud / 板块其他品种 Amihud 均值 (相对板块流动性比值).
    比值高 → 本品种相对板块流动性差 → 负向.
    方向: 负向.
    """
    name = "intraday_amihud_sector_relative_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "Amihud相对板块 (品种/板块其他均值, 差=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "amount"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, amount = panel["close"], panel["amount"]
        ret_1m = close.pct_change().abs()
        amihud = ret_1m / (amount + 1e-12)
        amihud = amihud.replace([np.inf, -np.inf], np.nan)
        day = amihud.index.normalize()
        daily_amihud: dict = {}
        for dt in sorted(set(day)):
            grp = amihud.loc[day == dt]
            if len(grp) < 10:
                continue
            vals = {}
            for col in grp.columns:
                a = grp[col].dropna()
                a = a[a < 10.0]
                if len(a) < 10:
                    continue
                vals[col] = float(a.mean())
            if vals:
                daily_amihud[dt] = pd.Series(vals)
        if not daily_amihud:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily_amihud).T
        df.index = pd.DatetimeIndex(df.index)
        sector_mean = _sector_mean(df.reindex(index=dates))
        ratio = df.reindex(index=dates) / sector_mean.replace(0, np.nan)
        return (-ratio).reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 385. intraday_amihud_vol_partial — Amihud-成交量偏相关 (控制波动)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_amihud_vol_partial_20d", category="intraday_advanced")
class IntradayAmihudVolPartial20d(Factor):
    """Amihud-成交量偏相关因子 (控制波动后).

    corr(Amihud_resid, volume_resid | 控制 |ret|) — 将 Amihud 与 volume 分别对 |ret| 回归取残差再相关.
    偏相关为正 → 放量时非流动性反升 → 异常 → 负向.
    偏相关为负 → 放量改善流动性 → 正常 → 正向.
    方向: 负向.
    """
    name = "intraday_amihud_vol_partial_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "Amihud量偏相关 (控制波动, 正偏=异常=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "amount", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, amount, volume = panel["close"], panel["amount"], panel["volume"]
        ret_1m = close.pct_change()
        ret_abs = ret_1m.abs()
        amihud = ret_abs / (amount + 1e-12)
        amihud = amihud.replace([np.inf, -np.inf], np.nan)
        day = amihud.index.normalize()
        partials: dict = {}
        for dt in sorted(set(day)):
            grp_a = amihud.loc[day == dt]
            grp_v = volume.loc[day == dt]
            grp_r = ret_abs.loc[day == dt]
            if len(grp_a) < 20:
                continue
            vals = {}
            for col in grp_a.columns:
                a = grp_a[col].dropna()
                v = grp_v[col].dropna()
                r = grp_r[col].dropna()
                common = a.index.intersection(v.index).intersection(r.index)
                if len(common) < 20:
                    continue
                a_c = a.loc[common].values
                v_c = v.loc[common].values
                r_c = r.loc[common].values
                a_c = a_c[a_c < 10.0]
                # 过滤后对齐 (先按同掩码过滤 v/r)
                mask = a.loc[common].values < 10.0
                if mask.sum() < 15:
                    continue
                a_c = a.loc[common].values[mask]
                v_c = v_c[mask]
                r_c = r_c[mask]
                if a_c.std(ddof=0) < 1e-12 or v_c.std(ddof=0) < 1e-12:
                    vals[col] = 0.0
                    continue
                x = np.column_stack([np.ones(len(r_c)), r_c])
                try:
                    b_a = np.linalg.lstsq(x, a_c, rcond=None)[0]
                    b_v = np.linalg.lstsq(x, v_c, rcond=None)[0]
                except np.linalg.LinAlgError:
                    vals[col] = 0.0
                    continue
                resid_a = a_c - x @ b_a
                resid_v = v_c - x @ b_v
                if resid_a.std(ddof=0) < 1e-12 or resid_v.std(ddof=0) < 1e-12:
                    vals[col] = 0.0
                else:
                    corr_val = float(np.corrcoef(resid_a, resid_v)[0, 1])
                    vals[col] = -corr_val if not np.isnan(corr_val) else 0.0
            if vals:
                partials[dt] = pd.Series(vals)
        if not partials:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(partials).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 386. intraday_amihud_resid_skew — Amihud 去波动残差偏度
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_amihud_resid_skew_20d", category="intraday_advanced")
class IntradayAmihudResidSkew20d(Factor):
    """Amihud 去波动残差偏度因子.

    Amihud 对 |ret| 回归残差的偏度 (去除波动后, 非流动性的偶发大冲击程度).
    正偏 → 偶发极端流动性枯竭 → 负向.
    方向: 负向.
    """
    name = "intraday_amihud_resid_skew_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "Amihud去波动残差偏度 (偶发枯竭=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "amount"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, amount = panel["close"], panel["amount"]
        ret_1m = close.pct_change()
        ret_abs = ret_1m.abs()
        amihud = ret_abs / (amount + 1e-12)
        amihud = amihud.replace([np.inf, -np.inf], np.nan)
        day = amihud.index.normalize()
        skews: dict = {}
        for dt in sorted(set(day)):
            grp_a = amihud.loc[day == dt]
            grp_r = ret_abs.loc[day == dt]
            if len(grp_a) < 20:
                continue
            vals = {}
            for col in grp_a.columns:
                a = grp_a[col].dropna()
                r = grp_r[col].dropna()
                common = a.index.intersection(r.index)
                if len(common) < 20:
                    continue
                a_c = a.loc[common].values
                r_c = r.loc[common].values
                mask = a_c < 10.0
                if mask.sum() < 15:
                    continue
                a_c = a_c[mask]
                r_c = r_c[mask]
                x = np.column_stack([np.ones(len(r_c)), r_c])
                try:
                    b_a = np.linalg.lstsq(x, a_c, rcond=None)[0]
                except np.linalg.LinAlgError:
                    vals[col] = 0.0
                    continue
                resid = a_c - x @ b_a
                n = len(resid)
                if n < 10 or resid.std(ddof=0) < 1e-12:
                    vals[col] = 0.0
                    continue
                # 标准偏度 (三阶矩/σ³)
                skew_val = float(np.mean((resid - resid.mean()) ** 3) / resid.std(ddof=0) ** 3)
                vals[col] = -skew_val  # 正偏=偶发枯竭=负向
            if vals:
                skews[dt] = pd.Series(vals)
        if not skews:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(skews).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 387. intraday_amihud_vol_conditional — 成交量条件 Amihud 比
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_amihud_vol_conditional_20d", category="intraday_advanced")
class IntradayAmihudVolConditional20d(Factor):
    """成交量条件 Amihud 比因子 (控制波动后).

    高成交量分钟(>中位)的 Amihud 均值 / 低成交量分钟(<中位)的 Amihud 均值.
    比值高 → 放量时冲击反而大 → 量能未改善流动性 → 负向.
    方向: 负向.
    """
    name = "intraday_amihud_vol_conditional_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "量条件Amihud比 (高量Amihud/低量, 放量仍冲击=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "amount", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, amount, volume = panel["close"], panel["amount"], panel["volume"]
        ret_1m = close.pct_change().abs()
        amihud = ret_1m / (amount + 1e-12)
        amihud = amihud.replace([np.inf, -np.inf], np.nan)
        day = amihud.index.normalize()
        ratios: dict = {}
        for dt in sorted(set(day)):
            grp_a = amihud.loc[day == dt]
            grp_v = volume.loc[day == dt]
            if len(grp_a) < 20:
                continue
            vals = {}
            for col in grp_a.columns:
                a = grp_a[col].dropna()
                v = grp_v[col].dropna()
                common = a.index.intersection(v.index)
                if len(common) < 20:
                    continue
                a_c = a.loc[common]
                v_c = v.loc[common]
                a_c = a_c[a_c < 10.0]
                v_c = v_c.loc[a_c.index]
                if len(a_c) < 15:
                    continue
                v_med = v_c.median()
                if v_med < 1e-12:
                    continue
                high_vol = a_c[v_c >= v_med]
                low_vol = a_c[v_c < v_med]
                h_mean = high_vol.mean()
                l_mean = low_vol.mean()
                if l_mean < 1e-12:
                    vals[col] = 1.0
                else:
                    vals[col] = float(h_mean / l_mean)
            if vals:
                ratios[dt] = pd.Series(vals)
        if not ratios:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(ratios).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 388. intraday_amihud_resid_vol — Amihud 去波动残差波动
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_amihud_resid_vol_20d", category="intraday_advanced")
class IntradayAmihudResidVol20d(Factor):
    """Amihud 去波动残差波动因子.

    Amihud 对 |ret| 回归残差的标准差 (去除波动后, 纯流动性冲击的不稳定性).
    残差波动高 → 流动性状态反复 → 不可预测 → 负向.
    方向: 负向.
    """
    name = "intraday_amihud_resid_vol_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "Amihud去波动残差波动 (流动性反复=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "amount"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, amount = panel["close"], panel["amount"]
        ret_1m = close.pct_change()
        ret_abs = ret_1m.abs()
        amihud = ret_abs / (amount + 1e-12)
        amihud = amihud.replace([np.inf, -np.inf], np.nan)
        day = amihud.index.normalize()
        vols: dict = {}
        for dt in sorted(set(day)):
            grp_a = amihud.loc[day == dt]
            grp_r = ret_abs.loc[day == dt]
            if len(grp_a) < 20:
                continue
            vals = {}
            for col in grp_a.columns:
                a = grp_a[col].dropna()
                r = grp_r[col].dropna()
                common = a.index.intersection(r.index)
                if len(common) < 20:
                    continue
                a_c = a.loc[common].values
                r_c = r.loc[common].values
                mask = a_c < 10.0
                if mask.sum() < 15:
                    continue
                a_c = a_c[mask]
                r_c = r_c[mask]
                x = np.column_stack([np.ones(len(r_c)), r_c])
                try:
                    b_a = np.linalg.lstsq(x, a_c, rcond=None)[0]
                except np.linalg.LinAlgError:
                    vals[col] = 0.0
                    continue
                resid = a_c - x @ b_a
                resid_std = float(resid.std(ddof=0))
                vals[col] = -resid_std  # 残差波动高=负向
            if vals:
                vols[dt] = pd.Series(vals)
        if not vols:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(vols).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 389. intraday_amihud_rank — Amihud 时间序列分位
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_amihud_rank_20d", category="intraday_advanced")
class IntradayAmihudRank20d(Factor):
    """Amihud 时间序列分位因子.

    品种日内 Amihud 在自身 20 日分布中的百分位 (时间序列排序).
    rank 高 → 自身流动性处历史低位 → 负向.
    方向: 负向.
    """
    name = "intraday_amihud_rank_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "Amihud时间序列分位 (自身流动性历史低位=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "amount"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, amount = panel["close"], panel["amount"]
        ret_1m = close.pct_change().abs()
        amihud = ret_1m / (amount + 1e-12)
        amihud = amihud.replace([np.inf, -np.inf], np.nan)
        day = amihud.index.normalize()
        daily_amihud: dict = {}
        for dt in sorted(set(day)):
            grp = amihud.loc[day == dt]
            if len(grp) < 10:
                continue
            vals = {}
            for col in grp.columns:
                a = grp[col].dropna()
                a = a[a < 10.0]
                if len(a) < 10:
                    continue
                vals[col] = float(a.mean())
            if vals:
                daily_amihud[dt] = pd.Series(vals)
        if not daily_amihud:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily_amihud).T
        df.index = pd.DatetimeIndex(df.index)
        rank = df.reindex(index=dates).rolling(20, min_periods=5).apply(
            lambda x: float((x.iloc[:-1] <= x.iloc[-1]).sum() / max(1, len(x) - 1)), raw=False)
        return (-rank).reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 390. intraday_amihud_rank_z — Amihud 时间序列 z-score
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_amihud_rank_z_20d", category="intraday_advanced")
class IntradayAmihudRankZ20d(Factor):
    """Amihud 时间序列 z-score 因子.

    品种日内 Amihud 的 20 日 z-score (自身历史偏离程度).
    z 高 → 流动性异常恶化 → 负向.
    方向: 负向.
    """
    name = "intraday_amihud_rank_z_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "Amihud时间序列z (流动性异常恶化=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "amount"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, amount = panel["close"], panel["amount"]
        ret_1m = close.pct_change().abs()
        amihud = ret_1m / (amount + 1e-12)
        amihud = amihud.replace([np.inf, -np.inf], np.nan)
        day = amihud.index.normalize()
        daily_amihud: dict = {}
        for dt in sorted(set(day)):
            grp = amihud.loc[day == dt]
            if len(grp) < 10:
                continue
            vals = {}
            for col in grp.columns:
                a = grp[col].dropna()
                a = a[a < 10.0]
                if len(a) < 10:
                    continue
                vals[col] = float(a.mean())
            if vals:
                daily_amihud[dt] = pd.Series(vals)
        if not daily_amihud:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily_amihud).T
        df.index = pd.DatetimeIndex(df.index)
        mean = df.reindex(index=dates).rolling(20, min_periods=5).mean()
        std = df.reindex(index=dates).rolling(20, min_periods=5).std().replace(0, np.nan)
        z = (df.reindex(index=dates) - mean) / std
        return (-z).reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 391. intraday_amihud_rank_change — Amihud 时间序列分位变化
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_amihud_rank_change_20d", category="intraday_advanced")
class IntradayAmihudRankChange20d(Factor):
    """Amihud 时间序列分位变化因子.

    品种 Amihud 的 20 日分位的跨日变化 (流动性排名恶化/改善).
    rank 上升 → 流动性恶化趋势 → 负向.
    方向: 负向.
    """
    name = "intraday_amihud_rank_change_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "Amihud分位变化 (rank升=流动性恶化=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "amount"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, amount = panel["close"], panel["amount"]
        ret_1m = close.pct_change().abs()
        amihud = ret_1m / (amount + 1e-12)
        amihud = amihud.replace([np.inf, -np.inf], np.nan)
        day = amihud.index.normalize()
        daily_amihud: dict = {}
        for dt in sorted(set(day)):
            grp = amihud.loc[day == dt]
            if len(grp) < 10:
                continue
            vals = {}
            for col in grp.columns:
                a = grp[col].dropna()
                a = a[a < 10.0]
                if len(a) < 10:
                    continue
                vals[col] = float(a.mean())
            if vals:
                daily_amihud[dt] = pd.Series(vals)
        if not daily_amihud:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily_amihud).T
        df.index = pd.DatetimeIndex(df.index)
        rank = df.reindex(index=dates).rolling(20, min_periods=5).apply(
            lambda x: float((x.iloc[:-1] <= x.iloc[-1]).sum() / max(1, len(x) - 1)), raw=False)
        rank_chg = rank.diff()
        return (-rank_chg).reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 392. intraday_amihud_rank_ma — Amihud 时间序列分位平滑
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_amihud_rank_ma_20d", category="intraday_advanced")
class IntradayAmihudRankMa20d(Factor):
    """Amihud 时间序列分位平滑因子.

    品种 Amihud 的 20 日分位的 5 日均 (持续性流动性状态, 平滑掉单日噪声).
    rank 持续高位 → 流动性持续低迷 → 负向.
    方向: 负向.
    """
    name = "intraday_amihud_rank_ma_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "Amihud分位5日均 (持续低迷=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {"close", "amount"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, amount = panel["close"], panel["amount"]
        ret_1m = close.pct_change().abs()
        amihud = ret_1m / (amount + 1e-12)
        amihud = amihud.replace([np.inf, -np.inf], np.nan)
        day = amihud.index.normalize()
        daily_amihud: dict = {}
        for dt in sorted(set(day)):
            grp = amihud.loc[day == dt]
            if len(grp) < 10:
                continue
            vals = {}
            for col in grp.columns:
                a = grp[col].dropna()
                a = a[a < 10.0]
                if len(a) < 10:
                    continue
                vals[col] = float(a.mean())
            if vals:
                daily_amihud[dt] = pd.Series(vals)
        if not daily_amihud:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily_amihud).T
        df.index = pd.DatetimeIndex(df.index)
        rank = df.reindex(index=dates).rolling(20, min_periods=5).apply(
            lambda x: float((x.iloc[:-1] <= x.iloc[-1]).sum() / max(1, len(x) - 1)), raw=False)
        rank_ma = rank.rolling(5, min_periods=3).mean()
        return (-rank_ma).reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 393. intraday_multiperiod_trend_vote — 多周期趋势投票 (A1 复现)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_multiperiod_trend_vote_20d", category="intraday_advanced")
class IntradayMultiperiodTrendVote20d(Factor):
    """多周期趋势投票因子 (资料A1, 基于5min).

    分别计算 5/15/30/60 根5min K线的均线斜率方向, 多头票数/总票数.
    votes_bullish 高 → 多周期共振看多 → 正向.
    与现有均线类(单周期)互补: 本因子是"多周期一致度".
    方向: 正向.
    """
    name = "intraday_multiperiod_trend_vote_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "多周期趋势投票 (5/15/30/60根5min斜率投票, 共振=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="5min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        votes: dict = {}
        windows = (5, 15, 30, 60)
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < max(windows) + 2:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < max(windows) + 2:
                    continue
                bullish = 0
                for w in windows:
                    ma_now = c.iloc[-1:].mean()
                    ma_prev = c.iloc[-w - 1:-1].mean()
                    if ma_now > ma_prev:
                        bullish += 1
                vals[col] = float(bullish / len(windows))
            if vals:
                votes[dt] = pd.Series(vals)
        if not votes:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(votes).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 394. intraday_multiperiod_vote_confirm — 多周期投票动量确认 (原创改进)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_multiperiod_vote_confirm_20d", category="intraday_advanced")
class IntradayMultiperiodVoteConfirm20d(Factor):
    """多周期投票动量确认因子 (原创: A1 改进).

    在 A1 投票基础上, 用投票结果与价格位移方向一致性加权:
    score = votes_bullish × sign(close_now - close_open) — 投票看多且实际收涨才计正分.
    防止"均线共振但价格未确认"的假信号, 提高信号质量.
    方向: 正向.
    """
    name = "intraday_multiperiod_vote_confirm_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "多周期投票动量确认 (投票×价格确认, 假信号过滤)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="5min")
        if not {"open", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        open_px, close = panel["open"], panel["close"]
        day = close.index.normalize()
        confirms: dict = {}
        windows = (5, 15, 30, 60)
        for dt in sorted(set(day)):
            grp_o = open_px.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < max(windows) + 2:
                continue
            vals = {}
            for col in grp_c.columns:
                o = grp_o[col].dropna()
                c = grp_c[col].dropna()
                if len(c) < max(windows) + 2 or len(o) < 1:
                    continue
                bullish = 0
                for w in windows:
                    ma_now = c.iloc[-1:].mean()
                    ma_prev = c.iloc[-w - 1:-1].mean()
                    if ma_now > ma_prev:
                        bullish += 1
                vote = bullish / len(windows)
                o_first = o.iloc[0]
                if o_first < 1e-12:
                    continue
                day_ret = np.sign(c.iloc[-1] / o_first - 1.0)
                vals[col] = float(vote * day_ret)
            if vals:
                confirms[dt] = pd.Series(vals)
        if not confirms:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(confirms).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 395. intraday_volume_price_corr_div — 量价相关快慢窗背离 (A2 复现)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_volume_price_corr_div_20d", category="intraday_advanced")
class IntradayVolumePriceCorrDiv20d(Factor):
    """量价相关快慢窗背离因子 (资料A2, 基于5min).

    Corr(ret, volume, 20根) − Corr(ret, volume, 60根) (快慢窗口相关之差).
    正背离 → 近期量价共振强于长期 → 趋势确认 → 正向.
    负背离 → 量价背离 → 反转提示 → 负向.
    方向: 随符号 (输出差值本身).
    """
    name = "intraday_volume_price_corr_div_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "量价相关快慢窗背离 (corr20-corr60, 共振=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="5min")
        if not {"close", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, volume = panel["close"], panel["volume"]
        ret_1m = close.pct_change()
        day = ret_1m.index.normalize()
        divs: dict = {}
        for dt in sorted(set(day)):
            grp_r = ret_1m.loc[day == dt]
            grp_v = volume.loc[day == dt]
            if len(grp_r) < 60:
                continue
            vals = {}
            for col in grp_r.columns:
                r = grp_r[col].dropna()
                v = grp_v[col].dropna()
                common = r.index.intersection(v.index)
                if len(common) < 60:
                    continue
                r_c = r.loc[common]
                v_c = v.loc[common]
                if r_c.std() < 1e-12 or v_c.std() < 1e-12:
                    vals[col] = 0.0
                    continue
                corr20 = r_c.iloc[-20:].corr(v_c.iloc[-20:], method="spearman")
                corr60 = r_c.iloc[-60:].corr(v_c.iloc[-60:], method="spearman")
                c20 = corr20 if not np.isnan(corr20) else 0.0
                c60 = corr60 if not np.isnan(corr60) else 0.0
                vals[col] = float(c20 - c60)
            if vals:
                divs[dt] = pd.Series(vals)
        if not divs:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(divs).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 396. intraday_vp_corr_div_slope — 量价相关背离趋势 (原创改进)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_vp_corr_div_slope_20d", category="intraday_advanced")
class IntradayVpCorrDivSlope20d(Factor):
    """量价相关背离趋势因子 (原创: A2 改进).

    快慢窗相关背离的边际变化 (今日背离 − 昨日背离):
    背离持续扩大 → 量价结构正在转变 (更可靠的结构信号, 而非单日背离).
    与 A2 互补: A2 看背离水平, 本因子看背离的演化方向.
    方向: 正向.
    """
    name = "intraday_vp_corr_div_slope_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "量价背离趋势 (背离的跨日变化, 结构转变=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="5min")
        if not {"close", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, volume = panel["close"], panel["volume"]
        ret_1m = close.pct_change()
        day = ret_1m.index.normalize()
        divs: dict = {}
        for dt in sorted(set(day)):
            grp_r = ret_1m.loc[day == dt]
            grp_v = volume.loc[day == dt]
            if len(grp_r) < 60:
                continue
            vals = {}
            for col in grp_r.columns:
                r = grp_r[col].dropna()
                v = grp_v[col].dropna()
                common = r.index.intersection(v.index)
                if len(common) < 60:
                    continue
                r_c = r.loc[common]
                v_c = v.loc[common]
                if r_c.std() < 1e-12 or v_c.std() < 1e-12:
                    vals[col] = 0.0
                    continue
                corr20 = r_c.iloc[-20:].corr(v_c.iloc[-20:], method="spearman")
                corr60 = r_c.iloc[-60:].corr(v_c.iloc[-60:], method="spearman")
                c20 = corr20 if not np.isnan(corr20) else 0.0
                c60 = corr60 if not np.isnan(corr60) else 0.0
                vals[col] = float(c20 - c60)
            if vals:
                divs[dt] = pd.Series(vals)
        if not divs:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(divs).T
        daily.index = pd.DatetimeIndex(daily.index)
        slope = daily.diff()
        return slope.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 397. intraday_price_range_volume_ratio — 价格区间量能偏斜 (A3 变体)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_price_range_volume_ratio_20d", category="intraday_advanced")
class IntradayPriceRangeVolumeRatio20d(Factor):
    """价格区间量能偏斜因子 (资料A3 变体, 基于5min).

    落在近20日价格区间上半部的成交量占比: Σvol[price > mid(N日)] / Σvol.
    与 #19 (日内中位) 不同: 本因子用 N 日区间中位, 捕捉跨日相对位置.
    上半部放量 → 资金在高位区承接/派发活跃 → 正向.
    方向: 正向.
    """
    name = "intraday_price_range_volume_ratio_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "价格区间量能偏斜 (N日区间上半部量占比, 高位承接=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="5min")
        if not {"close", "volume", "high", "low"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, volume, high, low = panel["close"], panel["volume"], panel["high"], panel["low"]
        day = close.index.normalize()
        ratios: dict = {}
        sorted_days = sorted(set(day))
        lookback = 20
        for idx, dt in enumerate(sorted_days):
            grp_c = close.loc[day == dt]
            if len(grp_c) < 10:
                continue
            if idx < lookback:
                continue
            vals = {}
            for col in grp_c.columns:
                c = grp_c[col].dropna()
                if len(c) < 10:
                    continue
                h_c = high.loc[day == dt, col].dropna()
                l_c = low.loc[day == dt, col].dropna()
                # N日区间中位: 用近20日的日high/low
                past_h = [high.loc[day == rd, col].dropna().max() for rd in sorted_days[idx - lookback:idx]]
                past_l = [low.loc[day == rd, col].dropna().min() for rd in sorted_days[idx - lookback:idx]]
                past_h = [x for x in past_h if not np.isnan(x)]
                past_l = [x for x in past_l if not np.isnan(x)]
                if len(past_h) < 5 or len(past_l) < 5:
                    continue
                hi = max(past_h); lo = min(past_l)
                if hi - lo < 1e-12:
                    continue
                mid = (hi + lo) / 2.0
                vol = volume.loc[day == dt, col].dropna()
                common = c.index.intersection(vol.index)
                if len(common) < 10:
                    continue
                c_c = c.loc[common]
                v_c = vol.loc[common]
                upper = v_c[c_c > mid].sum()
                lower = v_c[c_c <= mid].sum()
                total = upper + lower
                if total < 1e-12:
                    continue
                vals[col] = float(upper / total)
            if vals:
                ratios[dt] = pd.Series(vals)
        if not ratios:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(ratios).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 398. intraday_range_vol_skew_delta — 区间量能偏斜边际 (原创改进)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_range_vol_skew_delta_20d", category="intraday_advanced")
class IntradayRangeVolSkewDelta20d(Factor):
    """区间量能偏斜边际因子 (原创: A3 改进).

    A3 水平值的跨日变化 (偏斜的边际方向):
    偏斜快速上移 → 资金正加速向高位区集聚 → 更强的承接信号.
    与 A3 互补: A3 看偏斜状态, 本因子看偏斜变化速度 (领先指标).
    方向: 正向.
    """
    name = "intraday_range_vol_skew_delta_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "区间量能偏斜边际 (偏斜跨日变化, 加速集聚=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="5min")
        if not {"close", "volume", "high", "low"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, volume, high, low = panel["close"], panel["volume"], panel["high"], panel["low"]
        day = close.index.normalize()
        ratios: dict = {}
        sorted_days = sorted(set(day))
        lookback = 20
        for idx, dt in enumerate(sorted_days):
            grp_c = close.loc[day == dt]
            if len(grp_c) < 10:
                continue
            if idx < lookback:
                continue
            vals = {}
            for col in grp_c.columns:
                c = grp_c[col].dropna()
                if len(c) < 10:
                    continue
                past_h = [high.loc[day == rd, col].dropna().max() for rd in sorted_days[idx - lookback:idx]]
                past_l = [low.loc[day == rd, col].dropna().min() for rd in sorted_days[idx - lookback:idx]]
                past_h = [x for x in past_h if not np.isnan(x)]
                past_l = [x for x in past_l if not np.isnan(x)]
                if len(past_h) < 5 or len(past_l) < 5:
                    continue
                hi = max(past_h); lo = min(past_l)
                if hi - lo < 1e-12:
                    continue
                mid = (hi + lo) / 2.0
                vol = volume.loc[day == dt, col].dropna()
                common = c.index.intersection(vol.index)
                if len(common) < 10:
                    continue
                c_c = c.loc[common]
                v_c = vol.loc[common]
                upper = v_c[c_c > mid].sum()
                lower = v_c[c_c <= mid].sum()
                total = upper + lower
                if total < 1e-12:
                    continue
                vals[col] = float(upper / total)
            if vals:
                ratios[dt] = pd.Series(vals)
        if not ratios:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(ratios).T
        daily.index = pd.DatetimeIndex(daily.index)
        delta = daily.diff()
        return delta.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 399. intraday_false_breakout_retrace — 假突破回撤 (A5 复现)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_false_breakout_retrace_20d", category="intraday_advanced")
class IntradayFalseBreakoutRetrace20d(Factor):
    """假突破回撤因子 (资料A5, 基于5min).

    价格触及近20根5min高低点后, 5根内回撤幅度 (突破失败度量):
    max(突破幅度后回撤) / ATR.
    假突破频繁 → 突破信号不可靠 → 反转风险 → 负向.
    与现有突破类 (方向) 互补: 本因子是突破质量/失败检测.
    方向: 负向.
    """
    name = "intraday_false_breakout_retrace_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "假突破回撤 (突破失败度/ATR, 不可靠=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="5min")
        if not {"high", "low", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, low, close = panel["high"], panel["low"], panel["close"]
        day = close.index.normalize()
        fakes: dict = {}
        for dt in sorted(set(day)):
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 30:
                continue
            vals = {}
            for col in grp_c.columns:
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                c = grp_c[col].dropna()
                common = h.index.intersection(l.index).intersection(c.index)
                if len(common) < 30:
                    continue
                h_c, l_c, c_c = (x.loc[common] for x in (h, l, c))
                # ATR 近似: 5min (high-low) 的20根均值
                hl = (h_c - l_c).iloc[-20:]
                atr = hl.mean()
                if atr < 1e-12:
                    vals[col] = 0.0
                    continue
                retraces = []
                for i in range(20, len(c_c)):
                    window_h = h_c.iloc[i - 20:i]
                    window_l = l_c.iloc[i - 20:i]
                    px = c_c.iloc[i]
                    if px >= window_h.max():
                        # 向上突破后回撤: 之后5根从最高点回落
                        fwd = h_c.iloc[i:i + 5]
                        peak = fwd.max() if len(fwd) > 0 else px
                        close_after = c_c.iloc[min(i + 4, len(c_c) - 1)]
                        retraces.append((peak - close_after) / atr)
                    elif px <= window_l.min():
                        # 向下突破后回抽
                        fwd = l_c.iloc[i:i + 5]
                        trough = fwd.min() if len(fwd) > 0 else px
                        close_after = c_c.iloc[min(i + 4, len(c_c) - 1)]
                        retraces.append((close_after - trough) / atr)
                if not retraces:
                    vals[col] = 0.0
                else:
                    vals[col] = float(np.mean(retraces))
            if vals:
                fakes[dt] = pd.Series(vals)
        if not fakes:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(fakes).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 400. intraday_breakout_quality — 突破质量 (原创改进)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_breakout_quality_20d", category="intraday_advanced")
class IntradayBreakoutQuality20d(Factor):
    """突破质量因子 (原创: A5 改进).

    真实突破占比 = 突破后5根内站稳(未回撤超0.5×ATR)的次数 / 总突破次数.
    quality 高 → 突破可信度高 → 顺势信号 → 正向.
    与 A5 (失败度) 互补: 本因子是"成功度", 且按 ATR 归一可比.
    方向: 正向.
    """
    name = "intraday_breakout_quality_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "突破质量 (站稳占比, 可信突破=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="5min")
        if not {"high", "low", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        high, low, close = panel["high"], panel["low"], panel["close"]
        day = close.index.normalize()
        qualities: dict = {}
        for dt in sorted(set(day)):
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 30:
                continue
            vals = {}
            for col in grp_c.columns:
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                c = grp_c[col].dropna()
                common = h.index.intersection(l.index).intersection(c.index)
                if len(common) < 30:
                    continue
                h_c, l_c, c_c = (x.loc[common] for x in (h, l, c))
                hl = (h_c - l_c).iloc[-20:]
                atr = hl.mean()
                if atr < 1e-12:
                    vals[col] = 0.0
                    continue
                n_break = 0
                n_hold = 0
                for i in range(20, len(c_c)):
                    window_h = h_c.iloc[i - 20:i]
                    window_l = l_c.iloc[i - 20:i]
                    px = c_c.iloc[i]
                    if px >= window_h.max() and window_h.max() - window_h.iloc[-1] > 0:
                        n_break += 1
                        close_after = c_c.iloc[min(i + 4, len(c_c) - 1)]
                        if close_after >= px - 0.5 * atr:
                            n_hold += 1
                    elif px <= window_l.min() and window_l.iloc[-1] - window_l.min() > 0:
                        n_break += 1
                        close_after = c_c.iloc[min(i + 4, len(c_c) - 1)]
                        if close_after <= px + 0.5 * atr:
                            n_hold += 1
                vals[col] = float(n_hold / n_break) if n_break > 0 else 0.5
            if vals:
                qualities[dt] = pd.Series(vals)
        if not qualities:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(qualities).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 401. intraday_session_effect_z — 时段效应 z-score (A6 复现)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_session_effect_z_20d", category="intraday_advanced")
class IntradaySessionEffectZ20d(Factor):
    """时段效应 z-score 因子 (资料A6, 基于5min).

    将当日分为4个时段 (开盘30min/午前/午后/收盘30min),
    当前时段收益对近期(20日)同段均值的 z-score.
    时段 z 正 → 该时段显著强于历史常态 (开盘动量/尾盘反转等结构) → 顺势.
    与现有 session_* (一致性/形态) 互补: 本因子是时段收益的历史偏离度.
    方向: 随符号.
    """
    name = "intraday_session_effect_z_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "时段效应z (时段收益对20日同段均值偏离)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="5min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        sorted_days = sorted(set(day))
        # 先算每日每时段的收益
        seg_rets: dict = {}
        for dt in sorted_days:
            grp = close.loc[day == dt]
            n = len(grp)
            if n < 16:
                continue
            quart = n // 4
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < n:
                    continue
                segs = []
                for k in range(4):
                    s = k * quart
                    e = n if k == 3 else (k + 1) * quart
                    if e - s < 2:
                        segs.append(0.0)
                    else:
                        segs.append(float(c.iloc[e - 1] / c.iloc[s] - 1.0))
                vals[col] = segs[3]  # 用最后时段(收盘)收益做代表
            if vals:
                seg_rets[dt] = pd.Series(vals)
        if len(seg_rets) < 25:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(seg_rets).T
        df.index = pd.DatetimeIndex(df.index)
        mean = df.reindex(index=dates).rolling(20, min_periods=5).mean()
        std = df.reindex(index=dates).rolling(20, min_periods=5).std().replace(0, np.nan)
        z = (df.reindex(index=dates) - mean) / std
        return z.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 402. intraday_session_vol_state — 时段效应×波动状态 (原创改进)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_session_vol_state_20d", category="intraday_advanced")
class IntradaySessionVolState20d(Factor):
    """时段效应×波动状态因子 (原创: A6 改进).

    时段 z-score 与时段波动状态交互: z × sign(当前时段波动 vs 20日同段波动).
    时段强收益 + 波动同步放大 → 有资金驱动的真实时段行情 → 更强信号.
    高收益但缩量低波动 → 可能是流动性薄的虚假波动 → 信号打折.
    方向: 正向.
    """
    name = "intraday_session_vol_state_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "时段效应×波动状态 (收益强+波动同步放大=真实行情)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="5min")
        if not {"close", "high", "low"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, high, low = panel["close"], panel["high"], panel["low"]
        day = close.index.normalize()
        sorted_days = sorted(set(day))
        seg_rets: dict = {}
        seg_vols: dict = {}
        for dt in sorted_days:
            grp_c = close.loc[day == dt]
            n = len(grp_c)
            if n < 16:
                continue
            quart = n // 4
            rvals, vvals = {}, {}
            for col in grp_c.columns:
                c = grp_c[col].dropna()
                if len(c) < n:
                    continue
                s = 3 * quart
                e = n
                if e - s < 2:
                    rvals[col] = 0.0
                    vvals[col] = 0.0
                    continue
                rvals[col] = float(c.iloc[e - 1] / c.iloc[s] - 1.0)
                h_c = high.loc[day == dt, col].dropna()
                l_c = low.loc[day == dt, col].dropna()
                hh = h_c.iloc[s:e] if len(h_c) >= e else h_c.iloc[-quart:]
                ll = l_c.iloc[s:e] if len(l_c) >= e else l_c.iloc[-quart:]
                if len(hh) > 0 and len(ll) > 0:
                    vvals[col] = float(hh.max() - ll.min())
                else:
                    vvals[col] = 0.0
            if rvals:
                seg_rets[dt] = pd.Series(rvals)
                seg_vols[dt] = pd.Series(vvals)
        if len(seg_rets) < 25 or len(seg_vols) < 25:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret_df = pd.DataFrame(seg_rets).T
        vol_df = pd.DataFrame(seg_vols).T
        ret_df.index = pd.DatetimeIndex(ret_df.index)
        vol_df.index = pd.DatetimeIndex(vol_df.index)
        r_mean = ret_df.reindex(index=dates).rolling(20, min_periods=5).mean()
        r_std = ret_df.reindex(index=dates).rolling(20, min_periods=5).std().replace(0, np.nan)
        z = (ret_df.reindex(index=dates) - r_mean) / r_std
        v_mean = vol_df.reindex(index=dates).rolling(20, min_periods=5).mean()
        v_state = np.sign(vol_df.reindex(index=dates) - v_mean.replace(0, np.nan))
        score = z * v_state
        return score.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 403. intraday_oi_volume_crowding — OI-量能拥挤度 (A7 复现)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_oi_volume_crowding_20d", category="intraday_advanced")
class IntradayOiVolumeCrowding20d(Factor):
    """OI-量能拥挤度因子 (资料A7, 基于5min+OI).

    拥挤度 = OI增速20根z-score × 0.5 + (vol/OI)分位 × 0.5 合成.
    拥挤度高 → 持仓快速膨胀+高换手 → 拥挤交易 → 反转/降杠杆 → 负向.
    与现有 OI 因子(水平/变化)不同: 本因子是拥挤度合成.
    方向: 负向.
    """
    name = "intraday_oi_volume_crowding_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "OI-量能拥挤度 (OI增速z+换手分位合成, 拥挤=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="5min")
        if not {"position", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        oi, volume = panel["position"], panel["volume"]
        day = oi.index.normalize()
        crowding: dict = {}
        for dt in sorted(set(day)):
            grp_o = oi.loc[day == dt]
            grp_v = volume.loc[day == dt]
            if len(grp_o) < 30:
                continue
            vals = {}
            for col in grp_o.columns:
                o = grp_o[col].dropna()
                v = grp_v[col].dropna()
                common = o.index.intersection(v.index)
                if len(common) < 30:
                    continue
                o_c = o.loc[common]
                v_c = v.loc[common]
                oi_growth = o_c.pct_change().dropna()
                if len(oi_growth) < 20:
                    continue
                g_mean = oi_growth.mean()
                g_std = oi_growth.std(ddof=0)
                oi_z = float((oi_growth.iloc[-1] - g_mean) / g_std) if g_std > 1e-12 else 0.0
                oi_z = max(-3.0, min(3.0, oi_z))
                # vol/OI 换手分位 (日内时序分位)
                turnover = v_c / o_c.replace(0, np.nan)
                t_curr = turnover.iloc[-1]
                t_rank = float((turnover.dropna() <= t_curr).mean())
                crowding_val = 0.5 * (oi_z + 3.0) / 6.0 + 0.5 * t_rank
                vals[col] = crowding_val
            if vals:
                crowding[dt] = pd.Series(vals)
        if not crowding:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(crowding).T
        daily.index = pd.DatetimeIndex(daily.index)
        return (-daily).rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 404. intraday_crowding_extreme_reversal — 拥挤度极值反转 (原创改进)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_crowding_extreme_reversal_20d", category="intraday_advanced")
class IntradayCrowdingExtremeReversal20d(Factor):
    """拥挤度极值反转因子 (原创: A7 改进).

    A7 拥挤度取 20 日时序分位, 极高分位(>80%) 触发反转信号:
    拥挤度分位-0.8 (0~1), 仅在高拥挤区间给出负值.
    相比 A7 的线性负向, 本因子是"阈值触发"式反转, 避免对温和拥挤过度反应.
    方向: 负向.
    """
    name = "intraday_crowding_extreme_reversal_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "拥挤度极值反转 (20日分位-0.8, 极端拥挤才触发)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="5min")
        if not {"position", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        oi, volume = panel["position"], panel["volume"]
        day = oi.index.normalize()
        crowding: dict = {}
        for dt in sorted(set(day)):
            grp_o = oi.loc[day == dt]
            grp_v = volume.loc[day == dt]
            if len(grp_o) < 30:
                continue
            vals = {}
            for col in grp_o.columns:
                o = grp_o[col].dropna()
                v = grp_v[col].dropna()
                common = o.index.intersection(v.index)
                if len(common) < 30:
                    continue
                o_c = o.loc[common]
                v_c = v.loc[common]
                oi_growth = o_c.pct_change().dropna()
                if len(oi_growth) < 20:
                    continue
                g_mean = oi_growth.mean()
                g_std = oi_growth.std(ddof=0)
                oi_z = float((oi_growth.iloc[-1] - g_mean) / g_std) if g_std > 1e-12 else 0.0
                oi_z = max(-3.0, min(3.0, oi_z))
                turnover = v_c / o_c.replace(0, np.nan)
                t_curr = turnover.iloc[-1]
                t_rank = float((turnover.dropna() <= t_curr).mean())
                crowding_val = 0.5 * (oi_z + 3.0) / 6.0 + 0.5 * t_rank
                vals[col] = crowding_val
            if vals:
                crowding[dt] = pd.Series(vals)
        if not crowding:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(crowding).T
        daily.index = pd.DatetimeIndex(daily.index)
        rank = daily.rolling(20, min_periods=5).apply(
            lambda x: float((x.iloc[:-1] <= x.iloc[-1]).sum() / max(1, len(x) - 1)), raw=False)
        reversal = (rank - 0.8)
        return (-reversal).reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 405. intraday_volume_oi_divergence — 量仓四象限 (A8 复现)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_volume_oi_divergence_20d", category="intraday_advanced")
class IntradayVolumeOiDivergence20d(Factor):
    """量仓四象限因子 (资料A8, 基于5min+OI).

    四象限: sign(Δvol) × sign(ΔOI) 组合 — 放量增仓/放量减仓/缩量增仓/缩量减仓.
    放量减仓 (换手出货) 占比高 → 短线转弱 → 负向.
    缩量增仓 (惜售蓄势) 占比高 → 蓄势 → 正向.
    输出 = 缩量增仓占比 − 放量减仓占比.
    方向: 正向.
    """
    name = "intraday_volume_oi_divergence_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "量仓四象限 (缩量增仓-放量减仓占比, 惜售蓄势=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="5min")
        if not {"position", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        oi, volume = panel["position"], panel["volume"]
        day = oi.index.normalize()
        divergences: dict = {}
        for dt in sorted(set(day)):
            grp_o = oi.loc[day == dt]
            grp_v = volume.loc[day == dt]
            if len(grp_o) < 20:
                continue
            vals = {}
            for col in grp_o.columns:
                o = grp_o[col].dropna()
                v = grp_v[col].dropna()
                common = o.index.intersection(v.index)
                if len(common) < 20:
                    continue
                o_c = o.loc[common]
                v_c = v.loc[common]
                d_oi = o_c.diff().dropna()
                d_vol = v_c.diff().dropna()
                idx = d_oi.index.intersection(d_vol.index)
                if len(idx) < 10:
                    continue
                d_oi_c = d_oi.loc[idx]
                d_vol_c = d_vol.loc[idx]
                vol_up = (d_vol_c > 0).sum()
                vol_dn = (d_vol_c < 0).sum()
                oi_up = (d_oi_c > 0).sum()
                oi_dn = (d_oi_c < 0).sum()
                n = len(idx)
                # 放量减仓: vol_up & oi_dn; 缩量增仓: vol_dn & oi_up
                sell_off = ((d_vol_c > 0) & (d_oi_c < 0)).sum()
                accumulate = ((d_vol_c < 0) & (d_oi_c > 0)).sum()
                vals[col] = float((accumulate - sell_off) / n)
            if vals:
                divergences[dt] = pd.Series(vals)
        if not divergences:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(divergences).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 406. intraday_volume_oi_price_confirm — 量仓×价格交叉验证 (原创改进)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_volume_oi_price_confirm_20d", category="intraday_advanced")
class IntradayVolumeOiPriceConfirm20d(Factor):
    """量仓×价格交叉验证因子 (原创: A8 改进).

    四象限与价格方向交叉验证: 放量增仓+价格上涨 = 新多进场 (强多);
    放量增仓+价格下跌 = 新空打压 (强空).
    score = sign(Δvol) × sign(ΔOI) × sign(Δprice) — 全同号才出强信号.
    相比 A8 仅看量仓状态, 本因子加入价格方向确认.
    方向: 正向.
    """
    name = "intraday_volume_oi_price_confirm_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "量仓价格交叉 (signΔvol×signΔOI×signΔprice, 同号强信号)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="5min")
        if not {"position", "volume", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        oi, volume, close = panel["position"], panel["volume"], panel["close"]
        day = oi.index.normalize()
        confirms: dict = {}
        for dt in sorted(set(day)):
            grp_o = oi.loc[day == dt]
            grp_v = volume.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_o) < 20:
                continue
            vals = {}
            for col in grp_o.columns:
                o = grp_o[col].dropna()
                v = grp_v[col].dropna()
                c = grp_c[col].dropna()
                common = o.index.intersection(v.index).intersection(c.index)
                if len(common) < 20:
                    continue
                o_c = o.loc[common]
                v_c = v.loc[common]
                c_c = c.loc[common]
                d_oi = np.sign(o_c.diff())
                d_vol = np.sign(v_c.diff())
                d_px = np.sign(c_c.diff())
                valid = (d_oi != 0) & (d_vol != 0) & (d_px != 0)
                if valid.sum() < 5:
                    vals[col] = 0.0
                    continue
                score = (d_oi[valid] * d_vol[valid] * d_px[valid]).mean()
                vals[col] = float(score)
            if vals:
                confirms[dt] = pd.Series(vals)
        if not confirms:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(confirms).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 407. intraday_rv_fast_slow_divergence — RV 快慢窗背离 (A9 复现)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_rv_fast_slow_divergence_20d", category="intraday_advanced")
class IntradayRvFastSlowDivergence20d(Factor):
    """RV 快慢窗背离因子 (资料A9, 基于5min).

    RV5 (5根已实现波动) − RV20 (20根滚动) 的 z-score 背离 (类 IV-HV 背离).
    快线升+慢线降 → 波动结构变化预警 → 波动放大 → 负向.
    与现有波动类(单窗口)互补: 本因子是快慢窗口结构差.
    方向: 负向.
    """
    name = "intraday_rv_fast_slow_divergence_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "RV快慢窗背离 (rv5-rv20的z, 结构变化=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="5min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        ret_1m = close.pct_change()
        day = ret_1m.index.normalize()
        divergences: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 40:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 40:
                    continue
                rv5 = (r.iloc[-5:] ** 2).sum()
                rv20 = (r.iloc[-20:] ** 2).sum()
                if rv20 < 1e-12:
                    vals[col] = 0.0
                    continue
                ratio = rv5 / rv20  # 快慢背离 (归一化比值, 1=一致)
                vals[col] = float(ratio)
            if vals:
                divergences[dt] = pd.Series(vals)
        if not divergences:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(divergences).T
        daily.index = pd.DatetimeIndex(daily.index)
        mean = daily.rolling(20, min_periods=5).mean()
        std = daily.rolling(20, min_periods=5).std().replace(0, np.nan)
        z = (daily - mean) / std
        return (-z).reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 408. intraday_rv_divergence_persistence — RV 背离持续性 (原创改进)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_rv_divergence_persistence_20d", category="intraday_advanced")
class IntradayRvDivergencePersistence20d(Factor):
    """RV 背离持续性因子 (原创: A9 改进).

    A9 看单日快慢背离, 本因子统计背离(rv5>rv20 或 <)连续持续的根数占比:
    背离持续越久 → 波动 regime 越稳定 (而非单日噪声).
    持续高波动背离 → 波动放大已是常态 → 风险持续 → 负向.
    方向: 负向.
    """
    name = "intraday_rv_divergence_persistence_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "RV背离持续性 (背离连续根数占比, 波动regime持续=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="5min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        ret_1m = close.pct_change()
        day = ret_1m.index.normalize()
        persist: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 40:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 40:
                    continue
                # 滚动5根 RV vs 20根 RV 的方向序列
                rv5_series = (r.rolling(5).apply(lambda x: np.sum(x ** 2), raw=True))
                rv20_series = (r.rolling(20).apply(lambda x: np.sum(x ** 2), raw=True))
                diff = rv5_series - rv20_series
                diff = diff.dropna()
                if len(diff) < 10:
                    continue
                signs = np.sign(diff.values)
                # 最长连续同号 run / 总长
                max_run = 1
                cur = 1
                for i in range(1, len(signs)):
                    if signs[i] == signs[i - 1] and signs[i] != 0:
                        cur += 1
                        max_run = max(max_run, cur)
                    else:
                        cur = 1
                vals[col] = float(max_run / len(signs))
            if vals:
                persist[dt] = pd.Series(vals)
        if not persist:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(persist).T
        daily.index = pd.DatetimeIndex(daily.index)
        return (-daily).rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 409. intraday_rv_compression — RV 波动压缩度 (A10 改名复现)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_rv_compression_20d", category="intraday_advanced")
class IntradayRvCompression20d(Factor):
    """RV 波动压缩度因子 (资料A10, 改名避免与 #142 冲突).

    当前 RV 的 20 日历史分位 (压缩度): 分位越低 → 波动越压缩.
    压缩后 (分位<20%) → 预判波动放大 ("弹簧"效应) → 正向 (低压缩分位=正向).
    与 #142 (振幅比) 不同: 本因子用 RV 的时序分位.
    方向: 正向 (取 1-分位).
    """
    name = "intraday_rv_compression_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "RV波动压缩 (20日分位, 压缩后预判放大=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="5min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        ret_1m = close.pct_change()
        day = ret_1m.index.normalize()
        rvs: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 20:
                    continue
                vals[col] = float((r ** 2).sum())
            if vals:
                rvs[dt] = pd.Series(vals)
        if not rvs:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(rvs).T
        daily.index = pd.DatetimeIndex(daily.index)
        rank = daily.rolling(20, min_periods=5).apply(
            lambda x: float((x.iloc[:-1] <= x.iloc[-1]).sum() / max(1, len(x) - 1)), raw=False)
        compression = 1.0 - rank  # 分位越低(压缩) → 值越高(正向)
        return compression.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 410. intraday_rv_compression_breakout — 压缩后扩张强度 (原创改进)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_rv_compression_breakout_20d", category="intraday_advanced")
class IntradayRvCompressionBreakout20d(Factor):
    """压缩后扩张强度因子 (原创: A10 改进).

    A10 只识别压缩状态, 本因子捕捉压缩后的扩张幅度:
    今日 RV / 前5日压缩期最低 RV — 压缩越深、反弹越猛 → 波动扩张信号越强.
    高扩张 → 波动 regime 切换完成 → 风险释放 → 负向.
    方向: 负向.
    """
    name = "intraday_rv_compression_breakout_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "压缩后扩张 (今日RV/5日最低RV, 扩张完成=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="5min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        ret_1m = close.pct_change()
        day = ret_1m.index.normalize()
        rvs: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                r = grp[col].dropna()
                if len(r) < 20:
                    continue
                vals[col] = float((r ** 2).sum())
            if vals:
                rvs[dt] = pd.Series(vals)
        if not rvs:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(rvs).T
        daily.index = pd.DatetimeIndex(daily.index)
        min5 = daily.rolling(5, min_periods=3).min().shift(1)
        expansion = daily / min5.replace(0, np.nan)
        return (-expansion).reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 411. intraday_ma_count_bullish — 均线簇多空强度 (A11 复现)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_ma_count_bullish_20d", category="intraday_advanced")
class IntradayMaCountBullish20d(Factor):
    """均线簇多空强度因子 (资料A11, 基于5min).

    收盘价位于 5/10/20/30/60/120 根5min均线上方的数量 (0-6).
    均线簇上方数量多 → 全面多头排列 → 趋势确认 → 正向.
    与 A1 互补: A1 看斜率投票, 本因子看价格对均线的相对位置.
    方向: 正向.
    """
    name = "intraday_ma_count_bullish_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "均线簇多空 (收盘在5/10/20/30/60/120均线上方数/6)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="5min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        counts: dict = {}
        windows = (5, 10, 20, 30, 60, 120)
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 5:
                continue
            vals = {}
            for col in grp.columns:
                s = close[col].dropna()
                # 取截至当日最后一根的全期序列 (跨日 MA)
                last_ts = grp.index[-1]
                hist = s.loc[:last_ts]
                if len(hist) < 5:
                    continue
                px = hist.iloc[-1]
                above = 0
                for w in windows:
                    if len(hist) >= w:
                        ma_w = hist.iloc[-w:].mean()
                        if px > ma_w:
                            above += 1
                vals[col] = float(above / len(windows))
            if vals:
                counts[dt] = pd.Series(vals)
        if not counts:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(counts).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 412. intraday_ma_count_weighted — 加权均线簇强度 (原创改进)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_ma_count_weighted_20d", category="intraday_advanced")
class IntradayMaCountWeighted20d(Factor):
    """加权均线簇强度因子 (原创: A11 改进).

    在 A11 基础上按均线周期加权: 长周期均线权重更高 (趋势更可靠).
    score = Σ w_i × sign(px − MA_i) / Σ w_i, w = 周期长度.
    长均线确认的多头比短均线更可信 → 过滤短周期噪声.
    方向: 正向.
    """
    name = "intraday_ma_count_weighted_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "加权均线簇 (长周期均线权重大, 可靠趋势=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="5min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        scores: dict = {}
        windows = (5, 10, 20, 30, 60, 120)
        weights = {w: float(w) for w in windows}
        w_sum = sum(weights.values())
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 5:
                continue
            vals = {}
            for col in grp.columns:
                s = close[col].dropna()
                last_ts = grp.index[-1]
                hist = s.loc[:last_ts]
                if len(hist) < 5:
                    continue
                px = hist.iloc[-1]
                score = 0.0
                for w in windows:
                    if len(hist) >= w:
                        ma_w = hist.iloc[-w:].mean()
                        score += weights[w] * (1.0 if px > ma_w else -1.0)
                vals[col] = float(score / w_sum)
            if vals:
                scores[dt] = pd.Series(vals)
        if not scores:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(scores).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 413. intraday_overnight_gap_reaction — 跳空后反应模式 (A13 复现)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_overnight_gap_reaction_20d", category="intraday_advanced")
class IntradayOvernightGapReaction20d(Factor):
    """跳空后反应模式因子 (资料A13, 基于5min).

    开盘跳空 (open−prev_close) 后, 30分钟内的回补/延续方向.
    输出 = 跳空后30min收益方向 × 跳空方向 (正=缺口延续, 负=缺口回补).
    正 → 缺口延续 → 顺势强 → 正向. 负 → 高开低走/低开高走 → 反转 → 负向.
    与 #27 (日内一致性) 不同: 本因子聚焦开盘后30min窗口.
    方向: 随符号.
    """
    name = "intraday_overnight_gap_reaction_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "跳空后反应 (30min延续=正向, 回补=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="5min")
        if not {"open", "close"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        open_px, close = panel["open"], panel["close"]
        day = close.index.normalize()
        reactions: dict = {}
        prev_close: dict = {}
        sorted_days = sorted(set(day))
        for dt in sorted_days:
            grp_o = open_px.loc[day == dt]
            grp_c = close.loc[day == dt]
            if len(grp_c) < 6:
                continue
            vals = {}
            for col in grp_c.columns:
                o = grp_o[col].dropna()
                c = grp_c[col].dropna()
                if len(o) < 1 or len(c) < 6:
                    continue
                o_first = o.iloc[0]
                prev_c = prev_close.get(col)
                if prev_c is None or prev_c < 1e-12 or o_first < 1e-12:
                    prev_close[col] = c.iloc[-1]
                    continue
                gap = o_first / prev_c - 1.0
                # 跳空后30分钟 (6根5min) 收益
                c_30 = c.iloc[5] if len(c) >= 6 else c.iloc[-1]
                reaction = c_30 / o_first - 1.0
                vals[col] = float(np.sign(gap) * reaction)
                prev_close[col] = c.iloc[-1]
            if vals:
                reactions[dt] = pd.Series(vals)
        if not reactions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(reactions).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 414. intraday_gap_fill_speed — 缺口回补速度 (原创改进)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_gap_fill_speed_20d", category="intraday_advanced")
class IntradayGapFillSpeed20d(Factor):
    """缺口回补速度因子 (原创: A13 改进).

    A13 只判断回补与否, 本因子度量回补的深度与速度:
    跳空幅度 / 回补用时(根数) — 回补越快越深 → 开盘定价错误越严重 → 反转信号越强.
    高回补速度 → 开盘方向被迅速否定 → 负向.
    方向: 负向.
    """
    name = "intraday_gap_fill_speed_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "缺口回补速度 (跳空/回补用时, 快速回补=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="5min")
        if not {"open", "close", "high", "low"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        open_px, close, high, low = panel["open"], panel["close"], panel["high"], panel["low"]
        day = close.index.normalize()
        speeds: dict = {}
        prev_close: dict = {}
        sorted_days = sorted(set(day))
        for dt in sorted_days:
            grp_o = open_px.loc[day == dt]
            grp_c = close.loc[day == dt]
            grp_h = high.loc[day == dt]
            grp_l = low.loc[day == dt]
            if len(grp_c) < 6:
                continue
            vals = {}
            for col in grp_c.columns:
                o = grp_o[col].dropna()
                c = grp_c[col].dropna()
                h = grp_h[col].dropna()
                l = grp_l[col].dropna()
                if len(o) < 1 or len(c) < 6:
                    continue
                o_first = o.iloc[0]
                prev_c = prev_close.get(col)
                if prev_c is None or prev_c < 1e-12 or o_first < 1e-12:
                    prev_close[col] = c.iloc[-1]
                    continue
                gap = abs(o_first / prev_c - 1.0)
                # 找到回补(价格回到prev_close)所需根数
                fill_idx = None
                if o_first > prev_c:
                    l_c = l.loc[:c.index[-1]]
                    crossed = l_c < prev_c
                else:
                    h_c = h.loc[:c.index[-1]]
                    crossed = h_c > prev_c
                crossed_pos = np.where(crossed.values)[0]
                if len(crossed_pos) > 0:
                    fill_idx = int(crossed_pos[0])
                if fill_idx is not None and fill_idx >= 1:
                    speed = gap / max(1, fill_idx)
                    vals[col] = speed  # 高速度=快速回补=负向
                else:
                    vals[col] = 0.0  # 未回补: 缺口延续, 中性
                prev_close[col] = c.iloc[-1]
            if vals:
                speeds[dt] = pd.Series(vals)
        if not speeds:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(speeds).T
        daily.index = pd.DatetimeIndex(daily.index)
        return (-daily).rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 415. intraday_implied_sector_beta — 板块隐含 β (A14 复现)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_implied_sector_beta_20d", category="intraday_advanced")
class IntradayImpliedSectorBeta20d(Factor):
    """板块隐含 β 因子 (资料A14, 基于5min).

    品种5min收益对板块内其他品种平均收益回归的滚动 β (60根).
    高 β → 品种与板块强联动 → 系统性驱动主导.
    输出为 |β| 的 z-score (联动强度), 高联动=板块效应稳定 → 正向.
    与现有板块因子(静态映射)不同: 本因子是动态回归暴露.
    方向: 正向.
    """
    name = "intraday_implied_sector_beta_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "板块隐含β (品种对板块均值回归β的|值|, 强联动=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="5min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        ret_1m = close.pct_change()
        day = ret_1m.index.normalize()
        betas: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 60:
                continue
            sub = grp.dropna(how="all")
            if sub.shape[1] < 2:
                continue
            other_mean = _sector_mean(sub)
            vals = {}
            for col in sub.columns:
                r_i = sub[col].dropna()
                r_o = other_mean[col]
                common = r_i.index.intersection(r_o.index)
                if len(common) < 60:
                    continue
                a = r_i.loc[common].iloc[-60:].values
                b = r_o.loc[common].iloc[-60:].values
                if b.std(ddof=0) < 1e-12:
                    vals[col] = 0.0
                    continue
                cov = np.cov(a, b)[0, 1]
                var = np.var(b)
                beta = cov / var if var > 1e-12 else 0.0
                vals[col] = abs(float(beta))
            if vals:
                betas[dt] = pd.Series(vals)
        if not betas:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(betas).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 416. intraday_beta_change_signal — 板块 β 突变 (原创改进)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_beta_change_signal_20d", category="intraday_advanced")
class IntradayBetaChangeSignal20d(Factor):
    """板块 β 突变因子 (原创: A14 改进).

    A14 看 β 水平, 本因子捕捉 β 的跨日突变:
    β 突增 → 品种突然与板块强联动 → 板块联动切换/资金合流 → 波动放大 → 负向.
    突变方向通过符号传递: β 突变本身是结构变化信号.
    方向: 负向.
    """
    name = "intraday_beta_change_signal_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "板块β突变 (β跨日变化, 联动切换=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="5min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        ret_1m = close.pct_change()
        day = ret_1m.index.normalize()
        betas: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 60:
                continue
            sub = grp.dropna(how="all")
            if sub.shape[1] < 2:
                continue
            other_mean = _sector_mean(sub)
            vals = {}
            for col in sub.columns:
                r_i = sub[col].dropna()
                r_o = other_mean[col]
                common = r_i.index.intersection(r_o.index)
                if len(common) < 60:
                    continue
                a = r_i.loc[common].iloc[-60:].values
                b = r_o.loc[common].iloc[-60:].values
                if b.std(ddof=0) < 1e-12:
                    vals[col] = 0.0
                    continue
                cov = np.cov(a, b)[0, 1]
                var = np.var(b)
                beta = cov / var if var > 1e-12 else 0.0
                vals[col] = float(beta)
            if vals:
                betas[dt] = pd.Series(vals)
        if not betas:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(betas).T
        daily.index = pd.DatetimeIndex(daily.index)
        change = daily.diff()
        return (-change.abs()).reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 417. intraday_money_flow_margin — 资金流边际 (A15 复现)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_money_flow_margin_20d", category="intraday_advanced")
class IntradayMoneyFlowMargin20d(Factor):
    """资金流边际因子 (资料A15, 基于5min).

    主动买卖差 (tick test: 涨=主动买, 跌=主动卖) 的近5根变化率 (边际而非水平).
    边际为正 → 大资金正在加速介入 → 企稳反弹模式 → 正向.
    与 #38 (水平比率) 互补: 本因子是边际变化.
    方向: 正向.
    """
    name = "intraday_money_flow_margin_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "资金流边际 (主动买卖差的5根变化率, 加速介入=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="5min")
        if not {"close", "volume"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, volume = panel["close"], panel["volume"]
        ret_1m = close.pct_change()
        day = ret_1m.index.normalize()
        margins: dict = {}
        for dt in sorted(set(day)):
            grp_r = ret_1m.loc[day == dt]
            grp_v = volume.loc[day == dt]
            if len(grp_r) < 20:
                continue
            vals = {}
            for col in grp_r.columns:
                r = grp_r[col].dropna()
                v = grp_v[col].dropna()
                common = r.index.intersection(v.index)
                if len(common) < 20:
                    continue
                r_c = r.loc[common]
                v_c = v.loc[common]
                signed = np.sign(r_c.values) * v_c.values
                # 边际: 后5根累计 signed − 前5根累计 signed
                if len(signed) < 10:
                    continue
                recent = signed[-5:].sum()
                prior = signed[-10:-5].sum()
                denom = abs(prior) + 1e-12
                margin = (recent - prior) / denom
                vals[col] = float(margin)
            if vals:
                margins[dt] = pd.Series(vals)
        if not margins:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(margins).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 418. intraday_money_price_divergence — 资金-价格背离 (原创改进)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_money_price_divergence_20d", category="intraday_advanced")
class IntradayMoneyPriceDivergence20d(Factor):
    """资金-价格背离因子 (原创: A15 改进).

    标准化资金流 (净主动买卖/总量) − 标准化价格涨幅 (各自 z-score) 之差:
    正背离 = 资金流入但价格未涨 (逆势吸筹) → 后续补涨 → 正向.
    负背离 = 价格涨但资金流出 → 涨势不实 → 负向.
    与 #165 (量-价顶背离) 不同: 本因子用主动买卖资金流.
    方向: 正向.
    """
    name = "intraday_money_price_divergence_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "资金价格背离 (资金流z-涨幅z, 逆势吸筹=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="5min")
        if not {"close", "volume", "open"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close, volume, open_px = panel["close"], panel["volume"], panel["open"]
        ret_1m = close.pct_change()
        day = ret_1m.index.normalize()
        divergences: dict = {}
        for dt in sorted(set(day)):
            grp_r = ret_1m.loc[day == dt]
            grp_v = volume.loc[day == dt]
            grp_o = open_px.loc[day == dt]
            if len(grp_r) < 20:
                continue
            vals = {}
            for col in grp_r.columns:
                r = grp_r[col].dropna()
                v = grp_v[col].dropna()
                o = grp_o[col].dropna()
                common = r.index.intersection(v.index).intersection(o.index)
                if len(common) < 20:
                    continue
                r_c = r.loc[common]
                v_c = v.loc[common]
                signed = np.sign(r_c.values) * v_c.values
                total = v_c.sum()
                if total < 1e-12:
                    continue
                net_flow = signed.sum() / total  # [-1,1]
                o_first = o.loc[common].iloc[0]
                if o_first < 1e-12:
                    continue
                price_ret = float(r_c.sum())
                vals[col] = float(net_flow - price_ret)
            if vals:
                divergences[dt] = pd.Series(vals)
        if not divergences:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(divergences).T
        daily.index = pd.DatetimeIndex(daily.index)
        # 截面标准化再时序平均 (资金流与收益量纲不同)
        z = _cs_zscore(daily.reindex(index=dates))
        return z.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 419. intraday_annualized_basis_z — 基差分位极值反转 (B1 复现)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_annualized_basis_z_20d", category="intraday_advanced")
class IntradayAnnualizedBasisZ20d(Factor):
    """基差分位极值反转因子 (资料B1, 基于5min term).

    近远月基差率 (far−near)/near 的 20 日时序分位, 中间区线性、极值区(>80%/<20%)反转:
    backwardation (近月升水) → 现货偏紧 → 正向; 但基差处历史极值 → 反转.
    注: 无到期日数据, 用近远月价差率代替年化.
    与 term_slope (水平) 不同: 本因子是分位+极值反转的完整升级.
    方向: 随结构.
    """
    name = "intraday_annualized_basis_z_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "基差分位极值反转 (近远月价差率20日分位, 极值反转)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_term_structure_panel(data, dates, universe, freq="5min")
        if "near_close" not in panel or "far_close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        near, far = panel["near_close"], panel["far_close"]
        day = near.index.normalize()
        basis: dict = {}
        for dt in sorted(set(day)):
            grp_n = near.loc[day == dt]
            grp_f = far.loc[day == dt]
            if len(grp_n) < 20:
                continue
            vals = {}
            for col in grp_n.columns:
                n = grp_n[col].dropna()
                f = grp_f[col].dropna()
                common = n.index.intersection(f.index)
                if len(common) < 20:
                    continue
                b = ((f.loc[common] - n.loc[common]) / n.loc[common].replace(0, np.nan)).dropna()
                if len(b) < 10:
                    continue
                vals[col] = float(b.mean())
            if vals:
                basis[dt] = pd.Series(vals)
        if not basis:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(basis).T
        df.index = pd.DatetimeIndex(df.index)
        rank = df.reindex(index=dates).rolling(20, min_periods=5).apply(
            lambda x: float(np.nan) if (x.dropna().empty or pd.isna(x.iloc[-1]))
            else float((x.dropna().iloc[:-1] <= x.dropna().iloc[-1]).sum() / max(1, len(x.dropna()) - 1)), raw=False)
        # 中间区线性 (rank-0.5), 极值区反转
        z = rank - 0.5
        rev = np.where(rank > 0.8, -(rank - 0.8) * 5.0,
                       np.where(rank < 0.2, (0.2 - rank) * 5.0, z.values))
        out = pd.DataFrame(rev, index=rank.index, columns=rank.columns)
        return out.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 420. intraday_basis_reversion_conviction — 基差反转信念 (原创改进)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_basis_reversion_conviction_20d", category="intraday_advanced")
class IntradayBasisReversionConviction20d(Factor):
    """基差反转信念因子 (原创: B1 改进).

    在 B1 极值反转基础上, 用基差自身波动率加权信念:
    基差极值 + 低波动 (基差稳定在极端) → 反转信念强; 高波动 → 基差本身在剧烈波动, 反转信号弱.
    信念 = 反转信号 / (1 + 基差20日波动).
    相比 B1 的纯分位反转, 加入信号可靠性调节.
    方向: 随结构.
    """
    name = "intraday_basis_reversion_conviction_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "基差反转信念 (极值反转/波动加权, 稳定极端=强信念)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_term_structure_panel(data, dates, universe, freq="5min")
        if "near_close" not in panel or "far_close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        near, far = panel["near_close"], panel["far_close"]
        day = near.index.normalize()
        basis: dict = {}
        for dt in sorted(set(day)):
            grp_n = near.loc[day == dt]
            grp_f = far.loc[day == dt]
            if len(grp_n) < 20:
                continue
            vals = {}
            for col in grp_n.columns:
                n = grp_n[col].dropna()
                f = grp_f[col].dropna()
                common = n.index.intersection(f.index)
                if len(common) < 20:
                    continue
                b = ((f.loc[common] - n.loc[common]) / n.loc[common].replace(0, np.nan)).dropna()
                if len(b) < 10:
                    continue
                vals[col] = float(b.mean())
            if vals:
                basis[dt] = pd.Series(vals)
        if not basis:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(basis).T
        df.index = pd.DatetimeIndex(df.index)
        df_r = df.reindex(index=dates)
        rank = df_r.rolling(20, min_periods=5).apply(
            lambda x: float(np.nan) if (x.dropna().empty or pd.isna(x.iloc[-1]))
            else float((x.dropna().iloc[:-1] <= x.dropna().iloc[-1]).sum() / max(1, len(x.dropna()) - 1)), raw=False)
        vol = df_r.rolling(20, min_periods=5).std().replace(0, np.nan)
        z = rank - 0.5
        rev = np.where(rank > 0.8, -(rank - 0.8) * 5.0,
                       np.where(rank < 0.2, (0.2 - rank) * 5.0, z.values))
        conv = pd.DataFrame(rev, index=rank.index, columns=rank.columns) / (1.0 + vol)
        return conv.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 421. intraday_roll_yield_dualscore — 展期收益双打分 (B3 复现)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_roll_yield_dualscore_20d", category="intraday_advanced")
class IntradayRollYieldDualscore20d(Factor):
    """展期收益双打分因子 (资料B3, 基于5min term).

    展期收益 (far−near)/near (正=近月贴水=多头展期正收益) 双打分:
    截面分 = 品种间展期收益排名 (0-1) + 时序分 = 自身20日分位, 各半合成.
    双高 → 展期收益强且处高位 → 正向.
    与 roll_yield (水平) 不同: 本因子是截面+时序双打分框架.
    方向: 正向.
    """
    name = "intraday_roll_yield_dualscore_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "展期收益双打分 (截面rank+时序分位各半)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_term_structure_panel(data, dates, universe, freq="5min")
        if "near_close" not in panel or "far_close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        near, far = panel["near_close"], panel["far_close"]
        day = near.index.normalize()
        roll: dict = {}
        for dt in sorted(set(day)):
            grp_n = near.loc[day == dt]
            grp_f = far.loc[day == dt]
            if len(grp_n) < 20:
                continue
            vals = {}
            for col in grp_n.columns:
                n = grp_n[col].dropna()
                f = grp_f[col].dropna()
                common = n.index.intersection(f.index)
                if len(common) < 20:
                    continue
                ry = ((f.loc[common] - n.loc[common]) / n.loc[common].replace(0, np.nan)).dropna()
                if len(ry) < 10:
                    continue
                vals[col] = float(ry.mean())
            if vals:
                roll[dt] = pd.Series(vals)
        if not roll:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(roll).T
        df.index = pd.DatetimeIndex(df.index)
        df_r = df.reindex(index=dates)
        cross_rank = df_r.rank(axis=1, pct=True)
        time_rank = df_r.rolling(20, min_periods=5).apply(
            lambda x: float(np.nan) if (x.dropna().empty or pd.isna(x.iloc[-1]))
            else float((x.dropna().iloc[:-1] <= x.dropna().iloc[-1]).sum() / max(1, len(x.dropna()) - 1)), raw=False)
        score = 0.5 * cross_rank + 0.5 * time_rank
        return score.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 422. intraday_roll_dualscore_consistency — 双打分一致性 (原创改进)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_roll_dualscore_consistency_20d", category="intraday_advanced")
class IntradayRollDualscoreConsistency20d(Factor):
    """双打分一致性因子 (原创: B3 改进).

    B3 平均两个打分, 本因子强调两分同向时的信念:
    截面 rank × 时序 rank — 两者都高(或都低)时才产生强信号, 冲突时衰减.
    高一致性 → 展期收益的截面优势与自身历史优势共振 → 更可靠 → 正向.
    方向: 正向.
    """
    name = "intraday_roll_dualscore_consistency_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "双打分一致性 (截面×时序, 共振才强信号)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_term_structure_panel(data, dates, universe, freq="5min")
        if "near_close" not in panel or "far_close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        near, far = panel["near_close"], panel["far_close"]
        day = near.index.normalize()
        roll: dict = {}
        for dt in sorted(set(day)):
            grp_n = near.loc[day == dt]
            grp_f = far.loc[day == dt]
            if len(grp_n) < 20:
                continue
            vals = {}
            for col in grp_n.columns:
                n = grp_n[col].dropna()
                f = grp_f[col].dropna()
                common = n.index.intersection(f.index)
                if len(common) < 20:
                    continue
                ry = ((f.loc[common] - n.loc[common]) / n.loc[common].replace(0, np.nan)).dropna()
                if len(ry) < 10:
                    continue
                vals[col] = float(ry.mean())
            if vals:
                roll[dt] = pd.Series(vals)
        if not roll:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(roll).T
        df.index = pd.DatetimeIndex(df.index)
        df_r = df.reindex(index=dates)
        cross_rank = df_r.rank(axis=1, pct=True)
        time_rank = df_r.rolling(20, min_periods=5).apply(
            lambda x: float(np.nan) if (x.dropna().empty or pd.isna(x.iloc[-1]))
            else float((x.dropna().iloc[:-1] <= x.dropna().iloc[-1]).sum() / max(1, len(x.dropna()) - 1)), raw=False)
        # 一致性: 同向增强 (rank乘积, 0-1), 中心化到 [-0.5, 0.5]
        score = cross_rank * time_rank - 0.25
        return (2.0 * score).reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 423. intraday_cross_contract_spread_z — 板块内价比时序 z (B5 变体)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_cross_contract_spread_z_20d", category="intraday_advanced")
class IntradayCrossContractSpreadZ20d(Factor):
    """板块内价比时序 z 因子 (资料B5 变体, 基于5min).

    品种价格 / 板块其他品种平均价的比值, 取 20 日时序 z (配对价差的自身历史偏离).
    比值 z 高 → 该品种相对板块超涨 → 均值回归 → 负向.
    与 #376 (log价格截面偏离) 不同: 本因子是比值自身的时序偏离 (配对回归视角).
    方向: 负向.
    """
    name = "intraday_cross_contract_spread_z_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "板块价比时序z (品种/板块均值, 超涨回归=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="5min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        ratios: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 20:
                continue
            sub = grp.dropna(how="all")
            if sub.shape[1] < 2:
                continue
            sec_mean = _sector_mean(sub)
            vals = {}
            for col in sub.columns:
                c = sub[col].dropna()
                m = sec_mean[col]
                common = c.index.intersection(m.index)
                if len(common) < 20:
                    continue
                ratio = (c.loc[common] / m.loc[common].replace(0, np.nan)).dropna()
                if len(ratio) < 10:
                    continue
                vals[col] = float(ratio.mean())
            if vals:
                ratios[dt] = pd.Series(vals)
        if not ratios:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(ratios).T
        df.index = pd.DatetimeIndex(df.index)
        df_r = df.reindex(index=dates)
        mean = df_r.rolling(20, min_periods=5).mean()
        std = df_r.rolling(20, min_periods=5).std().replace(0, np.nan)
        z = (df_r - mean) / std
        return (-z).reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 424. intraday_cross_ratio_reversion_speed — 价比回归速度 (原创改进)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_cross_ratio_reversion_speed_20d", category="intraday_advanced")
class IntradayCrossRatioReversionSpeed20d(Factor):
    """价比回归速度因子 (原创: B5 改进).

    B5 只识别偏离, 本因子度量板块价比偏离后的回归速度:
    计算比值 5 日变化与 20 日偏离的比值 — 偏离大且快速回归 → 均值回归动能强.
    速度 = −(5日Δ比值) × sign(20日偏离) (正=朝均值回归).
    高回归速度 → 配对价差修复进行中 → 顺势 → 正向.
    方向: 正向.
    """
    name = "intraday_cross_ratio_reversion_speed_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "价比回归速度 (5日Δ×sign偏离, 回归动能=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="5min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        day = close.index.normalize()
        ratios: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 20:
                continue
            sub = grp.dropna(how="all")
            if sub.shape[1] < 2:
                continue
            sec_mean = _sector_mean(sub)
            vals = {}
            for col in sub.columns:
                c = sub[col].dropna()
                m = sec_mean[col]
                common = c.index.intersection(m.index)
                if len(common) < 20:
                    continue
                ratio = (c.loc[common] / m.loc[common].replace(0, np.nan)).dropna()
                if len(ratio) < 10:
                    continue
                vals[col] = float(ratio.mean())
            if vals:
                ratios[dt] = pd.Series(vals)
        if not ratios:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(ratios).T
        df.index = pd.DatetimeIndex(df.index)
        df_r = df.reindex(index=dates)
        mean20 = df_r.rolling(20, min_periods=5).mean()
        deviation = df_r - mean20
        chg5 = df_r.diff(5)
        # 朝均值回归: chg5 与 deviation 反向
        speed = -chg5 * np.sign(deviation)
        return speed.rolling(20, min_periods=5).mean().reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 425. intraday_sector_rotation_strength — 板块轮动强度 (B6 复现)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_sector_rotation_strength_20d", category="intraday_advanced")
class IntradaySectorRotationStrength20d(Factor):
    """板块轮动强度因子 (资料B6, 基于5min).

    板块内分化度 (品种日内收益std) × 轮动速度 (板块内排名变化幅度):
    高分化 + 快速轮动 → 结构性行情 → 板块内相对因子有效 → 正向.
    与 #374 (分化绝对z, 负向) / #377 (排名变化, 负向) 不同: 本因子是两者的乘积交互.
    方向: 正向.
    """
    name = "intraday_sector_rotation_strength_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "板块轮动强度 (分化度×轮动速度, 结构性行情=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="5min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        ret_1m = close.pct_change()
        day = ret_1m.index.normalize()
        strengths: dict = {}
        prev_rank: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 20:
                continue
            sub = grp.dropna(how="all")
            if sub.shape[1] < 2:
                continue
            day_ret = sub.iloc[-1]
            disp = float(day_ret.std(ddof=0))
            # 轮动速度: 今日排名 vs 昨日排名 的平均变化
            rank_today = day_ret.rank(pct=True)
            prev = prev_rank.get("_all")
            if prev is not None:
                move = float((rank_today - prev).abs().mean())
            else:
                move = 0.0
            prev_rank["_all"] = rank_today
            vals = {}
            for col in sub.columns:
                vals[col] = disp * move
            if vals:
                strengths[dt] = pd.Series(vals)
        if not strengths:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(strengths).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 426. intraday_sector_rotation_momentum — 板块轮动方向动量 (原创改进)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_sector_rotation_momentum_20d", category="intraday_advanced")
class IntradaySectorRotationMomentum20d(Factor):
    """板块轮动方向动量因子 (原创: B6 改进).

    B6 度量轮动强度, 本因子度量轮动方向的持续性:
    连续两日领涨/领跌品种是否一致 — 轮动方向延续 → 板块内相对强度可迁移 → 正向.
    轮动方向反转 (昨日领涨今日领跌) → 板块无稳定结构 → 负向.
    方向: 正向.
    """
    name = "intraday_sector_rotation_momentum_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "板块轮动动量 (领涨品种连续一致, 轮动延续=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_minute_panel(data, dates, universe, freq="5min")
        if "close" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        close = panel["close"]
        ret_1m = close.pct_change()
        day = ret_1m.index.normalize()
        momentum: dict = {}
        prev_ret: dict = {}
        for dt in sorted(set(day)):
            grp = ret_1m.loc[day == dt]
            if len(grp) < 20:
                continue
            sub = grp.dropna(how="all")
            if sub.shape[1] < 2:
                continue
            day_ret = sub.iloc[-1]
            vals = {}
            for col in sub.columns:
                r_now = float(day_ret[col]) if col in day_ret.index and not np.isnan(day_ret[col]) else 0.0
                r_prev = prev_ret.get(col, 0.0)
                # 轮动动量: 今日与昨日的板块内相对位置一致性
                vals[col] = np.sign(r_now) * np.sign(r_prev) if abs(r_prev) > 1e-12 else 0.0
            prev_ret = {col: (float(day_ret[col]) if col in day_ret.index and not np.isnan(day_ret[col]) else 0.0)
                        for col in sub.columns}
            if vals:
                momentum[dt] = pd.Series(vals)
        if not momentum:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(momentum).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)


# ═══════════════════════════════════════════════════════════════════════════════
# 427. intraday_seat_net_position_margin — 席位净持仓边际 (C4 复现)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_seat_net_position_margin_20d", category="intraday_advanced")
class IntradaySeatNetPositionMargin20d(Factor):
    """席位净持仓边际因子 (资料C4, 基于日度席位).

    席位净持仓 (net_position) 的 5 日边际变化 / 总持仓 — 机构资金的边际动向.
    净多边际为正 → 机构资金正在持续加多 → 正向.
    与 #282 (1日净变化) 不同: 本因子是5日边际且按总持仓归一 (跨品种可比).
    方向: 正向.
    """
    name = "intraday_seat_net_position_margin_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "席位净持仓边际 (5日Δnet/总持仓, 机构加多=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_seat_panel(data, dates, universe)
        if not {"net_position", "total_long", "total_short"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        net = panel["net_position"]
        denom = panel["total_long"] + panel["total_short"]
        margin = net.diff(5) / denom.replace(0, np.nan)
        return margin.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 428. intraday_seat_margin_consensus — 席位边际多空共识 (原创改进)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_seat_margin_consensus_20d", category="intraday_advanced")
class IntradaySeatMarginConsensus20d(Factor):
    """席位边际多空共识因子 (原创: C4 改进).

    将 C4 边际与多空比方向结合: 边际为正 且 多空比(净多头格局) 同向 → 共识强.
    score = sign(净持仓边际) × 多空比方向一致性 (同为净多=1, 冲突=0).
    相比 C4 只看边际水平, 本因子要求"边际方向"与"存量格局"共振.
    方向: 正向.
    """
    name = "intraday_seat_margin_consensus_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "席位边际共识 (边际方向×净多格局, 共振=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_seat_panel(data, dates, universe)
        if not {"net_position", "total_long", "total_short"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        net = panel["net_position"]
        denom = panel["total_long"] + panel["total_short"]
        margin = net.diff(5)
        net_strength = net / denom.replace(0, np.nan)
        # 边际与存量同向
        consensus = np.sign(margin) * np.sign(net_strength)
        consensus = consensus.where(margin.abs() > 1e-12)
        return consensus.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 429. intraday_seat_concentration — 席位持仓 HHI 集中度 (C5 复现)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_seat_concentration_20d", category="intraday_advanced")
class IntradaySeatConcentration20d(Factor):
    """席位持仓 HHI 集中度因子 (资料C5, 基于日度席位).

    席位净持仓的 Herfindahl 指数 Σ(seat_net / total_net)² (净持仓在席位间的集中度).
    集中度高 → 少数席位主导方向 → 拥挤交易 → 反转风险 → 负向.
    与 #321/322 (top5多头/空头占比) 不同: 本因子用全席位净持仓的 HHI 浓度.
    方向: 负向.
    """
    name = "intraday_seat_concentration_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "席位持仓HHI (净持仓集中度, 拥挤=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        df = _get_seat_table(data, dates, universe, "product_seat")
        if df is None or "net_position" not in df.columns:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        data = df[["_ts", "_root", "net_position"]].dropna()
        if data.empty:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        # 每 (ts, root) 的 HHI
        share = data.groupby(["_ts", "_root"])["net_position"].apply(
            lambda x: float(((x / max(float(x.sum()), 1e-12)) ** 2).sum()))
        share = share.reset_index(name="hhi")
        pivot = share.pivot(index="_ts", columns="_root", values="hhi")
        pivot.index = pd.DatetimeIndex(pivot.index)
        return (-pivot).reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 430. intraday_seat_concentration_change — 集中度变化 (原创改进)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_seat_concentration_change_20d", category="intraday_advanced")
class IntradaySeatConcentrationChange20d(Factor):
    """席位集中度变化因子 (原创: C5 改进).

    C5 看集中度水平, 本因子看集中度的跨日变化:
    集中度快速上升 → 席位正在向少数机构收敛 → 拥挤度加剧 → 反转风险提前预警.
    集中度下降 → 分歧扩散 → 拥挤缓解.
    方向: 负向.
    """
    name = "intraday_seat_concentration_change_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "席位集中度变化 (ΔHHI, 拥挤加剧=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        df = _get_seat_table(data, dates, universe, "product_seat")
        if df is None or "net_position" not in df.columns:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        data = df[["_ts", "_root", "net_position"]].dropna()
        if data.empty:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        share = data.groupby(["_ts", "_root"])["net_position"].apply(
            lambda x: float(((x / max(float(x.sum()), 1e-12)) ** 2).sum()))
        share = share.reset_index(name="hhi")
        pivot = share.pivot(index="_ts", columns="_root", values="hhi")
        pivot.index = pd.DatetimeIndex(pivot.index)
        change = pivot.diff()
        return (-change).reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 431. intraday_flow_price_divergence — 席位资金流-价格背离 (C6 复现)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_flow_price_divergence_20d", category="intraday_advanced")
class IntradayFlowPriceDivergence20d(Factor):
    """席位资金流-价格背离因子 (资料C6, 基于日度席位).

    席位净持仓边际的 20 日 z − 价格收益的 20 日 z (资金流与价格背离度).
    正背离 = 机构净流入但价格未同步上涨 → 逆势吸筹 → 补涨 → 正向.
    负背离 = 价格涨但机构资金流出 → 涨势不实 → 负向.
    与 #418 (日内 tick 流) 不同: 本因子用日度席位机构资金.
    方向: 正向.
    """
    name = "intraday_flow_price_divergence_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "席位资金价格背离 (净持仓z-收益z, 逆势吸筹=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_seat_panel(data, dates, universe)
        close = data.get("close", dates, universe)
        if not {"net_position", "total_long", "total_short"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        net = panel["net_position"]
        denom = panel["total_long"] + panel["total_short"]
        flow = (net / denom.replace(0, np.nan)).reindex(index=pd.DatetimeIndex(dates))
        if close is None or close.empty:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret = close.pct_change().reindex(index=pd.DatetimeIndex(dates))
        f_mean = flow.rolling(20, min_periods=5).mean()
        f_std = flow.rolling(20, min_periods=5).std().replace(0, np.nan)
        r_mean = ret.rolling(20, min_periods=5).mean()
        r_std = ret.rolling(20, min_periods=5).std().replace(0, np.nan)
        div = (flow - f_mean) / f_std - (ret - r_mean) / r_std
        return div.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 432. intraday_flow_price_catchup — 资金补涨/补跌 (原创改进)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_flow_price_catchup_20d", category="intraday_advanced")
class IntradayFlowPriceCatchup20d(Factor):
    """资金补涨/补跌因子 (原创: C6 改进).

    C6 只识别背离, 本因子捕捉背离后的兑现:
    score = 背离值 × 价格动量 (资金已流入+价格开始启动 → 补涨兑现).
    正背离且价格转涨 → 吸筹兑现 → 正向; 负背离且价格转跌 → 出货兑现 → 负向.
    相比 C6 (静态背离), 本因子要求背离开始修复.
    方向: 正向.
    """
    name = "intraday_flow_price_catchup_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "资金补涨补跌 (背离×价格动量, 兑现=正向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_seat_panel(data, dates, universe)
        close = data.get("close", dates, universe)
        if not {"net_position", "total_long", "total_short"}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        net = panel["net_position"]
        denom = panel["total_long"] + panel["total_short"]
        flow = (net / denom.replace(0, np.nan)).reindex(index=pd.DatetimeIndex(dates))
        if close is None or close.empty:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        ret = close.pct_change().reindex(index=pd.DatetimeIndex(dates))
        f_mean = flow.rolling(20, min_periods=5).mean()
        f_std = flow.rolling(20, min_periods=5).std().replace(0, np.nan)
        r_mean = ret.rolling(20, min_periods=5).mean()
        r_std = ret.rolling(20, min_periods=5).std().replace(0, np.nan)
        div = (flow - f_mean) / f_std - (ret - r_mean) / r_std
        # 5日价格动量
        mom5 = ret.rolling(5, min_periods=3).mean()
        catchup = div * np.sign(mom5)
        return catchup.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 433. intraday_annualized_basis — 真年化基差率 (A阶段面板语义化)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_annualized_basis_20d", category="intraday_advanced")
class IntradayAnnualizedBasis20d(Factor):
    """真年化基差率因子 (基于修复后的期限结构面板, 资料B1).

    年化基差率 = (far - near) / near * 365 / remaining_days, 其中 near/far 为
    按到期月最近两月 (排除合成合约), remaining_days 为 near 剩余自然日.
    正=contango (远高近低), 负=backwardation (近高远低).
    取 20 日分位, 中间区线性、极值区(>80%/<20%)反转:
    backwardation 深(近月大幅升水) → 现货偏紧 → 正向; 但处历史极值 → 反转.
    与 #419 annualized_basis_z 区别: #419 用近远月价差率近似 (当时无到期日数据),
    本因子用真实年化口径 (含剩余天数), 输入面板为修复后的近/次月合约.
    方向: 随结构 (极值反转).
    """
    name = "intraday_annualized_basis_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "真年化基差率20日分位极值反转 (年化=价差率×365/剩余天数)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_term_structure_panel(data, dates, universe, freq="5min")
        if "annualized_basis" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        basis = panel["annualized_basis"]
        day = basis.index.normalize()
        daily: dict = {}
        for dt in sorted(set(day)):
            grp = basis.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                b = grp[col].dropna()
                if len(b) >= 10:
                    vals[col] = float(b.mean())
            if vals:
                daily[dt] = pd.Series(vals)
        if not daily:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily).T
        df.index = pd.DatetimeIndex(df.index)
        df = df.reindex(index=dates)
        # 滚动分位: 窗口内先 dropna; 有效值不足或末值为 NaN 时返回 NaN (防伪信号)
        rank = df.rolling(20, min_periods=5).apply(
            lambda x: float(np.nan) if (x.dropna().empty or pd.isna(x.iloc[-1]))
            else float((x.dropna().iloc[:-1] <= x.dropna().iloc[-1]).sum() / max(1, len(x.dropna()) - 1)),
            raw=False)
        z = rank - 0.5
        rev = np.where(rank > 0.8, -(rank - 0.8) * 5.0,
                       np.where(rank < 0.2, (0.2 - rank) * 5.0, z.values))
        out = pd.DataFrame(rev, index=rank.index, columns=rank.columns)
        return out.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 434. intraday_basis_momentum — 年化基差率动量 (A阶段面板语义化)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_basis_momentum_20d", category="intraday_advanced")
class IntradayBasisMomentum20d(Factor):
    """年化基差率动量因子.

    年化基差率的 20 日滚动均值 (与 #225 slope_change 的一阶差区分: 本因子用
    真年化口径 + 动量视角, 反映期限结构整体松紧趋势):
    基差率上行 (contango 加深或 backwardation 收窄) → 结构转松 → 偏空近月 → 负向.
    方向: 负向.
    """
    name = "intraday_basis_momentum_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "年化基差率20日动量 (contango加深=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_term_structure_panel(data, dates, universe, freq="5min")
        if "annualized_basis" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        basis = panel["annualized_basis"]
        day = basis.index.normalize()
        daily: dict = {}
        for dt in sorted(set(day)):
            grp = basis.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                b = grp[col].dropna()
                if len(b) >= 10:
                    vals[col] = float(b.mean())
            if vals:
                daily[dt] = pd.Series(vals)
        if not daily:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily).T
        df.index = pd.DatetimeIndex(df.index)
        mom = -df.rolling(20, min_periods=5).mean()
        return mom.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 435. intraday_rollover_frequency — 换月频率 (A阶段面板语义化)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_rollover_frequency_20d", category="intraday_advanced")
class IntradayRolloverFrequency20d(Factor):
    """换月频率因子 (基于修复后面板的 rollover_flag).

    rollover_flag 标记换月点及其±1根K线. 本因子统计 20 日窗口内换月标记的比例:
    换月频繁 → 主力合约切换密集 → 期限结构/流动性处于切换期, 结构信号不稳定 → 负向.
    可作为其他期限结构因子的 regime 门控 (高换月频率时降权).
    方向: 负向.
    """
    name = "intraday_rollover_frequency_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "20日窗口换月标记比例 (换月频繁=结构不稳定=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        panel = _get_term_structure_panel(data, dates, universe, freq="5min")
        if "rollover_flag" not in panel:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        flag = panel["rollover_flag"]
        day = flag.index.normalize()
        daily: dict = {}
        for dt in sorted(set(day)):
            grp = flag.loc[day == dt]
            if len(grp) < 20:
                continue
            vals = {}
            for col in grp.columns:
                f = grp[col].dropna()
                if len(f) >= 10:
                    vals[col] = float(f.mean())
            if vals:
                daily[dt] = pd.Series(vals)
        if not daily:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        df = pd.DataFrame(daily).T
        df.index = pd.DatetimeIndex(df.index)
        freq = df.rolling(20, min_periods=5).mean()
        out = -freq
        return out.reindex(index=pd.DatetimeIndex(dates), columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# B 阶段: 换月事件工程 (436-438) — 基于换月日历/复权 settle/结构跳变
# ═══════════════════════════════════════════════════════════════════════════════

def _daily_mean_panel(panel_df, dates, min_bars=20, min_vals=10):
    """5min 面板按交易日求均值 → 日度面板 (B 阶段共享工具, 新增)."""
    day = panel_df.index.normalize()
    daily: dict = {}
    for dt in sorted(set(day)):
        grp = panel_df.loc[day == dt]
        if len(grp) < min_bars:
            continue
        vals = {}
        for col in grp.columns:
            b = grp[col].dropna()
            if len(b) >= min_vals:
                vals[col] = float(b.mean())
        if vals:
            daily[dt] = pd.Series(vals)
    df = pd.DataFrame(daily).T
    df.index = pd.DatetimeIndex(df.index)
    return df.reindex(index=pd.DatetimeIndex(dates))


def _roll_window_flags(dates, universe, cal, half_window=1):
    """换月窗口标记: 每个换月日±half_window 个交易日记为 1, 其余 0.

    Returns DataFrame(dates×universe, float). B 阶段新增.
    """
    idx = pd.DatetimeIndex(dates)
    win = pd.DataFrame(0.0, index=idx, columns=universe)
    for root in universe:
        if root not in win.columns:
            continue
        for rd in cal.get(root, []):
            rd = pd.Timestamp(rd)
            if rd < idx.min() or rd > idx.max():  # 研究期外的换月日不参与窗口标记
                continue
            pos = idx.get_indexer([rd], method="nearest")
            i = int(pos[0])
            colpos = win.columns.get_loc(root)
            for j in range(max(0, i - half_window), min(len(idx), i + half_window + 1)):
                win.iloc[j, colpos] = 1.0
    return win


# ═══════════════════════════════════════════════════════════════════════════════
# 436. intraday_days_to_rollover — 换月年龄(自最近换月以来的交易日数) (换月日历事件因子)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_days_to_rollover_20d", category="intraday_advanced")
class IntradayDaysToRollover20d(Factor):
    """换月年龄因子 (自最近一次换月以来的交易日数, 基于换月日历).

    每 (品种, 交易日) 计算自最近一次主力切换日以来的交易日数 (换月当日=0, 之后递增):
    换月年龄小 = 刚换月 = 主力切换期, 价差/基差噪声大、信号易失真;
    换月年龄大 = 远离换月 = 期限结构处于稳定期, 结构信号可靠.
    20 日滚动均值平滑 (平均换月年龄). 可作为期限结构因子的 regime 状态.
    方向: 正向 (换月后时间越长越稳定, 越偏好持有结构信号).
    """
    name = "intraday_days_to_rollover_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "自最近换月以来的交易日数20日均值 (刚换月=结构不稳定)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        idx = pd.DatetimeIndex(dates)
        cal = _get_rollover_calendar(idx, universe)
        out = pd.DataFrame(np.nan, index=idx, columns=universe)
        for root in universe:
            roll_days = sorted(cal.get(root, []))
            if not roll_days:
                continue
            roll_arr = np.array([pd.Timestamp(d) for d in roll_days])
            for i, dt in enumerate(idx):
                # 最近一次已发生的换月(可能早于研究期): 换月年龄 = 交易日位置差 (换月日=0, 次日=1)
                past = roll_arr[roll_arr <= dt]
                if len(past) == 0:
                    continue
                last_pos = int(np.searchsorted(idx.values, past[-1], side="right")) - 1
                out.iloc[i, out.columns.get_loc(root)] = float(i - last_pos)
        return out.rolling(20, min_periods=5).mean().reindex(index=idx, columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 437. intraday_rollover_settle_gap — 换月结算价跳变强度 (换月事件工程)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_rollover_settle_gap_20d", category="intraday_advanced")
class IntradayRolloverSettleGap20d(Factor):
    """换月结算价跳变强度因子.

    settle 面板在换月日已置 NaN (修复), 换月前后跨合约的结算跳变表现为
    换月窗口(±1交易日)内 |Δsettle| 的缺口. 本因子统计 20 日窗口内该缺口之和:
    高 = 近期换月伴随剧烈结算价跳变 = 主力切换期噪声大, 日度价格/收益信号不可靠.
    方向: 负向 (换月结算跳变剧烈 → 风险高 → 回避/低配).
    """
    name = "intraday_rollover_settle_gap_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "20日窗口换月结算价跳变缺口之和 (高=换月剧烈=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        idx = pd.DatetimeIndex(dates)
        settle = _read_local_daily(data, idx, universe, "settle")
        cal = _get_rollover_calendar(idx, universe)
        win = _roll_window_flags(idx, universe, cal, half_window=1)
        # 换月日 settle 已置 NaN, diff 在窗口内会断; 用 3 日滚动最值差捕捉跨合约跳变
        # (rolling max-min 对含 NaN 窗口与 nanmax-nanmin 行为一致, 免逐元素 lambda)
        spread = settle.rolling(3, min_periods=1).max() - settle.rolling(3, min_periods=1).min()
        gap = spread.where(win > 0)
        # 换月稀疏(每月1-2次), min_periods=1: 窗口内出现换月跳变即计入
        return (-gap.rolling(20, min_periods=1).sum()).reindex(index=idx, columns=universe).shift(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 438. intraday_rollover_basis_gap — 换月基差结构跳变 (换月事件工程)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("intraday_rollover_basis_gap_20d", category="intraday_advanced")
class IntradayRolloverBasisGap20d(Factor):
    """换月基差结构跳变强度因子.

    annualized_basis 日度均值在换月窗口(±1交易日)内的 |Δ| 缺口, 20 日求和:
    高 = 换月期间期限结构发生剧烈跳变 = 基差信号在切换期不可靠, 且换月后
    结构需要时间重新稳定.
    与 #437 区别: #437 用结算价跳变 (价格层面), 本因子用年化基差率跳变 (结构层面).
    方向: 负向 (换月期结构跳变剧烈 → 结构信号不可靠 → 回避).
    """
    name = "intraday_rollover_basis_gap_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "20日窗口换月期年化基差率跳变缺口之和 (高=结构不稳=负向)"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
        idx = pd.DatetimeIndex(dates)
        panel = _get_term_structure_panel(data, idx, universe, freq="5min")
        if "annualized_basis" not in panel:
            return pd.DataFrame(np.nan, index=idx, columns=universe)
        basis_d = _daily_mean_panel(panel["annualized_basis"], idx)
        cal = _get_rollover_calendar(idx, universe)
        win = _roll_window_flags(idx, universe, cal, half_window=1)
        # 换月稀疏, min_periods=1: 窗口内出现基差跳变即计入
        gap = basis_d.diff().abs().where(win > 0)
        return (-gap.rolling(20, min_periods=1).sum()).reindex(index=idx, columns=universe).shift(1)
