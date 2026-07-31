"""高频微观结构因子批次 (基于公开高频因子体系, 期货本地1分钟数据适配).

所有因子 category="intraday_advanced", frequency="daily" (日频信号),
依赖 _get_minute_panel 读取本地1分钟 Parquet, 按天聚合后 roll20 平滑 + shift(1) 防未来.

本批次 10 个因子 (K1~K10) 分别刻画:
  K1  realized_kurtosis      已实现峰度 (分钟收益四阶矩)
  K2  overnight_intraday_vol 隔夜/日内波动比
  K3  jump_persistence       跳跃持续性 (相邻跳跃同向比例)
  K4  intraday_amihud        日内 Amihud 非流动性 (分钟粒度)
  K5  liquidity_elasticity   流动性弹性 (冲击后价格恢复速度)
  K6  early_late_vol_asym    早盘/尾盘成交量不对称
  K7  tail_return_ratio      尾部收益比 (上下尾部均值比)
  K8  vol_clustering         波动率聚集 (日内波动自相关)
  K9  micro_price_impact     微价格冲击 (成交额加权收益)
  K10 signed_volume_pressure 买卖压力 (signed volume 代理)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.interfaces import Factor
from core.registry import register_factor
from factors.library.intraday import _get_minute_panel


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
