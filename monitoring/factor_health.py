# -*- coding: utf-8 -*-
"""monitoring.factor_health — A: 因子健康状态机 + 4 口径指标 + 回撤-反弹周期.

参考研报 067-071(国泰海通高频因子周报)的因子运营化做法:
- 4 口径滚动(全样本/60 日/20 日/上周)IC、ICIR、HAC t、多空收益、多头超额、胜率;
- 回撤-反弹周期: 多空累计收益相对全历史峰值回撤, 创 20 日新高且维持 2 日确认反弹;
- 状态机: ACTIVE / WATCH / RETIRED(可再激活), 阈值与检验流程一致(IC=0.02, 连续 20 日).

观察模块, 只读复用外部数据(信号面板 + 日度收益), 不修改生产配置.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import monitoring.config as C


def _daily_cross_section_ic(signal: pd.Series, returns: pd.Series) -> float:
    """单日截面排名 IC；有效品种不足 10 个时返回 NaN."""
    df = pd.concat([signal, returns], axis=1).dropna()
    if len(df) < 10:
        return float("nan")
    s, r = df.iloc[:, 0].rank(), df.iloc[:, 1].rank()
    if s.std(ddof=0) == 0 or r.std(ddof=0) == 0:
        return float("nan")
    return float(np.corrcoef(s, r)[0, 1])


def daily_ic_series(signal: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """逐日截面 IC：收盘 T 信号对齐下一交易日 T+1 收益."""
    common = signal.index.intersection(returns.index)
    if len(common) == 0:
        return pd.Series(dtype=float)
    sig, ret = signal.loc[common], returns.shift(-1).loc[common]
    out = {}
    for dt in common:
        out[dt] = _daily_cross_section_ic(sig.loc[dt], ret.loc[dt])
    return pd.Series(out, dtype=float).dropna().sort_index()


def directed_score(signal: pd.DataFrame, direction: int) -> pd.DataFrame:
    """方向化分位得分: direction=1 高值高分为多头候选; -1 反号."""
    if direction not in (-1, 1):
        raise ValueError(f"factor direction must be 1 or -1, got {direction!r}")
    r = signal.rank(axis=1, pct=True)
    return r if direction == 1 else (1.0 - r)


def signal_portfolio_returns(signal: pd.DataFrame, returns: pd.DataFrame,
                             top_n: int, multi_only: bool = False) -> pd.Series:
    """信号组合下一交易日收益: 前 top_n 多头等权 - 后 top_n 空头等权.

    multi_only=True 时仅多头(做多前 top_n), 供纯多头策略/回撤口径使用.
    """
    if isinstance(top_n, bool) or int(top_n) != top_n or top_n < 1:
        raise ValueError(f"top_n must be a positive integer, got {top_n!r}")
    common = signal.index.intersection(returns.index)
    if len(common) == 0:
        return pd.Series(dtype=float)
    sig, ret = signal.loc[common], returns.shift(-1).loc[common]
    long_ret = pd.Series(np.nan, index=common, dtype=float)
    short_ret = pd.Series(np.nan, index=common, dtype=float)
    for dt in common:
        row = sig.loc[dt].dropna().sort_values(ascending=False)
        if len(row) == 0:
            continue
        # Long/short legs must be disjoint.  A small or sparse universe
        # therefore uses the largest feasible symmetric pair instead of
        # counting the same asset in both legs.
        n = min(top_n, len(row)) if multi_only else min(top_n, len(row) // 2)
        if n == 0:
            continue
        longs = row.index[:n]
        r = ret.loc[dt]
        lv = r[longs].dropna()
        if len(lv):
            long_ret.loc[dt] = float(lv.mean())
        if not multi_only:
            shorts = row.index[-n:]
            sv = r[shorts].dropna()
            if len(sv):
                short_ret.loc[dt] = float(sv.mean())
    if multi_only:
        return long_ret.dropna()
    return long_ret.sub(short_ret, fill_value=np.nan).dropna()


def cumulative_return(daily: pd.Series) -> pd.Series:
    """日度收益 → 累计净值收益(第一个有效日为基准)."""
    v = daily.dropna()
    if v.empty:
        return pd.Series(dtype=float)
    return (1.0 + v).cumprod() - 1.0


class FactorHealthMonitor:
    """因子健康状态机.

    状态: ACTIVE(正常) / WATCH(回撤预警) / RETIRED(IC 失效, 可再激活).
    判定(与检验流程一致):
      - RETIRED: 连续 IC_DAYS_CONSECUTIVE 日 滚动 60 日 IC < IC_RETIRED(0.02);
      - WATCH: 20 日窗口多空累计收益回撤 > DD_THRESHOLD(30%);
      - ACTIVE: 创 20 日新高且维持 REBOUND_CONFIRM_DAYS 日(反弹确认), 或从未触发上述.
    """

    def __init__(self, factors: dict[str, int] | None = None):
        self.factors = dict(factors) if factors is not None else dict(C.PRODUCTION_FACTORS)
        # 每因子历史: daily_ic(序列), port_ret(多空日收益), state(当前)
        self._daily_ic: dict[str, pd.Series] = {}
        self._port_ret: dict[str, pd.Series] = {}
        self._state: dict[str, str] = {f: C.STATE_ACTIVE for f in self.factors}
        self._last_date: pd.Timestamp | None = None

    # ---- 增量计算 ----
    def update(self, signals: dict[str, pd.DataFrame], returns: pd.DataFrame) -> None:
        observed_dates: set[pd.Timestamp] = set()
        for name, direction in self.factors.items():
            sig = signals.get(name)
            if sig is None or sig.empty:
                continue
            score = directed_score(sig, direction)
            current_ic = daily_ic_series(sig * direction, returns)
            current_port = signal_portfolio_returns(
                score, returns, top_n=C.SIGNAL_TOP_N)
            observed_dates.update(current_ic.dropna().index)
            observed_dates.update(current_port.dropna().index)
            self._daily_ic[name] = self._append_history(
                self._daily_ic.get(name), current_ic
            )
            self._port_ret[name] = self._append_history(
                self._port_ret.get(name), current_port
            )
            self._state[name] = self._decide_state(
                self._daily_ic[name], self._port_ret[name], self._state.get(name, C.STATE_ACTIVE))
        if observed_dates:
            self._last_date = pd.Timestamp(max(observed_dates))

    @staticmethod
    def _append_history(
        previous: pd.Series | None, current: pd.Series
    ) -> pd.Series:
        """Append a new monitoring batch, with current observations winning overlaps."""
        if previous is None or previous.empty:
            return current.sort_index()
        merged = pd.concat([previous, current])
        return merged[~merged.index.duplicated(keep="last")].sort_index()

    # ---- 状态判定 ----
    def _rolling_ic(self, ic: pd.Series) -> pd.Series:
        return ic.rolling(C.IC_WINDOWS[0], min_periods=10).mean()

    def _decide_state(self, ic: pd.Series, port_ret: pd.Series, prev: str) -> str:
        r60 = self._rolling_ic(ic)
        bad_ic = r60 < C.IC_RETIRED
        consec = self._consecutive_days(bad_ic)
        if consec >= C.IC_DAYS_CONSECUTIVE:
            return C.STATE_RETIRED
        cum = cumulative_return(port_ret)
        dd = self._drawdown(cum)
        if not dd.empty and dd.iloc[-1] < -C.DD_THRESHOLD:
            return C.STATE_WATCH
        if self._is_rebounding(cum):
            # RETIRED 因子再激活必须 IC 已回升(最近 IC_REACTIVATE_DAYS 日滚动 IC ≥ 阈值),
            # 避免仅靠收益创新高触发 RETIRED↔ACTIVE 状态抖动
            if prev == C.STATE_RETIRED:
                ic_tail = r60.dropna().tail(C.IC_REACTIVATE_DAYS)
                if ic_tail.empty or not (ic_tail >= C.IC_RETIRED).all():
                    return C.STATE_RETIRED
            return C.STATE_ACTIVE
        return prev if prev in C.STATES else C.STATE_ACTIVE

    @staticmethod
    def _consecutive_days(mask: pd.Series) -> int:
        """序列尾部连续 True 的天数."""
        cnt = 0
        for v in mask.iloc[::-1]:
            if v:
                cnt += 1
            else:
                break
        return cnt

    @staticmethod
    def _drawdown(cum: pd.Series, window: int | None = None) -> pd.Series:
        """回撤: 相对历史最高净值 (nav/nav.cummax() - 1).

        用净值(nav=1+cum)而非 cum 本身, 避免组合整体亏损时 cummax 为 0/负
        导致回撤被放大成 -1000% 级伪值(此前用 20 日滚动窗口峰值有此问题).
        window 参数保留用于接口兼容(反弹判定另行使用 REBOUND_WINDOW).
        """
        if cum.empty:
            return pd.Series(dtype=float)
        nav = 1.0 + cum
        peak_nav = nav.cummax()
        return nav / peak_nav - 1.0

    @staticmethod
    def _is_rebounding(cum: pd.Series) -> bool:
        """创 REBOUND_WINDOW 日滚动最高(相对前日窗口峰值)且维持 REBOUND_CONFIRM_DAYS 日.

        用 shift(1) 前的滚动峰值, 避免单调下跌时 rolling_max=自身被误判为创新高.
        """
        if cum.empty:
            return False
        prev_peak = cum.shift(1).rolling(C.REBOUND_WINDOW, min_periods=1).max()
        at_high = cum >= prev_peak - 1e-12
        tail = at_high.iloc[-C.REBOUND_CONFIRM_DAYS:] if len(at_high) >= C.REBOUND_CONFIRM_DAYS else at_high
        return bool(tail.all()) if len(tail) else False

    # ---- 查询 ----
    def get_state(self, factor: str) -> str:
        return self._state.get(factor, C.STATE_ACTIVE)

    def get_states(self) -> dict[str, str]:
        return dict(self._state)

    def last_date(self) -> pd.Timestamp | None:
        return self._last_date

    # ---- 快照(周报数据源) ----
    def health_snapshot(self) -> dict:
        """4 口径指标 + 回撤-反弹周期 + 状态, 落盘/供周报."""
        snap: dict = {}
        for name, direction in self.factors.items():
            ic = self._daily_ic.get(name)
            port = self._port_ret.get(name)
            if ic is None or port is None:
                continue
            snap[name] = {
                "direction": direction,
                "state": self._state.get(name, C.STATE_ACTIVE),
                "daily_ic": _series_to_list(ic),
                "port_ret": _series_to_list(port),
                "metrics": self._metrics(ic, port),
                "drawdown_cycle": self._drawdown_cycle(port),
            }
        return {"as_of": str(self._last_date.date()) if self._last_date is not None else None,
                "factors": snap}

    def _metrics(self, ic: pd.Series, port_ret: pd.Series) -> dict:
        """4 口径: 全样本/60日/20日/上周 的 IC/ICIR/多空收益/多头超额/胜率/4周动量."""
        out: dict = {}
        ic_valid = ic.dropna()
        port_valid = port_ret.dropna()
        full_ic = ic_valid
        full_port = port_valid
        win60 = ic_valid.iloc[-C.IC_WINDOWS[0]:]
        win20 = ic_valid.iloc[-C.IC_WINDOWS[1]:]
        # 上周(最近 5 个交易日)
        week = ic_valid.iloc[-5:]
        week_port = port_valid.iloc[-5:]
        for label, icw, pw in (("all", full_ic, full_port),
                               ("60d", win60, port_valid.iloc[-60:]),
                               ("20d", win20, port_valid.iloc[-20:]),
                               ("week", week, week_port)):
            if icw.empty:
                out[label] = {"n": 0}
                continue
            ic_mean = float(icw.mean())
            ic_std = float(icw.std(ddof=0))
            out[label] = {
                "n": int(len(icw)),
                "ic": ic_mean,
                "icir": ic_mean / ic_std if ic_std > 0 else 0.0,
                "win_rate": float((icw > 0).mean()),
                "port_ret": float(pw.sum()) if not pw.empty else None,
            }
        # 4 周多空收益动量(最近 20 个交易日的多空收益和)
        mom = float(port_valid.iloc[-20:].sum()) if len(port_valid) else None
        out["momentum_20d"] = mom
        return out

    def _drawdown_cycle(self, port_ret: pd.Series) -> dict:
        """回撤-反弹周期: 当前回撤深度/峰值日期/是否处于回撤, 及最近一次完整周期."""
        cum = cumulative_return(port_ret)
        if cum.empty:
            return {"in_drawdown": False, "depth": 0.0}
        dd = self._drawdown(cum)
        vals = dd.to_numpy(dtype=float)
        if np.all(np.isnan(vals)):
            return {"in_drawdown": False, "depth": 0.0,
                    "peak_date": None, "trough_date": None, "cum_ret": _series_to_list(cum)}
        trough_pos = int(np.nanargmin(vals))
        depth = float(vals[trough_pos])
        nav = (1.0 + cum).to_numpy(dtype=float)
        peak_nav_at_trough = float(np.maximum.accumulate(nav)[trough_pos])
        peak_candidates = np.flatnonzero(
            np.isclose(nav[:trough_pos + 1], peak_nav_at_trough)
        )
        peak_idx = int(peak_candidates[-1]) if len(peak_candidates) else 0
        return {
            "depth": depth,
            "peak_date": str(cum.index[peak_idx].date()) if peak_idx >= 0 else None,
            "trough_date": str(cum.index[trough_pos].date()) if trough_pos >= 0 else None,
            "in_drawdown": bool(float(vals[-1]) < -C.DD_THRESHOLD),
            "cum_ret": _series_to_list(cum),
        }


def _series_to_list(s: pd.Series) -> list[dict]:
    """序列 → [{date, value}] 列表(JSON 友好)."""
    out = []
    for dt, v in s.dropna().items():
        out.append({"date": str(dt.date()) if hasattr(dt, "date") else str(dt),
                    "value": float(v)})
    return out
