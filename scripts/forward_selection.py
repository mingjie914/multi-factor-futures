"""前向选择: 6 有效因子 + OI/期限候选按 t 从强到弱逐个加入, 记录夏普/回撤曲线.

用法: python scripts/forward_selection.py
输出: 每次加入的因子、夏普、回撤、年化, 找最优子集.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.config import load_config
from factors.engine import FactorEngine

# 基础 6 有效因子
BASE6 = {
    "intraday_jump_intensity_20d": -1, "intraday_price_peak_count_20d": 1,
    "intraday_realised_skewness_20d": 1, "intraday_dtws_20d": 1,
    "intraday_drip_stone_20d": -1, "intraday_peak_ridge_ratio_20d": -1,
}
# OI/期限候选按 t 从强到弱
CANDIDATES = [
    ("intraday_oi_time_centroid_20d", -1, 6.99),
    ("intraday_settle_position_20d", -1, 5.25),
    ("intraday_term_vol_spread_20d", -1, 4.89),
    ("intraday_big_bar_ratio_20d", -1, 4.36),
    ("intraday_oi_ma_cross_20d", 1, 3.33),
    ("intraday_oi_trend_20d", 1, 2.67),
    ("intraday_oi_vol_price_corr_20d", -1, 2.39),
    ("intraday_term_breakout_20d", -1, 2.23),
    ("intraday_term_oi_ratio_20d", 1, 2.16),
]
MANUAL29 = [
    "A", "AG", "AL", "AU", "CU", "FU", "HC", "I", "IC", "IF", "IH", "J", "JM",
    "M", "MA", "NI", "P", "RB", "RM", "RU", "SA", "SN", "SR", "T", "TA", "TL",
    "TS", "Y", "ZN",
]


def backtest(engine, runner, cal, universe, factors: dict) -> dict:
    names = list(factors)
    computed = engine.compute_factors(names, cal, universe, parallel=True)
    score = pd.DataFrame(index=cal, columns=universe, dtype=float)
    for n, d in factors.items():
        r = computed[n].rank(axis=1, pct=True)
        score = score.add(r if d == 1 else (1 - r), fill_value=0)
    score = score.div(len(factors))
    close = runner.data_manager.get("close", cal, universe)
    fwd = close.pct_change().shift(-1)
    vol20 = close.pct_change().rolling(20, min_periods=10).std(ddof=0)
    rebal = score.resample("W-FRI").last()
    rets = []
    for t in rebal.index:
        row = rebal.loc[t].dropna()
        if len(row) < 20:
            continue
        rk = row.rank(ascending=False)
        top = rk[rk <= 10].index
        bot = rk[rk > len(rk) - 10].index
        vt = vol20.loc[t] if t in vol20.index else vol20.asof(t)
        def rp(pool):
            v = vt[pool].replace(0, np.nan).dropna()
            if v.empty:
                return None
            w = 1.0 / v
            return (w / w.sum()).to_dict()
        wl, ws = rp(top), rp(bot)
        w = pd.Series(0.0, index=universe)
        if wl:
            for k, v in wl.items():
                w[k] += v
        if ws:
            for k, v in ws.items():
                w[k] -= v
        wd = cal[(cal >= t) & (cal < t + pd.Timedelta(days=7))]
        for d in wd:
            if d in fwd.index:
                r = fwd.loc[d]
                long_ret = sum(r[c] * wi for c, wi in wl.items()) if wl else 0.0
                short_ret = sum(r[c] * wi for c, wi in ws.items()) if ws else 0.0
                rets.append((d, float(long_ret - short_ret)))
    s = pd.Series({d: v for d, v in rets}).sort_index()
    ann = s.mean() * 252
    vol = s.std(ddof=0) * np.sqrt(252)
    nav = (1 + s).cumprod()
    return {
        "sharpe": ann / vol if vol > 0 else np.nan,
        "annual": ann,
        "drawdown": (nav / nav.cummax() - 1).min(),
        "volatility": vol,
    }


def main():
    cfg = load_config("config/intraday_backtest.yaml")
    from pipeline.runner import PipelineRunner
    runner = PipelineRunner(config=cfg)
    engine = FactorEngine(runner.data_manager)
    cal = pd.DatetimeIndex(runner.data_manager.get_calendar(
        pd.Timestamp("2025-01-01"), pd.Timestamp("2026-05-31")))
    u = [s for s in MANUAL29 if s in cfg.universe] or MANUAL29

    print(f"{'因子集':<70}{'夏普':>7}{'年化':>8}{'回撤':>8}{'波动':>7}")
    print("-" * 100)
    current = dict(BASE6)
    best = {"sharpe": -9, "desc": "", "n": 0}
    # 先跑纯 6 因子基线
    m = backtest(engine, runner, cal, u, current)
    print(f"6因子基线{'':<60}{m['sharpe']:>7.2f}{m['annual']:>8.2%}{m['drawdown']:>8.2%}{m['volatility']:>7.2%}")
    best = {"sharpe": m["sharpe"], "desc": "6因子", "n": 6}
    # 逐个加入
    for name, direction, t_val in CANDIDATES:
        current[name] = direction
        m = backtest(engine, runner, cal, u, current)
        n_f = len(current)
        print(f"+{name[:38]:<38} (t={t_val}){'':<10}{m['sharpe']:>7.2f}{m['annual']:>8.2%}{m['drawdown']:>8.2%}{m['volatility']:>7.2%}")
        if m["sharpe"] > best["sharpe"]:
            best = {"sharpe": m["sharpe"], "desc": f"{n_f}因子(含{name[:20]})", "n": n_f}
    print("-" * 100)
    print(f"最优: {best['desc']} 夏普={best['sharpe']:.2f}")


if __name__ == "__main__":
    main()
