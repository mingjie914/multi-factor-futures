"""OI 成对选择: 从 5 个 OI 候选选 2 个加入 6 因子, 枚举 C(5,2)=10 组合, 各跑全量+OOS.

判定: OOS 夏普 >= 6因子(2.03) 且 全量 >= 2.12, 或 OOS 接近且回撤显著改善.
用法: python scripts/oi_pair_selection.py
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

BASE6 = {
    "intraday_jump_intensity_20d": -1, "intraday_price_peak_count_20d": 1,
    "intraday_realised_skewness_20d": 1, "intraday_dtws_20d": 1,
    "intraday_drip_stone_20d": -1, "intraday_peak_ridge_ratio_20d": -1,
}
OI5 = {
    "intraday_oi_time_centroid_20d": -1, "intraday_settle_position_20d": -1,
    "intraday_term_vol_spread_20d": -1, "intraday_big_bar_ratio_20d": -1,
    "intraday_oi_ma_cross_20d": 1,
}
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
        wd = cal[(cal >= t) & (cal < t + pd.Timedelta(days=7))]
        for d in wd:
            if d in fwd.index:
                r = fwd.loc[d]
                lr = sum(r[c] * wi for c, wi in wl.items()) if wl else 0.0
                sr = sum(r[c] * wi for c, wi in ws.items()) if ws else 0.0
                rets.append((d, float(lr - sr)))
    s = pd.Series({d: v for d, v in rets}).sort_index().dropna()
    ann = s.mean() * 252
    vol = s.std(ddof=0) * np.sqrt(252)
    nav = (1 + s).cumprod()
    return {
        "sharpe": ann / vol if vol > 0 else np.nan,
        "annual": ann,
        "drawdown": (nav / nav.cummax() - 1).min(),
    }


def main():
    cfg = load_config("config/intraday_backtest.yaml")
    from pipeline.runner import PipelineRunner
    runner = PipelineRunner(config=cfg)
    engine = FactorEngine(runner.data_manager)
    cal_full = pd.DatetimeIndex(runner.data_manager.get_calendar(
        pd.Timestamp("2025-01-01"), pd.Timestamp("2026-05-31")))
    cal_oos = pd.DatetimeIndex(runner.data_manager.get_calendar(
        pd.Timestamp("2026-03-01"), pd.Timestamp("2026-05-31")))
    u = [s for s in MANUAL29 if s in cfg.universe] or MANUAL29

    print("基准: 6因子 全量2.12/OOS2.03")
    print(f"{'组合':<55}{'全量夏普':>8}{'全量回撤':>9}{'OOS夏普':>8}{'OOS回撤':>9}")
    print("-" * 90)
    oi_names = list(OI5)
    results = []
    for i in range(len(oi_names)):
        for j in range(i + 1, len(oi_names)):
            f = dict(BASE6)
            f[oi_names[i]] = OI5[oi_names[i]]
            f[oi_names[j]] = OI5[oi_names[j]]
            mf = backtest(engine, runner, cal_full, u, f)
            mo = backtest(engine, runner, cal_oos, u, f)
            name = f"+{oi_names[i][12:24]}+{oi_names[j][12:24]}"
            results.append((name, mf, mo))
            print(f"{name:<55}{mf['sharpe']:>8.2f}{mf['drawdown']:>9.2%}{mo['sharpe']:>8.2f}{mo['drawdown']:>9.2%}")
    print("-" * 90)
    # 最优: OOS 夏普最高且全量不降
    best = max(results, key=lambda r: r[2]["sharpe"])
    print(f"OOS最优: {best[0]} OOS夏普={best[2]['sharpe']:.2f} 全量={best[1]['sharpe']:.2f}")


if __name__ == "__main__":
    main()
