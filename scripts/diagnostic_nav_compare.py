"""修正B1 vs B3 净值对比图生成 (只读脚本).

- 修正B1: 9品种全部持有做多, 因子Top5超配, 其余保持基准权重 (周度)
- B3: 6因子+manual29 多空Top10/Bottom10 + 池内逆波动率 (周度)

用法:
    python scripts/diagnostic_nav_compare.py [--end 2026-05-31]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_config
from factors.engine import FactorEngine

SYM9 = ["AU", "CU", "IF", "IC", "TF", "T", "SC", "RB", "M"]
MANUAL29 = [
    "A", "AG", "AL", "AU", "CU", "FU", "HC", "I", "IC", "IF", "IH", "J", "JM",
    "M", "MA", "NI", "P", "RB", "RM", "RU", "SA", "SN", "SR", "T", "TA", "TL",
    "TS", "Y", "ZN",
]
SIX = {
    "intraday_jump_intensity_20d": -1,
    "intraday_price_peak_count_20d": 1,
    "intraday_realised_skewness_20d": 1,
    "intraday_dtws_20d": 1,
    "intraday_drip_stone_20d": -1,
    "intraday_peak_ridge_ratio_20d": -1,
}


def build_strategy(runner, engine, universe, start, end):
    """周度 W-FRI 信号回测, 返回日收益 Series."""
    cal = pd.DatetimeIndex(runner.data_manager.get_calendar(
        pd.Timestamp(start), pd.Timestamp(end)))
    computed = engine.compute_factors(list(SIX), cal.tolist(), universe, parallel=False)
    score = pd.DataFrame(index=cal, columns=universe, dtype=float)
    for n, d in SIX.items():
        r = computed[n].rank(axis=1, pct=True)
        o = r if d == 1 else (1 - r)
        score = score.add(o, fill_value=0)
    score = score.div(len(SIX))
    close = runner.data_manager.get("close", cal, universe)
    dr = close.pct_change()
    fwd = dr.shift(-1)
    vol20 = dr.rolling(20, min_periods=10).std(ddof=0)
    rebal = score.resample("W-FRI").last()
    rets = []
    for t in rebal.index:
        row = rebal.loc[t].dropna()
        if len(row) < 5:
            continue
        ranked = row.rank(ascending=False)
        top5 = ranked[ranked <= 5].index
        w = pd.Series(0.0, index=universe)
        if len(universe) == 9:
            # 修正B1: 全部持有等权, Top5双倍
            w = pd.Series(1.0 / 9, index=universe)
            w[top5] = 2.0 / 9
            w = w / w.sum()
        else:
            # B3: Top10多/Bottom10空 + 池内逆波动率
            top10 = ranked[ranked <= 10].index
            bot10 = ranked[ranked > len(ranked) - 10].index
            vt = vol20.loc[t] if t in vol20.index else vol20.asof(t)
            def rp(pool):
                v = vt[pool].replace(0, np.nan).dropna()
                if v.empty:
                    return None
                ww = 1.0 / v
                return (ww / ww.sum()).to_dict()
            wl, ws = rp(top10), rp(bot10)
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
                rets.append((d, float((r * w).sum())))
    return pd.Series({d: v for d, v in rets}).sort_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/intraday_backtest.yaml")
    parser.add_argument("--end", default="2026-05-31")
    parser.add_argument("--output", default="runs/nav_compare")
    args = parser.parse_args()

    cfg = load_config(args.config)
    from pipeline.runner import PipelineRunner
    runner = PipelineRunner(config=cfg)
    engine = FactorEngine(runner.data_manager)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    for start, tag in [("2025-01-01", "2025至今"), ("2026-01-01", "2026至今")]:
        b1 = build_strategy(runner, engine, SYM9, start, args.end)
        b3 = build_strategy(runner, engine, MANUAL29, start, args.end)
        fig, ax = plt.subplots(figsize=(12, 6))
        for series, label, color in [
            (b1, "修正B1 (9品种全持有+Top5超配)", "#e67e22"),
            (b3, "B3 (6因子+manual29多空)", "#2ecc71"),
        ]:
            nav = (1 + series).cumprod()
            ax.plot(nav.index, nav.values, label=label, color=color, linewidth=1.5)
        ax.set_title(f"修正B1 vs B3 净值 ({tag}, {start}~{args.end})")
        ax.set_ylabel("净值")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        png = out / f"nav_compare_{start}.png"
        plt.savefig(png, dpi=150)
        plt.close()

        def m(s):
            s2 = s.dropna()
            if len(s2) < 5:
                return "样本不足"
            ann = s2.mean() * 252
            vol = s2.std(ddof=0) * np.sqrt(252)
            nav = (1 + s2).cumprod()
            return f"年化{ann:.1%} 夏普{ann/vol if vol>0 else 0:.2f} 回撤{(nav/nav.cummax()-1).min():.1%}"

        print(f"[{tag}]")
        print(f"  修正B1: {m(b1)}")
        print(f"  B3:     {m(b3)}")
        print(f"  图: {png}")


if __name__ == "__main__":
    main()
