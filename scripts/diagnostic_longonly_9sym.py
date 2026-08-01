"""9品种只做多 Top5 分支探索 (只读脚本, 不修改框架).

品种: AU,CU,IF,IC,TF,T,SC,RB,M (9个)
信号: 6个已验证有效因子等权打分 → 综合得分Top5做多 (等权, 不做空)
调仓: 周度
窗口: 可指定, 默认 2025-01-01 与 2026-01-01 起

用法:
    python scripts/diagnostic_longonly_9sym.py [--start 2025-01-01] [--end 2026-05-31]
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

# 9 品种
SYMBOLS = ["AU", "CU", "IF", "IC", "TF", "T", "SC", "RB", "M"]

# 6 个已验证有效因子 + 方向 (+1=高暴露看多, -1=高暴露看空)
FACTORS = {
    "intraday_jump_intensity_20d": -1,
    "intraday_price_peak_count_20d": 1,
    "intraday_realised_skewness_20d": 1,
    "intraday_dtws_20d": 1,
    "intraday_drip_stone_20d": -1,
    "intraday_peak_ridge_ratio_20d": -1,
}


def metrics(ret: pd.Series) -> dict:
    ret = ret.dropna()
    if len(ret) < 5:
        return {}
    ann = ret.mean() * 252
    vol = ret.std(ddof=0) * np.sqrt(252)
    nav = (1 + ret).cumprod()
    return {
        "annual_return": ann,
        "sharpe": ann / vol if vol > 0 else np.nan,
        "max_drawdown": (nav / nav.cummax() - 1).min(),
        "volatility": vol,
        "total_return": nav.iloc[-1] - 1,
        "n_days": len(ret),
    }


def run(config_path, start, end, topn=5, output_dir="runs/longonly_9sym"):
    cfg = load_config(config_path)
    from pipeline.runner import PipelineRunner
    runner = PipelineRunner(config=cfg)
    engine = FactorEngine(runner.data_manager)
    calendar = pd.DatetimeIndex(runner.data_manager.get_calendar(
        pd.Timestamp(start), pd.Timestamp(end)))
    symbols = [s for s in SYMBOLS if s in cfg.universe] or SYMBOLS

    # 因子打分
    names = list(FACTORS)
    computed = engine.compute_factors(names, calendar.tolist(), symbols, parallel=False)
    score = pd.DataFrame(index=calendar, columns=symbols, dtype=float)
    for name, direction in FACTORS.items():
        rank = computed[name].rank(axis=1, pct=True)
        oriented = rank if direction == 1 else (1 - rank)
        score = score.add(oriented, fill_value=0)
    score = score.div(len(names))

    # 日收益
    close = runner.data_manager.get("close", calendar, symbols)
    daily_ret = close.pct_change()
    fwd = daily_ret.shift(-1)

    # 周度调仓: 选Top5做多
    rebal = score.resample("W-FRI").last()
    rets = []
    for t in rebal.index:
        row = rebal.loc[t].dropna()
        if len(row) < topn:
            continue
        picks = row.rank(ascending=False)[row.rank(ascending=False) <= topn].index
        week_dates = calendar[(calendar >= t) & (calendar < t + pd.Timedelta(days=7))]
        for d in week_dates:
            if d not in fwd.index:
                continue
            r = fwd.loc[d]
            rets.append((d, r[picks].mean() if len(picks) else 0))
    ret_s = pd.Series({d: v for d, v in rets}).sort_index()
    m = metrics(ret_s)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{start}_{end}"
    print(f"[9品种只做多 Top{topn}] {start}~{end} ({len(m.get('n_days', [])) if isinstance(m.get('n_days'), list) else m.get('n_days', 0)}天)")
    print(f"  年化={m.get('annual_return', 0):.2%} 夏普={m.get('sharpe', 0):.2f} "
          f"回撤={m.get('max_drawdown', 0):.2%} 波动={m.get('volatility', 0):.2%} "
          f"总收益={m.get('total_return', 0):.2%}")

    # 净值图
    nav = (1 + ret_s).cumprod()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(nav.index, nav.values, color="#2ecc71", linewidth=1.5)
    ax.set_title(f"9品种只做多 Top{topn} 净值 ({start}~{end})")
    ax.set_ylabel("净值")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    png = out / f"nav_{tag}.png"
    plt.savefig(png, dpi=150)
    plt.close()
    print(f"  净值图: {png}")
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/intraday_backtest.yaml")
    parser.add_argument("--end", default="2026-05-31")
    parser.add_argument("--topn", type=int, default=5)
    parser.add_argument("--output", default="runs/longonly_9sym")
    args = parser.parse_args()
    run(args.config, "2025-01-01", args.end, args.topn, args.output)
    run(args.config, "2026-01-01", args.end, args.topn, args.output)


if __name__ == "__main__":
    main()
