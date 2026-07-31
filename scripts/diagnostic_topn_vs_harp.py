"""打分法 TopN vs HARP 对比 (只读脚本, 不修改框架).

将 11 个有效因子的截面排名相加 (方向统一: 排名高=看多), 取综合排名
前 TopN 做多 / 后 TopN 做空, 周度调仓等权, 计算绩效并与 HARP 对比.

用法:
    python scripts/diagnostic_topn_vs_harp.py [--topn 5] [--start ...] [--end ...]
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

# 11 个有效因子 + 方向 (+1=高暴露看多, -1=高暴露看空)
FACTORS = {
    "intraday_jump_intensity_20d": -1,
    "intraday_price_peak_count_20d": 1,
    "intraday_realised_skewness_20d": 1,
    "intraday_dtws_20d": 1,
    "intraday_drip_stone_20d": -1,
    "intraday_peak_ridge_ratio_20d": -1,
    "intraday_price_volume_elasticity_20d": -1,
    "intraday_roll_spread_20d": -1,
    "intraday_parkinson_vol_ratio_20d": -1,
    "intraday_kyle_lambda_20d": -1,
    "intraday_open_close_volume_ratio_20d": -1,
}

# HARP 基准 (当前框架实际回测结果)
HARP_BENCHMARK = {
    "annual_return": 0.0797,
    "sharpe": 1.21,
    "max_drawdown": -0.0399,
    "volatility": 0.0649,
}


def metrics_from_returns(ret: pd.Series) -> dict:
    ret = ret.dropna()
    if len(ret) < 5:
        return {}
    ann = ret.mean() * 52
    vol = ret.std(ddof=0) * np.sqrt(52)
    sharpe = ann / vol if vol > 0 else np.nan
    nav = (1 + ret).cumprod()
    dd = (nav / nav.cummax() - 1).min()
    return {
        "annual_return": ann,
        "sharpe": sharpe,
        "max_drawdown": dd,
        "volatility": vol,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/intraday_backtest.yaml")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-31")
    parser.add_argument("--topn", type=int, default=5)
    parser.add_argument("--output", default="runs/topn_vs_harp")
    args = parser.parse_args()

    cfg = load_config(args.config)
    from pipeline.runner import PipelineRunner
    runner = PipelineRunner(config=cfg)
    engine = FactorEngine(runner.data_manager)
    calendar = pd.DatetimeIndex(runner.data_manager.get_calendar(
        pd.Timestamp(args.start), pd.Timestamp(args.end)))
    universe = cfg.universe

    # 计算所有因子暴露
    names = list(FACTORS)
    computed = engine.compute_factors(names, calendar.tolist(), universe,
                                      parallel=False)

    # 统一方向后截面排名 (0~1), 求和
    score = pd.DataFrame(index=calendar, columns=universe, dtype=float)
    for name, direction in FACTORS.items():
        mat = computed[name]
        rank = mat.rank(axis=1, pct=True)
        oriented = rank if direction == 1 else (1 - rank)
        score = score.add(oriented, fill_value=0)
    score = score.div(len(names))  # 归一化到 0~1

    # 周度调仓: 取每周最后一个交易日的综合得分
    close = runner.data_manager.get("close", calendar, universe)
    fwd = close.pct_change(1).shift(-1)  # 次日收益 (日频)
    rebal = score.resample("W-FRI").last()

    topn = args.topn
    portfolio_ret = []
    for t in rebal.index:
        row = rebal.loc[t].dropna()
        if len(row) < 2 * topn:
            continue
        ranked = row.rank(ascending=False)
        long_picks = ranked[ranked <= topn].index
        short_picks = ranked[ranked > len(ranked) - topn].index
        # 找到下周的交易日
        week_dates = calendar[(calendar >= t) & (calendar < t + pd.Timedelta(days=7))]
        for d in week_dates:
            if d in fwd.index:
                r = fwd.loc[d]
                long_ret = r[long_picks].mean() if len(long_picks) else 0
                short_ret = r[short_picks].mean() if len(short_picks) else 0
                portfolio_ret.append((d, long_ret - short_ret))
    if not portfolio_ret:
        print("无有效组合收益")
        return
    ret_df = pd.DataFrame(portfolio_ret, columns=["date", "ret"]).set_index("date")
    ret_series = ret_df["ret"]

    m = metrics_from_returns(ret_series)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"=== 打分法 Top{topn} vs HARP ===")
    print(f"打分法: 年化={m.get('annual_return', 0):.2%} 夏普={m.get('sharpe', 0):.2f} "
          f"回撤={m.get('max_drawdown', 0):.2%} 波动={m.get('volatility', 0):.2%}")
    print(f"HARP:   年化={HARP_BENCHMARK['annual_return']:.2%} 夏普={HARP_BENCHMARK['sharpe']:.2f} "
          f"回撤={HARP_BENCHMARK['max_drawdown']:.2%} 波动={HARP_BENCHMARK['volatility']:.2%}")

    # 结论 (综合夏普+回撤)
    score_diff = m.get("sharpe", 0) / max(HARP_BENCHMARK["sharpe"], 1e-9)
    dd_ratio = abs(m.get("max_drawdown", 0)) / max(abs(HARP_BENCHMARK["max_drawdown"]), 1e-9)
    if score_diff >= 0.85 and dd_ratio <= 2.0:
        print(f"\n结论: 打分法夏普为 HARP 的 {score_diff:.0%} 且回撤相近({dd_ratio:.1f}x) → 打分法更优/相当, 可考虑简化框架")
    elif score_diff >= 0.85 and dd_ratio > 2.0:
        print(f"\n结论: 打分法夏普为 HARP 的 {score_diff:.0%} 但回撤达 HARP 的 {dd_ratio:.1f} 倍 → "
              f"HARP 回撤控制显著更优, 保留 HARP")
    else:
        print(f"\n结论: 打分法夏普为 HARP 的 {score_diff:.0%} → HARP 更优, 保留现状")

    # 净值图
    nav = (1 + ret_series).cumprod()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(nav.index, nav.values, label=f"打分法 Top{topn}")
    ax.set_title(f"打分法 Top{topn} 多空净值 (周度调仓)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "topn_nav.png", dpi=150)
    plt.close()
    print(f"净值图: {out / 'topn_nav.png'}")


if __name__ == "__main__":
    main()
