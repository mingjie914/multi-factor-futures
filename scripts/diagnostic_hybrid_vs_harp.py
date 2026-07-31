"""二合一对比: 纯HARP vs 打分选池+池内风险平价 (只读脚本).

逻辑:
  打分法: 11因子排名求和 → Top10/Bottom10 候选池 (选标的)
  风险平价: 池内按波动率倒数加权 (HARP核心思想简化) (定权重)
对比:
  纯HARP (回测引擎结果, 手工填入或读取CSV)
  纯打分法 (diagnostic_topn_vs_harp 结果)
  二合一 (本脚本)

用法:
    python scripts/diagnostic_hybrid_vs_harp.py [--topn 10] [--start ...] [--end ...]
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

# 11 个有效因子 + 方向
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

# 纯HARP基准 (61品种, 2025-01~2026-05 实际回测结果)
HARP_BENCHMARK = {
    "annual_return": 0.0328,
    "sharpe": 0.57,
    "max_drawdown": -0.0869,
    "volatility": 0.0599,
}
# 纯打分法 Top5 结果
SCORE_BENCHMARK = {
    "annual_return": 0.0950,
    "sharpe": 1.06,
    "max_drawdown": -0.1973,
    "volatility": 0.0895,
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
    return {"annual_return": ann, "sharpe": sharpe, "max_drawdown": dd, "volatility": vol}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/intraday_backtest.yaml")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-31")
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--output", default="runs/hybrid_vs_harp")
    args = parser.parse_args()

    cfg = load_config(args.config)
    from pipeline.runner import PipelineRunner
    runner = PipelineRunner(config=cfg)
    engine = FactorEngine(runner.data_manager)
    calendar = pd.DatetimeIndex(runner.data_manager.get_calendar(
        pd.Timestamp(args.start), pd.Timestamp(args.end)))
    universe = cfg.universe

    # 因子暴露
    names = list(FACTORS)
    computed = engine.compute_factors(names, calendar.tolist(), universe, parallel=False)
    score = pd.DataFrame(index=calendar, columns=universe, dtype=float)
    for name, direction in FACTORS.items():
        rank = computed[name].rank(axis=1, pct=True)
        oriented = rank if direction == 1 else (1 - rank)
        score = score.add(oriented, fill_value=0)
    score = score.div(len(names))

    # 日收益 + 波动率 (20日滚动)
    close = runner.data_manager.get("close", calendar, universe)
    daily_ret = close.pct_change()
    vol20 = daily_ret.rolling(20, min_periods=10).std(ddof=0)

    # 周度调仓
    fwd = daily_ret.shift(-1)  # 次日收益
    rebal = score.resample("W-FRI").last()
    topn = args.topn

    hybrid_ret = []
    score5_ret = []
    for t in rebal.index:
        row = rebal.loc[t].dropna()
        if len(row) < 2 * topn:
            continue
        ranked = row.rank(ascending=False)
        long_pool = ranked[ranked <= topn].index
        short_pool = ranked[ranked > len(ranked) - topn].index

        # 池内风险平价权重: 波动率倒数
        vol_t = vol20.loc[t] if t in vol20.index else vol20.asof(t)
        def rp_weights(pool):
            v = vol_t[pool].replace(0, np.nan).dropna()
            if v.empty:
                return None
            w = (1.0 / v)
            return w / w.sum()
        w_long = rp_weights(long_pool)
        w_short = rp_weights(short_pool)

        week_dates = calendar[(calendar >= t) & (calendar < t + pd.Timedelta(days=7))]
        for d in week_dates:
            if d not in fwd.index:
                continue
            r = fwd.loc[d]
            if w_long is not None:
                long_ret = (r[long_pool] * w_long).sum()
            else:
                long_ret = r[long_pool].mean() if len(long_pool) else 0
            if w_short is not None:
                short_ret = (r[short_pool] * w_short).sum()
            else:
                short_ret = r[short_pool].mean() if len(short_pool) else 0
            hybrid_ret.append((d, long_ret - short_ret))
            # 纯打分法 (等权)
            score5_ret.append((d, r[long_pool].mean() - r[short_pool].mean()))

    def to_series(rows):
        return pd.Series({d: v for d, v in rows}).sort_index()

    hybrid_s = to_series(hybrid_ret)
    score5_s = to_series(score5_ret)
    hybrid_m = metrics_from_returns(hybrid_s)
    score5_m = metrics_from_returns(score5_s)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"=== 二合一 (打分Top{topn} + 池内风险平价) vs 纯HARP vs 纯打分 ===")
    print(f"纯HARP:   年化={HARP_BENCHMARK['annual_return']:.2%} 夏普={HARP_BENCHMARK['sharpe']:.2f} "
          f"回撤={HARP_BENCHMARK['max_drawdown']:.2%} 波动={HARP_BENCHMARK['volatility']:.2%}")
    print(f"纯打分:   年化={score5_m.get('annual_return', 0):.2%} 夏普={score5_m.get('sharpe', 0):.2f} "
          f"回撤={score5_m.get('max_drawdown', 0):.2%} 波动={score5_m.get('volatility', 0):.2%}")
    print(f"二合一:   年化={hybrid_m.get('annual_return', 0):.2%} 夏普={hybrid_m.get('sharpe', 0):.2f} "
          f"回撤={hybrid_m.get('max_drawdown', 0):.2%} 波动={hybrid_m.get('volatility', 0):.2%}")

    # 汇总CSV
    rows = [
        {"strategy": "HARP", **HARP_BENCHMARK},
        {"strategy": "SCORE", **{k: float(v) for k, v in score5_m.items()}},
        {"strategy": "HYBRID", **{k: float(v) for k, v in hybrid_m.items()}},
    ]
    pd.DataFrame(rows).set_index("strategy").to_csv(out / "comparison.csv", encoding="utf-8-sig")

    # 净值对比图
    fig, ax = plt.subplots(figsize=(12, 6))
    for series, label, color in [
        (score5_s, f"纯打分 Top{topn}", "#e67e22"),
        (hybrid_s, f"二合一 Top{topn}+风险平价", "#2ecc71"),
    ]:
        nav = (1 + series).cumprod()
        ax.plot(nav.index, nav.values, label=label, color=color)
    ax.set_title("打分法 vs 二合一 多空净值")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "hybrid_nav.png", dpi=150)
    plt.close()
    print(f"净值图: {out / 'hybrid_nav.png'}")
    print(f"汇总: {out / 'comparison.csv'}")


if __name__ == "__main__":
    main()
