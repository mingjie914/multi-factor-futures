"""融合策略口径扫描 + 权重变体调优 (只读脚本, 不修改框架).

支持:
  口径 (--universe-mode):  all61 (全部) / liq45 (成交额前45) / liq29 (成交额前29)
  策略 (--strategy):       score (纯打分等权) / hybrid (打分选池+池内加权)
  权重方案 (--weight-scheme): 
      equal     纯打分等权 (score 默认)
      rp        池内风险平价
      half_rp   50%等权 + 50%风险平价
      ic        1个月滚动IC加权
      floored   风险平价 + 单品种下限0.5%
      erc       等风险贡献 (协方差 shrinkage=0.3, 框架默认, 默认权重方案)
      confirm2w 连续两周确认信号 + 等权

输出: 每组合的 年化/夏普/回撤/波动/月换手/换手超标周占比

用法:
    python scripts/diagnostic_scan.py --strategy hybrid --weight-scheme half_rp
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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


def metrics(ret: pd.Series) -> dict:
    ret = ret.dropna()
    if len(ret) < 5:
        return {}
    # 收益为日度序列, 年化用 252 (此前误用 52 周度年化, 导致夏普低估 2.2 倍)
    ann = ret.mean() * 252
    vol = ret.std(ddof=0) * np.sqrt(252)
    nav = (1 + ret).cumprod()
    return {
        "annual_return": ann,
        "sharpe": ann / vol if vol > 0 else np.nan,
        "max_drawdown": (nav / nav.cummax() - 1).min(),
        "volatility": vol,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/intraday_backtest.yaml")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-31")
    parser.add_argument("--universe-mode", default="all61",
                        choices=["all61", "liq45", "liq29", "manual29"])
    parser.add_argument("--strategy", default="score", choices=["score", "hybrid"])
    parser.add_argument("--factor-subset", default="all11",
                        choices=["all11", "six", "seven", "nine", "ten", "eleven", "fifteen"], help="all11=11因子, six=6因子, ten=6+OI最强4, fifteen=6+OI9")
    parser.add_argument("--weight-scheme", default="erc",
                        choices=["equal", "rp", "erc", "half_rp", "ic", "floored", "confirm2w"])
    parser.add_argument("--rebalance", default="D", help="调仓频率 (D=日度默认, W-FRI=周度, BM=月度)")
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--output", default="runs/scan")
    args = parser.parse_args()

    cfg = load_config(args.config)
    from pipeline.runner import PipelineRunner
    runner = PipelineRunner(config=cfg)
    engine = FactorEngine(runner.data_manager)
    calendar = pd.DatetimeIndex(runner.data_manager.get_calendar(
        pd.Timestamp(args.start), pd.Timestamp(args.end)))
    universe = list(cfg.universe)

    # 因子子集
    if args.factor_subset == "six":
        FACTORS_SUB = {k: v for k, v in FACTORS.items()
                       if k in ("intraday_jump_intensity_20d", "intraday_price_peak_count_20d",
                                "intraday_realised_skewness_20d", "intraday_dtws_20d",
                                "intraday_drip_stone_20d", "intraday_peak_ridge_ratio_20d")}
    elif args.factor_subset == "seven":
        FACTORS_SUB = {k: v for k, v in FACTORS.items()
                       if k in ("intraday_jump_intensity_20d", "intraday_price_peak_count_20d",
                                "intraday_realised_skewness_20d", "intraday_dtws_20d",
                                "intraday_drip_stone_20d", "intraday_peak_ridge_ratio_20d",
                                "intraday_vwap_crossings_20d")}
    elif args.factor_subset == "fifteen":
        FACTORS_SUB = {k: v for k, v in FACTORS.items()
                       if k in ("intraday_jump_intensity_20d", "intraday_price_peak_count_20d",
                                "intraday_realised_skewness_20d", "intraday_dtws_20d",
                                "intraday_drip_stone_20d", "intraday_peak_ridge_ratio_20d")}
        FACTORS_SUB.update({
            "intraday_oi_time_centroid_20d": -1, "intraday_settle_position_20d": -1,
            "intraday_term_vol_spread_20d": -1, "intraday_big_bar_ratio_20d": -1,
            "intraday_oi_ma_cross_20d": 1, "intraday_oi_trend_20d": 1,
            "intraday_oi_vol_price_corr_20d": -1, "intraday_term_breakout_20d": -1,
            "intraday_term_oi_ratio_20d": 1,
        })
    else:
        FACTORS_SUB = FACTORS

    # 口径筛选: 按日均成交额排序
    if args.universe_mode == "manual29":
        universe = ['A','AG','AL','AU','CU','FU','HC','I','IC','IF','IH','J','JM',
                    'M','MA','NI','P','RB','RM','RU','SA','SN','SR','T','TA','TL','TS','Y','ZN']
        print(f"[universe=manual29] 手动核心品种 {len(universe)}")
    elif args.universe_mode != "all61":
        amount = runner.data_manager.get("amount", calendar, universe)
        daily_amount = amount.mean(skipna=True).sort_values(ascending=False)
        n = 45 if args.universe_mode == "liq45" else 29
        universe = daily_amount.index[:n].tolist()
        print(f"[universe={args.universe_mode}] 按日均成交额取前 {len(universe)}")

    # 因子暴露 + 打分
    names = list(FACTORS_SUB)
    computed = engine.compute_factors(names, calendar.tolist(), universe, parallel=True)
    score = pd.DataFrame(index=calendar, columns=universe, dtype=float)
    for name, direction in FACTORS_SUB.items():
        rank = computed[name].rank(axis=1, pct=True)
        oriented = rank if direction == 1 else (1 - rank)
        score = score.add(oriented, fill_value=0)
    score = score.div(len(names))

    # 日收益、波动率、IC
    close = runner.data_manager.get("close", calendar, universe)
    daily_ret = close.pct_change()
    vol20 = daily_ret.rolling(20, min_periods=10).std(ddof=0)
    fwd = daily_ret.shift(-1)

    # 1个月滚动IC (仅 ic 加权需要, 向量化逐日截面 spearman)
    rolling_ic = None
    mean_ic = mean_ic_std = None
    if args.weight_scheme == "ic":
        rolling_ic = pd.DataFrame(index=calendar, columns=universe, dtype=float)
        fwd5 = close.pct_change(5).shift(-5)
        for t in calendar:
            row_f_mat = {name: computed[name].loc[t] for name in names}
            row_r = fwd5.loc[t] if t in fwd5.index else pd.Series(dtype=float)
            for name in names:
                f_row = row_f_mat[name]
                valid = f_row.notna() & row_r.reindex(f_row.index).notna()
                if valid.sum() >= 5:
                    rolling_ic.loc[t, name] = f_row[valid].corr(
                        row_r.reindex(f_row.index)[valid], method="spearman")
        mean_ic = rolling_ic.mean()
        mean_ic_std = rolling_ic.std()
        print(f"IC均值={mean_ic.mean():.4f} IC标准差均值={mean_ic_std.mean():.3f}")

    # 调仓频率
    rebal = score.resample(args.rebalance).last()
    topn = args.topn

    rets, turnovers = [], []
    prev_long, prev_short = set(), set()
    for t in rebal.index:
        row = rebal.loc[t].dropna()
        if len(row) < 2 * topn:
            continue
        ranked = row.rank(ascending=False)
        long_pool = list(ranked[ranked <= topn].index)
        short_pool = list(ranked[ranked > len(ranked) - topn].index)

        # 连续两周确认 (首周 prev 为空时跳过过滤, 作为基线)
        if args.weight_scheme == "confirm2w" and prev_long:
            long_pool = [x for x in long_pool if x in prev_long]
            short_pool = [x for x in short_pool if x in prev_short]

        # 权重方案
        vol_t = vol20.loc[t] if t in vol20.index else vol20.asof(t)
        def rp_w(pool):
            v = vol_t[pool].replace(0, np.nan).dropna()
            if v.empty:
                return None
            w = 1.0 / v
            total = w.sum()
            if not np.isfinite(total) or total <= 0:
                return None
            w = w / total
            return w.to_dict()
        def ic_w(pool):
            w = {c: max(mean_ic[c], 0.01) for c in pool}
            s = sum(w.values())
            return {c: v / s for c, v in w.items()}
        def erc_w(pool):
            """池内 ERC: 近60日协方差 shrinkage=0.3, 与 strategies/combined.py 一致."""
            if pool is None or len(pool) < 2:
                return None
            from optimization.risk_budgeting import RiskBudgetingOptimizer
            start = t - pd.Timedelta(days=90)
            cal = pd.DatetimeIndex(runner.data_manager.get_calendar(start, t))
            ret_sub = close.pct_change().reindex(cal)[list(pool)].dropna()
            if ret_sub.shape[0] < 10:
                return None
            cov_raw = ret_sub.cov().values
            target = np.diag(np.diag(cov_raw))
            cov = 0.7 * cov_raw + 0.3 * target
            try:
                w = RiskBudgetingOptimizer._erc_weights(cov, np.ones(len(pool)))
            except (RuntimeError, ValueError):
                v = ret_sub.std(ddof=0).replace(0, np.nan).dropna()
                if v.empty:
                    return None
                w = (1.0 / v).values
                w = w / w.sum()
            w = pd.Series(w, index=pool).clip(lower=0.005, upper=0.20)
            return (w / w.sum()).to_dict()

        scheme = args.weight_scheme
        if args.strategy == "score":
            w_long = {c: 1.0 / len(long_pool) for c in long_pool} if long_pool else None
            w_short = {c: 1.0 / len(short_pool) for c in short_pool} if short_pool else None
        elif scheme in ("rp", "half_rp", "floored", "erc"):
            if scheme == "erc":
                w_long = erc_w(long_pool)
                w_short = erc_w(short_pool)
            else:
                rl, rs = rp_w(long_pool), rp_w(short_pool)
                if scheme == "half_rp":
                    eq = {c: 1.0 / len(long_pool) for c in long_pool} if long_pool else None
                    if rl and eq:
                        w_long = {c: 0.5 * eq[c] + 0.5 * rl[c] for c in rl}
                    else:
                        w_long = rl or eq
                    eqs = {c: 1.0 / len(short_pool) for c in short_pool} if short_pool else None
                    w_short = {c: 0.5 * eqs[c] + 0.5 * rs[c] for c in rs} if (rs and eqs) else (rs or eqs)
                elif scheme == "floored" and rl:
                    w_long = {c: max(v, 0.005) for c, v in rl.items()}
                    s = sum(w_long.values())
                    w_long = {c: v / s for c, v in w_long.items()}
                    w_short = {c: max(v, 0.005) for c, v in rs.items()}
                    s = sum(w_short.values())
                    w_short = {c: v / s for c, v in w_short.items()}
                else:
                    w_long, w_short = rl, rs
        elif scheme == "ic":
            w_long = ic_w(long_pool)
            w_short = ic_w(short_pool)
        else:  # equal / confirm2w
            w_long = {c: 1.0 / len(long_pool) for c in long_pool} if long_pool else None
            w_short = {c: 1.0 / len(short_pool) for c in short_pool} if short_pool else None

        # 换手率 (相对上周)
        turn = (len(set(long_pool) ^ prev_long) + len(set(short_pool) ^ prev_short))
        prev_long, prev_short = set(long_pool), set(short_pool)

        week_dates = calendar[(calendar >= t) & (calendar < t + pd.Timedelta(days=7))]
        if args.rebalance != "W-FRI":
            # 非周度: 用下一个调仓点界定持有期
            idx = rebal.index.get_indexer([t], method="bfill")
            nxt = rebal.index[idx[0] + 1] if idx[0] + 1 < len(rebal.index) else rebal.index[-1] + pd.Timedelta(days=1)
            week_dates = calendar[(calendar >= t) & (calendar < nxt)]
        for d in week_dates:
            if d not in fwd.index:
                continue
            r = fwd.loc[d]
            long_ret = sum(r[c] * w for c, w in w_long.items()) if w_long else 0
            short_ret = sum(r[c] * w for c, w in w_short.items()) if w_short else 0
            rets.append((d, long_ret - short_ret))
        if len(week_dates):
            denom = max(len(long_pool) + len(short_pool), 1)
            turnovers.append((t, turn / denom))

    ret_s = pd.Series({d: v for d, v in rets}).sort_index()
    m = metrics(ret_s)
    if turnovers:
        turn_s = pd.Series({t: v for t, v in turnovers}).sort_index()
        monthly_turn = turn_s.mean()
        over80 = (turn_s > 0.8).mean()
    else:
        monthly_turn, over80 = np.nan, np.nan

    tag = f"{args.universe_mode}/{args.strategy}/{args.weight_scheme}"
    print(f"[{tag}] 年化={m.get('annual_return', 0):.2%} 夏普={m.get('sharpe', 0):.2f} "
          f"回撤={m.get('max_drawdown', 0):.2%} 波动={m.get('volatility', 0):.2%} "
          f"月换手={monthly_turn:.0%} 超标周占比={over80:.0%}")

    # 月度分解 (诊断辅助)
    if not ret_s.empty:
        monthly = ret_s.groupby(ret_s.index.to_period('M')).mean() * 21  # 月收益近似
        print(f"  月度收益: " + ", ".join(f"{k}={v*100:.2f}%" for k, v in monthly.items()))


if __name__ == "__main__":
    main()
