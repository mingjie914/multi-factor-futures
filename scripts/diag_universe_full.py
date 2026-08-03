"""diag_universe_full — 品种池 x 因子方案 全矩阵对比 (修正图例).

品种池: manual29(29) vs 扩展38(29+IM,TF,CF,OI,LH,JD,SC,V,UR)
因子方案: B3-6 / B3+席位(7,当前生产) / 8 / 10 / 12
权重: ERC | 调仓: 日度 | cap: 3
输出: 汇总表 + 2张净值图 (分池) + 1张子图网格.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from core.config import load_config
from pipeline.runner import PipelineRunner
from factors.engine import FactorEngine
from optimization.risk_budgeting import RiskBudgetingOptimizer

MANUAL29 = ["A", "AG", "AL", "AU", "CU", "FU", "HC", "I", "IC", "IF", "IH", "J", "JM",
            "M", "MA", "NI", "P", "RB", "RM", "RU", "SA", "SN", "SR", "T", "TA", "TL",
            "TS", "Y", "ZN"]
UNIV38 = MANUAL29 + ["IM", "TF", "CF", "OI", "LH", "JD", "SC", "V", "UR"]

SECTORS = {
    "有色": ["CU", "AL", "ZN", "NI", "SN", "AG", "AU"],
    "黑色": ["RB", "HC", "I", "J", "JM"],
    "能化": ["FU", "MA", "RU", "SA", "TA", "SC", "V", "UR"],
    "农产品": ["A", "M", "P", "RM", "Y", "SR", "CF", "OI", "LH", "JD"],
    "金融": ["IC", "IF", "IH", "T", "TL", "TS", "IM", "TF"],
}

B3 = {
    "intraday_jump_intensity_20d": -1, "intraday_price_peak_count_20d": 1,
    "intraday_realised_skewness_20d": 1, "intraday_dtws_20d": 1,
    "intraday_drip_stone_20d": -1, "intraday_peak_ridge_ratio_20d": -1,
}
F7 = dict(B3); F7["intraday_seat_long_short_seat_ratio_20d"] = 1
F8 = dict(F7); F8["intraday_seat_count_rank_20d"] = -1; F8["intraday_settle_close_basis_20d"] = -1
F10 = dict(F8); F10["intraday_seat_long_short_ratio_20d"] = 1; F10["intraday_price_rank_vol_20d"] = 1
F12 = dict(F10); F12["intraday_volume_rank_ratio_20d"] = 1; F12["intraday_settle_basis_rank_20d"] = 1
PLANS = {"B3-6因子": B3, "7因子(当前)": F7, "8因子": F8, "10因子": F10, "12因子": F12}
COLORS = {"B3-6因子": "#95a5a6", "7因子(当前)": "#2ecc71", "8因子": "#f39c12",
          "10因子": "#3498db", "12因子": "#9b59b6"}


def backtest(runner, factors, universe, start, end, cap=3):
    cal = pd.DatetimeIndex(runner.data_manager.get_calendar(
        pd.Timestamp(start), pd.Timestamp(end)))
    engine = FactorEngine(runner.data_manager)
    names = list(factors)
    computed = engine.compute_factors(names, cal, universe, parallel=True)
    score = pd.DataFrame(index=cal, columns=universe, dtype=float)
    for n, direction in factors.items():
        r = computed[n].rank(axis=1, pct=True)
        score = score.add(r if direction == 1 else (1 - r), fill_value=0)
    score = score.div(len(names))

    close = runner.data_manager.get("close", cal, universe)
    daily_ret = close.pct_change()
    fwd = daily_ret.shift(-1)
    vol20 = close.pct_change().rolling(20, min_periods=10).std(ddof=0)
    sector_of = {}
    for sec, mem in SECTORS.items():
        for m in mem:
            if m in universe:
                sector_of[m] = sec
    rebal = score.resample("D").last()
    rets = []

    def capped(order, cap_n):
        picks, counts = [], {}
        for s in order:
            sec = sector_of.get(s, "其他")
            if counts.get(sec, 0) >= cap_n:
                continue
            picks.append(s)
            counts[sec] = counts.get(sec, 0) + 1
            if len(picks) >= 10:
                break
        return picks

    for t in rebal.index:
        row = rebal.loc[t].dropna()
        if len(row) < 20:
            continue
        if cap > 0:
            top = capped(row.sort_values(ascending=False).index.tolist(), cap)
            bot = capped(row.sort_values(ascending=True).index.tolist(), cap)
        else:
            rk = row.rank(ascending=False)
            top = rk[rk <= 10].index
            bot = rk[rk > len(rk) - 10].index
        vt = vol20.loc[t] if t in vol20.index else vol20.asof(t)

        def erc_w(pool):
            if len(pool) < 2:
                return None
            start_d = t - pd.Timedelta(days=90)
            c = pd.DatetimeIndex(runner.data_manager.get_calendar(start_d, t))
            ret_sub = daily_ret.reindex(c)[list(pool)].dropna()
            if ret_sub.shape[0] < 10:
                return None
            cov_raw = ret_sub.cov().values
            target = np.diag(np.diag(cov_raw))
            cov = 0.7 * cov_raw + 0.3 * target
            try:
                w = RiskBudgetingOptimizer._erc_weights(cov, np.ones(len(pool)))
                return dict(zip(pool, w))
            except (RuntimeError, ValueError):
                v = ret_sub.std(ddof=0).replace(0, np.nan).dropna()
                if v.empty:
                    return None
                w = (1.0 / v).values
                return dict(zip(pool, w / w.sum()))

        wl, ws = erc_w(top), erc_w(bot)
        nxt = cal[(cal > t) & (cal <= t + pd.Timedelta(days=1))]
        for d in nxt:
            if d in fwd.index:
                r = fwd.loc[d]
                lr = sum(r[c] * wi for c, wi in wl.items()) if wl else 0
                sr = sum(r[c] * wi for c, wi in ws.items()) if ws else 0
                rets.append((d, lr - sr))
    return pd.Series({d: v for d, v in rets}).sort_index().dropna()


def stat(s):
    if len(s) < 3:
        return (np.nan, np.nan, np.nan, np.nan)
    nav = (1 + s).cumprod()
    ann = s.mean() * 252
    vol = s.std(ddof=0) * np.sqrt(252)
    mdd = (nav / nav.cummax() - 1).min()
    return (ann / vol if vol > 0 else np.nan, ann, mdd, vol)


def main():
    cfg = load_config("config/intraday_backtest.yaml")
    runner = PipelineRunner(config=cfg)
    pools = {"manual29": list(MANUAL29), "扩展38": list(UNIV38)}

    print("方案 | 品种池 | 全段夏普/回撤 | OOS夏普(3-1~5-15) | 实盘夏普(5-16~7-31)")
    print("-" * 95)
    results = {}  # (pool, plan) -> Series
    for pool_name, univ in pools.items():
        for plan_name, F in PLANS.items():
            s = backtest(runner, F, univ, "2025-01-01", "2026-07-31", cap=3)
            results[(pool_name, plan_name)] = s
            sh, ann, mdd, vol = stat(s)
            oos = s[(s.index >= pd.Timestamp("2026-03-01")) & (s.index <= pd.Timestamp("2026-05-15"))]
            live = s[s.index > pd.Timestamp("2026-05-15")]
            oos_sh, _, oos_mdd, _ = stat(oos)
            live_sh, _, live_mdd, _ = stat(live)
            print(f"{plan_name:<12} | {pool_name:<6} | {sh:.2f}/{mdd:.1%} | {oos_sh:.2f} | {live_sh:.2f}")

    # 图1: 分池净值 (每池一张)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, pool_name in zip(axes, pools):
        for plan_name in PLANS:
            s = results[(pool_name, plan_name)]
            nav = (1 + s).cumprod()
            ax.plot(nav.index, nav.values, label=plan_name, color=COLORS[plan_name], linewidth=1.3)
        ax.axvline(pd.Timestamp("2026-03-01"), color="gray", linestyle="--", alpha=0.8, label="OOS起点")
        ax.axvline(pd.Timestamp("2026-05-16"), color="red", linestyle="--", alpha=0.8, label="实盘起点")
        ax.set_title(f"品种池: {pool_name} (cap=3, ERC, 日度)")
        ax.set_ylabel("净值")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("runs/universe_full_nav.png", dpi=150)
    print("\n净值图(分池): runs/universe_full_nav.png")

    # 图2: 各方案内 品种池对比 (子图网格)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.flatten()
    for idx, plan_name in enumerate(PLANS):
        ax = axes[idx]
        for pool_name in pools:
            s = results[(pool_name, plan_name)]
            nav = (1 + s).cumprod()
            ax.plot(nav.index, nav.values, label=pool_name, linewidth=1.5)
        ax.axvline(pd.Timestamp("2026-03-01"), color="gray", linestyle="--", alpha=0.7)
        ax.axvline(pd.Timestamp("2026-05-16"), color="red", linestyle="--", alpha=0.7)
        ax.set_title(plan_name)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    axes[-1].axis("off")
    plt.suptitle("品种池对比 (manual29 vs 扩展38) — 按因子方案", fontsize=13)
    plt.tight_layout()
    plt.savefig("runs/universe_full_pool_compare.png", dpi=150)
    print("净值图(池对比): runs/universe_full_pool_compare.png")


if __name__ == "__main__":
    main()
