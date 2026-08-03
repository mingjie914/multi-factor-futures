"""diag_weight_matrix — 当前生产配置下的权重方案全对比.

固定: 7因子 + 38品种池 + cap=3 + 日度调仓 + 2025-01~2026-07.
因子层合成: 等权 (EW) vs 滚动60日ICIR加权.
品种层权重: equal / rp / erc / half_rp / ic / floored.
输出: 汇总表 (全段/OOS/实盘) + 净值图.
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

UNIV38 = ["A", "AG", "AL", "AU", "CU", "FU", "HC", "I", "IC", "IF", "IH", "J", "JM",
          "M", "MA", "NI", "P", "RB", "RM", "RU", "SA", "SN", "SR", "T", "TA", "TL",
          "TS", "Y", "ZN", "IM", "TF", "CF", "OI", "LH", "JD", "SC", "V", "UR"]
SECTORS = {
    "有色": ["CU", "AL", "ZN", "NI", "SN", "AG", "AU"],
    "黑色": ["RB", "HC", "I", "J", "JM"],
    "能化": ["FU", "MA", "RU", "SA", "TA", "SC", "V", "UR"],
    "农产品": ["A", "M", "P", "RM", "Y", "SR", "CF", "OI", "LH", "JD"],
    "金融": ["IC", "IF", "IH", "T", "TL", "TS", "IM", "TF"],
}
F7 = {
    "intraday_jump_intensity_20d": -1, "intraday_price_peak_count_20d": 1,
    "intraday_realised_skewness_20d": 1, "intraday_dtws_20d": 1,
    "intraday_drip_stone_20d": -1, "intraday_peak_ridge_ratio_20d": -1,
    "intraday_seat_long_short_seat_ratio_20d": 1,
}


def backtest(runner, factors, universe, start, end, cap=3,
             weight="erc", icir_window=0):
    cal = pd.DatetimeIndex(runner.data_manager.get_calendar(
        pd.Timestamp(start), pd.Timestamp(end)))
    engine = FactorEngine(runner.data_manager)
    names = list(factors)
    computed = engine.compute_factors(names, cal, universe, parallel=True)
    close = runner.data_manager.get("close", cal, universe)
    daily_ret = close.pct_change()
    fwd = daily_ret.shift(-1)

    # 因子层合成: 等权 or ICIR
    ranks = {}
    for n, direction in factors.items():
        r = computed[n].rank(axis=1, pct=True)
        ranks[n] = r if direction == 1 else (1 - r)
    if icir_window > 0:
        fwd_rank = fwd.rank(axis=1)
        ic = pd.DataFrame({n: ranks[n].corrwith(fwd_rank, axis=1) for n in names})
        icir = ic.rolling(icir_window, min_periods=20).mean() / \
               ic.rolling(icir_window, min_periods=20).std().replace(0, np.nan)
        icir = icir.abs().shift(1).fillna(1.0 / len(names))
        wsum = icir.sum(axis=1).replace(0, np.nan)
        w = icir.div(wsum, axis=0)
        score = pd.DataFrame(0.0, index=cal, columns=universe)
        for n in names:
            score = score.add(ranks[n].mul(w[n], axis=0), fill_value=0)
        score = score.div(score.sum(axis=1).replace(0, np.nan), axis=0)
    else:
        score = pd.DataFrame(index=cal, columns=universe, dtype=float)
        for n in names:
            score = score.add(ranks[n], fill_value=0)
        score = score.div(len(names))

    vol20 = close.pct_change().rolling(20, min_periods=10).std(ddof=0)
    sector_of = {}
    for sec, mem in SECTORS.items():
        for m in mem:
            if m in universe:
                sector_of[m] = sec
    rebal = score.resample("D").last()
    rets = []
    prev_long, prev_short = None, None

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
        if weight == "confirm2w" and prev_long is not None:
            top = [x for x in top if x in prev_long]
            bot = [x for x in bot if x in prev_short]
        vt = vol20.loc[t] if t in vol20.index else vol20.asof(t)

        def rp_w(pool):
            v = vt[pool].replace(0, np.nan).dropna()
            if v.empty:
                return None
            w = 1.0 / v
            return (w / w.sum()).to_dict()

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

        eq_l = {c: 1.0 / len(top) for c in top} if top else None
        eq_s = {c: 1.0 / len(bot) for c in bot} if bot else None
        if weight == "equal":
            wl, ws = eq_l, eq_s
        elif weight == "rp":
            wl, ws = rp_w(top), rp_w(bot)
        elif weight == "erc":
            wl, ws = erc_w(top), erc_w(bot)
        elif weight == "half_rp":
            rl, rs = rp_w(top), rp_w(bot)
            wl = {c: 0.5 * eq_l[c] + 0.5 * rl[c] for c in rl} if (rl and eq_l) else (rl or eq_l)
            ws = {c: 0.5 * eq_s[c] + 0.5 * rs[c] for c in rs} if (rs and eq_s) else (rs or eq_s)
        elif weight == "floored":
            rl, rs = rp_w(top), rp_w(bot)
            wl = {c: max(v, 0.005) for c, v in rl.items()} if rl else None
            ws = {c: max(v, 0.005) for c, v in rs.items()} if rs else None
            if wl:
                s_ = sum(wl.values()); wl = {c: v / s_ for c, v in wl.items()}
            if ws:
                s_ = sum(ws.values()); ws = {c: v / s_ for c, v in ws.items()}
        else:  # ic / confirm2w fallback
            wl, ws = eq_l, eq_s
        prev_long, prev_short = set(top), set(bot)
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
        return (np.nan, np.nan, np.nan)
    nav = (1 + s).cumprod()
    ann = s.mean() * 252
    vol = s.std(ddof=0) * np.sqrt(252)
    mdd = (nav / nav.cummax() - 1).min()
    return (ann / vol if vol > 0 else np.nan, ann, mdd)


def main():
    cfg = load_config("config/intraday_backtest.yaml")
    runner = PipelineRunner(config=cfg)
    univ = list(UNIV38)
    weights = ["equal", "rp", "erc", "half_rp", "ic", "floored", "confirm2w"]
    fig, ax = plt.subplots(figsize=(15, 8))
    colors = {"equal": "#95a5a6", "rp": "#f39c12", "erc": "#2ecc71",
              "half_rp": "#e67e22", "ic": "#3498db", "floored": "#9b59b6",
              "confirm2w": "#c0392b"}
    print("=== 因子层: 等权合成, 品种层权重方案对比 ===")
    print("权重 | 全段夏普/回撤 | OOS夏普 | 实盘夏普")
    print("-" * 75)
    ew_results = {}
    for w in weights:
        s = backtest(runner, F7, univ, "2025-01-01", "2026-07-31", cap=3, weight=w)
        ew_results[w] = s
        sh, ann, mdd = stat(s)
        oos = s[(s.index >= pd.Timestamp("2026-03-01")) & (s.index <= pd.Timestamp("2026-05-15"))]
        live = s[s.index > pd.Timestamp("2026-05-15")]
        oos_sh, _, _ = stat(oos)
        live_sh, _, _ = stat(live)
        print(f"{w:<10} | {sh:.2f}/{mdd:.1%} | {oos_sh:.2f} | {live_sh:.2f}")
        ax.plot((1 + s).cumprod().index, (1 + s).cumprod().values,
                label=f"EW-{w}", color=colors[w], linewidth=1.2, alpha=0.8)
    print()
    print("=== 因子层: ICIR加权合成 (品种层: erc) vs 等权-erc ===")
    s_icir = backtest(runner, F7, univ, "2025-01-01", "2026-07-31", cap=3, weight="erc", icir_window=60)
    sh_i, ann_i, mdd_i = stat(s_icir)
    oos_i = s_icir[(s_icir.index >= pd.Timestamp("2026-03-01")) & (s_icir.index <= pd.Timestamp("2026-05-15"))]
    live_i = s_icir[s_icir.index > pd.Timestamp("2026-05-15")]
    oos_ish, _, _ = stat(oos_i)
    live_ish, _, _ = stat(live_i)
    print(f"ICIR-erc  | {sh_i:.2f}/{mdd_i:.1%} | {oos_ish:.2f} | {live_ish:.2f}")
    sh_e, ann_e, mdd_e = stat(ew_results["erc"])
    oos_e = ew_results["erc"][(ew_results["erc"].index >= pd.Timestamp("2026-03-01")) & (ew_results["erc"].index <= pd.Timestamp("2026-05-15"))]
    live_e = ew_results["erc"][ew_results["erc"].index > pd.Timestamp("2026-05-15")]
    oos_esh, _, _ = stat(oos_e)
    live_esh, _, _ = stat(live_e)
    print(f"EW-erc(基准) | {sh_e:.2f}/{mdd_e:.1%} | {oos_esh:.2f} | {live_esh:.2f}")
    ax.plot((1 + s_icir).cumprod().index, (1 + s_icir).cumprod().values,
            label="ICIR-erc", color="black", linewidth=2.0)
    ax.axvline(pd.Timestamp("2026-03-01"), color="gray", linestyle="--", alpha=0.8, label="OOS起点")
    ax.axvline(pd.Timestamp("2026-05-16"), color="red", linestyle="--", alpha=0.8, label="实盘起点")
    ax.set_title("权重方案对比 (7因子, 38品种, cap=3, 日度)")
    ax.set_ylabel("净值")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("runs/weight_matrix_nav.png", dpi=150)
    print("\n净值图: runs/weight_matrix_nav.png")


if __name__ == "__main__":
    main()
