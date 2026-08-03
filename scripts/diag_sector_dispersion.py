"""diag_sector_dispersion — sector_dispersion 加入前后的 7 vs 8 因子对比.

7因子 = B3 + seat_long_short_seat_ratio (当前生产)
8因子 = 7 + sector_dispersion (候选, 完整检验通过)
品种池 38 | cap=3 | ERC | 日度 | 2025-01~2026-07. 输出净值图 (OOS/实盘标注).
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
F8 = dict(F7)
F8["intraday_sector_dispersion_20d"] = -1


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
        return "n/a"
    nav = (1 + s).cumprod()
    ann = s.mean() * 252
    vol = s.std(ddof=0) * np.sqrt(252)
    mdd = (nav / nav.cummax() - 1).min()
    return f"年化={ann:.1%} 夏普={ann/vol if vol > 0 else 0:.2f} 回撤={mdd:.1%} 波动={vol:.1%}"


def main():
    cfg = load_config("config/intraday_backtest.yaml")
    runner = PipelineRunner(config=cfg)
    univ = [s for s in UNIV38]
    plans = {
        "7因子 (当前生产)": (F7, "#2ecc71"),
        "8因子 (+sector_dispersion)": (F8, "#e67e22"),
    }
    fig, ax = plt.subplots(figsize=(15, 8))
    print("方案 | 全段 | OOS(3-1~5-15) | 实盘(5-16~7-31)")
    print("-" * 95)
    for label, (F, color) in plans.items():
        s = backtest(runner, F, univ, "2025-01-01", "2026-07-31", cap=3)
        nav = (1 + s).cumprod()
        oos = s[(s.index >= pd.Timestamp("2026-03-01")) & (s.index <= pd.Timestamp("2026-05-15"))]
        live = s[s.index > pd.Timestamp("2026-05-15")]
        print(f"{label:<28} | {stat(s)} | {stat(oos)} | {stat(live)}")
        ax.plot(nav.index, nav.values, label=label, color=color, linewidth=1.6)
    ax.axvline(pd.Timestamp("2026-03-01"), color="gray", linestyle="--", alpha=0.8, label="OOS起点 2026-03-01")
    ax.axvline(pd.Timestamp("2026-05-16"), color="red", linestyle="--", alpha=0.8, label="实盘起点 2026-05-16")
    ax.set_title("7因子 vs 8因子(+sector_dispersion) 净值对比 (38品种, cap=3, ERC, 日度)")
    ax.set_ylabel("净值")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("runs/7vs8_sector_dispersion_nav.png", dpi=150)
    print("\n净值图: runs/7vs8_sector_dispersion_nav.png")


if __name__ == "__main__":
    main()
