"""diag_production_nav — 当前生产方案净值图 (7因子+38池+cap3+ERC+日度).

数据窗口: 2025-01-01 ~ 最新数据日 (2026-07-31). 标注 OOS 起点 2026-03-01 与实盘起点 2026-05-16.
输出: runs/production_nav.png + 控制台分阶段统计.
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


def main():
    cfg = load_config("config/intraday_backtest.yaml")
    runner = PipelineRunner(config=cfg)
    univ = list(UNIV38)
    cal = pd.DatetimeIndex(runner.data_manager.get_calendar(
        pd.Timestamp("2025-01-01"), pd.Timestamp("2026-07-31")))
    engine = FactorEngine(runner.data_manager)
    comp = engine.compute_factors(list(F7), cal, univ, parallel=True)
    score = pd.DataFrame(index=cal, columns=univ, dtype=float)
    for n, direction in F7.items():
        r = comp[n].rank(axis=1, pct=True)
        score = score.add(r if direction == 1 else (1 - r), fill_value=0)
    score = score.div(len(F7))
    close = runner.data_manager.get("close", cal, univ)
    daily_ret = close.pct_change()
    fwd = daily_ret.shift(-1)
    vol20 = close.pct_change().rolling(20, min_periods=10).std(ddof=0)
    sector_of = {}
    for sec, mem in SECTORS.items():
        for m in mem:
            if m in univ:
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
        top = capped(row.sort_values(ascending=False).index.tolist(), 3)
        bot = capped(row.sort_values(ascending=True).index.tolist(), 3)
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
            cov = 0.7 * cov_raw + 0.3 * np.diag(np.diag(cov_raw))
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
    s = pd.Series({d: v for d, v in rets}).sort_index().dropna()
    nav = (1 + s).cumprod()

    def seg(d0, d1):
        x = s[(s.index >= pd.Timestamp(d0)) & (s.index <= pd.Timestamp(d1))]
        if len(x) < 3:
            return None
        n = (1 + x).cumprod()
        ann = x.mean() * 252
        vol = x.std(ddof=0) * np.sqrt(252)
        return ann, ann / vol if vol > 0 else 0, (n / n.cummax() - 1).min()

    print("=== 当前生产方案 (7因子+38池+cap3+ERC+日度) ===")
    full = seg("2025-01-01", "2026-07-31")
    print(f"全段: 年化={full[0]:.1%} 夏普={full[1]:.2f} 回撤={full[2]:.1%}")
    oos = seg("2026-03-01", "2026-05-15")
    live = seg("2026-05-16", "2026-07-31")
    print(f"OOS(3-1~5-15): 年化={oos[0]:.1%} 夏普={oos[1]:.2f} 回撤={oos[2]:.1%}")
    print(f"实盘(5-16~7-31): 年化={live[0]:.1%} 夏普={live[1]:.2f} 回撤={live[2]:.1%}")

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(nav.index, nav.values, color="#2ecc71", linewidth=1.8, label="生产方案净值")
    ax.axvline(pd.Timestamp("2026-03-01"), color="gray", linestyle="--", alpha=0.8, label="OOS起点 2026-03-01")
    ax.axvline(pd.Timestamp("2026-05-16"), color="red", linestyle="--", alpha=0.8, label="实盘起点 2026-05-16")
    ax.fill_between([pd.Timestamp("2026-03-01"), pd.Timestamp("2026-05-15")],
                    ax.get_ylim()[0], ax.get_ylim()[1], color="gray", alpha=0.08)
    ax.set_title("生产方案净值 (7因子+38品种+cap3+ERC+日度, 2025-01~2026-07)")
    ax.set_ylabel("净值")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("runs/production_nav.png", dpi=150)
    print("净值图: runs/production_nav.png")


if __name__ == "__main__":
    main()
