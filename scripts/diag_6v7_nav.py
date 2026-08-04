"""diag_6v7_nav — 6因子 vs 7因子 (均等权+ERC) 净值对比图."""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from core.config import load_config
from pipeline.runner import PipelineRunner
from factors.engine import FactorEngine
from optimization.risk_budgeting import RiskBudgetingOptimizer
from strategies.combined import SECTOR_MAP, SECTOR_CAP

UNIV38 = ["A", "AG", "AL", "AU", "CU", "FU", "HC", "I", "IC", "IF", "IH", "J", "JM",
          "M", "MA", "NI", "P", "RB", "RM", "RU", "SA", "SN", "SR", "T", "TA", "TL",
          "TS", "Y", "ZN", "IM", "TF", "CF", "OI", "LH", "JD", "SC", "V", "UR"]
F6 = {
    "intraday_jump_intensity_20d": -1, "intraday_price_peak_count_20d": 1,
    "intraday_realised_skewness_20d": 1, "intraday_dtws_20d": 1,
    "intraday_drip_stone_20d": -1, "intraday_peak_ridge_ratio_20d": -1,
}
F7 = dict(F6)
F7["intraday_seat_long_short_seat_ratio_20d"] = 1


def main():
    cfg = load_config('config/intraday_backtest.yaml')
    runner = PipelineRunner(config=cfg)
    cal = pd.DatetimeIndex(runner.data_manager.get_calendar(pd.Timestamp('2025-01-01'), pd.Timestamp('2026-07-31')))
    u = list(UNIV38)
    engine = FactorEngine(runner.data_manager)
    close = runner.data_manager.get('close', cal, u)
    daily_ret = close.pct_change()
    vol20 = close.pct_change().rolling(20, min_periods=10).std(ddof=0)
    sector_of = {}
    for sec, mem in SECTOR_MAP.items():
        for m in mem:
            if m in u:
                sector_of[m] = sec

    def capped(order, cap_n):
        picks, counts = [], {}
        for s in order:
            sec = sector_of.get(s, '其他')
            if counts.get(sec, 0) >= cap_n:
                continue
            picks.append(s)
            counts[sec] = counts.get(sec, 0) + 1
            if len(picks) >= 10:
                break
        return picks

    def erc_w(pool, t):
        if len(pool) < 2:
            return None
        sd = t - pd.Timedelta(days=90)
        c = pd.DatetimeIndex(runner.data_manager.get_calendar(sd, t))
        rs = daily_ret.reindex(c)[list(pool)].dropna()
        if rs.shape[0] < 10:
            return None
        cov_raw = rs.cov().values
        cov = 0.7 * cov_raw + 0.3 * np.diag(np.diag(cov_raw))
        try:
            w = RiskBudgetingOptimizer._erc_weights(cov, np.ones(len(pool)))
            return dict(zip(pool, w))
        except (RuntimeError, ValueError):
            v = rs.std(ddof=0).replace(0, np.nan).dropna()
            if v.empty:
                return None
            w = (1.0 / v).values
            return dict(zip(pool, w / w.sum()))

    def run(F):
        comp = engine.compute_factors(list(F), cal, u, parallel=True)
        score = pd.DataFrame(index=cal, columns=u, dtype=float)
        for n, direction in F.items():
            r = comp[n].rank(axis=1, pct=True)
            score = score.add(r if direction == 1 else (1 - r), fill_value=0)
        score = score.div(len(F))
        rets = []
        for t in score.index:
            row = score.loc[t].dropna()
            if len(row) < 20:
                continue
            top = capped(row.sort_values(ascending=False).index.tolist(), SECTOR_CAP)
            bot = capped(row.sort_values(ascending=True).index.tolist(), SECTOR_CAP)
            wl, ws = erc_w(top, t), erc_w(bot, t)
            if t in daily_ret.index:
                r = daily_ret.loc[t].fillna(0.0)
                lr = sum(r[c] * wi for c, wi in (wl or {}).items())
                sr = sum(r[c] * wi for c, wi in (ws or {}).items())
                rets.append((t, lr - sr))
        return pd.Series({d: v for d, v in rets}).sort_index().dropna()

    fig, ax = plt.subplots(figsize=(15, 8))
    for label, F, color, lw in [
        ('6因子-等权-ERC (生产)', F6, '#2ecc71', 2.2),
        ('7因子-等权-ERC', F7, '#e67e22', 1.8),
    ]:
        s = run(F)
        nav = (1 + s).cumprod()
        ann = s.mean() * 252
        vol = s.std(ddof=0) * np.sqrt(252)
        mdd = (nav / nav.cummax() - 1).min()
        oos = s[(s.index >= pd.Timestamp('2026-03-01')) & (s.index <= pd.Timestamp('2026-05-15'))]
        live = s[s.index > pd.Timestamp('2026-05-15')]
        oos_sh = oos.mean() * 252 / (oos.std(ddof=0) * np.sqrt(252)) if len(oos) > 2 and oos.std() > 0 else 0
        live_sh = live.mean() * 252 / (live.std(ddof=0) * np.sqrt(252)) if len(live) > 2 and live.std() > 0 else 0
        print(f"{label}: 年化={ann:.1%} 夏普={ann/vol:.2f} 回撤={mdd:.1%} OOS={oos_sh:.2f} 实盘={live_sh:.2f}")
        ax.plot(nav.index, nav.values, label=label, color=color, linewidth=lw)
    ax.axvline(pd.Timestamp('2026-03-01'), color='gray', linestyle='--', alpha=0.8, label='OOS起点 2026-03-01')
    ax.axvline(pd.Timestamp('2026-05-16'), color='red', linestyle='--', alpha=0.8, label='实盘起点 2026-05-16')
    ax.set_title('6因子 vs 7因子 净值对比 (等权+ERC, 时序修正后, 38池+cap3+日度)')
    ax.set_ylabel('净值')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('runs/6v7_nav_fixed.png', dpi=150)
    print('净值图: runs/6v7_nav_fixed.png')


if __name__ == '__main__':
    main()
