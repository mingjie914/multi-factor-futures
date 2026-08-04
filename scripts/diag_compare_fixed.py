"""diag_compare_fixed — 修正时序后的多方案净值对比.

统一口径: 因子shift(1) → 权重T基于≤T-1 → T日持有 → T日收益 (T-1信号×T日价格变动).
方案: 
  因子集: 6因子(B3) vs 7因子(B3+席位)
  合成:  等权 vs ICIR加权
  权重:  ERC (池内) vs RP(逆波动) vs 等权
品种池 38, cap=3, 日度, 2025-01-01~2026-07-31.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
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

    def rp_w(pool, t):
        vt = vol20.loc[t] if t in vol20.index else vol20.asof(t)
        v = vt[pool].replace(0, np.nan).dropna()
        if v.empty:
            return None
        w = 1.0 / v
        return (w / w.sum()).to_dict()

    def run(F, icir=False, weight='erc'):
        comp = engine.compute_factors(list(F), cal, u, parallel=True)
        ranks = {}
        for n, direction in F.items():
            r = comp[n].rank(axis=1, pct=True)
            ranks[n] = r if direction == 1 else (1 - r)
        if icir:
            fwd_rank = daily_ret.rank(axis=1)
            ic = pd.DataFrame({n: ranks[n].corrwith(fwd_rank, axis=1) for n in F})
            icir_m = ic.rolling(60, min_periods=20).mean() / ic.rolling(60, min_periods=20).std().replace(0, np.nan)
            icir_m = icir_m.abs().shift(1).fillna(1.0 / len(F))
            wsum = icir_m.sum(axis=1).replace(0, np.nan)
            w = icir_m.div(wsum, axis=0)
            score = pd.DataFrame(0.0, index=cal, columns=u)
            for n in F:
                score = score.add(ranks[n].mul(w[n], axis=0), fill_value=0)
            score = score.div(score.sum(axis=1).replace(0, np.nan), axis=0)
        else:
            score = pd.DataFrame(index=cal, columns=u, dtype=float)
            for n in F:
                score = score.add(ranks[n], fill_value=0)
            score = score.div(len(F))

        rets = []
        for t in score.index:
            row = score.loc[t].dropna()
            if len(row) < 20:
                continue
            top = capped(row.sort_values(ascending=False).index.tolist(), SECTOR_CAP)
            bot = capped(row.sort_values(ascending=True).index.tolist(), SECTOR_CAP)
            if weight == 'erc':
                wl, ws = erc_w(top, t), erc_w(bot, t)
            elif weight == 'rp':
                wl, ws = rp_w(top, t), rp_w(bot, t)
            else:  # equal
                wl = {c: 1.0 / len(top) for c in top} if top else None
                ws = {c: 1.0 / len(bot) for c in bot} if bot else None
            if t in daily_ret.index:
                r = daily_ret.loc[t].fillna(0.0)
                lr = sum(r[c] * wi for c, wi in (wl or {}).items())
                sr = sum(r[c] * wi for c, wi in (ws or {}).items())
                rets.append((t, lr - sr))
        s = pd.Series({d: v for d, v in rets}).sort_index().dropna()
        ann = s.mean() * 252
        vol = s.std(ddof=0) * np.sqrt(252)
        navs = (1 + s).cumprod()
        mdd = (navs / navs.cummax() - 1).min()
        oos = s[(s.index >= pd.Timestamp('2026-03-01')) & (s.index <= pd.Timestamp('2026-05-15'))]
        live = s[s.index > pd.Timestamp('2026-05-15')]
        oos_sh = oos.mean() * 252 / (oos.std(ddof=0) * np.sqrt(252)) if len(oos) > 2 and oos.std() > 0 else 0
        live_sh = live.mean() * 252 / (live.std(ddof=0) * np.sqrt(252)) if len(live) > 2 and live.std() > 0 else 0
        return ann, ann / vol if vol > 0 else 0, mdd, oos_sh, live_sh

    print("=== 修正时序后方案对比 (T-1信号×T日收益, 38池+cap3+日度) ===")
    print(f"{'方案':<28} | {'年化':>6} {'夏普':>5} {'回撤':>6} {'OOS夏普':>6} {'实盘夏普':>6}")
    print('-' * 80)
    cases = [
        ('6因子-等权-ERC', lambda: run(F6, icir=False, weight='erc')),
        ('7因子-等权-ERC', lambda: run(F7, icir=False, weight='erc')),
        ('7因子-ICIR-ERC', lambda: run(F7, icir=True, weight='erc')),
        ('7因子-等权-RP', lambda: run(F7, icir=False, weight='rp')),
        ('7因子-等权-等权', lambda: run(F7, icir=False, weight='equal')),
    ]
    for label, fn in cases:
        ann, sh, mdd, oos, live = fn()
        print(f"{label:<28} | {ann:>5.1%} {sh:>5.2f} {mdd:>5.1%} {oos:>6.2f} {live:>6.2f}")


if __name__ == '__main__':
    main()
