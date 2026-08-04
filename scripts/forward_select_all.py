"""前向选择: 生产6因子 + 27候选, 按|t|降序逐个加入, 记录夏普/回撤."""
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
PROD6 = ['intraday_jump_intensity_20d', 'intraday_price_peak_count_20d', 'intraday_realised_skewness_20d',
         'intraday_dtws_20d', 'intraday_drip_stone_20d', 'intraday_peak_ridge_ratio_20d']
DIR6 = {'intraday_jump_intensity_20d': -1, 'intraday_price_peak_count_20d': 1,
        'intraday_realised_skewness_20d': 1, 'intraday_dtws_20d': 1,
        'intraday_drip_stone_20d': -1, 'intraday_peak_ridge_ratio_20d': -1}


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

    def backtest(flist, dirs):
        comp = engine.compute_factors(flist, cal, u, parallel=True)
        score = pd.DataFrame(index=cal, columns=u, dtype=float)
        for n in flist:
            r = comp[n].rank(axis=1, pct=True)
            score = score.add(r if dirs[n] == 1 else (1 - r), fill_value=0)
        score = score.div(len(flist))
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
        s = pd.Series({d: v for d, v in rets}).sort_index().dropna()
        ann = s.mean() * 252
        vol = s.std(ddof=0) * np.sqrt(252)
        navs = (1 + s).cumprod()
        mdd = (navs / navs.cummax() - 1).min()
        oos = s[(s.index >= pd.Timestamp('2026-03-01')) & (s.index <= pd.Timestamp('2026-05-15'))]
        oos_sh = oos.mean() * 252 / (oos.std(ddof=0) * np.sqrt(252)) if len(oos) > 2 and oos.std() > 0 else 0
        return ann, ann / vol if vol > 0 else 0, mdd, oos_sh

    # 27 候选 (|t| 降序) — 读取清单
    cand = []
    for line in open('scripts/_final_candidates.txt', encoding='utf-8'):
        parts = line.strip().split(',')
        cand.append((parts[0], float(parts[1]), float(parts[2]), int(parts[3])))
    cand.sort(key=lambda x: -x[1])

    print('=== 前向选择 (6因子 + 逐个加入候选) ===')
    print(f"{'因子数':>4} {'组合':<30} {'年化':>6} {'夏普':>5} {'回撤':>6} {'OOS':>5}")
    flist = list(PROD6)
    dirs = dict(DIR6)
    ann, sh, mdd, oos = backtest(flist, dirs)
    print(f"{len(flist):>4} {'生产6因子':<30} {ann:>5.1%} {sh:>5.2f} {mdd:>5.1%} {oos:>5.2f}")
    best = (sh, len(flist), '生产6因子')
    for name, t, ic, p in cand:
        # 方向: IC>0 正向, IC<0 负向
        flist2 = flist + [name]
        dirs2 = dict(dirs)
        dirs2[name] = 1 if ic > 0 else -1
        ann, sh, mdd, oos = backtest(flist2, dirs2)
        flag = '⬆' if sh > best[0] else ''
        print(f"{len(flist2):>4} {('+'+name):<30} {ann:>5.1%} {sh:>5.2f} {mdd:>5.1%} {oos:>5.2f} {flag}")
        if sh > best[0]:
            best = (sh, len(flist2), name)
        flist, dirs = flist2, dirs2
    print(f'\n最佳: 夏普 {best[0]:.2f} @ {best[1]} 因子 (最后一个加入: {best[2]})')


if __name__ == '__main__':
    main()
