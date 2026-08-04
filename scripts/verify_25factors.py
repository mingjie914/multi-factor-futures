"""关键验证: 6因子 vs 25因子 的实盘段(5/16后)表现 — 防止过拟合."""
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
# 25因子 = 6 + 前19个候选 (|t|降序加入, 到 lowest_time)
CAND25 = [
    'intraday_zero_ret_freq_20d', 'intraday_open_close_drift_20d', 'intraday_volatility_clustering_20d',
    'intraday_oi_vol_corr_daily_20d', 'intraday_oi_time_centroid_20d', 'intraday_wash_trade_20d',
    'intraday_settle_position_20d', 'intraday_cross_vol_20d', 'intraday_amihud_vol_ratio_20d',
    'intraday_oi_skew_stability_20d', 'intraday_depth_trend_20d', 'intraday_open_close_volume_ratio_20d',
    'intraday_oi_quantile_range_20d', 'intraday_settle_gap_20d', 'intraday_amihud_trend_20d',
    'intraday_price_delay_20d', 'intraday_overnight_absorption_20d', 'intraday_session_symmetry_20d',
    'intraday_lowest_time_20d',
]


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
        return pd.Series({d: v for d, v in rets}).sort_index().dropna()

    def stat(s, label):
        ann = s.mean() * 252
        vol = s.std(ddof=0) * np.sqrt(252)
        navs = (1 + s).cumprod()
        mdd = (navs / navs.cummax() - 1).min()
        oos = s[(s.index >= pd.Timestamp('2026-03-01')) & (s.index <= pd.Timestamp('2026-05-15'))]
        live = s[s.index > pd.Timestamp('2026-05-15')]
        oos_sh = oos.mean() * 252 / (oos.std(ddof=0) * np.sqrt(252)) if len(oos) > 2 and oos.std() > 0 else 0
        live_sh = live.mean() * 252 / (live.std(ddof=0) * np.sqrt(252)) if len(live) > 2 and live.std() > 0 else 0
        print(f'{label}: 年化={ann:.1%} 夏普={ann/vol if vol>0 else 0:.2f} 回撤={mdd:.1%} OOS={oos_sh:.2f} 实盘={live_sh:.2f}')

    # 方向: IC>0 正向
    info = {}
    for line in open('scripts/_final_candidates.txt', encoding='utf-8'):
        p = line.strip().split(',')
        info[p[0]] = (float(p[2]), int(p[3]))  # ic, period

    flist6 = list(PROD6)
    dirs6 = dict(DIR6)
    s6 = backtest(flist6, dirs6)
    stat(s6, '6因子 (生产)')

    flist25 = list(PROD6) + CAND25
    dirs25 = dict(DIR6)
    for c in CAND25:
        dirs25[c] = 1 if info[c][0] > 0 else -1
    s25 = backtest(flist25, dirs25)
    stat(s25, '25因子 (前向选择峰值)')

    # 逐年
    print('\n=== 实盘段逐月收益对比 ===')
    for lab, s in [('6因子', s6), ('25因子', s25)]:
        live = s[s.index > pd.Timestamp('2026-05-15')]
        m = live.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        print(f'{lab}: ' + ' '.join(f'{d.month}月={v:+.1%}' for d, v in m.items()))


if __name__ == '__main__':
    main()
