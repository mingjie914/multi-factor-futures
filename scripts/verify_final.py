"""最终验证: 修正后 production 口径 (权重T×收益T) vs 权重文件复算.

两者应完全一致 (同一套 ERC/选池/时序).
"""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from core.config import load_config
from pipeline.runner import PipelineRunner
from factors.engine import FactorEngine
from optimization.risk_budgeting import RiskBudgetingOptimizer
from strategies.combined import FACTORS, SECTOR_MAP, SECTOR_CAP

cfg = load_config('config/intraday_backtest.yaml')
runner = PipelineRunner(config=cfg)
cal = pd.DatetimeIndex(runner.data_manager.get_calendar(pd.Timestamp('2025-01-01'), pd.Timestamp('2026-07-31')))
u = list(runner.config.universe)
engine = FactorEngine(runner.data_manager)
comp = engine.compute_factors(list(FACTORS), cal, u, parallel=True)
score = pd.DataFrame(index=cal, columns=u, dtype=float)
for n, direction in FACTORS.items():
    r = comp[n].rank(axis=1, pct=True)
    score = score.add(r if direction == 1 else (1 - r), fill_value=0)
score = score.div(len(FACTORS))
close = runner.data_manager.get('close', cal, u)
daily_ret = close.pct_change()
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


# ===== 方案A: production 口径 (修正后, 同日) =====
rets_a = []
for t in score.index:
    row = score.loc[t].dropna()
    if len(row) < 20:
        continue
    top = capped(row.sort_values(ascending=False).index.tolist(), SECTOR_CAP)
    bot = capped(row.sort_values(ascending=True).index.tolist(), SECTOR_CAP)
    wl = erc_w(top, t) or {}
    ws = erc_w(bot, t) or {}
    if t in daily_ret.index:
        r = daily_ret.loc[t].fillna(0.0)
        lr = sum(r[c] * wi for c, wi in wl.items())
        sr = sum(r[c] * wi for c, wi in ws.items())
        rets_a.append((t, lr - sr))
sa = pd.Series({d: v for d, v in rets_a}).sort_index().dropna()
ann_a = sa.mean() * 252
vol_a = sa.std(ddof=0) * np.sqrt(252)
nav_a = (1 + sa).cumprod()
mdd_a = (nav_a / nav_a.cummax() - 1).min()
print(f'方案A (production修正后, 权重T×收益T): n={len(sa)} 年化={ann_a:.1%} 夏普={ann_a/vol_a:.2f} 回撤={mdd_a:.1%}')

# ===== 方案B: 权重文件复算 (T-1权重×T日收益 = 权重[prev] × ret[T]) =====
w = pd.read_csv('weights/daily_weights.csv', parse_dates=['date'])
wmat = w.pivot(index='date', columns='symbol', values='weight').fillna(0.0).sort_index()
# 权重文件日期=T 代表 T-1收盘信号 → T日持有 → T日收益 (同日)
rets_b = []
for t in wmat.index:
    wt = wmat.loc[t]
    if t not in daily_ret.index:
        continue
    r = daily_ret.loc[t].fillna(0.0)
    common = [s for s in wt.index if s in r.index]
    rets_b.append((t, float((wt[common] * r[common]).sum())))
sb = pd.Series({d: v for d, v in rets_b}).sort_index().dropna()
ann_b = sb.mean() * 252
vol_b = sb.std(ddof=0) * np.sqrt(252)
nav_b = (1 + sb).cumprod()
mdd_b = (nav_b / nav_b.cummax() - 1).min()
print(f'方案B (权重文件复算, 同日): n={len(sb)} 年化={ann_b:.1%} 夏普={ann_b/vol_b:.2f} 回撤={mdd_b:.1%}')

# 逐日对比
common_dates = sa.index.intersection(sb.index)
diff = (sa.reindex(common_dates) - sb.reindex(common_dates)).abs()
print(f'\n共同日期: {len(common_dates)} | 日收益最大差异: {diff.max():.6f} | 平均差异: {diff.mean():.6f}')
