"""验证正确时序: 因子shift(1)后 T日权重应配 T日收益 (即 T-1信号×T日价格变动)."""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from core.config import load_config
from pipeline.runner import PipelineRunner
from factors.engine import FactorEngine
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
    # 手工 ERC (避免 scipy): 迭代等风险贡献
    n = len(pool)
    w = np.ones(n) / n
    for _ in range(200):
        sw = cov @ w
        mrc = sw / (w @ sw + 1e-12)  # 边际风险贡献
        target = (w @ sw) / n
        new_w = w * np.clip(target / (mrc + 1e-12), 0.2, 5.0)
        new_w = new_w / new_w.sum()
        if np.max(np.abs(new_w - w)) < 1e-10:
            w = new_w
            break
        w = new_w
    return dict(zip(pool, w))


# 正确时序: 因子T日值已shift(1) (基于≤T-1) → T-1收盘可算权重 → T日持有赚T日收益
# 即 权重[T] × daily_ret[T] (同日) — 这就是用户说的 T-1信号×T日价格变动
rets = []
for t in score.index:
    row = score.loc[t].dropna()
    if len(row) < 20:
        continue
    top = capped(row.sort_values(ascending=False).index.tolist(), SECTOR_CAP)
    bot = capped(row.sort_values(ascending=True).index.tolist(), SECTOR_CAP)
    wl = erc_w(top, t) or {}
    ws = erc_w(bot, t) or {}
    r = daily_ret.loc[t].fillna(0.0)
    lr = sum(r[c] * wi for c, wi in wl.items())
    sr = sum(r[c] * wi for c, wi in ws.items())
    rets.append((t, lr - sr))
s = pd.Series({d: v for d, v in rets}).sort_index().dropna()
ann = s.mean() * 252
vol = s.std(ddof=0) * np.sqrt(252)
navs = (1 + s).cumprod()
mdd = (navs / navs.cummax() - 1).min()
print(f'【正确时序 权重T×收益T(同日)】 n={len(s)} 年化={ann:.1%} 夏普={ann / vol if vol > 0 else 0:.2f} 回撤={mdd:.1%}')
print('= 因子shift1 → 权重T基于≤T-1 → T日持有 → T日收益 (即 T-1信号 × T日价格变动)')
