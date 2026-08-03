"""验证权重对齐语义: T日权重 × T日收益 vs T日权重 × T+1日收益.

用 weights/daily_weights.csv + 真实日收益, 两种对齐各算净值,
对比 production 报告的 2.27 夏普, 确定正确时序.
"""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from core.config import load_config
from pipeline.runner import PipelineRunner

cfg = load_config('config/intraday_backtest.yaml')
runner = PipelineRunner(config=cfg)
cal = pd.DatetimeIndex(runner.data_manager.get_calendar(pd.Timestamp('2025-01-01'), pd.Timestamp('2026-07-31')))
u = list(runner.config.universe)
close = runner.data_manager.get('close', cal, u)
ret = close.pct_change(fill_method=None)  # ret.loc[t] = t 日收益

w = pd.read_csv('weights/daily_weights.csv', parse_dates=['date'])
# 权重矩阵: date x symbol
wmat = w.pivot(index='date', columns='symbol', values='weight').fillna(0.0)
print(f'权重矩阵: {wmat.shape}, 日期 {wmat.index[0].date()} ~ {wmat.index[-1].date()}')

def backtest(aligned_days):
    """aligned_days: 权重日期与收益日期的偏移 (0=同日, 1=权重T日收益T+1日)."""
    nav = 1.0
    daily_ret_series = []
    dates = []
    for i, t in enumerate(wmat.index):
        wt = wmat.loc[t]
        # 收益日 = t + aligned_days 交易日
        t_ret = t
        for _ in range(aligned_days):
            nxt = cal[cal > t_ret]
            if len(nxt) == 0:
                break
            t_ret = nxt[0]
        if t_ret not in ret.index:
            continue
        r = ret.loc[t_ret].fillna(0.0)
        # 对齐权重品种
        common = [s for s in wt.index if s in r.index]
        pr = float((wt[common] * r[common]).sum())
        nav *= (1 + pr)
        daily_ret_series.append(pr)
        dates.append(t)
    s = pd.Series(daily_ret_series, index=pd.DatetimeIndex(dates))
    ann = s.mean() * 252
    vol = s.std(ddof=0) * np.sqrt(252)
    nav_series = (1 + s).cumprod()
    mdd = (nav_series / nav_series.cummax() - 1).min()
    return ann, ann / vol if vol > 0 else 0, mdd, len(s)

for off, label in [(0, 'T日权重 × T日收益 (同日)'), (1, 'T日权重 × T+1日收益 (次日)')]:
    ann, sh, mdd, n = backtest(off)
    print(f'{label}: n={n} 年化={ann:.1%} 夏普={sh:.2f} 回撤={mdd:.1%}')

print('\nproduction 报告: 年化 19.9% 夏普 2.27 回撤 -5.1%')
