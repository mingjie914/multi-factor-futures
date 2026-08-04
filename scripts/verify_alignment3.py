"""实测: CSV 权重三种对齐方式 vs production (2.04).

对齐A: weight_T × ret_T      (同日, CSV日期=执行日)
对齐B: weight_{T-1} × ret_T  (shift 1, CSV日期=信号日)
对齐C: weight_T × ret_{T+1}  (shift -1)
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
ret = close.pct_change()  # ret[T] = close[T]/close[T-1]-1

w = pd.read_csv('weights/daily_weights.csv', parse_dates=['date'])
wmat = w.pivot(index='date', columns='symbol', values='weight').fillna(0.0).sort_index()


def run(offset):
    """offset: 权重日期到收益日期的交易日偏移 (0=同日, 1=权重T配收益T+1, -1=权重T-1配收益T)."""
    rets, dates = [], []
    for t in wmat.index:
        wt = wmat.loc[t]
        t_ret = t
        if offset > 0:
            for _ in range(offset):
                nxt = cal[cal > t_ret]
                if len(nxt) == 0:
                    break
                t_ret = nxt[0]
        elif offset < 0:
            prv = cal[cal < t_ret]
            if len(prv) == 0:
                continue
            t_ret = prv[-1]
        if t_ret not in ret.index:
            continue
        r = ret.loc[t_ret].fillna(0.0)
        common = [s for s in wt.index if s in r.index]
        pr = float((wt[common] * r[common]).sum())
        rets.append(pr)
        dates.append(t_ret)
    s = pd.Series(rets, index=pd.DatetimeIndex(dates)).sort_index().dropna()
    ann = s.mean() * 252
    vol = s.std(ddof=0) * np.sqrt(252)
    navs = (1 + s).cumprod()
    mdd = (navs / navs.cummax() - 1).min()
    return ann, ann / vol if vol > 0 else 0, mdd, vol, len(s)


print('=== CSV 三种对齐方式复算 ===')
print(f'{"对齐":<28} {"年化":>7} {"夏普":>6} {"回撤":>7} {"波动":>7} {"n":>4}')
for offset, label in [(0, 'A: weight_T × ret_T (同日)'),
                      (1, 'C: weight_T × ret_{T+1} (次日)'),
                      (-1, 'B: weight_{T-1} × ret_T (前移)')]:
    ann, sh, mdd, vol, n = run(offset)
    print(f'{label:<28} {ann:>6.1%} {sh:>6.2f} {mdd:>6.1%} {vol:>6.1%} {n:>4}')
print()
print('production 报告: 年化 18.9% 夏普 2.04 回撤 -5.9% 波动 ~9.3%')
print('你的外部结果:   年化 20.31% 夏普 2.20 回撤 -5.90% 波动 9.21%')
