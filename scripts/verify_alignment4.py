"""实证: 当前 daily_weights.csv 是几因子? 用户 shift 用法复算 = 1.47? 同日 = 2.04?"""
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
ret = close.pct_change()

w = pd.read_csv('weights/daily_weights.csv', parse_dates=['date'])
wmat = w.pivot(index='date', columns='symbol', values='weight').fillna(0.0).sort_index()
print(f'CSV 形状: {wmat.shape}, 首日 {wmat.index[0].date()}, 末日 {wmat.index[-1].date()}')

# 验证是不是 6 因子: 检查 2025-01-16 的多头是否包含 seat_long_short_seat_ratio 相关品种差异
# 对比: 生产 6 因子在 2025-01-16 的持仓
import importlib
import strategies.combined as sc
importlib.reload(sc)
print(f'当前 combined.FACTORS 数: {len(sc.FACTORS)}')

# 用户用法: shift(1) → weight_{T-1} × ret_T? 或 weight_T × ret_{T+1}?
# 用户说"加回 shift(1): weight_{T-1} × return_T"
def run(offset):
    rets, dates = [], []
    for t in wmat.index:
        wt = wmat.loc[t]
        t_ret = t
        if offset > 0:
            for _ in range(offset):
                nxt = cal[cal > t_ret]
                if len(nxt) == 0: break
                t_ret = nxt[0]
        elif offset < 0:
            prv = cal[cal < t_ret]
            if len(prv) == 0: continue
            t_ret = prv[-1]
        if t_ret not in ret.index: continue
        r = ret.loc[t_ret].fillna(0.0)
        common = [s for s in wt.index if s in r.index]
        rets.append(float((wt[common]*r[common]).sum()))
        dates.append(t_ret)
    s = pd.Series(rets, index=pd.DatetimeIndex(dates)).sort_index().dropna()
    ann = s.mean()*252; vol = s.std(ddof=0)*np.sqrt(252)
    navs = (1+s).cumprod(); mdd = (navs/navs.cummax()-1).min()
    return ann, ann/vol if vol>0 else 0, mdd, vol, len(s)

print('\n=== 当前 CSV (6因子) 复算 ===')
print(f'{"对齐":<24} {"年化":>7} {"夏普":>6} {"回撤":>7} {"波动":>7}')
for off, lab in [(0,'同日 weight_T×ret_T'), (1,'次日 weight_T×ret_{T+1}'), (-1,'前移 weight_{T-1}×ret_T')]:
    ann, sh, mdd, vol, n = run(off)
    print(f'{lab:<24} {ann:>6.1%} {sh:>6.2f} {mdd:>6.1%} {vol:>6.1%}')
print('\n你的两次结果: 2.20 (旧CSV) / 1.47 (新CSV+shift)')
print('production 正确: 2.04 (同日)')
