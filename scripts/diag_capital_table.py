"""一次性: 资金规模-可执行对照表 (8/3 持仓)."""
import sys
sys.path.insert(0, '.')
import pandas as pd, re, glob, os
from data.contract_specs import CONTRACT_SPECS
from strategies.combined import CombinedStrategy

strat = CombinedStrategy('config/intraday_backtest.yaml')
w = strat.signal('2026-08-03')
base = r'E:\程明杰公司内容\期货行情数据\本地表\futureshistoryprices1d'
months = sorted(os.listdir(base))
df = pd.read_parquet(glob.glob(os.path.join(base, months[-1], '*.parquet'))[0])
latest = df['trade_datetime'].max()
df = df[df['trade_datetime'] == latest]
closes = {}
for sym, grp in df.groupby('symbol'):
    m = re.match(r'^([A-Za-z]+)', str(sym))
    if m:
        root = m.group(1).upper()
        c = grp['close'].dropna()
        if not c.empty:
            closes[root] = float(c.iloc[0])

info = {}
for sym, weight in w.items():
    spec = CONTRACT_SPECS.get(sym)
    price = closes.get(sym)
    if not spec or not price:
        continue
    info[sym] = {'weight': weight, 'one_lot': price * spec['multiplier']}


def feasible(C):
    ok = 0
    for sym, d in info.items():
        if abs(d['weight']) * C / d['one_lot'] >= 1.0:
            ok += 1
    return ok


print('=== 资金规模 - 可执行对照表 (按目标权重配比, 8/3 收盘) ===')
print('{:>12} | {:>6} | {:>6} | 说明'.format('总资金', '可开仓', '覆盖'))
print('-' * 72)
for C in [100_000, 200_000, 300_000, 500_000, 800_000, 1_000_000,
          1_500_000, 2_000_000, 3_000_000, 5_000_000, 10_000_000]:
    ok = feasible(C)
    print('{:>12,.0f} | {:>3}/20 | {:>5.0f}% | {}'.format(
        C, ok, ok / 20 * 100, '完整复制' if ok == 20 else '部分'))
C = 100_000
while feasible(C) < 20:
    C += 50_000
print('\n完整复制全部 20 持仓的最低资金: {:,.0f} 元'.format(C))
