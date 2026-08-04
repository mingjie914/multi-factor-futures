"""对 132 个通过 FDR 的因子做: 与生产6因子相关性去重 + 候选间去重."""
import sys, json
sys.path.insert(0, '.')

def json_load(p):
    return json.load(open(p, encoding='utf-8'))
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from core.config import load_config
from pipeline.runner import PipelineRunner
from factors.engine import FactorEngine

cfg = load_config('config/intraday_backtest.yaml')
runner = PipelineRunner(config=cfg)
cal = pd.DatetimeIndex(runner.data_manager.get_calendar(pd.Timestamp('2025-01-01'), pd.Timestamp('2026-02-28')))
u = list(runner.config.universe)
engine = FactorEngine(runner.data_manager)

# 通过 FDR 的 132 个 (含生产6)
passed = []
for bi in range(1, 5):
    d = json_load(f'runs/ia_full_b{bi}/ic_by_window_period.json')
    for r in d.get('all_results', []):
        if r.get('best_period', 0) > 0:
            passed.append(r['name'])
passed = list(dict.fromkeys(passed))

PROD6 = ['intraday_jump_intensity_20d', 'intraday_price_peak_count_20d', 'intraday_realised_skewness_20d',
         'intraday_dtws_20d', 'intraday_drip_stone_20d', 'intraday_peak_ridge_ratio_20d']
cand = [n for n in passed if n not in PROD6]
print(f'通过: {len(passed)}, 候选(除生产6): {len(cand)}')

# 分批计算相关性 (避免内存过大)
all_names = PROD6 + cand
comp = engine.compute_factors(all_names, cal, u, parallel=True)
daily = pd.DataFrame({n: comp[n].mean(axis=1) for n in all_names})
corr = daily.corr()

# 与生产6相关性
print('\n=== 与生产6因子 max|corr| < 0.5 的候选 ===')
ok_vs_prod = []
for n in cand:
    mc = max(abs(corr.loc[n, p]) for p in PROD6)
    if mc < 0.5:
        ok_vs_prod.append((n, mc))
print(f'通过: {len(ok_vs_prod)}')

# 候选间去重 (贪心: 保留 |t| 更大的)
import heapq
info = {}
for bi in range(1, 5):
    d = json_load(f'runs/ia_full_b{bi}/ic_by_window_period.json')
    for r in d.get('all_results', []):
        if r.get('best_period', 0) > 0:
            info[r['name']] = (abs(r.get('best_t', 0)), r.get('best_ic', 0), r.get('best_period', 0))

ok_names = [n for n, _ in ok_vs_prod]
ok_names.sort(key=lambda n: -info[n][0])  # |t| 降序
selected = []
for n in ok_names:
    if all(abs(corr.loc[n, s]) < 0.5 for s in selected):
        selected.append(n)
print(f'\n候选间去重后 (贪心, |t|降序, corr<0.5): {len(selected)}')
print('\n=== 最终独立候选 ===')
for n in selected:
    t, ic, p = info[n]
    print(f'  {n:<44} |t|={t:.2f} IC={ic:.4f} 周期={p}')

with open('scripts/_final_candidates.txt', 'w', encoding='utf-8') as f:
    for n in selected:
        t, ic, p = info[n]
        f.write(f'{n},{t:.2f},{ic:.4f},{p}\n')


