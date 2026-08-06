"""全量 53 因子 (6生产 + 47候选) 1min vs 5min 一致性验证.

目的: 确认全局切 5min 后, 哪些因子数值会漂移 (需标记 force_1min),
哪些完全一致 (可安全用 5min).
准入规则: 相对差 < 0.5% 且方向一致率 > 99% → 可 5min; 否则 → 必须真 1min.
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import factors.library.intraday as ID
from exp_core import ExpEnv, PROD6
from exp18_light_forward import KEPT47, DIRS

ALL53 = list(dict.fromkeys(list(PROD6) + KEPT47))


def main():
    env = ExpEnv(None)
    cal, u = env.cal, env.u
    sub_cal = cal[200:260]  # 60 天样本
    t0 = __import__('time').time()
    # 1min 全量
    ID._INTRADAY_FREQ = '1min'
    comp1 = env.engine.compute_factors(ALL53, sub_cal, u, parallel=False)
    t1 = __import__('time').time()
    print(f'1min compute {len(ALL53)} 因子: {t1-t0:.0f}s', flush=True)
    # 5min 全量
    ID._INTRADAY_FREQ = '5min'
    comp5 = env.engine.compute_factors(ALL53, sub_cal, u, parallel=False)
    t2 = __import__('time').time()
    ID._INTRADAY_FREQ = '5min'
    print(f'5min compute {len(ALL53)} 因子: {t2-t1:.0f}s', flush=True)

    print('\n=== 53 因子 1min vs 5min 一致性 ===')
    safe, drift = [], []
    for n in ALL53:
        if n not in comp1 or n not in comp5:
            print(f'  {n}: 计算失败跳过')
            continue
        f1, f5 = comp1[n], comp5[n]
        common = f1.index.intersection(f5.index)
        if len(common) < 10:
            print(f'  {n}: 样本不足')
            continue
        diff = (f1.loc[common] - f5.loc[common]).abs().mean().mean()
        scale = f1.loc[common].abs().mean().mean()
        rel = diff / scale if scale > 1e-9 else 0
        sm = (np.sign(f1.loc[common].fillna(0)) == np.sign(f5.loc[common].fillna(0))).mean().mean()
        ok = rel < 0.005 and sm > 0.99
        tag = 'SAFE-5min' if ok else 'DRIFT-needs-1min'
        if ok:
            safe.append(n)
        else:
            drift.append(n)
        print(f'  {n:<44} rel={rel:.2%} sign={sm:.1%} -> {tag}')
    print(f'\n=== 汇总 ===')
    print(f'安全用 5min: {len(safe)}/{len(ALL53)}')
    print(f'需真 1min (漂移): {len(drift)}')
    for n in drift:
        print(f'  ⚠️ {n}')


if __name__ == '__main__':
    main()
