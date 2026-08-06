"""全量 53 因子 1min vs 5min 覆盖率对比 (修正版 - 补覆盖率维度).

之前的 verify_53_consistency 只比较数值, 未比较覆盖率:
当因子在 5min 下也大量 NaN 时, 对齐比较剩余值会误判"一致",
掩盖覆盖恶化 (drip_stone 案例)。

修正: 对每个因子对比 1min/5min 的 NaN 率, 覆盖率差 > 5% 即需强制 1min.
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
from exp18_light_forward import KEPT47

ALL53 = list(dict.fromkeys(list(PROD6) + KEPT47))


def main():
    env = ExpEnv(None)
    cal, u = env.cal, env.u
    # 用全历史 (2024-01 ~ 2026-08) 保证覆盖率有代表性
    sub_cal = cal

    ID._INTRADAY_FREQ = '1min'
    comp1 = env.engine.compute_factors(ALL53, sub_cal, u, parallel=False)
    ID._INTRADAY_FREQ = '5min'
    comp5 = env.engine.compute_factors(ALL53, sub_cal, u, parallel=False)
    ID._INTRADAY_FREQ = '5min'

    print('=== 53 因子 1min vs 5min 覆盖率对比 (NaN 率) ===')
    print(f'{"因子":<44} {"1min NaN":>9} {"5min NaN":>9} {"Δ":>7} {"判定":>14}')
    need_1min = []
    for n in ALL53:
        if n not in comp1 or n not in comp5:
            print(f'  {n}: 计算失败')
            continue
        f1, f5 = comp1[n], comp5[n]
        na1 = f1.isna().mean().mean()
        na5 = f5.isna().mean().mean()
        delta = na5 - na1
        verdict = 'PASS-5min' if delta < 0.05 else '⚠️ NEED-1min'
        if delta >= 0.05:
            need_1min.append((n, na1, na5))
        print(f'{n:<44} {na1:>8.1%} {na5:>8.1%} {delta:>+6.1%} {verdict:>14}')
    print(f'\n=== 需强制 1min 的因子 ({len(need_1min)}) ===')
    for n, na1, na5 in need_1min:
        print(f'  ⚠️ {n}: 1min NaN={na1:.1%} -> 5min NaN={na5:.1%}')


if __name__ == '__main__':
    main()
