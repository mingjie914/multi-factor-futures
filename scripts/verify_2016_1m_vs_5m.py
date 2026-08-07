"""2016 起 1min vs 5min 一致性验证 (用户核心疑虑).

之前的 1m/5m 对比只做 2024 起; 本脚本验证 2016-03-31 起 (含冷启动前置 2015).
覆盖: 生产 6 因子 + 部分 term 因子.
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
from exp18_light_forward import DIRS

CHECK = list(PROD6) + [
    'intraday_term_slope_20d', 'intraday_term_roll_yield_20d',
    'intraday_basis_momentum_20d', 'intraday_rollover_basis_gap_20d',
]


def main():
    env = ExpEnv(None)
    cal, u = env.cal, env.u
    # 2016-03-31 净值起点, 前置到 2015 冷启动
    sub = cal[cal >= pd.Timestamp('2015-01-01')]
    print(f'样本: {sub[0].date()} ~ {sub[-1].date()} ({len(sub)} 天)', flush=True)

    ID._INTRADAY_FREQ = '1min'
    comp1 = env.engine.compute_factors(CHECK, sub, u, parallel=False)
    ID._INTRADAY_FREQ = '5min'
    comp5 = env.engine.compute_factors(CHECK, sub, u, parallel=False)
    ID._INTRADAY_FREQ = '5min'

    print(f'=== {len(CHECK)} 因子 1min vs 5min (2015 起) ===')
    print(f'{"因子":<40} {"1min NaN":>8} {"5min NaN":>8} {"rel":>7} {"sign":>6} {"判定":>10}')
    for n in CHECK:
        if n not in comp1 or n not in comp5:
            print(f'{n:<40} 计算失败')
            continue
        f1, f5 = comp1[n], comp5[n]
        na1, na5 = f1.isna().mean().mean(), f5.isna().mean().mean()
        common = f1.index.intersection(f5.index)
        if len(common) < 10:
            print(f'{n:<40} 样本不足')
            continue
        diff = (f1.loc[common] - f5.loc[common]).abs().mean().mean()
        scale = f1.loc[common].abs().mean().mean()
        rel = diff / scale if scale > 1e-9 else 0
        sm = (np.sign(f1.loc[common].fillna(0)) == np.sign(f5.loc[common].fillna(0))).mean().mean()
        ok = rel < 0.005 and sm > 0.99 and abs(na5 - na1) < 0.05
        verdict = 'PASS-5min' if ok else '⚠️ NEED-1min'
        print(f'{n:<40} {na1:>7.1%} {na5:>7.1%} {rel:>6.2%} {sm:>5.1%} {verdict:>10}')


if __name__ == '__main__':
    main()
