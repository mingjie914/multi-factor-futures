"""21 候选因子 1min vs 5min 一致性验证 (决定是否需要 1min 重检).

背景: 92 新因子用 5min 完成 IC 检验 (21 通过 FDR). 本脚本验证这 21 个
在 1min vs 5min 下数值/覆盖率是否一致.
- 一致: 5min 检验结果可靠, 无需 1min 重检
- 不一致: 该因子 5min 失真, 需 force_1min 或 1min 重检
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

CAND21 = [
    'intraday_lead_amount_surge_20d', 'intraday_price_diff_autocorr_20d',
    'intraday_diff_autocorr_long_20d', 'intraday_order_flow_autocorr_20d',
    'intraday_vp_corr_high_freq_20d', 'intraday_peak_moment_count_20d',
    'intraday_order_flow_memory_20d', 'intraday_jump_ret_follow_ratio_20d',
    'intraday_neg_ret_illiq_20d', 'intraday_ret_extreme_magnitude_20d',
    'intraday_peak_ridge_coherence_20d', 'intraday_smart_money_v4_vol_20d',
    'intraday_torrent_down_20d', 'intraday_jump_amount_lagcorr_20d',
    'intraday_csad_sigma120_20d', 'intraday_vol_bucket_entropy_20d',
    'intraday_trajectory_illiq_20d', 'intraday_vol_flow_vol_20d',
    'intraday_price_peak_interval_std_20d', 'intraday_following_price_confirm_20d',
    'intraday_flow_ret_resid_vol_20d',
]


def main():
    env = ExpEnv(None)
    cal, u = env.cal, env.u
    sub = cal[cal >= pd.Timestamp('2024-01-01')]  # 2024 起 (5min 覆盖全)

    ID._INTRADAY_FREQ = '1min'
    comp1 = env.engine.compute_factors(CAND21, sub, u, parallel=False)
    ID._INTRADAY_FREQ = '5min'
    comp5 = env.engine.compute_factors(CAND21, sub, u, parallel=False)
    ID._INTRADAY_FREQ = '5min'

    print(f'=== 21 候选 1min vs 5min ({len(CAND21)}) ===')
    print(f'{"因子":<42} {"1min NaN":>8} {"5min NaN":>8} {"rel":>7} {"sign":>6} {"判定":>10}')
    need = []
    for n in CAND21:
        if n not in comp1 or n not in comp5:
            print(f'{n:<42} 计算失败')
            continue
        f1, f5 = comp1[n], comp5[n]
        na1, na5 = f1.isna().mean().mean(), f5.isna().mean().mean()
        common = f1.index.intersection(f5.index)
        if len(common) < 10:
            continue
        diff = (f1.loc[common] - f5.loc[common]).abs().mean().mean()
        scale = f1.loc[common].abs().mean().mean()
        rel = diff / scale if scale > 1e-9 else 0
        sm = (np.sign(f1.loc[common].fillna(0)) == np.sign(f5.loc[common].fillna(0))).mean().mean()
        ok = rel < 0.01 and sm > 0.98 and abs(na5 - na1) < 0.05
        verdict = 'PASS-5min' if ok else 'NEED-1min'
        if not ok:
            need.append(n)
        print(f'{n:<42} {na1:>7.1%} {na5:>7.1%} {rel:>6.2%} {sm:>5.1%} {verdict:>10}')
    print(f'\n需 1min 重检/force_1min: {len(need)}')
    for n in need:
        print(f'  ⚠️ {n}')
    if not need:
        print('全部 PASS: 5min 检验结果可靠, 无需 1min 重检')


if __name__ == '__main__':
    main()
