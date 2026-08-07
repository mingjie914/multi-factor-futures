"""term 因子 1min vs 5min 一致性验证 (任务1).

验证 ~30 个使用 _get_term_structure_panel 的因子在 5min 下:
  1. 数值: 相对差 < 0.5% 且方向一致 > 99%
  2. 覆盖率: NaN 率差 < 5%
结论: 一致的 term 因子可安全走 5min (吃加速); 不一致的需 force_1min.
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
import warnings
warnings.filterwarnings('ignore')
import re
import numpy as np
import pandas as pd
import factors.library.intraday as ID
from exp_core import ExpEnv

# term 因子注册名 (从类名推导, 匹配 register)
TERM_NAMES = [
    'intraday_term_slope_change_20d', 'intraday_term_spread_vol_20d',
    'intraday_term_oi_ratio_20d', 'intraday_term_slope_ma_cross_20d',
    'intraday_term_vol_spread_20d', 'intraday_term_breakout_20d',
    'intraday_term_reversion_20d', 'intraday_oi_log_change_vol_20d',
    'intraday_term_jump_intensity_20d', 'intraday_term_peak_ridge_ratio_20d',
    'intraday_term_trend_efficiency_20d', 'intraday_term_dtws_20d',
    'intraday_term_vp_corr_20d', 'intraday_term_herding_20d',
    'intraday_term_peak_count_20d', 'intraday_term_range_position_20d',
    'intraday_term_quantile_skew_20d', 'intraday_oi_price_trend_align_20d',
    'intraday_term_position_ratio_20d', 'intraday_oi_vol_corr_daily_20d',
    'intraday_oi_mean_reversion_20d', 'intraday_term_spread_zscore_20d',
    'intraday_oi_vol_corr_change_20d', 'intraday_oi_surge_follow_20d',
    'intraday_vp_corr_stability_20d', 'intraday_basis_reversion_conviction_20d',
    'intraday_roll_yield_dualscore_20d', 'intraday_roll_dualscore_consistency_20d',
    'intraday_cross_contract_spread_z_20d', 'intraday_basis_momentum_20d',
    'intraday_rollover_frequency_20d', 'intraday_days_to_rollover_20d',
    'intraday_term_slope_20d', 'intraday_term_roll_yield_20d',
]


def main():
    env = ExpEnv(None)
    cal, u = env.cal, env.u
    # 用 2024 起 (5min 有完整数据)
    sub = cal[cal >= pd.Timestamp('2024-01-01')]

    ID._INTRADAY_FREQ = '1min'
    comp1 = env.engine.compute_factors(TERM_NAMES, sub, u, parallel=False)
    ID._INTRADAY_FREQ = '5min'
    comp5 = env.engine.compute_factors(TERM_NAMES, sub, u, parallel=False)
    ID._INTRADAY_FREQ = '5min'

    print(f'=== term 因子 1min vs 5min ({len(TERM_NAMES)} 个) ===')
    print(f'{"因子":<42} {"1min NaN":>8} {"5min NaN":>8} {"ΔNaN":>7} {"rel":>7} {"sign":>6} {"判定":>10}')
    need_1min = []
    for n in TERM_NAMES:
        if n not in comp1 or n not in comp5:
            print(f'{n:<42} 计算失败')
            continue
        f1, f5 = comp1[n], comp5[n]
        na1 = f1.isna().mean().mean()
        na5 = f5.isna().mean().mean()
        d_na = na5 - na1
        common = f1.index.intersection(f5.index)
        if len(common) < 10:
            print(f'{n:<42} 样本不足')
            continue
        diff = (f1.loc[common] - f5.loc[common]).abs().mean().mean()
        scale = f1.loc[common].abs().mean().mean()
        rel = diff / scale if scale > 1e-9 else 0
        sm = (np.sign(f1.loc[common].fillna(0)) == np.sign(f5.loc[common].fillna(0))).mean().mean()
        ok = rel < 0.005 and sm > 0.99 and d_na < 0.05
        verdict = 'PASS-5min' if ok else '⚠️ NEED-1min'
        if not ok:
            need_1min.append(n)
        print(f'{n:<42} {na1:>7.1%} {na5:>7.1%} {d_na:>+6.1%} {rel:>6.2%} {sm:>5.1%} {verdict:>10}')
    print(f'\n=== 需 force_1min ({len(need_1min)}) ===')
    for n in need_1min:
        print(f'  ⚠️ {n}')


if __name__ == '__main__':
    main()
