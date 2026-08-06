"""实验17: 全集搜索更优因子集 (6生产 + 31独立候选 = 37, 允许替换).

用户核心诉求: 生产 6 因子不一定最优, 需在 6+47 全集中寻找更优子集,
而不只是"52中选好再加入6因子".

方向表: 从有效因子库检验报告 t 值符号读取.
方法:
  A. 从空集前向选择 (完全自由, 最多8个) — 看全集中独立选出什么
  B. 从6生产前向 + 允许替换 — 看能否改进
评估: 全段夏普×0.5 + 实盘夏普×0.5
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from exp16_dedup_forward import Runner, NEW25
from exp_core import PROD6

KEPT31 = [
    'intraday_zero_ret_freq_20d', 'intraday_oi_time_centroid_20d',
    'intraday_amihud_vol_ratio_20d', 'intraday_oi_skew_stability_20d',
    'intraday_depth_trend_20d', 'intraday_open_close_volume_ratio_20d',
    'intraday_amihud_trend_20d', 'intraday_price_delay_20d',
    'intraday_session_symmetry_20d', 'intraday_lowest_time_20d',
    'intraday_seat_long_short_seat_ratio_20d', 'intraday_volume_rank_ratio_20d',
    'intraday_volume_time_shape_20d', 'intraday_extreme_freq_balance_20d',
    'intraday_term_vol_ratio_20d', 'intraday_turnover_velocity_20d',
    'intraday_settle_close_basis_20d', 'intraday_settle_drift_20d',
    'intraday_term_slope_20d', 'intraday_oi_log_change_vol_20d',
    'intraday_term_roll_yield_20d', 'intraday_term_vol_spread_20d',
    'intraday_basis_momentum_20d', 'intraday_roll_yield_dualscore_20d',
    'intraday_roll_dualscore_consistency_20d', 'intraday_volume_oi_price_confirm_20d',
    'intraday_ma_count_bullish_20d', 'intraday_rv_compression_breakout_20d',
    'intraday_amihud_resid_vol_20d', 'intraday_amihud_cross_z_20d',
    'intraday_overnight_gap_reaction_20d',
]

# 方向表 (有效因子库 t 值符号)
DIRS = {
    'intraday_jump_intensity_20d': -1, 'intraday_price_peak_count_20d': 1,
    'intraday_realised_skewness_20d': 1, 'intraday_dtws_20d': 1,
    'intraday_drip_stone_20d': -1, 'intraday_peak_ridge_ratio_20d': -1,
    'intraday_zero_ret_freq_20d': -1, 'intraday_oi_time_centroid_20d': -1,
    'intraday_amihud_vol_ratio_20d': 1, 'intraday_oi_skew_stability_20d': -1,
    'intraday_depth_trend_20d': 1, 'intraday_open_close_volume_ratio_20d': -1,
    'intraday_amihud_trend_20d': 1, 'intraday_price_delay_20d': -1,
    'intraday_session_symmetry_20d': -1, 'intraday_lowest_time_20d': 1,
    'intraday_seat_long_short_seat_ratio_20d': 1, 'intraday_volume_rank_ratio_20d': 1,
    'intraday_volume_time_shape_20d': 1, 'intraday_extreme_freq_balance_20d': -1,
    'intraday_term_vol_ratio_20d': 1, 'intraday_turnover_velocity_20d': 1,
    'intraday_settle_close_basis_20d': -1, 'intraday_settle_drift_20d': 1,
    'intraday_term_slope_20d': 1, 'intraday_oi_log_change_vol_20d': -1,
    'intraday_term_roll_yield_20d': 1, 'intraday_term_vol_spread_20d': -1,
    'intraday_basis_momentum_20d': 1, 'intraday_roll_yield_dualscore_20d': -1,
    'intraday_roll_dualscore_consistency_20d': -1,
    'intraday_volume_oi_price_confirm_20d': 1, 'intraday_ma_count_bullish_20d': 1,
    'intraday_rv_compression_breakout_20d': -1,
    'intraday_amihud_resid_vol_20d': 1, 'intraday_amihud_cross_z_20d': 1,
    'intraday_overnight_gap_reaction_20d': 1,
}


def main():
    r = Runner()
    ALL = list(dict.fromkeys(list(PROD6) + KEPT31))
    # 初始化全部 ranks (按方向表)
    for n in ALL:
        if n not in r.ranks and n in r.comp:
            rk = r.comp[n].rank(axis=1, pct=True)
            d = DIRS.get(n, 1)
            r.ranks[n] = rk if d == 1 else (1 - rk)
    print('=' * 70)
    print('实验17: 全集搜索 (6生产 + 31独立 = 37)')
    print('=' * 70)

    def backtest(names):
        # 确保方向 (增量因子可能已由 try_direction 改过, 重置)
        for n in names:
            d = DIRS.get(n, 1)
            rk = r.comp[n].rank(axis=1, pct=True)
            r.ranks[n] = rk if d == 1 else (1 - rk)
        return r.returns(names)

    def score(s):
        sh = s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0
        live = s[s.index > pd.Timestamp('2026-05-15')]
        lsh = live.mean() * 252 / (live.std() * np.sqrt(252)) if len(live) > 2 and live.std() > 0 else 0
        return sh * 0.5 + lsh * 0.5

    # A. 从空集前向选择
    print('\n--- A. 从空集前向选择 (最多8个) ---')
    current = []
    for step in range(4):
        best, best_name, best_s = -1e9, None, None
        for c in ALL:
            if c in current:
                continue
            s = backtest(current + [c])
            sc = score(s)
            if sc > best:
                best, best_name, best_s = sc, c, s
        if best_name is None:
            break
        current.append(best_name)
        sh = best_s.mean() / best_s.std() * np.sqrt(252) if best_s.std() > 0 else 0
        live_sh = score(best_s)
        print(f'  +{best_name:<44} 夏普={sh:.2f} 实盘得分={live_sh:.2f}')
    print(f'空集前向最终 ({len(current)}): {current}')
    st = backtest(current)
    sh = st.mean() / st.std() * np.sqrt(252) if st.std() > 0 else 0
    print(f'  最终: 夏普={sh:.2f} 实盘得分={score(st):.2f}')

    # B. 从6生产前向 + 替换
    print('\n--- B. 6生产为基底, 前向+替换 (最多+6) ---')
    current = list(PROD6)
    base_sc = score(backtest(current))
    print(f'基准 6f: 得分={base_sc:.2f}')
    for step in range(3):
        best_add, best_add_s, best_add_n = base_sc, None, None
        for c in KEPT31:
            if c in current:
                continue
            s = backtest(current + [c])
            sc = score(s)
            if sc > best_add:
                best_add, best_add_s, best_add_n = sc, s, c
        if best_add_n is None or best_add <= base_sc:
            print(f'  无可改进, 停止 (最佳新增 {best_add_n}, 得分 {best_add:.2f} <= 基准 {base_sc:.2f})')
            break
        current.append(best_add_n)
        # 替换: 尝试逐个去掉看是否仍优于 +1 前
        while True:
            improved = False
            for i in range(len(current)):
                trial = current[:i] + current[i+1:]
                s = backtest(trial)
                sc = score(s)
                if sc > score(backtest(current)):
                    removed = current[i]
                    current = trial
                    improved = True
                    break
            if not improved:
                break
        base_sc = score(backtest(current))
        print(f'  +{best_add_n:<44} 得分={base_sc:.2f} ({len(current)}因子)')
    print(f'B 最终 ({len(current)}): {current}')


if __name__ == '__main__':
    main()
