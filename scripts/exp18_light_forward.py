"""实验18: 轻量代理指标前向搜索 (方案一) — 58因子同等候选筛选更优生产集.

核心: 前向搜索阶段不用完整回测 (64s/次), 用 Rank IC 均值/ICIR 作为轻量筛选
指标 (<0.1s/次). 流程:
  1. compute 全部 58 因子一次 (60s, 缓存后同进程 0s)
  2. 对每个候选组合只算 IC 均值/ICIR (DataFrame.corr, 快)
  3. 前向选择: 每步选 ICIR 提升最大的候选 (1275 次 × 0.1s ≈ 2min)
  4. 最终 Top 组合跑完整回测 (64s) 验证
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')
from exp_core import ExpEnv, PROD6

# 58 因子 = 生产6 + 有效候选47 (去重后)
KEPT47 = [  # 从有效因子库第二层 #1-47
    'intraday_zero_ret_freq_20d', 'intraday_open_close_drift_20d',
    'intraday_volatility_clustering_20d', 'intraday_oi_vol_corr_daily_20d',
    'intraday_oi_time_centroid_20d', 'intraday_wash_trade_20d',
    'intraday_settle_position_20d', 'intraday_cross_vol_20d',
    'intraday_amihud_vol_ratio_20d', 'intraday_oi_skew_stability_20d',
    'intraday_depth_trend_20d', 'intraday_open_close_volume_ratio_20d',
    'intraday_oi_quantile_range_20d', 'intraday_settle_gap_20d',
    'intraday_amihud_trend_20d', 'intraday_price_delay_20d',
    'intraday_overnight_absorption_20d', 'intraday_session_symmetry_20d',
    'intraday_lowest_time_20d', 'intraday_oi_peak_ridge_ratio_20d',
    'intraday_seat_long_short_seat_ratio_20d', 'intraday_volume_rank_ratio_20d',
    'intraday_herding_20d', 'intraday_volume_time_shape_20d',
    'intraday_extreme_freq_balance_20d', 'intraday_term_vol_ratio_20d',
    'intraday_turnover_velocity_20d',
    'intraday_settle_close_basis_20d', 'intraday_settle_drift_20d',
    'intraday_settle_vol_ratio_20d', 'intraday_term_slope_20d',
    'intraday_oi_log_change_vol_20d', 'intraday_term_roll_yield_20d',
    'intraday_term_vol_spread_20d', 'intraday_oi_vol_price_corr_20d',
    'intraday_term_spread_vol_20d', 'intraday_rollover_basis_gap_20d',
    'intraday_basis_momentum_20d', 'intraday_roll_yield_dualscore_20d',
    'intraday_roll_dualscore_consistency_20d', 'intraday_amihud_resid_vol_20d',
    'intraday_volume_oi_price_confirm_20d', 'intraday_amihud_cross_z_20d',
    'intraday_overnight_gap_reaction_20d', 'intraday_false_breakout_retrace_20d',
    'intraday_ma_count_bullish_20d', 'intraday_rv_compression_breakout_20d',
]

# 方向表 (有效因子库 t 值符号)
DIRS = {
    'intraday_jump_intensity_20d': -1, 'intraday_price_peak_count_20d': 1,
    'intraday_realised_skewness_20d': 1, 'intraday_dtws_20d': 1,
    'intraday_drip_stone_20d': -1, 'intraday_peak_ridge_ratio_20d': -1,
    'intraday_zero_ret_freq_20d': -1, 'intraday_open_close_drift_20d': 1,
    'intraday_volatility_clustering_20d': -1, 'intraday_oi_vol_corr_daily_20d': 1,
    'intraday_oi_time_centroid_20d': -1, 'intraday_wash_trade_20d': 1,
    'intraday_settle_position_20d': -1, 'intraday_cross_vol_20d': 1,
    'intraday_amihud_vol_ratio_20d': 1, 'intraday_oi_skew_stability_20d': -1,
    'intraday_depth_trend_20d': 1, 'intraday_open_close_volume_ratio_20d': -1,
    'intraday_oi_quantile_range_20d': -1, 'intraday_settle_gap_20d': 1,
    'intraday_amihud_trend_20d': 1, 'intraday_price_delay_20d': -1,
    'intraday_overnight_absorption_20d': 1, 'intraday_session_symmetry_20d': -1,
    'intraday_lowest_time_20d': 1, 'intraday_oi_peak_ridge_ratio_20d': -1,
    'intraday_seat_long_short_seat_ratio_20d': 1, 'intraday_volume_rank_ratio_20d': 1,
    'intraday_herding_20d': 1, 'intraday_volume_time_shape_20d': 1,
    'intraday_extreme_freq_balance_20d': -1, 'intraday_term_vol_ratio_20d': 1,
    'intraday_turnover_velocity_20d': 1,
    'intraday_settle_close_basis_20d': -1, 'intraday_settle_drift_20d': 1,
    'intraday_settle_vol_ratio_20d': 1, 'intraday_term_slope_20d': 1,
    'intraday_oi_log_change_vol_20d': -1, 'intraday_term_roll_yield_20d': 1,
    'intraday_term_vol_spread_20d': -1, 'intraday_oi_vol_price_corr_20d': -1,
    'intraday_term_spread_vol_20d': 1, 'intraday_rollover_basis_gap_20d': 1,
    'intraday_basis_momentum_20d': 1, 'intraday_roll_yield_dualscore_20d': -1,
    'intraday_roll_dualscore_consistency_20d': -1, 'intraday_amihud_resid_vol_20d': 1,
    'intraday_volume_oi_price_confirm_20d': 1, 'intraday_amihud_cross_z_20d': 1,
    'intraday_overnight_gap_reaction_20d': 1, 'intraday_false_breakout_retrace_20d': 1,
    'intraday_ma_count_bullish_20d': 1, 'intraday_rv_compression_breakout_20d': -1,
}


def main():
    t_start = time.time()
    env = ExpEnv(PROD6)
    cal, u = env.cal, env.u
    ALL = list(dict.fromkeys(list(PROD6) + KEPT47))
    print(f'候选池: {len(ALL)} 因子 (6生产 + {len(KEPT47)}候选)', flush=True)

    # 1. compute 一次全部因子 (缓存后同进程快)
    t0 = time.time()
    comp = env.engine.compute_factors(ALL, cal, u, parallel=False)
    print(f'compute {len(ALL)} 因子: {time.time()-t0:.1f}s', flush=True)

    # 2. 各因子 rank (方向调整)
    ranks = {}
    for n in ALL:
        r = comp[n].rank(axis=1, pct=True)
        d = DIRS.get(n, 1)
        ranks[n] = r if d == 1 else (1 - r)
    fwd_rank = env.daily_ret.rank(axis=1)

    # 3. 预计算每个因子与 fwd 的 IC 序列 (一次性, 供组合 IC 快速合成)
    t0 = time.time()
    ic_all = pd.DataFrame({n: ranks[n].corrwith(fwd_rank, axis=1) for n in ALL})
    print(f'IC 预计算 {len(ALL)} 因子: {time.time()-t0:.1f}s', flush=True)

    def light_score(names):
        """轻量代理指标: 组合 IC 均值 × ICIR (用全期 IC 面板)."""
        ic_sub = ic_all[names].mean(axis=1)
        ic_mean = ic_sub.mean()
        ic_std = ic_sub.std(ddof=0)
        icir = ic_mean / ic_std if ic_std > 0 else 0
        return ic_mean, icir, ic_mean * icir

    # 4. 前向选择 (轻量)
    print('\n=== 前向选择 (轻量 Rank IC 指标) ===', flush=True)
    base6 = list(PROD6)
    m0, i0, s0 = light_score(base6)
    print(f'基准 6f: IC={m0:.4f} ICIR={i0:.3f} score={s0:.4f}', flush=True)

    current = list(base6)
    history = [(len(current), s0, m0, i0)]
    for step in range(12):  # 最多加到 18
        best, best_n = -1e9, None
        for c in ALL:
            if c in current:
                continue
            m, i, sc = light_score(current + [c])
            if sc > best:
                best, best_n, best_mi = sc, c, (m, i)
        if best_n is None or best <= s0:
            break
        current.append(best_n)
        s0 = best
        m0, i0 = best_mi
        history.append((len(current), s0, m0, i0))
        print(f'  +{best_n:<44} IC={m0:.4f} ICIR={i0:.3f} score={s0:.4f} ({len(current)}因子)', flush=True)

    print(f'\n轻量前向选择耗时: {time.time()-t_start:.1f}s', flush=True)
    print(f'最终 {len(current)} 因子: {current}', flush=True)

    # 5. 对 Top 组合跑完整回测验证 (只验证前 3 个候选规模)
    print('\n=== 完整回测验证 (Top 组合) ===', flush=True)
    from exp16_dedup_forward import Runner as R16
    r16 = R16()
    # 复用 R16 的 returns (方向表已有)
    for n in ALL:
        if n not in r16.ranks and n in r16.comp:
            rk = r16.comp[n].rank(axis=1, pct=True)
            d = DIRS.get(n, 1)
            r16.ranks[n] = rk if d == 1 else (1 - rk)
    for k in [len(base6), len(current), min(len(current), 12)]:
        if k <= len(base6):
            continue
        names = current[:k]
        s = r16.returns(names)
        sh = s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0
        live = s[s.index > pd.Timestamp('2026-05-15')]
        lsh = live.mean() * 252 / (live.std() * np.sqrt(252)) if len(live) > 2 and live.std() > 0 else 0
        print(f'  {k}因子: 夏普={sh:.2f} 实盘={lsh:.2f}', flush=True)


if __name__ == '__main__':
    main()
