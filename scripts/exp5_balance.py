"""实验5: 因子类别均衡优化.

现状: 33有效因子 波动/流动性8个 vs 席位1个, 等权时类别失衡.
方案对比 (全部用 IC_IR 最优加权 + 池内ERC + cap3 + 日度):
  A. F6-IC_IR (基准, 2.56, 实盘+2.24) — 生产候选
  B. F33-IC_IR (33因子, 2.25异常, 参数灾难) — 排除
  C. F33-类别均衡: 每类别先等权合成类别分, 再 IC_IR 加权类别间 (类别内1票)
  D. F12-类别均衡: 选每类别最强1-2个(共12), 再 IC_IR

类别均衡核心: 先按类别分组等权合成 → 类别分作为"因子" → IC_IR 加权
→ 每类别无论因子多少, 贡献1票 (缓解 波动8票vs席位1票)
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from exp_core import ExpEnv, stats, format_stats, PROD6, CAND27, CAND_DIR

# 类别映射 (33因子)
CATS = {
    '波动流动': ['intraday_volatility_clustering_20d','intraday_oi_vol_corr_daily_20d','intraday_cross_vol_20d',
                'intraday_amihud_vol_ratio_20d','intraday_open_close_volume_ratio_20d','intraday_amihud_trend_20d',
                'intraday_volume_rank_ratio_20d','intraday_term_vol_ratio_20d'],
    '混合其他': ['intraday_zero_ret_freq_20d','intraday_open_close_drift_20d','intraday_price_delay_20d',
                'intraday_overnight_absorption_20d','intraday_session_symmetry_20d','intraday_lowest_time_20d',
                'intraday_extreme_freq_balance_20d'],
    '跳跃峰': ['intraday_jump_intensity_20d','intraday_price_peak_count_20d','intraday_peak_ridge_ratio_20d',
              'intraday_oi_peak_ridge_ratio_20d'],
    '分布偏度': ['intraday_realised_skewness_20d','intraday_dtws_20d','intraday_oi_skew_stability_20d','intraday_herding_20d'],
    'OI持仓': ['intraday_oi_time_centroid_20d','intraday_settle_position_20d','intraday_oi_quantile_range_20d','intraday_settle_gap_20d'],
    '量价': ['intraday_wash_trade_20d','intraday_depth_trend_20d','intraday_turnover_velocity_20d'],
    '频谱路径': ['intraday_drip_stone_20d','intraday_volume_time_shape_20d'],
    '席位': ['intraday_seat_long_short_seat_ratio_20d'],
}


def ledoit_wolf_cov(ic_matrix):
    T, N = ic_matrix.shape
    sample_cov = np.cov(ic_matrix, rowvar=False, ddof=1)
    sample_corr = np.corrcoef(ic_matrix, rowvar=False)
    avg_corr = np.mean(sample_corr[np.triu_indices(N, k=1)]) if N > 1 else 0.0
    target_corr = np.eye(N) * (1 - avg_corr) + np.ones((N, N)) * avg_corr
    std_v = np.std(ic_matrix, axis=0, ddof=1)
    target_cov = np.outer(std_v, std_v) * target_corr
    centered = ic_matrix - ic_matrix.mean(axis=0)
    pi = sum(np.sum((centered.iloc[i].values.reshape(-1, 1) @ centered.iloc[i].values.reshape(1, -1) - sample_cov) ** 2)
             for i in range(T)) / T
    gamma = np.sum((target_cov - sample_cov) ** 2)
    lam = max(0.0, min(1.0, pi / gamma)) if gamma > 0 else 0.5
    return lam * target_cov + (1 - lam) * sample_cov


def main():
    # F33 全部
    F33 = dict(PROD6)
    for n in CAND27:
        F33[n] = CAND_DIR[n]
    env = ExpEnv(F33)
    cal, u, daily_ret = env.cal, env.u, env.daily_ret
    comp = env.engine.compute_factors(list(F33), cal, u, parallel=True)
    ranks = {}
    for n, direction in F33.items():
        r = comp[n].rank(axis=1, pct=True)
        ranks[n] = r if direction == 1 else (1 - r)
    fwd_rank = daily_ret.rank(axis=1)

    def ic_ir_score(factor_names, t_ic_map):
        """对给定因子名列表, 返回滚动IC_IR加权合成的截面得分."""
        names = list(factor_names)
        ic = pd.DataFrame({n: t_ic_map[n] for n in names})
        wmap = {}
        for t in cal:
            hist = ic.loc[:t].iloc[-60:-1]
            if len(hist) < 20:
                wmap[t] = pd.Series(1.0 / len(names), index=names)
                continue
            ic_mean = hist.mean()
            lw_cov = ledoit_wolf_cov(hist)
            try:
                wi = np.linalg.inv(lw_cov) @ ic_mean.values
            except np.linalg.LinAlgError:
                wi = ic_mean.abs().values
            wi = np.abs(wi)
            s = np.sum(wi)
            wmap[t] = pd.Series(np.asarray(wi, dtype=float) / s, index=names)
        sc = pd.DataFrame(index=cal, columns=u, dtype=float)
        for t in cal:
            if t not in wmap:
                continue
            wt = wmap[t]
            row = pd.Series(0.0, index=u)
            for n in names:
                if t in ranks[n].index:
                    row = row.add(ranks[n].loc[t] * wt[n], fill_value=0)
            tot = row.sum()
            if tot > 0:
                sc.loc[t] = row / tot
        return sc

    # 各因子滚动IC序列
    ic_map = {n: ranks[n].corrwith(fwd_rank, axis=1) for n in F33}

    def backtest(sc):
        rets = []
        for t in sc.index:
            row = sc.loc[t].dropna()
            if len(row) < 20:
                continue
            top = env.capped(row, ascending=False)
            bot = env.capped(row, ascending=True)
            wl = env.erc_w(top, t) or {}
            ws = env.erc_w(bot, t) or {}
            if t in daily_ret.index:
                r = daily_ret.loc[t].fillna(0.0)
                lr = sum(r[c] * wi for c, wi in wl.items())
                sr = sum(r[c] * wi for c, wi in ws.items())
                rets.append((t, lr - sr))
        return pd.Series({d: v for d, v in rets}).sort_index().dropna()

    # C. 类别均衡: 每类别先等权合成类别分, 再 IC_IR 加权类别间
    # 步骤1: 类别内等权 (每类别1票)
    cat_scores = {}
    for cat, members in CATS.items():
        sc = pd.DataFrame(index=cal, columns=u, dtype=float)
        valid = [n for n in members if n in F33]
        for n in valid:
            sc = sc.add(ranks[n], fill_value=0)
        cat_scores[cat] = sc.div(len(valid))
    # 步骤2: 类别间 IC_IR 加权
    # 用每类别分的滚动 IC 序列
    cat_ic = {}
    for cat, sc in cat_scores.items():
        cat_ic[cat] = sc.rank(axis=1, pct=True).corrwith(fwd_rank, axis=1)
    # IC_IR 加权类别
    cat_names = list(CATS.keys())
    ic_df = pd.DataFrame({c: cat_ic[c] for c in cat_names})
    wmap = {}
    for t in cal:
        hist = ic_df.loc[:t].iloc[-60:-1]
        if len(hist) < 20:
            wmap[t] = pd.Series(1.0 / len(cat_names), index=cat_names)
            continue
        ic_mean = hist.mean()
        lw_cov = ledoit_wolf_cov(hist)
        try:
            wi = np.linalg.inv(lw_cov) @ ic_mean.values
        except np.linalg.LinAlgError:
            wi = ic_mean.abs().values
        wi = np.abs(wi)
        s = np.sum(wi)
        wmap[t] = pd.Series(np.asarray(wi, dtype=float) / s, index=cat_names)
    scC = pd.DataFrame(index=cal, columns=u, dtype=float)
    for t in cal:
        if t not in wmap:
            continue
        wt = wmap[t]
        row = pd.Series(0.0, index=u)
        for c in cat_names:
            if t in cat_scores[c].index:
                row = row.add(cat_scores[c].loc[t] * wt[c], fill_value=0)
        tot = row.sum()
        if tot > 0:
            scC.loc[t] = row / tot
    sC = backtest(scC)

    # D. F12-类别均衡: 每类别选 IC 最强的 1 个 (8类→8因子), 不够则多选补到12
    top_per_cat = []
    for cat, members in CATS.items():
        valid = [n for n in members if n in F33]
        if not valid:
            continue
        # 按滚动IC均值选最强
        ic_means = {n: ic_map[n].mean() for n in valid}
        best = max(valid, key=lambda n: abs(ic_means[n]))
        top_per_cat.append(best)
    # 补足到12 (按IC均值)
    used = set(top_per_cat)
    rest = sorted([n for n in F33 if n not in used], key=lambda n: -abs(ic_map[n].mean()))
    F12_bal = top_per_cat + rest[:max(0, 12 - len(top_per_cat))]
    print(f'F12类别均衡选股: {len(F12_bal)} 因子')
    # 用这些因子构建类别均衡得分 (类别内可能多选, 每类等权后整体IC_IR)
    # 简化: 用 ic_ir_score 对 F12_bal 全体
    sD = backtest(ic_ir_score(F12_bal, ic_map))

    print('\n=== 实验5: 因子类别均衡 ===')
    print(f'C 类别均衡(F33-8类): {format_stats(stats(sC))}')
    print(f'D 类别均衡(F12精选): {format_stats(stats(sD))}')
    # 对照: 之前 F6-IC_IR 2.56, F33-IC_IR 2.25
    print('\n对照: F6-IC_IR 2.56(实盘+2.24) | F33-IC_IR 2.25(异常)')


if __name__ == '__main__':
    main()
