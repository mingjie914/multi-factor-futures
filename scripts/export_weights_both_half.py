"""统一导出: 生产6f等权 vs 12f-IC_IR, 1倍杠杆(half)日度权重, 数据至 2026-08-04.

输出 (gitignored weights/):
  weights/daily_weights_prod_half.csv   生产6f等权 (杠杆1.0)
  weights/daily_weights_12f_icir_half.csv  12f-IC_IR (杠杆1.0)
"""
import sys, os
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from exp_core import ExpEnv, PROD6

F12 = {
    'intraday_jump_intensity_20d': -1,
    'intraday_price_peak_count_20d': +1,
    'intraday_realised_skewness_20d': +1,
    'intraday_dtws_20d': +1,
    'intraday_drip_stone_20d': -1,
    'intraday_peak_ridge_ratio_20d': -1,
    'intraday_oi_time_centroid_20d': -1,
    'intraday_wash_trade_20d': -1,
    'intraday_price_delay_20d': -1,
    'intraday_amihud_trend_20d': +1,
    'intraday_amihud_vol_ratio_20d': +1,
    'intraday_volume_time_shape_20d': +1,
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


def run(F, mode, label):
    env = ExpEnv(F)
    cal, u, daily_ret = env.cal, env.u, env.daily_ret
    names = list(F)
    comp = {}
    for i in range(0, len(names), 10):
        part = env.engine.compute_factors(names[i:i+10], cal, u, parallel=False)
        for k, v in part.items():
            comp[k] = v
    ranks = {}
    for n, direction in F.items():
        if n not in comp:
            print(f'[{label}] skip {n}')
            continue
        r = comp[n].rank(axis=1, pct=True)
        ranks[n] = r if direction == 1 else (1 - r)
    fwd_rank = daily_ret.rank(axis=1)
    if mode == 'icir':
        ic = pd.DataFrame({n: ranks[n].corrwith(fwd_rank, axis=1) for n in names})

    rows = []
    for t in cal:
        sc = pd.Series(0.0, index=u)
        if mode == 'icir':
            hist = ic.loc[:t].iloc[-60:]
            if len(hist) < 20:
                continue
            ic_mean = hist.mean()
            lw_cov = ledoit_wolf_cov(hist)
            try:
                wi = np.linalg.inv(lw_cov) @ ic_mean.values
            except np.linalg.LinAlgError:
                wi = ic_mean.abs().values
            wi = np.abs(wi) / np.abs(wi).sum()
            for n in names:
                if t in ranks[n].index:
                    sc = sc.add(ranks[n].loc[t] * wi[names.index(n)], fill_value=0)
        else:
            for n in names:
                if t in ranks[n].index:
                    sc = sc.add(ranks[n].loc[t] / len(names), fill_value=0)
        tot = sc.sum()
        if tot > 0:
            sc = sc / tot
        sc = sc.dropna()
        if len(sc) < 20:
            continue
        top = env.capped(sc, ascending=False)
        bot = env.capped(sc, ascending=True)
        wl = env.erc_w(top, t)
        ws = env.erc_w(bot, t)
        wl = wl if wl is not None else pd.Series(1.0 / len(top), index=top)
        ws = ws if ws is not None else pd.Series(1.0 / len(bot), index=bot)
        for sym, w in wl.items():
            rows.append({'date': t.strftime('%Y-%m-%d'), 'symbol': sym, 'weight': w * 0.5})
        for sym, w in ws.items():
            rows.append({'date': t.strftime('%Y-%m-%d'), 'symbol': sym, 'weight': -w * 0.5})
    df = pd.DataFrame(rows)
    return df


def main():
    os.makedirs('weights', exist_ok=True)
    # 生产 6f 等权
    df6 = run(PROD6, 'ew', 'prod6')
    df6.to_csv('weights/daily_weights_prod_half.csv', index=False, encoding='utf-8')
    print(f'生产6f等权 half: {len(df6)} 行, {df6.date.nunique()} 交易日, {df6.date.min()}~{df6.date.max()}')
    # 12f IC_IR
    df12 = run(F12, 'icir', '12f')
    df12.to_csv('weights/daily_weights_12f_icir_half.csv', index=False, encoding='utf-8')
    print(f'12f-IC_IR half: {len(df12)} 行, {df12.date.nunique()} 交易日, {df12.date.min()}~{df12.date.max()}')


if __name__ == '__main__':
    main()
