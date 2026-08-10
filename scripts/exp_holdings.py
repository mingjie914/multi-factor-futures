"""输出最新持仓: 6因子等权 vs 6因子IC_IR, 含较上日增减列.

持仓 = 最新交易日 (2026-07-31 信号, 用于 08-04 执行). 上多下空, 按权重绝对值降序.
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from exp_core import ExpEnv, PROD6


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


def get_holdings(mode, last_t):
    env = ExpEnv(PROD6)
    cal, u, daily_ret = env.cal, env.u, env.daily_ret
    comp = env.engine.compute_factors(list(PROD6), cal, u, parallel=True)
    ranks = {}
    for n, direction in PROD6.items():
        r = comp[n].rank(axis=1, pct=True)
        ranks[n] = r if direction == 1 else (1 - r)
    fwd_rank = daily_ret.rank(axis=1)
    names = list(PROD6)

    # 构建权重序列 (每日), 取最后两天算增减
    if mode == 'ew':
        wmap = {}
        for t in cal:
            wmap[t] = pd.Series(1.0 / len(names), index=names)
    else:
        ic = pd.DataFrame({n: ranks[n].corrwith(fwd_rank, axis=1) for n in names})
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

    # 找最后两个有有效信号的交易日 (等权得分覆盖)
    ew_score = pd.DataFrame(index=cal, columns=u, dtype=float)
    for n in names:
        ew_score = ew_score.add(ranks[n], fill_value=0)
    ew_score = ew_score.div(len(names))
    valid_days = [t for t in cal if len(ew_score.loc[t].dropna()) > 20]
    if len(valid_days) < 2:
        print(f'{mode}: 有效交易日不足'); return
    t_prev, t_last = valid_days[-2], valid_days[-1]
    print(f'\n=== {mode} 持仓 (信号日 {t_last.date()}, 用于 {t_last + pd.Timedelta(days=1)} 执行) ===')

    def holdings_on(t):
        row = pd.Series(0.0, index=u)
        for n in names:
            if t in ranks[n].index:
                row = row.add(ranks[n].loc[t] * wmap[t][n], fill_value=0)
        tot = row.sum()
        if tot > 0:
            row = row / tot
        # 选池 + ERC
        top = env.capped(row.dropna(), ascending=False)
        bot = env.capped(row.dropna(), ascending=True)
        wl = env.erc_w(top, t) or {}
        ws = env.erc_w(bot, t) or {}
        h = {}
        for sym, w in wl.items():
            h[sym] = w
        for sym, w in ws.items():
            h[sym] = h.get(sym, 0.0) - w
        return pd.Series(h)

    h_prev = holdings_on(t_prev)
    h_last = holdings_on(t_last)
    # 合并: 上多下空, 按权重绝对值降序
    all_syms = list(h_last.index)
    prev_map = h_prev.to_dict()
    rows = []
    for sym in all_syms:
        w = h_last[sym]
        p = prev_map.get(sym, 0.0)
        rows.append({'symbol': sym, 'direction': '多' if w > 0 else '空', 'weight': w, 'prev': p, 'delta': w - p})
    df = pd.DataFrame(rows).sort_values('weight', key=lambda s: s.abs(), ascending=False)
    print(f"{'品种':<5} {'方向':<3} {'权重':>8} {'较上日':>8}")
    print('-' * 35)
    for r in df.itertuples():
        print(f"{r.symbol:<5} {r.direction:<3} {r.weight:>8.4f} {r.delta:>+8.4f}")
    return df


if __name__ == '__main__':
    print('=' * 50)
    print('6因子等权 vs 6因子IC_IR 最新持仓 (含较上日增减)')
    print('=' * 50)
    get_holdings('ew', None)
    get_holdings('icir', None)
