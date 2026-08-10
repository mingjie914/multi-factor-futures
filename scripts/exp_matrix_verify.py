"""验证 F6-LW-ICIR / F6-IC_IR 的真实性与稳定性 (逐年/逐月/换手)."""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from exp_core import ExpEnv, stats, format_stats, PROD6


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


def run(F, mode):
    env = ExpEnv(F)
    cal, u, daily_ret = env.cal, env.u, env.daily_ret
    comp = env.engine.compute_factors(list(F), cal, u, parallel=True)
    ranks = {}
    for n, direction in F.items():
        r = comp[n].rank(axis=1, pct=True)
        ranks[n] = r if direction == 1 else (1 - r)
    fwd_rank = daily_ret.rank(axis=1)
    names = list(F)
    ic = pd.DataFrame({n: ranks[n].corrwith(fwd_rank, axis=1) for n in names})
    wmap = {}
    for t in cal:
        hist = ic.loc[:t].iloc[-60:-1]
        if len(hist) < 20:
            wmap[t] = pd.Series(1.0 / len(names), index=names)
            continue
        ic_mean = hist.mean()
        lw_cov = ledoit_wolf_cov(hist)
        if mode == 'lw':
            ic_std = np.sqrt(np.diag(lw_cov)).clip(min=1e-8)
            wi = (ic_mean.abs() / ic_std).values
        else:
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


F6 = dict(PROD6)
print('=== F6 稳定性验证 (修复后) ===')
for mode, lab in [('lw', 'F6-LW-ICIR'), ('opt', 'F6-IC_IR')]:
    s = run(F6, mode)
    st = stats(s)
    print(f'{lab}: {format_stats(st)}')
    # 逐年
    for y in [2025, 2026]:
        sy = s[s.index.year == y]
        if len(sy) > 2:
            a = sy.mean() * 252
            v = sy.std(ddof=0) * np.sqrt(252)
            print(f'  {y}: 年化={a:.1%} 夏普={a/v if v > 0 else 0:.2f} n={len(sy)}')
    # 逐月 (2026)
    m26 = s[s.index.year == 2026].resample('ME').apply(lambda x: (1 + x).prod() - 1)
    mm = ['{}月={:+.1%}'.format(d.month, v) for d, v in m26.items()]
    print(f'  2026逐月: {mm}')
