"""对比 F33-LW vs F6-IC_IR 的实盘段(5/16后)表现."""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from exp_core import ExpEnv, stats, PROD6, CAND27, CAND_DIR


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

    if mode == 'ew':
        sc = pd.DataFrame(index=cal, columns=u, dtype=float)
        for n in names:
            sc = sc.add(ranks[n], fill_value=0)
        sc = sc.div(len(names))
    else:
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
F33 = dict(PROD6)
for n in CAND27:
    F33[n] = CAND_DIR[n]

print('=== 实盘段 (5/16后) 对比 ===')
for F, fn, lab in [(F6, 'F6', 'F6'), (F33, 'F33', 'F33')]:
    for mode in ('lw', 'opt'):
        s = run(F, mode)
        live = s[s.index > pd.Timestamp('2026-05-15')]
        oos = s[(s.index >= pd.Timestamp('2026-03-01')) & (s.index <= pd.Timestamp('2026-05-15'))]
        a_l = live.mean() * 252
        v_l = live.std(ddof=0) * np.sqrt(252)
        a_o = oos.mean() * 252
        v_o = oos.std(ddof=0) * np.sqrt(252)
        print(f'{fn}-{mode}: OOS夏普={a_o/v_o if v_o>0 else 0:.2f}(n={len(oos)}), 实盘夏普={a_l/v_l if v_l>0 else 0:.2f}(n={len(live)}), 实盘年化={a_l:.1%}')
        # 实盘逐月
        if len(live) > 0:
            m = live.resample('ME').apply(lambda x: (1 + x).prod() - 1)
            mm = ['{}月={:+.2%}'.format(d.month, v) for d, v in m.items()]
            print('   实盘逐月: ' + str(mm))
