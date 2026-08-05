"""4方案对比净值图: F6-等权 vs F6-LW-ICIR vs F6-IC_IR vs F33-IC_IR."""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from exp_core import ExpEnv, stats, format_stats, PROD6, CAND27, CAND_DIR


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
            hist = ic.loc[:t].iloc[-60:]
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

plans = [
    ('F6-等权(当前生产)', F6, 'ew', '#95a5a6', 1.6),
    ('F6-LW-ICIR', F6, 'lw', '#e67e22', 1.8),
    ('F6-IC_IR(最稳健)', F6, 'opt', '#2ecc71', 2.4),
    ('F33-IC_IR(参数灾难)', F33, 'opt', '#c0392b', 1.4),
]
fig, ax = plt.subplots(figsize=(16, 9))
print('=== 4方案对比 ===')
print(f'{"方案":<24} {"年化":>6} {"夏普":>5} {"回撤":>6} {"OOS":>5} {"实盘":>5}')
print('-' * 65)
for label, F, mode, color, lw in plans:
    s = run(F, mode)
    st = stats(s)
    print(f'{label:<24} {st["ann"]:>5.1%} {st["sharpe"]:>5.2f} {st["mdd"]:>5.1%} {st["oos"]:>5.2f} {st["live"]:>5.2f}')
    nav = (1 + s).cumprod()
    ax.plot(nav.index, nav.values, label=label, color=color, linewidth=lw)
ax.axvline(pd.Timestamp('2026-03-01'), color='gray', ls='--', alpha=0.8, label='OOS起点')
ax.axvline(pd.Timestamp('2026-05-16'), color='red', ls='--', alpha=0.8, label='实盘起点')
ax.set_title('4方案对比: F6-等权 vs F6-LW-ICIR vs F6-IC_IR vs F33-IC_IR (修复后)')
ax.set_ylabel('净值')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('runs/exp_4plans_nav.png', dpi=150)
print('\n净值图: runs/exp_4plans_nav.png')
