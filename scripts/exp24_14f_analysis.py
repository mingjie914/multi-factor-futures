"""实验24: 14因子(B1) 修复后分年度/分增量贡献分析.

1. 逐年夏普: 6因子 vs 14因子 (修复后)
2. 8个增量因子各自边际贡献 (从14因子逐个去掉)
3. OOS/实盘稳健性
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from exp_core import ExpEnv, PROD6
from exp23_6v14_nav import R22, B1, prepare_ic_history

INCR8 = B1[6:]  # 8 个增量因子


def lw(icm):
    T, N = icm.shape
    sc = np.cov(icm, rowvar=False, ddof=1)
    corr = np.corrcoef(icm, rowvar=False)
    avg = np.mean(corr[np.triu_indices(N, k=1)]) if N > 1 else 0
    tc = np.eye(N) * (1 - avg) + np.ones((N, N)) * avg
    sv = np.std(icm, axis=0, ddof=1)
    tgt = np.outer(sv, sv) * tc
    c = icm - icm.mean(axis=0)
    pi = sum(np.sum((c.iloc[i].values.reshape(-1, 1) @ c.iloc[i].values.reshape(1, -1) - sc) ** 2) for i in range(T)) / T
    g = np.sum((tgt - sc) ** 2)
    lam = max(0, min(1, pi / g)) if g > 0 else 0.5
    return lam * tgt + (1 - lam) * sc


def bt(r, names):
    ic = r.ic[names]
    rets = []
    for t in r.cal:
        hist = ic.loc[:t].iloc[-60:-1]  # 不含 ic[T] (防同日泄漏)
        hist = prepare_ic_history(hist)
        if hist.shape[1] < 2:
            continue
        names_eff = list(hist.columns)
        if len(hist) < 30:
            w = pd.Series(1.0 / len(names_eff), index=names_eff)
        else:
            im = hist.mean()
            lwc = lw(hist)
            try:
                wi = np.linalg.inv(lwc) @ im.values
            except np.linalg.LinAlgError:
                wi = im.abs().values
            wi = np.abs(wi)
            w = pd.Series(wi / wi.sum(), index=names_eff)
        sc = pd.Series(0.0, index=r.u)
        for n in names_eff:
            if t in r.ranks[n].index:
                sc = sc.add(r.ranks[n].loc[t].fillna(0.0) * w[n], fill_value=0)
        tot = sc.sum()
        if tot > 0:
            sc = sc / tot
        sc = sc.dropna()
        if len(sc) < 20:
            continue
        top = r.env.capped(sc, ascending=False, date=t)
        bot = r.env.capped(sc, ascending=True, date=t)
        wl = r.env.erc_w(top, t) or {}
        ws = r.env.erc_w(bot, t) or {}
        if t in r.daily_ret.index:
            rr = r.daily_ret.loc[t].fillna(0.0)
            rets.append((t, sum(rr[c] * wi for c, wi in wl.items()) - sum(rr[c] * wi for c, wi in ws.items())))
    return pd.Series({d: v for d, v in rets}).sort_index().dropna()


def main():
    r = R22(B1)
    print('=' * 60)
    print('实验24: 14因子修复后分析')
    print('=' * 60)
    s6 = bt(r, list(PROD6))
    s14 = bt(r, B1)

    # 1. 逐年夏普
    print('\n--- 逐年夏普 ---')
    print(f'{"年份":<6} {"6因子":>8} {"14因子":>8} {"差":>8}')
    for y in range(2016, 2027):
        a = s6[s6.index.year == y]
        b = s14[s14.index.year == y]
        if len(a) > 20 and len(b) > 20:
            sh_a = a.mean() * 252 / (a.std() * np.sqrt(252)) if a.std() > 0 else 0
            sh_b = b.mean() * 252 / (b.std() * np.sqrt(252)) if b.std() > 0 else 0
            print(f'{y:<6} {sh_a:>8.2f} {sh_b:>8.2f} {sh_b-sh_a:>+8.2f}')

    # 2. 增量边际贡献 (14因子逐个去掉增量)
    print('\n--- 8增量因子边际贡献 (从14因子去掉) ---')
    base_sh = s14.mean() * 252 / (s14.std() * np.sqrt(252)) if s14.std() > 0 else 0
    print(f'14因子全: 夏普={base_sh:.2f}')
    for n in INCR8:
        sub = [x for x in B1 if x != n]
        s = bt(r, sub)
        sh = s.mean() * 252 / (s.std() * np.sqrt(252)) if s.std() > 0 else 0
        print(f'  去 {n:<44} 夏普={sh:.2f} (Δ={sh-base_sh:+.2f})')

    # 3. OOS/实盘
    print('\n--- OOS/实盘 ---')
    for name, s in [('6因子', s6), ('14因子', s14)]:
        oos = s[(s.index >= pd.Timestamp('2026-03-01')) & (s.index <= pd.Timestamp('2026-05-15'))]
        live = s[s.index > pd.Timestamp('2026-05-15')]
        osh = oos.mean() * 252 / (oos.std() * np.sqrt(252)) if len(oos) > 2 and oos.std() > 0 else 0
        lsh = live.mean() * 252 / (live.std() * np.sqrt(252)) if len(live) > 2 and live.std() > 0 else 0
        print(f'{name}: OOS={osh:.2f} 实盘={lsh:.2f}')


if __name__ == '__main__':
    main()
