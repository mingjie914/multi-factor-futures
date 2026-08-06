"""实验19: 12因子候选 (ICIR 前向选择) 稳健性验证.

对比: 生产 6f-IC_IR vs 12f-IC_IR (exp18 前向选择的前 12 个).
验证: 全段/逐年/逐月/2024起样本/OOS/实盘.
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from exp18_light_forward import DIRS, KEPT47
from exp_core import ExpEnv, stats, PROD6

F12 = list(PROD6) + [
    'intraday_ma_count_bullish_20d', 'intraday_cross_vol_20d',
    'intraday_session_symmetry_20d', 'intraday_wash_trade_20d',
    'intraday_basis_momentum_20d', 'intraday_lowest_time_20d',
]
F6 = list(PROD6)


def lw(icm):
    T, N = icm.shape
    sc = np.cov(icm, rowvar=False, ddof=1)
    corr = np.corrcoef(icm, rowvar=False)
    avg = np.mean(corr[np.triu_indices(N, k=1)]) if N > 1 else 0
    tc = np.eye(N)*(1-avg) + np.ones((N,N))*avg
    sv = np.std(icm, axis=0, ddof=1)
    tgt = np.outer(sv, sv)*tc
    c = icm - icm.mean(axis=0)
    pi = sum(np.sum((c.iloc[i].values.reshape(-1,1) @ c.iloc[i].values.reshape(1,-1) - sc)**2) for i in range(T))/T
    g = np.sum((tgt - sc)**2)
    lam = max(0, min(1, pi/g)) if g > 0 else 0.5
    return lam*tgt + (1-lam)*sc


def main():
    env = ExpEnv(PROD6)
    cal, u, daily_ret = env.cal, env.u, env.daily_ret
    ALL = list(dict.fromkeys(list(PROD6) + KEPT47))
    comp = env.engine.compute_factors(ALL, cal, u, parallel=False)
    ranks = {}
    for n in ALL:
        r = comp[n].rank(axis=1, pct=True)
        d = DIRS.get(n, 1)
        ranks[n] = r if d == 1 else (1 - r)
    fwd = daily_ret.rank(axis=1)

    def returns(names):
        ic = pd.DataFrame({n: ranks[n].corrwith(fwd, axis=1) for n in names})
        rets = []
        for t in cal:
            hist = ic.loc[:t].iloc[-60:]
            if len(hist) < 30: continue
            im = hist.mean()
            lwc = lw(hist)
            try:
                wi = np.linalg.inv(lwc) @ im.values
            except np.linalg.LinAlgError:
                wi = im.abs().values
            wi = np.abs(wi); wi = wi/wi.sum()
            sc = pd.Series(0.0, index=u)
            for n in names:
                if t in ranks[n].index:
                    sc = sc.add(ranks[n].loc[t]*wi[names.index(n)], fill_value=0)
            tot = sc.sum()
            if tot > 0: sc = sc/tot
            sc = sc.dropna()
            if len(sc) < 20: continue
            top = env.capped(sc, ascending=False)
            bot = env.capped(sc, ascending=True)
            wl = env.erc_w(top, t) or {}
            ws = env.erc_w(bot, t) or {}
            if t in daily_ret.index:
                rr = daily_ret.loc[t].fillna(0.0)
                rets.append((t, sum(rr[c]*wi for c, wi in wl.items()) - sum(rr[c]*wi for c, wi in ws.items())))
        return pd.Series({d:v for d,v in rets}).sort_index().dropna()

    def report(name, s):
        st = stats(s)
        print(f'{name}: 夏普={st["sharpe"]:.2f} 年化={st["ann"]:.1%} 回撤={st["mdd"]:.1%} OOS={st["oos"]:.2f} 实盘={st["live"]:.2f}')
        for y, grp in s.groupby(s.index.year):
            sh = grp.mean()/grp.std()*np.sqrt(252) if grp.std()>0 else 0
            print(f'    {y}: 收益={grp.sum():.1%} 夏普={sh:.2f}')
        mos = s.groupby([s.index.year, s.index.month]).sum()
        print(f'    负月: {(mos<0).sum()}/{len(mos)} 最差月={mos.min():.2%}')

    print('='*60)
    print('实验19: 12因子候选稳健性验证')
    print('='*60)
    s6 = returns(F6)
    s12 = returns(F12)
    report('6因子 (生产)', s6)
    print()
    report('12因子 (前向候选)', s12)
    # 相关性检查: 12因子新增的6个与生产6是否正交 (用 IC 序列相关)
    print('\n=== 新增6因子 vs 生产6因子 相关 (应<0.5) ===')
    ic_all = pd.DataFrame({n: ranks[n].corrwith(fwd, axis=1) for n in ALL})
    for n in F12[6:]:
        maxc = max(abs(ic_all[n].corr(ic_all[p])) for p in F6)
        print(f'  {n}: max_corr(生产6)={maxc:.2f}')


if __name__ == '__main__':
    main()
