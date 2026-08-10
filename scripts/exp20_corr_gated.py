"""实验20: 带相关性门槛的轻量前向选择 (修正 exp18 的缺陷).

exp19 教训: 轻量前向选择未过滤高相关候选 (wash_trade 0.96/cross_vol 0.81),
选中同族因子导致跨样本不稳. 

修正: 每步只允许加入与**所有已选因子** corr<0.5 的候选.
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from exp18_light_forward import DIRS, KEPT47
from exp_core import ExpEnv, PROD6

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
    ic_all = pd.DataFrame({n: ranks[n].corrwith(fwd, axis=1) for n in ALL})

    def corr(a, b):
        return ic_all[a].corr(ic_all[b])

    def light_score(names):
        ic_sub = ic_all[names].mean(axis=1)
        ic_mean = ic_sub.mean()
        ic_std = ic_sub.std(ddof=0)
        icir = ic_mean / ic_std if ic_std > 0 else 0
        return ic_mean * icir

    def full_returns(names):
        ic = ic_all[names]
        rets = []
        for t in cal:
            hist = ic.loc[:t].iloc[-60:-1]
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

    print('='*60)
    print('实验20: 相关性门槛轻量前向选择 (corr<0.5)')
    print('='*60)
    current = list(F6)
    base_sc = light_score(current)
    print(f'基准 6f: score={base_sc:.4f}')
    chosen = []
    for step in range(10):
        best, best_n = -1e9, None
        for c in ALL:
            if c in current:
                continue
            # 相关性门槛: 与所有已选因子 corr<0.5
            if max(abs(corr(c, s)) for s in current) >= 0.5:
                continue
            sc = light_score(current + [c])
            if sc > best:
                best, best_n = sc, c
        if best_n is None or best <= base_sc:
            print(f'  无合格候选 (score={best:.4f} <= 基准 {base_sc:.4f}), 停止')
            break
        current.append(best_n)
        chosen.append(best_n)
        base_sc = best
        print(f'  +{best_n:<44} score={best:.4f} ({len(current)}因子)', flush=True)
    print(f'\n最终 {len(current)} 因子: {current}')

    # 完整回测验证 (每阶段)
    print('\n=== 完整回测验证 ===')
    s6 = full_returns(F6)
    sh6 = s6.mean()/s6.std()*np.sqrt(252) if s6.std()>0 else 0
    live6 = s6[s6.index > pd.Timestamp('2026-05-15')]
    lsh6 = live6.mean()*252/(live6.std()*np.sqrt(252)) if len(live6)>2 and live6.std()>0 else 0
    print(f'6因子: 夏普={sh6:.2f} 实盘={lsh6:.2f}')
    for k in [8, 10, 12, len(current)]:
        names = current[:k]
        s = full_returns(names)
        sh = s.mean()/s.std()*np.sqrt(252) if s.std()>0 else 0
        live = s[s.index > pd.Timestamp('2026-05-15')]
        lsh = live.mean()*252/(live.std()*np.sqrt(252)) if len(live)>2 and live.std()>0 else 0
        print(f'{k}因子: 夏普={sh:.2f} 实盘={lsh:.2f}')


if __name__ == '__main__':
    main()
