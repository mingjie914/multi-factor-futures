"""生产基准 (泄漏修复后, 高效向量化版): 6因子 IC_IR 全历史 + 分段.

复用 exp22 Runner (5min数据 + 分年compute), 生产6因子, 严格 T-1信号xT收益.
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from exp_core import PROD6
from exp22_full_pool import Runner as R22, prepare_ic_history


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


def main():
    r = R22(list(PROD6))
    names = list(PROD6)
    print(f'回测: 生产6因子 IC_IR, {len(r.cal)} 天', flush=True)
    ic = r.ic[names]
    rets = []
    skipped = 0
    for t in r.cal:
        hist = ic.loc[:t].iloc[-60:-1]  # 不含 ic[T] (严格无泄漏)
        hist = prepare_ic_history(hist)
        if hist.shape[1] < 2 or len(hist) < 30:
            skipped += 1
            continue
        names_eff = list(hist.columns)
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
            skipped += 1
            continue
        top = r.env.capped(sc, ascending=False, date=t)
        bot = r.env.capped(sc, ascending=True, date=t)
        wl = r.env.erc_w(top, t) or {}
        ws = r.env.erc_w(bot, t) or {}
        if t in r.daily_ret.index:
            rr = r.daily_ret.loc[t].fillna(0.0)
            rets.append((t, sum(rr[c] * wi for c, wi in wl.items()) - sum(rr[c] * wi for c, wi in ws.items())))
    s = pd.Series({d: v for d, v in rets}).sort_index().dropna()
    print(f'交易天数: {len(s)}, 跳过: {skipped}', flush=True)

    def seg(name, sr):
        ann = sr.mean() * 252
        vol = sr.std() * np.sqrt(252)
        nav = (1 + sr).cumprod()
        mdd = (nav / nav.cummax() - 1).min()
        oos = sr[(sr.index >= pd.Timestamp('2026-03-01')) & (sr.index <= pd.Timestamp('2026-05-15'))]
        live = sr[sr.index > pd.Timestamp('2026-05-15')]
        osh = oos.mean() * 252 / (oos.std() * np.sqrt(252)) if len(oos) > 2 and oos.std() > 0 else 0
        lsh = live.mean() * 252 / (live.std() * np.sqrt(252)) if len(live) > 2 and live.std() > 0 else 0
        print(f'{name}: 夏普={ann/vol if vol>0 else 0:.2f} 年化={ann:.1%} 回撤={mdd:.1%} OOS={osh:.2f} 实盘={lsh:.2f} ({len(sr)}天)')

    print('\n=== 生产 6因子 IC_IR (泄漏修复后) ===')
    seg('全段', s)
    for y in range(2016, 2027):
        yr = s[s.index.year == y]
        if len(yr) > 20:
            ann = yr.mean() * 252
            vol = yr.std() * np.sqrt(252)
            print(f'  {y}: 夏普={ann/vol if vol>0 else 0:.2f} 收益={yr.sum():.1%}')
    # 保存净值
    nav = (1 + s).cumprod()
    nav.to_csv('runs/production_benchmark_nav.csv')
    print('\n净值已存 runs/production_benchmark_nav.csv')


if __name__ == '__main__':
    main()
