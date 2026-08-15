"""实验25: 13因子 (14 - ma_count_bullish) 历史候选验证.

对比: 6因子(生产) vs 13因子 vs 14因子(B1).
评估: 全段/分年度/OOS/实盘/边际(13因子逐个去增量).
13f 继承了 exp22 的6f锚定和完整历史前向选择，仅是固定候选，不证明13个因子最优.
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

F6 = list(PROD6)
F13 = [x for x in B1 if x != 'intraday_ma_count_bullish_20d']
F14 = B1


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
    r = R22(F14)
    print('=' * 60)
    print('实验25: 13因子 (14 - ma_count_bullish) 验证')
    print('=' * 60)
    s6 = bt(r, F6)
    s13 = bt(r, F13)
    s14 = bt(r, F14)

    def seg(s):
        st = {}
        st['full'] = s.mean() * 252 / (s.std() * np.sqrt(252)) if s.std() > 0 else 0
        oos = s[(s.index >= pd.Timestamp('2026-03-01')) & (s.index <= pd.Timestamp('2026-05-15'))]
        live = s[s.index > pd.Timestamp('2026-05-15')]
        st['oos'] = oos.mean() * 252 / (oos.std() * np.sqrt(252)) if len(oos) > 2 and oos.std() > 0 else 0
        st['live'] = live.mean() * 252 / (live.std() * np.sqrt(252)) if len(live) > 2 and live.std() > 0 else 0
        mdd = ((1 + s).cumprod() / (1 + s).cumprod().cummax() - 1).min()
        st['mdd'] = mdd
        return st

    print('\n--- 全段/OOS/实盘 ---')
    print(f'{"方案":<12} {"全段夏普":>8} {"回撤":>8} {"OOS":>6} {"实盘":>6}')
    for name, s in [('6因子', s6), ('13因子', s13), ('14因子', s14)]:
        st = seg(s)
        print(f'{name:<12} {st["full"]:>8.2f} {st["mdd"]:>7.1%} {st["oos"]:>6.2f} {st["live"]:>6.2f}')

    # 分年度
    print('\n--- 逐年夏普 ---')
    print(f'{"年份":<6} {"6因子":>8} {"13因子":>8} {"14因子":>8}')
    for y in range(2016, 2027):
        row = []
        for s in [s6, s13, s14]:
            yr = s[s.index.year == y]
            row.append(yr.mean() * 252 / (yr.std() * np.sqrt(252)) if len(yr) > 20 and yr.std() > 0 else 0)
        print(f'{y:<6} {row[0]:>8.2f} {row[1]:>8.2f} {row[2]:>8.2f}')

    # 13因子边际 (去掉的7个增量各自)
    print('\n--- 13因子边际 (逐个去增量) ---')
    base = seg(s13)['full']
    INCR7 = [x for x in F13 if x not in F6]
    for n in INCR7:
        sub = [x for x in F13 if x != n]
        s = bt(r, sub)
        sh = s.mean() * 252 / (s.std() * np.sqrt(252)) if s.std() > 0 else 0
        print(f'  去 {n:<44} 夏普={sh:.2f} (Δ={sh-base:+.2f})')

    # 保存净值数据 (核对横线)
    s6.to_csv('runs/exp25_s6.csv')
    s13.to_csv('runs/exp25_s13.csv')
    s14.to_csv('runs/exp25_s14.csv')

    # 净值图 (6/13/14 三方案 + 年化) - main 内
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(13, 7))
    for name, s in [('6因子', s6), ('13因子', s13), ('14因子', s14)]:
        st = seg(s)
        ann = s.mean() * 252
        s_plot = s[s.index >= pd.Timestamp('2016-03-31')]
        nav = (1 + s_plot).cumprod()
        ax.plot(nav.index, nav.values, lw=1.6,
                label=f'{name} (夏普{st["full"]:.2f}/年化{ann:.1%}/实盘{st["live"]:.2f})')
    ax.axvline(pd.Timestamp('2026-03-01'), color='gray', ls='--', lw=1, label='OOS起点')
    ax.axvline(pd.Timestamp('2026-05-16'), color='red', ls='--', lw=1, label='实盘起点')
    ax.set_title('6/13/14因子 净值对比 (2016-03-31 起, 修复泄漏后)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = 'runs/nav_6f_13f_14f.png'
    fig.savefig(out, dpi=150)
    print(f'净值对比图: {out}')


if __name__ == '__main__':
    main()
