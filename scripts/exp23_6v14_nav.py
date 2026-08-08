"""实验23: 6因子 vs 14因子(B1) 净值对比图.

B1 = exp22 从6出发前向选的 14 因子 (6生产 + ma_count_bullish/torrent_down/
lowest_time/term_slope/open_close_volume_ratio/seat_long_short_seat_ratio/
turnover_velocity/price_delay).
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
from exp_core import ExpEnv, stats, PROD6
from exp22_full_pool import Runner as R22, NEW21_DIR
from exp18_light_forward import DIRS

B1 = list(PROD6) + [
    'intraday_ma_count_bullish_20d', 'intraday_torrent_down_20d',
    'intraday_lowest_time_20d', 'intraday_term_slope_20d',
    'intraday_open_close_volume_ratio_20d', 'intraday_seat_long_short_seat_ratio_20d',
    'intraday_turnover_velocity_20d', 'intraday_price_delay_20d',
]


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
    r = R22()
    print('=' * 60)
    print('实验23: 6因子 vs 14因子(B1) 净值对比')
    print('=' * 60)
    fig, ax = plt.subplots(figsize=(13, 7))
    results = {}
    for name, names in [('6因子 (生产)', list(PROD6)), ('14因子 (B1 前向)', B1)]:
        ic = r.ic[names]
        rets = []
        for t in r.cal:
            hist = ic.loc[:t].iloc[-60:-1]  # 不含 ic[T] (防同日泄漏)
            # 剔除 IC 全 NaN 的因子 (该段无信号, 如 seat 2020 前)
            hist = hist.dropna(axis=1, how='all')
            if hist.shape[1] < 2:
                continue
            if len(hist) < 30:
                w = pd.Series(1.0/hist.shape[1], index=hist.columns)
            else:
                im = hist.mean()
                lwc = lw(hist)
                try:
                    wi = np.linalg.inv(lwc) @ im.values
                except np.linalg.LinAlgError:
                    wi = im.abs().values
                wi = np.abs(wi)
                w = pd.Series(wi/wi.sum(), index=hist.columns)
            names_eff = list(hist.columns)
            sc = pd.Series(0.0, index=r.u)
            for n in names_eff:
                if t in r.ranks[n].index:
                    sc = sc.add(r.ranks[n].loc[t].fillna(0.0)*w[n], fill_value=0)  # 因子缺失不污染其他
            tot = sc.sum()
            if tot > 0:
                sc = sc/tot
            sc = sc.dropna()
            if len(sc) < 20:
                continue
            top = r.env.capped(sc, ascending=False)
            bot = r.env.capped(sc, ascending=True)
            wl = r.env.erc_w(top, t) or {}
            ws = r.env.erc_w(bot, t) or {}
            if t in r.daily_ret.index:
                rr = r.daily_ret.loc[t].fillna(0.0)
                rets.append((t, sum(rr[c]*wi for c, wi in wl.items()) - sum(rr[c]*wi for c, wi in ws.items())))
        s = pd.Series({d: v for d, v in rets}).sort_index().dropna()
        st = stats(s)
        results[name] = (s, st)
        # 净值图从 2016-03-31 起
        s_plot = s[s.index >= pd.Timestamp('2016-03-31')]
        nav = (1 + s_plot).cumprod()
        ax.plot(nav.index, nav.values, lw=1.6,
                label=f'{name} (全段夏普{st["sharpe"]:.2f}/实盘{st["live"]:.2f})')
        print(f'{name}: 夏普={st["sharpe"]:.2f} 年化={st["ann"]:.1%} 回撤={st["mdd"]:.1%} '
              f'OOS={st["oos"]:.2f} 实盘={st["live"]:.2f} ({len(s)}天)')
    ax.axvline(pd.Timestamp('2026-03-01'), color='gray', ls='--', lw=1, label='OOS起点')
    ax.axvline(pd.Timestamp('2026-05-16'), color='red', ls='--', lw=1, label='实盘起点')
    ax.set_title('6因子 vs 14因子(B1) 净值对比 (2016-03-31 起)')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = 'runs/nav_6f_vs_14f.png'
    fig.savefig(out, dpi=150)
    print(f'\n净值对比图: {out}')


if __name__ == '__main__':
    main()
