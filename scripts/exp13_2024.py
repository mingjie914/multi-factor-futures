"""实验13: 6f vs 12f 回测 (2024-01-01 至今, 数据至 2026-08-04).

仅跑一下看看延长样本后的表现. OOS/实盘区间规则不变:
  OOS = 2026-03-01 ~ 2026-05-15, 实盘 = 2026-05-16+
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from exp_core import ExpEnv, stats, PROD6, CAND27, CAND_DIR

F12 = ['intraday_jump_intensity_20d', 'intraday_price_peak_count_20d',
       'intraday_realised_skewness_20d', 'intraday_dtws_20d',
       'intraday_drip_stone_20d', 'intraday_peak_ridge_ratio_20d',
       'intraday_oi_time_centroid_20d', 'intraday_wash_trade_20d',
       'intraday_price_delay_20d', 'intraday_amihud_trend_20d',
       'intraday_amihud_vol_ratio_20d', 'intraday_volume_time_shape_20d']
F6 = list(PROD6)


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


class Runner:
    def __init__(self):
        self.env = ExpEnv(PROD6)
        self.cal, self.u, self.daily_ret = self.env.cal, self.env.u, self.env.daily_ret
        all_names = list(dict.fromkeys(F12 + F6))
        all_F = dict(PROD6)
        for n in CAND27:
            all_F[n] = CAND_DIR[n]
        self.all_F = all_F
        self.comp = {}
        for i in range(0, len(all_names), 10):
            part = self.env.engine.compute_factors(all_names[i:i+10], self.cal, self.u, parallel=False)
            for k, v in part.items():
                self.comp[k] = v
        self.ranks = {}
        for n, direction in all_F.items():
            if n not in self.comp:
                continue
            r = self.comp[n].rank(axis=1, pct=True)
            self.ranks[n] = r if direction == 1 else (1 - r)
        self.fwd_rank = self.daily_ret.rank(axis=1)

    def returns(self, names, window=60):
        ic = pd.DataFrame({n: self.ranks[n].corrwith(self.fwd_rank, axis=1) for n in names})
        wmap = {}
        for t in self.cal:
            hist = ic.loc[:t].iloc[-window:]
            if len(hist) < max(10, window // 2):
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
        sc = pd.DataFrame(index=self.cal, columns=self.u, dtype=float)
        for t in self.cal:
            if t not in wmap:
                continue
            wt = wmap[t]
            row = pd.Series(0.0, index=self.u)
            for n in names:
                if t in self.ranks[n].index:
                    row = row.add(self.ranks[n].loc[t] * wt[n], fill_value=0)
            tot = row.sum()
            if tot > 0:
                sc.loc[t] = row / tot
        rets = []
        for t in sc.index:
            row = sc.loc[t].dropna()
            if len(row) < 20:
                continue
            top = self.env.capped(row, ascending=False)
            bot = self.env.capped(row, ascending=True)
            wl = self.env.erc_w(top, t) or {}
            ws = self.env.erc_w(bot, t) or {}
            if t in self.daily_ret.index:
                r = self.daily_ret.loc[t].fillna(0.0)
                lr = sum(r[c] * wi for c, wi in wl.items())
                sr = sum(r[c] * wi for c, wi in ws.items())
                rets.append((t, lr - sr))
        return pd.Series({d: v for d, v in rets}).sort_index().dropna()


def report(name, s):
    st = stats(s)
    print(f'{name}: 夏普={st["sharpe"]:.2f} 年化={st["ann"]:.1%} 回撤={st["mdd"]:.1%} '
          f'OOS={st["oos"]:.2f} 实盘={st["live"]:.2f} ({len(s)}天)')
    # 逐年
    for y, grp in s.groupby(s.index.year):
        sh = grp.mean() / grp.std() * np.sqrt(252) if grp.std() > 0 else 0
        print(f'    {y}: 收益={grp.sum():.1%} 夏普={sh:.2f}')
    # 逐月负月
    mos = s.groupby([s.index.year, s.index.month]).sum()
    print(f'    负月: {(mos < 0).sum()}/{len(mos)}, 最差月={mos.min():.2%}')
    return st


def main():
    r = Runner()
    print('=' * 70)
    print('实验13: 6f vs 12f 回测 (2024-01-01 ~ 2026-08-04, IC_IR 60日)')
    print('=' * 70)
    print(f'样本: {r.cal[0].date()} ~ {r.cal[-1].date()} ({len(r.cal)} 交易日)')
    s6 = r.returns(F6)
    s12 = r.returns(F12)
    print()
    report('6f-IC_IR', s6)
    print()
    report('12f-IC_IR', s12)
    # 净值图
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, s in [('6f-IC_IR', s6), ('12f-IC_IR', s12)]:
        nav = (1 + s).cumprod()
        ax.plot(nav.index, nav.values, label=name, lw=1.5)
    ax.axvline(pd.Timestamp('2026-03-01'), color='gray', ls='--', lw=1, label='OOS起点')
    ax.axvline(pd.Timestamp('2026-05-16'), color='red', ls='--', lw=1, label='实盘起点')
    ax.set_title('6f vs 12f IC_IR 净值 (2024-01 起)')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('runs/exp13_2024_nav.png', dpi=150)
    print('\n净值图: runs/exp13_2024_nav.png')


if __name__ == '__main__':
    main()
