"""实验11: IC_IR 滚动窗口敏感性 (10/20/30/60/90日).

对比 12f-IC_IR 在不同 IC_IR 窗口下的表现:
  窗口短 -> 权重响应快但噪声大
  窗口长 -> 权重稳定但滞后
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
            hist = ic.loc[:t].iloc[-window:-1]
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


def main():
    r = Runner()
    print('=' * 70)
    print('实验11: IC_IR 滚动窗口敏感性 (6因子 vs 12因子, 10/20/30/60/90日)')
    print('=' * 70)
    for label, names in [('6因子', F6), ('12因子', F12)]:
        print(f'\n--- {label} ---')
        print(f'{"窗口":<6} {"夏普":>8} {"年化":>8} {"回撤":>8} {"OOS":>6} {"实盘":>6}')
        for win in [10, 20, 30, 60, 90]:
            s = r.returns(names, window=win)
            st = stats(s)
            print(f'{win:<6} {st["sharpe"]:>8.2f} {st["ann"]:>7.1%} {st["mdd"]:>8.1%} {st["oos"]:>6.2f} {st["live"]:>6.2f}')


if __name__ == '__main__':
    main()
