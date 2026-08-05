"""实验14: 12f 增量因子 2024 年拖累分析.

问题: 12f-IC_IR 2024 夏普仅 0.48 (vs 6f 2.38), 哪些增量因子拖累最重?

方法:
  A. 单因子 2024 回测 (Top5多空): 看每个增量因子独立在 2024 的表现
  B. 12f 逐个去掉增量因子, 看 2024 段夏普 (哪个去掉后 2024 提升最多)
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
INCR = F12[6:]  # 6 个增量因子


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

    def returns_icir(self, names, window=60, t0=None, t1=None):
        ic = pd.DataFrame({n: self.ranks[n].corrwith(self.fwd_rank, axis=1) for n in names})
        wmap = {}
        for t in self.cal:
            if t0 is not None and t < t0:
                continue
            if t1 is not None and t > t1:
                continue
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
            if t not in wmap or t < (t0 or self.cal[0]) or t > (t1 or self.cal[-1]):
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

    def returns_single(self, n, t0, t1):
        rk = self.ranks[n]
        rets = []
        for t in rk.index:
            if t < t0 or t > t1:
                continue
            row = rk.loc[t].dropna()
            if len(row) < 20:
                continue
            top5 = row.nlargest(5).index
            bot5 = row.nsmallest(5).index
            if t in self.daily_ret.index:
                r = self.daily_ret.loc[t].fillna(0.0)
                rets.append((t, r[top5].mean() - r[bot5].mean()))
        return pd.Series({d: v for d, v in rets}).sort_index().dropna()


def main():
    r = Runner()
    t0, t1 = pd.Timestamp('2024-01-01'), pd.Timestamp('2024-12-31')
    print('=' * 70)
    print('实验14: 12f 增量因子 2024 年拖累分析')
    print('=' * 70)

    print('\n--- A. 单因子 2024 表现 (Top5多空) ---')
    print(f'{"因子":<42} {"2024夏普":>10} {"2024收益":>10}')
    for n in INCR:
        s = r.returns_single(n, t0, t1)
        if len(s) < 50:
            print(f'{n:<42} {"样本不足":>10}')
            continue
        sh = s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0
        print(f'{n:<42} {sh:>10.2f} {s.sum():>10.1%}')

    print('\n--- B. 12f 逐个去掉增量因子, 2024 段夏普 ---')
    base12 = r.returns_icir(F12, t0=t0, t1=t1)
    base6 = r.returns_icir(F6, t0=t0, t1=t1)
    def shp(x):
        return x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
    print(f'12f 全 (2024): 夏普={shp(base12):.2f} 收益={base12.sum():.1%}')
    print(f'6f 全 (2024): 夏普={shp(base6):.2f} 收益={base6.sum():.1%}')
    print(f'{"去掉":<42} {"2024夏普":>10} {"Δ":>8} {"2024收益":>10}')
    base_sh = shp(base12)
    for n in INCR:
        sub = [x for x in F12 if x != n]
        s = r.returns_icir(sub, t0=t0, t1=t1)
        sh = shp(s)
        print(f'{n:<42} {sh:>10.2f} {sh-base_sh:>+8.2f} {s.sum():>10.1%}')


if __name__ == '__main__':
    main()
