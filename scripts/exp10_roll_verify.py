"""实验10: 6因子替换版滚动重验 — 确认实盘5.72是否偶然.

方法: 滚动子窗口 (每 60 交易日一个, 步进 30) 分别回测:
  A. 6因子 (生产) IC_IR
  B. 6因子替换 (skewness->open_close_drift) IC_IR
每个窗口比两者夏普, 看替换版是否持续占优 (若只在实盘段偶然则不稳).

同时做跨样本稳定性: 窗口1(前段)选出的最优替换, 在窗口2(后段)是否仍占优.
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from exp_core import ExpEnv, PROD6, CAND27, CAND_DIR

F6 = list(PROD6)
F6SWAP = ['intraday_jump_intensity_20d', 'intraday_price_peak_count_20d',
          'intraday_open_close_drift_20d', 'intraday_dtws_20d',
          'intraday_drip_stone_20d', 'intraday_peak_ridge_ratio_20d']


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
        all_names = list(dict.fromkeys(F6 + F6SWAP + ['intraday_open_close_drift_20d']))
        all_F = {}
        for n in all_names:
            if n in PROD6:
                all_F[n] = PROD6[n]
            elif n in CAND_DIR:
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

    def returns_window(self, names, t0, t1):
        """在 [t0, t1] 窗口内回测 (用全期IC, 但只取窗口内收益)."""
        ic = pd.DataFrame({n: self.ranks[n].corrwith(self.fwd_rank, axis=1) for n in names})
        wmap = {}
        cal_win = [t for t in self.cal if t0 <= t <= t1]
        for t in cal_win:
            hist = ic.loc[:t].iloc[-60:-1]
            if len(hist) < 20:
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
        sc = pd.DataFrame(index=cal_win, columns=self.u, dtype=float)
        for t in cal_win:
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
    cal = r.cal
    print('=== 实验10: 6因子替换版滚动重验 (双窗口 20+60日) ===')
    for win in [20, 60]:
        step = win // 2
        print(f'\n--- 窗口 {win} 交易日长, {step} 步进 ---')
        print(f'{"窗口":<16} {"6因子夏普":>10} {"6替换夏普":>10} {"替换更优?":>10}')
        wins = 0
        nwin = 0
        for i in range(0, len(cal) - win, step):
            t0, t1 = cal[i], cal[i + win]
            s6 = r.returns_window(F6, t0, t1)
            ss = r.returns_window(F6SWAP, t0, t1)
            if len(s6) < max(10, win // 3) or len(ss) < max(10, win // 3):
                continue
            sh6 = s6.mean() / s6.std() * np.sqrt(252) if s6.std() > 0 else 0
            shs = ss.mean() / ss.std() * np.sqrt(252) if ss.std() > 0 else 0
            better = '是' if shs > sh6 else '否'
            if shs > sh6:
                wins += 1
            nwin += 1
            print(f'{t0.date()}~{t1.date()} {sh6:>10.2f} {shs:>10.2f} {better:>10}')
        print(f'\n窗口{win}: 替换版占优 {wins}/{nwin} ({wins/max(nwin,1):.0%})')


if __name__ == '__main__':
    main()
