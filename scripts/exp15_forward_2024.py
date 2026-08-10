"""实验15: 2024 起样本的前向搜索 — 选出的因子是否与 2025 起不同.

方法: 与 exp7 相同 (前向加入, IC_IR 60日, cap3, ERC), 但回测窗口 2024-01 起.
对比 exp7 (2025 起) 选出的 6 个增量因子, 看 2024 起样本是否选不同.
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from exp_core import ExpEnv, stats, PROD6, CAND27, CAND_DIR

F6 = list(PROD6)
# exp7 (2025起) 选出的增量
EXP7_INCR = ['intraday_oi_time_centroid_20d', 'intraday_wash_trade_20d',
             'intraday_price_delay_20d', 'intraday_amihud_trend_20d',
             'intraday_amihud_vol_ratio_20d', 'intraday_volume_time_shape_20d']


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
        # 全部 33 候选 (6生产 + 27)
        all_names = list(PROD6) + CAND27
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
                print(f'  [skip] {n}')
                continue
            r = self.comp[n].rank(axis=1, pct=True)
            self.ranks[n] = r if direction == 1 else (1 - r)
        self.fwd_rank = self.daily_ret.rank(axis=1)

    def evaluate(self, names, t0=None):
        """IC_IR 回测从 t0 起, 返回 stats dict."""
        ic = pd.DataFrame({n: self.ranks[n].corrwith(self.fwd_rank, axis=1) for n in names})
        wmap = {}
        for t in self.cal:
            if t0 is not None and t < t0:
                continue
            hist = ic.loc[:t].iloc[-60:-1]
            if len(hist) < 30:
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
            if t not in wmap or (t0 is not None and t < t0):
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
        s = pd.Series({d: v for d, v in rets}).sort_index().dropna()
        return stats(s)


def main():
    r = Runner()
    t0 = pd.Timestamp('2024-01-01')
    print('=' * 70)
    print('实验15: 2024 起样本前向搜索 (对比 exp7 的 2025 起)')
    print('=' * 70)

    # 基准: 6f 2024起
    st6 = r.evaluate(F6, t0)
    print(f'基准 6f (2024起): 夏普={st6["sharpe"]:.2f} 实盘={st6["live"]:.2f}')

    # 前向加入 6 步
    current = list(F6)
    chosen = []
    print('\n=== 前向加入 (2024起) ===')
    for step in range(6):
        best, best_st, best_name = -1e9, None, None
        for c in CAND27:
            if c in current:
                continue
            st = r.evaluate(current + [c], t0)
            score = st['sharpe'] * 0.5 + st['live'] * 0.5
            if score > best:
                best, best_st, best_name = score, st, c
        if best_name is None:
            break
        current.append(best_name)
        chosen.append(best_name)
        print(f'  选 {best_name:<42} 夏普={best_st["sharpe"]:.2f} 回撤={best_st["mdd"]:.1%} 实盘={best_st["live"]:.2f}')

    print('\n=== 2024起 选出的 6 增量 vs exp7(2025起) ===')
    print('2024起:', chosen)
    print('2025起:', EXP7_INCR)
    common = set(chosen) & set(EXP7_INCR)
    print(f'重合: {len(common)}/6 -> {common}')
    new = [c for c in chosen if c not in EXP7_INCR]
    gone = [c for c in EXP7_INCR if c not in chosen]
    print(f'新入选(2024起特有): {new}')
    print(f'落选(2025起有, 2024起无): {gone}')


if __name__ == '__main__':
    main()
