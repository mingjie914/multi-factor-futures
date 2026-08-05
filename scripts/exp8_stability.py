"""实验8: 12因子IC_IR稳健性验证 (逐年/逐月/滚动重验).

对比: 6因子IC_IR (生产候选) vs 12因子IC_IR (前向搜索发现)
评估:
  1. 全段/逐年/逐月夏普
  2. 滚动重验: 每60日窗口内重新算IC_IR权重(已在exp7实现, 这里保持)
  3. OOS/实盘分段
  4. 12因子中每个新增因子的边际贡献(逐个去掉)
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
        all_names = list(dict.fromkeys(F12 + F6 + F6SWAP))  # 15个, 保序
        all_F = {}
        for n in all_names:
            if n in PROD6:
                all_F[n] = PROD6[n]
            elif n in CAND_DIR:
                all_F[n] = CAND_DIR[n]
        self.all_F = all_F
        # 分组计算 (每批10个, 避免批量 compute 不稳定丢因子)
        self.comp = {}
        for i in range(0, len(all_names), 10):
            batch = all_names[i:i+10]
            try:
                part = self.env.engine.compute_factors(batch, self.cal, self.u, parallel=False)
                for k, v in part.items():
                    self.comp[k] = v
            except Exception as e:
                print(f'  [batch fail] {batch}: {e}')
        self.ranks = {}
        for n, direction in all_F.items():
            if n not in self.comp:
                print(f'  [skip] {n} compute 失败/未返回')
                continue
            r = self.comp[n].rank(axis=1, pct=True)
            self.ranks[n] = r if direction == 1 else (1 - r)
        self.fwd_rank = self.daily_ret.rank(axis=1)

    def returns(self, names, window=60):
        ic = pd.DataFrame({n: self.ranks[n].corrwith(self.fwd_rank, axis=1) for n in names})
        wmap = {}
        for t in self.cal:
            hist = ic.loc[:t].iloc[-window:]
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

    def yearly_monthly(self, name, rets):
        print(f'\n=== {name} ({len(rets)}天) ===')
        st = stats(rets)
        print(f'全段: 年化={st["ann"]:.1%} 夏普={st["sharpe"]:.2f} 回撤={st["mdd"]:.1%} OOS={st["oos"]:.2f} 实盘={st["live"]:.2f}')
        # 逐年
        yrs = []
        for y, grp in rets.groupby(rets.index.year):
            yrs.append((y, grp.sum(), grp.mean() / grp.std() * np.sqrt(252) if grp.std() > 0 else 0))
        for y, ret, sh in yrs:
            print(f'  {y}: 收益={ret:.1%} 夏普={sh:.2f}')
        # 逐月
        mos = rets.groupby([rets.index.year, rets.index.month]).sum()
        print(f'  负月数: {(mos < 0).sum()}/{len(mos)} 最差月={mos.min():.2%} ({mos.idxmin()})')
        return rets


def main():
    r = Runner()
    print('=' * 60)
    print('任务1: 12因子IC_IR稳健性验证')
    print('=' * 60)
    r6 = r.returns(F6)
    r12 = r.returns(F12)
    r6s = r.returns(F6SWAP)
    r.yearly_monthly('6因子IC_IR (生产候选)', r6)
    r.yearly_monthly('12因子IC_IR (前向搜索)', r12)
    r.yearly_monthly('6因子替换IC_IR (skewness->open_close_drift)', r6s)

    # 边际贡献: 12因子逐个去掉
    print('\n=== 12因子边际贡献 (逐个去掉, 看夏普变化) ===')
    base = stats(r12)['sharpe']
    print(f'12因子全: {base:.2f}')
    for n in F12:
        sub = [x for x in F12 if x != n]
        st = stats(r.returns(sub))
        print(f'  去 {n:<38} 夏普={st["sharpe"]:.2f} 实盘={st["live"]:.2f} (Δ={st["sharpe"]-base:+.2f})')


if __name__ == '__main__':
    main()
