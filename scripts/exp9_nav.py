"""实验9: 版本对比净值图 + 单因子净值曲线.

A. 版本对比: 6因子IC_IR vs 12因子IC_IR vs 6因子替换IC_IR (净值图)
B. 单因子净值: 每个因子独立做 Top20%多空 (5长5空, 池内等权), 看因子本身信号强度
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from exp_core import ExpEnv, PROD6, CAND27, CAND_DIR

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
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


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
        all_names = list(dict.fromkeys(F12 + F6 + F6SWAP))
        all_F = dict(PROD6)
        for n in CAND27:
            all_F[n] = CAND_DIR[n]
        self.all_F = all_F
        self.comp = {}
        for i in range(0, len(all_names), 10):
            batch = all_names[i:i+10]
            try:
                part = self.env.engine.compute_factors(batch, self.cal, self.u, parallel=False)
                for k, v in part.items():
                    self.comp[k] = v
            except Exception:
                pass
        self.ranks = {}
        for n, direction in all_F.items():
            if n not in self.comp:
                continue
            r = self.comp[n].rank(axis=1, pct=True)
            self.ranks[n] = r if direction == 1 else (1 - r)
        self.fwd_rank = self.daily_ret.rank(axis=1)

    def returns_icir(self, names, window=60):
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

    def returns_single(self, n, n_long=5):
        """单因子: 截面Top5多空 (池内等权)."""
        rk = self.ranks[n]
        rets = []
        for t in rk.index:
            row = rk.loc[t].dropna()
            if len(row) < 20:
                continue
            top5 = row.nlargest(n_long).index
            bot5 = row.nsmallest(n_long).index
            if t in self.daily_ret.index:
                r = self.daily_ret.loc[t].fillna(0.0)
                lr = r[top5].mean()
                sr = r[bot5].mean()
                rets.append((t, lr - sr))
        return pd.Series({d: v for d, v in rets}).sort_index().dropna()


def main():
    r = Runner()
    # A. 版本对比
    rs = {
        '6因子 IC_IR (生产候选)': r.returns_icir(F6),
        '12因子 IC_IR (前向搜索)': r.returns_icir(F12),
        '6因子替换 IC_IR': r.returns_icir(F6SWAP),
    }
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, s in rs.items():
        nav = (1 + s).cumprod()
        ax.plot(nav.index, nav.values, label=f'{name} (夏普 {s.mean()/s.std()*np.sqrt(252):.2f})', lw=1.5)
    ax.axvline(pd.Timestamp('2026-03-01'), color='gray', ls='--', lw=1, label='OOS起点')
    ax.axvline(pd.Timestamp('2026-05-16'), color='red', ls='--', lw=1, label='实盘起点')
    ax.set_title('版本对比净值 (6/12/替换, IC_IR)')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('runs/exp9_versions_nav.png', dpi=150)
    print('净值图已存: runs/exp9_versions_nav.png')

    # B. 单因子净值 (12因子 + 关键候选)
    names = F12 + ['intraday_open_close_drift_20d', 'intraday_herding_20d',
                   'intraday_overnight_absorption_20d', 'intraday_session_symmetry_20d',
                   'intraday_depth_trend_20d', 'intraday_turnover_velocity_20d',
                   'intraday_volume_rank_ratio_20d', 'intraday_extreme_freq_balance_20d',
                   'intraday_lowest_time_20d', 'intraday_oi_quantile_range_20d',
                   'intraday_oi_peak_ridge_ratio_20d', 'intraday_oi_skew_stability_20d',
                   'intraday_oi_vol_corr_daily_20d', 'intraday_cross_vol_20d',
                   'intraday_volatility_clustering_20d', 'intraday_zero_ret_freq_20d',
                   'intraday_settle_position_20d', 'intraday_settle_gap_20d',
                   'intraday_amihud_vol_ratio_20d', 'intraday_amihud_trend_20d',
                   'intraday_price_delay_20d', 'intraday_volume_time_shape_20d']
    single_stats = []
    for n in names:
        if n not in r.ranks:
            continue
        s = r.returns_single(n)
        if len(s) < 50:
            continue
        sh = s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0
        single_stats.append((n, sh, s.sum()))
    single_stats.sort(key=lambda x: -x[1])
    print('\n=== 单因子净值 (Top5多空, 池内等权) ===')
    for n, sh, tot in single_stats:
        print(f'  {n:<40} 夏普={sh:.2f} 总收益={tot:.1%}')

    # 单因子净值图 (前12)
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    for n, sh, tot in single_stats[:12]:
        s = r.returns_single(n)
        nav = (1 + s).cumprod()
        ax2.plot(nav.index, nav.values, label=f'{n.split("_")[2] if len(n.split("_"))>2 else n} (夏普{sh:.2f})', lw=1.2)
    ax2.axvline(pd.Timestamp('2026-05-16'), color='red', ls='--', lw=1, label='实盘起点')
    ax2.set_title('单因子净值 (Top5多空, 前12强因子)')
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig('runs/exp9_single_nav.png', dpi=150)
    print('\n单因子净值图已存: runs/exp9_single_nav.png')


if __name__ == '__main__':
    main()
