"""实验12: 池内 ERC 协方差参数敏感性.

A. 协方差回看窗口: 30/60/90/120 日历日 (erc_w 用 sd = t - days)
B. 收缩系数: 0.5/0.7/0.9 (样本协方差 vs 对角目标)

基于 12因子 IC_IR (窗口60日最优) 评估.
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
        all_names = list(F12)
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

    def returns(self, window=60, cov_days=90, shrink=0.7):
        names = F12
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
            # 自定义 ERC: 按 cov_days/shrink
            wl = self._erc(top, t, cov_days, shrink)
            ws = self._erc(bot, t, cov_days, shrink)
            wl = wl if wl is not None else dict(zip(top, [1.0/len(top)]*len(top)))
            ws = ws if ws is not None else dict(zip(bot, [1.0/len(bot)]*len(bot)))
            if t in self.daily_ret.index:
                r = self.daily_ret.loc[t].fillna(0.0)
                lr = sum(r[c] * wi for c, wi in wl.items())
                sr = sum(r[c] * wi for c, wi in ws.items())
                rets.append((t, lr - sr))
        return pd.Series({d: v for d, v in rets}).sort_index().dropna()

    def _erc(self, pool, t, cov_days, shrink):
        if len(pool) < 2:
            return None
        sd = t - pd.Timedelta(days=cov_days)
        try:
            from pipeline.runner import PipelineRunner
            c = pd.DatetimeIndex(self.env.runner.data_manager.get_calendar(sd, t))
        except Exception:
            c = self.daily_ret.loc[:t].index[-cov_days:]
        rs = self.daily_ret.reindex(c)[list(pool)].dropna()
        if rs.shape[0] < 10:
            return None
        cov_raw = rs.cov().values
        cov = shrink * cov_raw + (1 - shrink) * np.diag(np.diag(cov_raw))
        try:
            from core.risk import RiskBudgetingOptimizer
            w = RiskBudgetingOptimizer._erc_weights(cov, np.ones(len(pool)))
            return dict(zip(pool, w))
        except (RuntimeError, ValueError, ImportError):
            v = rs.std(ddof=0).replace(0, np.nan).dropna()
            if v.empty:
                return None
            w = (1.0 / v).values
            return dict(zip(pool, w / w.sum()))


def main():
    r = Runner()
    print('=' * 70)
    print('实验12: 池内 ERC 协方差参数敏感性 (12因子 IC_IR 窗口60)')
    print('=' * 70)

    print('\n--- A. 协方差回看窗口 (收缩0.7固定) ---')
    print(f'{"窗口":<6} {"夏普":>8} {"年化":>8} {"回撤":>8} {"OOS":>6} {"实盘":>6}')
    for cd in [30, 60, 90, 120]:
        s = r.returns(cov_days=cd)
        st = stats(s)
        print(f'{cd:<6} {st["sharpe"]:>8.2f} {st["ann"]:>7.1%} {st["mdd"]:>8.1%} {st["oos"]:>6.2f} {st["live"]:>6.2f}')

    print('\n--- B. 收缩系数 (窗口90固定) ---')
    print(f'{"收缩":<6} {"夏普":>8} {"年化":>8} {"回撤":>8} {"OOS":>6} {"实盘":>6}')
    for sh in [0.5, 0.7, 0.9]:
        s = r.returns(cov_days=90, shrink=sh)
        st = stats(s)
        print(f'{sh:<6} {st["sharpe"]:>8.2f} {st["ann"]:>7.1%} {st["mdd"]:>8.1%} {st["oos"]:>6.2f} {st["live"]:>6.2f}')


if __name__ == '__main__':
    main()
