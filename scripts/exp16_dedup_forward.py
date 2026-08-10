"""实验16: 候选因子相关性去重 + 组合测试 (挖掘更优生产组合).

流程:
  1. 收集 52 个有效候选 (27旧 + 25新) + 生产 6 因子的因子矩阵
  2. 相关性矩阵: 候选与生产6因子 + 候选间两两相关
  3. 去重: 与生产6因子 corr>=0.5 剔除; 候选间贪心去重 (保留高|IC|)
  4. 前向选择: 从生产6因子出发, IC_IR 组合, 逐步加入独立候选
  5. 评估: 全段/OOS/实盘夏普, 找最优组合 (兼顾稳健性)
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from exp_core import ExpEnv, stats, PROD6, CAND27, CAND_DIR

# 新增候选 (2026-08-05 检验通过的 25 个, 从有效因子库第六七章)
NEW25 = [
    # settle 5
    'intraday_settle_position_20d', 'intraday_settle_close_basis_20d',
    'intraday_settle_drift_20d', 'intraday_settle_gap_20d',
    'intraday_settle_vol_ratio_20d',
    # 期限/OI 9
    'intraday_oi_vol_corr_daily_20d', 'intraday_term_slope_20d',
    'intraday_oi_quantile_range_20d', 'intraday_oi_log_change_vol_20d',
    'intraday_term_roll_yield_20d', 'intraday_term_vol_spread_20d',
    'intraday_oi_peak_ridge_ratio_20d', 'intraday_oi_vol_price_corr_20d',
    'intraday_term_spread_vol_20d',
    # 新因子 4
    'intraday_rollover_basis_gap_20d', 'intraday_basis_momentum_20d',
    'intraday_roll_yield_dualscore_20d', 'intraday_roll_dualscore_consistency_20d',
    # 补检 7
    'intraday_amihud_resid_vol_20d', 'intraday_volume_oi_price_confirm_20d',
    'intraday_amihud_cross_z_20d', 'intraday_overnight_gap_reaction_20d',
    'intraday_false_breakout_retrace_20d', 'intraday_ma_count_bullish_20d',
    'intraday_rv_compression_breakout_20d',
]


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
        # 全部候选方向: 旧27 (CAND_DIR) + 新25 (默认方向需从检验获取, 先按 1 存, 后补)
        all_names = list(dict.fromkeys(list(PROD6) + CAND27 + NEW25))
        all_F = dict(PROD6)
        for n in CAND27:
            all_F[n] = CAND_DIR[n]
        for n in NEW25:
            all_F.setdefault(n, 1)  # 方向待从检验结果确定
        self.all_F = all_F
        self.all_names = all_names
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

    def corr_with(self, a, b):
        """两因子的截面 rank 序列相关 (全期均值)."""
        ra = self.ranks[a]
        rb = self.ranks[b]
        corrs = []
        for t in self.cal:
            x = ra.loc[t].dropna()
            y = rb.loc[t].dropna()
            common = x.index.intersection(y.index)
            if len(common) >= 10:
                corrs.append(x[common].corr(y[common]))
        return np.mean(corrs) if corrs else np.nan

    def try_direction(self, name, base_names):
        """测 ± 两个方向, 返回 (夏普, 方向, 收益序列)."""
        best = None
        for d in [1, -1]:
            rk = self.comp[name].rank(axis=1, pct=True)
            self.ranks[name] = rk if d == 1 else (1 - rk)
            s = self.returns(base_names + [name])
            score = s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0
            if best is None or score > best[0]:
                best = (score, d, s)
        return best


def main():
    r = Runner()
    print('=' * 70)
    print('实验16: 候选去重 + 前向选择')
    print('=' * 70)

    # 1. 相关性去重: 候选 vs 生产6因子
    print('\n--- 1. 候选 vs 生产6因子 相关 (剔除 >=0.5) ---')
    kept = []
    dropped = []
    for n in r.all_names:
        if n in PROD6:
            continue
        maxc = 0
        for p in PROD6:
            c = r.corr_with(n, p)
            if not np.isnan(c):
                maxc = max(maxc, abs(c))
        if maxc >= 0.5:
            dropped.append((n, maxc))
        else:
            kept.append(n)
    print(f'保留: {len(kept)}  剔除(相关>=0.5): {len(dropped)}')
    for n, c in dropped:
        print(f'  剔除 {n}: max_corr={c:.2f}')

    # 2. 候选间贪心去重 (保留前向选择的独立集)
    print('\n--- 2. 前向选择 (从生产6出发, IC_IR) ---')
    base = list(PROD6)
    st = stats(r.returns(base))
    print(f'基准 6f-IC_IR: 夏普={st["sharpe"]:.2f} OOS={st["oos"]:.2f} 实盘={st["live"]:.2f}')
    current = list(base)
    chosen = []
    for step in range(8):  # 最多加到 14
        best, best_st, best_name = -1e9, None, None
        for c in kept:
            if c in current:
                continue
            scr, d, s = r.try_direction(c, current)
            live_sh = s[s.index > pd.Timestamp('2026-05-15')].mean() * 252 / (s[s.index > pd.Timestamp('2026-05-15')].std() * np.sqrt(252)) if len(s[s.index > pd.Timestamp('2026-05-15')]) > 2 else 0
            combo = scr * 0.5 + live_sh * 0.5
            if combo > best:
                best, best_st, best_name = combo, s, c
        if best_name is None:
            break
        current.append(best_name)
        chosen.append(best_name)
        st = stats(best_st)
        print(f'  +{best_name:<42} 夏普={st["sharpe"]:.2f} OOS={st["oos"]:.2f} 实盘={st["live"]:.2f}')
    print(f'\n最终组合 ({len(current)}因子): {current}')


if __name__ == '__main__':
    main()
