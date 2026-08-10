"""探索矩阵: 因子池{6,12,27} x 组合方式{等权, LW-ICIR, IC_IR最优}.

因子池:
  F6  = 生产6因子 (当前)
  F12 = 生产6 + 27候选中 |IC|最强6个
  F27 = 生产6 + 全部27候选 (33? 不, 27池=候选27, 6生产已含其中部分?) 
  注: 27候选是独立于生产6的, 全池=6+27=33? 但候选与生产无重叠, 全=33.
  实际: 用 6 / 12(=6+6强) / 33(全部有效)
组合方式:
  ew  = 等权 (生产)
  lw  = Ledoit-Wolf收缩ICIR
  opt = IC_IR解析最优 (Σ_IC^{-1}·IC̄)
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from exp_core import ExpEnv, stats, format_stats, PROD6, CAND27, CAND_DIR


def ledoit_wolf_cov(ic_matrix):
    T, N = ic_matrix.shape
    sample_cov = np.cov(ic_matrix, rowvar=False, ddof=1)
    sample_corr = np.corrcoef(ic_matrix, rowvar=False)
    idx = np.triu_indices(N, k=1)
    avg_corr = np.mean(sample_corr[idx]) if N > 1 else 0.0
    target_corr = np.eye(N) * (1 - avg_corr) + np.ones((N, N)) * avg_corr
    std_vector = np.std(ic_matrix, axis=0, ddof=1)
    target_cov = np.outer(std_vector, std_vector) * target_corr
    centered = ic_matrix - ic_matrix.mean(axis=0)
    pi = 0.0
    for t in range(T):
        r = centered.iloc[t].values.reshape(-1, 1)
        pi += np.sum((r @ r.T - sample_cov) ** 2)
    pi /= T
    gamma = np.sum((target_cov - sample_cov) ** 2)
    lam = max(0.0, min(1.0, pi / gamma)) if gamma > 0 else 0.5
    return lam * target_cov + (1 - lam) * sample_cov


def main():
    # ===== 因子池定义 =====
    # 27候选按 |IC| 排序 (来自 docs/有效因子库.md 第二层)
    cand_sorted = sorted(CAND27, key=lambda n: CAND_DIR.get(n, 0), reverse=True)
    # 用 |IC| 排序: 从 _final_candidates 提取的 |t| 排序在 docs 中, 这里用近似: CAND_DIR 值大小
    # 选 |方向| 最大的6个作为 F12 增量 (简化: 取前6)
    F6 = dict(PROD6)
    # F12: 生产6 + 27候选选6个 (用 IC 排序, 但这里没有 IC 数据, 取前6个候选)
    extra6 = ['intraday_zero_ret_freq_20d', 'intraday_open_close_drift_20d', 'intraday_volatility_clustering_20d', 'intraday_oi_vol_corr_daily_20d', 'intraday_oi_time_centroid_20d', 'intraday_wash_trade_20d']
    F12 = dict(PROD6)
    for n in extra6:
        F12[n] = CAND_DIR[n]
    # F33: 全部有效 (6生产 + 27候选)
    F33 = dict(PROD6)
    for n in CAND27:
        F33[n] = CAND_DIR[n]
    pools = {'F6': F6, 'F12': F12, 'F33': F33}

    # ===== 组合方式 =====
    # 每池预计算因子矩阵
    results = {}
    for pool_name, F in pools.items():
        env = ExpEnv(F)
        cal, u, daily_ret = env.cal, env.u, env.daily_ret
        comp = env.engine.compute_factors(list(F), cal, u, parallel=True)
        env._comp = comp
        ranks = {}
        for n, direction in F.items():
            r = comp[n].rank(axis=1, pct=True)
            ranks[n] = r if direction == 1 else (1 - r)
        fwd_rank = daily_ret.rank(axis=1)
        names = list(F)

        # 等权 score
        score_ew = pd.DataFrame(index=cal, columns=u, dtype=float)
        for n in names:
            score_ew = score_ew.add(ranks[n], fill_value=0)
        score_ew = score_ew.div(len(names))

        # 滚动 IC
        ic = pd.DataFrame({n: ranks[n].corrwith(fwd_rank, axis=1) for n in names})

        def backtest(sc):
            rets = []
            for t in sc.index:
                row = sc.loc[t].dropna()
                if len(row) < 20:
                    continue
                top = env.capped(row, ascending=False)
                bot = env.capped(row, ascending=True)
                wl = env.erc_w(top, t) or {}
                ws = env.erc_w(bot, t) or {}
                if t in daily_ret.index:
                    r = daily_ret.loc[t].fillna(0.0)
                    lr = sum(r[c] * wi for c, wi in wl.items())
                    sr = sum(r[c] * wi for c, wi in ws.items())
                    rets.append((t, lr - sr))
            return pd.Series({d: v for d, v in rets}).sort_index().dropna()

        # 等权
        results[(pool_name, 'ew')] = backtest(score_ew)

        # LW-ICIR 和 IC_IR (滚动60日)
        for mode in ('lw', 'opt'):
            wmap = {}
            for t in cal:
                hist = ic.loc[:t].iloc[-60:-1]
                if len(hist) < 20:
                    wmap[t] = pd.Series(1.0 / len(names), index=names)
                    continue
                ic_mean = hist.mean()
                if mode == 'lw':
                    lw_cov = ledoit_wolf_cov(hist)
                    ic_std = np.sqrt(np.diag(lw_cov)).clip(min=1e-8)
                    wi = (ic_mean.abs() / ic_std).values
                else:  # opt
                    lw_cov = ledoit_wolf_cov(hist)
                    try:
                        wi = np.linalg.inv(lw_cov) @ ic_mean.values
                    except np.linalg.LinAlgError:
                        wi = ic_mean.abs().values
                    wi = np.abs(wi)
                s = np.sum(wi)
                wmap[t] = pd.Series(np.asarray(wi, dtype=float) / s, index=names)
            # 合成 score
            sc = pd.DataFrame(index=cal, columns=u, dtype=float)
            for t in cal:
                if t not in wmap:
                    continue
                wt = wmap[t]
                row = pd.Series(0.0, index=u)
                for n in names:
                    if n in ranks and t in ranks[n].index:
                        row = row.add(ranks[n].loc[t] * wt[n], fill_value=0)
                tot = row.sum()
                if tot > 0:
                    sc.loc[t] = row / tot
            results[(pool_name, mode)] = backtest(sc)

    # ===== 汇总 =====
    print('=== 探索矩阵: 因子池 x 组合方式 ===')
    print(f"{'池':<4} {'组合':<6} {'年化':>7} {'夏普':>6} {'回撤':>7} {'OOS':>6} {'实盘':>6}")
    print('-' * 60)
    best = (0, '')
    for pool_name in ['F6', 'F12', 'F33']:
        for mode, lab in [('ew', '等权'), ('lw', 'LW-ICIR'), ('opt', 'IC_IR')]:
            st = stats(results[(pool_name, mode)])
            if st is None:
                continue
            print(f"{pool_name:<4} {lab:<6} {st['ann']:>6.1%} {st['sharpe']:>6.2f} {st['mdd']:>6.1%} {st['oos']:>6.2f} {st['live']:>6.2f}")
            if st['sharpe'] > best[0]:
                best = (st['sharpe'], f"{pool_name}-{lab}")
    print(f"\n最佳: {best[1]} (夏普 {best[0]:.2f})")

    # 净值图
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors = {'ew': '#2ecc71', 'lw': '#e67e22', 'opt': '#9b59b6'}
    for ax, pool_name in zip(axes, ['F6', 'F12', 'F33']):
        for mode, lab in [('ew', '等权'), ('lw', 'LW-ICIR'), ('opt', 'IC_IR')]:
            if (pool_name, mode) not in results:
                continue
            s = results[(pool_name, mode)]
            nav = (1 + s).cumprod()
            ax.plot(nav.index, nav.values, label=lab, color=colors[mode], linewidth=1.4)
        ax.axvline(pd.Timestamp('2026-03-01'), color='gray', ls='--', alpha=0.7)
        ax.axvline(pd.Timestamp('2026-05-16'), color='red', ls='--', alpha=0.7)
        ax.set_title(f'{pool_name} ({len(pools[pool_name])}因子)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.suptitle('探索矩阵: 因子池 x 组合方式', fontsize=13)
    plt.tight_layout()
    plt.savefig('runs/exp_matrix_9.png', dpi=150)
    print('\n净值图: runs/exp_matrix_9.png')


if __name__ == '__main__':
    main()
