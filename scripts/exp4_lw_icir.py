"""实验4: Ledoit-Wolf 收缩 ICIR 加权 vs 等权 vs 简单ICIR.

方法 (第二篇): w* = Σ_IC^{-1} · IC̄ (最大化 IC_IR)
  - 滚动窗口内 Spearman IC 矩阵 → Ledoit-Wolf 收缩协方差 → 最优权重
对比:
  A. 等权 (生产, 2.04)
  B. 简单 ICIR 加权 (滚动60日, 未收缩, 之前测过 1.70)
  C. Ledoit-Wolf 收缩 ICIR 加权 (第二篇方法)
  D. IC_IR 解析最优权重 (Σ_IC^{-1}·IC̄ + LW收缩)
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
from exp_core import ExpEnv, stats, format_stats, PROD6


def ledoit_wolf_cov(ic_matrix):
    """Ledoit-Wolf 收缩协方差 (第二篇 4.5)."""
    T, N = ic_matrix.shape
    sample_cov = np.cov(ic_matrix, rowvar=False, ddof=1)
    sample_corr = np.corrcoef(ic_matrix, rowvar=False)
    idx = np.triu_indices(N, k=1)
    if N > 1:
        avg_corr = np.mean(sample_corr[idx])
    else:
        avg_corr = 0.0
    target_corr = np.eye(N) * (1 - avg_corr) + np.ones((N, N)) * avg_corr
    std_vector = np.std(ic_matrix, axis=0, ddof=1)
    target_cov = np.outer(std_vector, std_vector) * target_corr
    # 最优收缩系数 (简化: 用固定 0.5 或标准 LW)
    # 完整 pi/rho 计算较重, 这里用解析近似 (Ledoit-Wolf 2004 标准)
    centered = ic_matrix - ic_matrix.mean(axis=0)
    pi = 0.0
    for t in range(T):
        r = centered.iloc[t].values.reshape(-1, 1)
        pi += np.sum((r @ r.T - sample_cov) ** 2)
    pi /= T
    gamma = np.sum((target_cov - sample_cov) ** 2)
    lam = max(0.0, min(1.0, pi / gamma)) if gamma > 0 else 0.5
    return lam * target_cov + (1 - lam) * sample_cov, lam


def main():
    env = ExpEnv(PROD6)
    cal, u, daily_ret = env.cal, env.u, env.daily_ret
    score = env.compute_scores()  # 等权基线得分
    names = list(PROD6)
    # 计算各因子 rank (方向已调整)
    ranks = {}
    comp = env._comp if env._comp is not None else env.engine.compute_factors(names, cal, u, parallel=True)
    env._comp = comp
    for n, direction in PROD6.items():
        r = comp[n].rank(axis=1, pct=True)
        ranks[n] = r if direction == 1 else (1 - r)
    fwd_rank = daily_ret.rank(axis=1)

    def build_score(mode, window=60):
        """mode: 'ew'等权 / 'icir'滚动ICIR / 'lw' Ledoit-Wolf ICIR / 'opt' IC_IR解析."""
        if mode == 'ew':
            return score
        # 滚动 IC
        ic = pd.DataFrame({n: ranks[n].corrwith(fwd_rank, axis=1) for n in names})
        w = {}
        for t in cal:
            hist = ic.loc[:t].iloc[-window:]
            if len(hist) < 20:
                w[t] = pd.Series(1.0 / len(names), index=names)
                continue
            ic_mean = hist.mean()
            if mode == 'icir':
                ic_std = hist.std().replace(0, np.nan)
                wi = (ic_mean.abs() / ic_std).fillna(1.0)
            elif mode in ('lw', 'opt'):
                lw_cov, lam = ledoit_wolf_cov(hist)
                if mode == 'opt':
                    try:
                        wi = np.linalg.inv(lw_cov) @ ic_mean.values
                    except np.linalg.LinAlgError:
                        wi = ic_mean.abs().values
                    wi = np.abs(wi)
                else:  # lw: 用收缩后的 ICIR (mean/std with LW implied)
                    ic_std = np.sqrt(np.diag(lw_cov)).clip(min=1e-8)
                    wi = (ic_mean.abs() / ic_std).fillna(1.0)
            else:
                wi = pd.Series(1.0 / len(names), index=names)
            wsum = np.sum(wi)
            w[t] = pd.Series(np.asarray(wi, dtype=float) / wsum, index=names)
        # 合成
        out = pd.DataFrame(index=cal, columns=u, dtype=float)
        for t in cal:
            if t not in w:
                continue
            wt = w[t]
            for n in names:
                out.loc[t] = out.loc[t].add(ranks[n].loc[t] * wt[n], fill_value=0) if n in ranks else out.loc[t]
            s = out.loc[t].sum()
            if s > 0:
                out.loc[t] = out.loc[t] / s
        return out

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

    print('=== 实验4: Ledoit-Wolf ICIR 加权 ===')
    sA = backtest(score)
    print(f'A 等权(生产): {format_stats(stats(sA))}')
    for mode, lab in [('icir', 'B 简单ICIR(60日)'), ('lw', 'C LW收缩ICIR'), ('opt', 'D IC_IR解析最优')]:
        sc = build_score(mode)
        s = backtest(sc)
        print(f'{lab:<20}: {format_stats(stats(s))}')

    # 净值图
    fig, ax = plt.subplots(figsize=(15, 8))
    sc_b = build_score('icir'); sB = backtest(sc_b)
    sc_c = build_score('lw'); sC = backtest(sc_c)
    sc_d = build_score('opt'); sD = backtest(sc_d)
    for s, lab, c in [(sA, 'A 等权', '#2ecc71'), (sB, 'B 简单ICIR', '#e67e22'), (sC, 'C LW-ICIR', '#3498db'), (sD, 'D IC_IR最优', '#9b59b6')]:
        nav = (1 + s).cumprod()
        ax.plot(nav.index, nav.values, label=lab, color=c, linewidth=1.5)
    ax.axvline(pd.Timestamp('2026-03-01'), color='gray', ls='--', alpha=0.8, label='OOS起点')
    ax.axvline(pd.Timestamp('2026-05-16'), color='red', ls='--', alpha=0.8, label='实盘起点')
    ax.set_title('实验4: Ledoit-Wolf ICIR 加权 (6因子)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('runs/exp4_lw_icir.png', dpi=150)
    print('\n净值图: runs/exp4_lw_icir.png')


if __name__ == '__main__':
    main()
