"""实验6: F33 参数灾难改进 — 降维(PCA) + 强收缩.

对比 (33因子, IC_IR框架, 池内ERC, cap3, 日度):
  A. F33-IC_IR (原始, 参数灾难, 2.25异常)
  B. F33-PCA降维: 33因子 → 前10个主成分 → 主成分得分IC_IR加权
  C. F33-强收缩: 协方差特征值裁剪 (eigenvalue clipping, 方差下限)
  D. F33-长窗口: 滚动窗口 60->120日 (更多样本)
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from exp_core import ExpEnv, stats, format_stats, PROD6, CAND27, CAND_DIR


def ledoit_wolf_cov(ic_matrix, extra_lam=0.0):
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
    lam = min(1.0, lam + extra_lam)  # 强收缩: 额外加大 λ
    return lam * target_cov + (1 - lam) * sample_cov


def eigen_clip(cov, floor=0.0):
    """特征值裁剪: 把过小特征值抬到下限, 稳定求逆."""
    eigval, eigvec = np.linalg.eigh(cov)
    eigval = np.maximum(eigval, floor)
    return eigvec @ np.diag(eigval) @ eigvec.T


def main():
    F33 = dict(PROD6)
    for n in CAND27:
        F33[n] = CAND_DIR[n]
    env = ExpEnv(F33)
    cal, u, daily_ret = env.cal, env.u, env.daily_ret
    comp = env.engine.compute_factors(list(F33), cal, u, parallel=True)
    ranks = {}
    for n, direction in F33.items():
        r = comp[n].rank(axis=1, pct=True)
        ranks[n] = r if direction == 1 else (1 - r)
    fwd_rank = daily_ret.rank(axis=1)
    names = list(F33)
    ic = pd.DataFrame({n: ranks[n].corrwith(fwd_rank, axis=1) for n in names})

    # PCA 主成分 (基于全期 IC 协方差, 用前10个)
    ic_full = ic.dropna()
    ic_std = (ic_full - ic_full.mean()) / np.clip(ic_full.std(), 1e-8, None)
    eigval, eigvec = np.linalg.eigh(ic_std.cov().values)
    top_idx = np.argsort(eigval)[::-1][:10]
    pca_load = eigvec[:, top_idx]  # 33x10

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

    # A. 原始 IC_IR
    def icir_weights(hist, mode='orig', floor=0.0, extra_lam=0.0):
        ic_mean = hist.mean()
        lw_cov = ledoit_wolf_cov(hist, extra_lam)
        if mode == 'clip':
            lw_cov = eigen_clip(lw_cov, floor)
        try:
            wi = np.linalg.inv(lw_cov) @ ic_mean.values
        except np.linalg.LinAlgError:
            wi = ic_mean.abs().values
        return np.abs(wi)

    def run_icir(mode='orig', window=60, floor=0.0, extra_lam=0.0):
        wmap = {}
        for t in cal:
            hist = ic.loc[:t].iloc[-window:-1]
            if len(hist) < 20:
                wmap[t] = pd.Series(1.0 / len(names), index=names)
                continue
            wi = icir_weights(hist, mode, floor, extra_lam)
            s = np.sum(wi)
            wmap[t] = pd.Series(np.asarray(wi, dtype=float) / s, index=names)
        sc = pd.DataFrame(index=cal, columns=u, dtype=float)
        for t in cal:
            if t not in wmap:
                continue
            wt = wmap[t]
            row = pd.Series(0.0, index=u)
            for n in names:
                if t in ranks[n].index:
                    row = row.add(ranks[n].loc[t] * wt[n], fill_value=0)
            tot = row.sum()
            if tot > 0:
                sc.loc[t] = row / tot
        return backtest(sc)

    # B. PCA: 用主成分得分做截面 (前10个主成分等权/IC_IR)
    def run_pca(n_comp=10):
        # 每期: 33因子得分 → 投影到前n_comp主成分 → 合成
        wmap = {}
        for t in cal:
            hist = ic.loc[:t].iloc[-60:-1]
            if len(hist) < 20:
                wmap[t] = pd.Series(1.0 / len(names), index=names)
                continue
            # 滚动 PCA (用滚动窗口的协方差)
            ic_win = (hist - hist.mean()) / np.clip(hist.std(), 1e-8, None)
            ev, evc = np.linalg.eigh(ic_win.cov().values)
            topi = np.argsort(ev)[::-1][:n_comp]
            load = evc[:, topi]  # N x n_comp
            # 主成分得分在品种层面: 对每个品种, 其因子值投影到主成分
            # 用 IC_IR 加权主成分 (对主成分的IC)
            pc_ic = np.zeros(n_comp)
            for k in range(n_comp):
                # 主成分IC ≈ 各因子IC的线性组合
                pc_ic[k] = abs(np.dot(load[:, k], hist.mean().values))
            w_pc = pc_ic / pc_ic.sum()
            # 因子权重 = 主成分负荷 × 主成分权重
            w_f = np.abs(load @ w_pc)
            s = w_f.sum()
            wmap[t] = pd.Series(np.asarray(w_f, dtype=float) / s, index=names)
        sc = pd.DataFrame(index=cal, columns=u, dtype=float)
        for t in cal:
            if t not in wmap:
                continue
            wt = wmap[t]
            row = pd.Series(0.0, index=u)
            for n in names:
                if t in ranks[n].index:
                    row = row.add(ranks[n].loc[t] * wt[n], fill_value=0)
            tot = row.sum()
            if tot > 0:
                sc.loc[t] = row / tot
        return backtest(sc)

    print('=== 实验6: F33 参数灾难改进 ===')
    sA = run_icir('orig')
    print(f'A 原始IC_IR(60日): {format_stats(stats(sA))}')
    for ncomp in [6, 10, 15]:
        s = run_pca(ncomp)
        print(f'B PCA-{ncomp}主成分: {format_stats(stats(s))}')
    sC = run_icir('clip', floor=0.5)
    print(f'C 特征值裁剪(floor=0.5): {format_stats(stats(sC))}')
    sD = run_icir('orig', window=120)
    print(f'D 长窗口(120日): {format_stats(stats(sD))}')
    sE = run_icir('orig', extra_lam=0.3)
    print(f'E 强收缩(λ+0.3): {format_stats(stats(sE))}')


if __name__ == '__main__':
    main()
