"""实验7: 前向搜索更优因子组合 — 回答"6因子是否最优".

方法: 从生产6因子出发, 用 IC_IR 加权 (最稳健组合方式), 逐个尝试:
  1. 加一个候选因子 (6+1=7... 直到 +6=12), 保留提升最大的
  2. 替换一个现有因子 (6选1换27候选), 保留提升最大的
评估: 全段夏普 + OOS + 实盘, 优先实盘稳健 (这是生产最关心的)
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from exp_core import ExpEnv, stats, format_stats, PROD6, CAND27, CAND_DIR


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


def main():
    env = ExpEnv(PROD6)  # 先只初始化数据 (用6因子)
    cal, u, daily_ret = env.cal, env.u, env.daily_ret
    # 全部候选的因子矩阵 (一次算全, 供各组合复用)
    all_names = list(PROD6) + CAND27
    all_F = dict(PROD6)
    for n in CAND27:
        all_F[n] = CAND_DIR[n]
    comp = env.engine.compute_factors(all_names, cal, u, parallel=True)
    ranks = {}
    for n, direction in all_F.items():
        r = comp[n].rank(axis=1, pct=True)
        ranks[n] = r if direction == 1 else (1 - r)
    fwd_rank = daily_ret.rank(axis=1)

    def evaluate(names):
        """对给定因子集做 IC_IR 加权回测, 返回 (全段夏普, OOS, 实盘)."""
        F = {n: all_F[n] for n in names}
        ic = pd.DataFrame({n: ranks[n].corrwith(fwd_rank, axis=1) for n in names})
        wmap = {}
        for t in cal:
            hist = ic.loc[:t].iloc[-60:]
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
        s = pd.Series({d: v for d, v in rets}).sort_index().dropna()
        st = stats(s)
        return st

    # ===== 基准: 6因子 IC_IR =====
    base6 = list(PROD6)
    st_base = evaluate(base6)
    print(f'基准 6因子-IC_IR: 夏普={st_base["sharpe"]:.2f} 回撤={st_base["mdd"]:.1%} OOS={st_base["oos"]:.2f} 实盘={st_base["live"]:.2f}')

    # ===== 步骤1: 逐个加入候选 (前向) =====
    print('\n=== 前向加入 (6->12) ===')
    current = list(base6)
    for step in range(6):
        best_sh, best_add, best_st = -1e9, None, None
        for c in CAND27:
            if c in current:
                continue
            st = evaluate(current + [c])
            # 优先实盘, 其次全段夏普
            score = st['live'] * 0.5 + st['sharpe'] * 0.5
            if score > best_sh:
                best_sh, best_add, best_st = score, c, st
        if best_add is None:
            break
        current.append(best_add)
        print(f'  加 {best_add:<40} 夏普={best_st["sharpe"]:.2f} 回撤={best_st["mdd"]:.1%} OOS={best_st["oos"]:.2f} 实盘={best_st["live"]:.2f}')
    print(f'前向加入最终 ({len(current)}因子): {current}')

    # ===== 步骤2: 替换搜索 (6因子逐个换候选) =====
    print('\n=== 替换搜索 (6因子 x 27候选) ===')
    best_swap, best_st_swap = None, None
    for i in range(len(base6)):
        for c in CAND27:
            if c in base6:
                continue
            swapped = list(base6)
            swapped[i] = c
            st = evaluate(swapped)
            score = st['live'] * 0.5 + st['sharpe'] * 0.5
            if best_st_swap is None or score > best_st_swap['live'] * 0.5 + best_st_swap['sharpe'] * 0.5:
                best_swap, best_st_swap = swapped, st
    if best_swap:
        print(f'最佳替换: {best_swap}')
        print(f'  夏普={best_st_swap["sharpe"]:.2f} 回撤={best_st_swap["mdd"]:.1%} OOS={best_st_swap["oos"]:.2f} 实盘={best_st_swap["live"]:.2f}')


if __name__ == '__main__':
    main()
