"""实验3: MOSEK式组合优化 — 中金组合优化核心.

对比 (6因子, 38池, 日度):
  A. 基线: cap3选池 + 池内ERC (当前生产, 2.04)
  B. MOSEK式: 全品种均值-方差优化 (Σw=0, 波动目标10%, 换手≤30%, |w_i|≤2.0)
  C. MOSEK式: 只在选池内(20个)优化 (中金在因子alpha全品种上优化)

用 cvxpy 实现二次规划 (MOSEK求解器不可用, 用OSQP/SCS替代, 数学等价)
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import cvxpy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from exp_core import ExpEnv, stats, format_stats, PROD6


def main():
    env = ExpEnv(PROD6)
    cal, u, daily_ret = env.cal, env.u, env.daily_ret
    score = env.compute_scores()

    def mosek_opt(alpha, cov, prev_w, lam=0.5, vol_target=0.10, turn_max=0.30, gross=2.0):
        """均值-方差优化 (cvxpy). alpha: 截面得分, cov: 60日协方差, prev_w: 前日权重."""
        n = len(alpha)
        w = cp.Variable(n)
        # 波动约束: w'Σw ≤ vol_target^2 / 252 (日度)
        Sigma = cov
        objective = cp.Maximize(alpha.values @ w - lam * cp.quad_form(w, Sigma))
        constraints = [
            cp.sum(w) == 0,
            cp.quad_form(w, Sigma) <= (vol_target / np.sqrt(252)) ** 2,
            cp.norm1(w - prev_w) <= turn_max,
            cp.norm_inf(w) <= gross / n,
        ]
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
            if prob.status in ('optimal', 'optimal_inaccurate'):
                return pd.Series(w.value, index=alpha.index)
        except Exception:
            pass
        # 回退: 归一化 alpha (多空)
        a = alpha - alpha.mean()
        s = a.abs().sum()
        return a / s if s > 0 else pd.Series(0.0, index=alpha.index)

    def backtest_b(mode='b'):
        """B: 全品种优化; C: 选池内优化."""
        rets = []
        prev_w = pd.Series(0.0, index=u)
        for t in score.index:
            row = score.loc[t].dropna()
            if len(row) < 20:
                continue
            # 60日协方差 (全品种或池内)
            sd = t - pd.Timedelta(days=90)
            c = pd.DatetimeIndex(env.runner.data_manager.get_calendar(sd, t))
            if mode == 'b':
                pool = u
                alpha = row.reindex(u).fillna(0.0)
            else:  # c: 选池内
                top = env.capped(row, ascending=False)
                bot = env.capped(row, ascending=True)
                pool = list(dict.fromkeys(top + bot))
                alpha = row.reindex(pool)
                prev_w_pool = prev_w.reindex(pool).fillna(0.0)
            ret_sub = daily_ret.reindex(c)[pool].dropna()
            if ret_sub.shape[0] < 20:
                continue
            cov = ret_sub.cov().values
            cov = 0.7 * cov + 0.3 * np.diag(np.diag(cov))
            prev_pool = prev_w.reindex(pool).fillna(0.0)
            w_t = mosek_opt(alpha, pd.DataFrame(cov, index=pool, columns=pool), prev_pool)
            if t in daily_ret.index:
                r = daily_ret.loc[t].fillna(0.0)
                gross = sum(r[c] * w for c, w in w_t.items())
                rets.append((t, gross))
                prev_w = prev_w.add(w_t.reindex(prev_w.index).fillna(0.0), fill_value=0)
        return pd.Series({d: v for d, v in rets}).sort_index().dropna()

    # A. 基线
    def backtest_base():
        rets = []
        for t in score.index:
            row = score.loc[t].dropna()
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

    print('=== 实验3: MOSEK式组合优化 ===')
    sA = backtest_base()
    print(f'A 基线(ERC): {format_stats(stats(sA))}')
    sB = backtest_b('b')
    print(f'B 全品种优化: {format_stats(stats(sB))}')
    sC = backtest_b('c')
    print(f'C 选池内优化: {format_stats(stats(sC))}')

    fig, ax = plt.subplots(figsize=(15, 8))
    for s, lab, c in [(sA, 'A 基线(ERC)', '#2ecc71'), (sB, 'B 全品种优化', '#e67e22'), (sC, 'C 选池内优化', '#3498db')]:
        nav = (1 + s).cumprod()
        ax.plot(nav.index, nav.values, label=lab, color=c, linewidth=1.5)
    ax.axvline(pd.Timestamp('2026-03-01'), color='gray', ls='--', alpha=0.8, label='OOS起点')
    ax.axvline(pd.Timestamp('2026-05-16'), color='red', ls='--', alpha=0.8, label='实盘起点')
    ax.set_title('实验3: MOSEK式组合优化 (6因子)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('runs/exp3_mosek.png', dpi=150)
    print('\n净值图: runs/exp3_mosek.png')


if __name__ == '__main__':
    main()
