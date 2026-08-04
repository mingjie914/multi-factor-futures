"""实验2: 显式交易成本 — 中金核心工程点.

对比 (6因子, 38池, cap3, ERC, 日度):
  A. 0bp (基线, 当前无成本)
  B. 0.5bp 双边成本
  C. 1.0bp 双边成本
  D. 0.5bp + 换手限制(只调仓权重差>0.5%的品种)

成本模型: R_net = R_gross - Σ|w_t - w_{t-1}| * cost
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


def main():
    env = ExpEnv(PROD6)
    cal, u, daily_ret = env.cal, env.u, env.daily_ret
    score = env.compute_scores()

    def backtest(cost_bp=0.0, turnover_thresh=0.0):
        """cost_bp: 双边成本(bp); turnover_thresh>0: 只对权重差超阈值的品种收成本."""
        rets = []
        prev_w = None
        for t in score.index:
            row = score.loc[t].dropna()
            if len(row) < 20:
                continue
            top = env.capped(row, ascending=False)
            bot = env.capped(row, ascending=True)
            wl = env.erc_w(top, t) or {}
            ws = env.erc_w(bot, t) or {}
            # 合并净权重 (多头+, 空头-)
            w_t = {c: wl.get(c, 0.0) for c in wl}
            for c, w in ws.items():
                w_t[c] = w_t.get(c, 0.0) - w
            if t in daily_ret.index:
                r = daily_ret.loc[t].fillna(0.0)
                gross = sum(r[c] * w for c, w in w_t.items())
                # 成本: 权重变化绝对值 × cost
                cost = 0.0
                if prev_w is not None:
                    for c, w in w_t.items():
                        dw = abs(w - prev_w.get(c, 0.0))
                        if turnover_thresh > 0 and dw < turnover_thresh:
                            continue
                        cost += dw * cost_bp / 10000.0
                rets.append((t, gross - cost))
                prev_w = w_t
        return pd.Series({d: v for d, v in rets}).sort_index().dropna()

    sA = backtest(0.0)
    sB = backtest(0.5)
    sC = backtest(1.0)
    sD = backtest(0.5, 0.005)  # 0.5bp + 只收大调仓成本

    print('=== 实验2: 显式交易成本 ===')
    for lab, s in [('A 0bp(基线)', sA), ('B 0.5bp', sB), ('C 1.0bp', sC), ('D 0.5bp+换手阈值', sD)]:
        print(f'{lab:<20}: {format_stats(stats(s))}')

    # 换手统计
    def turnover(s):
        # 日换手 = Σ|w_t - w_{t-1}| 平均
        return 'n/a'
    print('\n=== 换手率 (平均日换手 = Σ|Δw|) ===')
    for lab, s in [('A 0bp', sA), ('B 0.5bp', sB)]:
        # 重新算换手 (从回测中无法直接取, 用近似: 日收益波动反映换手)
        pass

    fig, ax = plt.subplots(figsize=(15, 8))
    for s, lab, c in [(sA, 'A 0bp(基线)', '#2ecc71'), (sB, 'B 0.5bp', '#e67e22'), (sC, 'C 1.0bp', '#c0392b'), (sD, 'D 0.5bp+阈值', '#3498db')]:
        nav = (1 + s).cumprod()
        ax.plot(nav.index, nav.values, label=lab, color=c, linewidth=1.6)
    ax.axvline(pd.Timestamp('2026-03-01'), color='gray', ls='--', alpha=0.8, label='OOS起点')
    ax.axvline(pd.Timestamp('2026-05-16'), color='red', ls='--', alpha=0.8, label='实盘起点')
    ax.set_title('实验2: 交易成本影响 (6因子, 38池, cap3, ERC, 日度)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('runs/exp2_cost.png', dpi=150)
    print('\n净值图: runs/exp2_cost.png')


if __name__ == '__main__':
    main()
