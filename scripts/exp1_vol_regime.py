"""实验1: 高低波分域 — 中金研报核心工程点.

对比:
  A. 基线: 6因子 无分域 (当前生产, 夏普2.04)
  B. 高波加权: 高波日仓位×1.5, 低波日×0.5 (波动缩放)
  C. 仅高波交易: 低波日空仓 (只在高波日持仓)
  D. 高波信号过滤: 只在|IC|更强的高波域用信号, 低波日减半

方法: 20日已实现波动率 + 240日滚动75%分位 → D_i,t ∈ {高波,低波}
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

    # ===== 高低波分域: 每品种 20日RV + 240日75%分位 =====
    # 用日收益滚动: RV20 = std(ret, 20) * sqrt(252)
    rv20 = daily_ret.rolling(20, min_periods=10).std(ddof=0) * np.sqrt(252)
    # 240日滚动75%分位 (逐日逐品种)
    q75 = rv20.rolling(240, min_periods=60).quantile(0.75)
    high_vol = rv20 > q75  # True=高波日
    # 分域覆盖检查
    cov = high_vol.dropna(how='all')
    print(f'高波分域: 有效样本 {cov.notna().sum().sum()}, 高波占比 {high_vol.sum().sum()/high_vol.notna().sum().sum():.1%}')

    # ===== 回测函数 (支持分域权重) =====
    def backtest(wscale=None):
        """wscale: dict 品种→每日子数组 (缩放系数). None=无分域."""
        rets = []
        for t in score.index:
            row = score.loc[t].dropna()
            if len(row) < 20:
                continue
            top = env.capped(row, ascending=False)
            bot = env.capped(row, ascending=True)
            wl = env.erc_w(top, t) or {}
            ws = env.erc_w(bot, t) or {}
            # 分域缩放
            if wscale is not None:
                for sym in list(wl):
                    s_ = wscale.get(sym, None)
                    if s_ is not None:
                        wl[sym] *= s_.get(t, 1.0) if hasattr(s_, 'get') else s_
                for sym in list(ws):
                    s_ = wscale.get(sym, None)
                    if s_ is not None:
                        ws[sym] *= s_.get(t, 1.0) if hasattr(s_, 'get') else s_
            if t in daily_ret.index:
                r = daily_ret.loc[t].fillna(0.0)
                lr = sum(r[c] * wi for c, wi in wl.items())
                sr = sum(r[c] * wi for c, wi in ws.items())
                rets.append((t, lr - sr))
        return pd.Series({d: v for d, v in rets}).sort_index().dropna()

    # A. 基线
    sA = backtest()
    stA = stats(sA)

    # B. 高波加权: 高波日×1.5, 低波日×0.5 (对每个品种)
    def scale_band(dict_scale):
        return dict_scale
    hv_b = {}
    for sym in u:
        if sym in high_vol.columns:
            hv_b[sym] = high_vol[sym].map(lambda x: 1.5 if x else 0.5)
    sB = backtest(hv_b)
    stB = stats(sB)

    # C. 仅高波交易: 低波日权重=0
    hv_c = {}
    for sym in u:
        if sym in high_vol.columns:
            hv_c[sym] = high_vol[sym].map(lambda x: 1.0 if x else 0.0)
    sC = backtest(hv_c)
    stC = stats(sC)

    print('\n=== 实验1: 高低波分域对比 ===')
    print(f'A 基线(无分域): {format_stats(stA)}')
    print(f'B 高波×1.5/低波×0.5: {format_stats(stB)}')
    print(f'C 仅高波交易: {format_stats(stC)}')

    # 净值图
    fig, ax = plt.subplots(figsize=(15, 8))
    for s, lab, c in [(sA, 'A基线(无分域)', '#2ecc71'), (sB, 'B高波加权', '#e67e22'), (sC, 'C仅高波', '#3498db')]:
        nav = (1 + s).cumprod()
        ax.plot(nav.index, nav.values, label=lab, color=c, linewidth=1.6)
    ax.axvline(pd.Timestamp('2026-03-01'), color='gray', ls='--', alpha=0.8, label='OOS起点')
    ax.axvline(pd.Timestamp('2026-05-16'), color='red', ls='--', alpha=0.8, label='实盘起点')
    ax.set_title('实验1: 高低波分域对比 (6因子, 38池, cap3, ERC, 日度)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('runs/exp1_vol_regime.png', dpi=150)
    print('\n净值图: runs/exp1_vol_regime.png')

    # 高波 vs 低波 IC 对比 (验证中金结论)
    print('\n=== 高波/低波 IC 对比 (验证中金: 高波IC更高?) ===')
    fwd5 = daily_ret.rolling(5).mean().shift(-5)
    for name in list(PROD6)[:3]:
        f = env._comp[name]
        ics_h, ics_l = [], []
        for t in score.index:
            if t not in high_vol.index:
                continue
            row = f.loc[t].dropna()
            fw = fwd5.loc[t].dropna()
            common = row.index.intersection(fw.index)
            if len(common) < 10:
                continue
            ic = row[common].corr(fw[common], method='spearman')
            is_h = high_vol.loc[t].reindex(common).mean() > 0.5
            if is_h:
                ics_h.append(ic)
            else:
                ics_l.append(ic)
        print(f'{name}: 高波IC均值={np.mean(ics_h):+.4f} (n={len(ics_h)}), 低波IC均值={np.mean(ics_l):+.4f} (n={len(ics_l)})')


if __name__ == '__main__':
    main()
