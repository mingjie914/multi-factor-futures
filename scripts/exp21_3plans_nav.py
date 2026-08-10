"""实验21: 三方案净值对比 (6icir / 12icir / 6等权).

统一框架: IC_IR(60日 LW) 或 等权合成 + cap3 选池 + 池内 ERC + 日度.
数据源: 当前 5min (drip_stone 强制 1min).
标注: OOS 起点 2026-03-01, 实盘起点 2026-05-16.
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
from exp_core import ExpEnv, stats, PROD6, CAND27, CAND_DIR
from exp18_light_forward import DIRS

# 三方案因子集
F6 = list(PROD6)
F12 = list(PROD6) + [
    'intraday_ma_count_bullish_20d', 'intraday_cross_vol_20d',
    'intraday_session_symmetry_20d', 'intraday_wash_trade_20d',
    'intraday_basis_momentum_20d', 'intraday_lowest_time_20d',
]


def lw(icm):
    T, N = icm.shape
    sc = np.cov(icm, rowvar=False, ddof=1)
    corr = np.corrcoef(icm, rowvar=False)
    avg = np.mean(corr[np.triu_indices(N, k=1)]) if N > 1 else 0
    tc = np.eye(N) * (1 - avg) + np.ones((N, N)) * avg
    sv = np.std(icm, axis=0, ddof=1)
    tgt = np.outer(sv, sv) * tc
    c = icm - icm.mean(axis=0)
    pi = sum(np.sum((c.iloc[i].values.reshape(-1, 1) @ c.iloc[i].values.reshape(1, -1) - sc) ** 2) for i in range(T)) / T
    g = np.sum((tgt - sc) ** 2)
    lam = max(0, min(1, pi / g)) if g > 0 else 0.5
    return lam * tgt + (1 - lam) * sc


class Runner:
    def __init__(self):
        self.env = ExpEnv(PROD6)
        self.cal, self.u, self.daily_ret = self.env.cal, self.env.u, self.env.daily_ret
        ALL = list(dict.fromkeys(list(PROD6) + CAND27 + F12))
        self.comp = {}
        # 分年段 compute: 避免 2537 天 1min 全量 (drip_stone FFT) 一次性内存峰值
        # (分年 drip_stone 正常 2-20% NaN, 全历史一起算会 OOM 返回全 NaN)
        chunk = 400  # 400 交易日一段
        for i in range(0, len(self.cal), chunk):
            sub = self.cal[i:i+chunk]
            for j in range(0, len(ALL), 10):
                part = self.env.engine.compute_factors(ALL[j:j+10], sub, self.u, parallel=False)
                for k, v in part.items():
                    if k not in self.comp:
                        self.comp[k] = v.reindex(self.cal)
                    else:
                        self.comp[k].loc[sub] = v
        self.ranks = {}
        for n in ALL:
            if n not in self.comp:
                continue
            r = self.comp[n].rank(axis=1, pct=True)
            d = DIRS.get(n, 1)
            self.ranks[n] = r if d == 1 else (1 - r)
        self.fwd = self.daily_ret.rank(axis=1)

    def returns(self, names, icir=True):
        ic = pd.DataFrame({n: self.ranks[n].corrwith(self.fwd, axis=1) for n in names})
        rets = []
        for t in self.cal:
            if icir:
                hist = ic.loc[:t].iloc[-60:-1]
                if len(hist) < 30:
                    w = pd.Series(1.0 / len(names), index=names)
                else:
                    im = hist.mean()
                    lwc = lw(hist)
                    try:
                        wi = np.linalg.inv(lwc) @ im.values
                    except np.linalg.LinAlgError:
                        wi = im.abs().values
                    wi = np.abs(wi)
                    w = pd.Series(wi / wi.sum(), index=names)
            else:
                w = pd.Series(1.0 / len(names), index=names)
            sc = pd.Series(0.0, index=self.u)
            for n in names:
                if t in self.ranks[n].index:
                    sc = sc.add(self.ranks[n].loc[t] * w[n], fill_value=0)
            tot = sc.sum()
            if tot > 0:
                sc = sc / tot
            sc = sc.dropna()
            if len(sc) < 20:
                continue
            top = self.env.capped(sc, ascending=False)
            bot = self.env.capped(sc, ascending=True)
            wl = self.env.erc_w(top, t) or {}
            ws = self.env.erc_w(bot, t) or {}
            if t in self.daily_ret.index:
                rr = self.daily_ret.loc[t].fillna(0.0)
                rets.append((t, sum(rr[c] * wi for c, wi in wl.items()) - sum(rr[c] * wi for c, wi in ws.items())))
        return pd.Series({d: v for d, v in rets}).sort_index().dropna()


def main():
    import time as _time
    import cProfile as _cProfile
    import pstats as _pstats
    import io as _io
    r = Runner()
    print('=' * 60)
    print('实验21: 三方案净值对比')
    print('=' * 60)
    _pr = _cProfile.Profile()
    _pr.enable()
    plans = {
        '6因子-IC_IR (生产)': (F6, True),
        '12因子-IC_IR (前向候选)': (F12, True),
        '6因子-等权 (旧生产)': (F6, False),
    }
    fig, ax = plt.subplots(figsize=(13, 7))
    results = {}
    for name, (names, icir) in plans.items():
        s = r.returns(names, icir)
        st = stats(s)
        # 净值图从 2016-03-31 起 (用户要求), 回测数据含前置冷启动
        s_plot = s[s.index >= pd.Timestamp('2016-03-31')]
        nav = (1 + s_plot).cumprod()
        ax.plot(nav.index, nav.values, lw=1.6,
                label=f'{name} (夏普{st["sharpe"]:.2f}/实盘{st["live"]:.2f})')
        results[name] = st
        print(f'{name}: 夏普={st["sharpe"]:.2f} 年化={st["ann"]:.1%} 回撤={st["mdd"]:.1%} '
              f'OOS={st["oos"]:.2f} 实盘={st["live"]:.2f} ({len(s)}天)')
    ax.axvline(pd.Timestamp('2026-03-01'), color='gray', ls='--', lw=1, label='OOS起点')
    ax.axvline(pd.Timestamp('2026-05-16'), color='red', ls='--', lw=1, label='实盘起点')
    ax.set_title('三方案净值对比 (6icir / 12icir / 6等权, 2016-03-31 起)')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = 'runs/nav_3plans_compare.png'
    fig.savefig(out, dpi=150)
    print(f'\n净值对比图: {out}')
    # 性能剖析 (任务4)
    _pr.disable()
    _st = _io.StringIO()
    _ps = _pstats.Stats(_pr, stream=_st).sort_stats('cumulative')
    _ps.print_stats(12)
    print('\n=== cProfile 性能热点 (三方案回测) ===')
    print(_st.getvalue()[:2500])


if __name__ == '__main__':
    main()
