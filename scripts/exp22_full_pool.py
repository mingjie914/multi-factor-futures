"""实验22: 全因子池无预设筛选 (验证新因子 vs 6因子 vs 组合更优).

因子池: 6生产 + 47旧候选 + 21新候选(#439-530通过FDR) = 74.
路径:
  A. 相关性去重: 与生产6因子 corr>=0.5 剔除 (同族冗余)
  B1. 从6因子出发前向加 (含替换: 允许换掉弱因子)
  B2. 从空集前向选 (完全不预设, 自然选最优)
  C. 跨样本验证: 全段/OOS/实盘
轻量指标: Rank IC均值xICIR (corr计算, 快), 相关性门槛防同族.
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import time
import factors.library.intraday as ID
from exp_core import ExpEnv, stats, PROD6
from exp18_light_forward import DIRS, KEPT47

# 21 新候选 (#439-530 通过 FDR)
NEW21 = [
    'intraday_lead_amount_surge_20d', 'intraday_price_diff_autocorr_20d',
    'intraday_diff_autocorr_long_20d', 'intraday_order_flow_autocorr_20d',
    'intraday_vp_corr_high_freq_20d', 'intraday_peak_moment_count_20d',
    'intraday_order_flow_memory_20d', 'intraday_jump_ret_follow_ratio_20d',
    'intraday_neg_ret_illiq_20d', 'intraday_ret_extreme_magnitude_20d',
    'intraday_peak_ridge_coherence_20d', 'intraday_smart_money_v4_vol_20d',
    'intraday_torrent_down_20d', 'intraday_jump_amount_lagcorr_20d',
    'intraday_csad_sigma120_20d', 'intraday_vol_bucket_entropy_20d',
    'intraday_trajectory_illiq_20d', 'intraday_vol_flow_vol_20d',
    'intraday_price_peak_interval_std_20d', 'intraday_following_price_confirm_20d',
    'intraday_flow_ret_resid_vol_20d',
]

# 新候选方向: 从检验 t 值符号 (负t -> 方向-1)
NEW21_DIR = {
    'intraday_lead_amount_surge_20d': -1, 'intraday_price_diff_autocorr_20d': -1,
    'intraday_diff_autocorr_long_20d': 1, 'intraday_order_flow_autocorr_20d': 1,
    'intraday_vp_corr_high_freq_20d': 1, 'intraday_peak_moment_count_20d': -1,
    'intraday_order_flow_memory_20d': 1, 'intraday_jump_ret_follow_ratio_20d': 1,
    'intraday_neg_ret_illiq_20d': -1, 'intraday_ret_extreme_magnitude_20d': 1,
    'intraday_peak_ridge_coherence_20d': -1, 'intraday_smart_money_v4_vol_20d': 1,
    'intraday_torrent_down_20d': -1, 'intraday_jump_amount_lagcorr_20d': 1,
    'intraday_csad_sigma120_20d': 1, 'intraday_vol_bucket_entropy_20d': 1,
    'intraday_trajectory_illiq_20d': -1, 'intraday_vol_flow_vol_20d': 1,
    'intraday_price_peak_interval_std_20d': 1, 'intraday_following_price_confirm_20d': 1,
    'intraday_flow_ret_resid_vol_20d': 1,
}


class Runner:
    def __init__(self):
        self.env = ExpEnv(PROD6)
        self.cal, self.u, self.daily_ret = self.env.cal, self.env.u, self.env.daily_ret
        ALL = list(dict.fromkeys(list(PROD6) + KEPT47 + NEW21))
        # 分年段 compute (drip_stone FFT 内存)
        self.comp = {}
        chunk = 400
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
            d = DIRS.get(n, NEW21_DIR.get(n, 1))
            self.ranks[n] = r if d == 1 else (1 - r)
        self.fwd = self.daily_ret.rank(axis=1)
        self.ic = pd.DataFrame({n: self.ranks[n].corrwith(self.fwd, axis=1) for n in ALL})

    def light_score(self, names):
        ic_sub = self.ic[names].mean(axis=1)
        im = ic_sub.mean()
        ist = ic_sub.std(ddof=0)
        return im * (im / ist if ist > 0 else 0)

    def corr(self, a, b):
        return self.ic[a].corr(self.ic[b])


def main():
    r = Runner()
    print('=' * 60)
    print('实验22: 全因子池无预设筛选')
    print('=' * 60)
    ALL = list(dict.fromkeys(list(PROD6) + KEPT47 + NEW21))
    print(f'全池: {len(ALL)} 因子 (6生产 + {len(KEPT47)}旧候选 + {len(NEW21)}新候选)', flush=True)

    # A. 相关性去重: 与生产6因子 corr>=0.5 剔除
    print('\n--- A. 与生产6因子相关性去重 (corr>=0.5) ---')
    kept = []
    dropped = []
    for n in ALL:
        if n in PROD6:
            kept.append(n)
            continue
        maxc = max(abs(r.corr(n, p)) for p in PROD6)
        if maxc >= 0.5:
            dropped.append((n, maxc))
        else:
            kept.append(n)
    print(f'保留 {len(kept)} 独立, 剔除 {len(dropped)} 高相关:')
    for n, c in sorted(dropped, key=lambda x: -x[1])[:10]:
        print(f'  {n}: corr={c:.2f}')

    # B1. 从6因子出发前向 (轻量 + 相关性门槛)
    print('\n--- B1. 从6因子出发前向 (轻量IC, 相关门槛) ---')
    base6 = list(PROD6)
    s0 = r.light_score(base6)
    print(f'基准 6因子: score={s0:.4f}')
    cur = list(base6)
    chosen = []
    for step in range(8):
        best, best_n = -1e9, None
        for c in kept:
            if c in cur:
                continue
            if max(abs(r.corr(c, s)) for s in cur) >= 0.5:
                continue
            sc = r.light_score(cur + [c])
            if sc > best:
                best, best_n = sc, c
        if best_n is None or best <= s0:
            break
        cur.append(best_n)
        chosen.append(best_n)
        s0 = best
        print(f'  +{best_n:<42} score={best:.4f} ({len(cur)}因子)', flush=True)
    print(f'B1 最终 ({len(cur)}): {cur}')

    # B2. 从空集前向 (完全不预设)
    print('\n--- B2. 从空集前向 (无预设) ---')
    cur2 = []
    s2 = -1e9
    for step in range(8):
        best, best_n = -1e9, None
        for c in kept:
            if c in cur2:
                continue
            if cur2 and max(abs(r.corr(c, s)) for s in cur2) >= 0.5:
                continue
            sc = r.light_score(cur2 + [c])
            if sc > best:
                best, best_n = sc, c
        if best_n is None:
            break
        cur2.append(best_n)
        s2 = best
        print(f'  +{best_n:<42} score={best:.4f} ({len(cur2)}因子)', flush=True)
    print(f'B2 最终 ({len(cur2)}): {cur2}')

    # C. 跨样本全回测验证 (3 组合 + 基准)
    print('\n--- C. 跨样本全回测验证 ---')
    import numpy as _np
    def _lw(icm):
        T, N = icm.shape
        sc = _np.cov(icm, rowvar=False, ddof=1)
        corr = _np.corrcoef(icm, rowvar=False)
        avg = _np.mean(corr[_np.triu_indices(N, k=1)]) if N > 1 else 0
        tc = _np.eye(N)*(1-avg) + _np.ones((N,N))*avg
        sv = _np.std(icm, axis=0, ddof=1)
        tgt = _np.outer(sv, sv)*tc
        c = icm - icm.mean(axis=0)
        pi = sum(_np.sum((c.iloc[i].values.reshape(-1,1) @ c.iloc[i].values.reshape(1,-1) - sc)**2) for i in range(T))/T
        g = _np.sum((tgt - sc)**2)
        lam = max(0, min(1, pi/g)) if g > 0 else 0.5
        return lam*tgt + (1-lam)*sc
    candidates = {'6因子(生产)': base6, 'B1 结果': cur, 'B2 结果': cur2}
    for name, names in candidates.items():
        ic = r.ic[names]
        rets = []
        for t in r.cal:
            hist = ic.loc[:t].iloc[-60:-1]  # 不含 ic[T] (防同日泄漏)
            # 剔除 IC 全 NaN 的因子 (该段无信号, 如 seat 2020 前)
            hist = hist.dropna(axis=1, how='all')
            if hist.shape[1] < 2:
                continue
            if len(hist) < 30:
                w = pd.Series(1.0/hist.shape[1], index=hist.columns)
            else:
                im = hist.mean()
                lwc = _lw(hist)
                try:
                    wi = np.linalg.inv(lwc) @ im.values
                except np.linalg.LinAlgError:
                    wi = im.abs().values
                wi = np.abs(wi)
                w = pd.Series(wi/wi.sum(), index=hist.columns)
            names_eff = list(hist.columns)
            sc = pd.Series(0.0, index=r.u)
            for n in names_eff:
                if t in r.ranks[n].index:
                    sc = sc.add(r.ranks[n].loc[t].fillna(0.0)*w[n], fill_value=0)  # 因子缺失不污染其他
            tot = sc.sum()
            if tot > 0:
                sc = sc/tot
            sc = sc.dropna()
            if len(sc) < 20:
                continue
            top = r.env.capped(sc, ascending=False)
            bot = r.env.capped(sc, ascending=True)
            wl = r.env.erc_w(top, t) or {}
            ws = r.env.erc_w(bot, t) or {}
            if t in r.daily_ret.index:
                rr = r.daily_ret.loc[t].fillna(0.0)
                rets.append((t, sum(rr[c]*wi for c, wi in wl.items()) - sum(rr[c]*wi for c, wi in ws.items())))
        s = pd.Series({d: v for d, v in rets}).sort_index().dropna()
        st = stats(s)
        print(f'{name} ({len(names)}因子): 夏普={st["sharpe"]:.2f} 回撤={st["mdd"]:.1%} OOS={st["oos"]:.2f} 实盘={st["live"]:.2f}')


if __name__ == '__main__':
    main()
