"""12因子 IC_IR 权重导出 (prod_snapshot_12f_icir 专用).

输出: weights_daily.csv (date|symbol|weight, 信号日, 用于 T+1 执行)
口径: T日信号 = 因子滚动IC_IR加权合成, 选池 cap3, 池内等权, 杠杆 2.0
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from combined import FACTORS, UNIVERSE, SECTOR_MAP, SECTOR_CAP, IC_IR_WINDOW
from exp_core import ExpEnv


def capped(row, ascending, cap=SECTOR_CAP, sector_map=SECTOR_MAP, universe=UNIVERSE):
    """cap=3 板块配额选池: 全市场排名, 每板块最多 cap 个."""
    if sector_map is None:
        return row.sort_values(ascending=not ascending).index[:10]
    picks = []
    sector_cnt = {}
    for sym in row.sort_values(ascending=not ascending).index:
        if sym not in universe:
            continue
        sector = next((k for k, v in sector_map.items() if sym in v), '其他')
        if sector_cnt.get(sector, 0) >= cap:
            continue
        picks.append(sym)
        sector_cnt[sector] = sector_cnt.get(sector, 0) + 1
        if len(picks) >= 10:
            break
    return picks


def erc_w(score_row, t, cov_lookback=60, shrinkage=0.3):
    """ERC 权重 (等风险贡献)."""
    try:
        import scipy.optimize as opt
        n = len(score_row)
        # 用过去收益协方差
        env = _ENV
        hist = env.daily_ret.loc[:t].iloc[-cov_lookback:][score_row.index]
        cov = hist.cov().values
        if cov.shape != (n, n):
            return None
        # Ledoit-Wolf 收缩
        sample_corr = np.corrcoef(hist.values, rowvar=False)
        avg_corr = np.mean(sample_corr[np.triu_indices(n, k=1)]) if n > 1 else 0.0
        target = np.eye(n) * (1 - avg_corr) + np.ones((n, n)) * avg_corr
        std_v = np.std(hist.values, axis=0, ddof=1)
        tgt_cov = np.outer(std_v, std_v) * target
        cov = (1 - shrinkage) * cov + shrinkage * tgt_cov

        def mrc(w):
            pw = np.sqrt(np.maximum(w @ cov @ w, 1e-12))
            return (cov @ w) / pw

        def obj(w):
            m = mrc(w)
            return np.sum((m - m.mean()) ** 2)

        w0 = np.ones(n) / n
        bounds = [(0.02, 0.5)] * n
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        res = opt.minimize(obj, w0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 200, 'ftol': 1e-8})
        if not res.success:
            return None
        w = res.x
        return pd.Series(w / w.sum(), index=score_row.index)
    except Exception:
        return None


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
    global _ENV
    env = ExpEnv(FACTORS)
    _ENV = env
    cal, u, daily_ret = env.cal, env.u, env.daily_ret
    names = list(FACTORS)
    comp = {}
    for i in range(0, len(names), 10):
        part = env.engine.compute_factors(names[i:i+10], cal, u, parallel=False)
        for k, v in part.items():
            comp[k] = v
    ranks = {}
    for n, direction in FACTORS.items():
        if n not in comp:
            print(f'[skip] {n}')
            continue
        r = comp[n].rank(axis=1, pct=True)
        ranks[n] = r if direction == 1 else (1 - r)
    fwd_rank = daily_ret.rank(axis=1)
    ic = pd.DataFrame({n: ranks[n].corrwith(fwd_rank, axis=1) for n in names})

    rows = []
    for t in cal:
        hist = ic.loc[:t].iloc[-IC_IR_WINDOW:]
        if len(hist) < 20:
            continue
        ic_mean = hist.mean()
        lw_cov = ledoit_wolf_cov(hist)
        try:
            wi = np.linalg.inv(lw_cov) @ ic_mean.values
        except np.linalg.LinAlgError:
            wi = ic_mean.abs().values
        wi = np.abs(wi)
        wi = wi / wi.sum()
        # 合成得分
        sc = pd.Series(0.0, index=u)
        for n in names:
            if t in ranks[n].index:
                sc = sc.add(ranks[n].loc[t] * wi[names.index(n)], fill_value=0)
        tot = sc.sum()
        if tot > 0:
            sc = sc / tot
        sc = sc.dropna()
        if len(sc) < 20:
            continue
        top = capped(sc, ascending=False)
        bot = capped(sc, ascending=True)
        wl = erc_w(sc.loc[top], t)
        ws = erc_w(sc.loc[bot], t)
        wl = wl if wl is not None else pd.Series(1.0 / len(top), index=top)
        ws = ws if ws is not None else pd.Series(1.0 / len(bot), index=bot)
        for sym, w in wl.items():
            rows.append({'date': t.strftime('%Y-%m-%d'), 'symbol': sym, 'weight': w})
        for sym, w in ws.items():
            rows.append({'date': t.strftime('%Y-%m-%d'), 'symbol': sym, 'weight': -w})
    df = pd.DataFrame(rows)
    df.to_csv('weights_daily.csv', index=False, encoding='utf-8')
    print(f'已导出 weights_daily.csv: {len(df)} 行, {df.date.nunique()} 交易日')
    # 校验
    bal = df.groupby('date').weight.sum()
    print(f'每日权重和: {bal.mean():.4f}±{bal.std():.4f} (应≈0)')


if __name__ == '__main__':
    main()
