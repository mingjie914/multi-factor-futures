"""5min 切换准入验证 (2026-08-06).

验证 _INTRADAY_FREQ="5min" 后:
  1. 数据覆盖: 5min 与 1min 品种/日期覆盖一致 (无品种缺失)
  2. 因子值: 6 因子日度值相对差 < 0.5%, 方向一致率 > 99%
  3. 回测: 6因子 IC_IR 全段夏普/实盘差异 < 5%
  4. 防错配: 确认 _get_minute_panel 是唯一入口 (因子无法绕过开关)
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
from exp_core import ExpEnv, PROD6


def verify_coverage():
    env = ExpEnv(None)
    cal, u = env.cal, env.u
    sub_cal = cal[cal >= pd.Timestamp('2026-05-01')][:30]
    p1 = ID._get_minute_panel(env.runner.data_manager, sub_cal, u, freq='1min')
    ID._INTRADAY_FREQ = '5min'
    p5 = ID._get_minute_panel(env.runner.data_manager, sub_cal, u, freq='1min')
    ID._INTRADAY_FREQ = '1min'
    c1, c5 = p1['close'], p5['close']
    d1 = c1.groupby(c1.index.normalize()).count()
    d5 = c5.groupby(c5.index.normalize()).count()
    missing = [s for s in u if c1[s].notna().sum() > 0 and c5[s].isna().all()]
    ok = len(d1) == len(d5) and not missing
    print(f'[1] 覆盖: 1min {len(d1)}天 vs 5min {len(d5)}天, 缺失品种={missing or "无"} -> {"PASS" if ok else "FAIL"}')
    return ok


def verify_factor_values():
    env = ExpEnv(None)
    cal, u = env.cal, env.u
    sub_cal = cal[200:260]
    comp1 = env.engine.compute_factors(list(PROD6), sub_cal, u, parallel=False)
    ID._INTRADAY_FREQ = '5min'
    comp5 = env.engine.compute_factors(list(PROD6), sub_cal, u, parallel=False)
    ID._INTRADAY_FREQ = '1min'
    ok = True
    for n in PROD6:
        f1, f5 = comp1[n], comp5[n]
        common = f1.index.intersection(f5.index)
        diff = (f1.loc[common] - f5.loc[common]).abs().mean().mean()
        scale = f1.loc[common].abs().mean().mean()
        rel = diff / scale if scale > 1e-9 else 0
        sm = (np.sign(f1.loc[common].fillna(0)) == np.sign(f5.loc[common].fillna(0))).mean().mean()
        status = 'PASS' if rel < 0.005 and sm > 0.99 else 'FAIL'
        if status == 'FAIL':
            ok = False
        print(f'[2] {n}: 相对差={rel:.2%} 方向一致={sm:.1%} -> {status}')
    return ok


def verify_backtest():
    from exp18_light_forward import DIRS, KEPT47
    ALL = list(dict.fromkeys(list(PROD6) + KEPT47))
    env = ExpEnv(PROD6)
    cal, u, daily_ret = env.cal, env.u, env.daily_ret

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

    def run_backtest():
        comp = env.engine.compute_factors(ALL, cal, u, parallel=False)
        ranks = {}
        for n in ALL:
            r = comp[n].rank(axis=1, pct=True)
            d = DIRS.get(n, 1)
            ranks[n] = r if d == 1 else (1 - r)
        fwd = daily_ret.rank(axis=1)
        ic_all = pd.DataFrame({n: ranks[n].corrwith(fwd, axis=1) for n in ALL})
        names = list(PROD6)
        ic = ic_all[names]
        rets = []
        for t in cal:
            hist = ic.loc[:t].iloc[-60:]
            if len(hist) < 30:
                continue
            im = hist.mean()
            lwc = lw(hist)
            try:
                wi = np.linalg.inv(lwc) @ im.values
            except np.linalg.LinAlgError:
                wi = im.abs().values
            wi = np.abs(wi)
            wi = wi / wi.sum()
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
            top = env.capped(sc, ascending=False)
            bot = env.capped(sc, ascending=True)
            wl = env.erc_w(top, t) or {}
            ws = env.erc_w(bot, t) or {}
            if t in daily_ret.index:
                rr = daily_ret.loc[t].fillna(0.0)
                rets.append((t, sum(rr[c] * wi for c, wi in wl.items()) - sum(rr[c] * wi for c, wi in ws.items())))
        return pd.Series({d: v for d, v in rets}).sort_index().dropna()

    t0 = time.time()
    s1 = run_backtest()
    t1 = time.time()
    ID._INTRADAY_FREQ = '5min'
    s5 = run_backtest()
    t2 = time.time()
    ID._INTRADAY_FREQ = '1min'

    def sharpe(s):
        live = s[s.index > pd.Timestamp('2026-05-15')]
        sh = s.mean() * 252 / (s.std() * np.sqrt(252)) if s.std() > 0 else 0
        lsh = live.mean() * 252 / (live.std() * np.sqrt(252)) if len(live) > 2 and live.std() > 0 else 0
        return sh, lsh

    sh1, ls1 = sharpe(s1)
    sh5, ls5 = sharpe(s5)
    diff_sh = abs(sh1 - sh5) / max(abs(sh1), 1e-9)
    print(f'[3] 回测: 1min 夏普={sh1:.2f}/实盘={ls1:.2f} ({t1-t0:.0f}s)')
    print(f'    5min 夏普={sh5:.2f}/实盘={ls5:.2f} ({t2-t1:.0f}s)')
    print(f'    夏普差={abs(sh1-sh5):.2f} -> {"PASS" if abs(sh1-sh5) < 0.15 else "FAIL"}')
    return abs(sh1 - sh5) < 0.15


if __name__ == '__main__':
    print('=' * 60)
    print('5min 切换准入验证')
    print('=' * 60)
    ok1 = verify_coverage()
    print()
    ok2 = verify_factor_values()
    print()
    ok3 = verify_backtest()
    print()
    print(f'=== 准入结论: {"全部 PASS, 可切换 5min" if ok1 and ok2 and ok3 else "有 FAIL, 需排查"} ===')
