"""生成适合外部代码 (weight_{T-1} × ret_T) 的权重.

外部代码: for T in dates: ret_T += weight_{T-1}(CSV) × return_T
→ CSV date 必须是"信号确定日", 且 weight 值 = score[date+1] (基于 ≤date 数据)
→ 即把执行日口径 CSV 的日期前移一个交易日 (权重值不变).
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from core.config import load_config
from pipeline.runner import PipelineRunner
from factors.engine import FactorEngine
from optimization.risk_budgeting import RiskBudgetingOptimizer
from strategies.combined import FACTORS, SECTOR_MAP, SECTOR_CAP

UNIV38 = ["A", "AG", "AL", "AU", "CU", "FU", "HC", "I", "IC", "IF", "IH", "J", "JM",
          "M", "MA", "NI", "P", "RB", "RM", "RU", "SA", "SN", "SR", "T", "TA", "TL",
          "TS", "Y", "ZN", "IM", "TF", "CF", "OI", "LH", "JD", "SC", "V", "UR"]


def main():
    cfg = load_config('config/intraday_backtest.yaml')
    runner = PipelineRunner(config=cfg)
    cal = pd.DatetimeIndex(runner.data_manager.get_calendar(pd.Timestamp('2025-01-01'), pd.Timestamp('2026-07-31')))
    u = list(UNIV38)
    engine = FactorEngine(runner.data_manager)
    comp = engine.compute_factors(list(FACTORS), cal, u, parallel=True)

    score = pd.DataFrame(index=cal, columns=u, dtype=float)
    for n, direction in FACTORS.items():
        r = comp[n].rank(axis=1, pct=True)
        score = score.add(r if direction == 1 else (1 - r), fill_value=0)
    score = score.div(len(FACTORS))

    close = runner.data_manager.get('close', cal, u)
    daily_ret = close.pct_change()
    sector_of = {}
    for sec, mem in SECTOR_MAP.items():
        for m in mem:
            if m in u:
                sector_of[m] = sec

    def capped(order, cap_n):
        picks, counts = [], {}
        for s in order:
            sec = sector_of.get(s, '其他')
            if counts.get(sec, 0) >= cap_n:
                continue
            picks.append(s)
            counts[sec] = counts.get(sec, 0) + 1
            if len(picks) >= 10:
                break
        return picks

    def erc_w(pool, t):
        if len(pool) < 2:
            return None
        sd = t - pd.Timedelta(days=90)
        c = pd.DatetimeIndex(runner.data_manager.get_calendar(sd, t))
        rs = daily_ret.reindex(c)[list(pool)].dropna()
        if rs.shape[0] < 10:
            return None
        cov_raw = rs.cov().values
        cov = 0.7 * cov_raw + 0.3 * np.diag(np.diag(cov_raw))
        try:
            w = RiskBudgetingOptimizer._erc_weights(cov, np.ones(len(pool)))
            return dict(zip(pool, w))
        except (RuntimeError, ValueError):
            v = rs.std(ddof=0).replace(0, np.nan).dropna()
            if v.empty:
                return None
            w = (1.0 / v).values
            return dict(zip(pool, w / w.sum()))

    # 计算执行日口径权重 (date=exec_date, weight 基于 ≤exec-1)
    rows = []
    exec_dates = []
    for t in score.index:
        row = score.loc[t].dropna()
        if len(row) < 20:
            continue
        top = capped(row.sort_values(ascending=False).index.tolist(), SECTOR_CAP)
        bot = capped(row.sort_values(ascending=True).index.tolist(), SECTOR_CAP)
        wl = erc_w(top, t) or {}
        ws = erc_w(bot, t) or {}
        exec_dates.append(t)
        for sym, w in wl.items():
            rows.append({'exec_date': t, 'symbol': sym, 'weight': round(w, 6)})
        for sym, w in ws.items():
            rows.append({'exec_date': t, 'symbol': sym, 'weight': round(-w, 6)})

    out = pd.DataFrame(rows)
    # 信号确定日 = 执行日的前一交易日 (外部 weight_{T-1} × ret_T)
    exec_set = pd.DatetimeIndex(exec_dates)
    sig_map = {}
    for t in exec_set:
        prv = exec_set[exec_set < t]
        sig_map[t] = prv[-1] if len(prv) else pd.NaT
    out['signal_date'] = out['exec_date'].map(sig_map)
    out = out.dropna(subset=['signal_date'])
    out = out[['signal_date', 'symbol', 'weight']].rename(columns={'signal_date': 'date'})
    out.to_csv('weights/daily_weights_ext.csv', index=False, encoding='utf-8')

    print(f'已导出: weights/daily_weights_ext.csv')
    print(f'形状: {len(out)} 行, 信号日 {out.date.min().date()} ~ {out.date.max().date()}, {out.date.nunique()} 天')
    nz = out.groupby('date').size()
    print(f'每日行数: min={nz.min()} max={nz.max()} (应20)')
    bal = out.groupby('date').weight.sum()
    print(f'每日权重合计: {bal.mean():.4f}±{bal.std():.4f} (多头+1空头-1 → 合计0)')
    print()
    print('=== 外部代码验证 ===')
    print('用法: for T: contrib += weight_{T-1}(CSV) × ret_T')
    # 验证
    wmat = out.pivot(index='date', columns='symbol', values='weight').fillna(0.0).sort_index()
    rets = []
    for t in wmat.index:
        wt = wmat.loc[t]
        nxt = cal[cal > t]
        if len(nxt) == 0:
            continue
        t1 = nxt[0]
        if t1 not in daily_ret.index:
            continue
        r = daily_ret.loc[t1].fillna(0.0)
        common = [s for s in wt.index if s in r.index]
        rets.append((t1, float((wt[common] * r[common]).sum())))
    s = pd.Series({d: v for d, v in rets}).sort_index().dropna()
    ann = s.mean() * 252
    vol = s.std(ddof=0) * np.sqrt(252)
    navs = (1 + s).cumprod()
    mdd = (navs / navs.cummax() - 1).min()
    print(f'weight_{{-1}}×ret_T 复算: n={len(s)} 年化={ann:.1%} 夏普={ann/vol:.2f} 回撤={mdd:.1%}')
    print('期望 ≈ production 2.04')


if __name__ == '__main__':
    main()
