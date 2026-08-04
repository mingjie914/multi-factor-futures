"""生成"信号确定日"口径的权重 (long 格式).

语义: date = 信号确定日 (T 日收盘后可用, 基于 ≤T 数据)
      外部用法: weight_T × ret_{T+1}  (T 日信号, T+1 日持有赚次日收益)

实现: 因子已内建 shift(1) → factor[t] 基于 ≤t-1 → t-1 收盘后可算
      故信号确定日 = t-1 (执行日前一交易日), 权重值不变.
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

    # 计算每个 t 的权重 (因子 t 值, 基于 ≤t-1)
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
            rows.append({'exec_date': t, 'signal_date': None, 'symbol': sym, 'direction': 'long', 'weight': round(w, 6)})
        for sym, w in ws.items():
            rows.append({'exec_date': t, 'signal_date': None, 'symbol': sym, 'direction': 'short', 'weight': round(-w, 6)})

    out = pd.DataFrame(rows)
    # 信号确定日 = 执行日前一个交易日 (factor[t] 基于 ≤t-1, t-1 收盘后可算)
    exec_set = pd.DatetimeIndex(exec_dates)
    signal_map = {}
    for t in exec_set:
        prev = exec_set[exec_set < t]
        signal_map[t] = prev[-1] if len(prev) else pd.NaT
    out['signal_date'] = out['exec_date'].map(signal_map)
    out = out.dropna(subset=['signal_date'])
    out = out.rename(columns={'signal_date': 'date'})[['date', 'symbol', 'direction', 'weight']]

    out.to_csv('weights/daily_weights_signal.csv', index=False, encoding='utf-8')
    print(f'已导出信号确定日口径: weights/daily_weights_signal.csv')
    print(f'形状: {len(out)} 行, 日期 {out.date.min().date()} ~ {out.date.max().date()}, 交易日 {out.date.nunique()}')
    print()
    print('=== 语义确认 ===')
    print('date = 信号确定日 (T 日收盘后可用, 基于 ≤T 数据)')
    print('外部用法: weight_T × ret_{T+1}  (T 日信号, T+1 日持有赚次日收益)')
    print()
    # 验证
    nz = out.groupby('date').size()
    print(f'每日行数: min={nz.min()} max={nz.max()} (应恒20)')
    bal = out.groupby(['date', 'direction']).weight.sum().unstack()
    print(f'多头合计: {bal["long"].mean():.4f}±{bal["long"].std():.4f}')
    print(f'空头合计: {bal["short"].mean():.4f}±{bal["short"].std():.4f}')
    print()
    print('样例 (最后3个信号日):')
    tail = out[out.date >= out.date.max() - pd.Timedelta(days=5)].sort_values(['date', 'weight'], ascending=[True, False])
    print(tail.head(24).to_string(index=False))


if __name__ == '__main__':
    main()
