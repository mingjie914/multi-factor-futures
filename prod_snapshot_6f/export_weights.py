"""export_weights — 导出生产方案每日权重 (2025-01-01 ~ 最新数据日).

输出: weights/daily_weights.csv
格式: date | symbol | direction | weight   (每日每品种一行, 权重=净权重, 多头正/空头负)
逻辑: 与 combined.py 完全一致 (7因子等权 → cap=3 选池 → 池内 ERC → 多空合并净权重)
      T 日权重 = T-1 信号 (shift 已内建), 用于 T 日收益计算.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from core.config import load_config
from pipeline.runner import PipelineRunner
from factors.engine import FactorEngine
from optimization.risk_budgeting import RiskBudgetingOptimizer
from strategies.combined import FACTORS, SECTOR_MAP, SECTOR_CAP

UNIV38 = ["A", "AG", "AL", "AU", "CU", "FU", "HC", "I", "IC", "IF", "IH", "J", "JM",
          "M", "MA", "NI", "P", "RB", "RM", "RU", "SA", "SN", "SR", "T", "TA", "TL",
          "TS", "Y", "ZN", "IM", "TF", "CF", "OI", "LH", "JD", "SC", "V", "UR"]


def main():
    cfg = load_config("config/intraday_backtest.yaml")
    runner = PipelineRunner(config=cfg)
    start, end = "2025-01-01", "2026-07-31"
    cal = pd.DatetimeIndex(runner.data_manager.get_calendar(pd.Timestamp(start), pd.Timestamp(end)))
    univ = [s for s in UNIV38]
    engine = FactorEngine(runner.data_manager)
    comp = engine.compute_factors(list(FACTORS), cal, univ, parallel=True)

    score = pd.DataFrame(index=cal, columns=univ, dtype=float)
    for n, direction in FACTORS.items():
        r = comp[n].rank(axis=1, pct=True)
        score = score.add(r if direction == 1 else (1 - r), fill_value=0)
    score = score.div(len(FACTORS))

    close = runner.data_manager.get("close", cal, univ)
    daily_ret = close.pct_change()
    vol20 = close.pct_change().rolling(20, min_periods=10).std(ddof=0)
    sector_of = {}
    for sec, mem in SECTOR_MAP.items():
        for m in mem:
            if m in univ:
                sector_of[m] = sec

    def capped(order, cap_n):
        picks, counts = [], {}
        for s in order:
            sec = sector_of.get(s, "其他")
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
        start_d = t - pd.Timedelta(days=90)
        c = pd.DatetimeIndex(runner.data_manager.get_calendar(start_d, t))
        ret_sub = daily_ret.reindex(c)[list(pool)].dropna()
        if ret_sub.shape[0] < 10:
            return None
        cov_raw = ret_sub.cov().values
        cov = 0.7 * cov_raw + 0.3 * np.diag(np.diag(cov_raw))
        try:
            w = RiskBudgetingOptimizer._erc_weights(cov, np.ones(len(pool)))
            return dict(zip(pool, w))
        except (RuntimeError, ValueError):
            v = ret_sub.std(ddof=0).replace(0, np.nan).dropna()
            if v.empty:
                return None
            w = (1.0 / v).values
            return dict(zip(pool, w / w.sum()))

    rows = []
    for t in cal:
        row = score.loc[t].dropna()
        if len(row) < 20:
            continue
        top = capped(row.sort_values(ascending=False).index.tolist(), SECTOR_CAP)
        bot = capped(row.sort_values(ascending=True).index.tolist(), SECTOR_CAP)
        wl = erc_w(top, t) or {}
        ws = erc_w(bot, t) or {}
        for sym, w in wl.items():
            rows.append({"date": t.date(), "symbol": sym, "direction": "long", "weight": round(w, 6)})
        for sym, w in ws.items():
            rows.append({"date": t.date(), "symbol": sym, "direction": "short", "weight": round(-w, 6)})

    out = pd.DataFrame(rows)
    out_dir = Path("weights")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "daily_weights.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"已导出 {len(out)} 行 ({(out['date'].nunique())} 个交易日) → {out_path}")
    print(out.head(10).to_string(index=False))
    print("...")
    print(out.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
