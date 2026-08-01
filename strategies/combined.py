"""combined — 融合策略模块 (最终固化版).

方案: 6因子打分选池 + 池内风险平价 (多空, 周度调仓)
  - 品种池: manual29 (流动性适中、数据完整, 见 docs/策略基准记录.md)
  - 信号: 6个已验证因子等权打分 → Top10做多 / Bottom10做空
  - 权重: 池内波动率倒数 (风险平价近似)
  - 调仓: 周度 (W-FRI)

验证数据 (2025-01~2026-05 / OOS 2026-03~05):
  全量夏普 0.96, 回撤 -5.32%
  OOS 夏普 0.92, 回撤 -3.11%
  月换手 18%

用法:
    from strategies.combined import CombinedStrategy
    strat = CombinedStrategy(config_path="config/intraday_backtest.yaml")
    weights = strat.signal(date="2026-05-29")   # 返回 {品种: 权重}
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.config import load_config
from factors.engine import FactorEngine

# 6 个已验证有效因子 + 方向 (+1=高暴露看多, -1=高暴露看空)
FACTORS = {
    "intraday_jump_intensity_20d": -1,
    "intraday_price_peak_count_20d": 1,
    "intraday_realised_skewness_20d": 1,
    "intraday_dtws_20d": 1,
    "intraday_drip_stone_20d": -1,
    "intraday_peak_ridge_ratio_20d": -1,
}

# manual29 品种池
MANUAL29 = [
    "A", "AG", "AL", "AU", "CU", "FU", "HC", "I", "IC", "IF", "IH", "J", "JM",
    "M", "MA", "NI", "P", "RB", "RM", "RU", "SA", "SN", "SR", "T", "TA", "TL",
    "TS", "Y", "ZN",
]

_DEFAULT_TOP_N = 10


class CombinedStrategy:
    """6因子打分选池 + 池内风险平价 融合策略."""

    def __init__(self, config_path: str = "config/intraday_backtest.yaml",
                 top_n: int = _DEFAULT_TOP_N):
        self.config_path = config_path
        self.top_n = top_n
        self.cfg = load_config(config_path)
        from pipeline.runner import PipelineRunner
        self.runner = PipelineRunner(config=self.cfg)
        self.engine = FactorEngine(self.runner.data_manager)
        self._universe = [s for s in MANUAL29 if s in self.cfg.universe] or MANUAL29

    @property
    def universe(self) -> list[str]:
        return list(self._universe)

    def factor_scores(self, start: str, end: str) -> pd.DataFrame:
        """计算 6 因子等权打分矩阵 (索引=日期, 列=品种, 0~1)."""
        calendar = pd.DatetimeIndex(self.runner.data_manager.get_calendar(
            pd.Timestamp(start), pd.Timestamp(end)))
        names = list(FACTORS)
        computed = self.engine.compute_factors(
            names, calendar.tolist(), self._universe, parallel=False)
        score = pd.DataFrame(index=calendar, columns=self._universe, dtype=float)
        for name, direction in FACTORS.items():
            rank = computed[name].rank(axis=1, pct=True)
            oriented = rank if direction == 1 else (1 - rank)
            score = score.add(oriented, fill_value=0)
        return score.div(len(names))

    def signal(self, date: str) -> dict[str, float]:
        """给定交易日, 返回目标持仓权重 {品种: 权重} (做多>0, 做空<0).

        内部: 取最近一周的因子打分 → Top N / Bottom N → 池内风险平价权重.
        """
        end = pd.Timestamp(date)
        start = end - pd.Timedelta(days=40)
        score = self.factor_scores(start.strftime("%Y-%m-%d"), date)
        if score.empty:
            return {}
        row = score.iloc[-1].dropna()
        if len(row) < 2 * self.top_n:
            row = score.dropna(axis=1).iloc[-1]
        if len(row) < 2:
            return {}

        ranked = row.rank(ascending=False)
        long_pool = ranked[ranked <= self.top_n].index.tolist()
        short_pool = ranked[ranked > len(ranked) - self.top_n].index.tolist()

        # 池内风险平价权重 (波动率倒数)
        weights = {}
        close = self.runner.data_manager.get(
            "close", pd.DatetimeIndex([end]), self._universe)
        vol = self._rolling_vol(end)
        for pool, sign in [(long_pool, 1.0), (short_pool, -1.0)]:
            if not pool:
                continue
            v = vol.reindex(pool).replace(0, np.nan).dropna()
            if v.empty:
                continue
            w = (1.0 / v)
            total = w.sum()
            if not np.isfinite(total) or total <= 0:
                continue
            w = w / total * sign
            for sym, wi in w.items():
                weights[sym] = float(wi)
        return weights

    def _rolling_vol(self, date: pd.Timestamp) -> pd.Series:
        """品种 20 日滚动波动率 (截至于 date)."""
        start = date - pd.Timedelta(days=60)
        cal = pd.DatetimeIndex(self.runner.data_manager.get_calendar(start, date))
        close = self.runner.data_manager.get("close", cal, self._universe)
        ret = close.pct_change()
        vol = ret.rolling(20, min_periods=10).std(ddof=0)
        return vol.iloc[-1] if not vol.empty else pd.Series(dtype=float)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="combined 融合策略信号")
    parser.add_argument("--date", default="2026-05-29", help="信号日期")
    parser.add_argument("--config", default="config/intraday_backtest.yaml")
    parser.add_argument("--topn", type=int, default=_DEFAULT_TOP_N)
    args = parser.parse_args()

    strat = CombinedStrategy(args.config, args.topn)
    w = strat.signal(args.date)
    if not w:
        print("无有效信号")
    else:
        print(f"信号日期 {args.date}: {len(w)} 个持仓")
        long = {k: v for k, v in w.items() if v > 0}
        short = {k: v for k, v in w.items() if v < 0}
        print(f"  多头: {', '.join(f'{k}({v*100:.1f}%)' for k, v in sorted(long.items(), key=lambda x:-x[1]))}")
        print(f"  空头: {', '.join(f'{k}({v*100:.1f}%)' for k, v in sorted(short.items(), key=lambda x:x[1]))}")
