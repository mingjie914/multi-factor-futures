"""combined — 融合策略模块 (最终固化版, ERC 升级).

方案: 7因子打分选池 + 池内 ERC 风险平价 (多空, 日度调仓)
  - 品种池: 38 (manual29 + 金融 IM/TF, 农产品 CF/OI/LH/JD, 能化 SC/V/UR, 见 docs/策略基准记录.md)
  - 信号: 7个已验证因子等权打分 → Top10做多 / Bottom10做空
  - 权重: 池内 ERC (等风险贡献, 协方差 shrinkage=0.3), 可回退到逆波动率
  - 选池: 板块配额 cap=3 (每板块最多3个多头/空头)
  - 调仓: 日度 (每天收盘生成信号, 次日持有)

验证数据 (2025-01~2026-05 / OOS 2026-03~05):
  逆波动率版: 全量夏普 0.96, OOS 夏普 0.92, 月换手 18%
  ERC 版: 待验证 (预期 OOS 回撤进一步收窄)

用法:
    from strategies.combined import CombinedStrategy
    strat = CombinedStrategy()
    weights = strat.signal("2026-05-29")   # pd.Series: 品种 → 净权重 (多空抵消)
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
# 2026-08-04: 时序 bug 修正后 6因子(夏普2.04) > 7因子(1.73), 席位因子 #326 不再增益
FACTORS = {
    "intraday_jump_intensity_20d": -1,
    "intraday_price_peak_count_20d": 1,
    "intraday_realised_skewness_20d": 1,
    "intraday_dtws_20d": 1,
    "intraday_drip_stone_20d": -1,
    "intraday_peak_ridge_ratio_20d": -1,
}

# 品种池 38 (2026-08-03 升级: manual29 + 金融 IM/TF, 农产品 CF/OI/LH/JD, 能化 SC/V/UR)
UNIVERSE38 = [
    "A", "AG", "AL", "AU", "CU", "FU", "HC", "I", "IC", "IF", "IH", "J", "JM",
    "M", "MA", "NI", "P", "RB", "RM", "RU", "SA", "SN", "SR", "T", "TA", "TL",
    "TS", "Y", "ZN",
    # 2026-08-03 新增 9 个
    "IM", "TF", "CF", "OI", "LH", "JD", "SC", "V", "UR",
]
# 兼容旧名
MANUAL29 = UNIVERSE38

_DEFAULT_TOP_N = 10

# 权重引擎开关: False = ERC (协方差), True = 简化逆波动率 (回退用)
USE_SIMPLE_RP = False
# 协方差收缩系数
COV_SHRINKAGE = 0.30
# 单品种权重硬限制
MAX_WEIGHT = 0.20
MIN_WEIGHT = 0.005

# 板块配额: 每板块最多 SECTOR_CAP 个多头/空头 (0 = 不限制, 全市场 Top10/Bottom10)
# 2026-08 验证: cap=3 夏普 2.64 vs 无配额 2.01, 回撤 -3.4% vs -7.3% (见 docs/策略基准记录.md)
SECTOR_CAP = 3

# 38 品种板块映射 (含金融板块; 用于配额选池, 不用于中性化)
SECTOR_MAP = {
    "有色": ["CU", "AL", "ZN", "NI", "SN", "AG", "AU"],
    "黑色": ["RB", "HC", "I", "J", "JM"],
    "能化": ["FU", "MA", "RU", "SA", "TA", "SC", "V", "UR"],
    "农产品": ["A", "M", "P", "RM", "Y", "SR", "CF", "OI", "LH", "JD"],
    "金融": ["IC", "IF", "IH", "T", "TL", "TS", "IM", "TF"],
}


class CombinedStrategy:
    """7因子打分选池 + 池内 ERC 风险平价 融合策略 (38品种, cap=3, 日度)."""

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
            names, calendar.tolist(), self._universe, parallel=True)
        score = pd.DataFrame(index=calendar, columns=self._universe, dtype=float)
        for name, direction in FACTORS.items():
            rank = computed[name].rank(axis=1, pct=True)
            oriented = rank if direction == 1 else (1 - rank)
            score = score.add(oriented, fill_value=0)
        return score.div(len(names))

    def _capped_picks(self, row: pd.Series, ascending: bool, cap: int) -> list[str]:
        """按得分排序取 top_n 个, 但每板块最多 cap 个 (全市场排名 + 板块配额).

        ascending=True 按升序取 (得分最低, 用于空头), False 按降序取 (得分最高, 用于多头).
        """
        order = row.sort_values(ascending=not ascending).index.tolist()
        picks, counts = [], {}
        for s in order:
            sec = next((k for k, mem in SECTOR_MAP.items() if s in mem), "其他")
            if counts.get(sec, 0) >= cap:
                continue
            picks.append(s)
            counts[sec] = counts.get(sec, 0) + 1
            if len(picks) >= self.top_n:
                break
        return picks

    def _pool_weights(self, pool: list[str], date: pd.Timestamp) -> pd.Series:
        """池内权重: ERC (默认) 或逆波动率 (USE_SIMPLE_RP=True)."""
        if not pool:
            return pd.Series(dtype=float)
        ret = self._recent_returns(date, pool)
        if ret.shape[1] < 2 or ret.shape[0] < 10:
            # 数据不足: 回退到等权
            return pd.Series(1.0 / len(pool), index=pool)

        if USE_SIMPLE_RP:
            vol = ret.std(ddof=0)
            w = 1.0 / vol.replace(0, np.nan)
            w = w / w.sum()
            return w

        # 协方差 ERC (框架正式 _erc_weights)
        from optimization.risk_budgeting import RiskBudgetingOptimizer
        cov_raw = ret.cov().values
        target = np.diag(np.diag(cov_raw))
        cov = (1.0 - COV_SHRINKAGE) * cov_raw + COV_SHRINKAGE * target
        try:
            w = RiskBudgetingOptimizer._erc_weights(cov, np.ones(len(pool)))
        except (RuntimeError, ValueError):
            # ERC 求解失败: 回退到逆波动率
            vol = ret.std(ddof=0)
            w = (1.0 / vol.replace(0, np.nan)).values
            w = w / w.sum()
        w = pd.Series(w, index=pool)

        # 极值保护: 单品种权重上限/下限
        w = w.clip(lower=MIN_WEIGHT, upper=MAX_WEIGHT)
        return w / w.sum()

    def _recent_returns(self, date: pd.Timestamp, symbols: list[str]) -> pd.DataFrame:
        """最近 60 日收益率 (截至于 date, 用于协方差估计)."""
        start = date - pd.Timedelta(days=90)
        cal = pd.DatetimeIndex(self.runner.data_manager.get_calendar(start, date))
        close = self.runner.data_manager.get("close", cal, symbols)
        if close is None or close.empty:
            return pd.DataFrame()
        return close.pct_change().dropna(how="all")

    def signal(self, date: str) -> pd.Series:
        """给定交易日, 返回净持仓权重 Series (索引=品种, 值=多空抵消后净权重).

        - 调仓日: 最近一个周五 (W-FRI) 的因子打分生成信号
        - 异常处理: 无数据品种自动跳过, 不报错
        """
        end = pd.Timestamp(date)
        start = end - pd.Timedelta(days=40)
        score = self.factor_scores(start.strftime("%Y-%m-%d"), date)
        if score.empty:
            return pd.Series(dtype=float)
        # 跳过无数据品种
        row = score.iloc[-1].dropna()
        if len(row) < 2 * self.top_n:
            row = score.dropna(axis=1).iloc[-1]
        if len(row) < 2:
            return pd.Series(dtype=float)

        ranked = row.rank(ascending=False)
        if SECTOR_CAP > 0:
            long_pool = self._capped_picks(row, ascending=True, cap=SECTOR_CAP)
            short_pool = self._capped_picks(row, ascending=False, cap=SECTOR_CAP)
        else:
            long_pool = ranked[ranked <= self.top_n].index.tolist()
            short_pool = ranked[ranked > len(ranked) - self.top_n].index.tolist()

        w_long = self._pool_weights(long_pool, end)
        w_short = self._pool_weights(short_pool, end)

        # 净权重: 多头 +, 空头 -
        net = pd.Series(0.0, index=self._universe)
        if not w_long.empty:
            net = net.add(w_long, fill_value=0)
        if not w_short.empty:
            net = net.sub(w_short, fill_value=0)
        return net[net.abs() > 1e-12]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="combined 融合策略信号")
    parser.add_argument("--date", default="2026-05-29", help="信号日期")
    parser.add_argument("--config", default="config/intraday_backtest.yaml")
    parser.add_argument("--topn", type=int, default=_DEFAULT_TOP_N)
    args = parser.parse_args()

    strat = CombinedStrategy(args.config, args.topn)
    w = strat.signal(args.date)
    if w.empty:
        print("无有效信号")
    else:
        print(f"信号日期 {args.date}: {len(w)} 个持仓")
        long = w[w > 0].sort_values(ascending=False)
        short = w[w < 0].sort_values()
        print(f"  多头: {', '.join(f'{k}({v*100:.1f}%)' for k, v in long.items())}")
        print(f"  空头: {', '.join(f'{k}({v*100:.1f}%)' for k, v in short.items())}")
