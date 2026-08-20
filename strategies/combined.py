"""combined — 主分支A：10f固定观察基线（ICIR + cap3 + ERC）。

方案: 10因子 IC_IR 动态加权打分选池 + 池内 ERC 风险平价 (多空, 日度调仓)
  - 品种池: 38 (manual29 + 金融 IM/TF, 农产品 CF/OI/LH/JD, 能化 SC/V/UR, 见 docs/策略基准记录.md)
  - 信号: 10f IC_IR动态加权（60日滚动Ledoit-Wolf）→ Top10做多 / Bottom10做空
  - 权重: 池内 ERC (等风险贡献目标, 协方差 shrinkage=0.3)，再投影到显式资产约束
  - 选池: 板块配额 cap=3 (每板块最多3个多头/空头)
  - 调仓: 日度 (每天收盘生成信号, 次日持有)

10f只作为修复后统一比较的固定基线；历史表现不足以构成生产批准，目标权重发布门保持关闭。

用法:
    from strategies.combined import CombinedStrategy
    strat = CombinedStrategy()
    weights = strat.signal("2026-05-29")   # pd.Series: 品种 → 净权重 (多空抵消)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.config import load_config
from core.sectors import PORTFOLIO_SELECTION_GROUPS, portfolio_selection_group_for
from data.manager import DataManager
from factors.engine import FactorEngine
from factors import library as _factor_library  # noqa: F401
from optimization.factor_weighting import (
    causal_history,
    combine_available_factor_scores,
    factor_weights,
    prepare_complete_history,
    rank_information_coefficients,
)
from optimization.portfolio_construction import (
    PortfolioConstraints,
    allocate_sleeve,
    causal_risk_window,
    combine_sleeves,
    prepare_risk_history,
    select_long_short_pools,
)

# 当前固定观察10f及方向（+1=高暴露看多，-1=高暴露看空）。
# 来源：13f剔除jump_intensity、dtws和seat_long_short_seat_ratio；
# 组合方法仍为ICIR + Top10/Bottom10 + cap3 + ERC。
FACTORS = {
    "intraday_price_peak_count_20d": 1,
    "intraday_realised_skewness_20d": 1,
    "intraday_drip_stone_20d": -1,
    "intraday_peak_ridge_ratio_20d": -1,
    "intraday_torrent_down_20d": -1,
    "intraday_lowest_time_20d": 1,
    "intraday_term_slope_20d": 1,
    "intraday_open_close_volume_ratio_20d": -1,
    "intraday_turnover_velocity_20d": 1,
    "intraday_price_delay_20d": -1,
}

# 品种池 38 (2026-08-03 升级: manual29 + 金融 IM/TF, 农产品 CF/OI/LH/JD, 能化 SC/V/UR)
UNIVERSE38 = [
    "A", "AG", "AL", "AU", "CU", "FU", "HC", "I", "IC", "IF", "IH", "J", "JM",
    "M", "MA", "NI", "P", "RB", "RM", "RU", "SA", "SN", "SR", "T", "TA", "TL",
    "TS", "Y", "ZN",
    # 2026-08-03 新增 9 个
    "IM", "TF", "CF", "OI", "LH", "JD", "SC", "V", "UR",
]
_DEFAULT_TOP_N = 10

# 生产选池使用五个宽组；因子研究仍使用 core.sectors 的细分类。
SECTOR_OF = {
    symbol: portfolio_selection_group_for(symbol) for symbol in UNIVERSE38
}
SECTOR_MAP = {
    group: [symbol for symbol in UNIVERSE38 if SECTOR_OF[symbol] == group]
    for group in PORTFOLIO_SELECTION_GROUPS
}


class CombinedStrategy:
    """10f IC_IR打分选池 + 池内ERC风险平价（38品种、cap3、日度）。"""

    def __init__(self, config_path: str = "config/intraday_backtest.yaml",
                 top_n: int = _DEFAULT_TOP_N):
        self.config_path = config_path
        if int(top_n) <= 0:
            raise ValueError("top_n must be positive")
        self.top_n = int(top_n)
        self.cfg = load_config(config_path)
        self.data_manager = DataManager.from_config(self.cfg)
        self.engine = FactorEngine(self.data_manager)
        configured = set(map(str, self.cfg.universe))
        missing = [symbol for symbol in UNIVERSE38 if symbol not in configured]
        if missing:
            raise ValueError(
                "生产配置缺少固定38品种: " + ", ".join(missing)
            )
        self._universe = list(UNIVERSE38)
        self.portfolio_cfg = self.cfg.production_portfolio
        self.constraints = PortfolioConstraints.from_config(
            self.portfolio_cfg, top_n=self.top_n
        )

    @property
    def universe(self) -> list[str]:
        return list(self._universe)

    def factor_scores(self, start: str, end: str) -> pd.DataFrame:
        """计算因子打分矩阵 (索引=日期, 列=品种, 0~1).

        合成方法和窗口来自 ``production_portfolio`` 配置；默认是
        ``lw_abs`` 与严格的60条历史IC。
        """
        calendar = pd.DatetimeIndex(self.data_manager.get_calendar(
            pd.Timestamp(start), pd.Timestamp(end)))
        if calendar.empty:
            raise RuntimeError(f"生产信号交易日历为空: {start} ~ {end}")
        names = list(FACTORS)
        computed = self.engine.compute_factors(
            names, calendar.tolist(), self._universe, parallel=True)
        unavailable = [
            name
            for name in names
            if name not in computed or not computed[name].notna().any().any()
        ]
        if unavailable:
            raise RuntimeError(
                "production factor computation failed or returned no values: "
                + ", ".join(unavailable)
            )
        discontinuities = {}
        for name in names:
            available = computed[name].notna().any(axis=1)
            missing_after_start = available.cummax() & ~available
            if bool(missing_after_start.any()):
                discontinuities[name] = [
                    str(pd.Timestamp(value).date())
                    for value in available.index[missing_after_start][:3]
                ]
        if discontinuities:
            raise RuntimeError(
                "production factor became unavailable after its first valid date: "
                + "; ".join(
                    f"{name}={dates}" for name, dates in discontinuities.items()
                )
            )
        # 各因子截面排名 (方向已调整: 高=好)
        ranks = {}
        for name, direction in FACTORS.items():
            rank = computed[name].rank(axis=1, pct=True)
            ranks[name] = rank if direction == 1 else (1 - rank)

        factor_weight_method = str(self.portfolio_cfg.factor_weight_method)
        if factor_weight_method == "equal":
            numerator = pd.DataFrame(0.0, index=calendar, columns=self._universe)
            denominator = pd.DataFrame(0.0, index=calendar, columns=self._universe)
            for name in names:
                available = ranks[name].notna()
                numerator = numerator.add(ranks[name].fillna(0.0), fill_value=0.0)
                denominator = denominator.add(available.astype(float), fill_value=0.0)
            return numerator.div(denominator.where(denominator > 0.0))

        # IC_IR 动态加权: 60日滚动 IC → Ledoit-Wolf 收缩协方差 → w* = Σ⁻¹·mean(IC)
        close = self.data_manager.get("close", calendar, self._universe)
        if close is None or close.empty:
            raise RuntimeError("生产因子权重缺少 close 数据")
        # rank[T]预测T+1收益；IC[T]要到T+1收盘后才完整可知。
        close_returns, _ = self.data_manager.prepare_close_data(close)
        ic = rank_information_coefficients(
            ranks, close_returns, minimum_cross_section=3
        ).reindex(calendar)
        score = pd.DataFrame(index=calendar, columns=self._universe, dtype=float)
        for t in calendar:
            hist = prepare_complete_history(
                causal_history(
                    ic,
                    t,
                    int(self.portfolio_cfg.ic_window),
                ),
                minimum_observations=30,
            )
            w = factor_weights(hist, factor_weight_method)
            if w.empty:
                continue
            score.loc[t] = combine_available_factor_scores(
                {name: ranks[name].loc[t] for name in w.index},
                w,
                self._universe,
            )
        return score

    def _pool_weights(
        self,
        pool: list[str],
        date: pd.Timestamp,
        recent_returns: pd.DataFrame | None = None,
    ) -> pd.Series:
        """使用共享构造器分配一个完整的单位多头或空头袖套。

        ``recent_returns`` 允许同一信号计算复用一次行情读取；仍在每个池内
        独立执行历史完整性筛选，因此不改变 ERC 样本或权重语义。
        """
        constraints = getattr(
            self,
            "constraints",
            PortfolioConstraints(top_n_per_side=self.top_n),
        )
        ret = (
            self._recent_returns(date, pool)
            if recent_returns is None
            else recent_returns.reindex(columns=pool)
        )
        history = prepare_risk_history(
            ret,
            pool,
            constraints.minimum_risk_observations,
        )
        portfolio_cfg = getattr(self, "portfolio_cfg", None)
        method = getattr(portfolio_cfg, "asset_weight_method", "erc")
        return allocate_sleeve(
            history,
            method=str(method),
            constraints=constraints,
            sector_of=SECTOR_OF,
        )

    def _recent_returns(self, date: pd.Timestamp, symbols: list[str]) -> pd.DataFrame:
        """配置窗口内、严格截至决策日前的收益率。"""
        portfolio_cfg = getattr(self, "portfolio_cfg", None)
        lookback = int(getattr(portfolio_cfg, "risk_lookback_calendar_days", 90))
        start = date - pd.Timedelta(days=lookback)
        # Preload exactly one earlier close so the return stamped on the first
        # window date is retained, matching the full-history evaluator.
        calendar = pd.DatetimeIndex(
            self.data_manager.get_calendar(start - pd.Timedelta(days=lookback), date)
        )
        before = calendar[calendar < start]
        in_window = calendar[(calendar >= start) & (calendar < date)]
        close_dates = before[-1:].append(in_window)
        close = self.data_manager.get("close", close_dates, symbols)
        if close is None or close.empty:
            raise RuntimeError("ERC 风险历史缺少 close 数据")
        close_returns, _ = self.data_manager.prepare_close_data(close)
        return causal_risk_window(close_returns, date, lookback).dropna(how="all")

    def signal(self, date: str) -> pd.Series:
        """给定交易日, 返回净持仓权重 Series (索引=品种, 值=多空抵消后净权重).

        - 调仓日: 使用当日可得的滞后因子打分生成次日持仓信号
        - 异常处理: 风险历史不足的品种不进入候选；入选后数据或约束异常则报错
        """
        end = pd.Timestamp(date)
        # IC_IR 需要 60 日历史算 IC, 放大窗口保证预热充足
        start = end - pd.Timedelta(days=160)
        score = self.factor_scores(start.strftime("%Y-%m-%d"), date)
        if score.empty:
            raise RuntimeError(f"{end.date()} 未生成生产因子分数")
        if pd.Timestamp(score.index[-1]).normalize() != end.normalize():
            raise RuntimeError(f"{end.date()} 不是可生成收盘信号的交易日")
        row = score.iloc[-1].dropna()
        constraints = getattr(
            self,
            "constraints",
            PortfolioConstraints(top_n_per_side=self.top_n),
        )
        recent_returns = self._recent_returns(end, list(row.index))
        eligible = recent_returns.columns[
            recent_returns.replace([np.inf, -np.inf], np.nan)
            .notna()
            .sum(axis=0)
            .ge(constraints.minimum_risk_observations)
        ].tolist()
        row = row.reindex(eligible).dropna()
        if len(row) < 2 * constraints.top_n_per_side:
            raise RuntimeError(
                f"{end.date()} 可用候选不足: {len(row)} < "
                f"{2 * constraints.top_n_per_side}"
            )

        long_pool, short_pool = select_long_short_pools(
            row,
            eligible=eligible,
            sector_of=SECTOR_OF,
            constraints=constraints,
        )

        w_long = self._pool_weights(long_pool, end, recent_returns)
        w_short = self._pool_weights(short_pool, end, recent_returns)

        return combine_sleeves(
            w_long,
            w_short,
            universe=self._universe,
            long_pool=long_pool,
            short_pool=short_pool,
            constraints=constraints,
            sector_of=SECTOR_OF,
        )


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
