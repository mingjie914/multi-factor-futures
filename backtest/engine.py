from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from core.types import (
    Date,
    DateIndex,
    NAVSeries,
    Universe,
    UniverseSchedule,
)
from core.interfaces import (
    ReturnModel,
    RiskModel,
    Optimizer,
    CostModel,
    Constraint,
)
from data.manager import DataManager
from factors.engine import FactorEngine
from factors.processor import FactorProcessor, build_processing_context
from backtest.metrics import TRADING_DAYS_PER_YEAR, compute_all_metrics
from backtest.research_ledger import (
    ResearchReturnLedger,
    align_transition_weights,
    close_marked_step,
    contract_transition_turnover,
    contract_transition_weight_vectors,
    default_research_ledger_metadata,
)
from data.market_quality import prepare_close_data
from core.logger import get_logger

logger = get_logger("multi_factor")


def _executed_turnover_intervals(result) -> pd.Series:
    """Return exchange-timed turnover when an audited ledger is available."""
    ledger = getattr(result, "research_ledger", None)
    daily = getattr(ledger, "daily", None)
    if daily is not None and "executed_traded_notional" in daily:
        values = daily["executed_traded_notional"].dropna()
        return values.iloc[1:] if len(values) else values
    return getattr(result, "turnover", pd.Series(dtype=float)).dropna()


class BacktestResult:
    def __init__(
        self,
        nav: NAVSeries,
        weights_history: pd.DataFrame,
        metrics: Dict[str, float],
        turnover: pd.Series = None,
        positions_history: pd.DataFrame = None,
        split_metrics: Dict[str, Dict[str, float]] = None,
        failure_ledger: List[dict] = None,
        costs: pd.Series = None,
        research_ledger: ResearchReturnLedger = None,
        decision_turnover: pd.Series = None,
    ):
        self.nav = nav
        self.weights_history = weights_history
        self.metrics = metrics
        self.turnover = turnover if turnover is not None else pd.Series(dtype=float)
        self.decision_turnover = (
            decision_turnover
            if decision_turnover is not None
            else pd.Series(dtype=float)
        )
        self.positions_history = positions_history if positions_history is not None else pd.DataFrame()
        # 固定历史分段诊断: {"train": {...}, "test": {...}}，默认前75%/后25%。
        self.split_metrics = split_metrics if split_metrics is not None else {}
        self.failure_ledger = failure_ledger if failure_ledger is not None else []
        self.costs = costs if costs is not None else pd.Series(dtype=float)
        self.research_ledger = research_ledger

    def summary(self) -> str:
        m = self.metrics
        base = (
            f"年化收益: {m.get('annual_return', 0):.2%} | "
            f"夏普: {m.get('sharpe', 0):.2f} | "
            f"最大回撤: {m.get('max_drawdown', 0):.2%} | "
            f"年化波动: {m.get('volatility', 0):.2%} | "
            f"胜率: {m.get('win_rate', 0):.2%} | "
            f"总收益: {m.get('total_return', 0):.2%}"
        )
        # 样本分段诊断 (若存在). 这不是独立样本外验证。
        if self.split_metrics and self.split_metrics.get("test"):
            tr = self.split_metrics["train"]
            te = self.split_metrics["test"]
            base += (
                f"\n  样本分段诊断(前75%/后25%, 非独立OOS): "
                f"训练期[年化={tr.get('annual_return', 0):.2%} 夏普={tr.get('sharpe', 0):.2f}] "
                f"测试期[年化={te.get('annual_return', 0):.2%} 夏普={te.get('sharpe', 0):.2f}]"
            )
        return base

    def save(self, output_dir: str, metadata: dict = None) -> None:
        """Persist a complete single-portfolio research result."""
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)

        def _json_default(value):
            if isinstance(value, (np.integer, np.floating)):
                return value.item()
            if isinstance(value, (pd.Timestamp, np.datetime64)):
                return str(pd.Timestamp(value))
            raise TypeError(f"not JSON serializable: {type(value).__name__}")

        payload = {
            "metadata": metadata or {},
            "metrics": self.metrics,
            "split_diagnostic": self.split_metrics,
            "failure_count": len(self.failure_ledger),
        }
        (root / "metrics.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
            + "\n",
            encoding="utf-8",
        )
        (root / "summary.txt").write_text(self.summary() + "\n", encoding="utf-8")
        self.nav.rename("nav").to_csv(root / "nav.csv")
        self.turnover.rename("turnover").to_csv(root / "turnover.csv")
        self.costs.rename("cost").to_csv(root / "costs.csv")
        if not self.weights_history.empty:
            self.weights_history.to_csv(root / "weights.csv")
        if not self.positions_history.empty:
            self.positions_history.to_csv(root / "positions.csv")
        if self.research_ledger is not None:
            self.research_ledger.save(root)
        (root / "failures.json").write_text(
            json.dumps(
                self.failure_ledger,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            )
            + "\n",
            encoding="utf-8",
        )

    def plot(self, save_dir: str = "./reports", version: str = ""):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 设置中文字体
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False

        os.makedirs(save_dir, exist_ok=True)

        # 检查 NAV 数据是否有效
        if self.nav.empty or self.nav.dropna().empty:
            print("  (NAV 数据为空, 跳过绘图)")
            return

        # 清理 NaN
        nav_clean = self.nav.dropna()
        fig, axes = plt.subplots(2, 1, figsize=(14, 9))

        # 上图: NAV 曲线 (归一化, 不显示资金数)
        nav_norm = nav_clean / nav_clean.iloc[0] if nav_clean.iloc[0] != 0 else nav_clean
        axes[0].plot(nav_norm.index, nav_norm.values,
                     color="#1a73e8", linewidth=1.5, label="净值")
        axes[0].fill_between(nav_norm.index, nav_norm.values,
                             nav_norm.iloc[0],
                             alpha=0.1, color="#1a73e8")
        title = "组合净值曲线 (归一化)"
        if version:
            title = f"{title} - {version}"
        m = self.metrics
        metrics_line = (
            f"年化 {m.get('annual_return', 0):.2%}  |  "
            f"夏普 {m.get('sharpe', 0):.2f}  |  "
            f"最大回撤 {m.get('max_drawdown', 0):.2%}  |  "
            f"卡玛 {m.get('calmar', 0):.2f}"
        )
        axes[0].set_title(f"{title}\n{metrics_line}", fontsize=13, pad=10)
        axes[0].set_ylabel("净值")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, alpha=0.3)

        # 下图: 权重热力图或累计收益
        if not self.weights_history.empty:
            wh = self.weights_history.fillna(0)
            # 只画权重变化最大的前 10 个品种
            weight_var = wh.std(axis=0).sort_values(ascending=False)
            top_cols = weight_var.head(10).index.tolist()
            wh_top = wh[top_cols]
            axes[1].stackplot(
                wh_top.index,
                wh_top.T.values,
                labels=top_cols,
                alpha=0.7,
            )
            axes[1].set_title("组合权重分布 (波动最大的前10个品种)")
            axes[1].set_ylabel("权重")
            axes[1].legend(loc="upper left", fontsize=7, ncol=2)
            axes[1].grid(True, alpha=0.3)
        else:
            # 如果没有权重数据, 画累计收益率
            returns = nav_clean.pct_change(fill_method=None).fillna(0)
            cum_ret = (1 + returns).cumprod() - 1
            axes[1].plot(cum_ret.index, cum_ret.values * 100,
                         color="#34a853", linewidth=1.5)
            axes[1].set_title("累计收益率 (%)")
            axes[1].set_ylabel("收益率 (%)")
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(save_dir, "backtest_nav.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  净值图已保存: {path}")

    def export_target_weights(self, path: str, as_of=None) -> str:
        """Export one close-observed target-weight snapshot to an explicit path.

        The exported weights are decisions observed at ``decision_date`` and are
        intended for adjustment during the following trading session.  This
        framework does not create or route orders.
        """
        if self.weights_history.empty:
            raise ValueError("no target-weight history is available")
        history = self.weights_history.copy()
        if history.index.has_duplicates or history.columns.has_duplicates:
            raise ValueError("target-weight history axes must be unique")
        history.index = pd.DatetimeIndex(history.index)
        history = history.sort_index()
        if as_of is not None:
            eligible = history.loc[history.index <= pd.Timestamp(as_of)]
            if eligible.empty:
                raise ValueError(f"no target weights are available by {as_of}")
            history = eligible
        decision_date = pd.Timestamp(history.index[-1])
        weights = pd.to_numeric(history.iloc[-1], errors="raise").fillna(0.0)
        if not np.isfinite(weights.to_numpy(dtype=float)).all():
            raise ValueError("target weights contain NaN or infinity")
        output = pd.DataFrame({
            "decision_date": decision_date.date().isoformat(),
            "ticker": weights.index.astype(str),
            "target_weight": weights.to_numpy(dtype=float),
            "execution_timing": "following_trading_session",
        }).sort_values("ticker", kind="stable")
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False)
        return str(output_path)


class MultiPortfolioResult:
    """多子组合叠加回测结果.

    Attributes:
        sub_results: 各子组合的 {name: BacktestResult}
        combined_result: 叠加后的组合 BacktestResult
        sub_configs: 各子组合配置信息
    """

    def __init__(
        self,
        sub_results: Dict[str, "BacktestResult"],
        combined_result: "BacktestResult",
        sub_configs: List[dict],
    ):
        self.sub_results = sub_results
        self.combined_result = combined_result
        self.sub_configs = sub_configs

    @classmethod
    def combine(
        cls, sub_results_raw: List[dict], total_capital: float
    ) -> "MultiPortfolioResult":
        """叠加各子组合净值.

        每个 sub_results_raw 元素: {"config": SubPortfolioConfig, "result": BacktestResult}

        叠加方式:
        - 各子组合净值归一化到1, 然后按资本占比加权求和
        - 叠加后的净值 × total_capital = 最终净值
        """
        from backtest.metrics import compute_all_metrics

        if not sub_results_raw:
            raise ValueError("at least one sub-portfolio result is required")
        total_capital = float(total_capital)
        if not np.isfinite(total_capital) or total_capital <= 0.0:
            raise ValueError("total_capital must be finite and positive")

        # 收集各子组合的归一化净值 (起始为1)
        sub_navs = {}
        sub_configs = []
        for item in sub_results_raw:
            cfg = item["config"]
            result = item["result"]
            sub_configs.append({
                "name": cfg.name,
                "factors": cfg.factors,
                "rebalance_freq": cfg.rebalance_freq,
                "holding_period": cfg.holding_period,
                "capital_weight": cfg.capital_weight,
                "metrics": result.metrics,
            })
            # 归一化净值
            nav = result.nav.dropna().sort_index()
            if nav.empty:
                raise ValueError(f"sub-portfolio {cfg.name!r} has no NAV")
            if nav.index.has_duplicates:
                raise ValueError(f"sub-portfolio {cfg.name!r} has duplicate NAV dates")
            values = nav.to_numpy(dtype=float)
            if not np.isfinite(values).all() or bool((values <= 0.0).any()):
                raise ValueError(f"sub-portfolio {cfg.name!r} NAV must be finite and positive")
            nav_norm = nav / nav.iloc[0]
            sub_navs[cfg.name] = nav_norm

        # 对齐到统一日期索引 (取并集)
        all_dates = sorted(set().union(*[set(n.index) for n in sub_navs.values()]))
        combined_idx = pd.DatetimeIndex(all_dates)

        # 向量化: 将所有子组合净值 stack 为 DataFrame, 一次性加权求和
        # 原始逐子组合循环 → 矩阵乘法
        sub_names = [s["name"] for s in sub_configs]
        if len(set(sub_names)) != len(sub_names):
            raise ValueError("sub-portfolio names must be unique")
        sub_nav_df = pd.DataFrame(sub_navs)
        # A sleeve may start later, but once started it must publish every
        # combined date.  Forward filling here would hide an internal gap or
        # an unexpectedly truncated run as a flat NAV segment.
        sub_nav_aligned = sub_nav_df[sub_names].reindex(combined_idx)
        for name in sub_names:
            first = sub_nav_aligned[name].first_valid_index()
            if first is None or sub_nav_aligned.loc[first:, name].isna().any():
                raise ValueError(
                    f"sub-portfolio {name!r} NAV has an internal or trailing gap"
                )
        # Capital allocated before a sleeve starts remains cash.
        sub_nav_aligned = sub_nav_aligned.fillna(1.0)
        weights = np.array([s["capital_weight"] for s in sub_configs], dtype=float)
        if (
            not np.isfinite(weights).all()
            or bool((weights < 0.0).any())
            or not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-10)
        ):
            raise ValueError("sub-portfolio capital weights must be non-negative and sum to 1")
        # 矩阵乘法: (n_dates, n_sub) @ (n_sub,) = (n_dates,)
        weighted_nav = pd.Series(
            sub_nav_aligned.values @ weights, index=combined_idx
        )

        # 乘以总资本
        combined_nav = weighted_nav * total_capital

        # 丢弃开头全为 NaN 的日期 (子组合起始日不同导致)
        first_valid = combined_nav.first_valid_index()
        if first_valid is not None:
            combined_nav = combined_nav.loc[first_valid:]

        # 计算组合收益系列
        combined_returns = combined_nav.pct_change(fill_method=None).dropna()

        # 计算叠加后组合的绩效指标
        combined_metrics = compute_all_metrics(combined_nav, returns=combined_returns)

        # 计算叠加后组合的平均换手率
        total_turnover = 0.0
        for item in sub_results_raw:
            cfg = item["config"]
            result = item["result"]
            executed = _executed_turnover_intervals(result)
            if not executed.empty:
                total_turnover += cfg.capital_weight * executed.mean()
        combined_metrics["avg_turnover"] = float(total_turnover)

        # 样本分段诊断 (固定权重叠加, 非独立OOS)
        from backtest.metrics import compute_split_metrics
        combined_split = compute_split_metrics(
            combined_nav, combined_returns, train_ratio=0.75
        )

        combined_result = BacktestResult(
            nav=combined_nav,
            weights_history=pd.DataFrame(),  # 子组合权重不合并
            metrics=combined_metrics,
            split_metrics=combined_split,
        )

        sub_results_dict = {
            item["config"].name: item["result"] for item in sub_results_raw
        }

        return cls(sub_results_dict, combined_result, sub_configs)

    @classmethod
    def combine_with_dynamic_weights(
        cls,
        sub_results_raw: List[dict],
        combined_nav: "pd.Series",
        weight_history: "pd.DataFrame",
        total_capital: float,
        turnover_history: "pd.Series" = None,
        cost_history: "pd.Series" = None,
        underlying_weights_history: "pd.DataFrame" = None,
        failure_ledger: List[dict] = None,
        exposure_diagnostics: List[dict] = None,
    ) -> "MultiPortfolioResult":
        """用元优化器生成的动态权重叠加各子组合.

        Args:
            sub_results_raw: [{"config": SubPortfolioConfig, "result": BacktestResult}, ...]
            combined_nav: 元优化器生成的叠加净值 (绝对值)
            weight_history: 各日期的资本权重 DataFrame (dates × sub_names)
            total_capital: 总初始资金
        """
        from backtest.metrics import compute_all_metrics

        sub_configs = []
        for item in sub_results_raw:
            cfg = item["config"]
            result = item["result"]
            # 平均权重
            avg_w = float(weight_history[cfg.name].mean()) if cfg.name in weight_history.columns else cfg.capital_weight
            sub_configs.append({
                "name": cfg.name,
                "factors": cfg.factors,
                "rebalance_freq": cfg.rebalance_freq,
                "holding_period": cfg.holding_period,
                "capital_weight": avg_w,
                "metrics": result.metrics,
            })

        # 叠加后的组合指标
        combined_returns = combined_nav.pct_change(fill_method=None).dropna()
        combined_metrics = compute_all_metrics(combined_nav, returns=combined_returns)

        # 子组合已审计交换手与元分配额外换手分别计量；没有订单/成交层，
        # 因此不假定不同 sleeve 之间可以净额成交。
        weighted_sub_turnover = 0.0
        for item in sub_results_raw:
            cfg = item["config"]
            result = item["result"]
            avg_w = float(weight_history[cfg.name].mean()) if cfg.name in weight_history.columns else cfg.capital_weight
            executed = _executed_turnover_intervals(result)
            if not executed.empty:
                weighted_sub_turnover += avg_w * executed.mean()
        combined_metrics["avg_subportfolio_turnover"] = float(weighted_sub_turnover)

        turnover_history = (
            turnover_history
            if turnover_history is not None
            else pd.Series(dtype=float)
        )
        cost_history = (
            cost_history if cost_history is not None else pd.Series(dtype=float)
        )
        active_turnover = turnover_history[turnover_history > 0].dropna()
        combined_metrics["avg_turnover"] = (
            float(active_turnover.mean()) if not active_turnover.empty else 0.0
        )
        combined_metrics["avg_daily_turnover"] = float(
            turnover_history.dropna().mean()
        ) if not turnover_history.empty else 0.0
        combined_metrics["annualized_turnover"] = (
            combined_metrics["avg_daily_turnover"] * TRADING_DAYS_PER_YEAR
        )
        combined_metrics["total_turnover"] = float(
            turnover_history.dropna().sum()
        )
        combined_metrics["total_transaction_cost"] = float(cost_history.fillna(0.0).sum())
        combined_metrics["avg_transaction_cost"] = (
            float(cost_history.fillna(0.0).mean()) if not cost_history.empty else 0.0
        )

        # 样本分段诊断 (叠加组合, 非独立OOS)
        from backtest.metrics import compute_split_metrics
        combined_split = compute_split_metrics(
            combined_nav, combined_returns, train_ratio=0.75
        )

        combined_result = BacktestResult(
            nav=combined_nav,
            weights_history=(
                underlying_weights_history
                if underlying_weights_history is not None
                else pd.DataFrame()
            ),
            metrics=combined_metrics,
            turnover=turnover_history,
            costs=cost_history,
            split_metrics=combined_split,
            failure_ledger=failure_ledger,
        )

        sub_results_dict = {
            item["config"].name: item["result"] for item in sub_results_raw
        }

        result = cls(sub_results_dict, combined_result, sub_configs)
        result.weight_history = weight_history  # 附加权重历史
        result.underlying_weights_history = combined_result.weights_history
        result.exposure_diagnostics = exposure_diagnostics or []
        return result

    def summary(self) -> str:
        lines = ["=" * 70, "日频多子组合叠加回测结果", "=" * 70]
        for cfg in self.sub_configs:
            m = cfg["metrics"]
            lines.append(
                f"  [{cfg['name']}] {cfg['rebalance_freq']}/{cfg['holding_period']}d "
                f"权重={cfg['capital_weight']:.0%} | "
                f"年化={m.get('annual_return', 0):.2%} "
                f"夏普={m.get('sharpe', 0):.2f} "
                f"回撤={m.get('max_drawdown', 0):.2%}"
            )
        lines.append("-" * 70)
        m = self.combined_result.metrics
        lines.append(
            f"  [叠加组合] | "
            f"年化={m.get('annual_return', 0):.2%} "
            f"夏普={m.get('sharpe', 0):.2f} "
            f"回撤={m.get('max_drawdown', 0):.2%} "
            f"波动={m.get('volatility', 0):.2%}"
        )
        # 样本分段诊断 (叠加组合, 非独立OOS)
        sm = self.combined_result.split_metrics
        if sm and sm.get("test"):
            tr, te = sm["train"], sm["test"]
            lines.append(
                f"  样本分段诊断(前75%/后25%, 非独立OOS): "
                f"训练期[年化={tr.get('annual_return', 0):.2%} 夏普={tr.get('sharpe', 0):.2f}] "
                f"测试期[年化={te.get('annual_return', 0):.2%} 夏普={te.get('sharpe', 0):.2f}]"
            )
        return "\n".join(lines)

    def save(self, output_dir: str, metadata: dict = None) -> None:
        """Persist one complete research result without relying on console logs."""
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        combined = self.combined_result
        payload = {
            "metadata": metadata or {},
            "combined_metrics": combined.metrics,
            "split_diagnostic": combined.split_metrics,
            "sub_portfolios": self.sub_configs,
            "failure_count": len(combined.failure_ledger),
        }

        def _json_default(value):
            if isinstance(value, (np.integer, np.floating)):
                return value.item()
            if isinstance(value, (pd.Timestamp, np.datetime64)):
                return str(pd.Timestamp(value))
            raise TypeError(f"not JSON serializable: {type(value).__name__}")

        (root / "metrics.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
            + "\n",
            encoding="utf-8",
        )
        (root / "summary.txt").write_text(self.summary() + "\n", encoding="utf-8")
        combined.nav.rename("nav").to_csv(root / "combined_nav.csv")
        combined.turnover.rename("turnover").to_csv(root / "turnover.csv")
        combined.costs.rename("cost").to_csv(root / "costs.csv")
        if not combined.weights_history.empty:
            combined.weights_history.to_csv(root / "underlying_weights.csv")
        if hasattr(self, "weight_history") and not self.weight_history.empty:
            self.weight_history.to_csv(root / "meta_weights.csv")
        for name, sub_result in self.sub_results.items():
            sub_result.save(
                root / "sub_portfolios" / str(name),
                metadata={
                    "parent_metadata": metadata or {},
                    "portfolio": str(name),
                },
            )
        (root / "failures.json").write_text(
            json.dumps(
                combined.failure_ledger,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            )
            + "\n",
            encoding="utf-8",
        )

    def plot(self, save_dir: str = "./reports", version: str = ""):
        """绘制子组合叠加净值图."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False

        os.makedirs(save_dir, exist_ok=True)

        fig, (ax, metrics_ax) = plt.subplots(
            2,
            1,
            figsize=(14, 9),
            gridspec_kw={"height_ratios": [5.0, 1.2]},
        )

        # 各子组合净值 (归一化)
        colors = ["#1a73e8", "#e8710a", "#1e8e3e", "#d93025", "#9334e6"]
        for idx, (name, result) in enumerate(self.sub_results.items()):
            nav = result.nav.dropna()
            if len(nav) > 0 and nav.iloc[0] != 0:
                nav_norm = nav / nav.iloc[0]
            else:
                nav_norm = nav
            color = colors[idx % len(colors)]
            ax.plot(nav_norm.index, nav_norm.values, label=name, color=color, linewidth=1.2, alpha=0.7)

        # 叠加组合净值 (归一化)
        combined_nav = self.combined_result.nav.dropna()
        if len(combined_nav) > 0 and combined_nav.iloc[0] != 0:
            combined_norm = combined_nav / combined_nav.iloc[0]
        else:
            combined_norm = combined_nav
        ax.plot(combined_norm.index, combined_norm.values, label="叠加组合",
                color="#202124", linewidth=2.0, linestyle="--")

        title = "日频多子组合叠加净值曲线"
        if version:
            title = f"{title} - {version}"
        ax.set_title(title, fontsize=14)
        ax.set_ylabel("归一化净值")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        # 指标表: 同时展示各子组合和叠加组合，避免只看净值曲线时
        # 丢失收益、风险与收益回撤比等关键口径。
        metric_rows = []
        row_labels = []
        for name, result in self.sub_results.items():
            m = result.metrics
            row_labels.append(name)
            metric_rows.append([
                f"{m.get('annual_return', 0):.2%}",
                f"{m.get('sharpe', 0):.2f}",
                f"{m.get('max_drawdown', 0):.2%}",
                f"{m.get('calmar', 0):.2f}",
            ])

        combined_metrics = self.combined_result.metrics
        row_labels.append("叠加组合")
        metric_rows.append([
            f"{combined_metrics.get('annual_return', 0):.2%}",
            f"{combined_metrics.get('sharpe', 0):.2f}",
            f"{combined_metrics.get('max_drawdown', 0):.2%}",
            f"{combined_metrics.get('calmar', 0):.2f}",
        ])

        metrics_ax.axis("off")
        table = metrics_ax.table(
            cellText=metric_rows,
            rowLabels=row_labels,
            colLabels=["年化收益", "夏普比率", "最大回撤", "卡玛比率"],
            cellLoc="center",
            rowLoc="center",
            loc="center",
            bbox=[0.08, 0.0, 0.84, 1.0],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.25)

        # 突出叠加组合行，便于从子策略对比快速落到最终组合。
        combined_row = len(row_labels)
        for col in range(-1, 4):
            cell = table.get_celld().get((combined_row, col))
            if cell is not None:
                cell.set_facecolor("#e8eaed")
                cell.set_text_props(fontweight="bold", color="#202124")

        path = os.path.join(save_dir, "multi_portfolio_nav.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  子组合净值图 -> {path}")


class Backtester:
    """回测引擎.

    负责: 数据准备 → 因子计算 → 模型拟合 → 逐期调仓 → 绩效评估.

    拆分为独立方法:
    - _prepare_backtest_data(): 日历对齐、因子预计算、风险模型估计
    - _refit_alpha(): 增量 alpha 重训
    - _optimize_period(): 单期组合优化
    - close_marked_step(): 单日持仓收益、成本与权重漂移
    """

    def __init__(
        self,
        rebalance_freq: str = "weekly",
        cost_model: Optional[CostModel] = None,
        market_name: str = "futures",
    ):
        self.initial_nav = 1.0
        self.rebalance_freq = rebalance_freq
        self.cost_model = cost_model
        self.market = market_name
        self.training_window = 750
        self.retrain_freq = 10
        self.holding_period = 5
        # 因子合成器 (可选): 聚类内等权合成, 降低多重共线性
        # None 表示不合成, 因子直接传入 alpha 模型
        self.factor_synthesizer = None
        # IC 监控器 (可选): 滚动 IC 跟踪 + 自动失效剔除
        # None 表示不监控, 所有因子始终参与
        self.ic_monitor = None
        # 最近一次 refit 使用的因子名集合 (确保预测时维度一致)
        self._last_fit_factors: Optional[set] = None
        self.asset_selector = None
        self.dynamic_risk_controller = None
        self.universe_selection_config = None
        self._eligibility_mask: Optional[pd.DataFrame] = None
        self.forecast_averaging_vintages = 1
        self._forecast_vintages: List[pd.Series] = []

    @staticmethod
    def _factor_frame_asof(
        factor: FactorMatrix, date: Date
    ) -> Optional[FactorMatrix]:
        """Return one point-in-time-safe factor row, relabelled to decision date."""
        if factor is None or factor.empty:
            return None
        date = pd.Timestamp(date)
        frame = factor if factor.index.is_monotonic_increasing else factor.sort_index()
        pos = int(frame.index.searchsorted(date, side="right")) - 1
        if pos < 0:
            return None
        row = frame.iloc[[pos]].copy()
        row.index = pd.DatetimeIndex([date])
        return row

    def _lookup_universe_schedule(
        self, universe_schedule: UniverseSchedule, date: Date,
        universe_static: pd.Index,
    ) -> pd.Index:
        """CR-004修复: 查询不晚于 date 的最近 schedule key.

        - 精确匹配优先
        - 否则取"不晚于 date 的最大 key"对应的 universe
        - 若 schedule 完全无匹配 (date 早于所有 schedule key), 返回空 Index,
          让上层显式处理 (len(universe)==0 时跳过本次调仓), 不静默回退完整 universe.

        Args:
            universe_schedule: {交易日: pd.Index(已上市品种)} 或非 dict
            date: 当前调仓日 (已经是有效交易日)
            universe_static: 仅在 universe_schedule 非 dict 时使用

        Returns:
            该调仓日可用的 universe (pd.Index)
        """
        if not isinstance(universe_schedule, dict):
            return universe_static

        date_ts = pd.Timestamp(date)
        # 1. 精确匹配
        if date_ts in universe_schedule:
            return universe_schedule[date_ts]
        # 2. 不晚于 date 的最大 key
        eligible_keys = [
            k for k in universe_schedule.keys()
            if pd.Timestamp(k) <= date_ts
        ]
        if eligible_keys:
            return universe_schedule[max(eligible_keys)]
        # 3. 完全找不到: 返回空 Index (不静默回退完整 universe)
        logger.warning(
            f"CR-004: 调仓日 {date_ts.date()} 找不到 universe schedule "
            f"(schedule keys={len(universe_schedule)} 个, 最早="
            f"{pd.Timestamp(min(universe_schedule.keys())).date() if universe_schedule else 'EMPTY'}); "
            f"返回空 universe, 跳过本次调仓"
        )
        return pd.Index([])

    def _prepare_backtest_data(
        self,
        data_manager: DataManager,
        factor_engine: FactorEngine,
        processor: FactorProcessor,
        factor_names: List[str],
        universe_schedule: UniverseSchedule,
        rebalance_dates: DateIndex,
        risk_model: RiskModel,
    ) -> Tuple[DateIndex, pd.Index, DateIndex, Dict[str, FactorMatrix], pd.DataFrame, pd.DataFrame]:
        """准备回测数据: 日历对齐、universe 并集、因子预计算、收益预计算、风险估计.

        Returns:
            (rebalance_dates, universe_static, all_dates, processed_all, fwd_returns, daily_returns)
            - fwd_returns: N日累计收益 (用于alpha训练)
            - daily_returns: 日度收益 (用于NAV累积)
        """
        from core.logger import get_logger
        log = get_logger("multi_factor")

        holding_period = self.holding_period
        training_window = self.training_window

        # universe 并集 (支持动态上市)
        # CR-004修复: universe_static 取所有 schedule 值的并集,
        # 不再用 .get(d, ...) 查找 (调仓日可能不是 schedule 的 key)
        universe_static = pd.Index([])
        if isinstance(universe_schedule, dict):
            for u in universe_schedule.values():
                universe_static = universe_static.union(u)
        else:
            universe_static = pd.Index(universe_schedule) if universe_schedule is not None else pd.Index([])

        # 日历范围
        bt_start = pd.Timestamp(rebalance_dates[0])
        bt_end = pd.Timestamp(rebalance_dates[-1])
        train_start = bt_start - pd.Timedelta(days=int(training_window * 1.5))
        tail_end = bt_end + pd.Timedelta(days=holding_period + 15)

        calendar = data_manager.get_calendar(train_start, tail_end)
        if calendar.empty:
            raise RuntimeError(
                "交易日历为空，回测失败关闭；不得用普通工作日替代交易所日历"
            )
        if hasattr(calendar, 'tz') and calendar.tz is not None:
            calendar = calendar.tz_localize(None)
        calendar = pd.DatetimeIndex(sorted(set(calendar)))

        # 调仓日对齐到交易日
        snapped = []
        for d in rebalance_dates:
            d = pd.Timestamp(d)
            if d.tzinfo is not None:
                d = d.tz_localize(None)
            before = calendar[calendar <= d]
            if len(before) > 0:
                snapped.append(before[-1])
            else:
                after = calendar[calendar >= d]
                snapped.append(after[0] if len(after) > 0 else d)
        rebalance_dates = pd.DatetimeIndex(sorted(set(snapped)))
        all_dates = calendar

        log.info(
            f"预计算因子: {len(factor_names)} 个, 日期 {all_dates[0].date()} ~ "
            f"{all_dates[-1].date()} ({len(all_dates)} 天), universe={len(universe_static)}, "
            f"holding={holding_period}d, {len(rebalance_dates)} 个调仓日"
        )

        ctx = build_processing_context(
            data_manager,
            all_dates,
            universe_static,
            self.universe_selection_config,
        )
        # The point-in-time eligibility mask must exist before snapshot factors
        # evaluate cross-sectional operators.
        raw_factors = factor_engine.compute_factors(
            factor_names, all_dates, universe_static, parallel=False
        )
        self._eligibility_mask = ctx.eligibility
        processed_all = processor.process_batch(raw_factors, ctx)

        # 因子合成: 聚类内等权合成 (降低多重共线性, 保留全部显著因子信息)
        # 合成在 processing 之后, alpha 拟合之前
        if self.factor_synthesizer is not None:
            processed_all = self.factor_synthesizer.synthesize(processed_all)
            logger.info(
                f"因子合成完成: {len(raw_factors)} 原始 → "
                f"{len(processed_all)} 合成因子"
            )

        # 预计算 forward returns (N日累计收益, 用于alpha训练和IC检验)
        fwd_returns = data_manager.get_forward_returns(
            all_dates, universe_static, period=holding_period
        )
        if ctx.eligibility is not None:
            fwd_returns = fwd_returns.where(ctx.eligibility)

        # 预计算日度收益 (用于日度NAV累积, 解决样本点不足和收益重叠问题)
        close = data_manager.get("close", all_dates, universe_static)
        if close.empty or close.dropna(how="all").empty:
            raise RuntimeError(
                "close行情为空，回测失败关闭；不得生成平坦净值"
            )
        prepare = getattr(data_manager, "prepare_close_data", None)
        if callable(prepare):
            daily_returns, self._close_tradable = prepare(close)
        else:
            daily_returns, self._close_tradable = prepare_close_data(close)

        schedule_getter = getattr(data_manager, "get_contract_schedule", None)
        self._contract_schedule = (
            schedule_getter(all_dates, universe_static)
            if callable(schedule_getter)
            else None
        )

        # 验证 fwd_returns 有效性
        if fwd_returns.empty or fwd_returns.dropna(how="all").empty:
            raise RuntimeError(
                "fwd_returns全为空，回测失败关闭；请检查close覆盖与持有期 "
                f"({all_dates[0].date()} ~ {all_dates[-1].date()})"
            )
        else:
            valid_count = fwd_returns.dropna(how="all").shape[0]
            log.info(f"fwd_returns 有效日期: {valid_count}/{len(fwd_returns)}")

        risk_prefetch = getattr(risk_model, "prepare_data", None)
        if risk_prefetch is not None:
            risk_prefetch(data_manager, all_dates, universe_static)
        if self.dynamic_risk_controller is not None:
            self.dynamic_risk_controller.prepare_data(
                data_manager, all_dates, universe_static
            )

        # D3修复: 风险模型不再全样本估计, 移入_refit_alpha的walk-forward流程
        # 首次估计在run()主循环的第一个调仓日触发

        return rebalance_dates, universe_static, all_dates, processed_all, fwd_returns, daily_returns

    def _refit_alpha(
        self,
        alpha_model: ReturnModel,
        processed_all: Dict[str, FactorMatrix],
        fwd_returns: pd.DataFrame,
        date: Date,
        all_dates: DateIndex,
        last_fit_idx: int,
        retrain_freq: int,
        universe_static: Universe,
        risk_model: Optional[RiskModel] = None,
        data_manager: Optional[DataManager] = None,
        daily_returns: Optional[pd.DataFrame] = None,
    ) -> Tuple[Optional[Date], int]:
        """增量重训 alpha 模型 (walk-forward, 无前视偏差).

        D2修复: train_cutoff用交易日索引而非日历天数,避免前视偏差.
        D3修复: risk_model在此处用训练期数据重新估计,避免全样本未来函数.
        D4修复: 训练数据用training_window截断为滚动窗口,而非全部历史.
        """
        from core.logger import get_logger
        log = get_logger("multi_factor")

        # D2修复: 用交易日索引计算train_cutoff_idx, 避免日历天数导致的前视
        date_np = np.datetime64(pd.Timestamp(date))
        date_idx = int(np.searchsorted(
            all_dates.values, date_np, side="left"
        ))
        # 训练截止 = 当前日期往前 holding_period+1 个交易日
        # (确保fwd_returns.loc[train_cutoff]的收益在date之前已实现)
        train_cutoff_idx = date_idx - self.holding_period - 1
        if train_cutoff_idx < 0:
            return None, last_fit_idx

        need_refit = (
            last_fit_idx < 0
            or train_cutoff_idx - last_fit_idx >= retrain_freq
        )
        if not need_refit:
            return None, last_fit_idx

        # D4修复: 用training_window截断为滚动窗口 (而非全部历史)
        # 滚动窗口 = 最近 training_window 个交易日
        start_idx = max(0, train_cutoff_idx + 1 - self.training_window)
        end_idx = train_cutoff_idx + 1
        if end_idx - start_idx < self.training_window:
            log.info(
                "alpha remains observation-only @ %s: %s/%s training bars",
                pd.Timestamp(date).date(), end_idx - start_idx, self.training_window,
            )
            return None, last_fit_idx

        # 滚动窗口切片 (iloc为O(1)视图)
        train_factors = {
            name: f.iloc[start_idx:end_idx]
            for name, f in processed_all.items()
        }
        train_fwd = fwd_returns.iloc[start_idx:end_idx]

        # IC 监控: 更新滚动 IC 历史, 在 fit 前剔除失效因子
        # (确保 fit 和 predict 使用同一套因子, 避免维度不匹配)
        if self.ic_monitor is not None:
            self.ic_monitor.update(train_factors, train_fwd)
            inactive = self.ic_monitor.inactive_factors
            if inactive:
                logger.info(
                    f"IC监控 @ {date.date()}: 剔除 {len(inactive)} 个失效因子 "
                    f"({', '.join(sorted(inactive)[:5])}...)"
                )
                train_factors = {
                    name: f for name, f in train_factors.items()
                    if name not in inactive
                }

        # CR-011: 删除 synthesizer.flip_signs 的动态更新.
        # 原因: processed_all 在 _prepare_backtest_data 阶段已合成, 此处修改 flip_signs
        # 不会重新合成历史矩阵; 且训练因子名为合成名, 无法匹配 flip_signs 的原始因子名.
        # 方向统一由 Alpha 模型系数决定 (见 CR-010 修复).

        if not train_factors:
            raise RuntimeError(f"no active factors remain for alpha fit @ {date}")
        try:
            alpha_model.fit(train_factors, train_fwd, universe_static)
        except Exception as exc:
            raise RuntimeError(f"alpha fit failed @ {date}: {exc}") from exc

        # 风险模型使用截至 t-1 的日收益，不使用 H 日重叠前向收益。
        if (
            risk_model is not None
            and data_manager is not None
            and daily_returns is not None
        ):
            risk_start_idx = max(0, date_idx - self.training_window)
            risk_factors = {
                name: factor.iloc[risk_start_idx:date_idx]
                for name, factor in processed_all.items()
            }
            risk_daily = daily_returns.iloc[risk_start_idx:date_idx]
            try:
                risk_model.estimate(
                    data_manager, risk_factors, risk_daily, universe_static
                )
            except Exception as exc:
                raise RuntimeError(f"risk-model fit failed @ {date}: {exc}") from exc

        # Record only after both fitted components are valid.
        self._last_fit_factors = set(train_factors)
        return all_dates[train_cutoff_idx], train_cutoff_idx

    def _predict_returns(
        self,
        alpha_model: ReturnModel,
        processed_all: Dict[str, FactorMatrix],
        date: Date,
        universe: Universe,
    ) -> pd.Series:
        """预测截面预期收益."""
        # 使用最近一次 refit 的因子集, 确保 predict 和 fit 维度一致
        if self._last_fit_factors is not None:
            factor_names = self._last_fit_factors & set(processed_all.keys())
        else:
            factor_names = set(processed_all.keys())
        if self._last_fit_factors is None:
            raise RuntimeError(f"alpha model has not been fitted @ {date}")
        current_factors = {}
        missing = []
        for name in sorted(factor_names):
            row = self._factor_frame_asof(processed_all[name], date)
            if row is None:
                missing.append(name)
            else:
                current_factors[name] = row
        if missing:
            raise ValueError(
                f"no point-in-time factor history for {len(missing)} factors: "
                f"{', '.join(missing[:5])}"
            )
        if not hasattr(alpha_model, "predict"):
            raise TypeError("alpha model has no predict method")
        try:
            predicted = alpha_model.predict(current_factors, universe, date)
        except Exception as exc:
            raise RuntimeError(f"alpha prediction failed @ {date}: {exc}") from exc
        if not isinstance(predicted, pd.Series):
            raise TypeError(
                f"alpha predict returned {type(predicted).__name__}, expected Series"
            )
        if predicted.index.has_duplicates:
            raise ValueError("alpha prediction index must be unique")
        predicted = predicted.reindex(universe).replace([np.inf, -np.inf], np.nan)
        if predicted.isna().any():
            raise ValueError("alpha prediction contains missing or non-finite values")
        predicted = predicted.astype(float)
        if self.asset_selector is not None:
            predicted = self.asset_selector.apply(predicted, date=date)
        vintages = max(int(self.forecast_averaging_vintages), 1)
        if vintages > 1:
            self._forecast_vintages.append(predicted.copy())
            self._forecast_vintages = self._forecast_vintages[-vintages:]
            predicted = pd.concat(self._forecast_vintages, axis=1).mean(axis=1)
        return predicted

    def _optimize_period(
        self,
        predicted: pd.Series,
        risk_model: RiskModel,
        current_weights: pd.Series,
        constraints: List[Constraint],
        optimizer: Optimizer,
        date: Date,
        universe: Universe,
        realized_vol: float,
        current_drawdown: float = 0.0,
    ) -> pd.Series:
        """单期组合优化."""
        from core.logger import get_logger
        log = get_logger("multi_factor")

        if not isinstance(predicted, pd.Series) or predicted.empty:
            raise ValueError("optimizer requires a non-empty prediction Series")
        if predicted.index.has_duplicates:
            raise ValueError("prediction index must be unique")
        predicted = predicted.reindex(universe)
        if predicted.isna().any() or not np.isfinite(predicted.values).all():
            raise ValueError("prediction contains NaN/Inf")

        try:
            optimization_universe = pd.Index(universe)
            optimization_predicted = predicted
            optimization_current = current_weights
            if self.asset_selector is not None:
                optimization_universe = pd.Index(
                    predicted.index[predicted.abs() > 1e-12]
                )
                if optimization_universe.empty:
                    return pd.Series(0.0, index=universe, dtype=float)
                optimization_predicted = predicted.reindex(optimization_universe)
                optimization_current = current_weights.reindex(
                    optimization_universe
                ).fillna(0.0)
            log.info(
                f"优化 @ {date.date()} | n={len(optimization_universe)} "
                f"predicted_std={float(optimization_predicted.std()):.6f} "
                f"realized_vol={realized_vol:.4f} dd={current_drawdown:.2%}"
            )
            target_w = optimizer.optimize(
                optimization_predicted, risk_model, optimization_current,
                constraints, self.cost_model, date, optimization_universe,
                realized_vol=realized_vol,
                current_drawdown=current_drawdown,
            )
            if not isinstance(target_w, pd.Series):
                raise TypeError(
                    f"optimizer returned {type(target_w).__name__}, expected Series"
                )
            if target_w.index.has_duplicates or not target_w.index.equals(
                optimization_universe
            ):
                raise ValueError("optimizer returned misaligned weight index")
            target_w = target_w.reindex(universe).fillna(0.0)
            if target_w.isna().any() or not np.isfinite(target_w.values).all():
                raise ValueError("optimizer returned NaN/Inf weights")
            if self.dynamic_risk_controller is not None:
                covariance = risk_model.covariance(date, universe)
                annual_volatility = (
                    self.dynamic_risk_controller.annual_volatility_asof(
                        date, universe
                    )
                )
                target_w, _ = self.dynamic_risk_controller.apply(
                    target_w,
                    covariance,
                    universe,
                    annual_volatility=annual_volatility,
                )
                if not isinstance(target_w, pd.Series):
                    raise TypeError("dynamic risk controller must return a Series")
                target_w = target_w.reindex(universe)
                if target_w.isna().any() or not np.isfinite(target_w.values).all():
                    raise ValueError("dynamic risk controller returned invalid weights")
            log.info(
                f"优化完成 @ {date.date()} | std={float(target_w.std()):.6f} "
                f"max={float(target_w.max()):.4f} "
                f"n_active={int((target_w > 0.001).sum())} "
                f"gross={float(target_w.abs().sum()):.4f}"
            )
            return target_w
        except Exception as exc:
            raise RuntimeError(f"portfolio optimization failed @ {date}: {exc}") from exc


    def _compute_realized_vol(
        self, returns_arr: np.ndarray, end_exclusive: int
    ) -> float:
        """计算决策时已经完成的近期收益波动率 (用于 vol targeting).

        优化: 直接对 numpy 数组切片, 避免 pandas .iloc 开销.

        CR-006修复:
        - returns_arr 是日度组合收益, 年化应使用 sqrt(252), 而非 sqrt(252/5)
          (旧因子 sqrt(252/HOLDING_PERIOD) 错误地按 holding_period 频率年化)
        ``end_exclusive`` 是已完成收益区间的右边界。收盘 T 决策调用时
        传入 T 所在位置加一，因此包含已知的 R[T]，但不会读取 R[T+1]。
        """
        if end_exclusive < 5:
            return 0.0
        start = max(0, end_exclusive - 20)
        recent = returns_arr[start:end_exclusive]
        # CR-006: 过滤 NaN (避免 NaN 传播导致 std 为 NaN)
        recent = recent[~np.isnan(recent)]
        if recent.size > 3:
            # numpy std (ddof=1) 比 pandas Series.std() 快
            std_val = float(np.std(recent, ddof=1))
            if std_val > 0:
                # CR-006修复: 日度收益年化用 sqrt(252), 而非 sqrt(252/5)
                return std_val * np.sqrt(TRADING_DAYS_PER_YEAR)
        return 0.0

    def run(
        self,
        data_manager: DataManager,
        factor_engine: FactorEngine,
        processor: FactorProcessor,
        factor_names: List[str],
        universe_schedule: UniverseSchedule,
        rebalance_dates: DateIndex,
        alpha_model: ReturnModel,
        risk_model: RiskModel,
        optimizer: Optimizer,
        constraints: List[Constraint],
    ) -> BacktestResult:
        """端到端回测主流程.

        Returns:
            BacktestResult: 包含 NAV、目标权重历史和绩效指标.
        """
        from core.logger import get_logger
        log = get_logger("multi_factor")
        if not factor_names:
            raise ValueError("backtest requires at least one configured factor")
        self._last_fit_factors = None
        self._forecast_vintages = []
        if self.asset_selector is not None and hasattr(self.asset_selector, "reset"):
            self.asset_selector.reset()

        # === Step 1: 数据准备 ===
        self._close_tradable = None
        self._contract_schedule = None
        rebalance_dates, universe_static, all_dates, processed_all, fwd_returns, daily_returns = (
            self._prepare_backtest_data(
                data_manager, factor_engine, processor, factor_names,
                universe_schedule, rebalance_dates, risk_model,
            )
        )

        if len(universe_static) == 0:
            raise RuntimeError(
                "backtest universe is empty after data and eligibility gates"
            )

        # === Step 2: 日度NAV累积 ===
        # 改为遍历所有交易日: 调仓日更新权重, 每日用 w·daily_ret 累积 NAV
        # 解决: (1)样本点不足 (2)收益重叠 (3)metrics年化频率错误
        bt_start_date = rebalance_dates[0]
        # CR-005修复: 区分 data_load_end (含 tail) 与 evaluation_end (评估结束日=配置结束日)
        # - data_load_end = tail_end: 在 _prepare_backtest_data 中用于加载行情数据 (含
        #   holding_period+15 天 tail, 供 fwd_returns 计算)
        # - evaluation_end = rebalance_dates[-1]: 评估结束日, 净值/收益/指标必须裁剪到此日期,
        #   避免绩效包含配置结束日后的约 holding_period+15 天
        evaluation_end = pd.Timestamp(rebalance_dates[-1])
        bt_dates = all_dates[(all_dates >= bt_start_date) & (all_dates <= evaluation_end)]
        n_bt = len(bt_dates)

        # 调仓日集合 (O(1) 查找)
        rebalance_set = set(rebalance_dates)
        # 预分配 numpy 数组 (日度频率)
        nav_arr = np.full(n_bt, np.nan, dtype=np.float64)
        returns_arr = np.full(n_bt, np.nan, dtype=np.float64)
        gross_returns_arr = np.full(n_bt, 0.0, dtype=np.float64)
        turnover_arr = np.full(n_bt, 0.0, dtype=np.float64)
        roll_turnover_arr = np.full(n_bt, 0.0, dtype=np.float64)
        executed_turnover_arr = np.full(n_bt, 0.0, dtype=np.float64)
        executed_roll_turnover_arr = np.full(n_bt, 0.0, dtype=np.float64)
        cost_arr = np.full(n_bt, 0.0, dtype=np.float64)
        trade_cost_arr = np.full(n_bt, 0.0, dtype=np.float64)
        holding_cost_arr = np.full(n_bt, 0.0, dtype=np.float64)
        asset_returns_arr = np.zeros(
            (n_bt, len(universe_static)), dtype=np.float64
        )
        effective_weights_arr = np.zeros(
            (n_bt, len(universe_static)), dtype=np.float64
        )
        contributions_arr = np.zeros(
            (n_bt, len(universe_static)), dtype=np.float64
        )
        if n_bt > 0:
            nav_arr[0] = self.initial_nav

        current_weights = pd.Series(dtype=float)
        held_contracts = pd.Series(dtype="object")
        weights_history: List[Tuple[pd.Timestamp, pd.Series]] = []

        last_fit_idx = -1
        # pending_cost 暂存本次收盘决策产生的成本，在新权重生效的次日扣除。
        pending_cost = 0.0
        pending_turnover = 0.0
        pending_roll_turnover = 0.0

        for i, date in enumerate(bt_dates):
            cost_today = 0.0  # 当日调仓产生的成本 (若调仓且权重有变化)
            pending_weights = None  # 调仓日决策的权重, 次日生效

            # --- 每日: 计算组合日收益 (用当日生效的权重) ---
            if date in daily_returns.index:
                daily_ret = daily_returns.loc[date].reindex(current_weights.index)
            else:
                daily_ret = pd.Series(
                    np.nan, index=current_weights.index, dtype=float
                )

            effective_trade_cost = float(pending_cost)
            pending_cost = 0.0
            executed_turnover_arr[i] = pending_turnover
            executed_roll_turnover_arr[i] = pending_roll_turnover
            pending_turnover = 0.0
            pending_roll_turnover = 0.0

            holding_cost = 0.0

            # 扣除昨日收盘决策暂存的交易成本（新权重生效的次日扣除）。
            # 顺序: 先用当日生效权重算 port_ret, 再扣 pending_cost (来自昨日调仓决策).
            if i > 0 and self.cost_model is not None:
                holding_estimator = getattr(
                    self.cost_model, "estimate_holding_cost", None
                )
                if holding_estimator is not None:
                    holding_cost = float(holding_estimator(current_weights, date))
                    if not np.isfinite(holding_cost) or holding_cost < 0:
                        raise RuntimeError(
                            f"invalid daily holding cost at {date}: {holding_cost}"
                        )
            step = close_marked_step(
                current_weights,
                daily_ret,
                trade_cost=effective_trade_cost,
                holding_cost=holding_cost,
            )
            port_ret = step.net_return
            gross_returns_arr[i] = step.gross_return
            cost_arr[i] = step.trade_cost + step.holding_cost
            trade_cost_arr[i] = step.trade_cost
            holding_cost_arr[i] = step.holding_cost
            effective_weights_arr[i, :] = (
                step.effective_weights.reindex(universe_static).fillna(0.0)
            )
            asset_returns_arr[i, :] = (
                step.asset_returns.reindex(universe_static).fillna(0.0)
            )
            contributions_arr[i, :] = (
                step.contributions.reindex(universe_static).fillna(0.0)
            )

            # 先完成 T 日收盘估值，再用真实漂移后的权重形成 T 日目标。
            # 因而收益口径严格为 W[T-1] * R[T]，目标 W[T] 于下一根生效。
            current_weights = step.end_weights
            if i == 0:
                nav_arr[i] = self.initial_nav * (1.0 + port_ret)
            else:
                nav_arr[i] = nav_arr[i - 1] * (1.0 + port_ret)
            returns_arr[i] = port_ret

            peak_so_far = np.nanmax(nav_arr[:i + 1])
            current_drawdown = (
                float((nav_arr[i] - peak_so_far) / peak_so_far)
                if peak_so_far > 0.0 else 0.0
            )

            # --- 调仓日: 收盘后决策新权重，下一交易日生效 ---
            if date in rebalance_set:
                # CR-004修复: 查询 universe 时使用"不晚于当前调仓日的最近 schedule key",
                # 不静默回退到完整 universe_static (找不到时返回空 Index, 跳过本次调仓)
                universe = self._lookup_universe_schedule(
                    universe_schedule, date, universe_static
                )
                if self._eligibility_mask is not None:
                    eligible_history = self._eligibility_mask.loc[
                        self._eligibility_mask.index <= date
                    ]
                    if eligible_history.empty:
                        universe = pd.Index([])
                    else:
                        eligible_row = eligible_history.iloc[-1].fillna(False)
                        universe = universe.intersection(
                            eligible_row.index[eligible_row.astype(bool)]
                        )
                if len(universe) == 0:
                    active = (
                        not current_weights.empty
                        and bool(current_weights.abs().gt(1e-12).any())
                    )
                    if active or self._last_fit_factors is not None:
                        raise RuntimeError(
                            f"eligible universe is empty at rebalance close {date.date()}"
                        )
                else:
                    # 增量 alpha 重训 (含D3: 风险模型walk-forward重估计)
                    _, last_fit_idx = self._refit_alpha(
                        alpha_model, processed_all, fwd_returns, date, all_dates,
                        last_fit_idx, self.retrain_freq, universe_static,
                        risk_model=risk_model, data_manager=data_manager,
                        daily_returns=daily_returns,
                    )

                    # Before the first complete training window, the strategy is
                    # explicitly observation-only and holds cash.
                    if self._last_fit_factors is not None:
                        predicted = self._predict_returns(
                            alpha_model, processed_all, date, universe
                        )
                        realized_vol = self._compute_realized_vol(returns_arr, i + 1)
                        target_w = self._optimize_period(
                            predicted, risk_model, current_weights, constraints,
                            optimizer, date, universe, realized_vol,
                            current_drawdown=current_drawdown,
                        )
                        pending_weights = target_w
                        # Keep the observed close decision separate from the
                        # effective exposure recorded by the daily ledger.
                        weights_history.append((date, pending_weights.copy()))

            # Rebalance from the post-mark drifted exposure. The new target is
            # effective for the next bar. Contract rolls are checked every day,
            # including days without a root-weight decision.
            if i < n_bt - 1 and (
                pending_weights is not None or self._contract_schedule is not None
            ):
                next_date = bt_dates[i + 1]
                desired_target = (
                    pending_weights if pending_weights is not None else step.end_weights
                )
                transition_target, transition_current = align_transition_weights(
                    desired_target, step.end_weights
                )
                can_trade = pd.Series(True, index=transition_target.index)
                if self._close_tradable is not None:
                    missing_dates = [
                        value for value in (date, next_date)
                        if value not in self._close_tradable.index
                    ]
                    if missing_dates:
                        raise RuntimeError(
                            "close tradability mask is missing transition dates: "
                            + ", ".join(str(pd.Timestamp(value).date()) for value in missing_dates)
                        )
                    can_trade = (
                        self._close_tradable.loc[date]
                        .reindex(transition_target.index)
                        .eq(True)
                        & self._close_tradable.loc[next_date]
                        .reindex(transition_target.index)
                        .eq(True)
                    )
                    transition_target = transition_target.where(
                        can_trade, transition_current
                    )

                target_contracts = None
                current_contracts = None
                cost_target = transition_target
                cost_current = transition_current
                if self._contract_schedule is not None:
                    target_contracts = (
                        self._contract_schedule.loc[next_date]
                        .reindex(transition_target.index)
                        .copy()
                    )
                    current_contracts = held_contracts.reindex(
                        transition_target.index
                    )
                    target_contracts = target_contracts.where(
                        can_trade, current_contracts
                    )
                    turnover, roll_turnover = contract_transition_turnover(
                        transition_target,
                        transition_current,
                        current_contracts=current_contracts,
                        target_contracts=target_contracts,
                    )
                    cost_target, cost_current = contract_transition_weight_vectors(
                        transition_target,
                        transition_current,
                        current_contracts=current_contracts,
                        target_contracts=target_contracts,
                    )
                    held_contracts = target_contracts.where(
                        transition_target.abs().gt(1e-12), pd.NA
                    )
                else:
                    turnover, roll_turnover = contract_transition_turnover(
                        transition_target, transition_current
                    )
                turnover_arr[i] = turnover
                roll_turnover_arr[i] = roll_turnover
                if self.cost_model is not None:
                    try:
                        if turnover > 1e-12:
                            cost_today = float(
                                self.cost_model.estimate_cost(
                                    cost_target,
                                    cost_current,
                                    next_date,
                                )
                            )
                        if not np.isfinite(cost_today) or cost_today < 0.0:
                            raise RuntimeError(
                                f"invalid transition cost at {date}: {cost_today}"
                            )
                    except Exception as e:
                        raise RuntimeError(
                            f"research transaction cost failed at {date}: {e}"
                        ) from e
                current_weights = transition_target
                pending_cost = cost_today
                pending_turnover = turnover
                pending_roll_turnover = roll_turnover

        # === Step 3: 绩效评估 ===
        # 日度 NAV → periods_per_year=252 天然正确, 无需手动调整
        if not np.isfinite(nav_arr).all():
            raise RuntimeError("backtest NAV contains an unaccounted NaN or infinity")
        nav = pd.Series(nav_arr, index=bt_dates)
        returns_series = pd.Series(returns_arr, index=bt_dates)
        gross_returns_series = pd.Series(gross_returns_arr, index=bt_dates)
        turnover_series = pd.Series(turnover_arr, index=bt_dates)
        roll_turnover_series = pd.Series(roll_turnover_arr, index=bt_dates)
        executed_turnover_series = pd.Series(
            executed_turnover_arr, index=bt_dates
        )
        executed_roll_turnover_series = pd.Series(
            executed_roll_turnover_arr, index=bt_dates
        )
        cost_series = pd.Series(cost_arr, index=bt_dates)
        asset_returns_frame = pd.DataFrame(
            asset_returns_arr, index=bt_dates, columns=universe_static
        )
        effective_weights_frame = pd.DataFrame(
            effective_weights_arr, index=bt_dates, columns=universe_static
        )
        contributions_frame = pd.DataFrame(
            contributions_arr, index=bt_dates, columns=universe_static
        )

        # CR-005修复: 防御性裁剪到评估结束日 (bt_dates 已裁剪, 这里再次确保 nav/returns/turnover
        # 不越过配置结束日 evaluation_end; weights_history 也一并裁剪)
        eval_mask = nav.index <= evaluation_end
        nav = nav.loc[eval_mask]
        returns_series = returns_series.loc[eval_mask]
        gross_returns_series = gross_returns_series.loc[eval_mask]
        turnover_series = turnover_series.loc[eval_mask]
        roll_turnover_series = roll_turnover_series.loc[eval_mask]
        executed_turnover_series = executed_turnover_series.loc[eval_mask]
        executed_roll_turnover_series = executed_roll_turnover_series.loc[
            eval_mask
        ]
        cost_series = cost_series.loc[eval_mask]
        asset_returns_frame = asset_returns_frame.loc[eval_mask]
        effective_weights_frame = effective_weights_frame.loc[eval_mask]
        contributions_frame = contributions_frame.loc[eval_mask]

        nav_before = nav.shift(1)
        if not nav_before.empty:
            nav_before.iloc[0] = self.initial_nav
        ledger_daily = pd.DataFrame(
            {
                "nav_before": nav_before,
                "nav_after": nav,
                "gross_return": gross_returns_series,
                "trade_cost": trade_cost_arr[eval_mask],
                "holding_cost": holding_cost_arr[eval_mask],
                "net_return": returns_series,
                "decision_turnover": turnover_series,
                "turnover": executed_turnover_series,
                "half_turnover": 0.5 * executed_turnover_series,
                "roll_turnover": roll_turnover_series,
                "executed_traded_notional": executed_turnover_series,
                "executed_roll_turnover": executed_roll_turnover_series,
                "gross_exposure": effective_weights_frame.abs().sum(axis=1),
                "net_exposure": effective_weights_frame.sum(axis=1),
                "active_instruments": effective_weights_frame.abs().gt(
                    1e-12
                ).sum(axis=1),
            },
            index=nav.index,
        )
        ledger_metadata = default_research_ledger_metadata()
        ledger_metadata.update(
            {
                "cost_model": (
                    type(self.cost_model).__name__
                    if self.cost_model is not None
                    else "none"
                ),
                "cost_stage": str(
                    getattr(self.cost_model, "cost_stage", "unspecified")
                ),
                "periods_per_year": float(
                    getattr(self.cost_model, "periods_per_year", 252.0)
                ),
                "turnover_cost_rate": float(
                    getattr(self.cost_model, "turnover_cost_rate", 0.0)
                ),
                "annual_roll_cost": getattr(
                    self.cost_model, "annual_roll_cost", None
                ),
                "transaction_cost_timing": (
                    "transition_cost_next_bar_plus_holding_cost_each_bar"
                ),
                "turnover_cost_policy": (
                    "delegated_to_cost_model"
                    if self.cost_model is not None
                    else "not_charged"
                ),
                "transition_cost_policy": (
                    "cost_model_on_concrete_contract_transition"
                    if self._contract_schedule is not None
                    else "cost_model_on_root_transition"
                ),
                "contract_schedule_policy": (
                    "point_in_time_schedule"
                    if self._contract_schedule is not None
                    else "unavailable"
                ),
                "rollover_cost_policy": (
                    "explicit_turnover_plus_declared_holding_cost_policy"
                    if self._contract_schedule is not None
                    else "declared_holding_cost_policy_only"
                ),
                "untradable_transition_policy": (
                    "require_decision_and_next_close_then_freeze"
                    if self._close_tradable is not None
                    else "not_provided"
                ),
            }
        )
        research_ledger = ResearchReturnLedger(
            daily=ledger_daily,
            asset_returns=asset_returns_frame,
            effective_weights=effective_weights_frame,
            contributions=contributions_frame,
            metadata=ledger_metadata,
        )
        research_ledger.validate()

        # 日度频率 periods_per_year=252 (默认值, 无需覆盖)
        # The first row is the NAV anchor and has no preceding holding interval.
        metrics = compute_all_metrics(nav, returns=returns_series.iloc[1:])
        # avg_turnover保留“有成交日均值”兼容口径，其余字段显式报告
        # 交换手的逐日、年化与全期口径。
        executed_intervals = executed_turnover_series.iloc[1:]
        reb_turnover = executed_intervals[executed_intervals > 0]
        if not reb_turnover.empty:
            metrics["avg_turnover"] = float(reb_turnover.mean())
        else:
            metrics["avg_turnover"] = 0.0
        metrics["avg_daily_turnover"] = float(
            executed_intervals.mean()
        ) if len(executed_intervals) else 0.0
        metrics["annualized_turnover"] = (
            metrics["avg_daily_turnover"] * TRADING_DAYS_PER_YEAR
        )
        metrics["total_turnover"] = float(executed_intervals.sum())
        metrics["total_transaction_cost"] = float(cost_series.sum())
        metrics["avg_transaction_cost"] = (
            float(cost_series.iloc[1:].mean()) if len(cost_series) > 1 else 0.0
        )
        metrics["total_trade_cost"] = float(trade_cost_arr[eval_mask].sum())
        metrics["total_holding_cost"] = float(holding_cost_arr[eval_mask].sum())
        metrics["total_roll_turnover"] = float(
            executed_roll_turnover_series.iloc[1:].sum()
        )

        # 样本分段诊断: 前75% / 后25%。不宣称独立样本外验证。
        from backtest.metrics import compute_split_metrics
        split_metrics = compute_split_metrics(nav, returns_series, train_ratio=0.75)

        # 权重历史: 仅调仓日记录 (CR-005: 裁剪到评估结束日, 不超过配置结束日)
        if weights_history:
            wh_pairs = [
                (d, w) for d, w in weights_history
                if pd.Timestamp(d) <= evaluation_end
            ]
            if wh_pairs:
                wh = pd.DataFrame(
                    [w for _, w in wh_pairs],
                    index=[d for d, _ in wh_pairs],
                )
            else:
                wh = pd.DataFrame()
        else:
            wh = pd.DataFrame()

        return BacktestResult(
            nav=nav,
            weights_history=wh,
            metrics=metrics,
            turnover=executed_turnover_series,
            decision_turnover=turnover_series,
            costs=cost_series,
            research_ledger=research_ledger,
            split_metrics=split_metrics,
            failure_ledger=[],
        )
