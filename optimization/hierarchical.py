"""层级优化器: 板块内优化 + 板块间权重分配.

两层级优化流程:
1. 第一层 (板块内): 对每个板块独立运行 MV/RB 优化器, 得到板块内相对权重
2. 第二层 (板块间): 按指定预算或等权分配板块权重
3. 全局校准: 用 cvxpy 软投影到全局约束可行域 (不改变板块内相对结构)

在弱 alpha 环境下, 分层方法比单层全品种优化更稳健, 避免单一板块主导.
"""
from __future__ import annotations

from typing import List, Optional, Dict, Any  # noqa: F401

import numpy as np  # noqa: F401
import pandas as pd

from core.types import (
    Date,
    ExpectedReturns,
    Universe,
    WeightVector,
)
from core.interfaces import (
    Optimizer,
    RiskModel,
    CostModel,
    Constraint,
    ConstraintContext,
)
from core.registry import register
from optimization.solver_utils import solve_validated


@register("optimizer", "hierarchical_sector")
class SectorLayeredOptimizer(Optimizer):
    """板块层级优化器.

    两层级优化: 先板块内优化品种权重, 再板块间分配权重.
    适用于弱 alpha 环境, 避免单一板块主导.

    参数:
        inner_optimizer_type: 板块内优化器类型 ("mean_variance" 或 "risk_budgeting")
        inner_kwargs: 传给板块内优化器的构造参数
        sector_weights: 板块权重预算, 如
            {"ferrous": 0.2, "nonferrous": 0.15, "precious": 0.1,
             "stock_index": 0.1, "bond": 0.1}, None=等权
        cost_penalty: 交易成本惩罚系数
        solver: cvxpy 求解器名称
    """

    # 求解器优先级 (与 MeanVarianceOptimizer 一致, 避免触发 0xC0000005 崩溃)
    _SOLVER_CHAIN = ["OSQP", "CLARABEL", "SCIPY"]

    def __init__(
        self,
        inner_optimizer_type: str = "mean_variance",
        inner_kwargs: dict = None,
        sector_weights: dict = None,
        cost_penalty: float = 0.5,
        solver: str = "OSQP",
    ):
        self.inner_optimizer_type = inner_optimizer_type
        self.inner_kwargs = inner_kwargs or {}
        self.sector_weights = sector_weights
        self.cost_penalty = cost_penalty
        self.solver = solver

    def optimize(
        self,
        expected_returns: ExpectedReturns,
        risk_model: RiskModel,
        current_weights: WeightVector,
        constraints: List[Constraint],
        cost_model: Optional[CostModel],
        date: Date,
        universe: Universe,
        realized_vol: float = 0.0,
        current_drawdown: float = 0.0,
    ) -> WeightVector:
        from core.logger import get_logger
        log = get_logger("multi_factor")

        n = len(universe)
        if n == 0:
            return pd.Series(dtype=float)

        # 使用统一的八板块映射；股指/国债、有色/贵金属分别优化。
        try:
            from core.sectors import SECTOR_MAP
        except ImportError as exc:
            raise RuntimeError("SECTOR_MAP is required by hierarchical optimizer") from exc

        sectors: Dict[str, list] = {}
        for i, sym in enumerate(universe):
            sec = SECTOR_MAP.get(str(sym), "other")
            sectors.setdefault(sec, []).append(i)

        log.info(f"层级优化 @ {date.date()} | n={n} 板块数={len(sectors)}")

        # ========== 第一层: 板块内优化 ==========
        sector_internal_weights: Dict[str, pd.Series] = {}
        for sector, indices in sectors.items():
            sector_universe = [universe[i] for i in indices]
            ns = len(sector_universe)

            # 单品种板块没有横截面可优化，沿用已有方向；首次建仓才使用 alpha 方向。
            if ns < 2:
                previous = current_weights.reindex(sector_universe).fillna(0.0)
                if float(previous.abs().sum()) > 1e-12:
                    sector_w = previous / float(previous.abs().sum())
                else:
                    signal = float(expected_returns.reindex(sector_universe).fillna(0.0).iloc[0])
                    sector_w = pd.Series(np.sign(signal), index=sector_universe)
                sector_internal_weights[sector] = sector_w
                log.info(f"  板块 {sector}: n={ns} (单品种方向权重)")
                continue

            sector_er = expected_returns.reindex(sector_universe).fillna(0)
            sector_cw = (
                current_weights.reindex(sector_universe).fillna(0)
                if not current_weights.empty
                else pd.Series(dtype=float)
            )

            inner = self._create_inner_optimizer()

            # 板块内优化 (空约束列表, 板块内不做约束)
            sector_w = inner.optimize(
                sector_er, risk_model, sector_cw, [], cost_model,
                date, sector_universe, realized_vol,
                current_drawdown=current_drawdown,
            )

            # 归一化: 板块内相对权重 (gross=1, 保留多空符号)
            gross = float(sector_w.abs().sum())
            if gross > 1e-10:
                sector_w = sector_w / gross
            else:
                raise RuntimeError(f"sector {sector!r} optimizer returned zero exposure")

            sector_internal_weights[sector] = sector_w
            log.info(
                f"  板块 {sector}: n={ns} 净={float(sector_w.sum()):.3f} "
                f"gross={float(sector_w.abs().sum()):.3f}"
            )

        # ========== 第二层: 板块间权重分配 ==========
        sector_alloc = self._allocate_sector_weights(sectors, log)

        # 拼接全局目标权重: 板块权重 × 板块内相对权重
        w_target = pd.Series(0.0, index=universe)
        for sector, indices in sectors.items():
            sector_universe = [universe[i] for i in indices]
            sw = sector_alloc.get(sector, 1.0 / max(len(sectors), 1))
            internal = sector_internal_weights[sector].reindex(sector_universe).fillna(0)
            for sym in sector_universe:
                w_target[sym] = sw * internal.get(sym, 0.0)

        log.info(
            "板块权重: " + ", ".join(
                f"{s}={sector_alloc.get(s, 0):.3f}" for s in sectors
            )
        )

        # ========== 全局约束校准 (软投影) ==========
        final_w = self._global_calibration(
            w_target, constraints, expected_returns, current_weights,
            risk_model, date, universe, realized_vol, log,
            current_drawdown=current_drawdown,
        )

        # 结果摘要
        n_long = int((final_w > 0.001).sum())
        n_short = int((final_w < -0.001).sum())
        gross = float(final_w.abs().sum())
        net = float(final_w.sum())
        log.info(
            f"层级优化完成 @ {date.date()} | 多={n_long} 空={n_short} "
            f"总={gross:.3f} 净={net:.3f} "
            f"max={float(final_w.max()):.3f} min={float(final_w.min()):.3f}"
        )
        return final_w

    def _create_inner_optimizer(self):
        """创建板块内优化器实例.

        根据 inner_optimizer_type 选择 MeanVariance 或 RiskBudgeting.
        若 RiskBudgeting 不可用, 回退到 MeanVariance.
        """
        if self.inner_optimizer_type == "risk_budgeting":
            try:
                from optimization.risk_budgeting import RiskBudgetingOptimizer
                return RiskBudgetingOptimizer(**self.inner_kwargs)
            except ImportError:
                from core.logger import get_logger
                get_logger("multi_factor").warning(
                    "RiskBudgetingOptimizer 不可用, 回退到 MeanVarianceOptimizer"
                )
                from optimization.mean_variance import MeanVarianceOptimizer
                return MeanVarianceOptimizer(**self.inner_kwargs)
        else:
            from optimization.mean_variance import MeanVarianceOptimizer
            return MeanVarianceOptimizer(**self.inner_kwargs)

    def _allocate_sector_weights(
        self,
        sectors: Dict[str, list],
        log,
    ) -> Dict[str, float]:
        """分配板块间权重.

        若 sector_weights 指定, 直接使用 (缺失板块补等权, 归一化到 sum=1);
        否则等权 (各板块 1/N).
        """
        n_sectors = len(sectors)
        if n_sectors == 0:
            return {}

        if self.sector_weights is not None:
            # 使用指定权重, 缺失板块补等权
            alloc: Dict[str, float] = {}
            for s in sectors:
                alloc[s] = float(self.sector_weights.get(s, 1.0 / n_sectors))
            total = sum(abs(v) for v in alloc.values())
            if total > 1e-10:
                alloc = {k: v / total for k, v in alloc.items()}
            return alloc
        else:
            # 等权 (各板块 1/N)
            w = 1.0 / n_sectors
            return {s: w for s in sectors}

    def _global_calibration(
        self,
        w_target: pd.Series,
        constraints: List[Constraint],
        expected_returns: ExpectedReturns,
        current_weights: WeightVector,
        risk_model: RiskModel,
        date: Date,
        universe: Universe,
        realized_vol: float,
        log,
        current_drawdown: float = 0.0,
    ) -> WeightVector:
        """全局约束软校准.

        最小化 ||w - w_target||^2 s.t. 全局约束.
        软投影, 不改变板块内相对结构 (仅微调以满足全局约束).
        """
        import cvxpy as cp

        n = len(universe)
        w = cp.Variable(n)
        target = w_target.reindex(universe).fillna(0.0).values

        # 目标: 最小化与目标权重的距离 (+ 交易成本惩罚)
        if not current_weights.empty:
            prev = current_weights.reindex(universe).fillna(0.0).values
            objective = cp.Minimize(
                cp.sum_squares(w - target)
                + self.cost_penalty * cp.norm1(w - prev)
            )
        else:
            objective = cp.Minimize(cp.sum_squares(w - target))

        # 应用全局约束 (复用 ConstraintContext 机制)
        ctx = ConstraintContext(
            expected_returns=expected_returns,
            current_weights=current_weights,
            risk_model=risk_model,
            industry=pd.Series(dtype=object),
            date=date,
            universe=universe,
            realized_vol=realized_vol,
            current_drawdown=current_drawdown,
        )
        var_dict = {"w": w}
        for c in constraints:
            try:
                c.apply(None, var_dict, ctx)
            except Exception as e:
                raise RuntimeError(
                    f"constraint {getattr(c, 'name', '?')} failed to apply: "
                    f"{type(e).__name__}: {e}"
                ) from e
        all_constraints = var_dict.get("_constraints", [])

        # 无约束时直接返回目标
        if not all_constraints:
            return w_target.reindex(universe).fillna(0.0)

        problem = cp.Problem(objective, all_constraints)

        solver_chain = [self.solver] + [
            name for name in self._SOLVER_CHAIN if name != self.solver
        ]
        outcome = solve_validated(problem, w, solver_chain)

        weights = pd.Series(w.value, index=universe)
        # 清理数值噪声 (|w|<1e-6 视为 0)
        weights = weights.where(weights.abs() > 1e-6, 0.0)

        log.info(
            f"全局校准 | solver={outcome.solver} status={outcome.status} "
            f"residual={outcome.max_constraint_violation:.2e} 约束数={len(all_constraints)} "
            f"总={float(weights.abs().sum()):.3f} 净={float(weights.sum()):.3f}"
        )
        return weights

    def _heuristic_fallback(
        self,
        sectors: Dict[str, list],
        sector_alloc: Dict[str, float],
        universe: Universe,
        current_weights: WeightVector,
        constraints: List[Constraint],
        log,
        current_drawdown: float = 0.0,
    ) -> WeightVector:
        """启发式回退: 板块内等权 × 板块权重, 再做简单约束截断.

        当 cvxpy 全局校准失败时使用. 不改变板块间相对结构.
        支持约束: position_limit, net_exposure, leverage, sector_exposure, drawdown_control.
        """
        n = len(universe)
        if n == 0:
            return pd.Series(dtype=float)

        # 板块内等权 × 板块权重
        weights = pd.Series(0.0, index=universe)
        for sector, indices in sectors.items():
            sector_universe = [universe[i] for i in indices]
            ns = len(sector_universe)
            if ns == 0:
                continue
            sw = sector_alloc.get(sector, 1.0 / max(len(sectors), 1))
            ew = sw / ns  # 板块内等权 × 板块权重
            for sym in sector_universe:
                weights[sym] = ew

        # 提取约束参数 (与 MeanVarianceOptimizer._heuristic_optimize 一致)
        pos_lower = -0.2
        pos_upper = 0.2
        net_lower = -0.5
        net_upper = 0.5
        gross_limit = 3.0
        sector_limit = 0.0
        for c in constraints:
            name = getattr(c, "name", "")
            if name == "position_limit":
                pos_lower = getattr(c, "lower", -0.2)
                pos_upper = getattr(c, "upper", 0.2)
            elif name == "net_exposure":
                net_lower = getattr(c, "lower", -0.5)
                net_upper = getattr(c, "upper", 0.5)
            elif name == "leverage":
                gross_limit = getattr(c, "limit", 3.0)
            elif name == "sector_exposure":
                sector_limit = getattr(c, "limit", 0.0)
            elif name == "drawdown_control":
                warning_dd = getattr(c, "warning_dd", -0.05)
                critical_dd = getattr(c, "critical_dd", -0.10)
                base_leverage = getattr(c, "leverage_limit", gross_limit)
                if current_drawdown <= critical_dd:
                    gross_limit = 0.1
                elif current_drawdown <= warning_dd:
                    gross_limit = base_leverage * 0.5

        # position_limit 截断
        weights = weights.clip(lower=pos_lower, upper=pos_upper)

        # net_exposure 调整
        net = float(weights.sum())
        if net < net_lower:
            weights = weights + (net_lower - net) / n
        elif net > net_upper and net > 1e-10:
            weights = weights * (net_upper / net)

        # 再次 clip
        weights = weights.clip(lower=pos_lower, upper=pos_upper)

        # leverage 截断
        gross = float(weights.abs().sum())
        if gross > gross_limit and gross > 0:
            weights = weights / gross * gross_limit

        # sector_exposure 截断
        if sector_limit > 0:
            try:
                from core.sectors import SECTOR_MAP
                sector_series = pd.Series(
                    [SECTOR_MAP.get(str(s), "other") for s in universe],
                    index=universe,
                )
                for sector in sector_series.unique():
                    mask = sector_series == sector
                    sector_net = float(weights[mask].sum())
                    if abs(sector_net) > sector_limit:
                        scale = sector_limit / abs(sector_net)
                        weights[mask] = weights[mask] * scale
            except ImportError:
                pass

        # 清理数值噪声
        weights = weights.where(weights.abs() > 1e-6, 0.0)

        log.info(
            f"启发式回退 | n={n} "
            f"总={float(weights.abs().sum()):.3f} 净={float(weights.sum()):.3f}"
        )
        return weights
