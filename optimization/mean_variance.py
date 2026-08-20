from __future__ import annotations

from typing import List, Optional, Dict, Any  # noqa: F401

import pandas as pd
import numpy as np  # noqa: F401

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
from optimization.solver_utils import solve_validated, validated_psd_covariance
from optimization.costs import marginal_turnover_cost_rate


@register("optimizer", "mean_variance")
class MeanVarianceOptimizer(Optimizer):
    """Research-only optimizer for calibrated expected-return forecasts.

    The objective is ``mu'w - 0.5*gamma*w'Sigma*w - cost*|w-w_prev|_1``.
    ``mu`` must be comparable out-of-sample return forecasts in the same unit
    and horizon. Cross-sectional ranks, z-scores, and direction-only signals
    do not satisfy that requirement and should use the formal hierarchical
    futures allocator instead.
    """

    allocation_role = "cross_sectional_forecast_utility_optimization"
    deployment_status = "research_only"

    # 求解器优先级: OSQP (QP专用, 纯Python绑定) > CLARABEL (Rust, 更精确) > SCIPY (纯Python兜底)
    # OSQP/CLARABEL 不触发 0xC0000005 崩溃 (与 ECOS/SCS 的 C 扩展不同)
    _SOLVER_CHAIN = ["OSQP", "CLARABEL", "SCIPY"]

    def __init__(
        self,
        risk_aversion: float = 2.0,
        cost_penalty: float = 0.5,
        solver: str = "OSQP",
    ):
        self.risk_aversion = risk_aversion
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

        aligned_returns = pd.Series(expected_returns, dtype=float).reindex(universe)
        if aligned_returns.isna().any() or not np.isfinite(aligned_returns).all():
            raise ValueError(f"expected returns contain missing/invalid values @ {date.date()}")
        mu = aligned_returns.to_numpy(dtype=float)
        cov = (
            risk_model.covariance(date, universe)
            .reindex(index=universe, columns=universe)
            .values
        )

        # 数值安全校验: NaN/Inf 会触发 C 求解器的 0xC0000005 硬崩溃
        try:
            cov = validated_psd_covariance(cov)
        except ValueError as exc:
            raise ValueError(f"invalid covariance @ {date.date()}: {exc}") from exc

        try:
            import cvxpy as cp

            w = cp.Variable(n)

            # 量级对齐: 期货日收益量级 ~0.001, cov量级 ~0.0003
            mu_std = float(np.std(mu))
            cov_scale = float(np.mean(np.diag(cov))) if cov.size > 0 else 1.0
            if mu_std > 1e-10 and cov_scale > 1e-12:
                alpha_scale = float(np.sqrt(cov_scale)) / mu_std
            else:
                alpha_scale = 1.0
            alpha_scale = max(1.0, min(alpha_scale, 1000.0))

            # 目标 (带收益放大)
            try:
                psd_cov = cp.psd_wrap(cov)
            except Exception as e:
                raise RuntimeError(f"psd_wrap failed @ {date.date()}: {e}") from e
            risk_term = cp.quad_form(w, psd_cov)
            ret_term = alpha_scale * (mu @ w)
            cost_term = 0
            if cost_model is not None and not current_weights.empty:
                prev_w = current_weights.reindex(universe).fillna(0.0).values
                cost_rate = marginal_turnover_cost_rate(cost_model, universe, date)
                cost_term = (
                    alpha_scale * self.cost_penalty * cost_rate
                    * cp.norm1(w - prev_w)
                )
            objective = cp.Maximize(
                ret_term - 0.5 * self.risk_aversion * risk_term - cost_term
            )

            # 应用约束: 收集到 var_dict["_constraints"] (cvxpy 1.7+ problem.constraints 只读)
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

            problem = cp.Problem(objective, all_constraints)
            solver_chain = [self.solver] + [
                name for name in self._SOLVER_CHAIN if name != self.solver
            ]
            outcome = solve_validated(problem, w, solver_chain)

            weights = pd.Series(w.value, index=universe)
            # 不再做 clip(0,1) —— 允许做空 (lower<0)
            # 仅清理数值噪声 (|w|<1e-6 视为 0)
            weights = weights.where(weights.abs() > 1e-6, 0.0)

            n_long = int((weights > 0.001).sum())
            n_short = int((weights < -0.001).sum())
            gross = float(weights.abs().sum())
            net = float(weights.sum())
            log.info(
                f"cvxpy优化 @ {date.date()} | solver={outcome.solver} "
                f"status={outcome.status} residual={outcome.max_constraint_violation:.2e} n={n} "
                f"多={n_long} 空={n_short} 总={gross:.3f} 净={net:.3f} "
                f"max={float(weights.max()):.3f} min={float(weights.min()):.3f} "
                f"std={float(weights.std()):.4f} scale={alpha_scale:.1f}"
            )
            return weights
        except Exception as e:
            raise RuntimeError(f"validated cvxpy optimization failed @ {date}: {e}") from e
