from __future__ import annotations

import logging
from typing import List, Optional, Dict, Any  # noqa: F401

import pandas as pd
import numpy as np  # noqa: F401

# cvxpy 启用: 使用 OSQP/CLARABEL 求解器 (纯 Python 绑定, 不触发 0xC0000005 崩溃)
# 早期 ECOS/SCS 的 C 扩展在 Windows 上触发访问违规, 已通过切换到 OSQP 解决.
_HAS_CVXPY = True
_CVXPY_WARNED = False

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
from optimization.costs import marginal_turnover_cost_rate


@register("optimizer", "mean_variance")
class MeanVarianceOptimizer(Optimizer):
    """均值-方差优化器. max mu'w - 0.5*gamma*w'Sigma*w - cost_penalty*|w-w_prev|_1"""

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
        import logging
        from core.logger import get_logger
        log = get_logger("multi_factor")

        n = len(universe)
        if n == 0:
            return pd.Series(dtype=float)

        mu = expected_returns.reindex(universe).fillna(0).values
        cov = (
            risk_model.covariance(date, universe)
            .reindex(index=universe, columns=universe)
            .values
        )

        # 数值安全校验: NaN/Inf 会触发 C 求解器的 0xC0000005 硬崩溃
        if np.any(np.isnan(mu)) or np.any(np.isinf(mu)):
            mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
            log.warning(f"mu 含 NaN/Inf @ {date.date()}, 已零填充")
        if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
            raise ValueError(f"covariance contains NaN/Inf @ {date.date()}")
        else:
            # 对称化和负对角线修复
            cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
            cov = (cov + cov.T) / 2  # 强制对称
            cov_diag = np.diag(cov)
            neg_mask = cov_diag < 0
            if np.any(neg_mask):
                cov[neg_mask, neg_mask] = np.abs(cov[neg_mask, neg_mask])
                log.warning(f"cov 对角线含负值 @ {date.date()}, 已取绝对值")

        # cvxpy 不可用 → 直接启发式 (仅首次打印警告)
        if not _HAS_CVXPY:
            global _CVXPY_WARNED
            if not _CVXPY_WARNED:
                log.warning("cvxpy 已禁用 (0xC0000005 崩溃), 全局使用启发式优化器")
                _CVXPY_WARNED = True
            return self._heuristic_optimize(
                expected_returns, current_weights, constraints, universe,
                date=date, current_drawdown=current_drawdown,
            )

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

    def _heuristic_optimize(
        self,
        expected_returns: ExpectedReturns,
        current_weights: WeightVector,
        constraints: List[Constraint],
        universe: Universe,
        date: Date = None,
        current_drawdown: float = 0.0,
    ) -> WeightVector:
        """无 cvxpy 时的启发式优化: 按 alpha 预测值符号分配权重.

        支持约束: net_exposure, leverage, position_limit (含负下限), turnover,
                  sector_exposure, drawdown_control.
        """
        import logging
        from core.logger import get_logger
        log = get_logger("multi_factor")

        n = len(universe)
        mu = expected_returns.reindex(universe).fillna(0.0)

        # 提取约束参数
        pos_lower = -0.2
        pos_upper = 0.2
        net_lower = -0.5
        net_upper = 0.5
        gross_limit = 3.0
        turnover_limit = None  # CR-022: 默认不施加换手限制 (仅当配置了turnover约束时才生效)
        sector_limit = 0.0  # 0 表示不约束
        has_turnover_constraint = False
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
            elif name == "turnover":
                has_turnover_constraint = True
                turnover_limit = getattr(c, "limit", 0.30)
            elif name == "sector_exposure":
                sector_limit = getattr(c, "limit", 0.15)
            elif name == "drawdown_control":
                # 回撤控制: 根据当前回撤动态调整 gross_limit
                warning_dd = getattr(c, "warning_dd", -0.05)
                critical_dd = getattr(c, "critical_dd", -0.10)
                base_leverage = getattr(c, "leverage_limit", gross_limit)
                if current_drawdown <= critical_dd:
                    gross_limit = 0.1  # 临界回撤: 清仓
                elif current_drawdown <= warning_dd:
                    gross_limit = base_leverage * 0.5  # 预警: 降杠杆50%

        # CR-022: 全零预测 → 保持上期权重 (不再自动做多等权)
        if mu.abs().max() < 1e-10:
            if current_weights is not None and not current_weights.empty:
                log.warning(f"预测全零 @ {date}, 保持上期权重 (不自动做多)")
                return current_weights.reindex(universe).fillna(0.0)
            else:
                log.warning(f"预测全零且无上期权重 @ {date}, 返回零仓位")
                return pd.Series(0.0, index=universe)

        # 按 alpha 符号分配权重: 正→多头, 负→空头
        # 权重 = alpha_i / sum(|alpha|) * target_gross
        target_gross = min(gross_limit, 1.0)  # 启发式默认 1x 杠杆
        abs_sum = float(mu.abs().sum())
        if abs_sum < 1e-10:
            return pd.Series(0.0, index=universe)
        weights = mu / abs_sum * target_gross

        # position_limit: 截断到 [pos_lower, pos_upper]
        weights = weights.clip(lower=pos_lower, upper=pos_upper)

        # net_exposure: 调整净敞口到 [net_lower, net_upper]
        net = float(weights.sum())
        if net < net_lower:
            # 净敞口过低, 需要增加多头 or 减少空头
            shortage = net_lower - net
            long_mask = weights > 0
            short_mask = weights < 0
            if long_mask.any() and short_mask.any():
                # 同时有多空: 增加多头, 减少空头各一半
                half = shortage / 2.0
                long_weights = weights[long_mask]
                weights[long_mask] = long_weights + half * long_weights / long_weights.sum()
                short_weights = weights[short_mask]
                weights[short_mask] = short_weights - half * short_weights.abs() / short_weights.abs().sum()
            elif long_mask.any():
                # 只有多头: 全部增加多头
                long_weights = weights[long_mask]
                weights[long_mask] = long_weights + shortage * long_weights / long_weights.sum()
            elif short_mask.any():
                # 只有空头: 减少空头 (即增加权重, 向零靠拢)
                short_weights = weights[short_mask]
                weights[short_mask] = short_weights + shortage * short_weights.abs() / short_weights.abs().sum()
        elif net > net_upper:
            # 净敞口过高: 等比缩放到目标净敞口
            weights = weights * (net_upper / net)

        # 再次 clip 确保 position_limit 不越界
        weights = weights.clip(lower=pos_lower, upper=pos_upper)

        # leverage: 总杠杆不超过 gross_limit
        gross = float(weights.abs().sum())
        if gross > gross_limit and gross > 0:
            weights = weights / gross * gross_limit

        # sector_exposure: 每个板块净敞口不超过 sector_limit
        if sector_limit > 0:
            try:
                from core.sectors import SECTOR_MAP
                sectors = pd.Series([SECTOR_MAP.get(str(s), "other") for s in universe], index=universe)
                for sector in sectors.unique():
                    sector_mask = sectors == sector
                    sector_net = float(weights[sector_mask].sum())
                    if abs(sector_net) > sector_limit:
                        # 等比缩放该板块权重到限制内
                        scale = sector_limit / abs(sector_net)
                        weights[sector_mask] = weights[sector_mask] * scale
            except ImportError:
                pass

        # CR-022: turnover 限制 — 仅当配置了 turnover 约束时才施加
        if has_turnover_constraint and turnover_limit is not None and not current_weights.empty:
            prev = current_weights.reindex(universe).fillna(0.0)
            turnover = float((weights - prev).abs().sum())
            if turnover > turnover_limit and turnover > 0:
                scale = turnover_limit / turnover
                weights = prev + (weights - prev) * scale

        # 清理数值噪声
        weights = weights.where(weights.abs() > 1e-6, 0.0)

        n_long = int((weights > 0.001).sum())
        n_short = int((weights < -0.001).sum())
        log.info(
            f"启发式优化 | n={n} 多={n_long} 空={n_short} "
            f"总={float(weights.abs().sum()):.3f} 净={float(weights.sum()):.3f} "
            f"max={float(weights.max()):.3f} min={float(weights.min()):.3f}"
        )

        return weights
