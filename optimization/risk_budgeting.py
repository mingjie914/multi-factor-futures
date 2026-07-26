from __future__ import annotations

import logging
from typing import List, Optional, Dict, Any  # noqa: F401

import pandas as pd
import numpy as np  # noqa: F401

# cvxpy 启用: 使用 OSQP/CLARABEL 求解器 (纯 Python 绑定, 不触发 0xC0000005 崩溃)
# 与 mean_variance.py 保持一致的求解器策略.
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


@register("optimizer", "risk_budgeting")
class RiskBudgetingOptimizer(Optimizer):
    """风险预算优化器.

    弱 alpha 环境 (IC~0.02) 下比均值-方差优化更稳健.
    核心思想: 以数值求解的真实风险预算组合为锚, alpha 仅做微调.

    目标函数 (凸, cvxpy 可解):
        min  0.5 * risk_aversion * w'Σw
             - alpha_scale * (mu'w)
             + cost_penalty * |w - w_prev|_1
             + rb_penalty * ||w - w_rp||^2
        s.t. 约束 (ConstraintContext 机制, 与 mean_variance.py 一致)

    其中 w_rp 由 Spinu 对数障碍形式求解，使风险贡献比例接近预算 b_i。

    弱 alpha 时: risk_aversion 和 rb_penalty 主导 → 接近风险平价 (稳健)
    强 alpha 时: alpha 项主导 → 偏离风险平价追逐收益 (进取)

    理论上风险预算软约束为 ``cp.sum(cp.abs(rc_i - target_rc_i)) <= tol``,
    其中 ``rc_i = w_i * (cov@w)_i``. 但该表达式含 w 的二次项相乘,
    非凸, cvxpy DCP 规则不接受. 这里用 ``||w - w_rp||^2`` 作为凸代理:
    w_rp 是风险预算的解析解, 跟踪它即可实现风险预算效果.
    """

    # 求解器优先级: OSQP (QP 专用, 纯 Python 绑定) > CLARABEL (Rust, 更精确) > SCIPY (纯 Python 兜底)
    # 与 mean_variance.py 一致, 避免 ECOS/SCS 的 C 扩展触发 0xC0000005 崩溃.
    _SOLVER_CHAIN = ["OSQP", "CLARABEL", "SCIPY"]

    def __init__(
        self,
        risk_budget: np.ndarray = None,
        risk_aversion: float = 2.0,
        cost_penalty: float = 0.5,
        solver: str = "OSQP",
    ):
        """
        Args:
            risk_budget: 各品种风险预算 (非负, 归一化后使用). None=等权 ERC.
            risk_aversion: 风险厌恶系数. 0=纯风险预算 (忽略 alpha); 大值=更接近风险平价.
            cost_penalty: 交易成本惩罚 (|w-w_prev|_1 的系数).
            solver: 首选求解器 (实际会按 _SOLVER_CHAIN 回退).
        """
        self.risk_budget = risk_budget
        self.risk_aversion = risk_aversion
        self.cost_penalty = cost_penalty
        self.solver = solver
        # 风险平价跟踪强度 (内部参数, 可手动调整).
        # 控制 w 偏离 w_rp 的惩罚强度: 大→接近纯风险平价; 小→接近 MV.
        self.rb_penalty = 1.0

    @staticmethod
    def _erc_weights(cov: np.ndarray, budget: np.ndarray) -> np.ndarray:
        """Solve long-only risk budgeting using the convex Spinu formulation."""
        from scipy.optimize import minimize

        n = len(budget)
        budget = np.asarray(budget, dtype=float)
        if len(cov) != n or not np.isfinite(cov).all():
            raise ValueError("invalid covariance for risk-budget solve")
        budget = np.maximum(budget, 1e-12)
        budget = budget / budget.sum()
        cov = (cov + cov.T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        cov = (eigenvectors * np.maximum(eigenvalues, 1e-10)) @ eigenvectors.T

        def objective(x):
            return 0.5 * float(x @ cov @ x) - float(budget @ np.log(x))

        def gradient(x):
            return cov @ x - budget / x

        vol = np.sqrt(np.maximum(np.diag(cov), 1e-12))
        initial = (1.0 / vol) / np.sum(1.0 / vol)
        result = minimize(
            objective,
            initial,
            jac=gradient,
            method="L-BFGS-B",
            bounds=[(1e-10, None)] * n,
            options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 2000},
        )
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f"risk-budget solve failed: {result.message}")
        weights = result.x / result.x.sum()
        contributions = weights * (cov @ weights)
        total = float(contributions.sum())
        if total <= 0 or not np.isfinite(total):
            raise RuntimeError("risk-budget solution has invalid total risk")
        residual = float(np.max(np.abs(contributions / total - budget)))
        if residual > 5e-4:
            raise RuntimeError(f"risk-budget contribution residual {residual:.3e}")
        return weights

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
        """风险预算优化.

        签名与 MeanVarianceOptimizer.optimize 完全一致, 可直接替换.
        """
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
            raise RuntimeError("cvxpy is required for validated risk-budget optimization")

        try:
            import cvxpy as cp

            # 量级对齐: 期货日收益量级 ~0.001, cov 量级 ~0.0003
            # (与 mean_variance.py 完全一致的 alpha_scale 计算)
            mu_std = float(np.std(mu))
            cov_scale = float(np.mean(np.diag(cov))) if cov.size > 0 else 1.0
            if mu_std > 1e-10 and cov_scale > 1e-12:
                alpha_scale = float(np.sqrt(cov_scale)) / mu_std
            else:
                alpha_scale = 1.0
            alpha_scale = max(1.0, min(alpha_scale, 1000.0))

            # === 真实风险预算参考解 ===
            if self.risk_budget is not None:
                budget = np.asarray(self.risk_budget, dtype=float)
                if len(budget) != n:
                    raise ValueError(
                        f"risk_budget length {len(budget)} != universe length {n}"
                    )
                else:
                    budget = np.maximum(budget, 1e-12)
                    budget = budget / budget.sum()
            else:
                # ERC: 等风险贡献, 各品种预算均等
                budget = np.ones(n) / n

            w_rp = self._erc_weights(cov, budget)

            # 从约束提取 gross_limit, 将 w_rp 缩放到可行量级
            # (避免 w_rp 在 gross=1 而 w 在 gross=3 时, 跟踪项被量级差主导)
            gross_limit = 1.0
            for c in constraints:
                if getattr(c, "name", "") == "leverage":
                    gross_limit = getattr(c, "limit", 1.0)
                    break
            w_rp = w_rp * min(gross_limit, 1.0)

            # === cvxpy 优化 ===
            w = cp.Variable(n)

            try:
                psd_cov = cp.psd_wrap(cov)
            except Exception as e:
                raise RuntimeError(f"psd_wrap failed @ {date.date()}: {e}") from e

            risk_term = cp.quad_form(w, psd_cov)
            ret_term = alpha_scale * (mu @ w)

            # 交易成本项 (与 mean_variance.py 一致)
            cost_term = 0
            if cost_model is not None and not current_weights.empty:
                prev_w = current_weights.reindex(universe).fillna(0.0).values
                cost_rate = marginal_turnover_cost_rate(cost_model, universe, date)
                cost_term = (
                    alpha_scale * self.cost_penalty * cost_rate
                    * cp.norm1(w - prev_w)
                )

            # 风险平价跟踪项 (凸代理):
            # 理论上风险预算软约束为 cp.sum(cp.abs(rc_i - target_rc_i)) <= tol,
            # 但 rc_i = w_i * (cov@w)_i 非凸, cvxpy 不接受.
            # 这里用 ||w - w_rp||^2 作为凸代理, w_rp 是风险预算的解析解.
            # rb_penalty 控制跟踪强度: 大→接近纯风险平价; 小→接近 MV.
            # 量级缩放: 乘 cov_scale 使 rb_term 与 risk_term (w'Σw) 量级一致,
            # 避免 rb_term (|w|²量级) 远大于 risk_term (|w|²×cov_diag量级) 导致
            # 风险平价跟踪项完全主导优化结果.
            cov_scale_val = float(np.mean(np.diag(cov))) if cov.size > 0 else 1.0
            rb_term = self.rb_penalty * cov_scale_val * cp.sum_squares(w - w_rp)

            objective = cp.Minimize(
                0.5 * self.risk_aversion * risk_term
                - ret_term
                + cost_term
                + rb_term
            )

            # 应用约束: 收集到 var_dict["_constraints"] (与 mean_variance.py 一致)
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
            # 不做 clip(0,1) —— 允许做空 (lower<0)
            # 仅清理数值噪声 (|w|<1e-6 视为 0)
            weights = weights.where(weights.abs() > 1e-6, 0.0)

            n_long = int((weights > 0.001).sum())
            n_short = int((weights < -0.001).sum())
            gross = float(weights.abs().sum())
            net = float(weights.sum())
            log.info(
                f"风险预算优化 @ {date.date()} | solver={outcome.solver} "
                f"status={outcome.status} residual={outcome.max_constraint_violation:.2e} n={n} "
                f"多={n_long} 空={n_short} 总={gross:.3f} 净={net:.3f} "
                f"max={float(weights.max()):.3f} min={float(weights.min()):.3f} "
                f"std={float(weights.std()):.4f} scale={alpha_scale:.1f} "
                f"rb={self.rb_penalty:.2f}"
            )
            return weights
        except Exception as e:
            raise RuntimeError(
                f"validated risk-budget optimization failed @ {date}: {e}"
            ) from e

    def _heuristic_optimize(
        self,
        expected_returns: ExpectedReturns,
        current_weights: WeightVector,
        constraints: List[Constraint],
        universe: Universe,
        current_drawdown: float = 0.0,
    ) -> WeightVector:
        """启发式风险平价回退.

        无 cvxpy 或 cov 含 NaN 时使用. 纯风险驱动, 不使用 alpha.
        无 cov 时无法计算 vol_i, 退化为等权 (ERC 假设各品种 vol 相等)
        或按 risk_budget 比例分配.

        支持约束: position_limit, net_exposure, leverage, turnover, sector_exposure,
                  drawdown_control.
        """
        from core.logger import get_logger
        log = get_logger("multi_factor")

        n = len(universe)
        if n == 0:
            return pd.Series(dtype=float)

        # 提取约束参数 (与 mean_variance.py 启发式一致)
        pos_lower = -0.2
        pos_upper = 0.2
        net_lower = -0.5
        net_upper = 0.5
        gross_limit = 3.0
        turnover_limit = None
        has_turnover_constraint = False
        sector_limit = 0.0  # 0 表示不约束
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
                warning_dd = getattr(c, "warning_dd", -0.05)
                critical_dd = getattr(c, "critical_dd", -0.10)
                base_leverage = getattr(c, "leverage_limit", gross_limit)
                if current_drawdown <= critical_dd:
                    gross_limit = 0.1
                elif current_drawdown <= warning_dd:
                    gross_limit = base_leverage * 0.5

        # 风险预算权重 (无 cov 时用预算比例作为权重代理, 或等权 ERC)
        if self.risk_budget is not None and len(self.risk_budget) == n:
            budget = np.maximum(np.asarray(self.risk_budget, dtype=float), 1e-12)
            budget = budget / budget.sum()
        else:
            budget = np.ones(n) / n

        target_gross = min(gross_limit, 1.0)  # 启发式默认 1x 杠杆
        weights = pd.Series(budget, index=universe, dtype=float)
        weights = weights / weights.abs().sum() * target_gross

        # position_limit: 截断到 [pos_lower, pos_upper]
        weights = weights.clip(lower=pos_lower, upper=pos_upper)

        # net_exposure: 风险平价全多头, 需缩放到 [net_lower, net_upper]
        net = float(weights.sum())
        if net > net_upper and net > 0:
            weights = weights * (net_upper / net)
        elif net < net_lower and net < 0:
            weights = weights * (net_lower / net)

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
                sectors = pd.Series(
                    [SECTOR_MAP.get(str(s), "other") for s in universe],
                    index=universe,
                )
                for sector in sectors.unique():
                    sector_mask = sectors == sector
                    sector_net = float(weights[sector_mask].sum())
                    if abs(sector_net) > sector_limit:
                        scale = sector_limit / abs(sector_net)
                        weights[sector_mask] = weights[sector_mask] * scale
            except ImportError:
                pass

        # turnover 限制: 与上期权重混合
        if (
            has_turnover_constraint
            and turnover_limit is not None
            and not current_weights.empty
        ):
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
            f"启发式风险平价 | n={n} 多={n_long} 空={n_short} "
            f"总={float(weights.abs().sum()):.3f} 净={float(weights.sum()):.3f} "
            f"max={float(weights.max()):.3f} min={float(weights.min()):.3f}"
        )

        return weights
