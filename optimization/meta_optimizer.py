"""元优化器 — 优化各子组合的资本配置权重.

基于各子组合的历史收益和协方差矩阵, 优化资本配置权重.
目标: 最大化整体组合夏普比率 (或其他目标).
整体组合的风险约束在此层执行.

层级关系:
- 子组合内部 (宽松约束): 各子组合满仓运行, 净敞口/杠杆可以更大
- 元优化器 (整体约束): 通过分配资本权重控制整体组合的风险
"""
from __future__ import annotations

import logging
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from optimization.solver_utils import solve_validated

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# cvxpy 启用: 使用 OSQP 求解器 (纯 Python 绑定, 不触发 0xC0000005 崩溃)
# 早期 ECOS/SCS 的 C 扩展在 Windows 上触发访问违规, 已通过切换到 OSQP 解决.
# ---------------------------------------------------------------------------
_HAS_CVXPY = True


class MetaOptimizer:
    """子组合资本权重元优化器.

    Methods:
        - max_sharpe: 最大化 (w·μ) / sqrt(w^T Σ w), 约束 Σw=1, w∈[min,max]
        - min_variance: 最小化 w^T Σ w, 约束 Σw=1
        - risk_parity: 各子组合风险贡献相等
    """

    def __init__(
        self,
        method: str = "max_sharpe",
        min_weight: float = 0.1,
        max_weight: float = 0.6,
        target_volatility: float = 0.15,
        estimation_window: int = 252,
        covariance_shrinkage: float = 0.30,
    ):
        self.method = method
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.target_volatility = target_volatility
        self.estimation_window = estimation_window
        self.covariance_shrinkage = float(covariance_shrinkage)
        # Relative sleeve weights sum to one. This separate scale controls the
        # amount of capital invested, so volatility targeting is not normalized away.
        self.last_capital_scale = 1.0
        self.last_failure: Optional[dict] = None

    def optimize(
        self,
        returns_df: pd.DataFrame,
        current_weights: Optional[np.ndarray] = None,
        date: Optional[pd.Timestamp] = None,
    ) -> np.ndarray:
        """优化子组合资本权重.

        CR-003 修复: 严格排除当日收益, 只用 t-1 及之前的数据估计参数.
        权重最早作用于 t 日收益, 不存在同日收益前视.

        Args:
            returns_df: 子组合日收益率 DataFrame (dates × sub_names), 完整历史
            current_weights: 当前权重 (用于平滑, 可选)
            date: 当前决策日期 (权重从 t 日开始生效, 估计只用 < t 的数据)

        Returns:
            归一化的资本权重数组 (sum=1, 各权重在 [min_weight, max_weight] 内)
        """
        n = returns_df.shape[1]
        self.last_failure = None
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([1.0])

        # CR-003: 严格排除当日收益 (side="left" 不包含 date 本身)
        if date is not None:
            pos = returns_df.index.searchsorted(date, side="left")
            returns_df = returns_df.iloc[:pos]
        if len(returns_df) > self.estimation_window:
            returns_df = returns_df.iloc[-self.estimation_window:]
        returns_df = returns_df.dropna(how="all")

        if len(returns_df) < 20:
            # 样本不足时保持现有配置；首次调用才使用等权。
            logger.warning(
                f"元优化器样本不足 ({len(returns_df)} < 20), 保持现有相对权重"
            )
            self.last_capital_scale = 1.0
            if current_weights is not None and len(current_weights) == n:
                return self._project_to_constraints(current_weights, n)
            return self._project_to_constraints(np.ones(n) / n, n)

        # 计算各子组合的年化收益和协方差矩阵 (向量化)
        mu = returns_df.mean().values * 252  # 年化收益
        cov = returns_df.cov().values * 252  # 样本协方差
        # 强制对称化 (消除数值误差导致的不对称)
        cov = (cov + cov.T) / 2
        # 数值安全: NaN/Inf填充
        if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
            diag_var = np.nan_to_num(np.diag(cov), nan=0.01, posinf=0.01, neginf=0.01)
            cov = np.diag(np.maximum(diag_var, 1e-6))

        try:
            if self.method == "max_sharpe":
                w = self._max_sharpe(mu, cov, n)
            elif self.method == "min_variance":
                w = self._min_variance(cov, n)
            elif self.method == "shrinkage_min_variance":
                cov = self._shrink_covariance(cov)
                w = self._min_variance(cov, n)
            elif self.method in {"risk_parity", "erc"}:
                w = self._risk_parity(cov, n)
            elif self.method == "inverse_volatility":
                w = self._inverse_volatility(cov, n)
            elif self.method == "hrp":
                w = self._hrp(cov, n)
            else:
                raise ValueError(f"unknown meta optimization method {self.method!r}")
        except Exception as exc:
            self.last_failure = {
                "stage": "meta_optimization",
                "date": str(pd.Timestamp(date).date()) if date is not None else None,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "fallback": "hold_previous_relative_weights",
            }
            if current_weights is None or len(current_weights) != n:
                raise RuntimeError(f"meta optimization failed without prior weights: {exc}") from exc
            logger.warning("元优化失败，保持上期相对权重: %s", exc)
            w = self._project_to_constraints(current_weights, n)

        # 与当前权重平滑 (避免剧烈换手)
        if current_weights is not None and len(current_weights) == n:
            smooth = 0.5  # 新旧权重各占一半
            w = smooth * w + (1 - smooth) * current_weights
            # 重新归一化到 [min, max] 约束
            w = self._project_to_constraints(w, n)

        # Relative weights and total capital scale are separate decisions.
        self.last_capital_scale = 1.0
        if self.target_volatility > 0 and len(w) > 0:
            port_var = float(w @ cov @ w)
            port_vol = np.sqrt(max(port_var, 1e-12))
            if port_vol > 1e-8:
                self.last_capital_scale = float(np.clip(
                    self.target_volatility / port_vol, 0.5, 2.0
                ))

        names = list(returns_df.columns)
        port_vol = (
            float(np.sqrt(w @ cov @ w)) * self.last_capital_scale
            if len(w) > 0 else 0.0
        )
        logger.info(
            f"元优化器({self.method}) @ {date}: "
            f"权重={dict(zip(names, [f'{x:.2f}' for x in w]))}, "
            f"预期年化波动={port_vol:.2%}, "
            f"目标波动={self.target_volatility:.2%}, "
            f"资本比例={self.last_capital_scale:.3f}"
        )
        return w

    def _max_sharpe(self, mu: np.ndarray, cov: np.ndarray, n: int) -> np.ndarray:
        """最大化夏普比率: (w·μ) / sqrt(w^T Σ w).

        使用 cvxpy 凸优化的近似: 最大化 w·μ - 0.5 * risk_aversion * w^T Σ w
        然后归一化. 或者直接用解析解 (当无约束时).
        """
        if _HAS_CVXPY:
            import cvxpy as cp
            w = cp.Variable(n)
            risk_aversion = 2.0
            objective = cp.Maximize(
                mu @ w - 0.5 * risk_aversion * cp.quad_form(w, cp.psd_wrap(cov))
            )
            constraints = [
                cp.sum(w) == 1,
                w >= self.min_weight,
                w <= self.max_weight,
            ]
            prob = cp.Problem(objective, constraints)
            solve_validated(prob, w, ["OSQP", "CLARABEL"])
            return np.asarray(w.value, dtype=float)
        raise RuntimeError("cvxpy is required for validated max_sharpe optimization")

    def _min_variance(self, cov: np.ndarray, n: int) -> np.ndarray:
        """最小化方差: w^T Σ w."""
        if _HAS_CVXPY:
            import cvxpy as cp
            w = cp.Variable(n)
            objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov)))
            constraints = [
                cp.sum(w) == 1,
                w >= self.min_weight,
                w <= self.max_weight,
            ]
            prob = cp.Problem(objective, constraints)
            solve_validated(prob, w, ["OSQP", "CLARABEL"])
            return np.asarray(w.value, dtype=float)
        raise RuntimeError("cvxpy is required for validated min_variance optimization")

    def _risk_parity(self, cov: np.ndarray, n: int) -> np.ndarray:
        """风险平价: 各子组合风险贡献相等."""
        from optimization.risk_budgeting import RiskBudgetingOptimizer

        budget = np.ones(n) / n
        w = RiskBudgetingOptimizer._erc_weights(cov, budget)
        return self._project_to_constraints(w, n)

    def _inverse_volatility(self, cov: np.ndarray, n: int) -> np.ndarray:
        volatility = np.sqrt(np.maximum(np.diag(cov), 1e-12))
        raw = 1.0 / volatility
        return self._project_to_constraints(raw / raw.sum(), n)

    def _shrink_covariance(self, cov: np.ndarray) -> np.ndarray:
        intensity = float(np.clip(self.covariance_shrinkage, 0.0, 1.0))
        diagonal = np.diag(np.diag(cov))
        shrunk = (1.0 - intensity) * cov + intensity * diagonal
        return (shrunk + shrunk.T) / 2.0

    def _hrp(self, cov: np.ndarray, n: int) -> np.ndarray:
        """Hierarchical Risk Parity with deterministic average linkage."""
        if n <= 1:
            return np.ones(n, dtype=float)
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform

        volatility = np.sqrt(np.maximum(np.diag(cov), 1e-12))
        corr = cov / np.outer(volatility, volatility)
        corr = np.clip((corr + corr.T) / 2.0, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)
        distance = np.sqrt(np.maximum((1.0 - corr) / 2.0, 0.0))
        order = leaves_list(
            linkage(squareform(distance, checks=False), method="average")
        ).tolist()

        weights = pd.Series(1.0, index=order, dtype=float)
        clusters = [order]
        while clusters:
            next_clusters = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue
                split = len(cluster) // 2
                left, right = cluster[:split], cluster[split:]
                left_var = self._cluster_variance(cov, left)
                right_var = self._cluster_variance(cov, right)
                total = left_var + right_var
                left_allocation = right_var / total if total > 0 else 0.5
                weights.loc[left] *= left_allocation
                weights.loc[right] *= 1.0 - left_allocation
                next_clusters.extend([left, right])
            clusters = next_clusters
        ordered = weights.reindex(range(n)).to_numpy(dtype=float)
        ordered = ordered / ordered.sum()
        return self._project_to_constraints(ordered, n)

    @staticmethod
    def _cluster_variance(cov: np.ndarray, indices: List[int]) -> float:
        sub_cov = cov[np.ix_(indices, indices)]
        inverse_variance = 1.0 / np.maximum(np.diag(sub_cov), 1e-12)
        weights = inverse_variance / inverse_variance.sum()
        return float(weights @ sub_cov @ weights)

    def _project_to_constraints(self, w: np.ndarray, n: int) -> np.ndarray:
        """将权重投影到 [min_weight, max_weight] 且 sum=1 的约束集.

        简单方法: clip 后归一化, 迭代几次直到收敛.
        """
        if n * self.min_weight > 1 + 1e-12 or n * self.max_weight < 1 - 1e-12:
            raise ValueError("infeasible meta weight bounds")
        values = np.asarray(w, dtype=float)
        if values.shape != (n,) or not np.isfinite(values).all():
            raise ValueError("invalid meta weights")
        low = float(np.min(values - self.max_weight))
        high = float(np.max(values - self.min_weight))
        for _ in range(100):
            shift = (low + high) / 2.0
            projected = np.clip(values - shift, self.min_weight, self.max_weight)
            if projected.sum() > 1.0:
                low = shift
            else:
                high = shift
        projected = np.clip(values - (low + high) / 2.0, self.min_weight, self.max_weight)
        if abs(float(projected.sum()) - 1.0) > 1e-8:
            raise RuntimeError("bounded simplex projection did not converge")
        return projected


class UnderlyingExposureController:
    """Project sleeve capital weights onto aggregate futures exposure limits.

    The meta optimizer allocates capital across sleeves.  This controller works
    one level lower: ``exposure_matrix @ sleeve_weights`` is the actual aggregate
    instrument target.  It only changes a target when that aggregate position
    violates a configured portfolio constraint, and it may de-lever but never
    increase the requested total capital scale.
    """

    _SUPPORTED = {
        "net_exposure",
        "leverage",
        "margin",
        "position_limit",
        "sector_exposure",
    }

    def __init__(
        self,
        constraint_specs: Sequence[Mapping],
        *,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        sector_map: Optional[Mapping[str, str]] = None,
        tolerance: float = 1e-7,
    ):
        self.constraint_specs = [
            dict(spec)
            for spec in constraint_specs
            if str(spec.get("type", "")) in self._SUPPORTED
        ]
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.sector_map = dict(sector_map or {})
        self.tolerance = float(tolerance)

    def apply(
        self,
        target_sleeve_weights: np.ndarray,
        exposure_matrix: np.ndarray,
        instruments: Sequence[str],
    ) -> Tuple[np.ndarray, dict]:
        """Return feasible effective sleeve weights and exposure diagnostics."""
        target = np.asarray(target_sleeve_weights, dtype=float).reshape(-1)
        matrix = np.asarray(exposure_matrix, dtype=float)
        instrument_names = [str(item) for item in instruments]
        self._validate_inputs(target, matrix, instrument_names)

        target_exposure = matrix @ target
        diagnostics = self.diagnostics(target_exposure, instrument_names)
        diagnostics["constraint_adjusted"] = False
        diagnostics["requested_capital_scale"] = float(target.sum())
        diagnostics["applied_capital_scale"] = float(target.sum())

        if not self.constraint_specs or diagnostics["feasible"]:
            return target.copy(), diagnostics
        if target.sum() <= self.tolerance:
            raise RuntimeError("zero target capital is infeasible under aggregate constraints")

        import cvxpy as cp

        n_sleeves = target.size
        effective = cp.Variable(n_sleeves)
        capital_scale = cp.sum(effective)
        aggregate = matrix @ effective
        requested_scale = float(target.sum())

        constraints = [
            effective >= 0.0,
            capital_scale >= 0.0,
            capital_scale <= requested_scale,
            effective >= self.min_weight * capital_scale,
            effective <= self.max_weight * capital_scale,
        ]
        self._append_cvx_constraints(cp, constraints, aggregate, instrument_names)

        # The first term preserves the desired sleeve mix.  The second strongly
        # prefers retaining the requested volatility scale when a mix change is
        # enough to restore feasibility.
        objective = cp.Minimize(
            cp.sum_squares(effective - target)
            + 25.0 * cp.square(capital_scale - requested_scale)
        )
        problem = cp.Problem(objective, constraints)
        solve_validated(problem, effective, ["CLARABEL", "OSQP"])

        projected = np.asarray(effective.value, dtype=float).reshape(-1)
        if projected.shape != target.shape or not np.isfinite(projected).all():
            raise RuntimeError("aggregate exposure projection returned invalid weights")
        projected[np.abs(projected) < self.tolerance] = 0.0

        applied_exposure = matrix @ projected
        diagnostics = self.diagnostics(applied_exposure, instrument_names)
        diagnostics["constraint_adjusted"] = True
        diagnostics["requested_capital_scale"] = requested_scale
        diagnostics["applied_capital_scale"] = float(projected.sum())
        if not diagnostics["feasible"]:
            raise RuntimeError(
                "aggregate exposure projection failed validation: "
                f"{diagnostics['violations']}"
            )
        return projected, diagnostics

    def diagnostics(self, exposure: np.ndarray, instruments: Sequence[str]) -> dict:
        values = np.asarray(exposure, dtype=float).reshape(-1)
        instrument_names = [str(item) for item in instruments]
        if values.size != len(instrument_names) or not np.isfinite(values).all():
            raise ValueError("aggregate exposure is misaligned or non-finite")

        sector_exposure: Dict[str, float] = {}
        for instrument, value in zip(instrument_names, values):
            sector = self.sector_map.get(instrument, "other")
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + float(value)

        violations: List[dict] = []
        for spec in self.constraint_specs:
            kind = str(spec["type"])
            if kind == "net_exposure":
                actual = float(values.sum())
                lower = float(spec.get("lower", -np.inf))
                upper = float(spec.get("upper", np.inf))
                if actual < lower - self.tolerance or actual > upper + self.tolerance:
                    violations.append({"type": kind, "actual": actual, "lower": lower, "upper": upper})
            elif kind in {"leverage", "margin"}:
                actual = float(np.abs(values).sum())
                limit = float(spec.get("limit", np.inf))
                if actual > limit + self.tolerance:
                    violations.append({"type": kind, "actual": actual, "limit": limit})
            elif kind == "position_limit":
                lower = float(spec.get("lower", -np.inf))
                upper = float(spec.get("upper", np.inf))
                below = np.flatnonzero(values < lower - self.tolerance)
                above = np.flatnonzero(values > upper + self.tolerance)
                if below.size or above.size:
                    offending = np.concatenate([below, above])
                    violations.append({
                        "type": kind,
                        "lower": lower,
                        "upper": upper,
                        "offending": {
                            instrument_names[i]: float(values[i]) for i in offending
                        },
                    })
            elif kind == "sector_exposure":
                limit = float(spec.get("limit", np.inf))
                offending = {
                    sector: value
                    for sector, value in sector_exposure.items()
                    if abs(value) > limit + self.tolerance
                }
                if offending:
                    violations.append({"type": kind, "limit": limit, "offending": offending})

        return {
            "feasible": not violations,
            "violations": violations,
            "net_exposure": float(values.sum()),
            "gross_exposure": float(np.abs(values).sum()),
            "max_abs_position": float(np.max(np.abs(values))) if values.size else 0.0,
            "sector_exposure": sector_exposure,
        }

    def _append_cvx_constraints(self, cp, constraints, aggregate, instruments) -> None:
        sector_indices: Dict[str, List[int]] = {}
        for index, instrument in enumerate(instruments):
            sector = self.sector_map.get(str(instrument), "other")
            sector_indices.setdefault(sector, []).append(index)

        for spec in self.constraint_specs:
            kind = str(spec["type"])
            if kind == "net_exposure":
                constraints.extend([
                    cp.sum(aggregate) >= float(spec.get("lower", -np.inf)),
                    cp.sum(aggregate) <= float(spec.get("upper", np.inf)),
                ])
            elif kind in {"leverage", "margin"}:
                constraints.append(cp.norm1(aggregate) <= float(spec.get("limit", np.inf)))
            elif kind == "position_limit":
                constraints.extend([
                    aggregate >= float(spec.get("lower", -np.inf)),
                    aggregate <= float(spec.get("upper", np.inf)),
                ])
            elif kind == "sector_exposure":
                limit = float(spec.get("limit", np.inf))
                for indices in sector_indices.values():
                    sector_net = cp.sum(aggregate[indices])
                    constraints.extend([sector_net >= -limit, sector_net <= limit])

    @staticmethod
    def _validate_inputs(target: np.ndarray, matrix: np.ndarray, instruments: Sequence[str]) -> None:
        if target.size == 0 or matrix.ndim != 2:
            raise ValueError("aggregate exposure projection requires non-empty 2D inputs")
        if matrix.shape != (len(instruments), target.size):
            raise ValueError("exposure matrix shape does not match instruments and sleeves")
        if not np.isfinite(target).all() or not np.isfinite(matrix).all():
            raise ValueError("aggregate exposure projection inputs contain NaN/Inf")
        if np.any(target < -1e-10):
            raise ValueError("effective sleeve capital weights must be non-negative")
