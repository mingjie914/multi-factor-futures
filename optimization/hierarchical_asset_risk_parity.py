"""Three-level signal-scaled risk-parity allocation for futures portfolios."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import List, Optional

import numpy as np
import pandas as pd

from core.interfaces import Constraint, CostModel, Optimizer, RiskModel
from core.registry import register
from core.sectors import asset_class_for, sector_for
from core.types import Date, ExpectedReturns, Universe, WeightVector
from optimization.risk_budgeting import RiskBudgetingOptimizer


@register("optimizer", "hierarchical_asset_risk_parity")
class HierarchicalAssetRiskParityOptimizer(Optimizer):
    """Allocate signed forecasts through instrument, sector, and asset layers.

    Relative instrument weights use the forecast divided by annual volatility
    exactly once. Commodity sector sleeves are then risk-budgeted into one
    commodity sleeve, followed by risk budgeting across stock, bond, and
    commodity sleeves. The final relative portfolio is scaled to an ex-ante
    volatility target and only de-risked by hard constraints.
    """

    allocation_role = "three_layer_futures_risk_budgeting"
    deployment_status = "formal_default"

    def __init__(
        self,
        target_volatility: float = 0.10,
        max_leverage: float = 2.0,
        covariance_shrinkage: float = 0.30,
        periods_per_year: float = 252.0,
        volatility_floor: float = 0.01,
        asset_class_budgets: Optional[Mapping[str, float]] = None,
        commodity_sector_budgets: Optional[Mapping[str, float]] = None,
    ):
        if target_volatility < 0:
            raise ValueError("target_volatility must be non-negative")
        if max_leverage <= 0:
            raise ValueError("max_leverage must be positive")
        if not 0.0 <= covariance_shrinkage <= 1.0:
            raise ValueError("covariance_shrinkage must be in [0, 1]")
        if periods_per_year <= 0 or volatility_floor <= 0:
            raise ValueError("annualization and volatility floor must be positive")
        self.target_volatility = float(target_volatility)
        self.max_leverage = float(max_leverage)
        self.covariance_shrinkage = float(covariance_shrinkage)
        self.periods_per_year = float(periods_per_year)
        self.volatility_floor = float(volatility_floor)
        self.asset_class_budgets = dict(asset_class_budgets or {})
        self.commodity_sector_budgets = dict(commodity_sector_budgets or {})
        self.last_diagnostics: dict = {}

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
        del cost_model, realized_vol
        names = pd.Index(universe)
        if names.empty:
            return pd.Series(dtype=float)

        forecasts = (
            pd.Series(expected_returns, dtype=float)
            .reindex(names)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        if float(forecasts.abs().sum()) <= 1e-12:
            self.last_diagnostics = self._empty_diagnostics("no_active_signal")
            return pd.Series(0.0, index=names)

        annual_covariance = self._annual_covariance(risk_model, date, names)
        annual_volatility = np.sqrt(np.maximum(np.diag(annual_covariance), 0.0))
        annual_volatility = np.maximum(annual_volatility, self.volatility_floor)

        leaf_vectors = self._build_leaf_vectors(
            forecasts.to_numpy(dtype=float), annual_volatility, names
        )
        class_vectors, sector_allocations = self._build_class_vectors(
            leaf_vectors, annual_covariance
        )
        if not class_vectors:
            self.last_diagnostics = self._empty_diagnostics("no_active_leaf")
            return pd.Series(0.0, index=names)

        class_names = list(class_vectors)
        class_matrix = np.column_stack([class_vectors[key] for key in class_names])
        class_covariance = self._sleeve_covariance(
            class_matrix, annual_covariance
        )
        class_allocations = self._risk_budget_weights(
            class_covariance, class_names, self.asset_class_budgets
        )
        raw = class_matrix @ class_allocations

        raw_gross = float(np.abs(raw).sum())
        if raw_gross <= 1e-12:
            self.last_diagnostics = self._empty_diagnostics("zero_raw_exposure")
            return pd.Series(0.0, index=names)
        raw /= raw_gross

        predicted_volatility = self._portfolio_volatility(raw, annual_covariance)
        scale = 1.0
        if self.target_volatility > 0 and predicted_volatility > 1e-12:
            scale = min(
                self.max_leverage,
                self.target_volatility / predicted_volatility,
            )
        target = pd.Series(raw * scale, index=names, dtype=float)
        target = self._apply_hard_constraints(
            target,
            current_weights,
            constraints,
            current_drawdown=current_drawdown,
        )
        final_values = target.to_numpy(dtype=float)
        final_volatility = self._portfolio_volatility(
            final_values, annual_covariance
        )
        self.last_diagnostics = {
            "reason": "allocated",
            "predicted_volatility_before_scale": predicted_volatility,
            "predicted_volatility": final_volatility,
            "target_volatility": self.target_volatility,
            "gross_exposure": float(target.abs().sum()),
            "net_exposure": float(target.sum()),
            "class_allocations": dict(zip(class_names, class_allocations)),
            "commodity_sector_allocations": sector_allocations,
        }
        return target.where(target.abs() > 1e-10, 0.0)

    def _build_leaf_vectors(
        self,
        forecasts: np.ndarray,
        annual_volatility: np.ndarray,
        names: pd.Index,
    ) -> dict[tuple[str, str], np.ndarray]:
        groups: dict[tuple[str, str], list[int]] = {}
        for index, instrument in enumerate(names):
            asset_class = asset_class_for(str(instrument))
            leaf = sector_for(str(instrument)) if asset_class == "commodity" else asset_class
            groups.setdefault((asset_class, leaf), []).append(index)

        result: dict[tuple[str, str], np.ndarray] = {}
        for key, indices in groups.items():
            local = forecasts[indices] / annual_volatility[indices]
            local_gross = float(np.abs(local).sum())
            if local_gross <= 1e-12:
                continue
            vector = np.zeros(len(names), dtype=float)
            vector[indices] = local / local_gross
            result[key] = vector
        return result

    def _build_class_vectors(
        self,
        leaf_vectors: Mapping[tuple[str, str], np.ndarray],
        annual_covariance: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        class_vectors: dict[str, np.ndarray] = {}
        commodity = {
            leaf: vector
            for (asset_class, leaf), vector in leaf_vectors.items()
            if asset_class == "commodity"
        }
        commodity_allocations: dict[str, float] = {}
        if commodity:
            sector_names = list(commodity)
            sector_matrix = np.column_stack(
                [commodity[key] for key in sector_names]
            )
            sector_covariance = self._sleeve_covariance(
                sector_matrix, annual_covariance
            )
            allocations = self._risk_budget_weights(
                sector_covariance,
                sector_names,
                self.commodity_sector_budgets,
            )
            class_vectors["commodity"] = sector_matrix @ allocations
            commodity_allocations = dict(zip(sector_names, allocations))

        for asset_class in ("stock", "bond"):
            vector = leaf_vectors.get((asset_class, asset_class))
            if vector is not None:
                class_vectors[asset_class] = vector
        return class_vectors, commodity_allocations

    def _annual_covariance(
        self, risk_model: RiskModel, date: Date, names: pd.Index
    ) -> np.ndarray:
        frame = risk_model.covariance(date, names).reindex(
            index=names, columns=names
        )
        matrix = frame.to_numpy(dtype=float)
        if matrix.shape != (len(names), len(names)) or not np.isfinite(matrix).all():
            raise ValueError("risk covariance is misaligned or non-finite")
        matrix = self._ensure_psd(matrix * self.periods_per_year)
        diagonal = np.maximum(np.diag(matrix), self.volatility_floor**2)
        np.fill_diagonal(matrix, diagonal)
        return self._ensure_psd(matrix)

    def _sleeve_covariance(
        self, sleeve_matrix: np.ndarray, annual_covariance: np.ndarray
    ) -> np.ndarray:
        covariance = sleeve_matrix.T @ annual_covariance @ sleeve_matrix
        diagonal = np.diag(np.diag(covariance))
        covariance = (
            (1.0 - self.covariance_shrinkage) * covariance
            + self.covariance_shrinkage * diagonal
        )
        return self._ensure_psd(covariance)

    @staticmethod
    def _risk_budget_weights(
        covariance: np.ndarray,
        labels: Sequence[str],
        configured_budgets: Mapping[str, float],
    ) -> np.ndarray:
        if len(labels) == 1:
            return np.ones(1, dtype=float)
        budget = np.asarray(
            [max(float(configured_budgets.get(label, 1.0)), 0.0) for label in labels],
            dtype=float,
        )
        if float(budget.sum()) <= 0:
            raise ValueError("active risk budgets must contain a positive value")
        budget /= budget.sum()
        return RiskBudgetingOptimizer._erc_weights(covariance, budget)

    def _apply_hard_constraints(
        self,
        target: pd.Series,
        current_weights: WeightVector,
        constraints: Sequence[Constraint],
        *,
        current_drawdown: float,
    ) -> pd.Series:
        result = self._project_hard_limits(
            target,
            constraints,
            current_drawdown=current_drawdown,
        )
        turnover_limit = next(
            (
                float(getattr(constraint, "limit", 0.0))
                for constraint in constraints
                if getattr(constraint, "name", "") == "turnover"
            ),
            None,
        )
        if turnover_limit is not None and turnover_limit >= 0:
            previous = (
                pd.Series(current_weights, dtype=float)
                .reindex(result.index)
                .fillna(0.0)
            )
            change = result - previous
            turnover = float(change.abs().sum())
            if turnover > turnover_limit and turnover > 0:
                result = previous + change * (turnover_limit / turnover)

        # A stale position can already violate today's limits. Turnover is a
        # soft transition limit in that conflict: final risk caps take
        # precedence and this projection only removes exposure.
        return self._project_hard_limits(
            result,
            constraints,
            current_drawdown=current_drawdown,
        )

    def _project_hard_limits(
        self,
        weights: pd.Series,
        constraints: Sequence[Constraint],
        *,
        current_drawdown: float,
    ) -> pd.Series:
        """Apply limits by clipping or proportional deleveraging only."""

        result = weights.copy()
        for constraint in constraints:
            kind = getattr(constraint, "name", "")
            if kind == "position_limit":
                result = result.clip(
                    lower=float(getattr(constraint, "lower", -np.inf)),
                    upper=float(getattr(constraint, "upper", np.inf)),
                )
            elif kind == "sector_exposure":
                limit = float(getattr(constraint, "limit", 0.0))
                if limit > 0:
                    result = self._cap_sector_net_exposure(result, limit)

        gross = max(float(result.abs().sum()), 1e-12)
        global_scale = min(1.0, self.max_leverage / gross)
        for constraint in constraints:
            kind = getattr(constraint, "name", "")
            if kind == "leverage":
                limit = float(getattr(constraint, "limit", self.max_leverage))
                global_scale = min(global_scale, limit / gross)
            elif kind == "net_exposure":
                net = float(result.sum())
                lower = float(getattr(constraint, "lower", -np.inf))
                upper = float(getattr(constraint, "upper", np.inf))
                if net > upper and net > 0:
                    global_scale = min(global_scale, upper / net)
                elif net < lower and net < 0:
                    global_scale = min(global_scale, lower / net)
            elif kind == "drawdown_control":
                warning = float(getattr(constraint, "warning_dd", -0.05))
                critical = float(getattr(constraint, "critical_dd", -0.10))
                limit = float(getattr(constraint, "leverage_limit", self.max_leverage))
                if current_drawdown <= critical:
                    limit = 0.1
                elif current_drawdown <= warning:
                    limit *= 0.5
                global_scale = min(global_scale, limit / gross)
        result *= max(global_scale, 0.0)
        return result

    @staticmethod
    def _cap_sector_net_exposure(weights: pd.Series, limit: float) -> pd.Series:
        result = weights.copy()
        groups: dict[str, list[str]] = {}
        for instrument in result.index:
            groups.setdefault(sector_for(str(instrument)), []).append(instrument)
        for instruments in groups.values():
            net = float(result.loc[instruments].sum())
            if abs(net) > limit:
                result.loc[instruments] *= limit / abs(net)
        return result

    @staticmethod
    def _portfolio_volatility(weights: np.ndarray, covariance: np.ndarray) -> float:
        return float(np.sqrt(max(float(weights @ covariance @ weights), 0.0)))

    @staticmethod
    def _ensure_psd(matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=float)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        values = (values + values.T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(values)
        eigenvalues = np.maximum(eigenvalues, 1e-12)
        return (eigenvectors * eigenvalues) @ eigenvectors.T

    def _empty_diagnostics(self, reason: str) -> dict:
        return {
            "reason": reason,
            "predicted_volatility": 0.0,
            "target_volatility": self.target_volatility,
            "gross_exposure": 0.0,
            "net_exposure": 0.0,
            "class_allocations": {},
            "commodity_sector_allocations": {},
        }


__all__ = ["HierarchicalAssetRiskParityOptimizer"]
