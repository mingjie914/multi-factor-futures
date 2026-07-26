"""Optional long-or-cash defensive trend/risk-allocation sleeve.

This module borrows the useful mechanics from the index documents without
changing the futures multi-factor strategy: multi-model averaging, T+1 timing,
explicit caps and risk-based allocation.  It is a standalone benchmark by
default and must earn promotion through locked out-of-sample evidence.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult
from backtest.metrics import compute_all_metrics, compute_split_metrics
from signals.selection import SectorForecastSelector


class DefensiveTrendRiskParity:
    name = "defensive_trend_risk_parity"
    role = "optional_defensive_sleeve"

    def __init__(
        self,
        *,
        lookbacks: Sequence[int] = (20, 60, 120),
        rebalance_freq: int = 5,
        volatility_window: int = 60,
        top_n_per_sector: int = 1,
        exit_buffer: int = 1,
        allocation: str = "inverse_volatility",
        covariance_shrinkage: float = 0.30,
        target_volatility: float = 0.10,
        asset_cap: float = 0.20,
        sector_cap: float = 0.35,
        turnover_cap: float = 0.50,
        annual_fee: float = 0.001,
        sector_map: Optional[Mapping[str, str]] = None,
    ):
        self.lookbacks = sorted({int(value) for value in lookbacks if int(value) > 0})
        if not self.lookbacks:
            raise ValueError("at least one positive trend lookback is required")
        self.rebalance_freq = max(int(rebalance_freq), 1)
        self.volatility_window = max(int(volatility_window), 20)
        self.allocation = str(allocation)
        self.covariance_shrinkage = float(covariance_shrinkage)
        self.target_volatility = float(target_volatility)
        self.asset_cap = float(asset_cap)
        self.sector_cap = float(sector_cap)
        self.turnover_cap = float(turnover_cap)
        self.annual_fee = float(annual_fee)
        if sector_map is None:
            from core.sectors import SECTOR_MAP

            sector_map = SECTOR_MAP
        self.sector_map = dict(sector_map)
        self.selector = SectorForecastSelector(
            mode="hysteresis_top_n",
            top_n_per_side=top_n_per_sector,
            exit_buffer=exit_buffer,
            sector_map=self.sector_map,
        )

    def run(self, prices: pd.DataFrame, *, cost_model=None) -> BacktestResult:
        clean = prices.sort_index().replace([np.inf, -np.inf], np.nan).ffill()
        clean = clean.dropna(axis=1, how="all")
        if len(clean) < max(self.lookbacks) + 2 or clean.shape[1] == 0:
            raise ValueError("insufficient price history for defensive sleeve")
        returns = clean.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        trend_models = [clean.pct_change(period) for period in self.lookbacks]

        dates = pd.DatetimeIndex(clean.index)
        assets = pd.Index(clean.columns.astype(str))
        clean.columns = assets
        returns.columns = assets
        trend_models = [model.set_axis(assets, axis=1) for model in trend_models]
        current = pd.Series(0.0, index=assets)
        pending: Optional[pd.Series] = None
        nav_values = np.ones(len(dates), dtype=float)
        return_values = np.zeros(len(dates), dtype=float)
        turnover_values = np.zeros(len(dates), dtype=float)
        cost_values = np.zeros(len(dates), dtype=float)
        decisions: list[tuple[pd.Timestamp, pd.Series]] = []
        failures: list[dict] = []
        self.selector.reset()

        warmup = max(max(self.lookbacks), self.volatility_window)
        for index, date in enumerate(dates):
            transaction_cost = 0.0
            if pending is not None:
                turnover_values[index] = float((pending - current).abs().sum())
                if cost_model is not None and turnover_values[index] > 0:
                    transaction_cost = float(
                        cost_model.estimate_cost(pending, current, date)
                    )
                    if not np.isfinite(transaction_cost) or transaction_cost < 0:
                        raise RuntimeError(
                            f"invalid defensive transaction cost at {date}: {transaction_cost}"
                        )
                current = pending
                pending = None

            # The first row is the initial valuation point, not a holding day.
            # Starting fees on the second row keeps returns, costs and NAV aligned.
            daily_return = float(current @ returns.loc[date]) if index > 0 else 0.0
            daily_fee = self.annual_fee / 252.0 if index > 0 else 0.0
            if index > 0 and cost_model is not None:
                estimator = getattr(cost_model, "estimate_holding_cost", None)
                if estimator is not None:
                    daily_fee += float(estimator(current, date))
            net_return = daily_return - transaction_cost - daily_fee
            return_values[index] = net_return
            cost_values[index] = transaction_cost + daily_fee
            if index > 0:
                nav_values[index] = nav_values[index - 1] * (1.0 + net_return)

            # Decision is made after date's close and becomes effective on the
            # next available trading date through ``pending``.
            if index >= warmup and (index - warmup) % self.rebalance_freq == 0:
                try:
                    forecast = pd.concat(
                        [model.loc[date] for model in trend_models], axis=1
                    ).mean(axis=1).clip(lower=0.0)
                    selected = self.selector.apply(forecast, date=date)
                    selected_assets = selected[selected > 0].index
                    history = returns.loc[:date].iloc[-self.volatility_window:]
                    target = self._allocate(history, selected_assets)
                    target = self._cap_turnover(target, current)
                    pending = target
                    decisions.append((date, target.copy()))
                except Exception as exc:
                    failures.append({
                        "date": str(pd.Timestamp(date)),
                        "stage": "defensive_allocation",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "action": "hold_previous_weights",
                    })

        nav = pd.Series(nav_values, index=dates, name=self.name)
        result_returns = pd.Series(return_values, index=dates, name="returns")
        turnover = pd.Series(turnover_values, index=dates, name="turnover")
        costs = pd.Series(cost_values, index=dates, name="transaction_cost")
        weights_history = (
            pd.DataFrame([weights for _, weights in decisions], index=[date for date, _ in decisions])
            if decisions else pd.DataFrame(columns=assets)
        )
        metrics = compute_all_metrics(nav, returns=result_returns)
        active_turnover = turnover[turnover > 0]
        metrics["avg_turnover"] = (
            float(active_turnover.mean()) if not active_turnover.empty else 0.0
        )
        metrics["total_transaction_cost"] = float(costs.sum())
        metrics["strategy_role"] = self.role
        return BacktestResult(
            nav=nav,
            weights_history=weights_history,
            signals_history=[],
            metrics=metrics,
            turnover=turnover,
            costs=costs,
            split_metrics=compute_split_metrics(nav, result_returns, train_ratio=0.6),
            failure_ledger=failures,
        )

    def _allocate(self, history: pd.DataFrame, selected_assets: pd.Index) -> pd.Series:
        assets = pd.Index(history.columns)
        target = pd.Series(0.0, index=assets)
        selected = pd.Index(selected_assets).intersection(assets)
        if len(selected) == 0:
            return target
        sample = history[selected].dropna(how="all").fillna(0.0)
        covariance = sample.cov().to_numpy(dtype=float) * 252.0
        covariance = self._finite_psd_covariance(covariance)

        if self.allocation == "inverse_volatility":
            raw = 1.0 / np.sqrt(np.maximum(np.diag(covariance), 1e-12))
            weights = raw / raw.sum()
        elif self.allocation == "correlation_adjusted_inverse_volatility":
            volatility = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
            corr = covariance / np.outer(volatility, volatility)
            average_corr = (corr.sum(axis=1) - 1.0) / max(len(selected) - 1, 1)
            multiplier = np.clip(1.0 - average_corr, 0.25, 1.75)
            raw = multiplier / volatility
            weights = raw / raw.sum()
        elif self.allocation in {"risk_parity", "erc"}:
            from optimization.risk_budgeting import RiskBudgetingOptimizer

            weights = RiskBudgetingOptimizer._erc_weights(
                covariance, np.ones(len(selected)) / len(selected)
            )
        elif self.allocation in {"shrinkage_min_variance", "hrp"}:
            from optimization.meta_optimizer import MetaOptimizer

            optimizer = MetaOptimizer(min_weight=0.0, max_weight=1.0)
            if self.allocation == "shrinkage_min_variance":
                covariance = optimizer._shrink_covariance(covariance)
                weights = optimizer._min_variance(covariance, len(selected))
            else:
                weights = optimizer._hrp(covariance, len(selected))
        else:
            raise ValueError(f"unsupported defensive allocation {self.allocation!r}")

        weights = self._apply_caps(pd.Series(weights, index=selected))
        predicted_volatility = float(
            np.sqrt(max(weights.to_numpy() @ covariance @ weights.to_numpy(), 0.0))
        )
        if self.target_volatility > 0 and predicted_volatility > 1e-12:
            weights *= min(1.0, self.target_volatility / predicted_volatility)
        target.loc[selected] = weights
        return target

    def _apply_caps(self, weights: pd.Series) -> pd.Series:
        capped = weights.clip(lower=0.0, upper=self.asset_cap)
        for _ in range(5):
            changed = False
            for sector in {self.sector_map.get(str(asset), "other") for asset in capped.index}:
                members = [
                    asset for asset in capped.index
                    if self.sector_map.get(str(asset), "other") == sector
                ]
                total = float(capped.loc[members].sum())
                if total > self.sector_cap and total > 0:
                    capped.loc[members] *= self.sector_cap / total
                    changed = True
            if not changed:
                break
        return capped

    def _cap_turnover(self, target: pd.Series, current: pd.Series) -> pd.Series:
        target = target.reindex(current.index).fillna(0.0)
        change = target - current
        turnover = float(change.abs().sum())
        if self.turnover_cap > 0 and turnover > self.turnover_cap:
            target = current + change * (self.turnover_cap / turnover)
        return target.clip(lower=0.0)

    def _finite_psd_covariance(self, covariance: np.ndarray) -> np.ndarray:
        matrix = np.asarray(covariance, dtype=float)
        diagonal = np.diag(matrix) if matrix.ndim == 2 else np.array([])
        fallback = float(np.nanmedian(diagonal[np.isfinite(diagonal)])) if diagonal.size else 1e-4
        if not np.isfinite(fallback) or fallback <= 0:
            fallback = 1e-4
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        matrix = (matrix + matrix.T) / 2.0
        values, vectors = np.linalg.eigh(matrix)
        values = np.maximum(values, max(fallback * 1e-6, 1e-12))
        return (vectors * values) @ vectors.T
