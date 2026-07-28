"""Standalone long-short Supertrend sleeve with volatility-aware risk caps."""
from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult
from backtest.metrics import compute_all_metrics, compute_split_metrics
from factors.user.supertrend import _coherent_ohlc_columns, _supertrend_components


class SupertrendATRRiskStrategy:
    name = "supertrend_atr_risk"
    role = "standalone_rule_sleeve"

    def __init__(
        self,
        *,
        rebalance_freq: int = 5,
        risk_window: int = 60,
        covariance_shrinkage: float = 0.30,
        target_volatility: float = 0.12,
        asset_vol_budget: float = 0.025,
        sector_vol_budget: float = 0.06,
        hard_asset_cap: float = 1.0,
        gross_cap: float = 2.0,
        net_cap: float = 0.50,
        turnover_cap: float = 0.50,
        rebalance_on_flip: bool = True,
        sector_map: Optional[Mapping[str, str]] = None,
    ):
        self.rebalance_freq = max(int(rebalance_freq), 1)
        self.risk_window = max(int(risk_window), 20)
        self.covariance_shrinkage = float(np.clip(covariance_shrinkage, 0.0, 1.0))
        self.target_volatility = float(target_volatility)
        self.asset_vol_budget = float(asset_vol_budget)
        self.sector_vol_budget = float(sector_vol_budget)
        self.hard_asset_cap = float(hard_asset_cap)
        self.gross_cap = float(gross_cap)
        self.net_cap = float(net_cap)
        self.turnover_cap = float(turnover_cap)
        self.rebalance_on_flip = bool(rebalance_on_flip)
        if sector_map is None:
            from core.sectors import SECTOR_MAP

            sector_map = SECTOR_MAP
        self.sector_map = dict(sector_map)

    def run(
        self,
        high: pd.DataFrame,
        low: pd.DataFrame,
        close: pd.DataFrame,
        *,
        cost_model=None,
    ) -> BacktestResult:
        high, low, close = self._align_prices(high, low, close)
        if close.shape[1] == 0 or len(close) < self.risk_window + 2:
            raise ValueError("insufficient OHLC history for Supertrend sleeve")

        direction, _, atr, _ = _supertrend_components(high, low, close)
        raw_returns = close.pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan
        )
        returns = raw_returns.fillna(0.0)
        atr_rate = atr / close.replace(0, np.nan)
        decision_direction = direction.ffill(limit=self.rebalance_freq)
        annual_atr_panel = (
            atr_rate.ffill(limit=self.rebalance_freq) * np.sqrt(252.0)
        )

        dates = pd.DatetimeIndex(close.index)
        assets = pd.Index(close.columns.astype(str))
        current = pd.Series(0.0, index=assets)
        pending: Optional[pd.Series] = None
        previous_decision_state = pd.Series(np.nan, index=assets)
        nav_values = np.ones(len(dates), dtype=float)
        return_values = np.zeros(len(dates), dtype=float)
        turnover_values = np.zeros(len(dates), dtype=float)
        cost_values = np.zeros(len(dates), dtype=float)
        trade_cost_values = np.zeros(len(dates), dtype=float)
        holding_cost_values = np.zeros(len(dates), dtype=float)
        decisions: list[tuple[pd.Timestamp, pd.Series]] = []
        failures: list[dict] = []
        diagnostics: list[dict] = []

        for index, date in enumerate(dates):
            transaction_cost = 0.0
            if pending is not None:
                turnover = float((pending - current).abs().sum())
                turnover_values[index] = turnover
                if cost_model is not None and turnover > 0:
                    transaction_cost = float(
                        cost_model.estimate_cost(pending, current, date)
                    )
                    if not np.isfinite(transaction_cost) or transaction_cost < 0:
                        raise RuntimeError(
                            f"invalid Supertrend transaction cost at {date}"
                        )
                current = pending
                pending = None

            daily_return = float(current @ returns.loc[date]) if index > 0 else 0.0
            holding_cost = 0.0
            if index > 0 and cost_model is not None:
                estimator = getattr(cost_model, "estimate_holding_cost", None)
                if estimator is not None:
                    holding_cost = float(estimator(current, date))
                    if not np.isfinite(holding_cost) or holding_cost < 0:
                        raise RuntimeError(
                            f"invalid Supertrend holding cost at {date}"
                        )
            net_return = daily_return - transaction_cost - holding_cost
            return_values[index] = net_return
            cost_values[index] = transaction_cost + holding_cost
            trade_cost_values[index] = transaction_cost
            holding_cost_values[index] = holding_cost
            if index > 0:
                nav_values[index] = nav_values[index - 1] * (1.0 + net_return)

            if index < self.risk_window:
                continue
            state = decision_direction.loc[date].reindex(assets)
            changed = (state != previous_decision_state) & state.notna()
            state_changed = bool(changed.any())
            scheduled = (index - self.risk_window) % self.rebalance_freq == 0
            if not (scheduled or (self.rebalance_on_flip and state_changed)):
                continue

            try:
                history = raw_returns.iloc[
                    max(0, index - self.risk_window + 1): index + 1
                ]
                target, _ = self._allocate(
                    state,
                    annual_atr_panel.loc[date].reindex(assets) / np.sqrt(252.0),
                    history,
                )
                if self.rebalance_on_flip and state_changed and not scheduled:
                    # A single contract flip should not resize every unchanged
                    # contract. Full covariance/risk resizing remains scheduled.
                    event_target = current.copy()
                    event_target.loc[changed] = target.loc[changed]
                    target = event_target
                target = self._cap_turnover(target, current)
                # Volatility limits are hard constraints.  If today's ATR or
                # covariance rises sharply, reducing stale risk takes priority
                # over the secondary turnover limit.
                annual_atr = annual_atr_panel.loc[date].reindex(assets)
                target = self._apply_risk_limits(target, annual_atr, history)
                effective_turnover = float((target - current).abs().sum())
                diag = self._diagnostics(target, annual_atr, history)
                diag.update({
                    "effective_turnover": effective_turnover,
                    "turnover_risk_override": bool(
                        self.turnover_cap > 0
                        and effective_turnover > self.turnover_cap + 1e-10
                    ),
                })
                pending = target
                previous_decision_state = state.copy()
                decisions.append((date, target.copy()))
                diagnostics.append({"date": str(date), **diag})
            except Exception as exc:
                failures.append({
                    "date": str(date),
                    "stage": "supertrend_atr_allocation",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "action": "hold_previous_weights",
                })

        nav = pd.Series(nav_values, index=dates, name=self.name)
        result_returns = pd.Series(return_values, index=dates, name="returns")
        turnover = pd.Series(turnover_values, index=dates, name="turnover")
        costs = pd.Series(cost_values, index=dates, name="transaction_cost")
        weights_history = (
            pd.DataFrame(
                [weights for _, weights in decisions],
                index=[date for date, _ in decisions],
            )
            if decisions
            else pd.DataFrame(columns=assets)
        )
        metrics = compute_all_metrics(nav, returns=result_returns)
        active_turnover = turnover[turnover > 0]
        metrics.update({
            "avg_turnover": (
                float(active_turnover.mean()) if not active_turnover.empty else 0.0
            ),
            "total_transaction_cost": float(costs.sum()),
            "total_trade_cost": float(trade_cost_values.sum()),
            "total_holding_cost": float(holding_cost_values.sum()),
            "strategy_role": self.role,
            "avg_gross_exposure": (
                float(weights_history.abs().sum(axis=1).mean())
                if not weights_history.empty else 0.0
            ),
            "max_gross_exposure": (
                float(weights_history.abs().sum(axis=1).max())
                if not weights_history.empty else 0.0
            ),
            "max_abs_net_exposure": (
                float(weights_history.sum(axis=1).abs().max())
                if not weights_history.empty else 0.0
            ),
            "max_abs_asset_weight": (
                float(weights_history.abs().max().max())
                if not weights_history.empty else 0.0
            ),
            "max_asset_vol_proxy": max(
                (float(item["max_asset_vol_proxy"]) for item in diagnostics),
                default=0.0,
            ),
            "max_sector_standalone_vol": max(
                (float(item["max_sector_standalone_vol"]) for item in diagnostics),
                default=0.0,
            ),
            "max_turnover": float(turnover.max()) if len(turnover) else 0.0,
            "risk_override_count": int(sum(
                bool(item.get("turnover_risk_override", False))
                for item in diagnostics
            )),
            "decision_count": len(decisions),
        })
        return BacktestResult(
            nav=nav,
            weights_history=weights_history,
            signals_history=[],
            metrics=metrics,
            turnover=turnover,
            costs=costs,
            split_metrics=compute_split_metrics(
                nav, result_returns, train_ratio=0.75
            ),
            failure_ledger=failures,
        )

    @staticmethod
    def _align_prices(high, low, close):
        common_index = high.index.intersection(low.index).intersection(close.index)
        common_columns = high.columns.intersection(low.columns).intersection(close.columns)
        high = high.reindex(index=common_index, columns=common_columns).astype(float)
        low = low.reindex(index=common_index, columns=common_columns).astype(float)
        close = close.reindex(index=common_index, columns=common_columns).astype(float)
        valid_columns = close.columns[close.notna().any()]
        high, low, close = (
            high[valid_columns], low[valid_columns], close[valid_columns]
        )
        coherent = _coherent_ohlc_columns(high, low, close)
        if not coherent.all():
            bad = ", ".join(str(item) for item in coherent.index[~coherent])
            raise ValueError(
                "OHLC price scales are inconsistent; fetch high/low/close in "
                f"one continuous-contract request. Offending instruments: {bad}"
            )
        return high, low, close

    def _allocate(
        self,
        state: pd.Series,
        atr_rate: pd.Series,
        history: pd.DataFrame,
    ) -> tuple[pd.Series, dict]:
        assets = pd.Index(history.columns.astype(str))
        target = pd.Series(0.0, index=assets)
        annual_atr = atr_rate.reindex(assets) * np.sqrt(252.0)
        valid_observations = history.reindex(columns=assets).notna().sum()
        valid = (
            state.reindex(assets).isin([-1.0, 1.0])
            & annual_atr.replace([np.inf, -np.inf], np.nan).notna()
            & (annual_atr > 1e-6)
            & (valid_observations >= max(20, self.risk_window // 2))
        )
        selected = assets[valid]
        if len(selected) == 0:
            return target, self._diagnostics(target, annual_atr, history)

        sample = history[selected].replace([np.inf, -np.inf], np.nan)
        covariance = self._covariance(sample)
        raw = state.reindex(selected).astype(float) / annual_atr.reindex(selected)
        raw /= max(float(raw.abs().sum()), 1e-12)
        predicted_vol = float(
            np.sqrt(max(raw.to_numpy() @ covariance @ raw.to_numpy(), 0.0))
        )
        if self.target_volatility > 0 and predicted_vol > 1e-12:
            raw *= self.target_volatility / predicted_vol

        dynamic_cap = (
            self.asset_vol_budget / annual_atr.reindex(selected)
        ).clip(upper=self.hard_asset_cap)
        raw = raw.clip(lower=-dynamic_cap, upper=dynamic_cap)
        raw = self._apply_sector_vol_caps(raw, covariance, selected)

        gross = float(raw.abs().sum())
        if self.gross_cap > 0 and gross > self.gross_cap:
            raw *= self.gross_cap / gross
        net = float(raw.sum())
        if self.net_cap > 0 and abs(net) > self.net_cap:
            raw *= self.net_cap / abs(net)
        target.loc[selected] = raw
        return target, self._diagnostics(target, annual_atr, history)

    def _apply_risk_limits(
        self,
        weights: pd.Series,
        annual_atr: pd.Series,
        history: pd.DataFrame,
    ) -> pd.Series:
        """Project a proposed trade onto the hard, volatility-aware limits."""
        assets = pd.Index(history.columns.astype(str))
        result = weights.reindex(assets).fillna(0.0).astype(float)
        risk = annual_atr.reindex(assets).replace([np.inf, -np.inf], np.nan)
        valid = risk.notna() & (risk > 1e-6)
        result.loc[~valid] = 0.0

        caps = (self.asset_vol_budget / risk.loc[valid]).clip(
            upper=self.hard_asset_cap
        )
        result.loc[valid] = result.loc[valid].clip(lower=-caps, upper=caps)

        covariance = self._covariance(history.reindex(columns=assets))
        result = self._apply_sector_vol_caps(result, covariance, assets)
        gross = float(result.abs().sum())
        if self.gross_cap > 0 and gross > self.gross_cap:
            result *= self.gross_cap / gross
        net = float(result.sum())
        if self.net_cap > 0 and abs(net) > self.net_cap:
            result *= self.net_cap / abs(net)
        return result

    def _apply_sector_vol_caps(
        self,
        weights: pd.Series,
        covariance: np.ndarray,
        selected: pd.Index,
    ) -> pd.Series:
        result = weights.copy()
        if self.sector_vol_budget <= 0:
            return result
        position = {asset: index for index, asset in enumerate(selected)}
        sectors = sorted({self.sector_map.get(str(asset), "other") for asset in selected})
        for sector in sectors:
            members = [
                asset for asset in selected
                if self.sector_map.get(str(asset), "other") == sector
            ]
            indices = [position[asset] for asset in members]
            sector_weights = result.reindex(members).to_numpy(dtype=float)
            sector_cov = covariance[np.ix_(indices, indices)]
            sector_vol = float(
                np.sqrt(max(sector_weights @ sector_cov @ sector_weights, 0.0))
            )
            if sector_vol > self.sector_vol_budget:
                result.loc[members] *= self.sector_vol_budget / sector_vol
        return result

    def _cap_turnover(self, target: pd.Series, current: pd.Series) -> pd.Series:
        target = target.reindex(current.index).fillna(0.0)
        change = target - current
        turnover = float(change.abs().sum())
        if self.turnover_cap > 0 and turnover > self.turnover_cap:
            target = current + change * (self.turnover_cap / turnover)
        return target

    def _shrink_covariance(self, covariance: np.ndarray) -> np.ndarray:
        matrix = np.asarray(covariance, dtype=float)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        matrix = (matrix + matrix.T) / 2.0
        diagonal = np.maximum(np.diag(matrix), 1e-8)
        target = np.diag(diagonal)
        matrix = (
            (1.0 - self.covariance_shrinkage) * matrix
            + self.covariance_shrinkage * target
        )
        values, vectors = np.linalg.eigh(matrix)
        values = np.maximum(values, 1e-10)
        return (vectors * values) @ vectors.T

    def _covariance(self, history: pd.DataFrame) -> np.ndarray:
        sample = history.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        covariance = sample.cov().to_numpy(dtype=float) * 252.0
        return self._shrink_covariance(covariance)

    def _diagnostics(
        self,
        weights: pd.Series,
        annual_atr: pd.Series,
        history: pd.DataFrame,
    ) -> dict:
        asset_proxy = weights.abs() * annual_atr.reindex(weights.index).fillna(0.0)
        covariance = self._covariance(history.reindex(columns=weights.index))
        position = {asset: index for index, asset in enumerate(weights.index)}
        sector_vols = []
        for sector in sorted(set(self.sector_map.values()) | {"other"}):
            members = [
                asset for asset in weights.index
                if self.sector_map.get(str(asset), "other") == sector
            ]
            if not members:
                continue
            indices = [position[asset] for asset in members]
            values = weights.reindex(members).to_numpy(dtype=float)
            sector_cov = covariance[np.ix_(indices, indices)]
            sector_vols.append(
                float(np.sqrt(max(values @ sector_cov @ values, 0.0)))
            )
        return {
            "gross": float(weights.abs().sum()),
            "net": float(weights.sum()),
            "max_asset_vol_proxy": float(asset_proxy.max()) if len(asset_proxy) else 0.0,
            "max_sector_standalone_vol": max(sector_vols, default=0.0),
        }


__all__ = ["SupertrendATRRiskStrategy"]
