"""Auditable close-marked return accounting for futures research.

This module deliberately models a *research* portfolio, not a broker account.
Assets are continuous futures series marked from close to close.  Target
notional weights become effective on the next bar, and exposures drift between
decisions so that the calculation does not assume free daily rebalancing.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd


LEDGER_SCHEMA_VERSION = 1
ACTIVE_WEIGHT_TOLERANCE = 1e-12


class ResearchLedgerError(RuntimeError):
    """Raised when close-marked research returns cannot be accounted for."""


class MissingActiveReturnError(ResearchLedgerError):
    """Raised when an active exposure has no return for the marked bar."""


@dataclass(frozen=True)
class CloseMarkedStep:
    asset_returns: pd.Series
    effective_weights: pd.Series
    contributions: pd.Series
    gross_return: float
    trade_cost: float
    holding_cost: float
    net_return: float
    end_weights: pd.Series


def align_transition_weights(
    target: pd.Series,
    current: Optional[pd.Series],
) -> tuple[pd.Series, pd.Series]:
    """Align both sides of a trade on their union, preserving exit trades."""
    target = pd.Series(target, dtype=float)
    current = (
        pd.Series(dtype=float)
        if current is None
        else pd.Series(current, dtype=float)
    )
    instruments = current.index.union(target.index, sort=False)
    return (
        target.reindex(instruments).fillna(0.0),
        current.reindex(instruments).fillna(0.0),
    )


def close_marked_step(
    weights: pd.Series,
    asset_returns: pd.Series,
    *,
    trade_cost: float = 0.0,
    holding_cost: float = 0.0,
) -> CloseMarkedStep:
    """Account for one close-to-close bar and drift its notional exposures."""
    weights = pd.Series(weights, dtype=float)
    returns = pd.Series(asset_returns, dtype=float).reindex(weights.index)

    if not np.isfinite(weights.to_numpy(dtype=float)).all():
        raise ResearchLedgerError("effective weights contain NaN or infinity")
    active = weights.abs().gt(ACTIVE_WEIGHT_TOLERANCE)
    missing = active & ~np.isfinite(returns.to_numpy(dtype=float))
    if bool(missing.any()):
        names = ", ".join(str(value) for value in weights.index[missing][:5])
        raise MissingActiveReturnError(
            f"active instruments have no close return: {names}"
        )

    returns = returns.fillna(0.0)
    if bool((returns.loc[active] <= -1.0).any()):
        names = ", ".join(
            str(value) for value in weights.index[active & returns.le(-1.0)][:5]
        )
        raise ResearchLedgerError(
            f"active instruments have invalid close returns <= -100%: {names}"
        )

    trade_cost = float(trade_cost)
    holding_cost = float(holding_cost)
    if (
        not np.isfinite(trade_cost)
        or not np.isfinite(holding_cost)
        or trade_cost < 0.0
        or holding_cost < 0.0
    ):
        raise ResearchLedgerError("research costs must be finite and non-negative")

    contributions = weights * returns
    gross_return = float(contributions.sum())
    net_return = gross_return - trade_cost - holding_cost
    if not np.isfinite(net_return) or 1.0 + net_return <= 0.0:
        raise ResearchLedgerError(
            f"research NAV is exhausted by bar return/costs: {net_return}"
        )

    # Fixed synthetic units change in notional value with price.  Dividing by
    # end NAV makes tomorrow's exposure explicit and avoids an uncharged daily
    # reset back to the last target weights.
    end_weights = weights * (1.0 + returns) / (1.0 + net_return)
    return CloseMarkedStep(
        asset_returns=returns,
        effective_weights=weights,
        contributions=contributions,
        gross_return=gross_return,
        trade_cost=trade_cost,
        holding_cost=holding_cost,
        net_return=net_return,
        end_weights=end_weights,
    )


@dataclass
class ResearchReturnLedger:
    daily: pd.DataFrame
    asset_returns: pd.DataFrame
    effective_weights: pd.DataFrame
    contributions: pd.DataFrame
    metadata: Mapping[str, object]

    @classmethod
    def empty(cls) -> "ResearchReturnLedger":
        return cls(
            daily=pd.DataFrame(),
            asset_returns=pd.DataFrame(),
            effective_weights=pd.DataFrame(),
            contributions=pd.DataFrame(),
            metadata=default_research_ledger_metadata(),
        )

    def validate(self, *, atol: float = 1e-12) -> None:
        if self.daily.empty:
            return
        required = {
            "nav_before",
            "nav_after",
            "gross_return",
            "trade_cost",
            "holding_cost",
            "net_return",
            "decision_turnover",
            "gross_exposure",
            "net_exposure",
            "active_instruments",
        }
        missing = required - set(self.daily.columns)
        if missing:
            raise ResearchLedgerError(
                "research ledger is missing daily columns: " + ", ".join(sorted(missing))
            )
        for frame_name, frame in {
            "asset_returns": self.asset_returns,
            "effective_weights": self.effective_weights,
            "contributions": self.contributions,
        }.items():
            if not frame.index.equals(self.daily.index):
                raise ResearchLedgerError(f"{frame_name} index does not match daily ledger")
        if not self.asset_returns.columns.equals(self.effective_weights.columns):
            raise ResearchLedgerError("asset return and effective-weight columns differ")
        if not self.contributions.columns.equals(self.effective_weights.columns):
            raise ResearchLedgerError("contribution and effective-weight columns differ")

        expected_contributions = self.asset_returns * self.effective_weights
        if not np.allclose(
            self.contributions.to_numpy(dtype=float),
            expected_contributions.to_numpy(dtype=float),
            rtol=0.0,
            atol=atol,
            equal_nan=False,
        ):
            raise ResearchLedgerError("per-asset return contribution identity failed")
        if not np.allclose(
            self.daily["gross_return"].to_numpy(dtype=float),
            self.contributions.sum(axis=1).to_numpy(dtype=float),
            rtol=0.0,
            atol=atol,
        ):
            raise ResearchLedgerError("gross return does not equal contribution sum")
        expected_net = (
            self.daily["gross_return"]
            - self.daily["trade_cost"]
            - self.daily["holding_cost"]
        )
        if not np.allclose(
            self.daily["net_return"].to_numpy(dtype=float),
            expected_net.to_numpy(dtype=float),
            rtol=0.0,
            atol=atol,
        ):
            raise ResearchLedgerError("net return cost identity failed")
        expected_nav = self.daily["nav_before"] * (1.0 + self.daily["net_return"])
        if not np.allclose(
            self.daily["nav_after"].to_numpy(dtype=float),
            expected_nav.to_numpy(dtype=float),
            rtol=0.0,
            atol=atol,
        ):
            raise ResearchLedgerError("research NAV identity failed")

    def save(self, output_dir: str | Path) -> None:
        self.validate()
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.daily.to_csv(root / "research_return_ledger.csv")
        self.asset_returns.to_csv(root / "research_asset_returns.csv")
        self.effective_weights.to_csv(root / "research_effective_weights.csv")
        self.contributions.to_csv(root / "research_return_contributions.csv")
        payload = dict(self.metadata)
        payload.setdefault("schema_version", LEDGER_SCHEMA_VERSION)
        (root / "research_return_ledger_metadata.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )


def default_research_ledger_metadata() -> dict[str, object]:
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "ledger_type": "close_marked_continuous_futures_research",
        "mark_price_field": "close",
        "asset_return_formula": "close_t / close_t_minus_1 - 1",
        "position_model": "target_notional_weights_with_between_decision_drift",
        "decision_timing": "decision_on_t_effective_for_next_bar",
        "transition_basis": "post_close_mark_drifted_exposure",
        "transaction_cost_timing": "annual_rate_prorated_each_held_bar",
        "turnover_cost_policy": "diagnostic_only_not_charged",
        "final_bar_target_policy": "not_executed_without_a_following_holding_bar",
        "missing_active_return_policy": "fail_closed",
        "continuous_contract_policy": "point_in_time_dominant_ratio_adjusted",
        "accounting_scope": "research_only_not_broker_settlement_or_delivery",
    }


__all__ = [
    "CloseMarkedStep",
    "MissingActiveReturnError",
    "ResearchLedgerError",
    "ResearchReturnLedger",
    "align_transition_weights",
    "close_marked_step",
    "default_research_ledger_metadata",
]
