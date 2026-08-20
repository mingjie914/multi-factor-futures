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

from backtest.metrics import TRADING_DAYS_PER_YEAR


LEDGER_SCHEMA_VERSION = 6
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
    if np.isinf(returns.to_numpy(dtype=float)).any():
        raise ResearchLedgerError("asset returns contain infinity")
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


def contract_transition_turnover(
    target: pd.Series,
    current: pd.Series,
    *,
    current_contracts: pd.Series | None = None,
    target_contracts: pd.Series | None = None,
) -> tuple[float, float]:
    """Return full traded notional and its explicit rollover component."""
    target, current = align_transition_weights(target, current)
    if current_contracts is None or target_contracts is None:
        return float((target - current).abs().sum()), 0.0

    contract_target, contract_current = contract_transition_weight_vectors(
        target,
        current,
        current_contracts=current_contracts,
        target_contracts=target_contracts,
    )

    current_contracts = pd.Series(current_contracts).reindex(current.index)
    target_contracts = pd.Series(target_contracts).reindex(current.index)
    rollover = 0.0
    for symbol in current.index:
        current_weight = float(current.loc[symbol])
        target_weight = float(target.loc[symbol])
        if (
            abs(current_weight) <= ACTIVE_WEIGHT_TOLERANCE
            and abs(target_weight) <= ACTIVE_WEIGHT_TOLERANCE
        ):
            continue
        current_contract = current_contracts.loc[symbol]
        target_contract = target_contracts.loc[symbol]
        if (
            abs(current_weight) > ACTIVE_WEIGHT_TOLERANCE
            and abs(target_weight) > ACTIVE_WEIGHT_TOLERANCE
            and str(current_contract) != str(target_contract)
        ):
            amount = abs(current_weight) + abs(target_weight)
            rollover += amount
    traded = float((contract_target - contract_current).abs().sum())
    return traded, float(rollover)


def contract_transition_weight_vectors(
    target: pd.Series,
    current: pd.Series,
    *,
    current_contracts: pd.Series,
    target_contracts: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Represent a root transition on concrete contracts for cost models."""
    target, current = align_transition_weights(target, current)
    current_contracts = pd.Series(current_contracts).reindex(current.index)
    target_contracts = pd.Series(target_contracts).reindex(current.index)
    current_by_contract: dict[str, float] = {}
    target_by_contract: dict[str, float] = {}

    for root in current.index:
        current_weight = float(current.loc[root])
        target_weight = float(target.loc[root])
        current_contract = current_contracts.loc[root]
        target_contract = target_contracts.loc[root]
        if abs(current_weight) > ACTIVE_WEIGHT_TOLERANCE:
            if pd.isna(current_contract):
                raise ResearchLedgerError(
                    f"active current weight has no contract: {root}"
                )
            key = str(current_contract)
            current_by_contract[key] = current_by_contract.get(key, 0.0) + current_weight
        if abs(target_weight) > ACTIVE_WEIGHT_TOLERANCE:
            if pd.isna(target_contract):
                raise ResearchLedgerError(
                    f"active target weight has no next-bar contract: {root}"
                )
            key = str(target_contract)
            target_by_contract[key] = target_by_contract.get(key, 0.0) + target_weight

    contracts = pd.Index(current_by_contract).union(
        pd.Index(target_by_contract), sort=False
    )
    return (
        pd.Series(target_by_contract, dtype=float).reindex(contracts).fillna(0.0),
        pd.Series(current_by_contract, dtype=float).reindex(contracts).fillna(0.0),
    )


def build_close_marked_ledger(
    target_weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    *,
    trade_cost_rate: float = 0.0,
    annual_fee: float = 0.0,
    annual_roll_cost: float = 0.0,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
    cost_multiplier: float = 1.0,
    contract_schedule: pd.DataFrame | None = None,
    decision_tradable: pd.DataFrame | None = None,
    initial_nav: float = 1000.0,
) -> "ResearchReturnLedger":
    """Build a causal daily ledger from close-indexed portfolio decisions.

    A target recorded on close ``T`` is effective for the ``T+1`` return.  At
    close ``T`` the next target is compared with the post-return drifted
    exposure.  If a contract schedule is supplied, a root-level rollover closes
    the old concrete contract and opens the new one even when root weight is
    unchanged.
    """
    rate = float(trade_cost_rate)
    fee = float(annual_fee)
    roll_rate = float(annual_roll_cost)
    periods = float(periods_per_year)
    multiplier = float(cost_multiplier)
    initial_nav = float(initial_nav)
    if (
        not np.isfinite([rate, fee, roll_rate, periods, multiplier, initial_nav]).all()
        or rate < 0.0
        or fee < 0.0
        or roll_rate < 0.0
        or periods <= 0.0
        or multiplier < 0.0
        or initial_nav <= 0.0
    ):
        raise ResearchLedgerError("invalid ledger cost or annualization inputs")

    target_dates = pd.DatetimeIndex(target_weights.index)
    return_dates = pd.DatetimeIndex(asset_returns.index)
    for name, index in (
        ("target_weights", target_dates),
        ("asset_returns", return_dates),
    ):
        if index.has_duplicates or not index.is_monotonic_increasing:
            raise ResearchLedgerError(f"{name} dates must be unique and sorted")
    if not target_dates.equals(return_dates):
        raise ResearchLedgerError(
            "target_weights and asset_returns must have identical daily dates"
        )
    if target_weights.columns.has_duplicates or asset_returns.columns.has_duplicates:
        raise ResearchLedgerError("ledger input instruments must be unique")
    dates = target_dates
    if dates.empty:
        raise ResearchLedgerError("research ledger inputs must contain daily dates")
    universe = pd.Index(target_weights.columns).union(
        pd.Index(asset_returns.columns), sort=False
    )
    try:
        numeric_targets = target_weights.astype(float)
    except (TypeError, ValueError) as exc:
        raise ResearchLedgerError("target weights must be numeric") from exc
    if not np.isfinite(numeric_targets.to_numpy(dtype=float)).all():
        raise ResearchLedgerError("target weights contain NaN or infinity")
    targets = numeric_targets.reindex(
        index=dates, columns=universe, fill_value=0.0
    )
    returns = asset_returns.reindex(index=dates, columns=universe).astype(float)
    schedule = None
    if contract_schedule is not None:
        schedule = contract_schedule.copy()
        schedule.index = pd.DatetimeIndex(schedule.index)
        if schedule.index.has_duplicates or schedule.columns.has_duplicates:
            raise ResearchLedgerError(
                "contract_schedule must have unique dates and instruments"
            )
        missing_dates = dates.difference(schedule.index)
        missing_instruments = universe.difference(schedule.columns)
        if len(missing_dates) or len(missing_instruments):
            raise ResearchLedgerError(
                "contract_schedule does not cover all ledger dates and instruments"
            )
        schedule = schedule.reindex(index=dates, columns=universe)
    tradable = None
    if decision_tradable is not None:
        tradable = decision_tradable.copy()
        tradable.index = pd.DatetimeIndex(tradable.index)
        if tradable.index.has_duplicates or tradable.columns.has_duplicates:
            raise ResearchLedgerError(
                "decision_tradable must have unique dates and instruments"
            )
        missing_dates = dates.difference(tradable.index)
        missing_instruments = universe.difference(tradable.columns)
        if len(missing_dates) or len(missing_instruments):
            raise ResearchLedgerError(
                "decision_tradable does not cover all ledger dates and instruments"
            )
        tradable = tradable.reindex(index=dates, columns=universe).eq(True)

    current = pd.Series(0.0, index=universe, dtype=float)
    held_contracts = pd.Series(pd.NA, index=universe, dtype="object")
    pending_traded_notional = 0.0
    pending_roll_turnover = 0.0
    nav = initial_nav
    daily_rows: list[dict[str, float | int]] = []
    return_rows: list[pd.Series] = []
    effective_rows: list[pd.Series] = []
    contribution_rows: list[pd.Series] = []

    for position, date in enumerate(dates):
        trade_cost = pending_traded_notional * rate * multiplier
        holding_cost = (
            0.0
            if position == 0
            else (fee + float(current.abs().sum()) * roll_rate) / periods
        )
        try:
            step = close_marked_step(
                current,
                returns.loc[date],
                trade_cost=trade_cost,
                holding_cost=holding_cost,
            )
        except MissingActiveReturnError as exc:
            raise MissingActiveReturnError(f"{date.date()}: {exc}") from exc
        nav_before = nav
        nav = nav_before * (1.0 + step.net_return)
        decision_turnover = 0.0
        rollover_turnover = 0.0
        blocked_target_notional = 0.0
        blocked_instruments = 0
        next_pending_notional = 0.0
        next_current = step.end_weights.reindex(universe).fillna(0.0)

        # A final-close target is observable but not an executed trade because
        # there is no following holding bar in this evaluation interval.
        if position < len(dates) - 1:
            desired_target = targets.loc[date]
            target = desired_target
            next_date = dates[position + 1]
            if tradable is not None:
                # A close-T target can only become effective for T+1 when the
                # root is observable at both boundaries.  This prevents a
                # pre-halt target from being opened on the halted bar and
                # preserves the reopening jump for positions actually held.
                can_trade = tradable.loc[date] & tradable.loc[next_date]
                target = desired_target.where(can_trade, next_current)
                blocked = (~can_trade) & (
                    desired_target.sub(next_current).abs().gt(ACTIVE_WEIGHT_TOLERANCE)
                )
                blocked_target_notional = float(
                    desired_target.sub(target).abs().sum()
                )
                blocked_instruments = int(blocked.sum())
            if schedule is None:
                decision_turnover, rollover_turnover = contract_transition_turnover(
                    target, next_current
                )
            else:
                target_contracts = schedule.loc[next_date].copy()
                if tradable is not None:
                    # A frozen root cannot roll.  Preserve the actually held
                    # contract and execute the delayed roll on the first later
                    # close at which that root is tradable.
                    target_contracts = target_contracts.where(can_trade, held_contracts)
                decision_turnover, rollover_turnover = contract_transition_turnover(
                    target,
                    next_current,
                    current_contracts=held_contracts,
                    target_contracts=target_contracts,
                )
                held_contracts = target_contracts.where(
                    target.abs().gt(ACTIVE_WEIGHT_TOLERANCE), pd.NA
                )
            next_pending_notional = decision_turnover
            next_current = target

        effective = step.effective_weights.reindex(universe).fillna(0.0)
        marked_returns = step.asset_returns.reindex(universe).fillna(0.0)
        contributions = step.contributions.reindex(universe).fillna(0.0)
        daily_rows.append(
            {
                "nav_before": nav_before,
                "nav_after": nav,
                "gross_return": step.gross_return,
                "trade_cost": step.trade_cost,
                "holding_cost": step.holding_cost,
                "management_fee": step.holding_cost,
                "net_return": step.net_return,
                "executed_traded_notional": pending_traded_notional,
                "executed_roll_turnover": pending_roll_turnover,
                "decision_turnover": decision_turnover,
                "turnover": pending_traded_notional,
                "half_turnover": 0.5 * pending_traded_notional,
                "roll_turnover": rollover_turnover,
                "blocked_target_notional": blocked_target_notional,
                "blocked_instruments": blocked_instruments,
                "gross_exposure": float(effective.abs().sum()),
                "net_exposure": float(effective.sum()),
                "active_instruments": int(
                    effective.abs().gt(ACTIVE_WEIGHT_TOLERANCE).sum()
                ),
            }
        )
        return_rows.append(marked_returns)
        effective_rows.append(effective)
        contribution_rows.append(contributions)
        current = next_current
        pending_traded_notional = next_pending_notional
        pending_roll_turnover = rollover_turnover

    daily = pd.DataFrame(daily_rows, index=dates)
    ledger = ResearchReturnLedger(
        daily=daily,
        asset_returns=pd.DataFrame(return_rows, index=dates, columns=universe),
        effective_weights=pd.DataFrame(
            effective_rows, index=dates, columns=universe
        ),
        contributions=pd.DataFrame(
            contribution_rows, index=dates, columns=universe
        ),
        metadata={
            **default_research_ledger_metadata(),
            "schema_version": LEDGER_SCHEMA_VERSION,
            "transaction_cost_timing": "next_bar_after_close_decision",
            "turnover_cost_policy": "full_traded_notional_times_rate",
            "turnover_definition": "full_l1_open_plus_close_notional",
            "half_turnover_definition": "0.5_times_full_traded_notional",
            "decision_turnover_timing": "decision_close_for_following_bar",
            "executed_turnover_timing": (
                "holding_bar_when_transition_becomes_effective"
            ),
            "rollover_cost_policy": "explicit_contract_close_and_open",
            "untradable_rollover_policy": "delay_until_tradable",
            "untradable_decision_policy": (
                "require_decision_and_next_close_then_freeze"
                if tradable is not None else "not_provided"
            ),
            "trade_cost_rate": rate,
            "annual_fee": fee,
            "annual_fee_policy": "account_level_each_elapsed_bar_after_anchor",
            "annual_roll_cost": roll_rate,
            "annual_roll_cost_policy": "gross_exposure_each_elapsed_bar_after_anchor",
            "periods_per_year": periods,
            "cost_multiplier": multiplier,
        },
    )
    ledger.validate()
    return ledger


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
        if self.daily.index.has_duplicates or not self.daily.index.is_monotonic_increasing:
            raise ResearchLedgerError("daily ledger dates must be unique and sorted")
        if self.daily.columns.has_duplicates:
            raise ResearchLedgerError("daily ledger columns must be unique")
        required = {
            "nav_before",
            "nav_after",
            "gross_return",
            "trade_cost",
            "holding_cost",
            "net_return",
            "decision_turnover",
            "executed_traded_notional",
            "roll_turnover",
            "executed_roll_turnover",
            "gross_exposure",
            "net_exposure",
            "active_instruments",
        }
        missing = required - set(self.daily.columns)
        if missing:
            raise ResearchLedgerError(
                "research ledger is missing daily columns: " + ", ".join(sorted(missing))
            )
        daily_numeric = self.daily.loc[:, sorted(required)].apply(
            pd.to_numeric, errors="coerce"
        )
        if not np.isfinite(daily_numeric.to_numpy(dtype=float)).all():
            raise ResearchLedgerError("daily ledger contains NaN or infinity")
        non_negative = [
            "trade_cost",
            "holding_cost",
            "decision_turnover",
            "executed_traded_notional",
            "roll_turnover",
            "executed_roll_turnover",
        ]
        non_negative.extend(
            name for name in ("turnover", "half_turnover")
            if name in self.daily.columns
        )
        if bool((self.daily[non_negative] < 0).any().any()):
            raise ResearchLedgerError("ledger costs and turnover must be non-negative")
        if "turnover" in self.daily.columns and not np.allclose(
            self.daily["turnover"],
            self.daily["executed_traded_notional"],
            rtol=0.0,
            atol=atol,
        ):
            raise ResearchLedgerError(
                "turnover does not match executed traded notional"
            )
        if "half_turnover" in self.daily.columns and not np.allclose(
            self.daily["half_turnover"],
            0.5 * self.daily["executed_traded_notional"],
            rtol=0.0,
            atol=atol,
        ):
            raise ResearchLedgerError(
                "half turnover does not match executed traded notional"
            )
        if bool(
            (
                self.daily["roll_turnover"]
                > self.daily["decision_turnover"] + atol
            ).any()
        ) or bool(
            (
                self.daily["executed_roll_turnover"]
                > self.daily["executed_traded_notional"] + atol
            ).any()
        ):
            raise ResearchLedgerError("roll turnover exceeds total traded notional")
        if bool((self.daily[["nav_before", "nav_after"]] <= 0).any().any()):
            raise ResearchLedgerError("ledger NAV must remain positive")
        for frame_name, frame in {
            "asset_returns": self.asset_returns,
            "effective_weights": self.effective_weights,
            "contributions": self.contributions,
        }.items():
            if not frame.index.equals(self.daily.index):
                raise ResearchLedgerError(f"{frame_name} index does not match daily ledger")
            if not np.isfinite(frame.to_numpy(dtype=float)).all():
                raise ResearchLedgerError(f"{frame_name} contains NaN or infinity")
        if not self.asset_returns.columns.equals(self.effective_weights.columns):
            raise ResearchLedgerError("asset return and effective-weight columns differ")
        if not self.contributions.columns.equals(self.effective_weights.columns):
            raise ResearchLedgerError("contribution and effective-weight columns differ")

        expected_gross = self.effective_weights.abs().sum(axis=1)
        expected_net_exposure = self.effective_weights.sum(axis=1)
        expected_active = self.effective_weights.abs().gt(ACTIVE_WEIGHT_TOLERANCE).sum(axis=1)
        if not np.allclose(
            self.daily["gross_exposure"], expected_gross, rtol=0.0, atol=atol
        ):
            raise ResearchLedgerError("gross exposure does not match effective weights")
        if not np.allclose(
            self.daily["net_exposure"], expected_net_exposure, rtol=0.0, atol=atol
        ):
            raise ResearchLedgerError("net exposure does not match effective weights")
        if not np.array_equal(
            self.daily["active_instruments"].to_numpy(dtype=int),
            expected_active.to_numpy(dtype=int),
        ):
            raise ResearchLedgerError("active-instrument count does not match effective weights")

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
        if len(self.daily) > 1 and not np.allclose(
            self.daily["nav_before"].iloc[1:].to_numpy(dtype=float),
            self.daily["nav_after"].iloc[:-1].to_numpy(dtype=float),
            rtol=0.0,
            atol=atol,
        ):
            raise ResearchLedgerError("research NAV chain is discontinuous")

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
        "decision_turnover_timing": "decision_close_for_following_bar",
        "executed_turnover_timing": "holding_bar_when_transition_becomes_effective",
        "turnover_definition": "full_l1_open_plus_close_notional",
        "half_turnover_definition": "0.5_times_full_traded_notional",
        "transition_basis": "post_close_mark_drifted_exposure",
        "transaction_cost_timing": "next_bar_after_close_decision",
        "turnover_cost_policy": "diagnostic_only_not_charged",
        "annual_fee_policy": "account_level_each_elapsed_bar_after_anchor",
        "final_bar_target_policy": "not_executed_without_a_following_holding_bar",
        "missing_active_return_policy": "fail_closed",
        "unavailable_close_marking_policy": (
            "last_observable_close_then_full_change_on_next_observation"
        ),
        "continuous_contract_policy": "point_in_time_dominant_ratio_adjusted",
        "accounting_scope": "research_only_not_broker_settlement_or_delivery",
    }


__all__ = [
    "CloseMarkedStep",
    "MissingActiveReturnError",
    "ResearchLedgerError",
    "ResearchReturnLedger",
    "align_transition_weights",
    "build_close_marked_ledger",
    "close_marked_step",
    "contract_transition_turnover",
    "contract_transition_weight_vectors",
    "default_research_ledger_metadata",
]
