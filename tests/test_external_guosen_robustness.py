import numpy as np
import pandas as pd
import pytest
from dataclasses import replace

from external_strategies.guosen_trend_index.robustness import _simulate_batch
from external_strategies.guosen_trend_index.strategy import GuosenTrendIndexSpec


def _spec() -> GuosenTrendIndexSpec:
    return GuosenTrendIndexSpec(
        universe=("A", "B"),
        selection_pct=1.0,
        target_volatility=0.04,
        volatility_window=2,
        correlation_window=None,
        correlation_multiplier_cap=2.0,
        minimum_risk_observations=1,
        execution_lag_days=0,
        periods_per_year=252,
        transaction_cost_rate=0.01,
        annual_management_fee=0.0,
        asset_caps={"A": 1.0, "B": 1.0},
    )


def test_target_at_close_is_effective_on_next_bar():
    targets = np.array([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
    returns = np.array([[np.nan, np.nan], [0.10, 0.0], [0.0, 0.0]])

    net, turnover = _simulate_batch(
        targets, returns, _spec(), dates=pd.date_range("2020-01-01", periods=3)
    )

    np.testing.assert_allclose(net[0, :2], [0.0, 0.09])
    np.testing.assert_allclose(turnover[0, 0], 1.0)


def test_turnover_uses_post_return_drift_not_target_difference():
    targets = np.array([
        [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
    ])
    returns = np.array([[np.nan, np.nan], [0.20, 0.0], [0.0, 0.0]])

    _, turnover = _simulate_batch(
        targets,
        returns,
        replace(_spec(), transaction_cost_rate=0.0),
        dates=pd.date_range("2020-01-01", periods=3),
    )

    # After A rises, the current notional is about (0.54545, 0.45455),
    # even though the target did not change.
    np.testing.assert_allclose(turnover[0, 1], 1.0 / 11.0)


def test_missing_active_return_fails_closed():
    targets = np.array([[[1.0, 0.0], [1.0, 0.0]]])
    returns = np.array([[np.nan, np.nan], [np.nan, 0.0]])

    with pytest.raises(ValueError, match="active asset return is missing"):
        _simulate_batch(
            targets,
            returns,
            _spec(),
            dates=pd.date_range("2020-01-01", periods=2),
        )


def test_untradable_decision_freezes_batch_weight_until_price_returns():
    targets = np.array([[[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
    returns = np.array(
        [[np.nan, np.nan], [0.0, 0.0], [0.0, 0.0], [-0.20, 0.0], [0.0, 0.0]]
    )
    tradable = np.array(
        [[True, True], [True, True], [False, True], [True, True], [True, True]]
    )

    net, turnover = _simulate_batch(
        targets,
        returns,
        replace(_spec(), transaction_cost_rate=0.0),
        decision_tradable=tradable,
        dates=pd.date_range("2020-01-01", periods=5),
    )

    np.testing.assert_allclose(turnover[0, 1:3], 0.0)
    np.testing.assert_allclose(net[0, 3], -0.20)


def test_contract_roll_counts_close_and_open():
    targets = np.array([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
    returns = np.array([[np.nan, np.nan], [0.0, 0.0], [0.0, 0.0]])
    schedule = pd.DataFrame(
        [["A2401", None], ["A2401", None], ["A2405", None]],
        index=pd.date_range("2020-01-01", periods=3),
        columns=["A", "B"],
    )
    from external_strategies.guosen_trend_index.robustness import (
        _prepare_contract_schedule,
    )

    prepared = _prepare_contract_schedule(
        schedule, pd.DatetimeIndex(schedule.index), ("A", "B")
    )
    _, turnover = _simulate_batch(
        targets,
        returns,
        replace(_spec(), transaction_cost_rate=0.0),
        contract_schedule=prepared,
        dates=pd.DatetimeIndex(schedule.index),
    )

    np.testing.assert_allclose(turnover[0], [1.0, 2.0, 0.0])


def test_batch_untradable_close_delays_contract_roll():
    targets = np.array([[[1.0, 0.0]] * 5])
    returns = np.array(
        [[np.nan, np.nan], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    )
    dates = pd.date_range("2020-01-01", periods=5)
    schedule = pd.DataFrame(
        [
            ["A2401", None],
            ["A2401", None],
            ["A2405", None],
            ["A2405", None],
            ["A2405", None],
        ],
        index=dates,
        columns=["A", "B"],
    )
    tradable = np.array(
        [[True, True], [True, True], [False, True], [True, True], [True, True]]
    )
    from external_strategies.guosen_trend_index.robustness import (
        _prepare_contract_schedule,
    )

    prepared = _prepare_contract_schedule(schedule, dates, ("A", "B"))
    _, turnover = _simulate_batch(
        targets,
        returns,
        replace(_spec(), transaction_cost_rate=0.0),
        contract_schedule=prepared,
        decision_tradable=tradable,
        dates=dates,
    )

    np.testing.assert_allclose(turnover[0], [1.0, 0.0, 0.0, 2.0, 0.0])


def test_batch_fails_closed_when_active_target_contract_is_blank():
    dates = pd.date_range("2020-01-01", periods=2)
    targets = np.array([[[1.0, 0.0], [1.0, 0.0]]])
    returns = np.array([[np.nan, np.nan], [0.0, 0.0]])
    schedule = pd.DataFrame(
        [["A2401", None], ["", None]],
        index=dates,
        columns=["A", "B"],
    )
    from external_strategies.guosen_trend_index.robustness import (
        _prepare_contract_schedule,
    )

    prepared = _prepare_contract_schedule(schedule, dates, ("A", "B"))
    with pytest.raises(ValueError, match="active target contract is missing"):
        _simulate_batch(
            targets,
            returns,
            replace(_spec(), transaction_cost_rate=0.0),
            contract_schedule=prepared,
            dates=dates,
        )
