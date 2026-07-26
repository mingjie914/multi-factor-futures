from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from strategies.defensive_trend_risk_parity import DefensiveTrendRiskParity


def _prices(periods: int = 90) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=periods, freq="B")
    steps = np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "A": 100.0 * np.exp(0.003 * steps),
            "B": 100.0 * np.exp(0.002 * steps + 0.01 * np.sin(steps / 4.0)),
            "C": 100.0 * np.exp(-0.001 * steps),
        },
        index=dates,
    )


def _strategy(**overrides) -> DefensiveTrendRiskParity:
    params = {
        "lookbacks": [5, 10],
        "rebalance_freq": 5,
        "volatility_window": 20,
        "top_n_per_sector": 2,
        "asset_cap": 0.40,
        "sector_cap": 0.60,
        "turnover_cap": 0.25,
        "annual_fee": 0.0,
        "sector_map": {"A": "s1", "B": "s1", "C": "s2"},
    }
    params.update(overrides)
    return DefensiveTrendRiskParity(**params)


def test_defensive_decisions_are_t_plus_one_and_point_in_time():
    prices = _prices()
    base = _strategy().run(prices)

    first_decision = base.weights_history.index[0]
    first_effective = prices.index[prices.index.get_loc(first_decision) + 1]
    assert base.turnover.loc[first_decision] == 0.0
    assert base.turnover.loc[first_effective] > 0.0

    changed_future = prices.copy()
    cutoff = prices.index[55]
    changed_future.loc[changed_future.index > cutoff, "A"] *= 4.0
    changed = _strategy().run(changed_future)
    pd.testing.assert_frame_equal(
        base.weights_history.loc[:cutoff], changed.weights_history.loc[:cutoff]
    )


@pytest.mark.parametrize(
    "allocation",
    ["inverse_volatility", "correlation_adjusted_inverse_volatility", "erc", "hrp", "shrinkage_min_variance"],
)
def test_defensive_allocations_are_finite_nonnegative_and_capped(allocation):
    strategy = _strategy(allocation=allocation)
    result = strategy.run(_prices())
    weights = result.weights_history

    assert np.isfinite(weights.to_numpy()).all()
    assert (weights >= -1e-12).all().all()
    assert weights.max().max() <= strategy.asset_cap + 1e-8
    assert weights[["A", "B"]].sum(axis=1).max() <= strategy.sector_cap + 1e-8
    effective = weights.reindex(result.turnover.index).ffill().shift(1).fillna(0.0)
    assert effective.diff().abs().sum(axis=1).max() <= strategy.turnover_cap + 1e-8


def test_defensive_costs_nav_and_failure_ledger_are_consistent():
    class CostModel:
        def estimate_cost(self, target, current, date):
            return float((target - current).abs().sum()) * 0.001

    prices = _prices()
    result = _strategy(annual_fee=0.0252).run(prices, cost_model=CostModel())
    assert result.costs.iloc[0] == 0.0
    assert result.nav.iloc[0] == 1.0
    assert result.costs.iloc[1] == pytest.approx(0.0001)
    assert result.nav.iloc[1] == pytest.approx(1.0 - result.costs.iloc[1])
    assert np.isfinite(result.nav.to_numpy()).all()
    assert result.metrics["total_transaction_cost"] == pytest.approx(result.costs.sum())
    assert result.failure_ledger == []


def test_defensive_runner_requires_explicit_standalone_enablement():
    from pipeline.runner import PipelineRunner

    runner = object.__new__(PipelineRunner)
    runner.config = SimpleNamespace(
        defensive_sleeve=SimpleNamespace(enabled=False, integration_mode="standalone")
    )
    with pytest.raises(RuntimeError, match="disabled"):
        runner.run_defensive_sleeve()

    runner.config.defensive_sleeve.enabled = True
    runner.config.defensive_sleeve.integration_mode = "meta"
    with pytest.raises(ValueError, match="standalone"):
        runner.run_defensive_sleeve()
