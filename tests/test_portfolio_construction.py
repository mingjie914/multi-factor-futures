from __future__ import annotations

import numpy as np
import pandas as pd

from optimization.factor_weighting import causal_history
from optimization.portfolio_construction import (
    PortfolioConstraints,
    allocate_sleeve,
    combine_sleeves,
)
from optimization.risk_budgeting import RiskBudgetingOptimizer


def test_causal_ic_history_uses_exactly_sixty_rows_before_decision():
    dates = pd.bdate_range("2026-01-01", periods=61)
    history = pd.DataFrame({"f1": np.arange(61), "f2": np.arange(61)}, index=dates)

    actual = causal_history(history, dates[-1], 60)

    assert len(actual) == 60
    assert actual.index.max() < dates[-1]
    assert actual.index.equals(dates[:-1])


def test_erc_asset_cap_still_holds_after_unit_sleeve_projection(monkeypatch):
    symbols = [f"S{i}" for i in range(10)]
    history = pd.DataFrame(
        np.random.default_rng(9).normal(0.0, 0.01, size=(40, 10)),
        columns=symbols,
    )
    monkeypatch.setattr(
        RiskBudgetingOptimizer,
        "_erc_weights",
        lambda covariance, budgets: np.array([0.91] + [0.01] * 9),
    )
    constraints = PortfolioConstraints(top_n_per_side=10, asset_max_fraction=0.20)

    weights = allocate_sleeve(
        history,
        method="erc",
        constraints=constraints,
        sector_of={},
    )

    assert np.isclose(weights.sum(), 1.0)
    assert weights.max() <= 0.20 + 1e-12
    assert weights.min() >= 0.005 - 1e-12


def test_asset_override_and_sector_weight_cap_are_independent_constraints():
    symbols = [f"S{i}" for i in range(10)]
    history = pd.DataFrame(
        np.random.default_rng(11).normal(0.0, 0.01, size=(40, 10)),
        columns=symbols,
    )
    sector_of = {symbol: ("sector_a" if index < 3 else "other") for index, symbol in enumerate(symbols)}
    constraints = PortfolioConstraints(
        top_n_per_side=10,
        asset_max_fraction=0.20,
        asset_max_overrides={"S0": 0.08},
        sector_weight_caps={"sector_a": 0.30},
    )

    weights = allocate_sleeve(
        history,
        method="inverse_volatility",
        constraints=constraints,
        sector_of=sector_of,
    )

    assert weights["S0"] <= 0.08 + 1e-9
    assert weights.loc[["S0", "S1", "S2"]].sum() <= 0.30 + 1e-9


def test_combined_sleeves_validate_exact_counts_and_exposures():
    long_pool = [f"L{i}" for i in range(10)]
    short_pool = [f"Q{i}" for i in range(10)]
    constraints = PortfolioConstraints(top_n_per_side=10)
    long_weights = pd.Series(0.1, index=long_pool)
    short_weights = pd.Series(0.1, index=short_pool)

    result = combine_sleeves(
        long_weights,
        short_weights,
        universe=long_pool + short_pool,
        long_pool=long_pool,
        short_pool=short_pool,
        constraints=constraints,
        sector_of={symbol: symbol for symbol in long_pool + short_pool},
    )

    assert (result > 0).sum() == 10
    assert (result < 0).sum() == 10
    assert np.isclose(result[result > 0].sum(), 1.0)
    assert np.isclose(result[result < 0].sum(), -1.0)
    assert np.isclose(result.abs().sum(), 2.0)
    assert np.isclose(result.sum(), 0.0)
