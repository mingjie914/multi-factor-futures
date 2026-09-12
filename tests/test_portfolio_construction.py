from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimization.factor_weighting import causal_history
from optimization.portfolio_construction import (
    PortfolioConstraints,
    PortfolioConstructionError,
    allocate_sleeve,
    combine_sleeves,
    causal_risk_window,
)
from optimization.risk_budgeting import RiskBudgetingOptimizer


def test_causal_ic_history_uses_exactly_sixty_rows_before_decision():
    dates = pd.bdate_range("2026-01-01", periods=61)
    history = pd.DataFrame({"f1": np.arange(61), "f2": np.arange(61)}, index=dates)

    actual = causal_history(history, dates[-1], 60)

    assert len(actual) == 60
    assert actual.index.max() < dates[-1]
    assert actual.index.equals(dates[:-1])


@pytest.mark.parametrize("timezone", [None, "Asia/Shanghai"])
def test_history_slices_match_boolean_reference_and_do_not_alias(timezone):
    dates = pd.bdate_range("2025-01-01", periods=100, tz=timezone)
    original = pd.DataFrame({"x": np.arange(100, dtype=float)}, index=dates)
    original.iloc[8] = np.nan
    original.iloc[12] = np.inf
    for frame in (original, original.iloc[::-1], original.iloc[[0, 1, 1, 2]], original.iloc[:0]):
        for decision in (dates[0], dates[45], dates[45] + pd.Timedelta(hours=12), dates[-1] + pd.Timedelta(days=4)):
            for window in (1, 60, 150):
                expected = frame.loc[frame.index < decision].tail(window)
                actual = causal_history(frame, decision, window)
                pd.testing.assert_frame_equal(actual, expected, check_exact=True)
                if len(actual):
                    snapshot = frame.copy()
                    actual.iloc[0, 0] = -999
                    pd.testing.assert_frame_equal(frame, snapshot, check_exact=True)
        pd.testing.assert_frame_equal(causal_history(frame, pd.NaT, 60), frame.loc[frame.index < pd.NaT].tail(60), check_exact=True)
    for decision in (dates[0], dates[60], dates[-1] + pd.Timedelta(days=7), pd.NaT):
        for lookback in (1, 90, 400):
            expected = original.loc[(dates >= decision - pd.Timedelta(days=lookback)) & (dates < decision)]
            actual = causal_risk_window(original, decision, lookback)
            pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    for invalid in (original.iloc[::-1], original.iloc[[0, 1, 1, 2]]):
        with pytest.raises(PortfolioConstructionError, match="unique and sorted"):
            causal_risk_window(invalid, dates[-1], 90)


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
