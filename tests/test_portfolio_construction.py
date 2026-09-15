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
    select_long_short_pools,
)
from optimization.risk_budgeting import RiskBudgetingOptimizer


def test_zero_buffer_and_empty_holdings_preserve_original_pools_exactly():
    from optimization.portfolio_construction import select_pool
    rng = np.random.default_rng(915)
    symbols = [f"S{i:02d}" for i in range(38)]
    sectors = {symbol: str(i % 8) for i, symbol in enumerate(symbols)}
    constraints = PortfolioConstraints()
    for _ in range(20):
        score = pd.Series(rng.integers(0, 20, len(symbols)), index=symbols)
        long = select_pool(score, eligible=symbols, sector_of=sectors,
                           top_n=10, sector_count_cap=3, ascending=False)
        short = select_pool(score, eligible=symbols, sector_of=sectors,
                            top_n=10, sector_count_cap=3, ascending=True, excluded=long)
        for kwargs in (dict(exit_buffer=0, previous_long=short, previous_short=long),
                       dict(exit_buffer=2)):
            actual = select_long_short_pools(score, eligible=symbols,
                sector_of=sectors, constraints=constraints, **kwargs)
            assert actual == (long, short)


@pytest.mark.parametrize("buffer,previous,expected", [
    (1, ["A", "C"], ["A", "C"]),
    (1, ["A", "D"], ["A", "B"]),
    (2, ["A", "D"], ["A", "D"]),
])
def test_buffer_retains_only_same_side_names_in_declared_band(buffer, previous, expected):
    score = pd.Series(range(10, 0, -1), index=list("ABCDEFGHIJ"), dtype=float)
    long, short = select_long_short_pools(score, eligible=score.index, sector_of={},
        constraints=PortfolioConstraints(top_n_per_side=2, sector_count_cap=0),
        previous_long=previous, previous_short=["J", "H"], exit_buffer=buffer)
    assert long == expected
    assert short == ["J", "H"]
    assert not set(long) & set(short)


def test_buffer_band_counts_sector_admissible_names_not_raw_ranks():
    score = pd.Series(range(10, 0, -1), index=list("ABCDEFGHIJ"), dtype=float)
    sectors = {symbol: ("first" if symbol in "ABCD" else symbol) for symbol in score.index}
    long, _ = select_long_short_pools(score, eligible=score.index, sector_of=sectors,
        constraints=PortfolioConstraints(top_n_per_side=2, sector_count_cap=1),
        previous_long=["A", "F"], exit_buffer=1)
    # Legal entrants are A,E and the one-name retention extension is F.
    assert long == ["A", "F"]


def test_buffer_falls_back_when_retention_makes_opposite_side_infeasible():
    score = pd.Series([6., 5., 4., 3., 2., 1.], index=["A1", "A2", "B", "C", "A3", "A4"])
    sectors = {symbol: symbol[0] for symbol in score.index}
    constraints = PortfolioConstraints(top_n_per_side=3, sector_count_cap=2)
    baseline = select_long_short_pools(score, eligible=score.index, sector_of=sectors,
                                     constraints=constraints)
    diagnostics = {}
    actual = select_long_short_pools(score, eligible=score.index, sector_of=sectors,
        constraints=constraints, previous_long=["A1", "B", "C"], exit_buffer=1,
        diagnostics=diagnostics)
    assert actual == baseline
    assert diagnostics["constraint_fallback"] is True
    assert diagnostics["fallback_reason"]


def test_buffer_handles_missing_scores_exit_reentry_and_stable_ties():
    score = pd.Series([10., 9., 8., 8., 6., 5., 4., 3., 2., 1.], index=list("ABCDEFGHIJ"))
    kwargs = dict(eligible=score.index, sector_of={},
                  constraints=PortfolioConstraints(top_n_per_side=2, sector_count_cap=0),
                  exit_buffer=1)
    assert select_long_short_pools(score, previous_long=["A", "C"], **kwargs)[0] == ["A", "C"]
    unavailable = score.copy()
    unavailable["C"] = np.nan
    exited = select_long_short_pools(unavailable, previous_long=["A", "C"], **kwargs)[0]
    assert exited == ["A", "B"]
    # Merely returning inside the retention band does not restore an exited name.
    assert select_long_short_pools(score, previous_long=exited, **kwargs)[0] == exited
    score["C"] = 11.
    assert select_long_short_pools(score, previous_long=[], **kwargs)[0] == ["C", "A"]
    unavailable["C"] = np.inf
    assert "C" not in select_long_short_pools(unavailable, previous_long=["C"], **kwargs)[0]


@pytest.mark.parametrize("value", [-1, 0.5, True])
def test_buffer_rejects_invalid_width(value):
    score = pd.Series(range(8), index=list("ABCDEFGH"), dtype=float)
    with pytest.raises(PortfolioConstructionError, match="nonnegative integer"):
        select_long_short_pools(score, eligible=score.index, sector_of={},
            constraints=PortfolioConstraints(top_n_per_side=2), exit_buffer=value)


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
