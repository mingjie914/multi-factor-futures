from __future__ import annotations

import numpy as np
import pandas as pd

from workflows.factor_adaptivity import (
    _compute_ic_by_sector,
    _cross_section_ic_and_slopes,
)


def test_combined_cross_section_pass_matches_standalone_statistics():
    from testing.ic_test import _vectorized_pearson_ic
    from testing.regression import _vectorized_univariate_ols

    rng = np.random.default_rng(20260727)
    dates = pd.date_range("2021-01-01", periods=500, freq="h")
    columns = [f"A{i}" for i in range(8)]
    factor = pd.DataFrame(
        rng.normal(size=(len(dates), len(columns))),
        index=dates,
        columns=columns,
    )
    returns = pd.DataFrame(
        0.01 * factor.to_numpy() + rng.normal(scale=0.02, size=factor.shape),
        index=dates,
        columns=columns,
    )
    factor.iloc[::17, :2] = np.nan
    returns.iloc[::23, 2:5] = np.nan

    actual_ic, actual_slopes = _cross_section_ic_and_slopes(
        factor.to_numpy(copy=False),
        returns.to_numpy(copy=False),
        dates,
        min_stocks=3,
    )
    expected_ic, _ = _vectorized_pearson_ic(factor, returns, min_stocks=3)
    expected_slopes = _vectorized_univariate_ols(
        factor, returns, min_stocks=3
    )

    pd.testing.assert_series_equal(actual_ic, expected_ic)
    pd.testing.assert_series_equal(actual_slopes, expected_slopes)


def test_sector_fast_path_preserves_zero_variance_row_decisions():
    from testing.ic_test import _vectorized_pearson_ic

    rng = np.random.default_rng(19)
    dates = pd.date_range("2022-01-01", periods=200, freq="h")
    columns = [f"A{i}" for i in range(10)]
    factor = pd.DataFrame(
        rng.normal(size=(len(dates), len(columns))),
        index=dates,
        columns=columns,
    )
    returns = pd.DataFrame(
        rng.normal(size=factor.shape), index=dates, columns=columns
    )
    factor.iloc[:5, :8] = 0.125
    sector_map = {
        name: ("ferrous" if position < 8 else "precious")
        for position, name in enumerate(columns)
    }

    expected_ic, _ = _vectorized_pearson_ic(
        factor[columns[:8]], returns[columns[:8]], min_stocks=3
    )
    result = _compute_ic_by_sector(
        factor, returns, sector_map, min_stocks=3, forward_period=1
    )["ferrous"]

    assert result["n_obs"] == len(expected_ic)
    assert result["ic_mean"] == expected_ic.mean()


def test_two_instrument_sector_uses_pooled_fixed_effects():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-03", periods=260, freq="B")
    factor = pd.DataFrame(
        rng.normal(size=(len(dates), 2)), index=dates, columns=["AU", "AG"]
    )
    returns = 0.02 * factor + pd.DataFrame(
        rng.normal(scale=0.01, size=factor.shape),
        index=dates,
        columns=factor.columns,
    )

    result = _compute_ic_by_sector(
        factor,
        returns,
        {"AU": "precious", "AG": "precious"},
        min_stocks=3,
        forward_period=5,
    )

    assert result["precious"]["test_type"] == "pooled_time_series_fixed_effects"
    assert result["precious"]["inference_model"].endswith("time_hac")
    assert result["precious"]["ols_beta"] > 0.0
    assert result["precious"]["ols_p_value"] < 0.01
    assert result["precious"]["n_obs"] >= 250


def test_three_instrument_sector_keeps_cross_sectional_inference():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2022-01-03", periods=120, freq="B")
    columns = ["IF", "IC", "IH"]
    factor = pd.DataFrame(
        rng.normal(size=(len(dates), len(columns))),
        index=dates,
        columns=columns,
    )
    returns = factor * 0.01

    result = _compute_ic_by_sector(
        factor,
        returns,
        {ticker: "stock_index" for ticker in columns},
        min_stocks=3,
        forward_period=1,
    )

    assert result["stock_index"]["test_type"] == "cross_section_fama_macbeth"


def test_short_single_instrument_sector_enters_observation_channel():
    dates = pd.date_range("2022-01-03", periods=100, freq="B")
    factor = pd.DataFrame({"AU": np.arange(len(dates), dtype=float)}, index=dates)
    returns = factor * 0.01

    result = _compute_ic_by_sector(
        factor,
        returns,
        {"AU": "precious"},
        min_stocks=3,
        forward_period=1,
    )

    record = result["precious"]
    assert record["test_type"] == "single_instrument_time_series"
    assert record["observation_channel"] is True
    assert record["sufficient_history"] is False
    assert record["n_trading_days"] == 100
    assert record["p_value"] == 1.0
