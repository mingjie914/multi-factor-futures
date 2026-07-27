from __future__ import annotations

import numpy as np
import pandas as pd

from testing.ic_test import ICTest
from testing.regression import _newey_west_t_stat, _vectorized_univariate_ols
from workflows.research import _joint_ic_ols_statistics


def test_joint_ic_ols_matches_separate_reference_paths():
    rng = np.random.default_rng(20260726)
    dates = pd.date_range("2020-01-02", periods=400, freq="B")
    columns = [f"F{i:02d}" for i in range(20)]
    factor = pd.DataFrame(
        rng.normal(size=(len(dates), len(columns))),
        index=dates,
        columns=columns,
    )
    returns = 0.01 * factor + pd.DataFrame(
        rng.normal(scale=0.02, size=factor.shape),
        index=dates,
        columns=columns,
    )
    factor.iloc[::11, :4] = np.nan
    returns.iloc[::13, 5:9] = np.nan

    expected_ic = ICTest(
        methods=["pearson"], decay_periods=[], forward_period=5
    ).run(factor, returns)
    expected_slopes = _vectorized_univariate_ols(
        factor, returns, min_stocks=10
    )
    actual = _joint_ic_ols_statistics(
        factor, returns, forward_period=5, min_stocks=10
    )

    assert np.isclose(actual["ic"], expected_ic.ic_mean)
    assert np.isclose(actual["ic_hac_t"], expected_ic.t_stat)
    assert np.isclose(actual["ir_nw"], expected_ic.ir_newey_west)
    assert np.isclose(
        actual["ic_pos_ratio"], (expected_ic.ic_series > 0).mean()
    )
    assert actual["ic_n"] == expected_ic.n_obs
    assert np.isclose(actual["ols_beta"], expected_slopes.mean())
    assert np.isclose(
        actual["ols_hac_t"], _newey_west_t_stat(expected_slopes, 5)
    )
    assert actual["ols_n"] == len(expected_slopes)


def test_joint_ic_ols_fails_closed_on_small_cross_section():
    dates = pd.date_range("2024-01-02", periods=20, freq="B")
    factor = pd.DataFrame(1.0, index=dates, columns=["A", "B"])
    returns = factor.copy()

    actual = _joint_ic_ols_statistics(
        factor, returns, forward_period=1, min_stocks=3
    )

    assert actual["ic_n"] == 0
    assert actual["ols_n"] == 0
    assert actual["ols_hac_t"] == 0.0
