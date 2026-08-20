from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.interfaces import ProcessingContext
from core.sectors import sector_matrix
from factors.processor import FactorProcessor, build_processing_steps
from processing.neutralize import NeutralizeStep
from processing.standardize import StandardizeStep


def test_neutralize_produces_zero_sector_means_each_date():
    dates = pd.bdate_range("2024-01-02", periods=8)
    universe = pd.Index(["RB", "HC", "I", "CU", "AL", "SN"])
    factor = pd.DataFrame(
        np.arange(len(dates) * len(universe), dtype=float).reshape(
            len(dates), len(universe)
        ),
        index=dates,
        columns=universe,
    )
    context = ProcessingContext(
        data=None,
        dates=dates,
        universe=universe,
        industry=sector_matrix(dates, universe),
    )

    result = NeutralizeStep(min_group_size=2).transform(factor, context)

    assert np.allclose(result[["RB", "HC", "I"]].mean(axis=1), 0.0)
    assert np.allclose(result[["CU", "AL", "SN"]].mean(axis=1), 0.0)


def test_neutralize_supports_dynamic_sector_labels_on_pandas_3():
    dates = pd.bdate_range("2024-01-02", periods=2)
    columns = ["A", "B", "C", "D"]
    factor = pd.DataFrame(
        [[1.0, 3.0, 10.0, 14.0], [2.0, 8.0, 12.0, 16.0]],
        index=dates,
        columns=columns,
    )
    industry = pd.DataFrame(
        [["x", "x", "y", "y"], ["x", "y", "x", "y"]],
        index=dates,
        columns=columns,
    )
    context = ProcessingContext(
        data=None,
        dates=dates,
        universe=pd.Index(columns),
        industry=industry,
    )

    result = NeutralizeStep(min_group_size=2).transform(factor, context)

    for date in dates:
        for label in ("x", "y"):
            members = industry.columns[industry.loc[date] == label]
            assert result.loc[date, members].mean() == pytest.approx(0.0)


def test_neutralize_fails_closed_when_industry_is_missing():
    factor = pd.DataFrame([[1.0, 2.0]], columns=["RB", "HC"])
    step = NeutralizeStep(missing_policy="error")

    with pytest.raises(RuntimeError, match="industry labels"):
        step.transform(factor, None)
    with pytest.raises(RuntimeError, match="industry labels"):
        FactorProcessor([step]).process(
            factor,
            ProcessingContext(
                data=None,
                dates=factor.index,
                universe=factor.columns,
            ),
        )


def test_eligibility_is_applied_before_sector_neutralization():
    dates = pd.DatetimeIndex(["2024-01-02"])
    universe = pd.Index(["RB", "HC", "I"])
    factor = pd.DataFrame([[1.0, 3.0, 1000.0]], index=dates, columns=universe)
    eligibility = pd.DataFrame(
        [[True, True, False]], index=dates, columns=universe
    )
    context = ProcessingContext(
        data=None,
        dates=dates,
        universe=universe,
        industry=sector_matrix(dates, universe),
        eligibility=eligibility,
    )

    result = FactorProcessor([
        NeutralizeStep(min_group_size=2)
    ]).process(factor, context)

    assert result.loc[dates[0], "RB"] == pytest.approx(-1.0)
    assert result.loc[dates[0], "HC"] == pytest.approx(1.0)
    assert pd.isna(result.loc[dates[0], "I"])


def test_processing_configuration_and_runtime_fail_closed():
    with pytest.raises(KeyError, match="not registered"):
        build_processing_steps([{"type": "typo_step", "params": {}}])

    class BrokenStep:
        name = "broken"

        def transform(self, factor, context):
            raise ValueError("bad configuration")

    factor = pd.DataFrame([[1.0]], columns=["RB"])
    context = ProcessingContext(
        data=None,
        dates=factor.index,
        universe=factor.columns,
    )
    with pytest.raises(RuntimeError, match="broken"):
        FactorProcessor([BrokenStep()]).process(factor, context)


def test_standardize_preserves_unobserved_values():
    factor = pd.DataFrame(
        [[1.0, 1.0, np.nan], [1.0, 3.0, np.nan]],
        columns=["RB", "HC", "I"],
    )
    result = StandardizeStep().transform(factor)

    assert result.loc[0, ["RB", "HC"]].eq(0.0).all()
    assert result.loc[1, "RB"] < 0 < result.loc[1, "HC"]
    assert result["I"].isna().all()


def test_standardize_global_zscore_uses_all_observed_values():
    factor = pd.DataFrame([[0.0, 10.0, 20.0], [0.0, 10.0, 30.0]])
    values = factor.to_numpy()
    expected = (values - values.mean()) / values.std(ddof=1)

    result = StandardizeStep(by_date=False).transform(factor)

    np.testing.assert_allclose(result.to_numpy(), expected)
