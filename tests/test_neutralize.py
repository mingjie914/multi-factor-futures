from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.interfaces import ProcessingContext
from core.sectors import sector_matrix
from factors.processor import FactorProcessor
from processing.neutralize import NeutralizeStep


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
