from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.registry import get
from factors.user.residual_structure import FACTOR_SPECS


class FrameProvider:
    def __init__(self, close=None):
        self.close = close

    def get(self, field, dates, universe):
        if field != "close" or self.close is None:
            return pd.DataFrame()
        return self.close.reindex(index=dates, columns=universe)


@pytest.fixture(scope="module")
def sample_data():
    dates = pd.bdate_range("2020-01-01", periods=320)
    universe = pd.Index([
        "RB", "HC", "CU", "AL", "A", "M", "IF", "IC", "T", "TL",
    ])
    rng = np.random.default_rng(42)
    market = rng.normal(0.0002, 0.006, size=(len(dates), 1))
    loadings = np.linspace(0.6, 1.4, len(universe))[None, :]
    residual = rng.normal(0.0, 0.004, size=(len(dates), len(universe)))
    returns = market * loadings + residual
    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(returns, axis=0)),
        index=dates,
        columns=universe,
    )
    return dates, universe, close


def test_specs_are_unique_registered_and_have_metadata():
    assert len(FACTOR_SPECS) == 12
    assert len({spec.slug for spec in FACTOR_SPECS}) == 12
    for spec in FACTOR_SPECS:
        factor = get("factor", spec.slug)()
        assert factor.name == spec.slug
        assert factor.category == "residual_structure"
        assert factor.frequency == "daily"
        assert factor.dependencies() == ["close"]
        assert factor.description


@pytest.mark.parametrize("spec", FACTOR_SPECS, ids=lambda spec: spec.slug)
def test_alignment_warmup_and_finite_values(spec, sample_data):
    dates, universe, close = sample_data
    result = get("factor", spec.slug)().compute(
        FrameProvider(close), dates, universe
    )
    assert result.index.equals(dates)
    assert result.columns.equals(universe)
    assert result.iloc[:20].isna().all().all()
    assert np.isfinite(result.iloc[160:].to_numpy()).any()
    assert not np.isinf(result.to_numpy()).any()


@pytest.mark.parametrize("spec", FACTOR_SPECS, ids=lambda spec: spec.slug)
def test_current_close_is_not_consumed(spec, sample_data):
    dates, universe, close = sample_data
    factor = get("factor", spec.slug)()
    baseline = factor.compute(FrameProvider(close), dates, universe)
    changed = close.copy()
    changed.iloc[-1] *= 100.0
    revised = factor.compute(FrameProvider(changed), dates, universe)
    pd.testing.assert_series_equal(baseline.iloc[-1], revised.iloc[-1])


def test_missing_close_fails_closed(sample_data):
    dates, universe, _ = sample_data
    result = get("factor", FACTOR_SPECS[0].slug)().compute(
        FrameProvider(), dates, universe
    )
    assert result.isna().all().all()


def test_sector_residual_requires_multiple_instruments():
    dates = pd.bdate_range("2020-01-01", periods=160)
    universe = pd.Index(["RB"])
    close = pd.DataFrame(np.linspace(100.0, 130.0, len(dates)), index=dates, columns=universe)
    result = get("factor", "sector_residual_momentum_20d")().compute(
        FrameProvider(close), dates, universe
    )
    assert result.isna().all().all()
