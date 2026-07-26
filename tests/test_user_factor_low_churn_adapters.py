from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import factors.library  # noqa: F401 - register built-ins before adapters
from core.registry import get
from factors.user.low_churn_adapters import FACTOR_SPECS, _rebalance_mask


class FrameProvider:
    def __init__(self, frames):
        self.frames = frames

    def get(self, field, dates, universe):
        if field not in self.frames:
            return pd.DataFrame()
        return self.frames[field].reindex(index=dates, columns=universe)


@pytest.fixture(scope="module")
def sample_data():
    dates = pd.bdate_range("2017-01-02", periods=620)
    universe = pd.Index(["RB", "HC", "CU", "AL", "A", "M", "IF", "IC"])
    rng = np.random.default_rng(41)
    returns = rng.normal(0.0001, 0.009, (len(dates), len(universe)))
    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(returns, axis=0)),
        index=dates,
        columns=universe,
    )
    open_price = close * np.exp(-0.3 * returns)
    width = 0.007 + np.abs(returns)
    high = np.maximum(open_price, close) * (1.0 + width)
    low = np.minimum(open_price, close) * (1.0 - width)
    volume = pd.DataFrame(
        rng.lognormal(8.0, 0.25, (len(dates), len(universe))),
        index=dates,
        columns=universe,
    )
    return dates, universe, {
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def test_specs_are_unique_registered_and_reference_existing_factors():
    assert len(FACTOR_SPECS) == 12
    assert len({spec.slug for spec in FACTOR_SPECS}) == 12
    for spec in FACTOR_SPECS:
        factor = get("factor", spec.slug)()
        base = get("factor", spec.base_factor)()
        assert factor.category == "low_churn_adapter"
        assert factor.dependencies() == base.dependencies()
        assert factor.frequency == "daily"
        assert factor.description


@pytest.mark.parametrize("spec", FACTOR_SPECS, ids=lambda spec: spec.slug)
def test_alignment_finite_values_and_declared_rebalance_days(spec, sample_data):
    dates, universe, frames = sample_data
    result = get("factor", spec.slug)().compute(
        FrameProvider(frames), dates, universe
    )

    assert result.index.equals(dates)
    assert result.columns.equals(universe)
    assert np.isfinite(result.iloc[300:].to_numpy()).any()
    assert not np.isinf(result.to_numpy()).any()

    changed = result.diff().abs().sum(axis=1).fillna(0.0) > 0.0
    assert changed.any()
    allowed = _rebalance_mask(dates, spec.schedule)
    assert allowed[changed].all()


@pytest.mark.parametrize("spec", FACTOR_SPECS, ids=lambda spec: spec.slug)
def test_current_bar_is_not_consumed(spec, sample_data):
    dates, universe, frames = sample_data
    factor = get("factor", spec.slug)()
    baseline = factor.compute(FrameProvider(frames), dates, universe)
    changed = {name: frame.copy() for name, frame in frames.items()}
    for frame in changed.values():
        frame.iloc[-1] *= 100.0
    revised = factor.compute(FrameProvider(changed), dates, universe)
    pd.testing.assert_series_equal(baseline.iloc[-1], revised.iloc[-1])


def test_missing_close_fails_closed(sample_data):
    dates, universe, frames = sample_data
    available = {name: frame for name, frame in frames.items() if name != "close"}
    result = get("factor", FACTOR_SPECS[0].slug)().compute(
        FrameProvider(available), dates, universe
    )
    assert result.isna().all().all()
