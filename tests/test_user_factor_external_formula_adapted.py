from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.registry import get
from factors.user.external_formula_adapted import FACTOR_SPECS


class FrameProvider:
    def __init__(self, frames):
        self.frames = frames

    def get(self, field, dates, universe):
        return self.frames.get(field, pd.DataFrame()).reindex(
            index=dates, columns=universe
        )


@pytest.fixture(scope="module")
def sample_data():
    dates = pd.bdate_range("2018-01-01", periods=520)
    universe = pd.Index(["RB", "HC", "CU", "AL", "A", "M", "IF", "IC"])
    rng = np.random.default_rng(31)
    drift = np.linspace(-0.0003, 0.0006, len(universe))[None, :]
    returns = drift + rng.normal(0.0, 0.009, (len(dates), len(universe)))
    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(returns, axis=0)),
        index=dates,
        columns=universe,
    )
    open_price = close * np.exp(-0.35 * returns)
    width = 0.006 + np.abs(returns)
    high = np.maximum(open_price, close) * (1.0 + width)
    low = np.minimum(open_price, close) * (1.0 - width)
    volume = pd.DataFrame(
        rng.lognormal(8.0, 0.3, (len(dates), len(universe))),
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


def test_specs_are_unique_registered_and_record_provenance():
    assert len(FACTOR_SPECS) == 12
    assert len({spec.slug for spec in FACTOR_SPECS}) == 12
    for spec in FACTOR_SPECS:
        factor = get("factor", spec.slug)()
        assert factor.name == spec.slug
        assert factor.category == "external_formula_adapted"
        assert factor.frequency == "daily"
        assert factor.dependencies() == ["open", "high", "low", "close", "volume"]
        assert "adapted from" in factor.description
        assert spec.source


@pytest.mark.parametrize("spec", FACTOR_SPECS, ids=lambda spec: spec.slug)
def test_alignment_warmup_finite_values_and_weekly_hold(spec, sample_data):
    dates, universe, frames = sample_data
    result = get("factor", spec.slug)().compute(
        FrameProvider(frames), dates, universe
    )

    assert result.index.equals(dates)
    assert result.columns.equals(universe)
    assert result.iloc[:15].isna().all().all()
    assert np.isfinite(result.iloc[250:].to_numpy()).any()
    assert not np.isinf(result.to_numpy()).any()

    changed = result.diff().abs().sum(axis=1).fillna(0.0) > 0.0
    assert changed.any()
    assert (result.index[changed].weekday == 4).all()


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


@pytest.mark.parametrize("missing", ["open", "high", "low", "close", "volume"])
def test_missing_dependency_fails_closed(missing, sample_data):
    dates, universe, frames = sample_data
    available = {name: frame for name, frame in frames.items() if name != missing}
    result = get("factor", FACTOR_SPECS[0].slug)().compute(
        FrameProvider(available), dates, universe
    )
    assert result.isna().all().all()

