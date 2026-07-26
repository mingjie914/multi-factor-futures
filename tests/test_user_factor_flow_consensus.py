from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.registry import get
from factors.user.flow_consensus import FACTOR_SPECS


class FrameProvider:
    def __init__(self, frames):
        self.frames = frames

    def get(self, field, dates, universe):
        return self.frames.get(field, pd.DataFrame()).reindex(
            index=dates, columns=universe
        )


@pytest.fixture(scope="module")
def sample_data():
    dates = pd.bdate_range("2020-01-01", periods=260)
    universe = pd.Index([
        "RB", "HC", "CU", "AL", "A", "M", "IF", "IC", "T", "TL",
    ])
    rng = np.random.default_rng(11)
    drift = np.linspace(-0.0005, 0.0008, len(universe))[None, :]
    returns = drift + rng.normal(0.0, 0.007, (len(dates), len(universe)))
    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(returns, axis=0)),
        index=dates,
        columns=universe,
    )
    open_price = close * np.exp(-0.35 * returns)
    spread = 0.008 + np.abs(returns)
    high = np.maximum(open_price, close) * (1.0 + spread)
    low = np.minimum(open_price, close) * (1.0 - spread)
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


def test_specs_are_unique_registered_and_have_metadata():
    assert len(FACTOR_SPECS) == 12
    assert len({spec.slug for spec in FACTOR_SPECS}) == 12
    for spec in FACTOR_SPECS:
        factor = get("factor", spec.slug)()
        assert factor.name == spec.slug
        assert factor.category == "liquidity_flow"
        assert factor.frequency == "daily"
        assert factor.dependencies() == ["open", "high", "low", "close", "volume"]
        assert factor.description


@pytest.mark.parametrize("spec", FACTOR_SPECS, ids=lambda spec: spec.slug)
def test_alignment_warmup_and_finite_values(spec, sample_data):
    dates, universe, frames = sample_data
    result = get("factor", spec.slug)().compute(
        FrameProvider(frames), dates, universe
    )
    assert result.index.equals(dates)
    assert result.columns.equals(universe)
    assert result.iloc[:10].isna().all().all()
    assert np.isfinite(result.iloc[140:].to_numpy()).any()
    assert not np.isinf(result.to_numpy()).any()


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
