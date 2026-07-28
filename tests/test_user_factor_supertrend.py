from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.registry import get
from factors.user.supertrend import FACTOR_SPECS


class FrameProvider:
    def __init__(self, frames):
        self.frames = frames

    def get(self, field, dates, universe):
        return self.frames.get(field, pd.DataFrame()).reindex(
            index=dates, columns=universe
        )


@pytest.fixture
def sample_data():
    dates = pd.bdate_range("2020-01-01", periods=100)
    universe = pd.Index(["UP", "DOWN", "SWITCH"])
    up = np.linspace(100.0, 160.0, len(dates))
    down = np.linspace(160.0, 100.0, len(dates))
    switch = np.r_[
        np.linspace(100.0, 135.0, 50),
        np.linspace(135.0, 75.0, 50),
    ]
    close = pd.DataFrame(
        {"UP": up, "DOWN": down, "SWITCH": switch}, index=dates
    )
    spread = pd.DataFrame(2.0, index=dates, columns=universe)
    frames = {
        "high": close + spread,
        "low": close - spread,
        "close": close,
    }
    return dates, universe, frames


def test_supertrend_factors_are_registered_with_fixed_metadata():
    assert len(FACTOR_SPECS) == 3
    assert {spec.slug for spec in FACTOR_SPECS} == {
        "supertrend_state_20_2",
        "supertrend_distance_20_2",
        "supertrend_flip_20_2",
    }
    for spec in FACTOR_SPECS:
        factor = get("factor", spec.slug)()
        assert factor.name == spec.slug
        assert factor.category == "trend"
        assert factor.frequency == "daily"
        assert factor.dependencies() == ["high", "low", "close"]
        assert factor.description


@pytest.mark.parametrize("spec", FACTOR_SPECS, ids=lambda spec: spec.slug)
def test_supertrend_alignment_warmup_and_finite_values(spec, sample_data):
    dates, universe, frames = sample_data
    result = get("factor", spec.slug)().compute(
        FrameProvider(frames), dates, universe
    )
    assert result.index.equals(dates)
    assert result.columns.equals(universe)
    assert result.iloc[:20].isna().all().all()
    assert np.isfinite(result.iloc[25:].to_numpy()).any()
    assert not np.isinf(result.to_numpy()).any()


@pytest.mark.parametrize("spec", FACTOR_SPECS, ids=lambda spec: spec.slug)
def test_supertrend_does_not_consume_current_bar(spec, sample_data):
    dates, universe, frames = sample_data
    factor = get("factor", spec.slug)()
    baseline = factor.compute(FrameProvider(frames), dates, universe)
    changed = {field: frame.copy() for field, frame in frames.items()}
    changed["high"].iloc[-1] *= 5.0
    changed["low"].iloc[-1] *= 0.2
    changed["close"].iloc[-1] *= 3.0
    revised = factor.compute(FrameProvider(changed), dates, universe)
    pd.testing.assert_series_equal(baseline.iloc[-1], revised.iloc[-1])


@pytest.mark.parametrize("missing", ["high", "low", "close"])
def test_supertrend_fails_closed_when_input_is_missing(missing, sample_data):
    dates, universe, frames = sample_data
    available = {field: frame for field, frame in frames.items() if field != missing}
    result = get("factor", "supertrend_distance_20_2")().compute(
        FrameProvider(available), dates, universe
    )
    assert result.isna().all().all()


def test_supertrend_state_distance_and_flip_are_coherent(sample_data):
    dates, universe, frames = sample_data
    provider = FrameProvider(frames)
    state = get("factor", "supertrend_state_20_2")().compute(
        provider, dates, universe
    )
    distance = get("factor", "supertrend_distance_20_2")().compute(
        provider, dates, universe
    )
    flip = get("factor", "supertrend_flip_20_2")().compute(
        provider, dates, universe
    )

    valid = state.notna() & distance.notna()
    assert ((np.sign(distance.where(valid)) == state.where(valid)) | ~valid).all().all()
    assert set(np.unique(state.stack().dropna())) <= {-1.0, 1.0}
    assert set(np.unique(flip.stack().dropna())) <= {-1.0, 0.0, 1.0}
    assert -1.0 in set(flip["SWITCH"].dropna())
    assert distance.abs().max().max() <= 5.0


def test_supertrend_factor_fails_closed_on_mixed_ohlc_scales(sample_data):
    dates, universe, frames = sample_data
    changed = {field: frame.copy() for field, frame in frames.items()}
    changed["close"][universe[0]] *= 1.30
    result = get("factor", "supertrend_state_20_2")().compute(
        FrameProvider(changed), dates, universe
    )
    assert result[universe[0]].isna().all()
