from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.registry import get
from factors.user.positioning_participation import FACTOR_SPECS


class FrameProvider:
    def __init__(self, frames):
        self.frames = frames

    def get(self, field, dates, universe):
        return self.frames.get(field, pd.DataFrame()).reindex(
            index=dates, columns=universe
        )


@pytest.fixture
def sample_data():
    dates = pd.bdate_range("2020-01-01", periods=45)
    universe = pd.Index(["A", "B", "C"])
    step = np.arange(len(dates), dtype=float)[:, None]
    close = pd.DataFrame(
        100.0 + step * np.array([[0.8, -0.35, 0.2]]),
        index=dates,
        columns=universe,
    )
    volume = pd.DataFrame(
        1000.0 + step * np.array([[5.0, 3.0, 2.0]]),
        index=dates,
        columns=universe,
    )
    oi = pd.DataFrame(
        5000.0 + step * np.array([[12.0, -6.0, 3.0]]),
        index=dates,
        columns=universe,
    )
    return dates, universe, {"close": close, "volume": volume, "oi": oi}


def test_all_specs_are_registered_with_declared_metadata():
    assert len(FACTOR_SPECS) == 12
    assert len({spec.slug for spec in FACTOR_SPECS}) == 12
    for spec in FACTOR_SPECS:
        factor = get("factor", spec.slug)()
        assert factor.name == spec.slug
        assert factor.frequency == "daily"
        assert factor.category == "positioning_participation"
        assert factor.dependencies() == ["close", "volume", "oi"]
        assert factor.description


@pytest.mark.parametrize("spec", FACTOR_SPECS, ids=lambda spec: spec.slug)
def test_factor_alignment_warmup_and_finite_values(spec, sample_data):
    dates, universe, frames = sample_data
    result = get("factor", spec.slug)().compute(
        FrameProvider(frames), dates, universe
    )
    assert result.index.equals(dates)
    assert result.columns.equals(universe)
    assert result.iloc[: spec.window].isna().all().all()
    assert np.isfinite(result.iloc[spec.window + 2 :].to_numpy()).any()


@pytest.mark.parametrize("spec", FACTOR_SPECS, ids=lambda spec: spec.slug)
def test_factor_does_not_consume_current_bar(spec, sample_data):
    dates, universe, frames = sample_data
    factor = get("factor", spec.slug)()
    baseline = factor.compute(FrameProvider(frames), dates, universe)
    changed = {name: frame.copy() for name, frame in frames.items()}
    for frame in changed.values():
        frame.iloc[-1] = frame.iloc[-1] * 100.0
    revised = factor.compute(FrameProvider(changed), dates, universe)
    pd.testing.assert_series_equal(baseline.iloc[-1], revised.iloc[-1])


@pytest.mark.parametrize("missing", ["close", "volume", "oi"])
def test_missing_dependency_returns_invalid_matrix(missing, sample_data):
    dates, universe, frames = sample_data
    available = {name: frame for name, frame in frames.items() if name != missing}
    result = get("factor", "opening_flow_pressure_10d")().compute(
        FrameProvider(available), dates, universe
    )
    assert result.index.equals(dates)
    assert result.columns.equals(universe)
    assert result.isna().all().all()


def test_zero_volume_and_open_interest_do_not_create_infinities(sample_data):
    dates, universe, frames = sample_data
    damaged = {name: frame.copy() for name, frame in frames.items()}
    damaged["volume"].iloc[15:20, 0] = 0.0
    damaged["oi"].iloc[20:25, 1] = 0.0
    result = get("factor", "low_churn_trend_5d")().compute(
        FrameProvider(damaged), dates, universe
    )
    assert not np.isinf(result.to_numpy()).any()
