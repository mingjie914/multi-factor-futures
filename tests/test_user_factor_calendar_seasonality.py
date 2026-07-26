from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.registry import get
from factors.user.calendar_seasonality import FACTOR_SPECS


class FrameProvider:
    def __init__(self, frames):
        self.frames = frames

    def get(self, field, dates, universe):
        return self.frames.get(field, pd.DataFrame()).reindex(
            index=dates, columns=universe
        )


@pytest.fixture(scope="module")
def sample_data():
    dates = pd.bdate_range("2012-01-02", "2021-12-31")
    universe = pd.Index(["A", "B", "C"])
    month = dates.month.to_numpy(dtype=float)[:, None]
    year_step = (dates.year.to_numpy(dtype=float) - 2012.0)[:, None]
    seasonal = np.sin(month / 12.0 * 2.0 * np.pi)
    base = 100.0 + year_step * np.array([[2.0, 1.0, 3.0]])
    close = pd.DataFrame(
        base + seasonal * np.array([[8.0, -5.0, 3.0]]),
        index=dates,
        columns=universe,
    )
    open_price = close * (1.0 - 0.001 * seasonal)
    high = np.maximum(open_price, close) * 1.01
    low = np.minimum(open_price, close) * 0.99
    volume = pd.DataFrame(
        1000.0 + month * np.array([[20.0, 10.0, 5.0]]),
        index=dates,
        columns=universe,
    )
    frames = {
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }
    return dates, universe, frames


def test_all_specs_are_unique_registered_and_declare_metadata():
    assert len(FACTOR_SPECS) == 12
    assert len({spec.slug for spec in FACTOR_SPECS}) == 12
    for spec in FACTOR_SPECS:
        factor = get("factor", spec.slug)()
        assert factor.name == spec.slug
        assert factor.category == "seasonality"
        assert factor.frequency == "daily"
        assert factor.dependencies() == list(spec.dependencies)
        assert factor.description


@pytest.mark.parametrize("spec", FACTOR_SPECS, ids=lambda spec: spec.slug)
def test_alignment_warmup_and_finite_values(spec, sample_data):
    dates, universe, frames = sample_data
    result = get("factor", spec.slug)().compute(
        FrameProvider(frames), dates, universe
    )
    assert result.index.equals(dates)
    assert result.columns.equals(universe)
    warmup_end = dates[dates.year <= 2012 + spec.years - 1]
    assert result.loc[warmup_end].isna().all().all()
    assert np.isfinite(result.loc[dates.year >= 2018].to_numpy()).any()
    assert not np.isinf(result.to_numpy()).any()


@pytest.mark.parametrize("spec", FACTOR_SPECS, ids=lambda spec: spec.slug)
def test_current_bar_changes_do_not_change_current_signal(spec, sample_data):
    dates, universe, frames = sample_data
    factor = get("factor", spec.slug)()
    baseline = factor.compute(FrameProvider(frames), dates, universe)
    changed = {name: frame.copy() for name, frame in frames.items()}
    for frame in changed.values():
        frame.iloc[-1] *= 100.0
    revised = factor.compute(FrameProvider(changed), dates, universe)
    pd.testing.assert_series_equal(baseline.iloc[-1], revised.iloc[-1])


@pytest.mark.parametrize("missing", ["open", "high", "low", "close", "volume"])
def test_missing_declared_dependency_fails_closed(missing, sample_data):
    dates, universe, frames = sample_data
    spec = next(spec for spec in FACTOR_SPECS if missing in spec.dependencies)
    available = {name: frame for name, frame in frames.items() if name != missing}
    result = get("factor", spec.slug)().compute(
        FrameProvider(available), dates, universe
    )
    assert result.isna().all().all()


def test_current_year_month_is_excluded_from_history(sample_data):
    dates, universe, frames = sample_data
    factor = get("factor", "calendar_return_mean_3y")()
    baseline = factor.compute(FrameProvider(frames), dates, universe)
    changed = {name: frame.copy() for name, frame in frames.items()}
    current_month = (dates.year == 2021) & (dates.month == 12)
    changed["close"].loc[current_month] *= np.linspace(1.0, 4.0, current_month.sum())[:, None]
    revised = factor.compute(FrameProvider(changed), dates, universe)
    pd.testing.assert_frame_equal(
        baseline.loc[current_month], revised.loc[current_month]
    )
