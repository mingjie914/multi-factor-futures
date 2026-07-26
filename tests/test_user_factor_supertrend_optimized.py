from __future__ import annotations

import numpy as np
import pandas as pd

from core.registry import get


class FrameProvider:
    def __init__(self, frames):
        self.frames = frames

    def get(self, field, dates, universe):
        return self.frames.get(field, pd.DataFrame()).reindex(
            index=dates, columns=universe
        )


def _sample_data():
    dates = pd.bdate_range("2020-01-01", periods=80)
    universe = pd.Index(["A", "B"])
    close = pd.DataFrame(
        {
            "A": np.linspace(100.0, 150.0, len(dates)),
            "B": np.r_[np.linspace(120.0, 145.0, 40), np.linspace(145.0, 90.0, 40)],
        },
        index=dates,
    )
    frames = {"high": close + 2.0, "low": close - 2.0, "close": close}
    return dates, universe, frames


def test_smoothed_supertrend_distance_registration_and_values():
    dates, universe, frames = _sample_data()
    factor = get("factor", "supertrend_distance_smooth3_20_2")()
    assert factor.category == "trend"
    assert factor.frequency == "daily"
    assert factor.dependencies() == ["high", "low", "close"]
    result = factor.compute(FrameProvider(frames), dates, universe)
    assert result.index.equals(dates)
    assert result.columns.equals(universe)
    assert result.iloc[:22].isna().all().all()
    assert np.isfinite(result.iloc[25:].to_numpy()).any()
    assert result.abs().max().max() <= 5.0


def test_smoothed_supertrend_distance_is_point_in_time():
    dates, universe, frames = _sample_data()
    factor = get("factor", "supertrend_distance_smooth3_20_2")()
    baseline = factor.compute(FrameProvider(frames), dates, universe)
    changed = {field: frame.copy() for field, frame in frames.items()}
    changed["high"].iloc[-1] *= 4.0
    changed["low"].iloc[-1] *= 0.25
    changed["close"].iloc[-1] *= 2.0
    revised = factor.compute(FrameProvider(changed), dates, universe)
    pd.testing.assert_series_equal(baseline.iloc[-1], revised.iloc[-1])


def test_smoothed_supertrend_distance_fails_closed():
    dates, universe, frames = _sample_data()
    frames.pop("low")
    result = get("factor", "supertrend_distance_smooth3_20_2")().compute(
        FrameProvider(frames), dates, universe
    )
    assert result.isna().all().all()
