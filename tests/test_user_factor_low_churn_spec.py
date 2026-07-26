from __future__ import annotations

import numpy as np
import pandas as pd

import factors.library  # noqa: F401 - triggers the project's SPEC registration path
from core.registry import get
from factors.spec_factor import compute_spec_factors_batch, is_spec_factor
from factors.specs import ALL_SPECS


class FrameProvider:
    def __init__(self, frames):
        self.frames = frames
        self.requested = []

    def get(self, field, dates, universe):
        self.requested.append(field)
        return self.frames.get(field, pd.DataFrame()).reindex(
            index=dates, columns=universe
        )


def _sample_frames():
    dates = pd.bdate_range("2020-01-01", periods=45)
    universe = pd.Index(["A", "B", "C"])
    step = np.arange(len(dates), dtype=float)[:, None]
    close = pd.DataFrame(
        100.0 + step * np.array([[0.8, -0.3, 0.2]]),
        index=dates,
        columns=universe,
    )
    volume = pd.DataFrame(
        1000.0 + step * np.array([[5.0, 3.0, 2.0]]),
        index=dates,
        columns=universe,
    )
    oi = pd.DataFrame(
        5000.0 + step * np.array([[12.0, -4.0, 3.0]]),
        index=dates,
        columns=universe,
    )
    frames = {
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": volume,
        "oi": oi,
    }
    return dates, universe, frames


def test_low_churn_specs_are_registered_and_declare_open_interest():
    specs = [spec for spec in ALL_SPECS if spec["base"] == "low_churn_trend"]
    assert len(specs) == 24
    assert len({spec["slug"] for spec in specs}) == 24
    for spec in specs:
        factor = get("factor", spec["slug"])()
        assert factor.category == "positioning_participation"
        assert factor.frequency == "daily"
        assert factor.dependencies() == [
            "open", "high", "low", "close", "volume", "oi"
        ]


def test_low_churn_spec_computes_finite_aligned_values():
    dates, universe, frames = _sample_frames()
    provider = FrameProvider(frames)
    result = get("factor", "low_churn_trend_10d_rank")().compute(
        provider, dates, universe
    )
    assert result.index.equals(dates)
    assert result.columns.equals(universe)
    assert provider.requested == ["open", "high", "low", "close", "volume", "oi"]
    assert np.isfinite(result.iloc[15:].to_numpy()).any()


def test_low_churn_spec_does_not_consume_current_bar():
    dates, universe, frames = _sample_frames()
    factor = get("factor", "low_churn_trend_5d_z")()
    baseline = factor.compute(FrameProvider(frames), dates, universe)
    revised_frames = {field: frame.copy() for field, frame in frames.items()}
    for frame in revised_frames.values():
        frame.iloc[-1] *= 100.0
    revised = factor.compute(FrameProvider(revised_frames), dates, universe)
    pd.testing.assert_series_equal(baseline.iloc[-1], revised.iloc[-1])


def test_low_churn_spec_fails_closed_without_open_interest():
    dates, universe, frames = _sample_frames()
    frames.pop("oi")
    result = get("factor", "low_churn_trend_20d_smooth")().compute(
        FrameProvider(frames), dates, universe
    )
    assert result.isna().all().all()


def test_batch_low_churn_spec_loads_declared_open_interest():
    dates, universe, frames = _sample_frames()
    provider = FrameProvider(frames)
    spec = next(
        spec for spec in ALL_SPECS
        if spec["slug"] == "low_churn_trend_10d_rank"
    )
    result = compute_spec_factors_batch([spec], provider, dates, universe)
    assert provider.requested == ["open", "high", "low", "close", "volume", "oi"]
    assert np.isfinite(result[spec["slug"]].iloc[15:].to_numpy()).any()


def test_batch_missing_open_interest_only_invalidates_dependent_specs():
    dates, universe, frames = _sample_frames()
    frames.pop("oi")
    provider = FrameProvider(frames)
    specs = [
        next(
            spec for spec in ALL_SPECS
            if spec["slug"] == "low_churn_trend_10d_z"
        ),
        next(
            spec for spec in ALL_SPECS
            if spec["slug"] == "log_momentum_10d_raw"
        ),
    ]

    result = compute_spec_factors_batch(specs, provider, dates, universe)

    assert result["low_churn_trend_10d_z"].isna().all().all()
    assert np.isfinite(
        result["log_momentum_10d_raw"].iloc[15:].to_numpy()
    ).any()
    assert provider.requested.count("oi") == 1


def test_intraday_batch_fails_closed_without_intraday_source():
    dates, universe, frames = _sample_frames()
    provider = FrameProvider(frames)
    spec = next(
        spec for spec in ALL_SPECS
        if spec.get("frequency") == "15min"
    )
    result = compute_spec_factors_batch([spec], provider, dates, universe)
    assert result[spec["slug"]].isna().all().all()
    assert provider.requested == []


def test_multiword_spec_transforms_use_the_batch_route():
    assert is_spec_factor("low_churn_trend_5d_vol_scaled")
    assert is_spec_factor("low_churn_trend_5d_confirm_volume")
    assert not is_spec_factor("low_churn_trend_5d")
