from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.registry import get
from factors.user.contract_curve_positioning import FACTOR_SPECS


class FrameProvider:
    def __init__(self, frames):
        self.frames = frames

    def get(self, field, dates, universe):
        return self.frames.get(field, pd.DataFrame()).reindex(
            index=dates, columns=universe
        )


@pytest.fixture
def sample_data():
    dates = pd.bdate_range("2020-01-01", periods=55)
    universe = pd.Index(["A", "B", "C"])
    step = np.arange(len(dates), dtype=float)[:, None]
    return dates, universe, {
        "close": pd.DataFrame(
            100.0 + step * np.array([[0.8, -0.3, 0.2]]),
            index=dates, columns=universe,
        ),
        "oi": pd.DataFrame(
            1000.0 + step * np.array([[5.0, 2.0, -1.0]]),
            index=dates, columns=universe,
        ),
        "curve_top2_oi": pd.DataFrame(
            1800.0 + step * np.array([[9.0, 3.0, -1.0]]),
            index=dates, columns=universe,
        ),
        "curve_total_oi": pd.DataFrame(
            2200.0 + step * np.array([[12.0, 4.0, -2.0]]),
            index=dates, columns=universe,
        ),
        "curve_oi_breadth": pd.DataFrame(
            np.tile(np.array([[0.8, 0.7, 0.4]]), (len(dates), 1)),
            index=dates, columns=universe,
        ),
    }


def test_all_contract_curve_specs_are_registered():
    assert len(FACTOR_SPECS) == 12
    assert len({spec.slug for spec in FACTOR_SPECS}) == 12
    for spec in FACTOR_SPECS:
        factor = get("factor", spec.slug)()
        assert factor.category == "positioning_curve"
        assert factor.frequency == "daily"
        assert spec.scope in factor.dependencies()


@pytest.mark.parametrize("spec", FACTOR_SPECS, ids=lambda spec: spec.slug)
def test_contract_curve_factor_alignment_and_finite_values(spec, sample_data):
    dates, universe, frames = sample_data
    result = get("factor", spec.slug)().compute(
        FrameProvider(frames), dates, universe
    )
    assert result.index.equals(dates)
    assert result.columns.equals(universe)
    assert result.iloc[: spec.window].isna().all().all()
    assert np.isfinite(result.iloc[spec.window + 3 :].to_numpy()).any()


@pytest.mark.parametrize("spec", FACTOR_SPECS, ids=lambda spec: spec.slug)
def test_contract_curve_factor_does_not_consume_current_bar(spec, sample_data):
    dates, universe, frames = sample_data
    factor = get("factor", spec.slug)()
    baseline = factor.compute(FrameProvider(frames), dates, universe)
    changed = {name: frame.copy() for name, frame in frames.items()}
    for frame in changed.values():
        frame.iloc[-1] *= 10.0
    revised = factor.compute(FrameProvider(changed), dates, universe)
    pd.testing.assert_series_equal(baseline.iloc[-1], revised.iloc[-1])


def test_missing_curve_field_fails_closed(sample_data):
    dates, universe, frames = sample_data
    frames = dict(frames)
    frames.pop("curve_total_oi")
    result = get("factor", "curve_total_oi_growth_5b")().compute(
        FrameProvider(frames), dates, universe
    )
    assert result.isna().all().all()


def test_intraday_lags_and_windows_follow_each_tickers_valid_bars():
    dates = pd.date_range("2024-01-02 09:00", periods=7, freq="15min")
    universe = pd.Index(["A", "B"])
    total_oi = pd.DataFrame(
        {
            "A": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
            "B": [200.0, np.nan, 202.0, np.nan, 204.0, np.nan, 206.0],
        },
        index=dates,
    )
    result = get("factor", "curve_total_oi_growth_1b")().compute(
        FrameProvider({"curve_total_oi": total_oi}), dates, universe
    )

    assert np.isclose(result.loc[dates[2], "A"], np.log(101.0 / 100.0))
    assert np.isclose(result.loc[dates[4], "B"], np.log(202.0 / 200.0))
    assert pd.isna(result.loc[dates[3], "B"])
