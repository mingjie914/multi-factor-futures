from __future__ import annotations

import numpy as np
import pandas as pd

from data.manager import DataManager, FrequencyDataProvider
from factors.spec_factor import (
    _resample_to_daily,
    compute_spec_factors_batch,
)


def test_intraday_spec_maps_night_session_then_lags_decision_day():
    dates = pd.DatetimeIndex(["2024-01-05", "2024-01-08", "2024-01-09"])
    minute = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0, 4.0]},
        index=pd.to_datetime([
            "2024-01-05 14:55",
            "2024-01-05 21:00",
            "2024-01-08 14:55",
            "2024-01-09 14:55",
        ]),
    )

    actual = _resample_to_daily(minute, dates)

    assert pd.isna(actual.loc["2024-01-05", "A"])
    assert actual.loc["2024-01-08", "A"] == 1.0
    assert actual.loc["2024-01-09", "A"] == 3.0


class _IntradaySource:
    def __init__(self, panels):
        self.panels = panels
        self.calls = []

    def fetch_price_at_frequency(
        self, tickers, start, end, fields, frequency="daily"
    ):
        self.calls.append((tuple(fields), frequency))
        return {
            field: self.panels[field].loc[start:end].reindex(columns=tickers)
            for field in fields
            if field in self.panels
        }


def _intraday_spec(frequency="5min"):
    return {
        "slug": "test_intraday_return_2p_raw",
        "base": "return",
        "transform": "raw",
        "frequency": frequency,
        "decision_lag_bars": 1,
        "params": {"window": 2},
    }


def test_intraday_spec_matching_frequency_returns_real_bar_signal():
    dates = pd.date_range("2024-01-02 09:00", periods=5, freq="5min")
    close = pd.DataFrame(
        {"A": [100.0, 101.0, 103.0, 102.0, 105.0]}, index=dates
    )
    panels = {
        field: close.copy()
        for field in ("open", "high", "low", "close", "volume")
    }
    source = _IntradaySource(panels)
    manager = DataManager(source, config={"cache": {"enabled": False}})
    provider = FrequencyDataProvider(
        manager, "5min", dates.min(), dates.max(), pd.Index(["A"])
    )

    result = compute_spec_factors_batch(
        [_intraday_spec()], provider, dates, pd.Index(["A"])
    )["test_intraday_return_2p_raw"]

    expected = close.pct_change(2, fill_method=None).shift(1)
    pd.testing.assert_frame_equal(result, expected)


def test_intraday_spec_incompatible_research_frequency_fails_closed():
    dates = pd.date_range("2024-01-02 09:00", periods=5, freq="15min")
    close = pd.DataFrame(
        {"A": np.arange(5, dtype=float) + 100.0}, index=dates
    )
    panels = {
        field: close.copy()
        for field in ("open", "high", "low", "close", "volume")
    }
    source = _IntradaySource(panels)
    manager = DataManager(source, config={"cache": {"enabled": False}})
    provider = FrequencyDataProvider(
        manager, "15min", dates.min(), dates.max(), pd.Index(["A"])
    )

    result = compute_spec_factors_batch(
        [_intraday_spec("5min")], provider, dates, pd.Index(["A"])
    )["test_intraday_return_2p_raw"]

    assert result.isna().all().all()
    assert source.calls == []
