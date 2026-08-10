from __future__ import annotations

import numpy as np
import pandas as pd

import factors.library.intraday as intraday


FACTOR_CLASSES = (
    intraday.IntradayAmtRatioEntropy60m20d,
    intraday.IntradayAmtRatioEntropyTrend20d,
    intraday.IntradayAmtRatioEntropyVolatility20d,
)


def _minute_panel(dates: pd.DatetimeIndex, last_day_scale: float = 1.0):
    timestamps = []
    for date in dates:
        timestamps.extend(date + pd.to_timedelta([9, 10, 11, 14], unit="h"))
    index = pd.DatetimeIndex(timestamps)
    close = pd.DataFrame(
        100.0 + np.linspace(0.0, 8.0, len(index)), index=index, columns=["AU"]
    )
    volume = pd.DataFrame(
        10.0 + (np.arange(len(index)) % 7), index=index, columns=["AU"]
    )
    last_mask = index.normalize() == dates[-1]
    close.loc[last_mask, "AU"] *= last_day_scale
    volume.loc[last_mask, "AU"] *= last_day_scale
    return {"close": close, "volume": volume}


def test_entropy_factors_are_daily_finite_and_causal(monkeypatch):
    dates = pd.bdate_range("2024-01-02", periods=30)
    active_panel = _minute_panel(dates)

    def fake_panel(data, requested_dates, universe, freq="1min", force_1min=False):
        assert force_1min is True
        return active_panel

    monkeypatch.setattr(intraday, "_get_minute_panel", fake_panel)
    baseline = {
        cls.name: cls().compute(None, dates, ["AU"])
        for cls in FACTOR_CLASSES
    }

    active_panel = _minute_panel(dates, last_day_scale=100.0)
    changed = {
        cls.name: cls().compute(None, dates, ["AU"])
        for cls in FACTOR_CLASSES
    }

    for name, result in baseline.items():
        assert result.index.equals(dates)
        assert result.columns.tolist() == ["AU"]
        assert np.isfinite(result.iloc[-1, 0])
        assert result.iloc[-1, 0] == changed[name].iloc[-1, 0]
