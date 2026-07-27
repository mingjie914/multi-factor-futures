from __future__ import annotations

import numpy as np
import pandas as pd

from core.period import PeriodContext, PeriodUnit
from data.manager import DataManager, FrequencyDataProvider


class IntradaySource:
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


def test_period_context_supports_one_and_five_minute_aliases():
    assert PeriodContext.from_string("1m").unit == PeriodUnit.MINUTE_1
    assert PeriodContext.from_string("5min").unit == PeriodUnit.MINUTE_5


def test_forward_returns_shift_each_instruments_own_valid_bars():
    dates = pd.date_range("2024-01-02 09:00", periods=5, freq="5min")
    close = pd.DataFrame(
        {
            "A": [100.0, 101.0, 102.0, 103.0, 104.0],
            "B": [200.0, np.nan, 202.0, np.nan, 204.0],
        },
        index=dates,
    )
    source = IntradaySource({"close": close})
    manager = DataManager(source, config={"cache": {"enabled": False}})
    provider = FrequencyDataProvider(
        manager, "5min", dates.min(), dates.max(), pd.Index(["A", "B"])
    )

    calendar = provider.get_calendar()
    forward = provider.get_forward_returns(calendar, pd.Index(["A", "B"]), 1)

    assert np.isclose(forward.loc[dates[0], "A"], 0.01)
    assert np.isclose(forward.loc[dates[0], "B"], 0.01)
    assert np.isclose(forward.loc[dates[2], "B"], 204.0 / 202.0 - 1.0)
    assert pd.isna(forward.loc[dates[1], "B"])


def test_prefetch_batches_frequency_dependencies():
    dates = pd.date_range("2024-01-02 09:00", periods=3, freq="5min")
    panels = {
        field: pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=dates)
        for field in ("close", "volume")
    }
    source = IntradaySource(panels)
    manager = DataManager(source, config={"cache": {"enabled": False}})
    provider = FrequencyDataProvider(
        manager, "5min", dates.min(), dates.max(), pd.Index(["A"])
    )

    class Factor:
        def dependencies(self):
            return ["close", "volume"]

    provider.prefetch([Factor()], dates, pd.Index(["A"]))
    provider.get("close", dates, pd.Index(["A"]))
    provider.get("volume", dates, pd.Index(["A"]))

    assert source.calls == [(('close', 'volume'), '5min')]


def test_daily_prefetch_reuses_fields_across_factor_batches():
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    calls = []

    class DailySource:
        def fetch_price(self, tickers, start, end, fields):
            calls.append(tuple(fields))
            return {
                field: pd.DataFrame(1.0, index=dates, columns=tickers)
                for field in fields
            }

    class Factor:
        def __init__(self, field):
            self.field = field

        def dependencies(self):
            return [self.field]

    manager = DataManager(
        DailySource(), config={"cache": {"enabled": False}}
    )
    universe = pd.Index(["A"])
    manager.prefetch([Factor("close")], dates, universe)
    manager.prefetch([Factor("volume")], dates, universe)
    manager.prefetch([Factor("close")], dates, universe)

    assert calls == [("close",), ("volume",)]
