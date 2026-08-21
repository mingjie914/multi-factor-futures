from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.period import PeriodContext, PeriodUnit
from data.manager import DataManager, FrequencyDataProvider
from data.market_quality import CloseDataQualityError
from factors.engine import FactorEngine
from pipeline.runner import _require_trade_calendar


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


def test_daily_forward_returns_fail_closed_on_unknown_gap():
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    close = pd.DataFrame({"A": [100.0, np.nan, 102.0, 103.0, 104.0]}, index=dates)

    class DailySource:
        def fetch_price(self, tickers, start, end, fields):
            return {"close": close.reindex(columns=tickers)}

    manager = DataManager(DailySource(), config={"cache": {"enabled": False}})

    with pytest.raises(CloseDataQualityError, match="unapproved close gaps"):
        manager.get_forward_returns(dates, pd.Index(["A"]), 1)


def test_daily_forward_returns_mark_audited_closure_on_calendar():
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    close = pd.DataFrame({"A": [100.0, np.nan, 102.0]}, index=dates)

    class DailySource:
        def fetch_price(self, tickers, start, end, fields):
            return {"close": close.reindex(columns=tickers)}

    manager = DataManager(
        DailySource(),
        config={
            "cache": {"enabled": False},
            "audited_nontrading_closes": {"A": [str(dates[1].date())]},
        },
    )
    forward = manager.get_forward_returns(dates, pd.Index(["A"]), 1)

    assert forward.loc[dates[0], "A"] == pytest.approx(0.0)
    assert forward.loc[dates[1], "A"] == pytest.approx(0.02)
    assert pd.isna(forward.loc[dates[2], "A"])


def test_factor_engine_rejects_factor_data_frequency_mismatch():
    dates = pd.date_range("2024-01-02 09:00", periods=3, freq="5min")
    close = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=dates)
    manager = DataManager(
        IntradaySource({"close": close}), config={"cache": {"enabled": False}}
    )
    provider = FrequencyDataProvider(
        manager, "5min", dates.min(), dates.max(), pd.Index(["A"])
    )

    class DailyFactor:
        name = "daily_only"
        frequency = "daily"

        def dependencies(self):
            return ["close"]

        def compute(self, data, requested_dates, universe):
            return data.get("close", requested_dates, universe)

    with pytest.raises(ValueError, match="frequency.*incompatible"):
        FactorEngine(provider).compute_factor(DailyFactor(), dates, pd.Index(["A"]))


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


def test_prefetch_dependency_override_skips_derived_fields():
    dates = pd.date_range("2024-01-02 09:00", periods=3, freq="5min")
    source = IntradaySource({})
    manager = DataManager(source, config={"cache": {"enabled": False}})
    provider = FrequencyDataProvider(
        manager, "5min", dates.min(), dates.max(), pd.Index(["A"])
    )

    class Factor:
        def dependencies(self):
            return ["derived_feature"]

        def prefetch_dependencies(self):
            return []

    provider.prefetch([Factor()], dates, pd.Index(["A"]))

    assert source.calls == []


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


def test_daily_prefetch_accepts_an_override_without_raw_dependencies():
    class Source:
        def fetch_price(self, *args, **kwargs):
            raise AssertionError("derived-only factor must not prefetch raw fields")

    class Factor:
        def prefetch_dependencies(self):
            return []

    manager = DataManager(Source(), config={"cache": {"enabled": False}})
    manager.prefetch(
        [Factor()], pd.date_range("2024-01-02", periods=2), pd.Index(["A"])
    )


def test_calendar_failures_are_not_silently_replaced_with_weekdays():
    class FailingCalendarSource:
        def fetch_calendar(self, start, end):
            raise OSError("calendar unavailable")

    manager = DataManager(
        FailingCalendarSource(), config={"cache": {"enabled": False}}
    )
    with pytest.raises(OSError, match="calendar unavailable"):
        manager.get_calendar("2026-01-01", "2026-01-31")

    with pytest.raises(RuntimeError, match="交易日历为空"):
        _require_trade_calendar([], "2026-01-01", "2026-01-31")


def test_price_source_failure_propagates_instead_of_returning_empty_panel():
    dates = pd.date_range("2026-01-01", periods=2, freq="B")

    class FailingPriceSource:
        def fetch_price(self, tickers, start, end, fields):
            raise OSError("price unavailable")

    manager = DataManager(
        FailingPriceSource(), config={"cache": {"enabled": False}}
    )
    with pytest.raises(OSError, match="price unavailable"):
        manager.get("close", dates, pd.Index(["RB"]))


def test_requested_price_field_cannot_be_silently_omitted():
    dates = pd.date_range("2024-01-02", periods=2, freq="B")

    class MissingFieldSource:
        def fetch_price(self, tickers, start, end, fields):
            return {}

    manager = DataManager(
        MissingFieldSource(), config={"cache": {"enabled": False}}
    )
    with pytest.raises(KeyError, match="omitted requested field 'close'"):
        manager.get("close", dates, pd.Index(["A"]))
