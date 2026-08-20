from __future__ import annotations

import pandas as pd
import pytest

import factors.library.intraday as intraday


class _DailyProvider:
    def __init__(self, fields):
        self.fields = fields

    def get(self, field, dates, universe):
        return self.fields[field].reindex(index=dates, columns=universe)


def test_daily_reader_uses_configured_provider_field_without_local_fallback():
    dates = pd.DatetimeIndex(["2026-05-11"])
    settle = pd.DataFrame({"RB": [3205.0]}, index=dates)
    oi = pd.DataFrame({"RB": [1000.0]}, index=dates)
    data = _DailyProvider({"settle": settle, "oi": oi})

    pd.testing.assert_frame_equal(
        intraday._read_local_daily(data, dates, ["RB"], "settle"), settle
    )
    pd.testing.assert_frame_equal(
        intraday._read_local_daily(data, dates, ["RB"], "oi"), oi
    )


def test_daily_reader_propagates_configured_source_failure():
    class FailingProvider:
        def get(self, field, dates, universe):
            raise OSError("broken configured parquet")

    with pytest.raises(OSError, match="broken configured parquet"):
        intraday._read_local_daily(
            FailingProvider(), pd.DatetimeIndex(["2026-05-11"]), ["RB"]
        )


def test_minute_panel_uses_source_session_clock_and_normalizes_oi_name():
    wall_clock = pd.DatetimeIndex([
        "2026-05-10 21:00", "2026-05-11 09:00"
    ])
    close = pd.DataFrame({"RB": [3200.0, 3210.0]}, index=wall_clock)
    oi = pd.DataFrame({"RB": [1000.0, 1100.0]}, index=wall_clock)

    class Source:
        cache_namespace = "test-session-source"

        def fetch_price_at_frequency(
            self, tickers, start, end, fields, frequency="daily"
        ):
            return {"close": close, "oi": oi}

        def trading_session_index(self, index):
            return pd.DatetimeIndex([
                "2026-05-11 03:00", "2026-05-11 15:00"
            ])

    class Data:
        source = Source()

    intraday._PANEL_CACHE.clear()
    panel = intraday._get_minute_panel(
        Data(), pd.DatetimeIndex(["2026-05-11"]), ["RB"], freq="5min"
    )

    assert "oi" not in panel
    assert "position" in panel
    assert panel["close"].index.normalize().unique().tolist() == [
        pd.Timestamp("2026-05-11")
    ]
    assert panel["position"].iloc[-1, 0] == 1100.0


def test_minute_panel_does_not_switch_sources_after_configured_failure():
    class Source:
        cache_namespace = "failing-source"

        def fetch_price_at_frequency(self, *args, **kwargs):
            raise OSError("broken configured parquet")

    class Data:
        source = Source()

        def get_at_frequency(self, *args, **kwargs):
            raise AssertionError("must not switch to provider fallback")

    intraday._PANEL_CACHE.clear()
    with pytest.raises(RuntimeError, match="拒绝切换到其他数据源"):
        intraday._get_minute_panel(
            Data(), pd.DatetimeIndex(["2026-05-11"]), ["RB"], freq="5min"
        )


def test_minute_panel_cache_is_isolated_between_source_instances():
    dates = pd.DatetimeIndex(["2026-05-11"])

    class Source:
        cache_namespace = "same-source-type"

        def __init__(self, value):
            self.value = value

        def fetch_price_at_frequency(self, tickers, start, end, fields, frequency):
            index = pd.DatetimeIndex(["2026-05-11 09:00"])
            return {"close": pd.DataFrame({tickers[0]: [self.value]}, index=index)}

    class Data:
        def __init__(self, value):
            self.source = Source(value)

    intraday._PANEL_CACHE.clear()
    first = intraday._get_minute_panel(Data(100.0), dates, ["RB"], freq="5min")
    second = intraday._get_minute_panel(Data(200.0), dates, ["RB"], freq="5min")

    assert first["close"].iloc[0, 0] == 100.0
    assert second["close"].iloc[0, 0] == 200.0


def test_no_frequency_provider_does_not_read_an_implicit_machine_path():
    intraday._PANEL_CACHE.clear()
    assert intraday._get_minute_panel(
        object(), pd.DatetimeIndex(["2026-05-11"]), ["RB"], freq="5min"
    ) == {}


def test_transient_intraday_caches_can_be_released_after_research_batch():
    intraday._PANEL_CACHE["panel"] = object()
    intraday._TERM_CACHE["term"] = object()
    intraday._SEAT_CACHE["seat"] = object()
    intraday._SEAT_DETAIL_CACHE["seat_detail"] = object()

    intraday.clear_transient_data_caches()

    assert not intraday._PANEL_CACHE
    assert not intraday._TERM_CACHE
    assert not intraday._SEAT_CACHE
    assert not intraday._SEAT_DETAIL_CACHE
