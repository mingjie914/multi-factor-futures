from __future__ import annotations

import numpy as np
import pandas as pd

from data.mysql_source import MySQLSource


def _source():
    return MySQLSource({
        "dominant_lag_days": 1,
        "tables": {
            "intraday_5m": {
                "table_name": "ths_data_5minute",
                "columns": {
                    "datetime": "time",
                    "ticker": "code",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                    "amount": "amt",
                    "oi": "oi",
                },
            }
        },
    })


def _contract_rows():
    rows = []
    for day, values in (
        ("2024-01-02", {"A2401": (100.0, 1000), "A2405": (200.0, 100)}),
        ("2024-01-03", {"A2401": (102.0, 100), "A2405": (202.0, 1200)}),
        ("2024-01-04", {"A2401": (104.0, 80), "A2405": (206.0, 1300)}),
    ):
        for minute in (5, 10):
            for symbol, (close, oi) in values.items():
                rows.append({
                    "trade_datetime": pd.Timestamp(day) + pd.Timedelta(hours=9, minutes=minute),
                    "symbol": symbol,
                    "open": close - 1.0,
                    "high": close + 1.0,
                    "low": close - 2.0,
                    "close": close + minute / 100.0,
                    "volume": 10 + minute,
                    "amount": (10 + minute) * close,
                    "oi": oi + minute,
                })
    return pd.DataFrame(rows)


def test_mysql_five_minute_uses_lagged_dominant_and_causal_roll(monkeypatch):
    source = _source()
    rows = _contract_rows()
    monkeypatch.setattr(source, "_read_sql", lambda sql: rows.copy())
    monkeypatch.setattr(
        source,
        "fetch_calendar",
        lambda start, end: pd.date_range("2024-01-02", "2024-01-05", freq="B"),
    )

    panel = source.fetch_price_at_frequency(
        ["A"], "2024-01-03", "2024-01-04", ["close", "oi"], "5min"
    )

    assert panel["oi"].loc["2024-01-03 09:10", "A"] == 110
    expected = 206.10 * (102.10 / 202.10)
    assert panel["close"].loc["2024-01-04 09:10", "A"] == expected


def test_mysql_five_minute_curve_uses_all_contracts(monkeypatch):
    source = _source()
    rows = _contract_rows()
    monkeypatch.setattr(source, "_read_sql", lambda sql: rows.copy())
    monkeypatch.setattr(
        source,
        "fetch_calendar",
        lambda start, end: pd.date_range("2024-01-02", "2024-01-05", freq="B"),
    )

    panel = source.fetch_price_at_frequency(
        ["A"], "2024-01-03", "2024-01-03",
        ["curve_total_oi", "curve_top2_oi", "curve_oi_concentration"],
        "5min",
    )

    assert panel["curve_total_oi"].loc["2024-01-03 09:10", "A"] == 1320
    assert panel["curve_top2_oi"].loc["2024-01-03 09:10", "A"] == 1320
    assert np.isclose(
        panel["curve_oi_concentration"].loc["2024-01-03 09:10", "A"], 1.0
    )


def test_mysql_night_bar_maps_to_next_trading_day(monkeypatch):
    source = _source()
    monkeypatch.setattr(
        source,
        "fetch_calendar",
        lambda start, end: pd.DatetimeIndex(["2024-01-05", "2024-01-08"]),
    )
    timestamps = pd.Series(pd.to_datetime([
        "2024-01-04 21:05", "2024-01-05 09:05", "2024-01-07 21:05"
    ]))

    actual = source._assign_trading_dates(timestamps)

    assert actual.tolist() == [
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-01-08"),
    ]
