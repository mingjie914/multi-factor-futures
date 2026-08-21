"""Regression tests for causal settlement data and effective roll dates."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import factors.library.intraday as intraday


DATES = pd.date_range("2026-05-11", "2026-05-13", freq="D")


class ScheduleData:
    def __init__(self):
        self.fields = {
            "settle": pd.DataFrame(
                {"RB": [3200.0, 3190.0, 3200.0]}, index=DATES
            ),
            "oi": pd.DataFrame(
                {"RB": [100000.0, 90000.0, 110000.0]}, index=DATES
            ),
        }
        self.schedule = pd.DataFrame(
            {"RB": ["RB2605", "RB2606", "RB2606"]}, index=DATES
        )

    def get(self, field, dates, universe):
        return self.fields[field].reindex(index=dates, columns=universe)

    def get_contract_schedule(self, dates, universe):
        return self.schedule.reindex(index=dates, columns=universe)


def test_daily_settle_and_oi_follow_the_configured_causal_series():
    data = ScheduleData()

    settle = intraday._read_local_daily(data, DATES, ["RB"], "settle")
    oi = intraday._read_local_daily(data, DATES, ["RB"], "oi")

    assert settle.loc["2026-05-12", "RB"] == pytest.approx(3190.0)
    assert oi.loc["2026-05-12", "RB"] == pytest.approx(90000.0)


def test_rollover_calendar_uses_effective_contract_schedule():
    data = ScheduleData()

    calendar = intraday._get_rollover_calendar(data, DATES, ["RB"])

    assert list(calendar["RB"]) == [pd.Timestamp("2026-05-12")]


def test_rollover_calendar_is_scoped_to_each_root():
    data = ScheduleData()
    data.fields["settle"]["M"] = [2800.0, 2810.0, 2820.0]
    data.fields["oi"]["M"] = [50000.0, 51000.0, 52000.0]
    data.schedule["M"] = ["M2607", "M2607", "M2607"]

    calendar = intraday._get_rollover_calendar(data, DATES, ["RB", "M"])

    assert list(calendar["RB"]) == [pd.Timestamp("2026-05-12")]
    assert calendar["M"].empty


def test_rollover_schedule_failure_is_not_silently_replaced_by_no_rolls():
    class FailingData(ScheduleData):
        def get_contract_schedule(self, dates, universe):
            raise OSError("schedule unavailable")

    with pytest.raises(OSError, match="schedule unavailable"):
        intraday._get_rollover_calendar(FailingData(), DATES, ["RB"])


def test_term_curve_excludes_unheld_delivery_contracts_before_maturity_rank():
    timestamp = pd.Timestamp("2026-08-14 15:00")

    class Source:
        def fetch_contract_curve_at_frequency(self, *args, **kwargs):
            return pd.DataFrame([
                {
                    "trade_datetime": timestamp,
                    "root": "M",
                    "symbol": "M2608",
                    "close": 3061.0,
                    "position": 0.0,
                    "volume": 0.0,
                },
                {
                    "trade_datetime": timestamp,
                    "root": "M",
                    "symbol": "M2609",
                    "close": 3120.0,
                    "position": 722303.0,
                    "volume": 10.0,
                },
                {
                    "trade_datetime": timestamp,
                    "root": "M",
                    "symbol": "M2611",
                    "close": 3180.0,
                    "position": 100000.0,
                    "volume": 5.0,
                },
            ])

        @staticmethod
        def trading_session_index(index):
            return pd.DatetimeIndex(index)

    data = type("Data", (), {"source": Source()})()
    panel = intraday._read_local_term(
        data, pd.DatetimeIndex(["2026-08-14"]), ["M"]
    )

    assert panel["near_close"].loc[timestamp, "M"] == pytest.approx(3120.0)
    assert panel["far_close"].loc[timestamp, "M"] == pytest.approx(3180.0)
    assert panel["near_expiry"].loc[timestamp, "M"] == 202609


def test_term_curve_chunking_preserves_cross_boundary_rollover(monkeypatch):
    dates = pd.date_range("2026-08-10", periods=4, freq="D")

    class Source:
        def __init__(self):
            self.calls = []

        def fetch_contract_curve_at_frequency(
            self, tickers, start, end, fields, frequency
        ):
            self.calls.append((pd.Timestamp(start), pd.Timestamp(end)))
            rows = []
            for day in dates[(dates >= start) & (dates <= end)]:
                symbols = (
                    ("M2609", "M2611") if day < dates[2]
                    else ("M2611", "M2701")
                )
                for rank, symbol in enumerate(symbols):
                    rows.append({
                        "trade_datetime": day + pd.Timedelta(hours=15),
                        "root": "M",
                        "symbol": symbol,
                        "close": 3000.0 + rank,
                        "position": 100.0 - rank,
                        "volume": 10.0,
                    })
            return pd.DataFrame(rows)

        @staticmethod
        def trading_session_index(index):
            return pd.DatetimeIndex(index)

    source = Source()
    data = type("Data", (), {"source": source})()
    monkeypatch.setattr(intraday, "_TERM_FETCH_CHUNK_DAYS", 2)

    panel = intraday._read_local_term(data, dates, ["M"])

    assert len(source.calls) == 2
    assert panel["near_expiry"]["M"].tolist() == [202609, 202609, 202611, 202611]
    assert panel["rollover_flag"]["M"].tolist() == [0.0, 1.0, 1.0, 1.0]


def test_rollover_factor_applies_decision_lag(monkeypatch):
    dates = pd.date_range("2026-03-01", "2026-08-01", freq="B")
    universe = ["RB"]

    class Data:
        def get_contract_schedule(self, requested_dates, requested_universe):
            values = np.where(
                np.arange(len(requested_dates)) < 20, "RB2605", "RB2606"
            )
            return pd.DataFrame(
                {"RB": values}, index=pd.DatetimeIndex(requested_dates)
            )

    result = intraday.IntradayDaysToRollover20d().compute(
        Data(), dates, universe
    )

    assert isinstance(result, pd.DataFrame)
    assert pd.isna(result.iloc[0, 0])
    assert result.index.equals(dates)
