from __future__ import annotations

import pandas as pd
import pytest

from scripts.contract_lots import _contract_snapshot, lots_for


class _FormalSource:
    @staticmethod
    def fetch_latest_trade_date():
        return pd.Timestamp("2026-08-14")

    @staticmethod
    def fetch_calendar(start, end):
        del start, end
        return pd.DatetimeIndex(["2026-08-14"])

    @staticmethod
    def fetch_contract_schedule(roots, start, end):
        del start, end
        values = {"RB": "RB2610", "AU": "AU2612"}
        return pd.DataFrame(
            {root: [values[root]] for root in roots},
            index=pd.DatetimeIndex(["2026-08-14"]),
        )

    @staticmethod
    def fetch_contract_curve_at_frequency(
        roots, start, end, fields, frequency="daily"
    ):
        del start, end, fields
        assert frequency == "daily"
        rows = {"RB": ("RB2610", 3300.0), "AU": ("AU2612", 800.0)}
        return pd.DataFrame([
            {
                "trade_datetime": pd.Timestamp("2026-08-14"),
                "trade_date": pd.Timestamp("2026-08-14"),
                "root": root,
                "symbol": rows[root][0],
                "close": rows[root][1],
            }
            for root in roots
        ])


def test_contract_snapshot_uses_formal_schedule_and_exact_contract_close():
    date, snapshot = _contract_snapshot(_FormalSource(), ["RB", "AU"])

    assert date == pd.Timestamp("2026-08-14")
    assert snapshot == {
        "RB": {"contract": "RB2610", "price": 3300.0},
        "AU": {"contract": "AU2612", "price": 800.0},
    }


def test_contract_snapshot_fails_when_scheduled_contract_quote_is_missing():
    class MissingQuoteSource(_FormalSource):
        @staticmethod
        def fetch_contract_curve_at_frequency(*args, **kwargs):
            frame = _FormalSource.fetch_contract_curve_at_frequency(*args, **kwargs)
            return frame.loc[frame["root"].ne("RB")]

    with pytest.raises(RuntimeError, match="RB2610"):
        _contract_snapshot(MissingQuoteSource(), ["RB", "AU"])


def test_lot_conversion_keeps_concrete_contract_in_handoff():
    _, snapshot = _contract_snapshot(_FormalSource(), ["RB"])

    rows = lots_for({"RB": 0.50}, 1_000_000.0, snapshot)

    assert rows[0]["contract"] == "RB2610"
    assert rows[0]["lots"] == 15
