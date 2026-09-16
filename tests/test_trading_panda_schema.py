import pytest

from trading.panda import normalize_snapshot


def _snapshot(row=None):
    return {
        "account": {
            "accountId": "sdk-schema-test",
            "ready": True,
            "tradeDate": "20260915",
            "totalProfit": 5_000_000,
            "availableFunds": 5_000_000,
            "margin": 0,
            "frozenCapital": 0,
        },
        "positions": [] if row is None else [row],
        "openOrders": [],
    }


def _policy():
    return {
        "position_schema": "panda_sdk_0_1_19",
        "known_contracts": {
            "SR701": {"contract": "SR2701", "root": "SR", "exchange": "CZCE"},
            "CU2610": {"contract": "CU2610", "root": "CU", "exchange": "SHFE"},
        },
        "exchange_aliases": {"CZC": "CZCE"},
    }


def _row(*, direction="long", td=0, yd=3, contract="SR701", exchange="CZC"):
    return {
        "contractCode": contract,
        "directionText": direction,
        "position": td + yd,
        "tdPosition": td,
        "ydPosition": yd,
        "closable": td + yd,
        "exchange": exchange,
    }


@pytest.mark.parametrize(
    ("direction", "td", "yd"),
    [("多", 0, 3), ("LONG", 3, 0), ("空头", 2, 4)],
)
def test_sdk_schema_maps_documented_position_fields(direction, td, yd):
    result = normalize_snapshot(
        _snapshot(_row(direction=direction, td=td, yd=yd)), _policy()
    )

    assert result["positions"] == [{
        "root": "SR",
        "contract": "SR2701",
        "exchange": "CZCE",
        "side": "long" if direction in {"多", "LONG"} else "short",
        "volume": td + yd,
        "today": td,
        "yesterday": yd,
        "available_today": td,
        "available_yesterday": yd,
        "broker_contract": "SR701",
    }]


def test_sdk_schema_allows_empty_positions_without_contract_map():
    result = normalize_snapshot(
        _snapshot(), {"position_schema": "panda_sdk_0_1_19"}
    )

    assert result["positions"] == []


def test_sdk_position_with_exchange_suffix_keeps_actual_broker_code():
    policy = _policy()
    policy["known_contracts"]["M2701"] = {"contract": "M2701", "root": "M", "exchange": "DCE"}
    row = _row(contract="M2701.DCE", direction="多头", td=1, yd=0)
    row.pop("exchange")  # Actual Panda position response has the venue in contractCode.
    result = normalize_snapshot(_snapshot(row), policy)["positions"][0]
    assert result["contract"] == "M2701" and result["broker_contract"] == "M2701.DCE"
    assert result["today"] == result["available_today"] == 1
    with pytest.raises(ValueError, match="exchange suffix"):
        normalize_snapshot(_snapshot({**row, "contractCode": "M2701.SHFE"}), policy)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (_row(contract="UNKNOWN1"), "known_contracts"),
        (_row(direction="buy"), "directionText"),
        ({**_row(), "closable": 2}, "closable"),
        ({**_row(), "position": 4}, "tdPosition"),
        ({**_row(), "exchange": "SHFE"}, "exchange"),
    ],
)
def test_sdk_schema_rejects_unverified_or_ambiguous_position(row, message):
    with pytest.raises(ValueError, match=message):
        normalize_snapshot(_snapshot(row), _policy())
