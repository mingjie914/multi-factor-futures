from __future__ import annotations

from decimal import Decimal

import pytest

from trading.sizing import size_targets


DATA_DATE = "2026-09-15"


def _account(
    *,
    equity: float = 1_000_000.0,
    available: float | None = None,
    max_margin_ratio: float = 1.0,
    max_gross_exposure: float = 1_000_000.0,
    max_abs_net_exposure: float = 1_000_000.0,
    require_one_lot: bool = False,
) -> dict:
    return {
        "equity": equity,
        "available": equity if available is None else available,
        "margin_used": 0.0,
        "frozen_margin": 0.0,
        "reserve": 0.0,
        "max_margin_ratio": max_margin_ratio,
        "max_gross_exposure": max_gross_exposure,
        "max_abs_net_exposure": max_abs_net_exposure,
        "require_one_lot": require_one_lot,
    }


def _spec(
    contract: str,
    *,
    multiplier: float | Decimal = 20.0,
    margin_long: float = 0.1,
    margin_short: float = 0.1,
    as_of: str = DATA_DATE,
) -> dict:
    return {
        "contract": contract,
        "exchange": "SHFE",
        "multiplier": multiplier,
        "margin_long": margin_long,
        "margin_short": margin_short,
        "tick": 1.0,
        "as_of": as_of,
        "source": "synthetic-complete-unit-test",
        "verified": True,
    }


def _weight(
    root: str,
    contract: str,
    weight: float | Decimal,
    *,
    close: float | Decimal = 10.0,
    data_date: str = DATA_DATE,
) -> dict:
    return {
        "root": root,
        "contract": contract,
        "close": close,
        "weight": weight,
        "data_date": data_date,
    }


def _route(
    strategy_id: str,
    account_id: str,
    basis: str,
    *,
    amount: float | Decimal | None = None,
    units: int | None = None,
) -> dict:
    row = {
        "strategy_id": strategy_id,
        "account_id": account_id,
        "capital_basis": basis,
    }
    if basis == "complete_unit":
        if units is not None:
            row["units"] = units
    elif amount is not None:
        row["amount"] = amount
    return row


def _run(
    weight_sets: dict[str, list[dict]],
    specs: list[dict],
    accounts: dict[str, dict],
    routes: list[dict],
    *,
    rounding: str = "half_up",
) -> dict:
    return size_targets(weight_sets, specs, accounts, routes, rounding=rounding)


def _targets(result: dict, account_id: str = "A") -> dict[str, dict]:
    return {row["contract"]: row for row in result["accounts"][account_id]["targets"]}


@pytest.mark.parametrize(("units", "capital", "small_raw", "large_raw", "small_lots", "large_lots"),
                         [(1, 60.31, 1.8, 60.3, 2, 60), (2, 120.62, 3.6, 120.6, 4, 121)])
def test_complete_unit_scales_before_rounding(units, capital, small_raw, large_raw, small_lots, large_lots):
    # The binding ratio is 60.301, so Cmin is 60.301 while the executable
    # reference capital is rounded UP to 60.31.  The other two legs then have
    # raw lots 1.8 and 60.3, exercising ordinary half-up lot rounding.
    rows = [
        _weight("AA", "AA2610", Decimal("0.9"), close=Decimal("1")),
        _weight("AB", "AB2610", Decimal("1"), close=Decimal("1")),
        _weight("AC", "AC2610", Decimal("1"), close=Decimal("1")),
    ]
    specs = [
        _spec("AA2610", multiplier=Decimal("30.155")),
        _spec("AB2610", multiplier=Decimal("60.301")),
        _spec("AC2610", multiplier=Decimal("60.31") / Decimal("60.3")),
    ]

    result = _run(
        {"S": rows},
        specs,
        {"A": _account()},
        [_route("S", "A", "complete_unit", units=units)],
    )

    account = result["accounts"]["A"]
    targets = _targets(result)
    route = account["routes"][0]
    assert route["capital_basis"] == "complete_unit"
    assert route["amount"] is None
    assert route["units"] == units
    assert route["minimum_complete_reference_capital"] == pytest.approx(60.301)
    assert route["reference_capital"] == pytest.approx(capital)
    assert targets["AA2610"]["raw_lots"] == pytest.approx(small_raw)
    assert targets["AC2610"]["raw_lots"] == pytest.approx(large_raw)
    # Cent ceiling makes the binding leg just over one theoretical lot.
    assert targets["AB2610"]["raw_lots"] == pytest.approx(capital / 60.301)
    assert targets["AA2610"]["target_lots"] == small_lots
    assert targets["AC2610"]["target_lots"] == large_lots
    assert targets["AB2610"]["target_lots"] == units
    assert account["completeness"]["S"] == {
        "selected_count": 3,
        "min_abs_raw_lots": pytest.approx(capital / 60.301),
        "below_one_contracts": [],
        "complete": True,
    }


def test_complete_unit_uses_max_ratio_when_smallest_weight_is_not_minimum_lot():
    rows = [
        _weight("AA", "AA2610", Decimal("0.1")),
        _weight("AB", "AB2610", Decimal("0.5")),
    ]
    result = _run(
        {"S": rows},
        [
            _spec("AA2610", multiplier=10.0),  # one-lot notional 100; ratio 1,000
            _spec("AB2610", multiplier=100.0),  # one-lot notional 1,000; ratio 2,000
        ],
        {"A": _account()},
        [_route("S", "A", "complete_unit", units=1)],
    )

    targets = _targets(result)
    route = result["accounts"]["A"]["routes"][0]
    assert route["minimum_complete_reference_capital"] == pytest.approx(2_000.0)
    assert route["reference_capital"] == pytest.approx(2_000.0)
    assert targets["AA2610"]["raw_lots"] == pytest.approx(2.0)
    assert targets["AB2610"]["raw_lots"] == pytest.approx(1.0)
    assert result["accounts"]["A"]["completeness"]["S"]["min_abs_raw_lots"] == 1


def test_complete_unit_keeps_all_twenty_instruments_and_zero_weight_row():
    roots = [
        "AA",
        "AB",
        "AC",
        "AD",
        "AE",
        "AF",
        "AG",
        "AH",
        "AI",
        "AJ",
        "AK",
        "AL",
        "AM",
        "AN",
        "AO",
        "AP",
        "AQ",
        "AR",
        "AS",
        "AT",
        "AU",
    ]
    rows = [_weight(root, f"{root}2610", 0 if i == 20 else 0.05) for i, root in enumerate(roots)]
    result = _run(
        {"S": rows},
        [_spec(f"{root}2610", multiplier=10.0) for root in roots],
        {"A": _account()},
        [_route("S", "A", "complete_unit", units=1)],
    )

    account = result["accounts"]["A"]
    targets = _targets(result)
    assert len(targets) == 21
    assert len(account["attribution"]) == 21
    assert sum(abs(row["raw_lots"]) >= 1 for row in targets.values()) == 20
    assert targets["AU2610"]["raw_lots"] == 0
    assert targets["AU2610"]["target_lots"] == 0
    assert account["completeness"]["S"] == {
        "selected_count": 20,
        "min_abs_raw_lots": pytest.approx(1.0),
        "below_one_contracts": [],
        "complete": True,
    }


def test_complete_unit_units_are_independent_per_account_and_equity_is_unchanged():
    rows = [_weight("AA", "AA2610", Decimal("0.5"))]
    result = _run(
        {"S": rows},
        [_spec("AA2610", multiplier=10.0)],
        {"A": _account(equity=500.0), "B": _account(equity=50.0)},
        [
            _route("S", "A", "complete_unit", units=1),
            _route("S", "B", "complete_unit", units=2),
        ],
    )

    for account_id, units, capital, target_notional, equity in (
        ("A", 1, 200.0, 100.0, 500.0),
        ("B", 2, 400.0, 200.0, 50.0),
    ):
        account = result["accounts"][account_id]
        route = account["routes"][0]
        assert route["units"] == units
        assert route["reference_capital"] == pytest.approx(capital)
        assert route["minimum_complete_reference_capital"] == pytest.approx(200.0)
        assert route["amount"] is None
        assert account["account"]["equity"] == equity
        assert _targets(result, account_id)["AA2610"]["target_notional"] == pytest.approx(
            target_notional
        )
        assert not any("EQUITY_ROUTE_BUDGET" in blocker for blocker in account["blockers"])


def test_tiny_weight_keeps_complete_unit_targets_but_risk_limits_block():
    result = _run(
        {
            "S": [
                _weight("AA", "AA2610", Decimal("0.0001")),
                _weight("AB", "AB2610", Decimal("1"), close=1.0),
            ]
        },
        [
            _spec("AA2610", multiplier=20.0),  # one-lot notional 200; Cmin=2,000,000
            _spec("AB2610", multiplier=20.0),
        ],
        {"A": _account(equity=1_000.0, max_gross_exposure=0.1)},
        [_route("S", "A", "complete_unit", units=1)],
    )

    account = result["accounts"]["A"]
    targets = _targets(result)
    assert account["routes"][0]["reference_capital"] == pytest.approx(2_000_000.0)
    assert targets["AA2610"]["target_lots"] == 1
    assert targets["AB2610"]["target_lots"] == 100_000
    assert account["summary"]["gross_notional"] == pytest.approx(2_000_200.0)
    assert any("GROSS_EXPOSURE" in blocker for blocker in account["blockers"])
    assert any("MARGIN" in blocker for blocker in account["blockers"])
    assert account["tradable"] is False


def test_reference_basis_uses_fixed_amount_without_equity_route_budget():
    result = _run(
        {"S": [_weight("AA", "AA2610", Decimal("0.5"))]},
        [_spec("AA2610", multiplier=10.0)],
        {"A": _account(equity=100.0)},
        [_route("S", "A", "reference", amount=300.0)],
    )

    account = result["accounts"]["A"]
    route = account["routes"][0]
    target = _targets(result)["AA2610"]
    assert route["capital_basis"] == "reference"
    assert route["amount"] == pytest.approx(300.0)
    assert route["reference_capital"] == pytest.approx(300.0)
    assert route["units"] is None
    assert route["minimum_complete_reference_capital"] == pytest.approx(200.0)
    assert target["target_notional"] == pytest.approx(150.0)
    assert target["raw_lots"] == pytest.approx(1.5)
    assert "EQUITY_ROUTE_BUDGET" not in " ".join(account["blockers"])
    assert account["account"]["equity"] == pytest.approx(100.0)
    assert account["tradable"] is True


def test_reference_amount_remains_fixed_when_next_day_weight_is_smaller():
    specs = [_spec("AA2610", multiplier=10.0)]
    account = {"A": _account(equity=1_000.0, require_one_lot=True)}
    route = [_route("S", "A", "reference", amount=100.0)]
    first = _run({"S": [_weight("AA", "AA2610", 1.0)]}, specs, account, route)
    next_day = _run({"S": [_weight("AA", "AA2610", 0.5, data_date="2026-09-16")]},
                    [_spec("AA2610", multiplier=10.0, as_of="2026-09-16")], account, route)

    assert first["accounts"]["A"]["routes"][0]["reference_capital"] == pytest.approx(100.0)
    assert next_day["accounts"]["A"]["routes"][0]["reference_capital"] == pytest.approx(100.0)
    assert first["accounts"]["A"]["completeness"]["S"]["complete"] is True
    assert next_day["accounts"]["A"]["completeness"]["S"] == {
        "selected_count": 1,
        "min_abs_raw_lots": pytest.approx(0.5),
        "below_one_contracts": ["AA2610"],
        "complete": False,
    }
    assert _targets(next_day)["AA2610"]["target_lots"] == 1
    assert any("REQUIRE_ONE_LOT" in blocker for blocker in next_day["accounts"]["A"]["blockers"])


@pytest.mark.parametrize("weight", [Decimal("0.5"), Decimal("0.9")])
def test_require_one_lot_blocks_pre_round_raw_half_or_nine_even_when_rounds_one(weight):
    result = _run(
        {"S": [_weight("AA", "AA2610", weight)]},
        [_spec("AA2610", multiplier=10.0)],
        {"A": _account(equity=1_000.0, require_one_lot=True)},
        [_route("S", "A", "reference", amount=100.0)],
    )

    account = result["accounts"]["A"]
    assert _targets(result)["AA2610"]["target_lots"] == 1
    assert account["completeness"]["S"]["min_abs_raw_lots"] == pytest.approx(float(weight))
    assert account["completeness"]["S"]["complete"] is False
    assert account["completeness"]["S"]["below_one_contracts"] == ["AA2610"]
    assert any("REQUIRE_ONE_LOT" in blocker for blocker in account["blockers"])


def test_require_one_lot_merges_multiple_routes_of_one_strategy_before_checking():
    result = _run(
        {"S": [_weight("AA", "AA2610", 1.0)]},
        [_spec("AA2610", multiplier=10.0)],
        {"A": _account(require_one_lot=True)},
        [
            _route("S", "A", "reference", amount=40.0),
            _route("S", "A", "notional", amount=60.0),
        ],
    )

    account = result["accounts"]["A"]
    target = _targets(result)["AA2610"]
    assert [route["reference_capital"] for route in account["routes"]] == [40, 60]
    assert [route["amount"] for route in account["routes"]] == [40, 60]
    assert target["raw_lots"] == pytest.approx(1.0)
    assert target["target_lots"] == 1
    assert account["completeness"]["S"] == {
        "selected_count": 1,
        "min_abs_raw_lots": pytest.approx(1.0),
        "below_one_contracts": [],
        "complete": True,
    }
    assert not any("REQUIRE_ONE_LOT" in blocker for blocker in account["blockers"])
    assert account["tradable"] is True


@pytest.mark.parametrize("second_weight", [-1.0, 1.0])
def test_incomplete_strategies_cannot_pass_by_cross_strategy_netting(second_weight):
    result = _run(
        {
            "LONG": [_weight("AA", "AA2610", 1.0)],
            "SECOND": [_weight("AA", "AA2610", second_weight)],
        },
        [_spec("AA2610", multiplier=10.0)],
        {"A": _account(require_one_lot=True)},
        [
            _route("LONG", "A", "reference", amount=50.0),
            _route("SECOND", "A", "reference", amount=50.0),
        ],
    )

    account = result["accounts"]["A"]
    target = _targets(result)["AA2610"]
    assert target["target_notional"] == 50 + 50 * second_weight
    assert target["target_lots"] == (0 if second_weight < 0 else 1)
    for strategy_id in ("LONG", "SECOND"):
        assert account["completeness"][strategy_id]["min_abs_raw_lots"] == pytest.approx(0.5)
        assert account["completeness"][strategy_id]["complete"] is False
        assert account["completeness"][strategy_id]["below_one_contracts"] == ["AA2610"]
    assert sum("REQUIRE_ONE_LOT" in blocker for blocker in account["blockers"]) == 2
    assert account["summary"]["net_notional"] == 50 + 50 * second_weight
    assert account["tradable"] is False


@pytest.mark.parametrize("short_amount", [100, 140])
def test_complete_opposite_strategies_can_net_below_one_without_lot_amplification(short_amount):
    result = _run(
        {
            "LONG": [_weight("AA", "AA2610", 1.0)],
            "SHORT": [_weight("AA", "AA2610", -1.0)],
        },
        [_spec("AA2610", multiplier=10.0)],
        {"A": _account(require_one_lot=True)},
        [
            _route("LONG", "A", "reference", amount=100.0),
            _route("SHORT", "A", "reference", amount=short_amount),
        ],
    )

    account = result["accounts"]["A"]
    target = _targets(result)["AA2610"]
    assert account["completeness"]["LONG"]["complete"] is True
    assert account["completeness"]["SHORT"]["complete"] is True
    assert target["target_notional"] == 100 - short_amount
    assert target["target_lots"] == 0
    assert account["summary"]["net_notional"] == 0
    assert not any("REQUIRE_ONE_LOT" in blocker for blocker in account["blockers"])
    assert account["tradable"] is True


@pytest.mark.parametrize("bad_units", [0, -1, 1.5, True, None, "1"])
def test_complete_unit_requires_positive_integer_units(bad_units):
    route = {
        "strategy_id": "S",
        "account_id": "A",
        "capital_basis": "complete_unit",
    }
    if bad_units is not None:
        route["units"] = bad_units
    with pytest.raises(ValueError, match="units"):
        _run(
            {"S": [_weight("AA", "AA2610", 1.0)]},
            [_spec("AA2610")],
            {"A": _account()},
            [route],
        )


def test_complete_unit_forbids_amount_and_other_bases_require_amount_without_units():
    with pytest.raises(ValueError, match="uses units, not amount"):
        _run(
            {"S": [_weight("AA", "AA2610", 1.0)]},
            [_spec("AA2610")],
            {"A": _account()},
            [
                {
                    "strategy_id": "S",
                    "account_id": "A",
                    "capital_basis": "complete_unit",
                    "units": 1,
                    "amount": 100,
                }
            ],
        )

    with pytest.raises(ValueError, match="amount"):
        _run(
            {"S": [_weight("AA", "AA2610", 1.0)]},
            [_spec("AA2610")],
            {"A": _account()},
            [_route("S", "A", "reference")],
        )

    with pytest.raises(ValueError, match="requires complete_unit"):
        _run(
            {"S": [_weight("AA", "AA2610", 1.0)]},
            [_spec("AA2610")],
            {"A": _account()},
            [
                {
                    "strategy_id": "S",
                    "account_id": "A",
                    "capital_basis": "reference",
                    "amount": 100,
                    "units": 1,
                }
            ],
        )
