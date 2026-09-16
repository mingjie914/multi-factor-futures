from __future__ import annotations

from copy import deepcopy

import pytest

from trading.sizing import size_targets


def _account(
    *,
    equity: float = 1_000_000.0,
    available: float | None = None,
    margin_used: float = 0.0,
    frozen_margin: float = 0.0,
    reserve: float = 0.0,
    max_margin_ratio: float = 1.0,
    max_gross_exposure: float = 10_000_000.0,
    max_abs_net_exposure: float = 10_000_000.0,
    require_one_lot: bool = False,
) -> dict:
    return {
        "equity": equity,
        "available": equity if available is None else available,
        "margin_used": margin_used,
        "frozen_margin": frozen_margin,
        "reserve": reserve,
        "max_margin_ratio": max_margin_ratio,
        "max_gross_exposure": max_gross_exposure,
        "max_abs_net_exposure": max_abs_net_exposure,
        "require_one_lot": require_one_lot,
    }


def _spec(
    contract: str,
    *,
    multiplier: float = 20.0,
    margin_long: float = 0.1,
    margin_short: float = 0.1,
    as_of: str = "2026-09-15",
    source: str = "daily-spec",
    verified: bool = True,
    max_order_lots: int | None = None,
    max_position_lots: int | None = None,
) -> dict:
    row = {
        "contract": contract,
        "exchange": "SHFE",
        "multiplier": multiplier,
        "margin_long": margin_long,
        "margin_short": margin_short,
        "tick": 1.0,
        "as_of": as_of,
        "source": source,
        "verified": verified,
    }
    if max_order_lots is not None:
        row["max_order_lots"] = max_order_lots
    if max_position_lots is not None:
        row["max_position_lots"] = max_position_lots
    return row


def _weight(root: str, contract: str, weight: float, *, close: float = 10.0) -> dict:
    return {
        "root": root,
        "contract": contract,
        "close": close,
        "weight": weight,
        "data_date": "2026-09-15",
    }


def _one_strategy(
    rows: list[dict],
    *,
    account: dict | None = None,
    strategy_id: str = "S",
    account_id: str = "A",
    basis: str = "equity",
    amount: float = 1_000.0,
    specs: list[dict] | None = None,
    rounding: str = "half_up",
) -> dict:
    if specs is None:
        specs = [_spec(row["contract"]) for row in rows]
    return size_targets(
        {strategy_id: rows},
        specs,
        {account_id: _account() if account is None else account},
        [
            {
                "strategy_id": strategy_id,
                "account_id": account_id,
                "capital_basis": basis,
                "amount": amount,
            }
        ],
        rounding=rounding,
    )


def _targets(result: dict, account_id: str = "A") -> dict[str, dict]:
    return {row["contract"]: row for row in result["accounts"][account_id]["targets"]}


def test_half_up_is_symmetric_for_positive_and_negative_half_and_one_point_five():
    rows = [
        _weight("RB", "RB2610", 0.5),
        _weight("CU", "CU2610", 1.5),
        _weight("AL", "AL2610", -0.5),
        _weight("AG", "AG2610", -1.5),
    ]

    result = _one_strategy(rows, amount=200.0)
    targets = _targets(result)

    assert [targets[c]["raw_lots"] for c in ("RB2610", "CU2610", "AL2610", "AG2610")] == [
        0.5,
        1.5,
        -0.5,
        -1.5,
    ]
    assert [targets[c]["target_lots"] for c in ("RB2610", "CU2610", "AL2610", "AG2610")] == [
        1,
        2,
        -1,
        -2,
    ]

    toward_zero = _targets(_one_strategy(rows, amount=200.0, rounding="toward_zero"))
    assert [toward_zero[c]["target_lots"] for c in ("RB2610", "CU2610", "AL2610", "AG2610")] == [
        0,
        1,
        0,
        -1,
    ]


def test_same_account_merges_opposite_strategies_before_rounding_and_keeps_zero_target():
    weights = {
        "LONG": [_weight("RB", "RB2610", 1.0)],
        "SHORT": [_weight("RB", "RB2610", -1.0)],
    }
    routes = [
        {"strategy_id": "LONG", "account_id": "A", "capital_basis": "equity", "amount": 1_000.0},
        {"strategy_id": "LONG", "account_id": "B", "capital_basis": "equity", "amount": 500.0},
        {"strategy_id": "SHORT", "account_id": "A", "capital_basis": "equity", "amount": 1_000.0},
        {"strategy_id": "SHORT", "account_id": "B", "capital_basis": "equity", "amount": 500.0},
    ]
    result = size_targets(
        weights,
        [_spec("RB2610")],
        {"A": _account(), "B": _account()},
        routes,
    )

    for account_id in ("A", "B"):
        account = result["accounts"][account_id]
        target = _targets(result, account_id)["RB2610"]
        assert target["target_notional"] == 0
        assert target["raw_lots"] == 0
        assert target["target_lots"] == 0
        assert target["actual_notional"] == 0
        assert account["summary"] == {
            "gross_notional": 0,
            "net_notional": 0,
            "estimated_margin": 0,
            "gross_exposure": 0,
            "net_exposure": 0,
        }
        assert len(account["attribution"]) == 2
        assert {row["strategy_id"] for row in account["attribution"]} == {"LONG", "SHORT"}


def test_weight_is_not_used_as_a_second_leverage_multiplier_and_minimum_equity_is_reported():
    rows = [
        _weight("RB", "RB2610", 0.2),
        _weight("CU", "CU2610", 0.8),
    ]
    result = _one_strategy(rows, amount=1_000.0)
    targets = _targets(result)

    assert targets["RB2610"]["target_notional"] == pytest.approx(200.0)
    assert targets["CU2610"]["target_notional"] == pytest.approx(800.0)
    assert targets["RB2610"]["actual_notional"] == pytest.approx(200.0)
    assert result["capital_requirements"]["S"] == {
        "proportional_one_lot_equity": pytest.approx(1000.0),
        "rounded_one_lot_equity": pytest.approx(500.0),
        "rows": [
            {
                "root": "RB",
                "contract": "RB2610",
                "weight": 0.2,
                "notional_per_lot": 200.0,
                "proportional_one_lot_equity": 1000.0,
                "rounded_one_lot_equity": 500.0,
            },
            {
                "root": "CU",
                "contract": "CU2610",
                "weight": 0.8,
                "notional_per_lot": 200.0,
                "proportional_one_lot_equity": 250.0,
                "rounded_one_lot_equity": 125.0,
            },
        ],
    }


def test_equity_notional_and_margin_budget_formulas_are_distinct():
    rows = [
        _weight("RB", "RB2610", 0.5),
        _weight("CU", "CU2610", -0.5),
    ]
    specs = [
        _spec("RB2610", margin_long=0.1, margin_short=0.2),
        _spec("CU2610", margin_long=0.1, margin_short=0.2),
    ]
    accounts = {name: _account() for name in ("E", "N", "M")}
    routes = [
        {"strategy_id": "S", "account_id": "E", "capital_basis": "equity", "amount": 1_000.0},
        {"strategy_id": "S", "account_id": "N", "capital_basis": "notional", "amount": 1_000.0},
        {"strategy_id": "S", "account_id": "M", "capital_basis": "margin", "amount": 1_000.0},
    ]
    result = size_targets({"S": rows}, specs, accounts, routes)

    for account_id in ("E", "N"):
        targets = _targets(result, account_id)
        assert targets["RB2610"]["target_notional"] == pytest.approx(500.0)
        assert targets["CU2610"]["target_notional"] == pytest.approx(-500.0)
    margin_targets = _targets(result, "M")
    assert margin_targets["RB2610"]["target_notional"] == pytest.approx(1000.0 / 0.15 * 0.5)
    assert margin_targets["CU2610"]["target_notional"] == pytest.approx(-1000.0 / 0.15 * 0.5)


def test_attribution_preserves_route_level_premerge_amounts():
    result = size_targets(
        {
            "S1": [_weight("RB", "RB2610", 1.0)],
            "S2": [_weight("RB", "RB2610", 1.0)],
        },
        [_spec("RB2610")],
        {"A": _account()},
        [
            {"strategy_id": "S1", "account_id": "A", "capital_basis": "equity", "amount": 100.0},
            {"strategy_id": "S2", "account_id": "A", "capital_basis": "equity", "amount": 200.0},
        ],
    )

    target = _targets(result)["RB2610"]
    assert target["target_notional"] == pytest.approx(300.0)
    assert [row["target_notional"] for row in result["accounts"]["A"]["attribution"]] == [
        100.0,
        200.0,
    ]


def test_zero_lot_requirement_and_five_million_capital_requirement_are_diagnosed():
    rows = [_weight("RB", "RB2610", 0.0001)]
    account = _account(equity=5_000_000.0, require_one_lot=True)
    result = _one_strategy(
        rows,
        account=account,
        amount=5_000_000.0,
        specs=[_spec("RB2610", multiplier=100_000.0)],
    )
    target = _targets(result)["RB2610"]

    assert target["target_lots"] == 0
    assert any("REQUIRE_ONE_LOT" in blocker for blocker in result["accounts"]["A"]["blockers"])
    assert result["accounts"]["A"]["tradable"] is False
    assert result["capital_requirements"]["S"]["proportional_one_lot_equity"] == pytest.approx(10_000_000_000.0)
    assert result["capital_requirements"]["S"]["rounded_one_lot_equity"] == pytest.approx(5_000_000_000.0)


def test_single_order_limit_is_carried_for_execution_without_capping_total_target_position():
    result = _one_strategy(
        [_weight("RB", "RB2610", 1.0)],
        amount=200_000.0,
        specs=[_spec("RB2610", max_order_lots=100, max_position_lots=1_200)],
    )
    target = _targets(result)["RB2610"]

    # The sizing target is 1,000 lots.  If execution already holds 999 lots,
    # its one-lot delta is within the 100-lot single-order limit; sizing has no
    # position snapshot and must not reject the full target here.
    assert target["target_lots"] == 1_000
    assert target["max_order_lots"] == 100
    assert not any("MAX_ORDER_LOTS" in blocker for blocker in result["accounts"]["A"]["blockers"])
    assert not any("MAX_POSITION_LOTS" in blocker for blocker in result["accounts"]["A"]["blockers"])


def test_max_position_lots_is_the_sizing_layer_position_gate():
    result = _one_strategy(
        [_weight("RB", "RB2610", 1.0)],
        amount=200_000.0,
        specs=[_spec("RB2610", max_order_lots=100, max_position_lots=999)],
    )
    blockers = result["accounts"]["A"]["blockers"]
    assert any("MAX_POSITION_LOTS" in blocker for blocker in blockers)
    assert not any("MAX_ORDER_LOTS" in blocker for blocker in blockers)


def test_unverified_or_stale_spec_metadata_never_marks_account_tradable():
    rows = [_weight("RB", "RB2610", 1.0)]
    for spec in (
        _spec("RB2610", verified=False),
        _spec("RB2610", as_of="2026-09-14"),
    ):
        result = _one_strategy(rows, specs=[spec])
        account = result["accounts"]["A"]
        assert account["tradable"] is False
        assert any("METADATA_UNVERIFIED" in blocker for blocker in account["blockers"])


def test_unverified_indicative_spec_may_omit_tick_for_amount_diagnostics():
    spec = _spec("RB2610", verified=False)
    spec["tick"] = None
    result = _one_strategy(
        [_weight("RB", "RB2610", 1.0)],
        specs=[spec],
    )
    target = _targets(result)["RB2610"]
    assert target["tick"] is None
    assert target["actual_notional"] > 0
    assert any("METADATA_UNVERIFIED" in blocker for blocker in result["accounts"]["A"]["blockers"])


def test_post_rounding_gross_net_margin_and_available_limits_are_all_blockers():
    account = _account(
        equity=1_000.0,
        available=90.0,
        max_margin_ratio=0.09,
        max_gross_exposure=0.15,
        max_abs_net_exposure=0.15,
    )
    result = _one_strategy(
        [_weight("RB", "RB2610", 1.0)],
        account=account,
        amount=150.0,
        specs=[_spec("RB2610", margin_long=0.5)],
    )
    target = _targets(result)["RB2610"]
    blockers = result["accounts"]["A"]["blockers"]

    assert target["raw_lots"] == pytest.approx(0.75)
    assert target["target_lots"] == 1
    assert result["accounts"]["A"]["summary"]["gross_exposure"] == pytest.approx(0.2)
    assert any("GROSS_EXPOSURE" in blocker for blocker in blockers)
    assert any("NET_EXPOSURE" in blocker for blocker in blockers)
    assert any("MARGIN" in blocker for blocker in blockers)
    assert any("AVAILABLE" in blocker for blocker in blockers)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda rows, specs, accounts, routes: routes[0].update(amount=float("nan")), "amount"),
        (lambda rows, specs, accounts, routes: routes[0].update(amount=-1.0), "amount"),
        (lambda rows, specs, accounts, routes: rows[0].update(weight=float("inf")), "weight"),
        (lambda rows, specs, accounts, routes: specs.append(deepcopy(specs[0])), "duplicate"),
        (lambda rows, specs, accounts, routes: rows[0].update(contract="RB8888"), "continuous"),
    ],
)
def test_nonfinite_negative_duplicate_and_continuous_inputs_are_rejected(mutator, message):
    rows = [_weight("RB", "RB2610", 1.0)]
    specs = [_spec("RB2610")]
    accounts = {"A": _account()}
    routes = [{"strategy_id": "S", "account_id": "A", "capital_basis": "equity", "amount": 100.0}]
    mutator(rows, specs, accounts, routes)

    with pytest.raises(ValueError, match=message):
        size_targets({"S": rows}, specs, accounts, routes)


def test_missing_strategy_account_duplicate_route_and_price_conflict_are_rejected():
    row = _weight("RB", "RB2610", 1.0)
    with pytest.raises(ValueError, match="strategy"):
        size_targets(
            {"S": [row]},
            [_spec("RB2610")],
            {"A": _account()},
            [{"strategy_id": "MISSING", "account_id": "A", "capital_basis": "equity", "amount": 1.0}],
        )
    with pytest.raises(ValueError, match="account"):
        size_targets(
            {"S": [row]},
            [_spec("RB2610")],
            {"A": _account()},
            [{"strategy_id": "S", "account_id": "MISSING", "capital_basis": "equity", "amount": 1.0}],
        )
    with pytest.raises(ValueError, match="duplicate route"):
        size_targets(
            {"S": [row]},
            [_spec("RB2610")],
            {"A": _account()},
            [
                {"strategy_id": "S", "account_id": "A", "capital_basis": "equity", "amount": 1.0},
                {"strategy_id": "S", "account_id": "A", "capital_basis": "equity", "amount": 2.0},
            ],
        )
    with pytest.raises(ValueError, match="price conflict"):
        size_targets(
            {
                "S1": [row],
                "S2": [_weight("RB", "RB2610", 1.0, close=11.0)],
            },
            [_spec("RB2610")],
            {"A": _account()},
            [
                {"strategy_id": "S1", "account_id": "A", "capital_basis": "equity", "amount": 1.0},
                {"strategy_id": "S2", "account_id": "A", "capital_basis": "equity", "amount": 1.0},
            ],
        )


def test_optional_weight_exchange_must_match_the_contract_spec():
    row = _weight("RB", "RB2610", 1.0)
    row["exchange"] = "DCE"
    with pytest.raises(ValueError, match="exchange conflict"):
        size_targets(
            {"S": [row]},
            [_spec("RB2610")],
            {"A": _account()},
            [{"strategy_id": "S", "account_id": "A", "capital_basis": "equity", "amount": 100.0}],
        )


def test_all_weight_rows_in_a_size_run_must_share_one_data_date():
    first = _weight("RB", "RB2610", 1.0)
    second = _weight("CU", "CU2610", -1.0)
    second["data_date"] = "2026-09-14"
    with pytest.raises(ValueError, match="data_date"):
        size_targets(
            {"S1": [first], "S2": [second]},
            [_spec("RB2610"), _spec("CU2610")],
            {"A": _account()},
            [
                {"strategy_id": "S1", "account_id": "A", "capital_basis": "equity", "amount": 100.0},
                {"strategy_id": "S2", "account_id": "A", "capital_basis": "equity", "amount": 100.0},
            ],
        )


def test_every_configured_account_requires_a_route_and_empty_routes_are_rejected():
    rows = {"S": [_weight("RB", "RB2610", 1.0)]}
    specs = [_spec("RB2610")]
    with pytest.raises(ValueError, match="missing route"):
        size_targets(
            rows,
            specs,
            {"A": _account(), "UNROUTED": _account()},
            [{"strategy_id": "S", "account_id": "A", "capital_basis": "equity", "amount": 100.0}],
        )
    with pytest.raises(ValueError, match="routes must not be empty"):
        size_targets(rows, specs, {"A": _account()}, [])


def test_equity_routes_cannot_exceed_equity_and_mixed_bases_use_actual_risk_gates():
    account = _account(equity=100.0)
    over = size_targets(
        {
            "S1": [_weight("RB", "RB2610", 1.0)],
            "S2": [_weight("RB", "RB2610", 1.0)],
        },
        [_spec("RB2610")],
        {"A": account},
        [
            {"strategy_id": "S1", "account_id": "A", "capital_basis": "equity", "amount": 60.0},
            {"strategy_id": "S2", "account_id": "A", "capital_basis": "equity", "amount": 60.0},
        ],
    )
    assert any("EQUITY_ROUTE_BUDGET" in blocker for blocker in over["accounts"]["A"]["blockers"])
    assert over["accounts"]["A"]["tradable"] is False

    result = size_targets(
        {
            "S1": [_weight("RB", "RB2610", 1.0)],
            "S2": [_weight("RB", "RB2610", 1.0)],
        },
        [_spec("RB2610")],
        {"A": account},
        [
            {"strategy_id": "S1", "account_id": "A", "capital_basis": "equity", "amount": 60.0},
            {"strategy_id": "S2", "account_id": "A", "capital_basis": "notional", "amount": 40.0},
        ],
    )
    # The desired 60 + 40 is merged first, then 100 / 200 per lot rounds up
    # to one actual lot.  Risk gates therefore inspect 200, not the raw 100.
    assert result["accounts"]["A"]["summary"]["gross_notional"] == pytest.approx(200.0)


def test_margin_used_can_exceed_equity_and_zero_available_is_a_diagnostic_state():
    account = _account(equity=100.0, available=0.0, margin_used=1_000.0)
    result = _one_strategy(
        [_weight("RB", "RB2610", 1.0)],
        account=account,
        amount=100.0,
        specs=[_spec("RB2610", margin_long=1.2)],
    )
    assert result["accounts"]["A"]["account"]["margin_used"] == 1_000.0
    # Existing margin_used covers the estimate, so available=0 is accepted by
    # the specified max(0, target_margin - margin_used) rule.
    assert "AVAILABLE" not in " ".join(result["accounts"]["A"]["blockers"])
    assert result["accounts"]["A"]["targets"][0]["estimated_margin"] == pytest.approx(240.0)


def test_available_already_excludes_frozen_margin_but_total_margin_cap_keeps_it():
    account = _account(
        equity=1_000.0,
        available=50.0,
        margin_used=100.0,
        frozen_margin=100.0,
        reserve=50.0,
        max_margin_ratio=0.5,
    )
    result = _one_strategy(
        [_weight("RB", "RB2610", 1.0)],
        account=account,
        amount=1_000.0,
        specs=[_spec("RB2610", margin_long=0.1)],
    )
    account_result = result["accounts"]["A"]
    assert account_result["summary"]["estimated_margin"] == pytest.approx(100.0)
    assert "AVAILABLE" not in " ".join(account_result["blockers"])
    assert "MARGIN" not in " ".join(account_result["blockers"])
    assert account_result["tradable"] is True

    capped = _one_strategy(
        [_weight("RB", "RB2610", 1.0)],
        account={**account, "available": 500.0, "max_margin_ratio": 0.2},
        amount=1_000.0,
        specs=[_spec("RB2610", margin_long=0.1)],
    )["accounts"]["A"]
    assert any("MARGIN" in blocker for blocker in capped["blockers"])
    assert "AVAILABLE" not in " ".join(capped["blockers"])


def test_zero_equity_is_invalid_even_when_the_other_account_fields_are_finite():
    with pytest.raises(ValueError, match="equity"):
        _one_strategy(
            [_weight("RB", "RB2610", 1.0)],
            account=_account(equity=0.0),
            amount=1.0,
        )


def test_specs_as_of_allows_t_minus_one_weights_with_t_specs_and_preserves_signal_date():
    row = _weight("RB", "RB2610", 1.0)
    row["data_date"] = "2026-09-14"
    original = deepcopy(row)
    result = size_targets(
        {"S": [row]},
        [_spec("RB2610", as_of="2026-09-15")],
        {"A": _account()},
        [{"strategy_id": "S", "account_id": "A", "capital_basis": "equity", "amount": 100.0}],
        specs_as_of="2026-09-15",
    )

    account = result["accounts"]["A"]
    assert account["tradable"] is True
    assert not any("METADATA_UNVERIFIED" in blocker for blocker in account["blockers"])
    assert row == original
    assert account["attribution"][0]["data_date"] == "2026-09-14"


def test_specs_as_of_blocks_stale_spec_for_the_requested_execution_date():
    row = _weight("RB", "RB2610", 1.0)
    row["data_date"] = "2026-09-14"
    result = size_targets(
        {"S": [row]},
        [_spec("RB2610", as_of="2026-09-14")],
        {"A": _account()},
        [{"strategy_id": "S", "account_id": "A", "capital_basis": "equity", "amount": 100.0}],
        specs_as_of="2026-09-15",
    )

    assert result["accounts"]["A"]["tradable"] is False
    assert any("METADATA_UNVERIFIED" in blocker for blocker in result["accounts"]["A"]["blockers"])


def test_specs_as_of_blocks_future_spec_for_the_requested_execution_date():
    result = size_targets(
        {"S": [_weight("RB", "RB2610", 1.0)]},
        [_spec("RB2610", as_of="2026-09-16")],
        {"A": _account()},
        [{"strategy_id": "S", "account_id": "A", "capital_basis": "equity", "amount": 100.0}],
        specs_as_of="2026-09-15",
    )

    assert result["accounts"]["A"]["tradable"] is False
    assert any("METADATA_UNVERIFIED" in blocker for blocker in result["accounts"]["A"]["blockers"])


def test_specs_as_of_none_keeps_the_original_weight_date_metadata_rule():
    row = _weight("RB", "RB2610", 1.0)
    row["data_date"] = "2026-09-14"
    result = size_targets(
        {"S": [row]},
        [_spec("RB2610", as_of="2026-09-15")],
        {"A": _account()},
        [{"strategy_id": "S", "account_id": "A", "capital_basis": "equity", "amount": 100.0}],
    )

    assert result["accounts"]["A"]["tradable"] is False
    assert any("METADATA_UNVERIFIED" in blocker for blocker in result["accounts"]["A"]["blockers"])


def test_specs_as_of_must_be_valid_and_not_earlier_than_weight_data_date():
    row = _weight("RB", "RB2610", 1.0)
    with pytest.raises(ValueError, match="specs_as_of"):
        size_targets(
            {"S": [row]},
            [_spec("RB2610")],
            {"A": _account()},
            [{"strategy_id": "S", "account_id": "A", "capital_basis": "equity", "amount": 100.0}],
            specs_as_of="2026-02-30",
        )

    with pytest.raises(ValueError, match="earlier than weight data_date"):
        size_targets(
            {"S": [row]},
            [_spec("RB2610")],
            {"A": _account()},
            [{"strategy_id": "S", "account_id": "A", "capital_basis": "equity", "amount": 100.0}],
            specs_as_of="2026-09-14",
        )
