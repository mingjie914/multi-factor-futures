from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from itertools import product
import math

import pytest

from trading.sizing import size_targets


def _account(
    *,
    equity: float = 1_000.0,
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
    multiplier: float,
    margin_long: float = 0.1,
    margin_short: float = 0.1,
    verified: bool = True,
    max_position_lots: int | None = None,
) -> dict:
    row = {
        "contract": contract,
        "exchange": "SHFE",
        "multiplier": multiplier,
        "margin_long": margin_long,
        "margin_short": margin_short,
        "tick": 1.0,
        "as_of": "2026-09-15",
        "source": "daily-spec",
        "verified": verified,
    }
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


def _size(
    weight_sets: dict[str, list[dict]],
    specs: list[dict],
    *,
    account: dict | None = None,
    routes: list[dict] | None = None,
    rounding: str | None = "balanced",
) -> dict:
    if routes is None:
        routes = [
            {
                "strategy_id": "S",
                "account_id": "A",
                "capital_basis": "equity",
                "amount": 1_000.0,
            }
        ]
    kwargs = {} if rounding is None else {"rounding": rounding}
    return size_targets(
        weight_sets,
        specs,
        {"A": _account() if account is None else account},
        routes,
        **kwargs,
    )


def _targets(result: dict) -> dict[str, dict]:
    return {row["contract"]: row for row in result["accounts"]["A"]["targets"]}


def _neutral_case(*, sign: int = 1) -> tuple[list[dict], list[dict]]:
    # Raw notionals are +1600, +800, -2400, so the unrounded net is zero.
    # Independent half-up rounding produces +600 net notional.  BB is the
    # cheaper feasible one-lot correction than the larger AA contract.
    rows = [
        _weight("AA", "AA2610", sign * 1.6),
        _weight("BB", "BB2610", sign * 0.8),
        _weight("CC", "CC2610", sign * -2.4),
    ]
    specs = [
        _spec("AA2610", multiplier=100.0),
        _spec("BB2610", multiplier=50.0),
        _spec("CC2610", multiplier=10.0),
    ]
    return rows, specs


@pytest.mark.parametrize("sign", [1, -1], ids=["net_long", "net_short"])
def test_balanced_rounding_chooses_the_lowest_error_feasible_one_lot_adjustment(sign: int):
    rows, specs = _neutral_case(sign=sign)
    result = _size(
        {"S": rows},
        specs,
        account=_account(max_abs_net_exposure=0.45),
    )
    targets = _targets(result)
    account = result["accounts"]["A"]
    attribution = {row["contract"]: row for row in account["attribution"]}

    assert [targets[c]["raw_lots"] for c in ("AA2610", "BB2610", "CC2610")] == [
        pytest.approx(sign * 1.6),
        pytest.approx(sign * 1.6),
        pytest.approx(sign * -24.0),
    ]
    assert [targets[c]["independent_rounded_lots"] for c in ("AA2610", "BB2610", "CC2610")] == [
        sign * 2,
        sign * 2,
        sign * -24,
    ]
    assert [targets[c]["target_lots"] for c in ("AA2610", "BB2610", "CC2610")] == [
        sign * 2,
        sign * 1,
        sign * -24,
    ]
    assert [
        (row["contract"], row["from_lots"], row["to_lots"])
        for row in account["rounding_adjustments"]
    ] == [("BB2610", sign * 2, sign * 1)]
    assert not any("NET_EXPOSURE" in blocker for blocker in account["blockers"])
    assert account["tradable"] is True
    for contract, target in targets.items():
        route_row = attribution[contract]
        assert route_row["rounded_lots"] == target["target_lots"]
        assert route_row["independent_rounded_lots"] == target["independent_rounded_lots"]
        assert route_row["estimated_margin"] == pytest.approx(target["estimated_margin"])
        assert route_row["rounded_weight"] == pytest.approx(
            target["actual_notional"] / route_row["reference_capital"]
        )

    assert sum(row["actual_notional"] for row in account["targets"]) == pytest.approx(
        account["summary"]["net_notional"]
    )
    assert sum(abs(row["actual_notional"]) for row in account["targets"]) == pytest.approx(
        account["summary"]["gross_notional"]
    )
    assert sum(row["estimated_margin"] for row in account["targets"]) == pytest.approx(
        account["summary"]["estimated_margin"]
    )
    assert account["summary"]["net_notional"] == pytest.approx(sign * 100.0)
    for target in account["targets"]:
        assert target["actual_notional"] == pytest.approx(
            target["target_lots"] * target["close"] * target["multiplier"]
        )
        assert target["estimated_margin"] == pytest.approx(
            abs(target["actual_notional"]) * target["margin_rate"]
        )


def test_default_rounding_remains_half_up_and_balanced_is_opt_in():
    rows = [
        _weight("RB", "RB2610", 0.1),
        _weight("CU", "CU2610", 0.3),
        _weight("AL", "AL2610", -0.1),
        _weight("AG", "AG2610", -0.3),
    ]
    specs = [_spec(row["contract"], multiplier=20.0) for row in rows]

    default_targets = _targets(_size({"S": rows}, specs, rounding=None))
    explicit_targets = _targets(_size({"S": rows}, specs, rounding="half_up"))
    assert [default_targets[c]["target_lots"] for c in ("RB2610", "CU2610", "AL2610", "AG2610")] == [
        1,
        2,
        -1,
        -2,
    ]
    assert [explicit_targets[c]["target_lots"] for c in ("RB2610", "CU2610", "AL2610", "AG2610")] == [
        1,
        2,
        -1,
        -2,
    ]
    assert [default_targets[c]["independent_rounded_lots"] for c in default_targets] == [
        default_targets[c]["target_lots"] for c in default_targets
    ]


def test_balanced_rounding_minimizes_net_error_even_when_already_inside_net_limit():
    rows, specs = _neutral_case()
    result = _size(
        {"S": rows},
        specs,
        account=_account(max_abs_net_exposure=0.7),
    )
    targets = _targets(result)
    account = result["accounts"]["A"]

    assert [targets[c]["target_lots"] for c in ("AA2610", "BB2610", "CC2610")] == [2, 1, -24]
    assert [targets[c]["independent_rounded_lots"] for c in ("AA2610", "BB2610", "CC2610")] == [
        2,
        2,
        -24,
    ]
    assert len(account["rounding_adjustments"]) == 1
    assert account["summary"]["net_notional"] == pytest.approx(100.0)
    assert account["tradable"] is True
    assert not any("NET_EXPOSURE" in blocker for blocker in account["blockers"])


def test_balanced_rounding_finds_joint_integer_changes_not_a_greedy_local_minimum():
    rows = [_weight("AA", "AA2610", 0.8), _weight("BB", "BB2610", -0.64),
            _weight("CC", "CC2610", 0.24), _weight("DD", "DD2610", -0.4)]
    specs = [_spec(c, multiplier=m) for c, m in
             [("AA2610", 50), ("BB2610", 40), ("CC2610", 15), ("DD2610", 10)]]
    result = _size({"S": rows}, specs, account=_account(require_one_lot=True, max_abs_net_exposure=0.25))
    targets = _targets(result)
    assert [targets[c]["target_lots"] for c in ("AA2610", "BB2610", "CC2610", "DD2610")] == [1, -1, 2, -4]
    assert result["accounts"]["A"]["summary"]["net_notional"] == 0
    assert all(isinstance(r["target_lots"], int) and abs(r["target_lots"]) >= 1 for r in targets.values())
    # The existing downstream path must consume these corrected integers and
    # stop ordering once that target is reached, without re-running sizing.
    from trading.execution import build_plan, paper_execute
    now = datetime(2026, 9, 15, 8, tzinfo=timezone.utc)
    snapshot = {"identity": "paper-one", "trade_date": "2026-09-15", "as_of": now.isoformat(),
                "positions": [], "open_orders": [], "equity": 1000, "available": 1000, "margin_used": 0}
    policy = {"expected_identity": "paper-one", "managed_roots": ["AA", "BB", "CC", "DD"],
              "max_snapshot_age_seconds": 60, "plan_ttl_seconds": 60, "order_type": "market",
              "max_order_lots": 100, "max_price_deviation": .05, "limit_prices": {},
              "channel": "paper", "cold_start": "adopt_managed"}
    arguments = dict(account_id="A", policy=policy, signal_id="integer-target", signal_date="2026-09-15", now=now)
    plan = build_plan(result["accounts"]["A"], snapshot, **arguments)
    assert plan["ready"]
    assert {o["contract"]: o["volume"] for o in plan["orders"]} == {
        c: abs(r["target_lots"]) for c, r in targets.items()}
    assert all(isinstance(o["volume"], int) for o in plan["orders"])
    filled = paper_execute(plan, snapshot, now=now)
    assert build_plan(result["accounts"]["A"], filled["snapshot"], **arguments)["orders"] == []


def test_balanced_rounding_preserves_a_nonzero_theoretical_net_instead_of_forcing_zero():
    rows = [_weight("AA", "AA2610", 1.6), _weight("BB", "BB2610", -1.0)]
    specs = [_spec("AA2610", multiplier=100), _spec("BB2610", multiplier=100)]
    account = _size({"S": rows}, specs)["accounts"]["A"]
    # Theoretical net is +600: +1000 is closer than the achievable zero.
    assert account["summary"]["net_notional"] == 1000
    assert account["rounding_adjustments"] == []


def test_balanced_rounding_keeps_numerically_integral_complete_unit_at_one_lot():
    rows = [_weight("AA", "AA2610", 1.000000000142), _weight("BB", "BB2610", -1.51)]
    specs = [_spec("AA2610", multiplier=100), _spec("BB2610", multiplier=100)]
    target = _targets(_size({"S": rows}, specs))["AA2610"]
    assert target["target_lots"] == 1


@pytest.mark.parametrize("raw_lots", [(1.3, 2.6, -1.4, -2.3), (-1.3, -2.6, 1.4, 2.3),
                                     (1.49, -2.51, 3.5, -1.01), (2.7, 1.2, -2.2, -1.8)])
def test_balanced_rounding_matches_an_independent_exhaustive_integer_oracle(raw_lots):
    units = [130, 210, 340, 550]
    roots = ["AA", "BB", "CC", "DD"]
    rows = [_weight(r, r + "2610", q * u / 1000) for r, q, u in zip(roots, raw_lots, units)]
    specs = [_spec(r + "2610", multiplier=u / 10) for r, u in zip(roots, units)]
    account = _account(equity=100000, require_one_lot=True)
    before = _targets(_size({"S": rows}, specs, account=account, rounding="half_up"))
    after = _targets(_size({"S": rows}, specs, account=account))
    keys = sorted(before)
    theoretical = [before[c]["target_notional"] for c in keys]
    def objective(lots):
        values = [q * u for q, u in zip(lots, units)]
        return (round(abs(sum(values) - sum(theoretical)), 8),
                round(sum(abs(v - t) for v, t in zip(values, theoretical)), 8),
                sum(q != before[c]["target_lots"] for c, q in zip(keys, lots)))
    alternatives = [(math.floor(before[c]["raw_lots"]), math.ceil(before[c]["raw_lots"])) for c in keys]
    expected = min(objective(q) for q in product(*alternatives))
    assert objective([after[c]["target_lots"] for c in keys]) == expected


def test_balanced_rounding_rejects_unbounded_search_before_allocating_candidates():
    roots = ["A" + chr(ord("A") + i) for i in range(25)]
    rows = [_weight(r, r + "2610", 1.6 if i % 2 else -1.6) for i, r in enumerate(roots)]
    specs = [_spec(r + "2610", multiplier=100) for r in roots]
    with pytest.raises(ValueError, match="BALANCED_ROUNDING_SEARCH_LIMIT"):
        _size({"S": rows}, specs, account=_account(equity=100000))


def test_balanced_rounding_does_not_mask_a_raw_vector_already_over_net_limit():
    rows = [
        _weight("AA", "AA2610", 1.6),
        _weight("CC", "CC2610", -1.0),
    ]
    specs = [
        _spec("AA2610", multiplier=100.0),
        _spec("CC2610", multiplier=100.0),
    ]
    result = _size(
        {"S": rows},
        specs,
        account=_account(max_abs_net_exposure=0.45),
    )
    targets = _targets(result)
    account = result["accounts"]["A"]

    assert sum(target["target_notional"] for target in account["targets"]) == pytest.approx(600.0)
    assert [targets[c]["target_lots"] for c in ("AA2610", "CC2610")] == [2, -1]
    assert [targets[c]["independent_rounded_lots"] for c in ("AA2610", "CC2610")] == [2, -1]
    assert account["rounding_adjustments"] == []
    assert account["tradable"] is False
    assert any("NET_EXPOSURE" in blocker for blocker in account["blockers"])


def test_balanced_rounding_restores_half_up_when_no_candidate_satisfies_all_limits():
    rows, specs = _neutral_case()
    result = _size(
        {"S": rows},
        specs,
        account=_account(max_abs_net_exposure=0.45, max_margin_ratio=0.04),
    )
    targets = _targets(result)
    account = result["accounts"]["A"]

    assert [targets[c]["target_lots"] for c in ("AA2610", "BB2610", "CC2610")] == [2, 2, -24]
    assert [targets[c]["independent_rounded_lots"] for c in ("AA2610", "BB2610", "CC2610")] == [
        2,
        2,
        -24,
    ]
    assert account["rounding_adjustments"] == []
    assert account["tradable"] is False
    assert any("NET_EXPOSURE" in blocker for blocker in account["blockers"])
    assert any("MARGIN" in blocker for blocker in account["blockers"])


def test_balanced_rounding_never_reduces_a_one_lot_target_to_zero():
    rows = [
        _weight("AA", "AA2610", 0.6),
        _weight("BB", "BB2610", 0.3),
        _weight("CC", "CC2610", -0.9),
    ]
    specs = [
        _spec("AA2610", multiplier=100.0),
        _spec("BB2610", multiplier=50.0),
        _spec("CC2610", multiplier=10.0),
    ]
    result = _size(
        {"S": rows},
        specs,
        account=_account(max_abs_net_exposure=0.45),
    )
    targets = _targets(result)
    account = result["accounts"]["A"]

    # Zeroing AA or BB would bring the +600 half-up net notional inside the
    # +450 cap, but both raw lot amounts are below one and therefore ineligible.
    assert [targets[c]["raw_lots"] for c in ("AA2610", "BB2610", "CC2610")] == [
        pytest.approx(0.6),
        pytest.approx(0.6),
        pytest.approx(-9.0),
    ]
    assert [targets[c]["target_lots"] for c in ("AA2610", "BB2610", "CC2610")] == [1, 1, -9]
    assert all(targets[c]["target_lots"] != 0 for c in ("AA2610", "BB2610"))
    assert account["rounding_adjustments"] == []
    assert account["tradable"] is False
    assert any("NET_EXPOSURE" in blocker for blocker in account["blockers"])


def test_balanced_rounding_merges_routes_before_account_correction_without_fake_attribution():
    rows = {
        "S1": [
            _weight("AA", "AA2610", 1.6),
            _weight("BB", "BB2610", 0.4),
            _weight("CC", "CC2610", -2.4),
        ],
        "S2": [_weight("BB", "BB2610", 0.4)],
    }
    specs = [
        _spec("AA2610", multiplier=100.0),
        _spec("BB2610", multiplier=50.0),
        _spec("CC2610", multiplier=10.0),
    ]
    routes = [
        {"strategy_id": "S1", "account_id": "A", "capital_basis": "equity", "amount": 1_000.0},
        {"strategy_id": "S2", "account_id": "A", "capital_basis": "equity", "amount": 1_000.0},
    ]
    result = _size(
        rows,
        specs,
        account=_account(equity=3_000.0, max_abs_net_exposure=0.15),
        routes=routes,
    )
    account = result["accounts"]["A"]
    targets = _targets(result)
    attribution = account["attribution"]
    bb_attribution = [row for row in attribution if row["contract"] == "BB2610"]

    assert targets["BB2610"]["independent_rounded_lots"] == 2
    assert targets["BB2610"]["target_lots"] == 1
    assert [row["independent_rounded_lots"] for row in bb_attribution] == [1, 1]
    assert [row["rounded_lots"] for row in bb_attribution] == [1, 1]
    assert sum(row["rounded_lots"] for row in bb_attribution) == 2
    assert [row["estimated_margin"] for row in bb_attribution] == [
        pytest.approx(50.0),
        pytest.approx(50.0),
    ]
    assert targets["BB2610"]["actual_notional"] == pytest.approx(500.0)
    assert targets["BB2610"]["estimated_margin"] == pytest.approx(50.0)
    assert sum(row["target_notional"] for row in attribution) == pytest.approx(
        sum(row["target_notional"] for row in account["targets"])
    )
    assert [(row["contract"], row["from_lots"], row["to_lots"]) for row in account["rounding_adjustments"]] == [
        ("BB2610", 2, 1)
    ]


def test_balanced_rounding_is_deterministic_and_uses_contract_code_for_ties():
    rows = [
        _weight("BB", "BB2610", 1.6),
        _weight("AA", "AA2610", 1.6),
        _weight("CC", "CC2610", -3.2),
    ]
    specs = [
        _spec("BB2610", multiplier=100.0),
        _spec("AA2610", multiplier=100.0),
        _spec("CC2610", multiplier=10.0),
    ]
    account = _account(max_abs_net_exposure=0.5)
    first = _size({"S": rows}, specs, account=account)
    second = _size(
        {"S": list(reversed(rows))},
        list(reversed(specs)),
        account=deepcopy(account),
    )

    def decision(result: dict) -> tuple[dict[str, int], list[tuple[str, int, int]]]:
        account_result = result["accounts"]["A"]
        return (
            {row["contract"]: row["target_lots"] for row in account_result["targets"]},
            [
                (row["contract"], row["from_lots"], row["to_lots"])
                for row in account_result["rounding_adjustments"]
            ],
        )

    expected = ({"AA2610": 1, "BB2610": 2, "CC2610": -32}, [("AA2610", 2, 1)])
    assert decision(first) == expected
    assert decision(second) == expected


def test_unverified_spec_for_zero_target_does_not_block_sizing():
    rows = [
        _weight("RB", "RB2610", 0.0),
        _weight("CU", "CU2610", 0.2),
    ]
    specs = [
        _spec("RB2610", multiplier=20.0, verified=False),
        _spec("CU2610", multiplier=20.0),
    ]
    account = _size({"S": rows}, specs)["accounts"]["A"]
    target = {row["contract"]: row for row in account["targets"]}["RB2610"]

    assert target["target_lots"] == 0
    assert not any("METADATA_UNVERIFIED: RB2610" in blocker for blocker in account["blockers"])
    assert account["tradable"] is True


def test_unverified_spec_for_nonzero_target_still_blocks_sizing():
    rows = [_weight("RB", "RB2610", 0.2)]
    specs = [_spec("RB2610", multiplier=20.0, verified=False)]
    account = _size({"S": rows}, specs)["accounts"]["A"]

    assert account["targets"][0]["target_lots"] == 1
    assert any("METADATA_UNVERIFIED: RB2610" in blocker for blocker in account["blockers"])
    assert account["tradable"] is False
