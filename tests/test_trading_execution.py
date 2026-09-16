from __future__ import annotations

from datetime import datetime, timezone
import copy

import pytest

from trading.execution import build_plan, paper_execute


NOW = datetime(2026, 9, 15, 8, 0, tzinfo=timezone.utc)


def inputs():
    sized = {"tradable": True, "blockers": [], "account": {"equity": 5e6, "available": 5e6,
             "reserve": 10000, "max_margin_ratio": .7},
             "summary": {"estimated_margin": 36000}, "targets": [{"root": "RB", "contract": "RB2701",
             "exchange": "SHFE", "target_lots": 3, "tick": 1, "close": 3600,
             "multiplier": 10, "margin_rate": .1}]}
    snapshot = {"identity": "paper-one", "trade_date": "2026-09-15",
                "as_of": NOW.isoformat(), "positions": [], "open_orders": [],
                "equity": 5e6, "available": 5e6, "margin_used": 0}
    policy = {"expected_identity": "paper-one", "managed_roots": ["RB"],
              "max_snapshot_age_seconds": 60, "plan_ttl_seconds": 60,
              "order_type": "market", "max_order_lots": 100,
              "max_price_deviation": .05, "limit_prices": {}, "channel": "paper",
              "cold_start": "adopt_managed"}
    return sized, snapshot, policy


def plan(sized, snapshot, policy):
    return build_plan(sized, snapshot, account_id="a", policy=policy,
                      signal_id="weights-a", signal_date="2026-09-15", now=NOW)


def test_new_positions_then_reconcile_makes_no_duplicate_orders():
    sized, snap, policy = inputs()
    p = plan(sized, snap, policy)
    assert p["ready"] and len(p["orders"]) == 1
    assert p["orders"][0]["side"] == "buy"
    assert p["orders"][0]["offset"] == "open"
    assert p["orders"][0]["price"] is None
    filled = paper_execute(p, snap, now=NOW)
    assert plan(sized, filled["snapshot"], policy)["orders"] == []


@pytest.mark.parametrize("quantity", [1.9, -1.9, True])
def test_noninteger_target_is_rejected_instead_of_truncated(quantity):
    sized, snapshot, policy = inputs()
    sized["targets"][0]["target_lots"] = quantity
    with pytest.raises(ValueError, match="target_lots"):
        plan(sized, snapshot, policy)


def test_market_rollover_requires_the_old_contracts_own_verified_specs():
    sized, snapshot, policy = inputs()
    snapshot["positions"] = [{"root": "RB", "contract": "RB2610", "exchange": "SHFE",
        "side": "long", "volume": 1, "today": 0, "yesterday": 1,
        "available_today": 0, "available_yesterday": 1}]
    blocked = plan(sized, snapshot, policy)
    assert not blocked["ready"] and "OLD_CONTRACT_SPEC_REQUIRED" in blocked["blockers"]
    policy["old_contract_specs"] = {"RB2610": {"tick": 1, "close": 3500, "metadata_verified": True}}
    assert plan(sized, snapshot, policy)["ready"]


@pytest.mark.parametrize("has_position", [False, True])
def test_unverified_zero_target_only_blocks_when_a_real_close_is_needed(has_position):
    sized, snap, policy = inputs()
    sized["targets"][0].update(target_lots=0, metadata_verified=False)
    if has_position:
        snap["positions"] = [{"root": "RB", "contract": "RB2701", "exchange": "SHFE",
            "side": "long", "volume": 1, "today": 1, "yesterday": 0,
            "available_today": 1, "available_yesterday": 0}]
    result = plan(sized, snap, policy)
    assert result["ready"] is (not has_position)
    assert ("METADATA_UNVERIFIED: RB2701" in result["blockers"]) is has_position


def test_rollover_closes_exact_old_today_and_yesterday_before_open():
    sized, snap, policy = inputs()
    policy["old_contract_specs"] = {"RB2610": {"tick": 1, "close": 3500, "metadata_verified": True}}
    snap["positions"] = [{"root": "RB", "contract": "RB2610", "exchange": "SHFE",
                          "side": "long", "volume": 3, "today": 1, "yesterday": 2,
                          "available_today": 1, "available_yesterday": 2}]
    p = plan(sized, snap, policy)
    assert [(o["contract"], o["offset"], o["volume"]) for o in p["orders"]] == [
        ("RB2610", "close_yesterday", 2), ("RB2610", "close_today", 1), ("RB2701", "open", 3)]
    assert p["orders"][-1]["depends_on"] == [o["client_order_id"] for o in p["orders"][:-1]]


def test_reverse_position_and_reject_unavailable_close():
    sized, snap, policy = inputs()
    snap["positions"] = [{"root": "RB", "contract": "RB2701", "exchange": "SHFE",
                          "side": "short", "volume": 2, "today": 0, "yesterday": 2,
                          "available_today": 0, "available_yesterday": 1}]
    p = plan(sized, snap, policy)
    assert not p["ready"] and "INSUFFICIENT_CLOSEABLE" in p["blockers"]


@pytest.mark.parametrize("change,expected", [
    ({"identity": "wrong"}, "ACCOUNT_IDENTITY_MISMATCH"),
    ({"as_of": "2026-09-15T07:50:00+00:00"}, "STALE_SNAPSHOT"),
    ({"open_orders": [{"orderId": "open1"}]}, "ACTIVE_ORDERS"),
])
def test_account_and_snapshot_gates(change, expected):
    sized, snap, policy = inputs()
    snap.update(change)
    p = plan(sized, snap, policy)
    assert not p["ready"] and expected in p["blockers"]


def test_limit_requires_explicit_price_and_tick_alignment():
    sized, snap, policy = inputs()
    policy["order_type"] = "limit"
    assert "LIMIT_PRICE_REQUIRED" in plan(sized, snap, policy)["blockers"]
    policy["limit_prices"] = {"RB2701": 3600.5}
    assert "PRICE_TICK_MISMATCH" in plan(sized, snap, policy)["blockers"]


def test_paper_refuses_modified_expired_or_already_changed_snapshot():
    sized, snap, policy = inputs()
    p = plan(sized, snap, policy)
    altered = copy.deepcopy(p)
    altered["orders"][0]["volume"] = 99
    with pytest.raises(ValueError, match="hash"):
        paper_execute(altered, snap, now=NOW)
    with pytest.raises(ValueError, match="expired"):
        paper_execute(p, snap, now=datetime(2026, 9, 15, 8, 2, tzinfo=timezone.utc))
    changed = {**snap, "available": 10}
    with pytest.raises(ValueError, match="snapshot"):
        paper_execute(p, changed, now=NOW)


def test_old_signal_never_becomes_current_and_missing_position_fields_fail():
    sized, snap, policy = inputs()
    p = build_plan(sized, snap, account_id="a", policy=policy,
                   signal_id="a", signal_date="2026-09-11", now=NOW)
    assert "STALE_SIGNAL" in p["blockers"]
    snap["positions"] = [{"root": "RB", "contract": "RB2610", "exchange": "SHFE",
                          "side": "long", "volume": 2}]
    with pytest.raises(ValueError, match="today"):
        plan(sized, snap, policy)


def test_version_change_keeps_existing_target_and_old_roots_close_only_differences():
    sized, snap, policy = inputs()
    snap["positions"] = [{"root": "RB", "contract": "RB2701", "exchange": "SHFE",
                          "side": "long", "volume": 2, "today": 0, "yesterday": 2,
                          "available_today": 0, "available_yesterday": 2}]
    p = plan(sized, snap, policy)
    assert [(o["offset"], o["volume"]) for o in p["orders"]] == [("open", 1)]
    switched = copy.deepcopy(sized)
    switched["targets"] = []
    p = build_plan(switched, snap, account_id="a", policy={**policy, "managed_roots": []},
                   signal_id="new-version", signal_date="2026-09-15", previous_plan=p, now=NOW)
    assert p["managed_roots"] == ["RB"]
    assert [(o["offset"], o["volume"]) for o in p["orders"]] == [("close_yesterday", 2)]


def test_margin_over_one_hundred_percent_allows_reductions_and_suppresses_opens():
    sized, snap, policy = inputs()
    policy["old_contract_specs"] = {"RB2610": {"tick": 1, "close": 3500, "metadata_verified": True}}
    snap.update(equity=10000, available=0, margin_used=11000)
    snap["positions"] = [{"root": "RB", "contract": "RB2610", "exchange": "SHFE",
                          "side": "short", "volume": 2, "today": 0, "yesterday": 2,
                          "available_today": 0, "available_yesterday": 2}]
    p = plan(sized, snap, policy)
    assert p["risk_mode"] == "reduce_only"
    assert p["ready"]
    assert all(o["offset"] != "open" for o in p["orders"])
    assert p["suppressed_open_lots"] == 3
    assert p["risk"]["current_margin_ratio"] == 1.1


def test_cold_start_requires_explicit_adoption_if_account_is_not_flat():
    sized, snap, policy = inputs()
    policy["cold_start"] = "require_flat"
    snap["positions"] = [{"root": "RB", "contract": "RB2701", "exchange": "SHFE",
                          "side": "long", "volume": 3, "today": 0, "yesterday": 3,
                          "available_today": 0, "available_yesterday": 3}]
    assert "COLD_START_NOT_FLAT" in plan(sized, snap, policy)["blockers"]


def test_nonpositive_equity_negative_available_do_not_prohibit_risk_reduction():
    sized, snap, policy = inputs()
    policy["old_contract_specs"] = {"RB2610": {"tick": 1, "close": 3500, "metadata_verified": True}}
    snap.update(equity=-1, available=-11000, margin_used=11000)
    snap["positions"] = [{"root": "RB", "contract": "RB2610", "exchange": "SHFE",
                          "side": "long", "volume": 2, "today": 0, "yesterday": 2,
                          "available_today": 0, "available_yesterday": 2}]
    p = plan(sized, snap, policy)
    assert p["ready"] and p["risk_mode"] == "reduce_only"
    assert p["risk"]["current_margin_ratio"] is None
    result = paper_execute(p, snap, now=NOW)
    assert result["snapshot"]["positions"] == []
    assert result["snapshot"]["margin_used"] == 0


def test_contract_aliases_cannot_create_false_rollovers_or_wrong_close_offsets():
    sized, snap, policy = inputs()
    pos = {"root": "rb", "contract": "rb2701", "exchange": "SHFE", "side": "long",
           "volume": 3, "today": 0, "yesterday": 3, "available_today": 0, "available_yesterday": 3}
    snap["positions"] = [pos]
    with pytest.raises(ValueError, match="canonical"):
        plan(sized, snap, policy)
    pos.update(root="RB", contract="RB2701", exchange="DCE")
    assert "POSITION_EXCHANGE_MISMATCH" in plan(sized, snap, policy)["blockers"]


def test_panda_channel_does_not_mark_unimplemented_close_offsets_ready():
    sized, snap, policy = inputs()
    policy["channel"] = "panda"
    snap["positions"] = [{"root": "RB", "contract": "RB2610", "broker_contract": "rb2610", "exchange": "SHFE",
                          "side": "long", "volume": 3, "today": 0, "yesterday": 3,
                          "available_today": 0, "available_yesterday": 3}]
    p = plan(sized, snap, policy)
    assert not p["ready"] and "PANDA_CLOSE_OFFSET_UNVERIFIED" in p["blockers"]
    assert p["orders"][0]["broker_contract"] == "rb2610"
    assert p["execution_phase"] == "close_then_reconcile"
