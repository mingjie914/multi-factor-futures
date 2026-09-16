from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path

import pytest

from trading.artifacts import digest
from trading.automation import (
    dated_schedule,
    position_key,
    prepare_budget,
    previous_trade_date,
    refresh_fixed_targets,
    run_rebalance,
)
from trading.execution import build_plan, validate_plan
from trading.weights import panel_cache_key


SHANGHAI = timezone(timedelta(hours=8))
TRADE_DATE = "2026-09-15"
SCHEDULE_CONFIG = {
    "prepare_at": "08:45:00",
    "freeze_at": "14:30:00",
    "preflight_at": "14:35:00",
    "start_at": "14:40:00",
    "submit_until": "14:50:00",
    "completion_by": "15:00:00",
}


def _config(**updates) -> dict:
    result = dict(SCHEDULE_CONFIG)
    result.update(updates)
    return result


def _calendar(**updates) -> dict:
    result = {
        "valid_from": "2026-09-01",
        "valid_through": "2026-09-30",
        "trading_days": ["2026-09-11", "2026-09-15", "2026-09-16"],
        "source": "synthetic-shanghai-calendar",
    }
    result.update(updates)
    return result


def _shanghai(hour: int, minute: int = 0, second: int = 0, *, day: int = 15) -> datetime:
    return datetime(2026, 9, day, hour, minute, second, tzinfo=SHANGHAI)


def test_dated_schedule_returns_all_times_on_the_trade_date_with_timezone():
    schedule = dated_schedule(TRADE_DATE, SCHEDULE_CONFIG)

    assert list(schedule) == list(SCHEDULE_CONFIG)
    assert all(value.tzinfo is not None for value in schedule.values())
    assert all(value.utcoffset() == timedelta(hours=8) for value in schedule.values())
    assert [value.strftime("%H:%M:%S") for value in schedule.values()] == list(SCHEDULE_CONFIG.values())
    assert {value.date().isoformat() for value in schedule.values()} == {TRADE_DATE}
    assert schedule["submit_until"] == schedule["completion_by"] - timedelta(minutes=10)


def test_dated_schedule_requires_each_named_time_field():
    for field in SCHEDULE_CONFIG:
        config = _config()
        del config[field]
        with pytest.raises(ValueError):
            dated_schedule(TRADE_DATE, config)


def test_dated_schedule_requires_strict_hh_mm_ss_values():
    for field, value in (
        ("prepare_at", "8:45:00"),
        ("freeze_at", "14:30"),
        ("start_at", "24:00:00"),
    ):
        with pytest.raises(ValueError):
            dated_schedule(TRADE_DATE, _config(**{field: value}))


def test_dated_schedule_enforces_order_through_submit_and_completion():
    invalid_configs = (
        _config(freeze_at="08:45:00"),
        _config(preflight_at="14:29:00"),
        _config(start_at="14:34:00"),
        _config(submit_until="14:39:00"),
        _config(completion_by="14:49:00"),
    )
    for config in invalid_configs:
        with pytest.raises(ValueError):
            dated_schedule(TRADE_DATE, config)


def test_dated_schedule_allows_completion_at_the_submit_cutoff():
    schedule = dated_schedule(TRADE_DATE, _config(completion_by="14:50:00"))
    assert schedule["completion_by"] == schedule["submit_until"]


def test_prepare_budget_allows_cold_start_and_caps_a_13h_recalculation():
    schedule = dated_schedule(TRADE_DATE, SCHEDULE_CONFIG)

    assert prepare_budget(schedule, _shanghai(8, 45), 900.0) == pytest.approx(900.0)
    # At 13:00 there are exactly 90 minutes before the 14:30 freeze.
    assert prepare_budget(schedule, _shanghai(13), 10_000.0) == pytest.approx(5_400.0)


def test_prepare_budget_rejects_before_prepare_and_at_or_after_freeze():
    schedule = dated_schedule(TRADE_DATE, SCHEDULE_CONFIG)
    for now in (_shanghai(8, 44, 59), _shanghai(14, 30), _shanghai(14, 30, 1)):
        with pytest.raises(ValueError):
            prepare_budget(schedule, now, 60.0)


def test_prepare_budget_requires_timezone_and_the_same_shanghai_date():
    schedule = dated_schedule(TRADE_DATE, SCHEDULE_CONFIG)
    invalid_now = (
        datetime(2026, 9, 15, 13, 0),
        datetime(2026, 9, 14, 16, 0, tzinfo=timezone.utc),
    )
    for now in invalid_now:
        with pytest.raises(ValueError):
            prepare_budget(schedule, now, 60.0)


def test_prepare_budget_rejects_nonfinite_nonpositive_or_boolean_timeout():
    schedule = dated_schedule(TRADE_DATE, SCHEDULE_CONFIG)
    for timeout in (0, -1, math.inf, math.nan, True):
        with pytest.raises(ValueError):
            prepare_budget(schedule, _shanghai(13), timeout)


def test_previous_trade_date_skips_weekend_and_holiday_using_the_supplied_calendar():
    calendar = _calendar(trading_days=["2026-09-10", "2026-09-11", "2026-09-15"])
    assert previous_trade_date("2026-09-15", calendar) == "2026-09-11"


def test_previous_trade_date_requires_the_execution_date_to_be_listed():
    calendar = _calendar(trading_days=["2026-09-11", "2026-09-15"])
    for non_trading_day in ("2026-09-12", "2026-09-14"):
        with pytest.raises(ValueError):
            previous_trade_date(non_trading_day, calendar)


def test_previous_trade_date_rejects_dates_outside_calendar_bounds():
    calendar = _calendar()
    for outside in ("2026-08-31", "2026-10-01"):
        with pytest.raises(ValueError):
            previous_trade_date(outside, calendar)


def test_previous_trade_date_rejects_missing_or_empty_calendar_metadata():
    for field in ("valid_from", "valid_through", "trading_days", "source"):
        calendar = _calendar()
        if field == "source":
            calendar[field] = ""
        else:
            del calendar[field]
        with pytest.raises(ValueError):
            previous_trade_date("2026-09-15", calendar)


def test_previous_trade_date_rejects_invalid_calendar_dates_and_range():
    invalid_calendars = (
        _calendar(valid_from="2026-09-31"),
        _calendar(valid_through="2026-09-00"),
        _calendar(trading_days=["2026-09-11", "not-a-date", "2026-09-15"]),
        _calendar(trading_days=["2026-09-11", "2026-10-01", "2026-09-15"]),
    )
    for calendar in invalid_calendars:
        with pytest.raises(ValueError):
            previous_trade_date("2026-09-15", calendar)


def test_previous_trade_date_rejects_unsorted_or_duplicate_trading_days():
    for days in (
        ["2026-09-15", "2026-09-11"],
        ["2026-09-11", "2026-09-11", "2026-09-15"],
    ):
        with pytest.raises(ValueError):
            previous_trade_date("2026-09-15", _calendar(trading_days=days))


def test_previous_trade_date_requires_a_prior_listed_day():
    calendar = _calendar(trading_days=["2026-09-15"])
    with pytest.raises(ValueError):
        previous_trade_date("2026-09-15", calendar)


# ---------------------------------------------------------------------------
# Offline rebalance/controller contract fixtures

REBALANCE_NOW = datetime(2026, 9, 15, 9, 0, tzinfo=SHANGHAI)
REBALANCE_TRADE_DATE = "2026-09-15"
REBALANCE_SIGNAL_DATE = "2026-09-14"


class _FakeClock:
    def __init__(self, current: datetime = REBALANCE_NOW):
        self.current = current

    def __call__(self) -> datetime:
        return self.current

    def sleep(self, seconds: float) -> None:
        self.current += timedelta(seconds=float(seconds))


def _rebalance_policy(
    *,
    submit_after_minutes: int = 10,
    completion_after_minutes: int = 20,
    poll_interval_seconds: int = 2,
    max_attempts_per_leg: int = 3,
) -> dict:
    stop = REBALANCE_NOW + timedelta(minutes=submit_after_minutes)
    completion = REBALANCE_NOW + timedelta(minutes=completion_after_minutes)
    return {
        "expected_identity": "A",
        "managed_roots": ["M", "P"],
        "max_snapshot_age_seconds": 300,
        "plan_ttl_seconds": 60,
        "order_type": "market",
        "max_order_lots": 100,
        "max_price_deviation": 0.05,
        "limit_prices": {},
        "channel": "paper",
        "cold_start": "adopt_managed",
        "expected_signal_date": REBALANCE_SIGNAL_DATE,
        "execution_window": {
            "start_at": REBALANCE_NOW.isoformat(),
            "submit_until": stop.isoformat(),
            "completion_by": completion.isoformat(),
        },
        "poll_interval_seconds": poll_interval_seconds,
        "max_attempts_per_leg": max_attempts_per_leg,
    }


def _target(
    contract: str = "M2701",
    *,
    root: str = "M",
    lots: int = 1,
    close: float = 100.0,
    margin_rate: float = 0.1,
) -> dict:
    return {
        "root": root,
        "contract": contract,
        "exchange": "DCE",
        "target_lots": lots,
        "tick": 1.0,
        "close": close,
        "multiplier": 1.0,
        "margin_rate": margin_rate,
    }


def _sized(
    targets: list[dict] | None = None,
    *,
    blockers: list[str] | None = None,
    tradable: bool = True,
    equity: float = 10_000.0,
    max_margin_ratio: float = 0.7,
) -> dict:
    targets = copy.deepcopy(targets or [_target()])
    gross = sum(abs(row["target_lots"] * row["close"] * row["multiplier"]) for row in targets)
    net = sum(row["target_lots"] * row["close"] * row["multiplier"] for row in targets)
    margin = sum(
        abs(row["target_lots"] * row["close"] * row["multiplier"]) * row["margin_rate"]
        for row in targets
    )
    return {
        "tradable": tradable,
        "blockers": list(blockers or []),
        "account": {
            "equity": equity,
            "available": equity,
            "margin_used": 0.0,
            "frozen_margin": 0.0,
            "reserve": 0.0,
            "max_margin_ratio": max_margin_ratio,
            "max_gross_exposure": 10.0,
            "max_abs_net_exposure": 10.0,
            "require_one_lot": False,
        },
        "summary": {
            "gross_notional": gross,
            "net_notional": net,
            "estimated_margin": margin,
            "gross_exposure": gross / equity,
            "net_exposure": net / equity,
        },
        "targets": targets,
        "routes": [],
    }


def _position(contract: str, *, root: str = "M", side: str = "long", volume: int = 1) -> dict:
    return {
        "root": root,
        "contract": contract,
        "exchange": "DCE",
        "side": side,
        "volume": volume,
        "today": volume,
        "yesterday": 0,
        "available_today": volume,
        "available_yesterday": 0,
    }


class _FakeBroker:
    """Offline broker: send only receives plan.orders[0], fills happen in poll."""

    def __init__(
        self,
        clock: _FakeClock,
        *,
        positions: list[dict] | None = None,
        outcomes: list[dict] | None = None,
        delayed_snapshot: bool = False,
        release_on_snapshot: int = 3,
        equity: float = 10_000.0,
        margin_used: float = 0.0,
        available: float | None = None,
    ):
        self.clock = clock
        self.positions = copy.deepcopy(positions or [])
        self.outcomes = list(outcomes or [])
        self.delayed_snapshot = delayed_snapshot
        self.release_on_snapshot = release_on_snapshot
        self.equity = equity
        self.margin_used = margin_used
        self.available = equity if available is None else available
        self.submissions: list[dict] = []
        self.plan_order_counts: list[int] = []
        self.poll_calls = 0
        self.snapshot_calls = 0
        self._delayed_fills: list[tuple[dict, int]] = []
        self._receipts: dict[str, dict] = {}

    def snapshot(self) -> dict:
        self.snapshot_calls += 1
        if self.delayed_snapshot and self.snapshot_calls >= self.release_on_snapshot:
            for order, quantity in self._delayed_fills:
                self._apply(order, quantity)
            self._delayed_fills.clear()
        return {
            "identity": "A",
            "trade_date": REBALANCE_TRADE_DATE,
            "as_of": self.clock().isoformat(),
            "positions": copy.deepcopy(self.positions),
            "open_orders": [],
            "equity": self.equity,
            "available": self.available,
            "margin_used": self.margin_used,
            "frozen_margin": 0.0,
        }

    def submit(self, plan: dict) -> dict:
        self.plan_order_counts.append(len(plan["orders"]))
        order = copy.deepcopy(plan["orders"][0])
        self.submissions.append(order)
        receipt = {
            "receipt_id": f"receipt-{len(self.submissions)}",
            "status": "pending",
            "order": order,
        }
        self._receipts[receipt["receipt_id"]] = receipt
        return receipt

    def poll(self, receipt: dict) -> dict:
        self.poll_calls += 1
        outcome = copy.deepcopy(self.outcomes.pop(0)) if self.outcomes else {
            "status": "reconciled",
            "filled_quantity": receipt["order"]["volume"],
        }
        result = {**outcome, "receipt_id": receipt["receipt_id"]}
        if result.get("status") == "reconciled" and "filled_quantity" in result:
            quantity = int(result["filled_quantity"])
            if quantity < 0 or quantity > receipt["order"]["volume"]:
                return {"status": "unknown", "reason": "INVALID_FILLED_QUANTITY"}
            if quantity:
                if self.delayed_snapshot:
                    self._delayed_fills.append((receipt["order"], quantity))
                else:
                    self._apply(receipt["order"], quantity)
        return result

    def _apply(self, order: dict, quantity: int) -> None:
        if order["offset"] == "open":
            side = "long" if order["side"] == "buy" else "short"
            current = next(
                (row for row in self.positions
                 if row["contract"] == order["contract"] and row["side"] == side),
                None,
            )
            if current is None:
                current = _position(order["contract"], root=order["root"], side=side, volume=0)
                self.positions.append(current)
            current["volume"] += quantity
            current["today"] += quantity
            current["available_today"] += quantity
            return
        side = "long" if order["side"] == "sell" else "short"
        current = next(
            row for row in self.positions
            if row["contract"] == order["contract"] and row["side"] == side
        )
        left = quantity
        take_today = min(left, current["available_today"])
        current["available_today"] -= take_today
        current["today"] -= take_today
        current["volume"] -= take_today
        left -= take_today
        if left:
            take_yesterday = min(left, current["available_yesterday"])
            current["available_yesterday"] -= take_yesterday
            current["yesterday"] -= take_yesterday
            current["volume"] -= take_yesterday
        self.positions = [row for row in self.positions if row["volume"]]


def _run_rebalance(
    tmp_path,
    sized: dict,
    broker: _FakeBroker,
    *,
    policy: dict | None = None,
    clock: _FakeClock | None = None,
    prepared_positions_key: str | None = None,
    previous_plan: dict | None = None,
):
    clock = broker.clock if clock is None else clock
    policy = _rebalance_policy() if policy is None else policy
    return run_rebalance(
        sized,
        account_id="A",
        policy=policy,
        signal_id="signal-a",
        signal_date=REBALANCE_SIGNAL_DATE,
        trade_date=REBALANCE_TRADE_DATE,
        journal_root=tmp_path,
        snapshot_fn=broker.snapshot,
        submit_fn=broker.submit,
        poll_fn=broker.poll,
        prepared_positions_key=prepared_positions_key,
        previous_plan=previous_plan,
        clock=clock,
        sleep=clock.sleep,
    )


def test_run_rebalance_empty_account_completes_multiple_targets_and_sends_first_order_only(tmp_path):
    sized = _sized([_target("M2701", lots=1), _target("P2701", root="P", lots=2)])
    clock = _FakeClock()
    broker = _FakeBroker(clock)

    report = _run_rebalance(tmp_path, sized, broker, clock=clock)

    assert report["status"] == "completed"
    assert [order["contract"] for order in broker.submissions] == ["M2701", "P2701"]
    assert broker.plan_order_counts[0] == 2
    assert {row["contract"]: row["volume"] for row in broker.positions} == {
        "M2701": 1,
        "P2701": 2,
    }


def test_run_rebalance_same_existing_targets_completes_without_sending(tmp_path):
    sized = _sized([_target("M2701", lots=2)])
    clock = _FakeClock()
    broker = _FakeBroker(clock, positions=[_position("M2701", volume=2)])

    report = _run_rebalance(tmp_path, sized, broker, clock=clock)

    assert report["status"] == "completed"
    assert broker.submissions == []


def test_run_rebalance_closes_old_contract_before_opening_new_contract(tmp_path):
    sized = _sized([_target("M2702", lots=2)])
    clock = _FakeClock()
    broker = _FakeBroker(clock, positions=[_position("M2701", volume=2)])

    policy = _rebalance_policy()
    policy["old_contract_specs"] = {"M2701": {"tick": 1, "close": 3500, "metadata_verified": True}}
    report = _run_rebalance(tmp_path, sized, broker, clock=clock, policy=policy)

    assert report["status"] == "completed"
    assert [(order["contract"], order["offset"]) for order in broker.submissions] == [
        ("M2701", "close"),
        ("M2702", "open"),
    ]
    assert broker.plan_order_counts[0] == 2


def test_run_rebalance_ioc_partial_fill_replans_only_the_remaining_delta(tmp_path):
    sized = _sized([_target("M2701", lots=3)])
    clock = _FakeClock()
    broker = _FakeBroker(
        clock,
        outcomes=[
            {"status": "reconciled", "filled_quantity": 2},
            {"status": "reconciled", "filled_quantity": 1},
        ],
    )

    report = _run_rebalance(tmp_path, sized, broker, clock=clock)

    assert report["status"] == "completed"
    assert [order["volume"] for order in broker.submissions] == [3, 1]


def test_run_rebalance_pending_poll_does_not_resend_the_same_order(tmp_path):
    sized = _sized([_target("M2701", lots=1)])
    clock = _FakeClock()
    broker = _FakeBroker(clock, outcomes=[{"status": "pending"}, {"status": "reconciled", "filled_quantity": 1}])

    report = _run_rebalance(tmp_path, sized, broker, clock=clock)

    assert report["status"] == "completed"
    assert len(broker.submissions) == 1
    assert broker.poll_calls == 2


def test_run_rebalance_unknown_receipt_stops_without_blind_retry(tmp_path):
    sized = _sized([_target("M2701", lots=1)])
    clock = _FakeClock()
    broker = _FakeBroker(clock, outcomes=[{"status": "unknown"}])

    report = _run_rebalance(tmp_path, sized, broker, clock=clock)

    assert report["status"] == "unknown"
    assert len(broker.submissions) == 1


def test_run_rebalance_pending_order_at_completion_is_explicit_partial(tmp_path):
    sized = _sized([_target("M2701", lots=1)])
    clock = _FakeClock()
    broker = _FakeBroker(clock, outcomes=[{"status": "pending"}] * 1_000)
    policy = _rebalance_policy(submit_after_minutes=3, completion_after_minutes=4)

    report = _run_rebalance(tmp_path, sized, broker, policy=policy, clock=clock)

    assert report["status"] == "unknown"
    assert "COMPLETION_DEADLINE" in report["reason"]
    assert len(broker.submissions) == 1


class _FillThenDeadlineBroker(_FakeBroker):
    def __init__(self, clock: _FakeClock, *, deadline: datetime, **kwargs):
        super().__init__(clock, **kwargs)
        self.deadline = deadline

    def poll(self, receipt: dict) -> dict:
        result = super().poll(receipt)
        if result.get("status") == "reconciled":
            self.clock.current = self.deadline
        return result


def test_terminal_fill_syncs_but_remaining_target_hits_submission_deadline_as_partial(tmp_path):
    sized = _sized([_target("M2701", lots=2)])
    clock = _FakeClock()
    stop = REBALANCE_NOW + timedelta(minutes=10)
    broker = _FillThenDeadlineBroker(
        clock,
        deadline=stop,
        outcomes=[{"status": "reconciled", "filled_quantity": 1}],
    )

    report = _run_rebalance(tmp_path, sized, broker, clock=clock)

    assert report["status"] == "partial"
    assert report["reason"] == "SUBMISSION_DEADLINE"
    assert [order["volume"] for order in broker.submissions] == [2]
    assert broker.positions[0]["volume"] == 1


def test_run_rebalance_same_account_day_restart_reuses_session_without_resend(tmp_path):
    sized = _sized([_target("M2701", lots=1)])
    clock = _FakeClock()
    broker = _FakeBroker(clock)
    first = _run_rebalance(tmp_path, sized, broker, clock=clock)
    second_broker = _FakeBroker(clock, positions=broker.positions)

    second = _run_rebalance(tmp_path, sized, second_broker, clock=clock)

    assert first["status"] == "completed"
    assert second["status"] == "completed"
    assert second["replayed"] is False
    assert second_broker.submissions == []


def test_run_rebalance_prepared_position_change_is_blocked_before_sizing(tmp_path):
    sized = _sized([_target("M2701", lots=1)])
    clock = _FakeClock()
    prepared_snapshot = {
        "identity": "A",
        "positions": [],
    }
    prepared_key = position_key(prepared_snapshot)
    broker = _FakeBroker(clock, positions=[_position("M2701", volume=1)])

    report = _run_rebalance(
        tmp_path,
        sized,
        broker,
        clock=clock,
        prepared_positions_key=prepared_key,
    )

    assert report["status"] == "blocked"
    assert report["reason"] == "HOLDINGS_CHANGED_AFTER_PREPARATION"
    assert broker.submissions == []


def test_run_rebalance_over_margin_limit_does_not_top_up_positions(tmp_path):
    sized = _sized([_target("M2701", lots=1)])
    clock = _FakeClock()
    broker = _FakeBroker(clock, equity=1_000.0, available=0.0, margin_used=1_100.0)

    report = _run_rebalance(tmp_path, sized, broker, clock=clock)

    assert report["status"] == "partial"
    assert report["reason"] == "REDUCE_ONLY_TARGET_NOT_REACHABLE"
    assert broker.submissions == []


def test_refresh_fixed_targets_updates_risk_fields_without_changing_frozen_target_lots():
    sized = _sized([_target("M2701", lots=3)], equity=10_000.0)
    snapshot = {
        "identity": "A",
        "trade_date": REBALANCE_TRADE_DATE,
        "as_of": REBALANCE_NOW.isoformat(),
        "positions": [],
        "open_orders": [],
        "equity": 2_000.0,
        "available": 2_000.0,
        "margin_used": 0.0,
        "frozen_margin": 0.0,
    }

    fresh = refresh_fixed_targets(sized, snapshot)

    assert fresh["account"]["equity"] == 2_000.0
    assert fresh["account"]["available"] == 2_000.0
    assert fresh["targets"] == sized["targets"]
    assert fresh["summary"]["gross_notional"] == sized["summary"]["gross_notional"]


def test_run_rebalance_preserves_metadata_blocker_even_when_account_can_trade(tmp_path):
    sized = _sized([_target("M2701", lots=1)], blockers=["METADATA_UNVERIFIED: M2701"], tradable=False)
    clock = _FakeClock()
    broker = _FakeBroker(clock)

    report = _run_rebalance(tmp_path, sized, broker, clock=clock)

    assert report["status"] == "blocked"
    assert "METADATA_UNVERIFIED" in report["reason"]
    assert broker.submissions == []


def test_run_rebalance_existing_account_lock_blocks_session(tmp_path):
    sized = _sized([_target("M2701", lots=1)])
    policy = _rebalance_policy()
    account_dir = Path(tmp_path) / digest({"identity": policy["expected_identity"]})
    account_dir.mkdir(parents=True)
    (account_dir / "execute.lock").write_text("other-process", encoding="utf-8")
    clock = _FakeClock()
    broker = _FakeBroker(clock)

    with pytest.raises(ValueError, match="locked"):
        _run_rebalance(tmp_path, sized, broker, policy=policy, clock=clock)
    assert broker.submissions == []


def test_reconciled_fill_waits_for_snapshot_before_replanning_or_duplicate_send(tmp_path):
    sized = _sized([_target("M2701", lots=1)])
    clock = _FakeClock()
    broker = _FakeBroker(
        clock,
        delayed_snapshot=True,
        release_on_snapshot=3,
        outcomes=[{"status": "reconciled", "filled_quantity": 1}],
    )

    report = _run_rebalance(tmp_path, sized, broker, clock=clock)

    assert report["status"] == "completed"
    assert len(broker.submissions) == 1
    assert broker.snapshot_calls >= 3


def test_reconciled_without_filled_quantity_is_unknown_and_not_retried(tmp_path):
    sized = _sized([_target("M2701", lots=1)])
    clock = _FakeClock()
    broker = _FakeBroker(clock, outcomes=[{"status": "reconciled"}])

    report = _run_rebalance(tmp_path, sized, broker, clock=clock)

    assert report["status"] == "unknown"
    assert len(broker.submissions) == 1


def test_zero_fill_ioc_retry_gets_a_new_intent_and_client_order_id(tmp_path):
    sized = _sized([_target("M2701", lots=1)])
    clock = _FakeClock()
    broker = _FakeBroker(
        clock,
        outcomes=[
            {"status": "reconciled", "filled_quantity": 0},
            {"status": "reconciled", "filled_quantity": 1},
        ],
    )

    report = _run_rebalance(tmp_path, sized, broker, clock=clock)

    assert report["status"] == "completed"
    assert len(broker.submissions) == 2
    assert broker.submissions[0]["client_order_id"] != broker.submissions[1]["client_order_id"]


def test_panel_cache_key_ignores_holdings_but_tracks_data_code_and_compute_start():
    identity = {
        "data_date": REBALANCE_TRADE_DATE,
        "code_sha256": "code-a",
        "data_fingerprint": "data-a",
        "runtime": {"python": "3.12"},
        "config": {"window": 20},
        "holding_states": {"M": 1},
    }
    factors = ["factor-a", "factor-b"]
    base = panel_cache_key(identity, factors, "2026-09-01")
    holdings_changed = panel_cache_key({**identity, "holding_states": {"M": 999}}, factors, "2026-09-01")

    assert base == holdings_changed
    assert panel_cache_key({**identity, "data_fingerprint": "data-b"}, factors, "2026-09-01") != base
    assert panel_cache_key({**identity, "code_sha256": "code-b"}, factors, "2026-09-01") != base
    assert panel_cache_key(identity, factors, "2026-09-02") != base


def _plan_for_mark_to_market_validation() -> tuple[dict, dict]:
    snapshot = _FakeBroker(_FakeClock()).snapshot()
    sized = _sized([_target("M2701", lots=1)])
    policy = _rebalance_policy()
    plan = build_plan(
        sized,
        snapshot,
        account_id="A",
        policy={**policy, "allow_mark_to_market": True},
        signal_id="signal-a",
        signal_date=REBALANCE_SIGNAL_DATE,
        now=REBALANCE_NOW,
    )
    return plan, snapshot


def test_validate_plan_allows_only_mark_to_market_equity_and_available_changes():
    plan, snapshot = _plan_for_mark_to_market_validation()
    now = REBALANCE_NOW + timedelta(seconds=10)
    marked = copy.deepcopy(snapshot)
    marked.update(equity=11_000.0, available=11_000.0, as_of=now.isoformat())

    validate_plan(plan, marked, now=now)


def test_validate_plan_allow_mark_to_market_still_rejects_position_changes():
    plan, snapshot = _plan_for_mark_to_market_validation()
    now = REBALANCE_NOW + timedelta(seconds=10)
    changed = copy.deepcopy(snapshot)
    changed["positions"] = [_position("M2701", volume=1)]
    changed["as_of"] = now.isoformat()

    with pytest.raises(ValueError, match="snapshot"):
        validate_plan(plan, changed, now=now)


def test_validate_plan_allow_mark_to_market_rejects_funding_that_cannot_open_target():
    plan, snapshot = _plan_for_mark_to_market_validation()
    now = REBALANCE_NOW + timedelta(seconds=10)
    underfunded = copy.deepcopy(snapshot)
    underfunded.update(equity=20.0, available=0.0, as_of=now.isoformat())

    with pytest.raises(ValueError, match="risk"):
        validate_plan(plan, underfunded, now=now)


def test_slice_orders_emits_only_one_allowed_first_segment_and_default_blocks():
    sized = _sized([_target("M2701", lots=5)])
    snapshot = _FakeBroker(_FakeClock()).snapshot()
    policy = {**_rebalance_policy(), "max_order_lots": 2}

    sliced = build_plan(
        sized,
        snapshot,
        account_id="A",
        policy={**policy, "slice_orders": True},
        signal_id="signal-a",
        signal_date=REBALANCE_SIGNAL_DATE,
        now=REBALANCE_NOW,
    )
    assert sliced["ready"] is True
    assert len(sliced["orders"]) == 1
    assert sliced["orders"][0]["volume"] == 2
    assert "ORDER_LOT_LIMIT" not in sliced["blockers"]

    unsliced = build_plan(
        sized,
        snapshot,
        account_id="A",
        policy=policy,
        signal_id="signal-a",
        signal_date=REBALANCE_SIGNAL_DATE,
        now=REBALANCE_NOW,
    )
    assert unsliced["ready"] is False
    assert unsliced["orders"][0]["volume"] == 5
    assert "ORDER_LOT_LIMIT" in unsliced["blockers"]
