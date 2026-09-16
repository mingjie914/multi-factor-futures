from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trading.execution import build_plan, validate_execution_window, validate_plan
from trading.panda import execute, normalize_snapshot, prepare


SHANGHAI = timezone(timedelta(hours=8))


def _at(hour: int, minute: int = 0, *, day: int = 15) -> datetime:
    return datetime(2026, 9, day, hour, minute, tzinfo=SHANGHAI)


def _window(
    start: datetime = _at(9),
    submit_until: datetime = _at(9, 30),
    completion_by: datetime = _at(10),
) -> dict:
    return {
        "start_at": start.isoformat(),
        "submit_until": submit_until.isoformat(),
        "completion_by": completion_by.isoformat(),
    }


def _policy(*, execution_window: dict | None = None, execute_enabled: bool = False) -> dict:
    policy = {
        "expected_identity": "paper-one",
        "managed_roots": ["RB"],
        "max_snapshot_age_seconds": 60,
        "plan_ttl_seconds": 600,
        "order_type": "market",
        "max_order_lots": 100,
        "max_price_deviation": 0.05,
        "limit_prices": {},
        "channel": "paper",
        "cold_start": "adopt_managed",
        "execute_enabled": execute_enabled,
    }
    if execution_window is not None:
        policy["execution_window"] = execution_window
        policy["expected_signal_date"] = "2026-09-14"
    return policy


def _sized() -> dict:
    return {
        "tradable": True,
        "blockers": [],
        "account": {
            "equity": 5_000_000.0,
            "available": 5_000_000.0,
            "reserve": 10_000.0,
            "max_margin_ratio": 0.7,
        },
        "summary": {"estimated_margin": 3_600.0},
        "targets": [
            {
                "root": "RB",
                "contract": "RB2701",
                "exchange": "SHFE",
                "target_lots": 1,
                "tick": 1.0,
                "close": 3_600.0,
                "multiplier": 10.0,
                "margin_rate": 0.1,
            }
        ],
    }


def _snapshot(now: datetime, *, identity: str = "paper-one") -> dict:
    return {
        "identity": identity,
        "trade_date": now.astimezone(SHANGHAI).date().isoformat(),
        "as_of": now.isoformat(),
        "positions": [],
        "open_orders": [],
        "equity": 5_000_000.0,
        "available": 5_000_000.0,
        "margin_used": 0.0,
        "frozen_margin": 0.0,
    }


def _build(now: datetime, *, policy: dict | None = None) -> tuple[dict, dict, dict]:
    policy = _policy() if policy is None else policy
    snapshot = _snapshot(now)
    plan = build_plan(
        _sized(),
        snapshot,
        account_id="paper-one",
        policy=policy,
        signal_id="weights-a",
        signal_date=policy.get("expected_signal_date", snapshot["trade_date"]),
        now=now,
    )
    return _sized(), snapshot, plan


def test_missing_execution_window_preserves_legacy_validation_and_ttl():
    now = _at(9, 10)
    policy = _policy()

    assert validate_execution_window(policy, now=now) is None
    _, snapshot, plan = _build(now, policy=policy)
    assert plan["ready"] is True
    assert datetime.fromisoformat(plan["expires_at"]) == now + timedelta(seconds=600)
    validate_plan(plan, snapshot, now=now)


@pytest.mark.parametrize(
    ("now", "allowed"),
    [(_at(9), True), (_at(9, 30), False)],
)
def test_execution_window_start_is_inclusive_and_submit_until_is_exclusive(now, allowed):
    policy = _policy(execution_window=_window())
    if allowed:
        assert validate_execution_window(policy, now=now) == _at(9, 30)
    else:
        with pytest.raises(ValueError, match="window"):
            validate_execution_window(policy, now=now)


@pytest.mark.parametrize("now", [_at(8, 59), _at(10, 1)])
def test_execution_window_rejects_future_and_expired_now(now):
    with pytest.raises(ValueError, match="window"):
        validate_execution_window(_policy(execution_window=_window()), now=now)


@pytest.mark.parametrize("field", ["start_at", "submit_until", "completion_by"])
def test_execution_window_requires_all_three_fields(field):
    window = _window()
    del window[field]
    with pytest.raises(ValueError):
        validate_execution_window(_policy(execution_window=window), now=_at(9, 10))


@pytest.mark.parametrize("field", ["start_at", "submit_until", "completion_by"])
def test_execution_window_requires_timezone_on_each_datetime(field):
    window = _window()
    window[field] = "2026-09-15T09:00:00"
    with pytest.raises(ValueError):
        validate_execution_window(_policy(execution_window=window), now=_at(9, 10))


@pytest.mark.parametrize(
    "window",
    [
        _window(_at(9), _at(9), _at(10)),
        _window(_at(9, 31), _at(9, 30), _at(10)),
        _window(_at(9), _at(10, 1), _at(10)),
    ],
)
def test_execution_window_requires_strict_order(window):
    with pytest.raises(ValueError):
        validate_execution_window(_policy(execution_window=window), now=_at(9, 10))


def test_execution_window_rejects_crossing_an_asia_shanghai_natural_day():
    # The ISO dates are both 15th in UTC, but the submit time is already the
    # 16th in Asia/Shanghai.
    window = _window(
        datetime(2026, 9, 15, 15, 0, tzinfo=timezone.utc),
        datetime(2026, 9, 15, 16, 10, tzinfo=timezone.utc),
        datetime(2026, 9, 15, 16, 30, tzinfo=timezone.utc),
    )
    with pytest.raises(ValueError):
        validate_execution_window(_policy(execution_window=window), now=_at(9, 10))


@pytest.mark.parametrize("signal_date", ["2026-09-15", "2026-09-16"])
def test_expected_signal_date_must_precede_execution_date(signal_date):
    policy = _policy(execution_window=_window())
    policy["expected_signal_date"] = signal_date
    with pytest.raises(ValueError):
        validate_execution_window(policy, now=_at(9, 10))


def test_build_plan_outside_window_is_blocked_without_throwing():
    policy = _policy(execution_window=_window())
    for now in (_at(8, 59), _at(9, 30)):
        _, _, plan = _build(now, policy=policy)
        assert plan["ready"] is False
        assert "OUTSIDE_EXECUTION_WINDOW" in plan["blockers"]


def test_build_plan_inside_window_truncates_ttl_at_submit_until():
    now = _at(9, 20)
    stop = _at(9, 30)
    _, snapshot, plan = _build(now, policy=_policy(execution_window=_window()))

    assert plan["ready"] is True
    assert datetime.fromisoformat(plan["expires_at"]) == stop
    validate_plan(plan, snapshot, now=now)
    with pytest.raises(ValueError, match="window"):
        validate_plan(plan, snapshot, now=stop)


def test_validate_plan_rechecks_window_at_deadline_even_when_plan_expiry_is_inclusive():
    now = _at(9, 10)
    _, snapshot, plan = _build(now, policy=_policy(execution_window=_window()))
    # The legacy expiry comparison accepts now == expires_at; the execution
    # window check must still reject submission at submit_until.
    with pytest.raises(ValueError, match="window"):
        validate_plan(plan, snapshot, now=_at(9, 30))


class _WindowPanda:
    def __init__(self, *, trade_date: str, clock):
        self.trade_date = trade_date
        self.clock = clock
        self.remote_reads = 0
        self.execute_calls = 0
        self.remote = None

    def snapshot(self):
        return {
            "account": {
                "accountId": "paper-one",
                "ready": True,
                "tradeDate": self.trade_date.replace("-", ""),
                "totalProfit": 5_000_000.0,
                "availableFunds": 5_000_000.0,
                "margin": 0.0,
                "frozenCapital": 0.0,
            },
            "positions": [],
            "openOrders": [],
        }

    def prepare_order(self, *args, **kwargs):
        parameters = dict(zip(("contractCode", "side", "offset", "volume"), args))
        parameters.update(clientOrderId=kwargs["client_order_id"])
        if kwargs.get("price") is not None:
            parameters["price"] = kwargs["price"]
        self.remote = {"planId": "remote-one", "status": "prepared", "accountId": "paper-one",
                       "operation": "place_order", "parameters": parameters,
                       "legs": [{**parameters, "preview": {"wouldSucceed": True}}]}
        return self.remote

    def plan(self, plan_id):
        assert plan_id == "remote-one"
        self.remote_reads += 1
        self.clock.current = self.clock.stop
        return self.remote

    def execute_plan(self, plan_id, **kwargs):
        self.execute_calls += 1
        return {"status": "queued", "operationId": "should-not-happen"}


class _AdvancingDateTime(datetime):
    current: datetime
    stop: datetime

    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return cls.current.replace(tzinfo=None)
        return cls.current.astimezone(tz)


def test_panda_remote_plan_read_rechecks_window_before_send(tmp_path, monkeypatch):
    # Build and prepare while inside a short window, then let the remote-plan
    # read advance local time to submit_until.  The send must never be called.
    real_now = datetime.now(timezone.utc)
    inside = real_now
    start = inside - timedelta(minutes=1)
    stop = inside + timedelta(seconds=30)
    completion = stop + timedelta(minutes=5)
    trade_date = inside.astimezone(SHANGHAI).date().isoformat()
    expected_signal_date = (inside.astimezone(SHANGHAI).date() - timedelta(days=1)).isoformat()
    policy = _policy(
        execution_window=_window(start, stop, completion),
        execute_enabled=True,
    )
    policy["expected_signal_date"] = expected_signal_date
    policy["channel"] = "panda"
    policy["max_snapshot_age_seconds"] = 600

    sized = _sized()
    snapshot = _snapshot(inside, identity="paper-one")
    snapshot["trade_date"] = trade_date
    plan = build_plan(
        sized,
        snapshot,
        account_id="paper-one",
        policy=policy,
        signal_id="weights-a",
        signal_date=expected_signal_date,
        now=inside,
    )
    clock = _AdvancingDateTime
    # Keep the synthetic execution clock just after the real snapshot timestamp
    # so the first snapshot-age check remains valid.
    clock.current = inside + timedelta(seconds=2)
    clock.stop = stop
    client = _WindowPanda(trade_date=trade_date, clock=clock)
    prepared = prepare(plan, 0, client)
    monkeypatch.setattr("trading.execution.datetime", clock)

    with pytest.raises(ValueError, match="window"):
        execute(plan, prepared, client, Path(tmp_path), confirmation=prepared["confirmation"])

    assert client.remote_reads == 1
    assert client.execute_calls == 0
