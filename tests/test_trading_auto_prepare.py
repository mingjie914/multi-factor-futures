from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import pytest

from trading.artifacts import read_artifact, write_artifact
from trading.automation import prepare_day


SHANGHAI = timezone(timedelta(hours=8))
TRADE_DATE = "2026-09-15"
SIGNAL_DATE = "2026-09-14"
NOW = datetime(2026, 9, 15, 9, 0, tzinfo=SHANGHAI)
FREEZE = datetime(2026, 9, 15, 10, 0, tzinfo=SHANGHAI)
SCHEDULE = {
    "prepare_at": "08:00:00",
    "freeze_at": "10:00:00",
    "preflight_at": "10:05:00",
    "start_at": "10:10:00",
    "submit_until": "10:20:00",
    "completion_by": "10:30:00",
}


class _Clock:
    def __init__(self, current: datetime = NOW):
        self.current = current

    def __call__(self) -> datetime:
        return self.current


def _settings() -> dict:
    return {
        "automation": {**SCHEDULE, "preparation_timeout_seconds": 7_200.0},
        "execution": {
            "A": {
                "expected_identity": "A",
                "managed_roots": ["RB"],
                "max_snapshot_age_seconds": 300,
            }
        },
        "routes": [
            {
                "strategy_id": "S",
                "account_id": "A",
                "capital_basis": "equity",
                "amount": 1_000.0,
            }
        ],
    }


def _calendar() -> dict:
    return {
        "valid_from": "2026-09-01",
        "valid_through": "2026-09-30",
        "trading_days": [SIGNAL_DATE, TRADE_DATE],
        "source": "synthetic-calendar",
    }


def _snapshot(
    *,
    identity: str = "A",
    positions: list[dict] | None = None,
    open_orders: list[dict] | None = None,
) -> dict:
    return {
        "identity": identity,
        "trade_date": TRADE_DATE,
        "as_of": NOW.isoformat(),
        "positions": [] if positions is None else positions,
        "open_orders": [] if open_orders is None else open_orders,
        "equity": 1_000_000.0,
        "available": 1_000_000.0,
        "margin_used": 0.0,
    }


def _position(volume: int = 1) -> dict:
    return {
        "root": "RB",
        "contract": "RB2610",
        "exchange": "SHFE",
        "side": "long",
        "volume": volume,
        "today": volume,
        "yesterday": 0,
        "available_today": volume,
        "available_yesterday": 0,
    }


def _weights_artifact(tmp_path, *, data_date: str = SIGNAL_DATE):
    return write_artifact(
        tmp_path / "weights",
        {"stage": "weights", "data_date": data_date, "mode": "simulation", "fixture": "prepare"},
        {
            "weights.csv": [
                {
                    "strategy_id": "S",
                    "root": "RB",
                    "contract": "RB2610",
                    "exchange": "SHFE",
                    "close": 10.0,
                    "weight": 1.0,
                    "data_date": data_date,
                }
            ]
        },
    )


def _sizing_artifact(tmp_path, weights, *, specs_as_of: str = TRADE_DATE):
    return write_artifact(
        tmp_path / "sizing",
        {"stage": "size", "weights_id": weights.name, "specs_as_of": specs_as_of},
        {"sizing.json": {"accounts": {"A": {"tradable": True, "blockers": []}}}},
    )


def _read_state(state_path):
    return json.loads(state_path.read_text(encoding="utf-8"))


def test_prepare_day_handoffs_snapshot_weights_snapshot_size_with_budget_and_dates(tmp_path):
    clock = _Clock()
    snapshots = [_snapshot(), _snapshot()]
    events = []

    def snapshot_fn():
        events.append("snapshot")
        return deepcopy(snapshots.pop(0))

    def weights_fn(signal_date, holdings, timeout_seconds):
        events.append(("weights", signal_date, holdings, timeout_seconds))
        assert signal_date == SIGNAL_DATE
        assert holdings["S"]["data_date"] == SIGNAL_DATE
        assert timeout_seconds == pytest.approx(3_600.0)
        return _weights_artifact(tmp_path)

    def sizing_fn(weights, fresh_snapshot, trade_date):
        events.append(("sizing", trade_date, fresh_snapshot))
        assert read_artifact(weights)["identity"]["data_date"] == SIGNAL_DATE
        assert trade_date == TRADE_DATE
        return _sizing_artifact(tmp_path, weights, specs_as_of=TRADE_DATE)

    state_path = tmp_path / "preparation.json"
    state = prepare_day(
        _settings(),
        account_id="A",
        trade_date=TRADE_DATE,
        calendar=_calendar(),
        state_path=state_path,
        snapshot_fn=snapshot_fn,
        weights_fn=weights_fn,
        sizing_fn=sizing_fn,
        clock=clock,
    )

    assert [event if isinstance(event, str) else event[0] for event in events] == [
        "snapshot",
        "weights",
        "snapshot",
        "sizing",
    ]
    assert state["status"] == "prepared"
    assert state["signal_date"] == SIGNAL_DATE
    assert state["preparation_budget_seconds"] == pytest.approx(3_600.0)
    assert _read_state(state_path)["status"] == "prepared"


@pytest.mark.parametrize(
    ("late_stage", "reason"),
    [
        ("weights", "WEIGHTS_FINISHED_AFTER_FREEZE"),
        ("sizing", "PREPARATION_FINISHED_AFTER_FREEZE"),
    ],
)
def test_prepare_day_rejects_after_freeze_and_overwrites_old_success_state(
    tmp_path, late_stage, reason
):
    clock = _Clock()
    snapshots = [_snapshot(), _snapshot()]
    state_path = tmp_path / "preparation.json"
    state_path.write_text(
        json.dumps(
            {
                "status": "prepared",
                "weights_directory": "old-success-weights",
                "sizing_directory": "old-success-sizing",
            }
        ),
        encoding="utf-8",
    )

    def snapshot_fn():
        return deepcopy(snapshots.pop(0))

    def weights_fn(signal_date, holdings, timeout_seconds):
        path = _weights_artifact(tmp_path)
        if late_stage == "weights":
            clock.current = FREEZE
        return path

    def sizing_fn(weights, fresh_snapshot, trade_date):
        path = _sizing_artifact(tmp_path, weights)
        if late_stage == "sizing":
            clock.current = FREEZE
        return path

    with pytest.raises(ValueError, match=reason):
        prepare_day(
            _settings(),
            account_id="A",
            trade_date=TRADE_DATE,
            calendar=_calendar(),
            state_path=state_path,
            snapshot_fn=snapshot_fn,
            weights_fn=weights_fn,
            sizing_fn=sizing_fn,
            clock=clock,
        )

    state = _read_state(state_path)
    assert state["status"] == "blocked"
    assert state["reason"] == reason
    assert "old-success" not in json.dumps(state)


def test_prepare_day_rejects_holdings_change_between_weight_and_sizing(tmp_path):
    clock = _Clock()
    snapshots = [_snapshot(), _snapshot(positions=[_position()])]
    events = []

    def snapshot_fn():
        events.append("snapshot")
        return deepcopy(snapshots.pop(0))

    def weights_fn(signal_date, holdings, timeout_seconds):
        events.append("weights")
        return _weights_artifact(tmp_path)

    def sizing_fn(weights, fresh_snapshot, trade_date):
        events.append("sizing")
        return _sizing_artifact(tmp_path, weights)

    state_path = tmp_path / "preparation.json"
    with pytest.raises(ValueError, match="HOLDINGS_CHANGED_DURING_WEIGHT_PREPARATION"):
        prepare_day(
            _settings(),
            account_id="A",
            trade_date=TRADE_DATE,
            calendar=_calendar(),
            state_path=state_path,
            snapshot_fn=snapshot_fn,
            weights_fn=weights_fn,
            sizing_fn=sizing_fn,
            clock=clock,
        )

    assert events == ["snapshot", "weights", "snapshot"]
    state = _read_state(state_path)
    assert state["status"] == "blocked"
    assert state["reason"] == "HOLDINGS_CHANGED_DURING_WEIGHT_PREPARATION"


@pytest.mark.parametrize("wrong_date", [TRADE_DATE, "2026-09-11"])
def test_prepare_day_rejects_wrong_signal_date_artifact(tmp_path, wrong_date):
    clock = _Clock()
    sizing_calls = []

    def weights_fn(signal_date, holdings, timeout_seconds):
        return _weights_artifact(tmp_path, data_date=wrong_date)

    def sizing_fn(weights, fresh_snapshot, trade_date):
        sizing_calls.append(True)
        return _sizing_artifact(tmp_path, weights)

    state_path = tmp_path / "preparation.json"
    with pytest.raises(ValueError, match="WEIGHTS_NOT_FOR_EXPECTED_SIGNAL_DATE"):
        prepare_day(
            _settings(),
            account_id="A",
            trade_date=TRADE_DATE,
            calendar=_calendar(),
            state_path=state_path,
            snapshot_fn=lambda: _snapshot(),
            weights_fn=weights_fn,
            sizing_fn=sizing_fn,
            clock=clock,
        )

    assert not sizing_calls
    assert _read_state(state_path)["status"] == "blocked"


def test_prepare_day_rejects_tampered_immutable_weight_artifact(tmp_path):
    clock = _Clock()

    def weights_fn(signal_date, holdings, timeout_seconds):
        path = _weights_artifact(tmp_path)
        (path / "weights.csv").write_text("tampered\n", encoding="utf-8")
        return path

    state_path = tmp_path / "preparation.json"
    with pytest.raises(ValueError, match="artifact file hash mismatch"):
        prepare_day(
            _settings(),
            account_id="A",
            trade_date=TRADE_DATE,
            calendar=_calendar(),
            state_path=state_path,
            snapshot_fn=lambda: _snapshot(),
            weights_fn=weights_fn,
            sizing_fn=lambda weights, fresh_snapshot, trade_date: _sizing_artifact(tmp_path, weights),
            clock=clock,
        )

    assert _read_state(state_path)["status"] == "blocked"


@pytest.mark.parametrize(
    ("snapshot_kwargs", "reason"),
    [
        ({"identity": "OTHER"}, "ACCOUNT_OR_TRADE_DATE_MISMATCH"),
        ({"open_orders": [{"order_id": "active"}]}, "ACTIVE_ORDERS_BEFORE_WEIGHT_PREPARATION"),
    ],
)
def test_prepare_day_rejects_wrong_account_or_active_orders_before_weights(
    tmp_path, snapshot_kwargs, reason
):
    clock = _Clock()
    calls = []

    def weights_fn(signal_date, holdings, timeout_seconds):
        calls.append("weights")
        return _weights_artifact(tmp_path)

    def sizing_fn(weights, fresh_snapshot, trade_date):
        calls.append("sizing")
        return _sizing_artifact(tmp_path, weights)

    state_path = tmp_path / "preparation.json"
    with pytest.raises(ValueError, match=reason):
        prepare_day(
            _settings(),
            account_id="A",
            trade_date=TRADE_DATE,
            calendar=_calendar(),
            state_path=state_path,
            snapshot_fn=lambda: _snapshot(**snapshot_kwargs),
            weights_fn=weights_fn,
            sizing_fn=sizing_fn,
            clock=clock,
        )

    assert calls == []
    assert _read_state(state_path)["status"] == "blocked"
