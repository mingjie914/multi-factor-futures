from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import core.config as core_config
import data.manager as data_manager
import run_trading_workflow as workflow
import trading.automation as automation
import trading.weights as weights_module
from trading.artifacts import digest, read_artifact, write_artifact


UTC = timezone.utc
SHANGHAI = timezone(timedelta(hours=8))
TRADE_DATE = "2026-09-15"
SIGNAL_DATE = "2026-09-14"
HOLIDAY_DATE = "2026-09-16"
START_UTC = datetime(2026, 9, 15, 0, 0, tzinfo=UTC)
AFTER_CUTOFF_UTC = datetime(2026, 9, 15, 2, 21, tzinfo=UTC)
SCHEDULE = {
    "prepare_at": "08:00:00",
    "freeze_at": "10:00:00",
    "preflight_at": "10:05:00",
    "start_at": "10:10:00",
    "submit_until": "10:20:00",
    "completion_by": "10:30:00",
}
FAKE_RUNTIME = {"python": "test-runtime"}
FAKE_STRATEGIES = [{"strategy_id": "S", "panel_start": "2026-01-01"}]


class _Clock:
    def __init__(self, current: datetime = START_UTC):
        self.current = current
        self.sleeps: list[float] = []

    def __call__(self) -> datetime:
        return self.current

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(float(seconds))
        self.current += timedelta(seconds=float(seconds))


class _PatchedDateTime(datetime):
    clock: _Clock | None = None

    @classmethod
    def now(cls, tz=None):
        current = cls.clock.current
        return current.astimezone(tz) if tz is not None else current.replace(tzinfo=None)


def _account() -> dict:
    return {
        "equity": 1_000_000.0,
        "available": 1_000_000.0,
        "margin_used": 0.0,
        "frozen_margin": 0.0,
        "reserve": 0.0,
        "max_margin_ratio": 1.0,
        "max_gross_exposure": 10.0,
        "max_abs_net_exposure": 10.0,
        "require_one_lot": False,
    }


def _settings(tmp_path, *, execute_enabled: bool = True, contest_verified: bool = True) -> tuple[dict, Path]:
    calendar_path = tmp_path / "calendar.json"
    calendar_path.write_text(
        json.dumps(
            {
                "valid_from": "2026-09-01",
                "valid_through": "2026-09-30",
                "trading_days": [SIGNAL_DATE, TRADE_DATE],
                "source": "synthetic-calendar",
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "trading.yaml"
    config_path.write_text("synthetic: true\n", encoding="utf-8")
    return (
        {
            "schema_version": 1,
            "mode": "simulation",
            "output_root": str(tmp_path / "output"),
            "catalog": "synthetic-catalog",
            "strategy_ids": {"S": "synthetic-strategy"},
            "accounts": {"A": _account()},
            "routes": [
                {"strategy_id": "S", "account_id": "A", "capital_basis": "equity", "amount": 1_000.0}
            ],
            "execution": {
                "A": {
                    "expected_identity": "A",
                    "managed_roots": ["RB"],
                    "max_snapshot_age_seconds": 300,
                    "channel": "panda",
                    "allow_simulation_strategy": True,
                    "execute_enabled": execute_enabled,
                }
            },
            "automation": {
                **SCHEDULE,
                "enabled": True,
                "calendar_path": str(calendar_path),
                "sdk_python": str(tmp_path / "synthetic-sdk-python.exe"),
                "preparation_timeout_seconds": 7_200.0,
                "contest_execution_verified": contest_verified,
                "poll_interval_seconds": 2,
                "max_attempts_per_leg": 3,
            },
        },
        config_path,
    )


def _weights_artifact(
    tmp_path,
    *,
    code_sha256: str = "CODE",
    data_fingerprint: str = "DATA",
    runtime: dict | None = None,
    strategies: list[dict] | None = None,
):
    identity = {
        "stage": "weights",
        "mode": "simulation",
        "data_date": SIGNAL_DATE,
        "code_sha256": code_sha256,
        "data_fingerprint": data_fingerprint,
        "runtime": FAKE_RUNTIME if runtime is None else runtime,
        "strategies": FAKE_STRATEGIES if strategies is None else strategies,
        "config": {"market": "futures", "data": {"source": "random"}},
    }
    return write_artifact(
        tmp_path / "weights",
        identity,
        {
            "weights.csv": [
                {
                    "strategy_id": "S",
                    "root": "RB",
                    "contract": "RB2610",
                    "exchange": "SHFE",
                    "close": 10.0,
                    "weight": 1.0,
                    "data_date": SIGNAL_DATE,
                }
            ]
        },
    )


def _sizing_artifact(tmp_path, weights_path, *, revision: int) -> Path:
    return write_artifact(
        tmp_path / "sizing",
        {"stage": "size", "weights_id": weights_path.name, "revision": revision},
        {
            "sizing.json": {
                "accounts": {
                    "A": {
                        "tradable": True,
                        "blockers": [],
                        "targets": [
                            {"contract": "RB2610", "root": "RB", "exchange": "SHFE", "target_lots": 1}
                        ],
                    }
                }
            }
        },
    )


class _Harness:
    def __init__(
        self,
        monkeypatch,
        tmp_path,
        *,
        clock: _Clock | None = None,
        code_sha256: str = "CODE",
        data_fingerprint: str = "DATA",
        subprocess_returncode: int = 0,
    ):
        self.clock = _Clock() if clock is None else clock
        self.events: list[tuple] = []
        self.snapshot_calls = 0
        self.subprocess_calls = 0
        self.sizing_calls = 0
        self.session_calls = 0
        self.subprocess_returncode = subprocess_returncode
        self.weights_path = _weights_artifact(
            tmp_path,
            code_sha256=code_sha256,
            data_fingerprint=data_fingerprint,
        )

        _PatchedDateTime.clock = self.clock
        monkeypatch.setattr(workflow, "datetime", _PatchedDateTime)
        monkeypatch.setattr(automation, "datetime", _PatchedDateTime)
        monkeypatch.setattr(workflow, "time", SimpleNamespace(sleep=self.clock.sleep))
        monkeypatch.setattr(workflow, "panda_call", self.panda_call)
        monkeypatch.setattr(workflow.subprocess, "run", self.subprocess_run)
        monkeypatch.setattr(workflow, "size_artifact", self.size_artifact)
        monkeypatch.setattr(weights_module, "code_fingerprint", lambda: "CODE")
        monkeypatch.setattr(weights_module, "runtime_contract", lambda: FAKE_RUNTIME)
        monkeypatch.setattr(weights_module, "strategy_contracts", lambda strategy_ids, catalog: FAKE_STRATEGIES)

        class _Config:
            def model_dump(self, mode="json"):
                return {"synthetic": True}

        monkeypatch.setattr(core_config, "load_config", lambda path: _Config())

        harness = self

        class _Source:
            def checkpoint_source_fingerprint(self, calendar_start, signal_date):
                harness.events.append(("source_fingerprint", calendar_start, signal_date))
                return data_fingerprint

            def close(self):
                harness.events.append(("source_close",))

        class _Manager:
            def __init__(self):
                self.source = _Source()

            @classmethod
            def from_config(cls, config):
                harness.events.append(("market_open",))
                return cls()

            def get_calendar(self, panel_start, signal_date):
                harness.events.append(("market_calendar", panel_start, signal_date))
                return ["synthetic-calendar-start"]

        monkeypatch.setattr(data_manager, "DataManager", _Manager)
        original_write_state = automation.write_state

        def record_state(path, value):
            if Path(path).name == "runner.json" and isinstance(value, dict) and "phase" in value:
                self.events.append(("status", value["phase"]))
            return original_write_state(path, value)

        monkeypatch.setattr(automation, "write_state", record_state)

    def snapshot(self):
        self.snapshot_calls += 1
        self.events.append(("snapshot", self.snapshot_calls))
        return {
            "identity": "A",
            "trade_date": TRADE_DATE,
            "as_of": self.clock().isoformat(),
            "positions": [],
            "open_orders": [],
            "equity": 1_000_000.0,
            "available": 1_000_000.0,
            "margin_used": 0.0,
        }

    def panda_call(self, request, *, interpreter=None, timeout=120):
        action = request["action"]
        if action == "snapshot":
            return self.snapshot()
        if action == "session":
            self.session_calls += 1
            self.events.append(("session", request["signal_date"]))
            return {"status": "completed", "reason": None, "remaining": []}
        raise AssertionError(f"unexpected panda action: {action}")

    def subprocess_run(self, command, *, stdout=None, **kwargs):
        self.subprocess_calls += 1
        self.events.append(("weights_process", command))
        if self.subprocess_returncode == 0:
            stdout.write(str(self.weights_path) + "\n")
            stdout.flush()
        return SimpleNamespace(returncode=self.subprocess_returncode)

    def size_artifact(self, weights_path, settings, specs=None, *, specs_as_of=None):
        self.sizing_calls += 1
        self.events.append(("sizing", specs_as_of))
        read_artifact(weights_path)
        return _sizing_artifact(Path(weights_path).parent.parent, Path(weights_path), revision=self.sizing_calls)


def _run(settings, config_path, *, account_id="A", trade_date=TRADE_DATE, prepare_only=False):
    return workflow.run_automatic_day(
        settings,
        config_path=config_path,
        account_id=account_id,
        trade_date=trade_date,
        prepare_only=prepare_only,
    )


@pytest.mark.parametrize(
    ("trade_date", "clock", "phase"),
    [
        (HOLIDAY_DATE, _Clock(), "non_trading_day"),
        (TRADE_DATE, _Clock(AFTER_CUTOFF_UTC), "not_started"),
    ],
)
def test_run_automatic_day_holiday_or_after_cutoff_makes_no_external_calls(
    monkeypatch, tmp_path, trade_date, clock, phase
):
    settings, config_path = _settings(tmp_path)
    harness = _Harness(monkeypatch, tmp_path, clock=clock)

    result = _run(settings, config_path, trade_date=trade_date)

    assert result.name == "runner.json"
    assert json.loads(result.read_text(encoding="utf-8"))["phase"] == phase
    assert harness.snapshot_calls == 0
    assert harness.subprocess_calls == 0
    assert harness.sizing_calls == 0
    assert harness.session_calls == 0


def test_run_automatic_day_prepare_only_writes_real_handoffs_without_session(monkeypatch, tmp_path):
    settings, config_path = _settings(tmp_path)
    harness = _Harness(monkeypatch, tmp_path)

    result = _run(settings, config_path, prepare_only=True)

    state = json.loads(result.read_text(encoding="utf-8"))
    assert result.name == "preparation.json"
    assert state["status"] == "prepared"
    assert harness.snapshot_calls == 2
    assert harness.subprocess_calls == 1
    assert harness.sizing_calls == 1
    assert harness.session_calls == 0
    assert read_artifact(state["weights_directory"])["identity"]["data_date"] == SIGNAL_DATE
    assert read_artifact(state["sizing_directory"])["identity"]["weights_id"] == state["weights_id"]
    assert harness.events[-1][0] != "session"


def test_waiting_runner_refreshes_liveness_and_preserves_phase_history(monkeypatch, tmp_path):
    settings, config_path = _settings(tmp_path, execute_enabled=False)
    _Harness(monkeypatch, tmp_path)
    observed = []
    original = automation.write_state

    def record(path, state):
        if Path(path).name == "runner.heartbeat.json":
            observed.append(dict(state))
        return original(path, state)

    monkeypatch.setattr(automation, "write_state", record)
    result = _run(settings, config_path)
    waiting = [s for s in observed if s["phase"] == "waiting_to_freeze"]
    assert len(waiting) > 2
    assert waiting[-1]["at"] > waiting[0]["at"]
    events = [json.loads(line) for line in (result.parent / "runner.events.jsonl").read_text().splitlines()]
    assert [e["phase"] for e in events][-3:] == ["waiting_to_freeze", "frozen", "prepared_execution_disabled"]
    assert not (result.parent / "runner.lock").exists()


def test_run_automatic_day_full_day_preserves_freeze_preflight_session_order(monkeypatch, tmp_path):
    settings, config_path = _settings(tmp_path)
    harness = _Harness(monkeypatch, tmp_path)

    result = _run(settings, config_path)

    manifest = read_artifact(result)
    phases = [event[1] for event in harness.events if event[0] == "status"]
    assert manifest["identity"]["stage"] == "session"
    assert harness.snapshot_calls == 3
    assert harness.subprocess_calls == 1
    assert harness.sizing_calls == 2
    assert harness.session_calls == 1
    assert harness.clock.current == datetime(2026, 9, 15, 2, 5, tzinfo=UTC)
    assert sum(harness.clock.sleeps) == pytest.approx(7_500.0)
    assert phases == [
        "waiting_to_prepare",
        "computing_weights",
        "prepared",
        "waiting_to_freeze",
        "frozen",
        "waiting_for_execution",
        "completed",
    ]
    assert [event[1] for event in harness.events if event[0] == "sizing"] == [TRADE_DATE, TRADE_DATE]


def test_run_automatic_day_preparation_failure_never_starts_a_session(monkeypatch, tmp_path):
    settings, config_path = _settings(tmp_path)
    harness = _Harness(monkeypatch, tmp_path, subprocess_returncode=1)

    result = _run(settings, config_path)

    runner = json.loads(result.read_text(encoding="utf-8"))
    preparation = json.loads(
        (result.parent / "preparation.json").read_text(encoding="utf-8")
    )
    assert runner["phase"] == "blocked"
    assert runner["reason"] == "LATEST_PREPARATION_FAILED"
    assert preparation["status"] == "blocked"
    assert "WEIGHT_GENERATION_FAILED" in preparation["reason"]
    assert harness.snapshot_calls == 1
    assert harness.subprocess_calls == 1
    assert harness.sizing_calls == 0
    assert harness.session_calls == 0


@pytest.mark.parametrize(
    ("code_sha256", "data_fingerprint", "message"),
    [
        ("CODE", "OLD-DATA", "certified data changed after preparation"),
    ],
)
def test_run_automatic_day_rejects_source_fingerprint_change_before_session(
    monkeypatch, tmp_path, code_sha256, data_fingerprint, message
):
    settings, config_path = _settings(tmp_path)
    harness = _Harness(
        monkeypatch,
        tmp_path,
        code_sha256=code_sha256,
        data_fingerprint=data_fingerprint,
    )
    if data_fingerprint == "OLD-DATA":
        # The certified source reports the current value while the frozen
        # weights artifact retains the old value.
        class _CurrentSource:
            def checkpoint_source_fingerprint(self, calendar_start, signal_date):
                return "CURRENT-DATA"

            def close(self):
                pass

        class _CurrentManager:
            def __init__(self):
                self.source = _CurrentSource()

            @classmethod
            def from_config(cls, config):
                return cls()

            def get_calendar(self, panel_start, signal_date):
                return ["synthetic-calendar-start"]

        monkeypatch.setattr(data_manager, "DataManager", _CurrentManager)

    with pytest.raises(ValueError, match=message):
        _run(settings, config_path)

    assert harness.session_calls == 0
    assert harness.snapshot_calls == 2
    assert harness.sizing_calls == 1
    runner_path = (
        tmp_path
        / "output"
        / "automation"
        / digest({"identity": "A"})
        / TRADE_DATE
        / "runner.json"
    )
    assert json.loads(runner_path.read_text(encoding="utf-8"))["phase"] == "blocked"


def test_completed_weights_survive_later_research_code_and_config_changes(monkeypatch, tmp_path):
    settings, config_path = _settings(tmp_path, execute_enabled=False)
    harness = _Harness(monkeypatch, tmp_path, code_sha256="PRIOR-RESEARCH-CODE")

    def research_was_changed(*args, **kwargs):
        raise AssertionError("completed weights must not be reinterpreted through live research settings")

    monkeypatch.setattr(core_config, "load_config", research_was_changed)
    monkeypatch.setattr(weights_module, "strategy_contracts", research_was_changed)
    monkeypatch.setattr(weights_module, "code_fingerprint", research_was_changed)
    result = _run(settings, config_path)
    assert json.loads(result.read_text(encoding="utf-8"))["phase"] == "prepared_execution_disabled"
    assert harness.snapshot_calls == 3
    assert harness.session_calls == 0


def test_run_automatic_day_prepares_but_never_starts_session_when_execution_disabled(monkeypatch, tmp_path):
    settings, config_path = _settings(tmp_path, execute_enabled=False)
    harness = _Harness(monkeypatch, tmp_path)

    result = _run(settings, config_path)

    runner = json.loads(result.read_text(encoding="utf-8"))
    assert result.name == "runner.json"
    assert runner["phase"] == "prepared_execution_disabled"
    assert harness.snapshot_calls == 3
    assert harness.sizing_calls == 2
    assert harness.session_calls == 0


def test_preview_date_cannot_submit_even_when_execution_switches_are_enabled(monkeypatch, tmp_path):
    settings, config_path = _settings(tmp_path, execute_enabled=True, contest_verified=True)
    settings["automation"]["preview_only_dates"] = [TRADE_DATE]
    harness = _Harness(monkeypatch, tmp_path)

    result = _run(settings, config_path)

    runner = json.loads(result.read_text(encoding="utf-8"))
    assert runner["phase"] == "preview_day_no_submission"
    assert runner["reason"] == "EXPLICIT_PREVIEW_ONLY_DATE"
    assert harness.sizing_calls == 2
    assert harness.session_calls == 0
