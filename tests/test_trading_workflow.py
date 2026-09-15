from __future__ import annotations

from datetime import datetime, timezone

import pytest

from run_trading_workflow import artifact_json, size_artifact
from trading.artifacts import read_artifact, read_csv, write_artifact
from trading.execution import build_plan, paper_execute


DATA_DATE = "2026-09-15"
NOW = datetime(2026, 9, 15, 8, 0, tzinfo=timezone.utc)


def _account(equity: float = 1_000_000.0) -> dict:
    return {
        "equity": equity,
        "available": equity,
        "margin_used": 0.0,
        "frozen_margin": 0.0,
        "reserve": 0.0,
        "max_margin_ratio": 1.0,
        "max_gross_exposure": 10.0,
        "max_abs_net_exposure": 10.0,
        "require_one_lot": False,
    }


def _spec(contract: str, *, verified: bool = True) -> dict:
    return {
        "contract": contract,
        "exchange": "SHFE",
        "multiplier": 20.0,
        "margin_long": 0.1,
        "margin_short": 0.1,
        "tick": 1.0 if verified else None,
        "as_of": DATA_DATE,
        "source": "synthetic-test-spec",
        "verified": verified,
    }


def _weights() -> list[dict]:
    return [
        {
            "strategy_id": "S1",
            "root": "RB",
            "contract": "RB2610",
            "close": 10.0,
            "weight": 0.5,
            "data_date": DATA_DATE,
        },
        {
            "strategy_id": "S2",
            "root": "CU",
            "contract": "CU2610",
            "close": 10.0,
            "weight": -0.5,
            "data_date": DATA_DATE,
        },
    ]


def _settings(output_root) -> dict:
    return {
        "schema_version": 1,
        "output_root": str(output_root),
        "rounding": "half_up",
        "accounts": {"A1": _account(), "A2": _account()},
        "routes": [
            {"strategy_id": "S1", "account_id": "A1", "capital_basis": "equity", "amount": 1_000.0},
            {"strategy_id": "S2", "account_id": "A2", "capital_basis": "equity", "amount": 1_000.0},
        ],
        "execution": {
            "A1": _policy("A1", ["RB"]),
            "A2": _policy("A2", ["CU"]),
        },
    }


def _policy(account_id: str, managed_roots: list[str]) -> dict:
    return {
        "expected_identity": account_id,
        "managed_roots": managed_roots,
        "cold_start": "adopt",
        "max_snapshot_age_seconds": 300,
        "plan_ttl_seconds": 300,
        "order_type": "market",
        "max_order_lots": 100,
        "max_price_deviation": 0.05,
        "limit_prices": {},
        "channel": "paper",
    }


def _weights_artifact(tmp_path):
    return write_artifact(
        tmp_path / "weights",
        {"stage": "weights", "data_date": DATA_DATE, "mode": "simulation", "fixture": "workflow"},
        {"weights.csv": _weights()},
    )


def _snapshot(account_id: str, *, positions: list[dict] | None = None) -> dict:
    return {
        "trade_date": DATA_DATE,
        "as_of": NOW.isoformat(),
        "identity": account_id,
        "positions": [] if positions is None else positions,
        "open_orders": [],
        "equity": 1_000_000.0,
        "available": 1_000_000.0,
        "margin_used": 0.0,
    }


def _position(*, contract: str, root: str, side: str, volume: int, exchange: str = "SHFE") -> dict:
    return {
        "root": root,
        "contract": contract,
        "exchange": exchange,
        "side": side,
        "volume": volume,
        "today": volume,
        "yesterday": 0,
        "available_today": volume,
        "available_yesterday": 0,
    }


def _sized(tmp_path, *, verified: bool = True):
    settings = _settings(tmp_path / "output")
    weights = _weights_artifact(tmp_path)
    sized_path = size_artifact(
        weights,
        settings,
        [_spec("RB2610", verified=verified), _spec("CU2610", verified=verified)],
    )
    return settings, sized_path, artifact_json(sized_path, "sizing.json")


def test_size_artifact_manifest_float_csv_and_capital_requirements_round_trip(tmp_path):
    settings, sized_path, sizing = _sized(tmp_path)
    manifest = read_artifact(sized_path)

    assert manifest["schema_version"] == 1
    assert manifest["identity"]["stage"] == "size"
    assert {"targets.csv", "attribution.csv", "sizing.json", "capital_requirements.csv"} <= set(
        manifest["files"]
    )
    targets = read_csv(sized_path, "targets.csv")
    assert {row["status"] for row in targets} == {"READY"}
    assert {row["account_id"] for row in targets} == {"A1", "A2"}
    assert all(isinstance(float(row["target_lots"]), float) for row in targets)
    assert all(isinstance(float(row["target_notional"]), float) for row in targets)
    assert {(row["account_id"], int(float(row["target_lots"]))) for row in targets} == {
        ("A1", 3),
        ("A2", -3),
    }
    assert all(sizing["accounts"][account]["tradable"] for account in ("A1", "A2"))
    assert sizing["capital_requirements"]["S1"]["rows"]
    assert sizing["capital_requirements"]["S2"]["rows"]
    assert sizing["capital_requirements"]["S1"]["proportional_one_lot_equity"] == 400
    assert sizing["capital_requirements"]["S2"]["proportional_one_lot_equity"] == 400
    assert settings["output_root"] in str(sized_path)
    # The immutable handoff can be read again after parsing all exported files.
    assert read_artifact(sized_path) == manifest


def test_existing_position_one_lot_delta_paper_fill_then_no_difference(tmp_path):
    settings, _, sizing = _sized(tmp_path)
    sized = sizing["accounts"]["A1"]
    target_lots = next(row["target_lots"] for row in sized["targets"] if row["contract"] == "RB2610")
    assert target_lots == 3
    snapshot = _snapshot(
        "A1",
        positions=[_position(contract="RB2610", root="RB", side="long", volume=target_lots - 1)],
    )
    plan = build_plan(
        sized,
        snapshot,
        account_id="A1",
        policy=settings["execution"]["A1"],
        signal_id="weights-v1",
        signal_date=DATA_DATE,
        now=NOW,
    )
    assert plan["ready"]
    assert [(row["offset"], row["volume"]) for row in plan["orders"]] == [("open", 1)]

    filled = paper_execute(plan, snapshot, now=NOW)
    regenerated = build_plan(
        sized,
        filled["snapshot"],
        account_id="A1",
        policy=settings["execution"]["A1"],
        signal_id="weights-v1",
        signal_date=DATA_DATE,
        now=NOW,
        previous_plan=plan,
    )
    assert regenerated["ready"]
    assert regenerated["orders"] == []


def test_oversized_existing_position_is_closed_then_regenerates_without_difference(tmp_path):
    settings, _, sizing = _sized(tmp_path)
    sized = sizing["accounts"]["A1"]
    snapshot = _snapshot(
        "A1",
        positions=[_position(contract="RB2610", root="RB", side="long", volume=4)],
    )
    plan = build_plan(
        sized,
        snapshot,
        account_id="A1",
        policy=settings["execution"]["A1"],
        signal_id="weights-v1",
        signal_date=DATA_DATE,
        now=NOW,
    )
    assert plan["ready"]
    assert [(row["offset"], row["volume"]) for row in plan["orders"]] == [("close_today", 1)]
    filled = paper_execute(plan, snapshot, now=NOW)
    regenerated = build_plan(
        sized,
        filled["snapshot"],
        account_id="A1",
        policy=settings["execution"]["A1"],
        signal_id="weights-v1",
        signal_date=DATA_DATE,
        now=NOW,
        previous_plan=plan,
    )
    assert regenerated["orders"] == []


def test_strategy_version_change_with_same_target_does_not_reopen_or_close(tmp_path):
    settings, _, sizing = _sized(tmp_path)
    sized = sizing["accounts"]["A1"]
    target_lots = next(row["target_lots"] for row in sized["targets"] if row["contract"] == "RB2610")
    snapshot = _snapshot(
        "A1",
        positions=[_position(contract="RB2610", root="RB", side="long", volume=target_lots)],
    )
    previous = build_plan(
        sized,
        snapshot,
        account_id="A1",
        policy=settings["execution"]["A1"],
        signal_id="weights-v1",
        signal_date=DATA_DATE,
        now=NOW,
    )
    switched = build_plan(
        sized,
        snapshot,
        account_id="A1",
        policy=settings["execution"]["A1"],
        signal_id="weights-v2",
        signal_date=DATA_DATE,
        now=NOW,
        previous_plan=previous,
    )
    assert previous["orders"] == []
    assert switched["ready"]
    assert switched["previous_signal_id"] == "weights-v1"
    assert switched["orders"] == []


def test_unverified_spec_keeps_diagnostics_but_blocks_plan_without_provider_call(tmp_path, monkeypatch):
    settings, sized_path, sizing = _sized(tmp_path, verified=False)
    assert {row["status"] for row in read_csv(sized_path, "targets.csv")} == {"BLOCKED"}
    sized = sizing["accounts"]["A1"]
    assert sized["tradable"] is False
    assert any("METADATA_UNVERIFIED" in blocker for blocker in sized["blockers"])

    called = []
    monkeypatch.setattr("run_trading_workflow.panda_call", lambda request: called.append(request))
    plan = build_plan(
        sized,
        _snapshot("A1"),
        account_id="A1",
        policy=settings["execution"]["A1"],
        signal_id="weights-v1",
        signal_date=DATA_DATE,
        now=NOW,
    )
    assert plan["ready"] is False
    assert any("METADATA_UNVERIFIED" in blocker for blocker in plan["blockers"])
    assert called == []
    with pytest.raises(ValueError, match="plan is blocked"):
        paper_execute(plan, _snapshot("A1"), now=NOW)
