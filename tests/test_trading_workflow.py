from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import pytest

from run_trading_workflow import artifact_json, size_artifact
from trading.artifacts import digest, read_artifact, read_csv, write_artifact
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


def test_size_plan_and_paper_cli_do_not_import_research_or_weights(tmp_path):
    import yaml
    settings = _settings(tmp_path / "output")
    config = tmp_path / "trading.yaml"
    config.write_text(yaml.safe_dump(settings), encoding="utf-8")
    specs = tmp_path / "specs.json"
    specs.write_text(json.dumps([_spec("RB2610"), _spec("CU2610")]), encoding="utf-8")
    source = _weights_artifact(tmp_path)
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps({**_snapshot("A1"), "as_of": datetime.now(timezone.utc).isoformat()}), encoding="utf-8")
    code = f'''
import builtins
original_import = builtins.__import__
def independent_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and (name.split('.')[0] in {{'research', 'factors', 'data', 'core'}} or name == 'trading.weights'):
        raise AssertionError('downstream imported upstream: ' + name)
    return original_import(name, globals, locals, fromlist, level)
builtins.__import__ = independent_import
from run_trading_workflow import main, artifact_json
prefix = ['--config', {str(config)!r}]
sizing = main(prefix + ['size', '--weights', {str(source)!r}, '--specs', {str(specs)!r}])
plan = main(prefix + ['plan', '--sizing', str(sizing), '--account', 'A1', '--snapshot', {str(snapshot)!r}])
assert artifact_json(plan, 'plan.json')['ready']
result = main(prefix + ['paper', '--plan', str(plan), '--snapshot', {str(snapshot)!r}])
assert artifact_json(result, 'snapshot.json')['positions'][0]['volume'] == 3
'''
    completed = subprocess.run([sys.executable, "-X", "utf8", "-c", code],
                               cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, encoding="utf-8")
    assert completed.returncode == 0, completed.stderr


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


def test_daily_report_preserves_source_and_money_units(tmp_path):
    settings = _settings(tmp_path / "output")
    source = _weights_artifact(tmp_path)
    short_spec = {**_spec("CU2610"), "margin_short": 0.2}
    path = size_artifact(source, settings, [_spec("RB2610"), short_spec])
    rows = read_csv(path, "daily_positions.csv")
    assert path.parent.name == DATA_DATE
    assert (path / "weights.csv").read_bytes() == (source / "weights.csv").read_bytes()
    assert len(rows) == 2
    for row, sign in zip(rows, (1, -1)):
        assert row["数据日期"] == DATA_DATE
        assert row["权重快照"] == source.name
        assert row["合约类型"] == "具体合约"
        assert row["主力滞后交易日"] == ""
        assert float(row["基准权重"]) == sign * 0.5
        assert float(row["基准仓位_元"]) == sign * 500
        assert float(row["理论手数"]) == sign * 2.5
        assert int(row["理论手数取整"]) == sign * 3
        assert int(row["账户合并目标手数"]) == sign * 3
        assert float(row["参考价格"]) == 10
        assert float(row["单手价值_元"]) == 200
        assert float(row["单手保证金_元"]) == (20 if sign == 1 else 40)
        assert float(row["预计占用保证金_元"]) == (60 if sign == 1 else 120)
        assert row["账户状态"] == "READY"
    assert rows[0]["理论合约"] == "RB2610"
    summary = read_csv(path, "account_summary.csv")
    assert [float(r["预计占用保证金_元"]) for r in summary] == [60, 120]
    assert [float(r["预计保证金占用率_%"]) for r in summary] == [0.006, 0.012]


@pytest.mark.parametrize("basis,amount,notional", [("equity", 1000, 500),
    ("notional", 1000, 1000), ("margin", 100, 1000)])
def test_daily_report_uses_sizing_budget_semantics(tmp_path, basis, amount, notional):
    settings = _settings(tmp_path / "output")
    settings["routes"][0].update(capital_basis=basis, amount=amount)
    path = size_artifact(_weights_artifact(tmp_path), settings, [_spec("RB2610"), _spec("CU2610")])
    row = read_csv(path, "daily_positions.csv")[0]
    assert row["资金口径"] == basis
    assert float(row["基准仓位_元"]) == notional


def test_daily_report_routes_do_not_replace_account_net_rounding(tmp_path):
    settings = _settings(tmp_path / "output")
    settings["accounts"] = {"A1": _account()}
    settings["routes"] = [{"strategy_id": sid, "account_id": "A1", "capital_basis": "equity",
                           "amount": 160} for sid in ("S1", "S2")]
    weights = [{**_weights()[0], "strategy_id": sid} for sid in ("S1", "S2")]
    source = write_artifact(tmp_path / "weights", {"stage": "weights", "data_date": DATA_DATE,
        "mode": "simulation"}, {"weights.csv": weights})
    path = size_artifact(source, settings, [_spec("RB2610")])
    rows = read_csv(path, "daily_positions.csv")
    assert [int(r["理论手数取整"]) for r in rows] == [0, 0]
    assert [int(r["账户合并目标手数"]) for r in rows] == [1, 1]
    assert read_csv(path, "targets.csv")[0]["target_lots"] == "1"
    assert [float(r["预计占用保证金_元"]) for r in rows] == [0, 0]
    assert float(read_csv(path, "account_summary.csv")[0]["预计占用保证金_元"]) == 20


def test_daily_margin_summary_nets_opposing_routes(tmp_path):
    settings = _settings(tmp_path / "output")
    settings["accounts"] = {"A1": _account()}
    settings["routes"] = [{"strategy_id": sid, "account_id": "A1", "capital_basis": "equity",
                           "amount": 1000} for sid in ("S1", "S2")]
    weights = [{**_weights()[0], "strategy_id": sid, "weight": weight}
               for sid, weight in (("S1", 0.5), ("S2", -0.5))]
    source = write_artifact(tmp_path / "weights", {"stage": "weights", "data_date": DATA_DATE,
        "mode": "simulation"}, {"weights.csv": weights})
    path = size_artifact(source, settings, [{**_spec("RB2610"), "margin_short": 0.2}])
    rows = read_csv(path, "daily_positions.csv")
    assert [float(r["预计占用保证金_元"]) for r in rows] == [60, 120]
    summary = read_csv(path, "account_summary.csv")
    assert len(summary) == 1
    assert float(summary[0]["预计占用保证金_元"]) == 0
    assert float(summary[0]["预计保证金占用率_%"]) == 0


def test_daily_sort_signed_weights_then_zeros_with_stable_ties(tmp_path):
    settings = _settings(tmp_path / "output")
    settings["accounts"] = {"A1": _account()}
    settings["routes"] = settings["routes"][:1]
    weights = [{**_weights()[0], "root": root, "contract": root + "2610", "weight": weight}
               for root, weight in (("ZN", 0), ("CU", -0.4), ("AL", 0.2),
                                    ("RB", 0.7), ("AG", 0.2), ("AU", -0.1), ("NI", 0))]
    for revision, source_rows in enumerate((weights, list(reversed(weights)))):
        source = write_artifact(tmp_path / "weights", {"stage": "weights", "data_date": DATA_DATE,
            "mode": "simulation", "revision": revision}, {"weights.csv": source_rows})
        path = size_artifact(source, settings, [_spec(r["contract"]) for r in weights])
        rows = read_csv(path, "daily_positions.csv")
        assert [r["品种"] for r in rows] == ["RB", "AG", "AL", "AU", "CU", "NI", "ZN"]
        assert [float(r["预计占用保证金_元"]) for r in rows[-2:]] == [0, 0]
        assert (path / "weights.csv").read_bytes() == (source / "weights.csv").read_bytes()


def test_complete_unit_report_separates_reference_capital_and_account_equity(tmp_path):
    settings = _settings(tmp_path / "output")
    settings["routes"] = [{"strategy_id": sid, "account_id": aid, "capital_basis": "complete_unit", "units": 1}
                          for sid, aid in (("S1", "A1"), ("S2", "A2"))]
    for account in settings["accounts"].values():
        account.update(equity=100, available=100, require_one_lot=True, max_gross_exposure=1)
    source = _weights_artifact(tmp_path)
    path = size_artifact(source, settings, [_spec("RB2610"), _spec("CU2610")])
    rows = read_csv(path, "daily_positions.csv")
    assert [float(r["折算基准金额_元"]) for r in rows] == [400, 400]
    assert [int(r["组合份数"]) for r in rows] == [1, 1]
    assert [r["组合完整"] for r in rows] == ["True", "True"]
    assert [abs(float(r["理论手数"])) for r in rows] == [1, 1]
    assert [float(r["取整权重偏差"]) for r in rows] == [0, 0]
    summary = read_csv(path, "account_summary.csv")
    assert all(float(r["基准权益_元"]) == 100 for r in summary)
    assert all(float(r["总名义杠杆"]) == 2 for r in summary)
    assert all(r["账户状态"] == "BLOCKED" and "GROSS_EXPOSURE" in r["拦截原因"] for r in summary)
    routes = read_csv(path, "route_sizing.csv")
    assert all(float(r["reference_capital"]) == 400 and r["amount"] == "" for r in routes)
    assert (path / "weights.csv").read_bytes() == (source / "weights.csv").read_bytes()


def test_default_trading_config_uses_complete_units_without_raising_risk_limits():
    from run_trading_workflow import load_settings
    settings = load_settings("config/trading.yaml")
    assert settings["routes"] == [{"strategy_id": "formal_main", "account_id": "paper_500w",
                                   "capital_basis": "complete_unit", "units": 1}]
    assert settings["accounts"]["paper_500w"]["equity"] == 5000000
    assert settings["accounts"]["paper_500w"]["max_gross_exposure"] == 2.1
    assert settings["execution"]["paper_500w"]["execute_enabled"] is False


def test_daily_report_keeps_revisions_dates_and_causal_main_contract(tmp_path):
    settings = _settings(tmp_path / "output")
    paths = []
    for revision, day in enumerate((DATA_DATE, DATA_DATE, "2026-09-16")):
        contract = {"strategy_id": "S1", "catalog_strategy_id": "formal", "revision": revision}
        identity = {"stage": "weights", "data_date": day, "mode": "simulation",
                    "strategies": [contract], "config": {"data": {"source": "duckdb_futures",
                    "parquet": {"dominant_lag_days": 1}}}}
        source = write_artifact(tmp_path / "weights", identity,
            {"weights.csv": [{**r, "data_date": day} for r in _weights()]})
        path = size_artifact(source, settings, [_spec("RB2610"), _spec("CU2610")])
        contents = (path / "daily_positions.csv").read_bytes()
        assert size_artifact(source, settings, [_spec("RB2610"), _spec("CU2610")]) == path
        assert (path / "daily_positions.csv").read_bytes() == contents
        assert path.parent.name == day
        row = read_csv(path, "daily_positions.csv")[0]
        assert row["合约类型"] == "滞后主力合约"
        assert row["主力滞后交易日"] == "1"
        assert row["策略版本"] == digest(contract)
        if day != DATA_DATE:
            assert row["参数当日核实"] == "False"
            assert row["账户状态"] == "BLOCKED"
        paths.append(path)
    assert len(set(paths)) == 3
    assert all(read_artifact(path) for path in paths)


def test_daily_report_rejects_date_manifest_mismatch_and_tampering(tmp_path):
    settings = _settings(tmp_path / "output")
    source = write_artifact(tmp_path / "weights", {"stage": "weights", "data_date": "2026-09-16",
        "mode": "simulation"}, {"weights.csv": _weights()})
    with pytest.raises(ValueError, match="data_date"):
        size_artifact(source, settings, [_spec("RB2610"), _spec("CU2610")])
    _, path, _ = _sized(tmp_path)
    (path / "daily_positions.csv").write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="file hash mismatch"):
        read_artifact(path)


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
