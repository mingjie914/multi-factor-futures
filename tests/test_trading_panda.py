from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from trading.panda import normalize_snapshot, prepare, execute
from trading.execution import build_plan


class FakePanda:
    calls = 0
    fail = False

    def snapshot(self):
        return {"account": {"accountId": "test", "ready": True, "tradeDate": "20260915",
                "totalProfit": 5000000, "availableFunds": 5000000, "margin": 0, "frozenCapital": 0},
                "positions": [], "openOrders": []}

    def prepare_order(self, *args, **kwargs):
        return {"planId": "remote-one", "status": "prepared", "parameters": [list(args), kwargs]}

    def plan(self, plan_id):
        return self.prepared

    def execute_plan(self, plan_id, **kwargs):
        self.calls += 1
        if self.fail:
            raise TimeoutError("uncertain")
        return {"status": "queued", "operationId": "op-one"}


def make_plan(client):
    snap = normalize_snapshot(client.snapshot(), {})
    policy = {"expected_identity": "test", "managed_roots": ["RB"], "channel": "panda",
              "order_type": "market", "max_order_lots": 100, "max_price_deviation": .05,
              "max_snapshot_age_seconds": 60, "plan_ttl_seconds": 60, "execute_enabled": True}
    sized = {"tradable": True, "blockers": [], "account": {"equity": 5e6, "reserve": 1e4,
             "max_margin_ratio": .7}, "summary": {"estimated_margin": 3600},
             "targets": [{"root": "RB", "contract": "RB2701", "exchange": "SHFE",
                          "target_lots": 1, "tick": 1, "close": 3600}]}
    return build_plan(sized, snap, account_id="test-local", policy=policy, signal_id="w",
                       signal_date="2026-09-15")


def test_readonly_snapshot_maps_equity_from_documented_total_profit():
    result = normalize_snapshot(FakePanda().snapshot(), {})
    assert result["equity"] == 5e6 and result["identity"] == "test"
    assert result["trade_date"] == "2026-09-15"
    raw = FakePanda().snapshot()
    raw["positions"] = [{"unknownField": 1}]
    with pytest.raises(ValueError, match="mapping"):
        normalize_snapshot(raw, {})


def test_canonical_position_keeps_exact_broker_code_for_closing():
    client = FakePanda()
    raw = client.snapshot()
    row = {"root": "SR", "contract": "SR701", "exchange": "CZCE", "side": "long",
           "volume": 1, "today": 0, "yesterday": 1, "available_today": 0, "available_yesterday": 1}
    raw["positions"] = [row]
    client.snapshot = lambda: raw
    policy = make_plan(FakePanda())["policy"]
    policy.update(managed_roots=["SR"], cold_start="adopt_managed",
                  position_fields={key: key for key in row}, contract_aliases={"SR701": "SR2701"})
    snap = normalize_snapshot(raw, policy)
    sized = {"tradable": True, "blockers": [], "account": {"equity": 5e6, "reserve": 1e4,
             "max_margin_ratio": .7}, "summary": {"estimated_margin": 0}, "targets": []}
    plan = build_plan(sized, snap, account_id="test-local", policy=policy, signal_id="w", signal_date="2026-09-15")
    assert plan["orders"][0]["contract"] == "SR2701"
    prepared = prepare(plan, 0, client)
    assert prepared["remote"]["parameters"][0][0] == "SR701"


def test_queued_or_timeout_is_persisted_and_never_resubmitted(tmp_path):
    for failure in (False, True):
        client = FakePanda()
        client.fail = failure
        plan = make_plan(client)
        remote = prepare(plan, 0, client)
        client.prepared = remote["remote"]
        folder = tmp_path / str(failure)
        assert execute(plan, remote, client, folder, confirmation=remote["confirmation"])["status"] in {"queued", "unknown"}
        assert client.calls == 1
        with pytest.raises(ValueError, match="already attempted"):
            execute(plan, remote, client, folder, confirmation=remote["confirmation"])
        assert client.calls == 1


def test_no_confirm_no_execution(tmp_path):
    client = FakePanda()
    plan = make_plan(client)
    remote = prepare(plan, 0, client)
    with pytest.raises(ValueError, match="confirmation"):
        execute(plan, remote, client, tmp_path, confirmation="yes")
    assert client.calls == 0


@pytest.mark.parametrize("prior_status", ["queued", "failed"])
def test_unresolved_attempt_blocks_new_plan_and_account_lock_blocks_parallel_send(tmp_path, prior_status):
    from trading.artifacts import digest
    client = FakePanda()
    first = make_plan(client)
    remote = prepare(first, 0, client)
    client.prepared = remote["remote"]
    execute(first, remote, client, tmp_path, confirmation=remote["confirmation"])
    import json
    journal = tmp_path / digest({"identity": "test"}) / (remote["client_order_id"] + ".json")
    record = json.loads(journal.read_text())
    record["status"] = prior_status
    journal.write_text(json.dumps(record))
    second = make_plan(client)
    second["policy"]["intent_revision"] = "new"
    second["orders"][0]["client_order_id"] = "mf_" + "b" * 28
    second["plan_id"] = digest({k: v for k, v in second.items() if k != "plan_id"})
    new = prepare(second, 0, client)
    client.prepared = new["remote"]
    with pytest.raises(ValueError, match="unresolved"):
        execute(second, new, client, tmp_path, confirmation=new["confirmation"])
    lock = tmp_path / digest({"identity": "test"}) / "execute.lock"
    lock.write_text("9999")
    with pytest.raises(ValueError, match="locked"):
        execute(second, new, client, tmp_path, confirmation=new["confirmation"])
