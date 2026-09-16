from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from trading.artifacts import digest, file_hash
from trading.automation import previous_account_plan
from trading.execution import build_plan
from trading.panda import normalize_snapshot, session


TRADE_DATE = "2026-09-15"


def _raw_snapshot(identity: str = "test") -> dict:
    return {
        "account": {
            "accountId": identity,
            "ready": True,
            "tradeDate": "20260915",
            "totalProfit": 5_000_000,
            "availableFunds": 5_000_000,
            "margin": 0,
            "frozenCapital": 0,
        },
        "positions": [],
        "openOrders": [],
    }


def _policy(**updates) -> dict:
    policy = {
        "expected_identity": "test",
        "managed_roots": ["RB"],
        "channel": "panda",
        "allow_simulation_strategy": True,
        "execute_enabled": True,
        "order_type": "market",
        "max_order_lots": 100,
        "max_price_deviation": 0.05,
        "max_snapshot_age_seconds": 300,
        "plan_ttl_seconds": 3600,
        "cold_start": "adopt_managed",
    }
    policy.update(updates)
    return policy


def _sized() -> dict:
    return {
        "tradable": True,
        "blockers": [],
        "account": {"equity": 5_000_000, "reserve": 10_000, "max_margin_ratio": 0.7},
        "summary": {"estimated_margin": 3_600},
        "targets": [{
            "root": "RB",
            "contract": "RB2701",
            "exchange": "SHFE",
            "target_lots": 1,
            "tick": 1,
            "close": 3_600,
        }],
    }


class FakePanda:
    def __init__(self, *, orders=None, operation=None):
        self.remote = {"planId": "remote-one", "status": "prepared"}
        self.orders_rows = [] if orders is None else orders
        self.operation_row = operation or {"status": "completed"}
        self.prepare_calls = []
        self.plan_calls = []
        self.execute_calls = []
        self.order_queries = []

    def snapshot(self):
        return _raw_snapshot()

    def prepare_order(self, *args, **kwargs):
        self.prepare_calls.append((args, kwargs))
        parameters = dict(zip(("contractCode", "side", "offset", "volume"), args))
        parameters.update(clientOrderId=kwargs["client_order_id"])
        if kwargs.get("price") is not None:
            parameters["price"] = kwargs["price"]
        self.remote.update(accountId=self.snapshot()["account"]["accountId"], operation="place_order",
                           parameters=parameters, legs=[{**parameters, "preview": {"wouldSucceed": True}}])
        return deepcopy(self.remote)

    def plan(self, plan_id):
        self.plan_calls.append(plan_id)
        return deepcopy(self.remote)

    def execute_plan(self, plan_id, **kwargs):
        self.execute_calls.append((plan_id, kwargs))
        return {"status": "queued", "operationId": "operation-one"}

    def orders(self, **kwargs):
        self.order_queries.append(kwargs)
        return deepcopy(self.orders_rows)

    def operation(self, operation_id):
        assert operation_id == "operation-one"
        return deepcopy(self.operation_row)


class PagedFakePanda(FakePanda):
    def __init__(self, pages):
        super().__init__()
        self.pages = list(pages)
        self.page_queries = []

    def orders_page(self, *, client_order_id, count, last_id=None):
        assert count == 200
        self.page_queries.append((client_order_id, last_id))
        index = len(self.page_queries) - 1
        return deepcopy(self.pages[index])


def _plan(client: FakePanda, policy=None):
    policy = policy or _policy()
    snapshot = normalize_snapshot(client.snapshot(), policy)
    return build_plan(
        _sized(), snapshot, account_id="test-local", policy=policy,
        signal_id="weights-v1", signal_date=TRADE_DATE,
    )


def _request(tmp_path, client=None, **updates):
    client = client or FakePanda()
    config = tmp_path / "settings.toml"
    config.write_text("stable-config\n", encoding="utf-8")
    policy = _policy()
    policy.update(updates.pop("policy", {}))
    request = {
        "sized": _sized(),
        "account_id": "test-local",
        "policy": policy,
        "signal_id": "weights-v1",
        "signal_date": TRADE_DATE,
        "trade_date": TRADE_DATE,
        "journal_root": str(tmp_path / "journal"),
        "mode": "simulation",
        "config_path": str(config),
        "config_sha256": file_hash(config),
    }
    request.update(updates)
    return request, client, config


def _capture_session(monkeypatch, request, client):
    captured = {}

    def fake_run(sized, **callbacks):
        captured["sized"] = sized
        captured.update(callbacks)
        return {"status": "captured"}

    monkeypatch.setattr("trading.automation.run_rebalance", fake_run)
    result = session(request, client)
    return captured, result


def _submit_under_controller_lock(request, plan, captured):
    journal = Path(request["journal_root"]) / digest({"identity": plan["identity"]})
    journal.mkdir(parents=True, exist_ok=True)
    lock = journal / "execute.lock"
    lock.write_text("controller-owned", encoding="utf-8")
    try:
        return captured["submit_fn"](plan)
    finally:
        lock.unlink(missing_ok=True)


def test_session_prepares_and_executes_one_exact_frozen_order_with_shared_lock(tmp_path, monkeypatch):
    request, client, _ = _request(tmp_path)
    captured, result = _capture_session(monkeypatch, request, client)
    plan = _plan(client, request["policy"])

    receipt = _submit_under_controller_lock(request, plan, captured)

    client_order_id = plan["orders"][0]["client_order_id"]
    assert result == {"status": "captured"}
    assert len(client.prepare_calls) == 1
    assert client.prepare_calls[0][1]["client_order_id"] == client_order_id
    assert client.plan_calls == ["remote-one"]
    assert len(client.execute_calls) == 1
    assert client.execute_calls[0] == (
        "remote-one", {"confirmed": True, "client_request_id": client_order_id}
    )
    assert receipt["client_order_id"] == client_order_id


def _prepared_receipt(tmp_path, monkeypatch, *, client=None, orders=None, operation=None):
    client = client or FakePanda(orders=orders, operation=operation)
    request, client, _ = _request(tmp_path, client=client)
    captured, _ = _capture_session(monkeypatch, request, client)
    plan = _plan(client, request["policy"])
    receipt = _submit_under_controller_lock(request, plan, captured)
    return request, client, captured, receipt


def test_session_poll_accepts_only_terminal_unique_integer_fill_evidence(tmp_path, monkeypatch):
    rows = [{
        "status": "filled", "orderId": "broker-1", "contractCode": "rb2701",
        "tradeDirection": "open_long", "filledQuantity": 1, "quantity": 1,
    }]
    _, client, captured, receipt = _prepared_receipt(tmp_path, monkeypatch, orders=rows)

    evidence = captured["poll_fn"](receipt)

    assert evidence["status"] == "reconciled"
    assert evidence["filled_quantity"] == 1
    assert client.order_queries == [{"client_order_id": receipt["client_order_id"], "count": 200}]


@pytest.mark.parametrize(
    "changes",
    [
        {"contractCode": "CU2701"},
        {"tradeDirection": "open_short"},
        {"quantity": 2, "filledQuantity": 2},
        {"filledQuantity": 0},
        {"clientOrderId": "different-client-order"},
    ],
)
def test_session_poll_rejects_order_evidence_that_does_not_match_frozen_intent(
    tmp_path, monkeypatch, changes
):
    row = {
        "status": "filled", "orderId": "broker-1", "contractCode": "rb2701",
        "tradeDirection": "open_long", "filledQuantity": 1, "quantity": 1,
    }
    row.update(changes)
    _, _, captured, receipt = _prepared_receipt(tmp_path, monkeypatch, orders=[row])

    evidence = captured["poll_fn"](receipt)

    assert evidence == {
        "status": "unknown", "reason": "ORDER_EVIDENCE_DOES_NOT_MATCH_INTENT"
    }


def test_session_poll_does_not_treat_completed_operation_without_orders_as_a_fill(tmp_path, monkeypatch):
    _, _, captured, receipt = _prepared_receipt(
        tmp_path, monkeypatch, orders=[], operation={"status": "completed"}
    )

    evidence = captured["poll_fn"](receipt)

    assert evidence["status"] == "queued"
    assert "filled_quantity" not in evidence


@pytest.mark.parametrize("venue", ["SHFE", "DCE"])
def test_session_order_contract_suffix_must_match_frozen_exchange(tmp_path, monkeypatch, venue):
    row = {"status": "filled", "orderId": "broker-1", "contractCode": f"RB2701.{venue}",
           "exchange": venue, "tradeDirection": "open_long", "filledQuantity": 1, "quantity": 1}
    _, _, captured, receipt = _prepared_receipt(tmp_path, monkeypatch, orders=[row])
    result = captured["poll_fn"](receipt)
    assert result["status"] == ("reconciled" if venue == "SHFE" else "unknown")


@pytest.mark.parametrize(
    ("rows", "reason"),
    [
        ([{"status": "filled", "orderId": "broker-1", "quantity": 1}],
         "FILL_QUANTITY_NOT_VERIFIED"),
        ([
            {"status": "filled", "orderId": "same", "filledQuantity": 1, "quantity": 1},
            {"status": "cancelled", "orderId": "same", "filledQuantity": 0, "quantity": 1},
        ], "ORDER_IDENTIFIERS_INCOMPLETE"),
        ([
            {"status": "filled", "orderId": f"broker-{index}", "filledQuantity": 0, "quantity": 1}
            for index in range(200)
        ], "ORDER_EVIDENCE_INCOMPLETE"),
    ],
)
def test_session_poll_rejects_incomplete_or_ambiguous_order_evidence(
    tmp_path, monkeypatch, rows, reason
):
    _, _, captured, receipt = _prepared_receipt(tmp_path, monkeypatch, orders=rows)

    evidence = captured["poll_fn"](receipt)

    assert evidence["status"] == "unknown"
    if reason == "ORDER_EVIDENCE_INCOMPLETE":
        assert evidence["orders_complete"] is False
    else:
        assert evidence["reason"] == reason


def test_config_change_blocks_new_send_but_allows_polling_the_original_order(
    tmp_path, monkeypatch
):
    request, client, config = _request(tmp_path)
    captured, _ = _capture_session(monkeypatch, request, client)
    plan = _plan(client, request["policy"])
    receipt = _submit_under_controller_lock(request, plan, captured)
    assert len(client.execute_calls) == 1

    config.write_text("changed-config\n", encoding="utf-8")
    with pytest.raises(ValueError, match="configuration changed"):
        captured["submit_fn"](plan)
    assert len(client.execute_calls) == 1

    client.orders_rows = [{
        "status": "filled", "orderId": "broker-1", "contractCode": "RB2701",
        "tradeDirection": "open_long", "filledQuantity": 1, "quantity": 1,
    }]
    evidence = captured["poll_fn"](receipt)

    assert evidence["status"] == "reconciled"
    assert client.order_queries[-1]["client_order_id"] == receipt["client_order_id"]


def test_reconcile_follows_pages_until_explicitly_complete(tmp_path, monkeypatch):
    client = PagedFakePanda([
        {
            "items": [{"status": "filled", "orderId": "broker-1", "filledQuantity": 1, "quantity": 1}],
            "meta": {"hasMore": True, "nextLastId": "cursor-1"},
        },
        {
            "items": [{"status": "cancelled", "orderId": "broker-2", "filledQuantity": 0, "quantity": 1}],
            "meta": {"hasMore": False},
        },
    ])
    request, client, _, receipt = _prepared_receipt(tmp_path, monkeypatch, client=client)

    from trading.panda import reconcile
    record = reconcile(client, request["journal_root"], "test", receipt["client_order_id"])

    assert record["status"] == "reconciled"
    assert record["reconciliation"]["orders_complete"] is True
    assert [row["orderId"] for row in record["reconciliation"]["orders"]] == [
        "broker-1", "broker-2"
    ]
    assert client.page_queries == [
        (receipt["client_order_id"], None),
        (receipt["client_order_id"], "cursor-1"),
    ]


def test_reconcile_rejects_a_repeated_page_cursor_as_incomplete(tmp_path, monkeypatch):
    client = PagedFakePanda([
        {
            "items": [{"status": "filled", "orderId": "broker-1", "filledQuantity": 1, "quantity": 1}],
            "meta": {"hasMore": True, "nextLastId": "same-cursor"},
        },
        {
            "items": [{"status": "filled", "orderId": "broker-2", "filledQuantity": 1, "quantity": 1}],
            "meta": {"hasMore": True, "nextLastId": "same-cursor"},
        },
    ])
    request, client, _, receipt = _prepared_receipt(tmp_path, monkeypatch, client=client)

    from trading.panda import reconcile
    record = reconcile(client, request["journal_root"], "test", receipt["client_order_id"])

    assert record["status"] == "unknown"
    assert record["reconciliation"]["orders_complete"] is False
    assert len(record["reconciliation"]["orders"]) == 2
    assert len(client.page_queries) == 2


@pytest.mark.parametrize(
    ("request_updates", "message"),
    [
        ({"mode": "production"}, "automatic execution"),
        ({"policy": {"channel": "paper"}}, "automatic execution"),
        ({"policy": {"allow_simulation_strategy": False}}, "automatic execution"),
        ({"policy": {"execute_enabled": False}}, "automatic execution"),
    ],
)
def test_session_rejects_non_simulation_or_disabled_automatic_execution(
    tmp_path, monkeypatch, request_updates, message
):
    request, client, _ = _request(tmp_path, **request_updates)
    monkeypatch.setattr("trading.automation.run_rebalance", pytest.fail)

    with pytest.raises(ValueError, match=message):
        session(request, client)


def _write_previous(root, plan, *, status="completed", filename="session_2026-09-14.json"):
    directory = Path(root) / digest({"identity": "test"})
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_text(
        json.dumps({"status": status, "last_plan": plan}, ensure_ascii=False),
        encoding="utf-8",
    )


def test_previous_account_plan_returns_intact_hashed_plan_for_same_account(tmp_path):
    client = FakePanda()
    plan = _plan(client)
    _write_previous(tmp_path / "journal", plan)

    inherited = previous_account_plan(
        tmp_path / "journal", "test", "test-local", "2026-09-15"
    )

    assert inherited == plan
    assert inherited["plan_id"] == digest({k: v for k, v in inherited.items() if k != "plan_id"})
    assert inherited["managed_roots"] == ["RB"]


def test_previous_account_plan_rejects_hash_tampering(tmp_path):
    client = FakePanda()
    plan = _plan(client)
    tampered = deepcopy(plan)
    tampered["managed_roots"] = ["RB", "CU"]
    _write_previous(tmp_path / "journal", tampered)

    with pytest.raises(ValueError, match="identity/hash mismatch"):
        previous_account_plan(tmp_path / "journal", "test", "test-local", "2026-09-15")


def test_previous_account_plan_rejects_unresolved_unknown_session(tmp_path):
    client = FakePanda()
    _write_previous(tmp_path / "journal", _plan(client), status="unknown")

    with pytest.raises(ValueError, match="UNRESOLVED_PREVIOUS_SESSION"):
        previous_account_plan(tmp_path / "journal", "test", "test-local", "2026-09-15")


def test_previous_account_plan_rejects_a_rehashed_plan_from_another_identity(tmp_path):
    client = FakePanda()
    plan = _plan(client)
    other = deepcopy(plan)
    other["identity"] = "other-account"
    other["plan_id"] = digest({k: v for k, v in other.items() if k != "plan_id"})
    _write_previous(tmp_path / "journal", other)

    with pytest.raises(ValueError, match="identity/hash mismatch"):
        previous_account_plan(tmp_path / "journal", "test", "test-local", "2026-09-15")
