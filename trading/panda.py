"""Optional Panda SDK worker. Importable without installing SDK in the research venv."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys

from trading.artifacts import digest, encoded
from trading.execution import validate_plan, validate_snapshot


def normalize_snapshot(raw, policy):
    account = raw["account"]
    if account.get("ready") is not True:
        raise ValueError("Panda account snapshot is not ready")
    positions = []
    mapping = policy.get("position_fields", {})
    required = ("root", "contract", "exchange", "side", "volume", "today", "yesterday",
                "available_today", "available_yesterday")
    if raw["positions"] and not all(name in mapping for name in required):
        raise ValueError("Panda nonempty position schema requires verified field mapping")
    for row in raw["positions"]:
        position = {field: row[mapping[field]] for field in required}
        position["broker_contract"] = str(position["contract"])
        position["side"] = policy.get("position_side_values", {}).get(position["side"], position["side"])
        position["contract"] = policy.get("contract_aliases", {}).get(position["contract"], position["contract"])
        position["contract"] = str(position["contract"]).upper()
        position["root"] = str(position["root"]).upper()
        position["exchange"] = policy.get("exchange_aliases", {}).get(position["exchange"], position["exchange"])
        positions.append(position)
    result = {"identity": str(account["accountId"]),
              "trade_date": datetime.strptime(account["tradeDate"], "%Y%m%d").date().isoformat(),
              "as_of": datetime.now(timezone.utc).isoformat(),
              # SDK 0.1.19 doctor explicitly labels totalProfit as equity.
              "equity": account["totalProfit"], "available": account["availableFunds"],
              "margin_used": account["margin"], "frozen_margin": account["frozenCapital"],
              "positions": positions, "open_orders": raw["openOrders"]}
    validate_snapshot(result)
    return result


def prepare(plan, order_index, client):
    current = normalize_snapshot(client.snapshot(), plan["policy"])
    validate_plan(plan, current)
    if plan["channel"] != "panda":
        raise ValueError("Panda preparation requires channel=panda")
    order = plan["orders"][order_index]
    if order["depends_on"]:
        raise ValueError("closing legs must fill; reconcile a new plan before opening")
    if order["offset"] not in {"open", "close"}:
        raise ValueError("Panda close_today/close_yesterday mapping is not verified; use manual execution")
    remote = client.prepare_order(order.get("broker_contract", order["contract"]), order["side"], order["offset"], order["volume"],
                                  price=order["price"], client_order_id=order["client_order_id"])
    if not isinstance(remote, dict) or not isinstance(remote.get("planId"), str):
        raise ValueError("Panda frozen plan schema changed: planId missing")
    body = {"local_plan_id": plan["plan_id"], "order_index": order_index,
            "client_order_id": order["client_order_id"], "remote": remote}
    # Confirmation identifies the exact local order and entire remote frozen result.
    return {**body, "confirmation": digest(body)}


def execute(plan, prepared, client, journal_root, *, confirmation):
    journal = Path(journal_root) / digest({"identity": plan["identity"]})
    journal.mkdir(parents=True, exist_ok=True)
    lock = journal / "execute.lock"
    try:
        with lock.open("x", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
    except FileExistsError as exc:
        raise ValueError("account execution is locked; reconcile a crashed process before recovery") from exc
    try:
        return _execute_locked(plan, prepared, client, journal_root, confirmation=confirmation)
    finally:
        lock.unlink()


def _execute_locked(plan, prepared, client, journal_root, *, confirmation):
    expected = digest({k: v for k, v in prepared.items() if k != "confirmation"})
    if confirmation != expected or prepared["confirmation"] != expected:
        raise ValueError("exact frozen-plan confirmation is required")
    if (prepared["local_plan_id"] != plan["plan_id"] or plan["channel"] != "panda"
            or plan["policy"].get("execute_enabled") is not True):
        raise ValueError("Panda execution is disabled or plan binding changed")
    order = plan["orders"][prepared["order_index"]]
    if order["client_order_id"] != prepared["client_order_id"] or order["depends_on"]:
        raise ValueError("frozen order binding or dependencies changed")
    journal = Path(journal_root) / digest({"identity": plan["identity"]})
    journal.mkdir(parents=True, exist_ok=True)
    attempt = journal / (order["client_order_id"] + ".json")
    if attempt.exists():
        raise ValueError("order already attempted; query original receipt, never resend")
    for entry in journal.glob("mf_*.json"):
        state = json.loads(entry.read_text(encoding="utf-8"))
        if state["status"] != "reconciled" and not (
                state["status"] == "expired" and state.get("reconciliation")):
            raise ValueError("account has an unresolved attempt; reconcile before another order")
    validate_plan(plan, normalize_snapshot(client.snapshot(), plan["policy"]))
    remote = client.plan(prepared["remote"]["planId"])
    if digest(remote) != digest(prepared["remote"]):
        raise ValueError("Panda frozen plan changed; preview and confirm again")
    initial = {"status": "sending", "client_order_id": order["client_order_id"],
               "local_plan_id": plan["plan_id"], "remote_plan_id": remote["planId"]}
    try:
        with attempt.open("xb") as handle:
            handle.write(encoded(initial))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ValueError("order already attempted by another process") from exc
    try:
        result = client.execute_plan(remote["planId"], confirmed=True,
                                     client_request_id=order["client_order_id"])
    except Exception as exc:
        # Even transport failure remains a recorded attempt. There is no retry loop.
        result = {"status": "unknown", "error_type": type(exc).__name__,
                  "next_action": "query_orders_by_clientOrderId"}
    record = {**initial, "status": result.get("status", "unknown"), "response": result}
    temp = attempt.with_suffix(f".{os.getpid()}.tmp")
    with temp.open("wb") as handle:
        handle.write(encoded(record))
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(attempt)
    return record


def reconcile(client, journal_root, identity, client_order_id):
    if not re.fullmatch(r"mf_[a-f0-9]{28}", client_order_id):
        raise ValueError("invalid client order ID")
    current = client.snapshot()
    if str(current["account"]["accountId"]) != str(identity):
        raise ValueError("account identity mismatch")
    path = Path(journal_root) / digest({"identity": str(identity)}) / (client_order_id + ".json")
    record = json.loads(path.read_text(encoding="utf-8"))
    orders = client.orders(client_order_id=client_order_id, count=200)
    operation_id = record.get("response", {}).get("operationId")
    operation = client.operation(operation_id) if operation_id else None
    # Command completion is not a fill. Require terminal order evidence, or
    # the service's explicit expired operation after its own queue reconciliation.
    if orders and all(o.get("status") in {"filled", "cancelled", "rejected"} for o in orders):
        record["status"] = "reconciled"
    elif not orders and operation and operation.get("status") == "expired":
        record["status"] = "expired"
    record["reconciliation"] = {"orders": orders, "operation": operation}
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    temp.write_bytes(encoded(record))
    temp.replace(path)
    return record


def main():
    from panda_trade import AgentClient
    request = json.load(sys.stdin)
    client = AgentClient()
    action = request["action"]
    if action == "snapshot":
        result = normalize_snapshot(client.snapshot(), request.get("policy", {}))
    elif action == "prepare":
        result = prepare(request["plan"], request["order_index"], client)
    elif action == "execute":
        result = execute(request["plan"], request["prepared"], client,
                         request["journal_root"], confirmation=request["confirmation"])
    elif action == "operation":
        result = client.operation(request["operation_id"])
    elif action == "reconcile":
        result = reconcile(client, request["journal_root"], request["identity"], request["client_order_id"])
    else:
        raise ValueError("unsupported Panda worker action")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
