"""Optional Panda SDK worker. Importable without installing SDK in the research venv."""
from __future__ import annotations

from datetime import datetime, timezone
import math
import json
import os
from pathlib import Path
import re
import sys

from trading.artifacts import digest, encoded
from trading.execution import integer, validate_plan, validate_snapshot


def normalize_snapshot(raw, policy):
    account = raw["account"]
    if account.get("ready") is not True:
        raise ValueError("Panda account snapshot is not ready")
    positions = []
    position_schema = policy.get("position_schema")
    if position_schema == "panda_sdk_0_1_19":
        known_contracts = policy.get("known_contracts")
        if not isinstance(known_contracts, dict):
            if raw["positions"]:
                raise ValueError("Panda SDK position schema requires known_contracts")
            known_contracts = {}

        def sdk_integer(row, field):
            if field not in row or isinstance(row[field], bool):
                raise ValueError(f"Panda SDK position requires nonnegative integer {field}")
            try:
                value = float(row[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Panda SDK position requires nonnegative integer {field}"
                ) from exc
            if not math.isfinite(value) or value < 0 or int(value) != value:
                raise ValueError(f"Panda SDK position requires nonnegative integer {field}")
            return int(value)

        direction_values = {
            "多": "long", "多头": "long", "long": "long",
            "空": "short", "空头": "short", "short": "short",
        }
        exchange_aliases = policy.get("exchange_aliases", {})
        for row in raw["positions"]:
            if "contractCode" not in row:
                raise ValueError("Panda SDK position requires contractCode")
            broker_contract = str(row["contractCode"])
            bare_contract, _, suffix = broker_contract.upper().partition(".")
            known = known_contracts.get(broker_contract)
            if known is None:
                known = known_contracts.get(broker_contract.upper())
            if known is None:
                known = known_contracts.get(bare_contract)
            if not isinstance(known, dict) or not all(
                    field in known for field in ("contract", "root", "exchange")):
                raise ValueError(
                    f"Panda SDK position contract is not in known_contracts: {broker_contract}"
                )
            contract = str(known["contract"]).upper()
            root = str(known["root"]).upper()
            exchange = str(known["exchange"]).upper()
            if suffix and exchange_aliases.get(suffix, suffix) != exchange:
                raise ValueError("Panda position exchange suffix conflicts with known_contracts")
            if "exchange" in row:
                raw_exchange = row["exchange"]
                normalized_exchange = exchange_aliases.get(
                    raw_exchange,
                    exchange_aliases.get(str(raw_exchange).upper(), raw_exchange),
                )
                if str(normalized_exchange).upper() != exchange:
                    raise ValueError(
                        f"Panda SDK position exchange conflicts with known_contracts: {broker_contract}"
                    )
            direction_text = str(row.get("directionText", "")).strip().casefold()
            side = direction_values.get(direction_text)
            if side is None:
                raise ValueError(
                    "Panda SDK position directionText must be 多/多头/long or 空/空头/short"
                )
            volume = sdk_integer(row, "position")
            today = sdk_integer(row, "tdPosition")
            yesterday = sdk_integer(row, "ydPosition")
            closable = sdk_integer(row, "closable")
            if volume != today + yesterday:
                raise ValueError("Panda SDK position must equal tdPosition + ydPosition")
            if closable != volume:
                raise ValueError(
                    "Panda SDK closable must equal position; cannot infer today/yesterday closeability"
                )
            positions.append({
                "root": root,
                "contract": contract,
                "exchange": exchange,
                "side": side,
                "volume": volume,
                "today": today,
                "yesterday": yesterday,
                "available_today": today,
                "available_yesterday": yesterday,
                "broker_contract": broker_contract,
            })
    mapping = policy.get("position_fields", {})
    required = ("root", "contract", "exchange", "side", "volume", "today", "yesterday",
                "available_today", "available_yesterday")
    if position_schema != "panda_sdk_0_1_19":
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
    _validate_frozen_order(remote, order, plan["identity"])
    body = {"local_plan_id": plan["plan_id"], "order_index": order_index,
            "client_order_id": order["client_order_id"], "remote": remote}
    # Confirmation identifies the exact local order and entire remote frozen result.
    return {**body, "confirmation": digest(body)}


def _validate_frozen_order(remote, order, identity):
    """Bind the documented single-order preview to the local immutable intent."""
    try:
        if (str(remote["accountId"]) != identity or remote["operation"] != "place_order"
                or remote["status"] != "prepared" or len(remote["legs"]) != 1):
            raise ValueError("account, operation or legs changed")
        leg = remote["legs"][0]
        if leg["preview"]["wouldSucceed"] is not True:
            raise ValueError("remote preview did not pass")
        aliases = {order["contract"].upper(), order.get("broker_contract", order["contract"]).upper().split(".")[0]}
        rows = [remote["parameters"], leg]
        detail = leg["preview"].get("detail")
        if detail is not None:
            rows.append(detail)
        for row in rows:
            contract, _, venue = row["contractCode"].upper().partition(".")
            if (contract not in aliases or (venue and venue != order["exchange"])
                    or row["side"] != order["side"] or row["offset"] != order["offset"]
                    or integer(row["volume"], "volume") != order["volume"]
                    or row.get("price") != order["price"]):
                raise ValueError("contract, side, offset, quantity or price changed")
            if row is not detail and row["clientOrderId"] != order["client_order_id"]:
                raise ValueError("client order identity changed")
            if (row.get("orderType", order["order_type"]) != order["order_type"]
                    or row.get("timeInForce", order["time_in_force"]).lower() != order["time_in_force"]):
                raise ValueError("order type changed")
        if detail is not None and detail.get("contract", {}).get("exchange", order["exchange"]) != order["exchange"]:
            raise ValueError("exchange changed")
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError("Panda frozen order does not match the local order") from exc


def _frozen_plan_signature(plan):
    # prepare returns an ISO expiry; GET plan returns epoch timestamps.
    value = dict(plan)
    value.pop("createdAtEpoch", None)
    epoch = value.pop("expiresAtEpoch", None)
    expiry = value.pop("expiresAt", None)
    if epoch is not None:
        if expiry is not None and int(float(epoch)) != int(datetime.fromisoformat(expiry).timestamp()):
            raise ValueError("Panda plan expiry representations disagree")
        value["expiresAt"] = int(float(epoch))
    elif expiry is not None:
        value["expiresAt"] = int(datetime.fromisoformat(expiry).timestamp())
    return digest(value)


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
    if _frozen_plan_signature(remote) != _frozen_plan_signature(prepared["remote"]):
        raise ValueError("Panda frozen plan changed; preview and confirm again")
    _validate_frozen_order(remote, order, plan["identity"])
    # Fetching the remote plan can consume the remaining local validity/window.
    validate_plan(plan)
    initial = {"status": "sending", "client_order_id": order["client_order_id"],
               "local_plan_id": plan["plan_id"], "remote_plan_id": remote["planId"],
               "intent": {"contract": order.get("broker_contract", order["contract"]),
                          "side": order["side"], "offset": order["offset"], "volume": order["volume"],
                          "exchange": order["exchange"]}}
    try:
        with attempt.open("xb") as handle:
            handle.write(encoded(initial))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ValueError("order already attempted by another process") from exc
    try:
        validate_plan(plan)
    except ValueError:
        # The durable journal itself may straddle the cutoff. Nothing was sent.
        result = {"status": "not_sent", "reason": "local_plan_expired_before_send"}
    else:
        try:
            result = client.execute_plan(remote["planId"], confirmed=True,
                                         client_request_id=order["client_order_id"])
        except Exception as exc:
            # Even transport failure remains a recorded attempt. There is no retry loop.
            result = {"status": "unknown", "error_type": type(exc).__name__,
                      "next_action": "query_orders_by_clientOrderId"}
    record = {**initial, "status": ("reconciled" if result.get("status") == "not_sent"
                                   else result.get("status", "unknown")), "response": result}
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
    orders, complete = [], False
    if hasattr(client, "orders_page"):
        cursor, seen_cursors = None, set()
        for _ in range(10):
            page = client.orders_page(client_order_id=client_order_id, count=200, last_id=cursor)
            orders.extend(page["items"])
            meta = page.get("meta", {})
            if meta.get("hasMore") is False:
                complete = True
                break
            cursor = meta.get("nextLastId")
            if meta.get("hasMore") is not True or cursor is None or str(cursor) in seen_cursors:
                break
            seen_cursors.add(str(cursor))
    else:
        orders = client.orders(client_order_id=client_order_id, count=200)
        complete = len(orders) < 200
    operation_id = record.get("response", {}).get("operationId")
    operation = client.operation(operation_id) if operation_id else None
    # Command completion is not a fill. Require terminal order evidence, or
    # the service's explicit expired operation after its own queue reconciliation.
    if not complete:
        record["status"] = "unknown"
    elif orders and all(o.get("status") in {"filled", "cancelled", "rejected"} for o in orders):
        record["status"] = "reconciled"
    elif not orders and operation and operation.get("status") == "expired":
        record["status"] = "expired"
    record["reconciliation"] = {"orders": orders, "operation": operation, "orders_complete": complete}
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    temp.write_bytes(encoded(record))
    temp.replace(path)
    return record


def session(request, client):
    """Run an explicitly enabled contest day using one persistent SDK client."""
    from trading.artifacts import file_hash
    from trading.automation import run_rebalance
    policy = request["policy"]
    if (request.get("mode") != "simulation" or policy.get("channel") != "panda"
            or policy.get("allow_simulation_strategy") is not True
            or policy.get("execute_enabled") is not True):
        raise ValueError("automatic execution requires an enabled simulation contest account")

    def unchanged():
        if file_hash(request["config_path"]) != request["config_sha256"]:
            raise ValueError("automatic configuration changed; stop sending")

    def snapshot():
        unchanged()
        return normalize_snapshot(client.snapshot(), policy)

    def submit(plan):
        unchanged()
        frozen = prepare(plan, 0, client)
        unchanged()
        # run_rebalance already holds the same per-account execute.lock.
        return _execute_locked(plan, frozen, client, request["journal_root"],
                               confirmation=frozen["confirmation"])

    def poll(receipt):
        if receipt.get("response", {}).get("status") == "not_sent":
            return {"status": "reconciled", "filled_quantity": 0, "not_sent": True}
        # Query the original ID even if configuration changed after submission.
        # A stop request must not prevent reconciliation of an existing attempt.
        record = reconcile(client, request["journal_root"], policy["expected_identity"],
                           receipt["client_order_id"])
        evidence = record.get("reconciliation", {})
        orders = evidence.get("orders", [])
        if record["status"] == "reconciled":
            if not orders or evidence.get("orders_complete") is not True:
                return {"status": "unknown", "reason": "ORDER_EVIDENCE_INCOMPLETE"}
            identifiers = [row.get("orderId") for row in orders]
            if None in identifiers or len(set(identifiers)) != len(identifiers):
                return {"status": "unknown", "reason": "ORDER_IDENTIFIERS_INCOMPLETE"}
            try:
                fills = [integer(row["filledQuantity"], "filledQuantity") for row in orders]
                quantities = [integer(row["quantity"], "quantity") for row in orders]
            except (KeyError, ValueError, TypeError):
                return {"status": "unknown", "reason": "FILL_QUANTITY_NOT_VERIFIED"}
            if any(f > q for f, q in zip(fills, quantities)):
                return {"status": "unknown", "reason": "FILL_QUANTITY_CONFLICT"}
            intent = record.get("intent", {})
            direction = {("buy", "open"): "open_long", ("sell", "open"): "open_short",
                         ("sell", "close"): "close_long", ("buy", "close"): "close_short"}.get(
                             (intent.get("side"), intent.get("offset")))
            if (direction is None or sum(quantities) != intent.get("volume") or any(
                    str(row.get("contractCode", "")).upper().split(".")[0]
                    != str(intent.get("contract", "")).upper().split(".")[0]
                    or (intent.get("exchange") and any(
                        venue and venue.upper() != intent["exchange"].upper() for venue in
                        (row.get("exchange"), str(row.get("contractCode", "")).partition(".")[2])))
                    or row.get("tradeDirection") != direction
                    or (row.get("clientOrderId") is not None and row["clientOrderId"] != receipt["client_order_id"])
                    or (row["status"] == "filled" and fill != quantity)
                    for row, fill, quantity in zip(orders, fills, quantities))):
                return {"status": "unknown", "reason": "ORDER_EVIDENCE_DOES_NOT_MATCH_INTENT"}
            return {"status": "reconciled", "filled_quantity": sum(fills),
                    "rejected": any(row["status"] == "rejected" for row in orders), **evidence}
        return {"status": record["status"], **evidence}

    unchanged()
    return run_rebalance(request["sized"], account_id=request["account_id"], policy=policy,
        signal_id=request["signal_id"], signal_date=request["signal_date"], trade_date=request["trade_date"],
        journal_root=request["journal_root"], snapshot_fn=snapshot, submit_fn=submit, poll_fn=poll,
        prepared_positions_key=request.get("prepared_positions_key"), previous_plan=request.get("previous_plan"))


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
    elif action == "session":
        result = session(request, client)
    else:
        raise ValueError("unsupported Panda worker action")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
