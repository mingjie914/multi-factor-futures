"""Pure account reconciliation and explicit paper execution of frozen plans."""
from __future__ import annotations

import copy
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from math import isfinite
import re

from trading.artifacts import digest


def timestamp(value):
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        raise ValueError("snapshot timestamp must include a timezone")
    return parsed


def validate_execution_window(policy, *, now=None):
    """Validate an explicitly dated send window; absence preserves manual use.

    Completion is a target, not a promise by the broker. Orders must stop being
    submitted earlier, with room for the channel's queue and reconciliation.
    Calendar selection belongs to the caller; never infer T-1 from weekdays.
    """
    if "execution_window" not in policy:
        return None
    window = policy["execution_window"]
    try:
        start, stop, completion = (
            timestamp(window[key]) for key in ("start_at", "submit_until", "completion_by")
        )
        shanghai = timezone(timedelta(hours=8))
        execution_date = start.astimezone(shanghai).date()
        if (not start < stop <= completion
                or any(t.astimezone(shanghai).date() != execution_date for t in (stop, completion))
                or date.fromisoformat(policy["expected_signal_date"]) >= execution_date):
            raise ValueError("invalid dated window or signal date")
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None or not start <= current < stop:
            raise ValueError("outside execution window")
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"execution window: {exc}") from exc
    return stop


def snapshot_hash(snapshot):
    # Fresh read timestamps differ even when the complete account state does not.
    return digest({k: v for k, v in snapshot.items() if k != "as_of"})


def snapshot_positions_hash(snapshot):
    return digest({**{key: snapshot[key] for key in ("identity", "trade_date", "open_orders")},
                   "positions": sorted(snapshot["positions"], key=lambda p: (p["contract"], p["side"]))})


def integer(value, field):
    if isinstance(value, bool) or not isfinite(float(value)) or int(value) != float(value) or int(value) < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return int(value)


def signed_integer(value, field):
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    quantity = integer(abs(value), field)
    return -quantity if value < 0 else quantity


def number(value, field, *, positive=False):
    value = float(value)
    if not isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{field} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def validate_snapshot(snapshot):
    timestamp(snapshot["as_of"])
    if not snapshot.get("identity"):
        raise ValueError("snapshot requires account identity")
    for field in ("equity", "available"):
        if not isfinite(float(snapshot[field])):
            raise ValueError(f"{field} must be finite")
    number(snapshot["margin_used"], "margin_used")
    if not isinstance(snapshot["open_orders"], list) or not isinstance(snapshot["positions"], list):
        raise ValueError("positions and open_orders must be complete lists")
    seen = set()
    for row in snapshot["positions"]:
        contract = str(row["contract"])
        if (row["root"] != str(row["root"]).upper() or contract != contract.upper()
                or not re.fullmatch(re.escape(row["root"]) + r"\d{2}(0[1-9]|1[0-2])", contract)):
            raise ValueError("positions require canonical concrete contracts")
        if row["exchange"] not in {"SHFE", "INE", "DCE", "CZCE", "CFFEX", "GFEX"}:
            raise ValueError("positions require a canonical exchange")
        if row["side"] not in {"long", "short"}:
            raise ValueError("position side must be long or short")
        key = (contract, row["side"])
        if key in seen:
            raise ValueError("duplicate position direction")
        seen.add(key)
        for field in ("volume", "today", "yesterday", "available_today", "available_yesterday"):
            if field not in row:
                raise ValueError(f"position requires {field}")
            integer(row[field], field)
        if (row["today"] + row["yesterday"] != row["volume"]
                or row["available_today"] > row["today"]
                or row["available_yesterday"] > row["yesterday"]):
            raise ValueError("position today/yesterday/available quantities conflict")


def build_plan(sized, snapshot, *, account_id, policy, signal_id, signal_date,
               previous_plan=None, now=None):
    now = now or datetime.now(timezone.utc)
    validate_snapshot(snapshot)
    blockers = list(sized["blockers"])
    if not sized["tradable"] and not blockers:
        blockers.append("SIZING_NOT_TRADABLE")
    if not policy.get("expected_identity") or snapshot["identity"] != policy["expected_identity"]:
        blockers.append("ACCOUNT_IDENTITY_MISMATCH")
    max_age = number(policy["max_snapshot_age_seconds"], "snapshot age", positive=True)
    ttl = number(policy["plan_ttl_seconds"], "plan ttl", positive=True)
    expires = now + timedelta(seconds=ttl)
    try:
        window_stop = validate_execution_window(policy, now=now)
        if window_stop is not None:
            expires = min(expires, window_stop)
            execution_date = window_stop.astimezone(timezone(timedelta(hours=8))).date().isoformat()
            if snapshot["trade_date"] != execution_date:
                blockers.append("EXECUTION_TRADE_DATE_MISMATCH")
    except ValueError:
        blockers.append("OUTSIDE_EXECUTION_WINDOW")
    if not 0 <= (now - timestamp(snapshot["as_of"])).total_seconds() <= max_age:
        blockers.append("STALE_SNAPSHOT")
    # Exchange trading date alone is insufficient: the explicit previous close
    # date is necessary at night / next morning and never inferred from weekdays.
    expected_close = policy.get("expected_signal_date", snapshot["trade_date"])
    if signal_date != expected_close:
        blockers.append("STALE_SIGNAL")
    if snapshot["open_orders"]:
        blockers.append("ACTIVE_ORDERS")
    equity = float(snapshot["equity"])
    margin_ratio = float(snapshot["margin_used"]) / equity if equity > 0 else None
    reduce_only = (equity <= 0 or margin_ratio >= float(sized["account"]["max_margin_ratio"])
                   or float(snapshot["available"]) <= float(sized["account"]["reserve"]))
    if not reduce_only and abs(float(sized["account"]["equity"]) - equity) > .01:
        blockers.append("EQUITY_CHANGED_RESIZE_REQUIRED")
    managed = set(policy["managed_roots"])
    if previous_plan is not None:
        old_body = {k: v for k, v in previous_plan.items() if k != "plan_id"}
        if digest(old_body) != previous_plan["plan_id"]:
            raise ValueError("previous plan hash mismatch")
        if previous_plan["identity"] != snapshot["identity"] or previous_plan["account_id"] != account_id:
            raise ValueError("previous plan belongs to a different account")
        managed.update(previous_plan["managed_roots"])
    elif policy.get("cold_start", "require_flat") == "require_flat" and snapshot["positions"]:
        blockers.append("COLD_START_NOT_FLAT")
    targets = {row["contract"]: row for row in sized["targets"]}
    if len(targets) != len(sized["targets"]):
        raise ValueError("duplicate target contract")
    for row in targets.values():
        signed_integer(row["target_lots"], "target_lots")
    if any(r["contract"] in targets and r["exchange"] != targets[r["contract"]]["exchange"]
           for r in snapshot["positions"]):
        blockers.append("POSITION_EXCHANGE_MISMATCH")
    if any(row["root"] not in managed for row in targets.values()):
        blockers.append("TARGET_OUTSIDE_MANAGED_SCOPE")
    unmanaged = [r for r in snapshot["positions"] if r["root"] not in managed and r["volume"]]
    if unmanaged:
        # Preserve these positions, and require their risk to be budgeted explicitly.
        blockers.append("UNMANAGED_POSITION_RISK_NOT_BUDGETED")
    typ = policy["order_type"]
    if typ not in {"market", "limit"}:
        raise ValueError("only market or explicit limit orders are supported")
    max_lots = integer(policy["max_order_lots"], "max_order_lots")
    if not max_lots:
        raise ValueError("max_order_lots must be positive")
    raw_orders = []

    def append_order(contract, root, exchange, side, offset, volume, broker_contract=None):
        if not volume:
            return
        spec = targets.get(contract)
        # Old contracts must supply their own specs; never borrow the new month's price.
        spec = spec or policy.get("old_contract_specs", {}).get(contract)
        if spec is None:
            blockers.append("OLD_CONTRACT_SPEC_REQUIRED")
        if spec is not None and spec.get("metadata_verified") is False:
            blockers.append(f"METADATA_UNVERIFIED: {contract}")
        price = None
        order_limit = min(max_lots, int(spec["max_order_lots"])) if spec and spec.get("max_order_lots") else max_lots
        if volume > order_limit:
            if policy.get("slice_orders") is True:
                # One allowed slice now; subsequent quantities are based on
                # confirmed fills, not pre-sent as an unchecked batch.
                volume = order_limit
            else:
                blockers.append("ORDER_LOT_LIMIT")
        if typ == "limit":
            value = policy.get("limit_prices", {}).get(contract)
            if value is None:
                blockers.append("LIMIT_PRICE_REQUIRED")
            elif spec is None or spec.get("tick") is None:
                blockers.append("OLD_CONTRACT_SPEC_REQUIRED")
            else:
                price = number(value, "limit price", positive=True)
                tick = Decimal(str(number(spec["tick"], "tick", positive=True)))
                if Decimal(str(price)) % tick:
                    blockers.append("PRICE_TICK_MISMATCH")
                close = number(spec["close"], "valuation price", positive=True)
                if abs(price / close - 1) > float(policy["max_price_deviation"]):
                    blockers.append("PRICE_DEVIATION")
                # Channel preview must additionally validate current exchange price bands.
                if spec.get("limit_down") is not None and price < float(spec["limit_down"]):
                    blockers.append("DAILY_PRICE_LIMIT")
                if spec.get("limit_up") is not None and price > float(spec["limit_up"]):
                    blockers.append("DAILY_PRICE_LIMIT")
        raw_orders.append({"root": root, "contract": contract, "exchange": exchange,
                           "broker_contract": broker_contract or contract,
                           "side": side, "offset": offset, "volume": int(volume),
                           "order_type": typ, "time_in_force": "ioc" if typ == "market" else "gfd",
                           "price": price})

    actual = {(r["contract"], r["side"]): r for r in snapshot["positions"] if r["root"] in managed}
    for (contract, direction), pos in sorted(actual.items()):
        desired = int(targets.get(contract, {}).get("target_lots", 0))
        keep = max(desired if direction == "long" else -desired, 0)
        close_volume = max(0, int(pos["volume"]) - keep)
        if close_volume > pos["available_today"] + pos["available_yesterday"]:
            blockers.append("INSUFFICIENT_CLOSEABLE")
            continue
        side = "sell" if direction == "long" else "buy"
        if pos["exchange"] in {"SHFE", "INE"}:
            old = min(close_volume, pos["available_yesterday"])
            append_order(contract, pos["root"], pos["exchange"], side, "close_yesterday", old, pos.get("broker_contract"))
            append_order(contract, pos["root"], pos["exchange"], side, "close_today", close_volume - old, pos.get("broker_contract"))
        else:
            append_order(contract, pos["root"], pos["exchange"], side, "close", close_volume, pos.get("broker_contract"))
    suppressed_open_lots = 0
    for contract, row in sorted(targets.items()):
        lots = int(row["target_lots"])
        direction = "long" if lots > 0 else "short"
        current = actual.get((contract, direction), {}).get("volume", 0)
        opening = max(0, abs(lots) - current)
        if reduce_only:
            suppressed_open_lots += opening
            continue
        append_order(contract, row["root"], row["exchange"],
                     "buy" if lots > 0 else "sell", "open", opening)
    margin = number(sized["summary"]["estimated_margin"], "target margin")
    if policy["channel"] == "panda" and any(o["offset"] in {"close_today", "close_yesterday"} for o in raw_orders):
        blockers.append("PANDA_CLOSE_OFFSET_UNVERIFIED")
    reserve = number(sized["account"]["reserve"], "reserve")
    if not reduce_only and max(0, margin - float(snapshot["margin_used"])) + reserve > float(snapshot["available"]):
        blockers.append("AVAILABLE_FUNDS_CHANGED")
    suppressed_checks = []
    if reduce_only:
        capital_codes = {"GROSS_EXPOSURE", "NET_EXPOSURE", "EQUITY_ROUTE_BUDGET", "MARGIN", "AVAILABLE", "REQUIRE_ONE_LOT"}
        suppressed_checks = [b for b in blockers if b.split(":", 1)[0] in capital_codes]
        blockers = [b for b in blockers if b not in suppressed_checks]
    order_seed = {"signal": signal_id, "account": account_id, "policy": policy,
                  "snapshot": snapshot_hash(snapshot)}
    close_ids = []
    for index, row in enumerate(raw_orders):
        row["client_order_id"] = "mf_" + digest({"seed": order_seed, "index": index, "order": row})[:28]
        row["depends_on"] = list(close_ids) if row["offset"] == "open" else []
        if row["offset"] != "open":
            close_ids.append(row["client_order_id"])
    body = {"schema_version": 1, "account_id": account_id, "identity": snapshot["identity"],
            "channel": policy["channel"], "signal_id": signal_id, "signal_date": signal_date,
            "previous_signal_id": previous_plan["signal_id"] if previous_plan else None,
            "managed_roots": sorted(managed), "risk_mode": "reduce_only" if reduce_only else "normal",
            "suppressed_open_lots": suppressed_open_lots,
            "execution_phase": "close_then_reconcile" if close_ids else "single_phase",
            "suppressed_capital_checks": suppressed_checks,
            "risk": {"current_margin_ratio": margin_ratio,
                     "margin_reduction_to_cap": max(0, float(snapshot["margin_used"])
                         - equity * float(sized["account"]["max_margin_ratio"]))},
            "created_at": now.isoformat(), "expires_at": expires.isoformat(),
            "snapshot_hash": snapshot_hash(snapshot), "snapshot_positions_hash": snapshot_positions_hash(snapshot),
            "policy": policy, "sizing": sized,
            "orders": raw_orders, "blockers": sorted(set(blockers)), "ready": not blockers}
    return {**body, "plan_id": digest(body)}


def validate_plan(plan, snapshot=None, *, now=None):
    body = {key: value for key, value in plan.items() if key != "plan_id"}
    if digest(body) != plan["plan_id"]:
        raise ValueError("plan hash mismatch")
    now = now or datetime.now(timezone.utc)
    validate_execution_window(plan["policy"], now=now)
    if not timestamp(plan["created_at"]) <= now <= timestamp(plan["expires_at"]):
        raise ValueError("plan expired or not yet valid")
    if not plan["ready"] or plan["blockers"]:
        raise ValueError("plan is blocked")
    for order in plan["orders"]:
        if not integer(order["volume"], "order volume"):
            raise ValueError("order volume must be positive")
    if snapshot is not None and snapshot_hash(snapshot) != plan["snapshot_hash"]:
        if (plan["policy"].get("allow_mark_to_market") is not True
                or snapshot_positions_hash(snapshot) != plan.get("snapshot_positions_hash")):
            raise ValueError("account snapshot changed; reconcile again")
        # Prices continuously change cash marks in a nonempty futures account.
        # Keep quantities immutable and recheck current risk, rather than making
        # a penny of PnL require an endless new factor/target calculation.
        from trading.automation import refresh_fixed_targets
        fresh = build_plan(refresh_fixed_targets(plan["sizing"], snapshot), snapshot,
            account_id=plan["account_id"], policy=plan["policy"], signal_id=plan["signal_id"],
            signal_date=plan["signal_date"], previous_plan=plan, now=now)
        if not fresh["ready"] or (fresh["risk_mode"] == "reduce_only"
                                  and any(o["offset"] == "open" for o in plan["orders"])):
            raise ValueError("account marked funds no longer pass execution risk")
    if snapshot is not None and not 0 <= (now - timestamp(snapshot["as_of"])).total_seconds() <= float(
            plan["policy"]["max_snapshot_age_seconds"]):
        raise ValueError("account snapshot expired or not yet valid")


def paper_execute(plan, snapshot, *, now=None):
    """Deterministic ideal fills for integration tests, never a brokerage claim."""
    validate_plan(plan, snapshot, now=now)
    if plan["channel"] != "paper":
        raise ValueError("paper execution requires channel=paper")
    result = copy.deepcopy(snapshot)
    positions = {(r["contract"], r["side"]): r for r in result["positions"]}
    fills = []
    for order in plan["orders"]:
        opening = order["offset"] == "open"
        direction = ("long" if order["side"] == "buy" else "short") if opening else (
            "long" if order["side"] == "sell" else "short")
        key = (order["contract"], direction)
        if opening:
            pos = positions.setdefault(key, {"root": order["root"], "contract": order["contract"],
                "exchange": order["exchange"], "side": direction, "volume": 0, "today": 0,
                "yesterday": 0, "available_today": 0, "available_yesterday": 0})
            for field in ("volume", "today", "available_today"):
                pos[field] += order["volume"]
        else:
            pos = positions[key]
            left = order["volume"]
            buckets = ["today"] if order["offset"] == "close_today" else ["yesterday", "today"]
            for field in buckets:
                take = min(left, pos["available_" + field])
                pos[field] -= take
                pos["available_" + field] -= take
                pos["volume"] -= take
                left -= take
            if left:
                raise ValueError("paper close unavailable")
        fills.append({**order, "status": "filled", "simulation": True})
    result["positions"] = [r for r in positions.values() if r["volume"]]
    # Use executed positions, including deliberately suppressed opens in reduce-only mode.
    specs = {r["contract"]: r for r in plan["sizing"]["targets"]}
    specs.update(plan["policy"].get("old_contract_specs", {}))
    margin = (sum(r["volume"] * float(specs[r["contract"]]["close"])
                  * float(specs[r["contract"]]["multiplier"])
                  * float(specs[r["contract"]]["margin_rate"]) for r in result["positions"])
              if all(r["contract"] in specs for r in result["positions"])
              else result["margin_used"])
    result["available"] += result["margin_used"] - margin
    result["margin_used"] = margin
    result["as_of"] = (now or datetime.now(timezone.utc)).isoformat()
    return {"status": "completed", "simulation": True, "fill_model": "ideal_quantity_no_fees_no_pnl",
            "plan_id": plan["plan_id"], "fills": fills, "snapshot": result}
