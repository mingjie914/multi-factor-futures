"""Dated preparation budgets and a bounded, broker-independent rebalance loop.

Heavy research is deliberately absent from the execution loop. The caller hands
over a verified, immutable sizing artifact and supplies its channel operations.
"""
from __future__ import annotations

import copy
from datetime import date, datetime, time, timedelta, timezone
import json
from math import isfinite
import os
from pathlib import Path
import re
import time as clock_module

from trading.artifacts import digest, encoded, read_artifact
from trading.execution import build_plan, integer, signed_integer, number, timestamp, validate_execution_window, validate_snapshot


SHANGHAI = timezone(timedelta(hours=8))
TIME_FIELDS = ("prepare_at", "freeze_at", "preflight_at", "start_at", "submit_until", "completion_by")


def dated_schedule(trade_date, config):
    try:
        day = date.fromisoformat(trade_date)
        result = {}
        for key in TIME_FIELDS:
            value = config[key]
            if not isinstance(value, str) or not re.fullmatch(r"\d{2}:\d{2}:\d{2}", value):
                raise ValueError("schedule times require HH:MM:SS")
            result[key] = datetime.combine(day, time.fromisoformat(value), SHANGHAI)
        values = list(result.values())
        if not all(a < b for a, b in zip(values[:4], values[1:5])) or values[4] > values[5]:
            raise ValueError("schedule times are not ordered")
        return result
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid daily schedule: {exc}") from exc


def prepare_budget(schedule, now, timeout_seconds):
    if (isinstance(timeout_seconds, bool) or not isfinite(float(timeout_seconds))
            or float(timeout_seconds) <= 0 or now.tzinfo is None
            or not schedule["prepare_at"] <= now < schedule["freeze_at"]):
        raise ValueError("weight preparation is outside its budget or time window")
    return min(float(timeout_seconds), (schedule["freeze_at"] - now).total_seconds())


def previous_trade_date(trade_date, calendar):
    try:
        first, last = (date.fromisoformat(calendar[key]) for key in ("valid_from", "valid_through"))
        days = [date.fromisoformat(value) for value in calendar["trading_days"]]
        day = date.fromisoformat(trade_date)
        if (not isinstance(calendar["source"], str) or not calendar["source"].strip()
                or first > last or days != sorted(set(days))
                or any(d < first or d > last for d in days) or day < first or day > last):
            raise ValueError("calendar metadata, bounds or ordering invalid")
        index = days.index(day)
        if index == 0:
            raise ValueError("preceding trading day is not covered")
        return days[index - 1].isoformat()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"trading calendar cannot resolve {trade_date}: {exc}") from exc


def write_state(path, value):
    """Atomically replace one local state file after its payload reaches disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(encoded(value))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def position_key(snapshot):
    # Exclude marks, PnL and query timestamps, but retain broker closeability.
    return digest({"identity": snapshot["identity"],
                   "positions": sorted(snapshot["positions"], key=lambda p: (p["contract"], p["side"]))})


def previous_account_plan(journal_root, identity, account_id, trade_date):
    """Carry managed scope forward only from intact prior local session plans."""
    directory = Path(journal_root) / digest({"identity": identity})
    previous = None
    for path in sorted(directory.glob("session_*.json")):
        if path.stem.removeprefix("session_") >= trade_date:
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["status"] in {"waiting", "executing", "unknown"}:
            raise ValueError("UNRESOLVED_PREVIOUS_SESSION")
        plan = record.get("last_plan")
        if plan is not None:
            if (digest({k: v for k, v in plan.items() if k != "plan_id"}) != plan["plan_id"]
                    or plan["identity"] != identity or plan["account_id"] != account_id):
                raise ValueError("previous account plan identity/hash mismatch")
            previous = plan
    return previous


def prepare_day(settings, *, account_id, trade_date, calendar, state_path,
                snapshot_fn, weights_fn, sizing_fn, clock=None):
    """Produce a timed, auditable handoff; only weights_fn may do heavy work.

    weights_fn(signal_date, holdings, timeout_seconds) returns an immutable
    weights directory. sizing_fn(directory, fresh_snapshot, trade_date) returns
    an immutable sizing directory. Neither callback is allowed to submit orders.
    """
    clock = clock or (lambda: datetime.now(timezone.utc))
    schedule = dated_schedule(trade_date, settings["automation"])
    budget = prepare_budget(schedule, clock(), settings["automation"]["preparation_timeout_seconds"])
    signal_date = previous_trade_date(trade_date, calendar)
    policy = settings["execution"][account_id]
    started = clock_module.perf_counter()
    state = {"schema_version": 1, "account_id": account_id, "trade_date": trade_date,
             "signal_date": signal_date, "status": "preparing", "started_at": clock().isoformat(),
             "settings_sha256": digest(settings), "preparation_budget_seconds": budget}
    write_state(state_path, state)
    try:
        before = snapshot_fn()
        validate_snapshot(before)
        if before["identity"] != policy["expected_identity"] or before["trade_date"] != trade_date:
            raise ValueError("ACCOUNT_OR_TRADE_DATE_MISMATCH")
        if before["open_orders"]:
            raise ValueError("ACTIVE_ORDERS_BEFORE_WEIGHT_PREPARATION")
        if not 0 <= (clock() - timestamp(before["as_of"])).total_seconds() <= float(policy["max_snapshot_age_seconds"]):
            raise ValueError("STALE_PREPARATION_SNAPSHOT")
        instances = {r["strategy_id"] for r in settings["routes"] if r["account_id"] == account_id}
        if len(instances) != 1:
            raise ValueError("AUTOMATIC_BUFFER_STATE_REQUIRES_ONE_STRATEGY_PER_ACCOUNT")
        roots = {}
        for position in before["positions"]:
            if position["root"] not in policy["managed_roots"]:
                raise ValueError("UNMANAGED_POSITION_BEFORE_WEIGHT_PREPARATION")
            roots[position["root"]] = roots.get(position["root"], 0) + position["volume"] * (
                1 if position["side"] == "long" else -1)
        holdings = {next(iter(instances)): {"data_date": signal_date,
                    "source": "reconciled-position:" + position_key(before), "holdings": roots}}
        state.update(holdings=holdings, preparation_snapshot=before,
                     prepared_positions_key=position_key(before))
        write_state(state_path, state)
        # Recalculate after account lookup: network latency consumes the budget.
        budget = prepare_budget(schedule, clock(), settings["automation"]["preparation_timeout_seconds"])
        weights = Path(weights_fn(signal_date, holdings, budget)).resolve()
        manifest = read_artifact(weights)
        if manifest["identity"].get("stage") != "weights" or manifest["identity"].get("data_date") != signal_date:
            raise ValueError("WEIGHTS_NOT_FOR_EXPECTED_SIGNAL_DATE")
        state.update(weights_directory=str(weights), weights_id=weights.name,
                     weights_elapsed_seconds=clock_module.perf_counter() - started)
        if clock() >= schedule["freeze_at"]:
            raise ValueError("WEIGHTS_FINISHED_AFTER_FREEZE")
        after = snapshot_fn()
        validate_snapshot(after)
        if (after["trade_date"] != trade_date or after["open_orders"]
                or position_key(after) != position_key(before)):
            raise ValueError("HOLDINGS_CHANGED_DURING_WEIGHT_PREPARATION")
        if not 0 <= (clock() - timestamp(after["as_of"])).total_seconds() <= float(policy["max_snapshot_age_seconds"]):
            raise ValueError("STALE_POST_PREPARATION_SNAPSHOT")
        sizing = Path(sizing_fn(weights, after, trade_date)).resolve()
        sized_manifest = read_artifact(sizing)
        if (sized_manifest["identity"].get("stage") != "size"
                or sized_manifest["identity"].get("weights_id") != weights.name):
            raise ValueError("SIZING_HANDOFF_NOT_BOUND_TO_WEIGHTS")
        sized = json.loads((sizing / "sizing.json").read_text(encoding="utf-8"))["accounts"][account_id]
        if clock() >= schedule["freeze_at"]:
            raise ValueError("PREPARATION_FINISHED_AFTER_FREEZE")
        state.update(status="prepared" if sized["tradable"] else "blocked",
                     blockers=sized["blockers"], sizing_directory=str(sizing), sizing_id=sizing.name,
                     last_snapshot=after, finished_at=clock().isoformat(),
                     elapsed_seconds=clock_module.perf_counter() - started)
        write_state(state_path, state)
        return state
    except Exception as exc:
        # A failed/late new attempt supersedes the old pointer; never fall back
        # to yesterday's successful signal just because a file is available.
        state.update(status="blocked", error_type=type(exc).__name__,
                     reason=str(exc) if isinstance(exc, (ValueError, TimeoutError)) else "preparation failed",
                     finished_at=clock().isoformat(), elapsed_seconds=clock_module.perf_counter() - started)
        write_state(state_path, state)
        raise


def residual_positions(sized, snapshot):
    targets = {p["contract"]: p for p in sized["targets"]}
    positions = {(p["contract"], p["side"]): p for p in snapshot["positions"]}
    keys = set(positions)
    keys.update((code, "long" if row["target_lots"] > 0 else "short")
                for code, row in targets.items() if row["target_lots"])
    result = []
    for contract, side in sorted(keys):
        signed = signed_integer(targets.get(contract, {}).get("target_lots", 0), "target_lots")
        target = max(signed if side == "long" else -signed, 0)
        actual = int(positions.get((contract, side), {}).get("volume", 0))
        if actual != target:
            result.append({"contract": contract, "side": side, "target_lots": target,
                           "actual_lots": actual, "remaining_lots": target - actual})
    return result


def refresh_fixed_targets(sized, snapshot):
    """Refresh account risk without changing the frozen integer target vector."""
    validate_snapshot(snapshot)
    result = copy.deepcopy(sized)
    account = result["account"]
    for key in ("equity", "available", "margin_used", "frozen_margin"):
        account[key] = float(snapshot.get(key, 0))
    risk_codes = {"GROSS_EXPOSURE", "NET_EXPOSURE", "MARGIN", "AVAILABLE", "EQUITY_ROUTE_BUDGET"}
    blockers = [b for b in result["blockers"] if b.split(":", 1)[0] not in risk_codes]
    summary = result["summary"]
    equity, reserve = account["equity"], float(account["reserve"])
    if equity > 0:
        summary["gross_exposure"] = float(summary["gross_notional"]) / equity
        summary["net_exposure"] = float(summary["net_notional"]) / equity
        if summary["gross_exposure"] > float(account["max_gross_exposure"]):
            blockers.append("GROSS_EXPOSURE: current equity")
        if abs(summary["net_exposure"]) > float(account["max_abs_net_exposure"]):
            blockers.append("NET_EXPOSURE: current equity")
        equity_budget = sum(float(r["amount"]) for r in result.get("routes", [])
                            if r["capital_basis"] == "equity")
        if equity_budget > equity:
            blockers.append("EQUITY_ROUTE_BUDGET: current equity")
    # build_plan's reduce-only path removes funding gates, never metadata gates.
    margin = float(summary["estimated_margin"])
    if margin + account["frozen_margin"] + reserve > equity * float(account["max_margin_ratio"]):
        blockers.append("MARGIN: current equity and frozen funds")
    if max(0, margin - account["margin_used"]) + reserve > account["available"]:
        blockers.append("AVAILABLE: current available funds")
    result.update(blockers=blockers, tradable=not blockers)
    return result


def run_rebalance(sized, *, account_id, policy, signal_id, signal_date, trade_date,
                  journal_root, snapshot_fn, submit_fn, poll_fn,
                  prepared_positions_key=None, previous_plan=None,
                  clock=None, sleep=None):
    """One account/day session; one outstanding attempt, no research or re-sizing.

    submit_fn(plan) submits ONLY plan['orders'][0]. poll_fn(receipt) must provide
    terminal order evidence as status=reconciled (command completion alone is
    insufficient). Known IOC remainders are replanned, never re-sent verbatim.
    The account execution lock is shared with the manual Panda entrypoint.
    """
    clock = clock or (lambda: datetime.now(timezone.utc))
    sleep = sleep or clock_module.sleep
    window = policy.get("execution_window") or {}
    start, stop, completion = (timestamp(window[key]) for key in ("start_at", "submit_until", "completion_by"))
    validate_execution_window(policy, now=start)
    if start.astimezone(SHANGHAI).date().isoformat() != trade_date:
        raise ValueError("execution window does not belong to the requested trade date")
    interval = number(policy.get("poll_interval_seconds", 2), "poll interval", positive=True)
    if interval < 2:
        raise ValueError("poll interval must be at least 2 seconds")
    max_attempts = policy.get("max_attempts_per_leg", 3)
    if isinstance(max_attempts, bool) or int(max_attempts) != max_attempts or max_attempts < 1:
        raise ValueError("max_attempts_per_leg must be a positive integer")
    account_dir = Path(journal_root) / digest({"identity": policy["expected_identity"]})
    account_dir.mkdir(parents=True, exist_ok=True)
    lock = account_dir / "execute.lock"
    try:
        with lock.open("x", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
    except FileExistsError as exc:
        raise ValueError("account execution is locked") from exc
    path = account_dir / f"session_{trade_date}.json"
    identity = {"account_id": account_id, "identity": policy["expected_identity"],
                "trade_date": trade_date, "signal_id": signal_id, "signal_date": signal_date,
                "sized": sized, "policy": policy}
    report = {"session_id": digest(identity), "identity": identity, "status": "waiting",
              "orders": [], "snapshots": [], "remaining": [], "reason": None}
    pending = None
    expected_position = None
    terminal_stop_reason = None
    last = None
    attempts = {}
    first = True
    owns_report = False

    def finish(status, reason=None):
        report.update(status=status, reason=reason, finished_at=clock().isoformat())
        if last is not None:
            report["remaining"] = residual_positions(sized, last)
        write_state(path, report)
        return report

    try:
        if path.exists():
            prior = json.loads(path.read_text(encoding="utf-8"))
            if prior["session_id"] != report["session_id"]:
                raise ValueError("account/day session already bound to a different target or policy")
            # A restart cannot silently replay a partly sent or completed day.
            return {**prior, "replayed": False, "restart_requires_reconciliation":
                    prior["status"] in {"waiting", "executing", "unknown"}}
        owns_report = True
        previous_account_plan(journal_root, policy["expected_identity"], account_id, trade_date)
        # Do not start a new day while a prior broker attempt remains uncertain.
        for entry in account_dir.glob("mf_*.json"):
            previous = json.loads(entry.read_text(encoding="utf-8"))
            if previous["status"] != "reconciled" and not (
                    previous["status"] == "expired" and previous.get("reconciliation")):
                return finish("blocked", "UNRESOLVED_PREVIOUS_ATTEMPT")
        write_state(path, report)
        while clock() < start:
            sleep(min(10, max(0, (start - clock()).total_seconds())))
        while True:
            if pending is not None:
                evidence = poll_fn(pending)
                report["orders"][-1]["reconciliation"] = evidence
                if evidence.get("status") == "reconciled":
                    if "filled_quantity" not in evidence:
                        return finish("unknown", "TERMINAL_ORDER_MISSING_FILL_QUANTITY")
                    filled = integer(evidence["filled_quantity"], "filled quantity")
                    sent = report["orders"][-1]["order"]
                    if filled > sent["volume"]:
                        return finish("unknown", "ORDER_FILL_EXCEEDS_INTENT")
                    side = ("long" if sent["side"] == "buy" else "short") if sent["offset"] == "open" else (
                        "long" if sent["side"] == "sell" else "short")
                    before = next((p["volume"] for p in last["positions"]
                                   if (p["contract"], p["side"]) == (sent["contract"], side)), 0)
                    expected_position = (sent["contract"], side,
                                         before + filled * (1 if sent["offset"] == "open" else -1))
                    if evidence.get("rejected"):
                        terminal_stop_reason = "ORDER_REJECTED"
                    pending = None
                elif evidence.get("status") in {"unknown", "expired", "failed", "rejected"}:
                    return finish("unknown" if evidence["status"] == "unknown" else "partial",
                                  f"ORDER_{evidence['status'].upper()}")
                elif clock() >= completion:
                    return finish("unknown", "COMPLETION_DEADLINE_WITH_PENDING_ORDER")
                else:
                    write_state(path, report)
                    sleep(min(interval, max(0, (completion - clock()).total_seconds())))
                    continue
            last = snapshot_fn()
            validate_snapshot(last)
            report["snapshots"].append(last)
            if last["identity"] != policy["expected_identity"] or last["trade_date"] != trade_date:
                return finish("blocked", "ACCOUNT_OR_TRADE_DATE_MISMATCH")
            if not 0 <= (clock() - timestamp(last["as_of"])).total_seconds() <= float(policy["max_snapshot_age_seconds"]):
                return finish("blocked", "STALE_SNAPSHOT")
            if last["open_orders"]:
                if expected_position is not None and clock() < completion:
                    sleep(min(interval, max(0, (completion - clock()).total_seconds())))
                    continue
                return finish("unknown" if expected_position is not None else "blocked",
                              "ACTIVE_ORDERS_NOT_RECONCILED")
            if expected_position is not None:
                contract, side, volume = expected_position
                actual = next((p["volume"] for p in last["positions"]
                               if (p["contract"], p["side"]) == (contract, side)), 0)
                if actual != volume:
                    if clock() >= completion:
                        return finish("unknown", "FILL_POSITION_SNAPSHOT_NOT_RECONCILED")
                    sleep(min(interval, max(0, (completion - clock()).total_seconds())))
                    continue
                expected_position = None
            if first and prepared_positions_key is not None and position_key(last) != prepared_positions_key:
                return finish("blocked", "HOLDINGS_CHANGED_AFTER_PREPARATION")
            first = False
            report["remaining"] = residual_positions(sized, last)
            if not report["remaining"]:
                return finish("completed")
            if terminal_stop_reason:
                return finish("partial", terminal_stop_reason)
            if clock() >= stop:
                return finish("partial" if report["orders"] else "not_started", "SUBMISSION_DEADLINE")
            fresh = refresh_fixed_targets(sized, last)
            attempt_policy = {**policy, "intent_revision": f"{report['session_id']}:{len(report['orders'])}"}
            plan = build_plan(fresh, last, account_id=account_id, policy=attempt_policy,
                              signal_id=signal_id, signal_date=signal_date,
                              previous_plan=previous_plan, now=clock())
            report["last_plan"] = plan
            if not plan["ready"]:
                return finish("blocked", "; ".join(plan["blockers"]))
            if not plan["orders"]:
                return finish("partial", "REDUCE_ONLY_TARGET_NOT_REACHABLE")
            order = plan["orders"][0]
            leg = (order["contract"], order["side"], order["offset"])
            attempts[leg] = attempts.get(leg, 0) + 1
            if attempts[leg] > max_attempts:
                return finish("partial", "IOC_ATTEMPT_LIMIT")
            if order["depends_on"]:
                return finish("blocked", "CLOSE_NOT_RECONCILED")
            report["status"] = "executing"
            report["orders"].append({"plan_id": plan["plan_id"], "order": order,
                                     "before_send": clock().isoformat()})
            # Durable intent precedes even the callback. Callback owns the broker
            # client ID journal and repeats the window check immediately at send.
            write_state(path, report)
            if clock() >= stop:
                return finish("partial", "SUBMISSION_DEADLINE")
            pending = submit_fn(plan)
            report["orders"][-1]["receipt"] = pending
            previous_plan = plan
            write_state(path, report)
            if pending.get("status") == "unknown":
                return finish("unknown", "TRANSPORT_OR_RECEIPT_UNKNOWN")
    except Exception as exc:
        # Never print raw provider error bodies or guess whether a sent order filled.
        if owns_report:
            finish("unknown" if pending is not None or report["orders"] else "blocked", type(exc).__name__)
        raise
    finally:
        lock.unlink()
