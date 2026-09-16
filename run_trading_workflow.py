"""Independent IDE/CLI entry: weights -> size -> plan -> explicit execution.

The weights stage generates simulation close weights only; buffered strategies
require --holdings. Later stages require an explicit immutable input directory.
See docs/交易模块设计与验收.md.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from trading.artifacts import digest, file_hash, read_artifact, read_csv, write_artifact, resolve
from trading.execution import build_plan, paper_execute


# ============================== IDE SETTINGS ==============================
CONFIG_PATH = "config/trading.yaml"
# No-argument use intentionally stops after generating weights.
# ==========================================================================


def load_settings(path):
    settings = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    if not isinstance(settings, dict) or settings.get("schema_version") != 1:
        raise ValueError("trading config requires schema_version=1")
    return settings


def reference_specs(rows):
    """Indicative capital arithmetic only: no invented current tick or account margin."""
    from data.contract_specs import CONTRACT_SPECS
    specs = {}
    for row in rows:
        spec = CONTRACT_SPECS[row["root"]]
        specs[row["contract"]] = {"contract": row["contract"], "exchange": row["exchange"],
            "multiplier": spec["multiplier"], "margin_long": spec["margin"],
            "margin_short": spec["margin"], "tick": None, "as_of": "2026-08-04",
            "source": "data/contract_specs.py; historical reference, not account margin", "verified": False}
    return list(specs.values())


def size_artifact(weights_directory, settings, specs=None, *, specs_as_of=None):
    from trading.sizing import size_targets
    manifest = read_artifact(weights_directory)
    if manifest["identity"]["stage"] != "weights":
        raise ValueError("size requires a weights artifact")
    rows = read_csv(weights_directory, "weights.csv")
    groups = {}
    for row in rows:
        groups.setdefault(row["strategy_id"], []).append({**row, "close": float(row["close"]),
                                                         "weight": float(row["weight"])})
    specs = reference_specs(rows) if specs is None else specs
    result = size_targets(groups, specs, settings["accounts"], settings["routes"], settings["rounding"],
                          specs_as_of=specs_as_of)
    if {row["data_date"] for row in rows} != {manifest["identity"]["data_date"]}:
        raise ValueError("weights data_date does not match its manifest")
    identity = {"stage": "size", "weights_id": Path(weights_directory).name,
                "implementation_sha256": file_hash(resolve("trading/sizing.py")),
                "entrypoint_sha256": file_hash(__file__),
                "weights_manifest": manifest, "data_date": manifest["identity"]["data_date"],
                "mode": manifest["identity"]["mode"], "specs": specs,
                "accounts": settings["accounts"], "routes": settings["routes"],
                "rounding": settings["rounding"]}
    if specs_as_of is not None:
        identity["specs_as_of"] = specs_as_of
    targets = [{"account_id": account, "status": "READY" if value["tradable"] else "BLOCKED", **row}
               for account, value in result["accounts"].items() for row in value["targets"]]
    attribution = [{"account_id": account, **row} for account, value in result["accounts"].items() for row in value["attribution"]]
    daily = daily_position_rows(result, identity)
    account_summary = [{"数据日期": identity["data_date"], "账户": account,
                        "基准权益_元": value["account"]["equity"],
                        "预计占用保证金_元": value["summary"]["estimated_margin"],
                        "预计保证金占用率_%": round(
                            value["summary"]["estimated_margin"] / value["account"]["equity"] * 100, 6),
                        "总名义杠杆": value["summary"]["gross_exposure"],
                        "净敞口占权益": value["summary"]["net_exposure"],
                        "策略实例数": len(value["completeness"]),
                        "完整策略数": sum(c["complete"] for c in value["completeness"].values()),
                        "策略入选品种数": len({r["root"] for r in value["attribution"] if r["weight"] != 0}),
                        "目标持仓品种数": len({r["root"] for r in value["targets"] if r["target_lots"] != 0}),
                        "账户状态": "READY" if value["tradable"] else "BLOCKED",
                        "拦截原因": "; ".join(value["blockers"])}
                       for account, value in result["accounts"].items()]
    return write_artifact(resolve(settings["output_root"]) / "sizing" / identity["data_date"], identity,
                          {"targets.csv": targets, "attribution.csv": attribution, "sizing.json": result,
                           "weights.csv": rows, "daily_positions.csv": daily,
                           "account_summary.csv": account_summary,
                           "route_sizing.csv": [{"account_id": account, **row}
                               for account, value in result["accounts"].items() for row in value["routes"]],
                           "capital_requirements.csv": [{"strategy_id": sid, **row}
                               for sid, req in result["capital_requirements"].items() for row in req["rows"]]})


def daily_position_rows(result, identity):
    """Human-readable route history, derived from the one sizing calculation."""
    source = identity["weights_manifest"]["identity"]
    strategies = {row["strategy_id"]: row for row in source.get("strategies", [])}
    versions = {sid: digest(strategy) for sid, strategy in strategies.items()}
    specs = {row["contract"]: row for row in identity["specs"]}
    data = source.get("config", {}).get("data", {})
    lag = data.get("parquet", {}).get("dominant_lag_days")
    causal_main = data.get("source") in {"duckdb_futures", "parquet_futures"} and lag is not None
    rows = []
    for account_id, account in result["accounts"].items():
        targets = {row["contract"]: row for row in account["targets"]}
        for row in account["attribution"]:
            strategy = strategies.get(row["strategy_id"])
            spec = specs[row["contract"]]
            rows.append({
                "数据日期": row["data_date"],
                "执行交易日": identity.get("specs_as_of", row["data_date"]),
                "策略实例": row["strategy_id"],
                "来源策略": strategy.get("catalog_strategy_id", "") if strategy else "",
                "策略版本": versions.get(row["strategy_id"], ""),
                "账户": account_id,
                "分配序号": row["route_index"],
                "品种": row["root"],
                "合约类型": "滞后主力合约" if causal_main else "具体合约",
                "主力滞后交易日": lag if causal_main else None,
                "理论合约": row["contract"],
                "交易所": spec["exchange"],
                "基准权重": row["weight"],
                "资金口径": row["capital_basis"],
                "基准资金_元": row["amount"] if row["amount"] is not None else row["reference_capital"],
                "组合份数": row["units"],
                "折算基准金额_元": row["reference_capital"],
                "基准仓位_元": row["target_notional"],
                "理论手数": row["raw_lots"],
                "理论手数取整": row["rounded_lots"],
                "独立取整手数": row.get("independent_rounded_lots", row["rounded_lots"]),
                "取整方法": identity["rounding"],
                "取整后参考权重": row["rounded_weight"],
                "取整权重偏差": row["weight_error"],
                "组合完整": account["completeness"][row["strategy_id"]]["complete"],
                "组合最小理论手数": account["completeness"][row["strategy_id"]]["min_abs_raw_lots"],
                "参考价格": row["close"],
                "计价乘数": row["multiplier"],
                "单手价值_元": row["notional_per_lot"],
                "保证金率": row["margin_rate"],
                "单手保证金_元": row["margin_per_lot"],
                "预计占用保证金_元": row["estimated_margin"],
                "参数日期": spec["as_of"],
                "参数来源": spec["source"],
                "参数当日核实": spec["verified"] and spec["as_of"] == identity.get("specs_as_of", row["data_date"]),
                "账户合并目标手数": targets[row["contract"]]["target_lots"],
                "账户组合校正手数": targets[row["contract"]]["target_lots"] - targets[row["contract"]].get("independent_rounded_lots", targets[row["contract"]]["target_lots"]),
                "账户状态": "READY" if account["tradable"] else "BLOCKED",
                "权重快照": identity["weights_id"],
            })
    return sorted(rows, key=lambda row: (
        row["账户"], row["策略实例"], row["分配序号"],
        row["基准权重"] == 0, -row["基准权重"], row["品种"], row["理论合约"]))


def panda_call(request, *, interpreter=None, timeout=120):
    interpreter = interpreter or os.environ.get("MF_PANDA_PYTHON", "")
    if not interpreter or not Path(interpreter).is_file():
        raise ValueError("set MF_PANDA_PYTHON to the independently installed SDK Python executable")
    result = subprocess.run([interpreter, "-X", "utf8", "-m", "trading.panda"],
                            input=json.dumps(request), text=True, encoding="utf-8", capture_output=True,
                            cwd=Path(__file__).resolve().parent, timeout=timeout)
    if result.returncode:
        # Exceptions may include provider bodies. Keep detailed identity data local.
        raise RuntimeError("Panda worker failed; verify the SDK, account and frozen plan before retrying")
    return json.loads(result.stdout)


def artifact_json(path, filename):
    manifest = read_artifact(path)
    if filename not in manifest["files"]:
        raise ValueError(f"artifact missing {filename}")
    return json.loads((Path(path) / filename).read_text(encoding="utf-8"))


def run_automatic_day(settings, *, config_path, account_id, trade_date=None, prepare_only=False):
    """One local day process; research children have deadlines and close their DBs."""
    from trading.automation import (SHANGHAI, dated_schedule, prepare_day, previous_trade_date,
                                    write_state, position_key, previous_account_plan)
    auto = settings.get("automation", {})
    if auto.get("enabled") is not True:
        raise ValueError("automatic preparation is disabled")
    if settings["mode"] != "simulation":
        raise ValueError("automatic contest runner is restricted to simulation")
    now = lambda: datetime.now(timezone.utc)
    day = trade_date or now().astimezone(SHANGHAI).date().isoformat()
    calendar = json.loads(resolve(auto["calendar_path"]).read_text(encoding="utf-8"))
    schedule = dated_schedule(day, auto)
    policy = copy.deepcopy(settings["execution"][account_id])
    if policy["channel"] != "panda" or policy.get("allow_simulation_strategy") is not True:
        raise ValueError("automatic contest account must explicitly select Panda simulation")
    root = resolve(settings["output_root"])
    job = root / "automation" / digest({"identity": policy["expected_identity"]}) / day
    job.mkdir(parents=True, exist_ok=True)
    # Valid official holiday: no runner, no account calls, no weekday inference.
    if calendar["valid_from"] <= day <= calendar["valid_through"] and day not in calendar["trading_days"]:
        output = job / "runner.json"
        write_state(output, {"trade_date": day, "phase": "non_trading_day"})
        return output
    signal_date = previous_trade_date(day, calendar)
    previous = previous_account_plan(root / "execution_journal", policy["expected_identity"], account_id, day)
    if previous:
        policy["managed_roots"] = sorted(set(policy["managed_roots"]) | set(previous["managed_roots"]))
    known = copy.deepcopy(policy.get("known_contracts", {}))

    def remember_contracts(rows):
        for row in rows:
            metadata = {key: row[key] for key in ("contract", "root", "exchange")}
            aliases = {row["contract"], row["contract"].lower(), row.get("broker_contract", row["contract"])}
            for alias in aliases:
                if alias in known and known[alias] != metadata:
                    raise ValueError("conflicting known contract mapping")
                known[alias] = metadata
        policy["known_contracts"] = known

    if previous:
        remember_contracts(previous["sizing"]["targets"] + previous["orders"])
    lock = job / "runner.lock"
    try:
        with lock.open("x", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
    except FileExistsError as exc:
        raise ValueError("a daily runner already owns this account/date; inspect its state before recovery") from exc
    state_path = job / "preparation.json"
    status_path = job / "runner.json"
    config_path = resolve(config_path)
    original_config_hash = file_hash(config_path)
    interpreter = auto["sdk_python"]
    routes = [r for r in settings["routes"] if r["account_id"] == account_id]
    instance_ids = {r["strategy_id"] for r in routes}
    day_settings = copy.deepcopy(settings)
    day_settings["execution"][account_id] = policy
    day_settings["accounts"] = {account_id: day_settings["accounts"][account_id]}
    day_settings["routes"] = routes
    day_settings["strategy_ids"] = {key: value for key, value in settings["strategy_ids"].items() if key in instance_ids}

    runner_state = {}
    heartbeat_at = now()

    def heartbeat():
        nonlocal heartbeat_at
        heartbeat_at = now()
        write_state(job / "runner.heartbeat.json", {"pid": os.getpid(), "at": heartbeat_at.isoformat(),
                    "phase": runner_state.get("phase"), "trade_date": day})

    def status(phase, **extra):
        runner_state.clear()
        runner_state.update(trade_date=day, signal_date=signal_date, account_id=account_id,
                            phase=phase, at=now().isoformat(), pid=os.getpid(), **extra)
        write_state(status_path, runner_state)
        with (job / "runner.events.jsonl").open("a", encoding="utf-8") as events:
            events.write(json.dumps(runner_state, ensure_ascii=False) + "\n")
        heartbeat()

    def unchanged():
        if file_hash(config_path) != original_config_hash:
            raise ValueError("automatic configuration changed; current day must be prepared again")

    def wait_until(moment):
        while now() < moment:
            unchanged()
            if (now() - heartbeat_at).total_seconds() >= 60:
                heartbeat()
            time.sleep(min(10, max(0, (moment - now()).total_seconds())))

    def snapshot():
        unchanged()
        return panda_call({"action": "snapshot", "policy": policy}, interpreter=interpreter,
                          timeout=auto.get("query_timeout_seconds", 60))

    def weights(signal, holdings, budget):
        # The generated child settings limit the run to this account's instances.
        child_config = job / "weight_settings.yaml"
        child_config.write_text(yaml.safe_dump(day_settings, allow_unicode=True, sort_keys=False), encoding="utf-8")
        holdings_file = job / "holding_inputs.json"
        holdings_file.write_text(json.dumps(holdings, ensure_ascii=False), encoding="utf-8")
        status("computing_weights", timeout_seconds=budget)
        log = job / f"weights_{now().strftime('%H%M%S%f')}.log"
        with log.open("w", encoding="utf-8") as handle:
            try:
                completed = subprocess.run([sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()),
                    "--config", str(child_config), "weights", "--as-of", signal, "--holdings", str(holdings_file)],
                    cwd=Path(__file__).resolve().parent, stdout=handle, stderr=subprocess.STDOUT,
                    timeout=budget)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError("WEIGHT_COMPUTATION_BUDGET_EXCEEDED") from exc
        unchanged()
        if completed.returncode:
            raise ValueError(f"WEIGHT_GENERATION_FAILED: inspect {log.name}")
        lines = log.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValueError("weight generation returned no artifact")
        return resolve(lines[-1].strip())

    def sizing(weights_path, current, execution_date):
        current_settings = copy.deepcopy(day_settings)
        for key in ("equity", "available", "margin_used", "frozen_margin"):
            current_settings["accounts"][account_id][key] = current.get(key, 0)
        spec_path = auto.get("specs_path")
        specs = json.loads(resolve(spec_path).read_text(encoding="utf-8")) if spec_path and resolve(spec_path).is_file() else None
        return size_artifact(weights_path, current_settings, specs, specs_as_of=execution_date)

    try:
        unchanged()
        if now() >= schedule["submit_until"]:
            status("not_started", reason="SUBMISSION_DEADLINE")
            return status_path
        # A morning start and two inexpensive cache-aware retries; no late heavy work.
        retry_times = [datetime.combine(schedule["prepare_at"].date(),
                       datetime.strptime(value, "%H:%M:%S").time(), SHANGHAI)
                       for value in auto.get("retry_prepare_at", [])]
        if any(not schedule["prepare_at"] < value < schedule["freeze_at"] for value in retry_times):
            raise ValueError("weight retry times must be before freeze")
        moments = sorted(set([max(now(), schedule["prepare_at"])] + [t for t in retry_times if t > now()]))
        for moment in moments:
            if moment >= schedule["freeze_at"]:
                continue
            status("waiting_to_prepare", next_at=moment.isoformat())
            wait_until(moment)
            try:
                prepared = prepare_day(day_settings, account_id=account_id, trade_date=day,
                    calendar=calendar, state_path=state_path, snapshot_fn=snapshot,
                    weights_fn=weights, sizing_fn=sizing)
                status(prepared["status"], preparation=str(state_path), blockers=prepared.get("blockers", []))
            except (ValueError, TimeoutError, RuntimeError, subprocess.TimeoutExpired) as exc:
                status("preparation_blocked", error_type=type(exc).__name__, preparation=str(state_path))
            if prepare_only:
                return state_path if state_path.exists() else status_path
        if not state_path.is_file():
            status("blocked", reason="NO_PREPARED_SIGNAL")
            return status_path
        prepared = json.loads(state_path.read_text(encoding="utf-8"))
        if prepared.get("error_type") or not prepared.get("weights_directory"):
            status("blocked", reason="LATEST_PREPARATION_FAILED")
            return status_path
        status("waiting_to_freeze", preparation=str(state_path))
        wait_until(schedule["freeze_at"])
        weights_path = resolve(prepared["weights_directory"])
        manifest = read_artifact(weights_path)
        identity = manifest["identity"]
        if identity["data_date"] != signal_date:
            raise ValueError("prepared signal date does not match execution day")
        # The completed artifact owns its recipe/code version. Later research
        # edits do not invalidate its numbers; each new weights child reads anew.
        # Recheck the actual source used by this artifact, not today's research config.
        from types import SimpleNamespace
        from core.config import DataSourceConfig
        from data.manager import DataManager
        recorded_config = SimpleNamespace(market=identity["config"]["market"],
                                          data=DataSourceConfig(**identity["config"]["data"]))
        market = DataManager.from_config(recorded_config)
        try:
            calendar_start = market.get_calendar(identity["strategies"][0]["panel_start"], signal_date)[0]
            if market.source.checkpoint_source_fingerprint(calendar_start, signal_date) != identity["data_fingerprint"]:
                raise ValueError("certified data changed after preparation; do not recalculate after freeze")
        finally:
            market.source.close()
        status("frozen", weights_directory=str(weights_path))
        wait_until(schedule["preflight_at"])
        current = snapshot()
        if current["open_orders"] or position_key(current) != prepared["prepared_positions_key"]:
            raise ValueError("holdings/orders changed after the frozen preparation")
        sized_path = sizing(weights_path, current, day)
        sized = artifact_json(sized_path, "sizing.json")["accounts"][account_id]
        remember_contracts(sized["targets"])
        if day in auto.get("preview_only_dates", []):
            status("preview_day_no_submission", sizing_directory=str(sized_path),
                   blockers=sized["blockers"], reason="EXPLICIT_PREVIEW_ONLY_DATE")
            return status_path
        if not sized["tradable"]:
            status("blocked", reason="SIZING_NOT_TRADABLE", blockers=sized["blockers"], sizing_directory=str(sized_path))
            return status_path
        if policy.get("execute_enabled") is not True or auto.get("contest_execution_verified") is not True:
            status("prepared_execution_disabled", sizing_directory=str(sized_path))
            return status_path
        execution_policy = {**policy, "allow_mark_to_market": True, "slice_orders": True,
            "expected_signal_date": signal_date,
            "execution_window": {key: schedule[key].isoformat() for key in ("start_at", "submit_until", "completion_by")},
            "poll_interval_seconds": auto.get("poll_interval_seconds", 2),
            "max_attempts_per_leg": auto.get("max_attempts_per_leg", 3)}
        status("waiting_for_execution", sizing_directory=str(sized_path))
        # One SDK process survives the whole window, reusing its HTTP connection.
        result = panda_call({"action": "session", "sized": sized, "policy": execution_policy,
            "account_id": account_id, "trade_date": day, "signal_id": sized_path.name, "signal_date": signal_date,
            "prepared_positions_key": prepared["prepared_positions_key"], "mode": "simulation",
            "previous_plan": previous,
            "config_path": str(config_path), "config_sha256": original_config_hash,
            "journal_root": str(root / "execution_journal")}, interpreter=interpreter,
            timeout=max(1, (schedule["completion_by"] - now()).total_seconds()) + 120)
        output = write_artifact(root / "sessions", {"stage": "session", "result": result},
                               {"result.json": result, "remaining.csv": result["remaining"]})
        status(result["status"], result_directory=str(output), reason=result.get("reason"))
        return output
    except Exception as exc:
        status("blocked", error_type=type(exc).__name__,
               reason=str(exc) if isinstance(exc, (ValueError, TimeoutError)) else "inspect launcher error log")
        raise
    finally:
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG_PATH)
    commands = parser.add_subparsers(dest="stage")
    weights = commands.add_parser("weights", help="generate close weights, no sizing or trading")
    weights.add_argument("--as-of")
    weights.add_argument("--holdings", help="dated reconciled signed root holdings per strategy instance (JSON)")
    weights.add_argument("--full-panel", action="store_true", help="independent full-history parity audit")
    size = commands.add_parser("size", help="convert one explicit weights artifact")
    size.add_argument("--weights", required=True)
    size.add_argument("--specs", help="verified per-contract JSON; absent means indicative reference only")
    size.add_argument("--specs-as-of", help="execution date of specs, separate from signal date")
    automatic = commands.add_parser("auto", help="one dated preparation/execution day; separate research and send windows")
    automatic.add_argument("--account", required=True)
    automatic.add_argument("--trade-date")
    automatic.add_argument("--prepare-only", action="store_true", help="prepare once and exit; never send orders")
    plan = commands.add_parser("plan", help="reconcile real positions into a reviewable order CSV")
    plan.add_argument("--sizing", required=True)
    plan.add_argument("--snapshot", required=True)
    plan.add_argument("--account", required=True)
    plan.add_argument("--previous-plan", help="retain managed scope when switching version or strategy")
    paper = commands.add_parser("paper", help="local ideal quantity simulation only")
    paper.add_argument("--plan", required=True)
    paper.add_argument("--snapshot", required=True)
    snap = commands.add_parser("panda-snapshot", help="only read the OAuth-bound account")
    snap.add_argument("--account", required=True)
    prep = commands.add_parser("panda-prepare", help="freeze ONE exact order for user review; never submit")
    prep.add_argument("--plan", required=True)
    prep.add_argument("--order-index", type=int, required=True)
    submit = commands.add_parser("panda-execute", help="requires an explicitly enabled account and exact confirmation hash")
    submit.add_argument("--plan", required=True)
    submit.add_argument("--prepared", required=True)
    submit.add_argument("--confirm", required=True)
    operation = commands.add_parser("panda-operation", help="query original operation once, without resubmission")
    operation.add_argument("--operation-id", required=True)
    reconcile = commands.add_parser("panda-reconcile", help="read order/operation evidence and close a local attempt")
    reconcile.add_argument("--account", required=True)
    reconcile.add_argument("--client-order-id", required=True)
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    stage = args.stage or "weights"
    root = resolve(settings["output_root"])
    if stage == "auto":
        output = run_automatic_day(settings, config_path=args.config, account_id=args.account,
                                   trade_date=args.trade_date, prepare_only=args.prepare_only)
    elif stage == "weights":
        from trading.weights import generate_weights
        gate = yaml.safe_load(resolve("config/target_publication.yaml").read_text(encoding="utf-8"))
        output = generate_weights(settings["strategy_ids"], as_of=getattr(args, "as_of", None),
            output_root=root / "weights", catalog_path=settings["catalog"], mode=settings["mode"], gate=gate,
            full_panel=getattr(args, "full_panel", False),
            holdings=json.loads(resolve(args.holdings).read_text(encoding="utf-8"))
                if getattr(args, "holdings", None) else None)
    elif stage == "size":
        specs = json.loads(resolve(args.specs).read_text(encoding="utf-8")) if args.specs else None
        output = size_artifact(resolve(args.weights), settings, specs, specs_as_of=args.specs_as_of)
    elif stage == "plan":
        source = resolve(args.sizing)
        manifest = read_artifact(source)
        sized = artifact_json(source, "sizing.json")["accounts"][args.account]
        snapshot_path = resolve(args.snapshot)
        snapshot = (artifact_json(snapshot_path, "snapshot.json") if snapshot_path.is_dir()
                    else json.loads(snapshot_path.read_text(encoding="utf-8")))
        policy = settings["execution"][args.account]
        if manifest["identity"].get("mode") != "production" and policy["channel"] == "panda":
            if policy.get("allow_simulation_strategy") is not True:
                raise ValueError("Panda contest use of a simulation strategy must be explicitly enabled")
        previous = artifact_json(resolve(args.previous_plan), "plan.json") if args.previous_plan else None
        value = build_plan(sized, snapshot, account_id=args.account, policy=policy,
            signal_id=source.name, signal_date=manifest["identity"]["data_date"], previous_plan=previous)
        output = write_artifact(root / "plans", {"stage": "plan", "plan": value},
            {"orders.csv": [{"plan_id": value["plan_id"], "status": ("BLOCKED" if not value["ready"] else
                                "WAIT_CLOSE" if row["depends_on"] else "READY"),
                             "signal_date": value["signal_date"], "risk_mode": value["risk_mode"], **row}
                            for row in value["orders"]], "plan.json": value, "snapshot.json": snapshot})
    elif stage == "paper":
        value = artifact_json(resolve(args.plan), "plan.json")
        identity = {"stage": "paper", "plan_id": value["plan_id"]}
        existing = root / "paper" / digest(identity)
        if existing.exists():
            read_artifact(existing)
            print(str(existing))
            return existing
        snapshot = json.loads(resolve(args.snapshot).read_text(encoding="utf-8"))
        result = paper_execute(value, snapshot)
        output = write_artifact(root / "paper", identity,
                                {"result.json": result, "snapshot.json": result["snapshot"]})
    else:
        if stage == "panda-snapshot":
            request = {"action": "snapshot", "policy": settings["execution"][args.account]}
            filename = "snapshot.json"
        elif stage == "panda-operation":
            request = {"action": "operation", "operation_id": args.operation_id}
            filename = "operation.json"
        elif stage == "panda-reconcile":
            request = {"action": "reconcile", "identity": settings["execution"][args.account]["expected_identity"],
                       "client_order_id": args.client_order_id, "journal_root": str(root / "execution_journal")}
            filename = "reconciliation.json"
        else:
            value = artifact_json(resolve(args.plan), "plan.json")
            request = {"action": stage.removeprefix("panda-"), "plan": value}
            filename = "prepared.json" if stage == "panda-prepare" else "receipt.json"
            if stage == "panda-prepare":
                request["order_index"] = args.order_index
            else:
                current_policy = settings["execution"][value["account_id"]]
                if current_policy.get("execute_enabled") is not True or current_policy != value["policy"]:
                    raise ValueError("execution policy changed or disabled; generate a new plan")
                request.update(prepared=artifact_json(resolve(args.prepared), "prepared.json"),
                               confirmation=args.confirm, journal_root=str(root / "execution_journal"))
        result = panda_call(request)
        output = write_artifact(root / stage, {"stage": stage, "result": result}, {filename: result})
    print(str(output))
    return output


if __name__ == "__main__":
    main()
