"""Independent IDE/CLI entry: weights -> size -> plan -> explicit execution.

The weights stage generates simulation close weights only; buffered strategies
require --holdings. Later stages require an explicit immutable input directory.
See docs/交易模块设计与验收.md.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import yaml

from trading.artifacts import digest, file_hash, read_artifact, read_csv, write_artifact
from trading.execution import build_plan, paper_execute
from trading.weights import generate_weights, resolve


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


def size_artifact(weights_directory, settings, specs=None):
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
    result = size_targets(groups, specs, settings["accounts"], settings["routes"], settings["rounding"])
    identity = {"stage": "size", "weights_id": Path(weights_directory).name,
                "implementation_sha256": file_hash(resolve("trading/sizing.py")),
                "entrypoint_sha256": file_hash(__file__),
                "weights_manifest": manifest, "data_date": manifest["identity"]["data_date"],
                "mode": manifest["identity"]["mode"], "specs": specs,
                "accounts": settings["accounts"], "routes": settings["routes"],
                "rounding": settings["rounding"]}
    targets = [{"account_id": account, "status": "READY" if value["tradable"] else "BLOCKED", **row}
               for account, value in result["accounts"].items() for row in value["targets"]]
    attribution = [{"account_id": account, **row} for account, value in result["accounts"].items() for row in value["attribution"]]
    return write_artifact(resolve(settings["output_root"]) / "sizing", identity,
                          {"targets.csv": targets, "attribution.csv": attribution, "sizing.json": result,
                           "capital_requirements.csv": [{"strategy_id": sid, **row}
                               for sid, req in result["capital_requirements"].items() for row in req["rows"]]})


def panda_call(request):
    interpreter = os.environ.get("MF_PANDA_PYTHON", "")
    if not interpreter or not Path(interpreter).is_file():
        raise ValueError("set MF_PANDA_PYTHON to the independently installed SDK Python executable")
    result = subprocess.run([interpreter, "-X", "utf8", "-m", "trading.panda"],
                            input=json.dumps(request), text=True, encoding="utf-8", capture_output=True,
                            cwd=Path(__file__).resolve().parent, timeout=120)
    if result.returncode:
        # Exceptions may include provider bodies. Keep detailed identity data local.
        raise RuntimeError("Panda worker failed; verify the SDK, account and frozen plan before retrying")
    return json.loads(result.stdout)


def artifact_json(path, filename):
    manifest = read_artifact(path)
    if filename not in manifest["files"]:
        raise ValueError(f"artifact missing {filename}")
    return json.loads((Path(path) / filename).read_text(encoding="utf-8"))


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
    if stage == "weights":
        gate = yaml.safe_load(resolve("config/target_publication.yaml").read_text(encoding="utf-8"))
        output = generate_weights(settings["strategy_ids"], as_of=getattr(args, "as_of", None),
            output_root=root / "weights", catalog_path=settings["catalog"], mode=settings["mode"], gate=gate,
            full_panel=getattr(args, "full_panel", False),
            holdings=json.loads(resolve(args.holdings).read_text(encoding="utf-8"))
                if getattr(args, "holdings", None) else None)
    elif stage == "size":
        specs = json.loads(resolve(args.specs).read_text(encoding="utf-8")) if args.specs else None
        output = size_artifact(resolve(args.weights), settings, specs)
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
