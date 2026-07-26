"""Fail-closed end-of-day decision gate.

The close workflow is deliberately separate from research and backtesting.
It never promotes a research candidate or an old report implicitly. Until a
reviewed deployment package is explicitly enabled, the only valid decision is
NO_TRADE.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APPROVED_STATUSES = {"approved_for_paper_trade", "approved_for_live"}


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _load_gate(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("trading gate config must contain a mapping")
    return raw


def build_close_decision(config_path: str | Path, as_of: str) -> dict[str, Any]:
    """Build an auditable close decision without assuming deployment approval."""
    decision_date = date.fromisoformat(as_of).isoformat()
    config_file = _resolve_project_path(str(config_path))
    gate = _load_gate(config_file)
    enabled = gate.get("enabled") is True
    approval_status = str(gate.get("approval_status", "disabled"))
    deployment_value = str(gate.get("deployment_package", "")).strip()
    deployment_path = (
        _resolve_project_path(deployment_value) if deployment_value else None
    )

    reason_code = "TRADING_DISABLED"
    if enabled and approval_status not in APPROVED_STATUSES:
        reason_code = "DEPLOYMENT_NOT_APPROVED"
    elif enabled and deployment_path is None:
        reason_code = "DEPLOYMENT_PACKAGE_MISSING"
    elif enabled and not deployment_path.exists():
        reason_code = "DEPLOYMENT_PACKAGE_NOT_FOUND"
    elif enabled:
        # The project has no approved deployment package yet. Refuse to infer
        # live weights from historical candidates or backtest configuration.
        reason_code = "DEPLOYMENT_EXECUTOR_NOT_RELEASED"

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "as_of": decision_date,
        "status": "NO_TRADE",
        "reason_code": reason_code,
        "approval_status": approval_status,
        "approved_study_id": gate.get("approved_study_id") or None,
        "protocol_sha256": gate.get("protocol_sha256") or None,
        "target_weights": {},
        "orders": [],
        "config": str(config_file),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed end-of-day trading decision gate."
    )
    parser.add_argument(
        "--config", default="config/trading.yaml",
        help="Trading approval gate (default: config/trading.yaml)",
    )
    parser.add_argument(
        "--as-of", default=date.today().isoformat(),
        help="Decision date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional JSON path; defaults to signals_output/close_decision_DATE.json",
    )
    args = parser.parse_args()

    decision = build_close_decision(args.config, args.as_of)
    output = (
        _resolve_project_path(args.output) if args.output
        else PROJECT_ROOT / "signals_output" / f"close_decision_{args.as_of}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**decision, "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
