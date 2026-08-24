"""Small, version-controlled effective-factor library backed by run evidence."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_library(path: str | Path) -> dict[str, Any]:
    library_path = Path(path).expanduser().resolve()
    if not library_path.is_file():
        return {"schema_version": SCHEMA_VERSION, "factors": []}
    payload = json.loads(library_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported effective factor library schema")
    factors = payload.get("factors")
    if not isinstance(factors, list):
        raise ValueError("effective factor library factors must be a list")
    names = [str(row.get("factor", "")) for row in factors]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("effective factor library contains empty or duplicate names")
    return payload


def effective_factor_names(path: str | Path) -> list[str]:
    return sorted(
        row["factor"]
        for row in load_library(path)["factors"]
        if row.get("status") == "effective"
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def export_current_csv(library_path: str | Path) -> Path:
    path = Path(library_path).expanduser().resolve()
    payload = load_library(path)
    output = path.with_name("current.csv")
    fields = [
        "factor", "status", "frequency", "registered_horizons",
        "selected_period", "approved_periods", "direction", "admitted_at", "source_run",
        "research_cutoff",
        "is_ic", "is_t", "is_q", "oos_ic", "oos_ic_hac_t",
        "oos_ols_hac_t", "oos_days", "evidence_sha256",
    ]
    temporary = output.with_name(f"{output.name}.{os.getpid()}.tmp")
    output.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(payload["factors"], key=lambda item: item["factor"]):
            writer.writerow({key: row.get(key, "") for key in fields})
    os.replace(temporary, output)
    return output


def admit_validation_run(
    run_dir: str | Path,
    library_path: str | Path,
    *,
    admitted_at: str,
) -> dict[str, Any]:
    """Merge one completed factor-validation run into the current library."""
    run = Path(run_dir).expanduser().resolve()
    passed_path = run / "passed_factors.csv"
    full_path = run / "factor_validation_full.csv"
    summary_path = run / "validation_summary.json"
    contract_path = run / "run_contract.json"
    if not all(path.is_file() for path in (passed_path, summary_path, contract_path)):
        raise FileNotFoundError(
            "validation run is missing passed_factors.csv, summary, or contract"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("run_id") != run.name:
        raise ValueError("validation contract run_id does not match directory")
    contract_files = contract.get("files", {})
    summary_hash = contract.get("validation_summary_sha256") or (
        contract_files.get("validation_summary.json", {}).get("sha256")
    )
    passed_hash = contract.get("passed_results_sha256") or (
        contract_files.get("passed_factors.csv", {}).get("sha256")
    )
    if summary_hash != _sha256(summary_path):
        raise ValueError("validation summary hash does not match run contract")
    if passed_hash != _sha256(passed_path):
        raise ValueError("passed factor hash does not match run contract")
    for filename, metadata in contract_files.items():
        artifact = run / filename
        expected_hash = metadata.get("sha256") if isinstance(metadata, dict) else None
        if not artifact.is_file() or not expected_hash:
            raise ValueError(f"validation contract artifact is missing: {filename}")
        if _sha256(artifact) != expected_hash:
            raise ValueError(f"validation artifact hash does not match: {filename}")
    if len(summary.get("is", [])) != 3 or len(summary.get("oos", [])) != 3:
        raise ValueError("validation summary is missing the fixed IS/OOS contract")
    with passed_path.open(encoding="utf-8-sig", newline="") as handle:
        passed = list(csv.DictReader(handle))
    if len(passed) != int(summary.get("final_pass_count", -1)):
        raise ValueError("validation summary and passed factor rows disagree")
    if full_path.is_file():
        with full_path.open(encoding="utf-8-sig", newline="") as handle:
            full_count = sum(1 for _ in csv.DictReader(handle))
        if full_count != int(summary.get("factor_count", -1)):
            raise ValueError("validation summary and full factor rows disagree")
    if any(str(row.get("final_pass", "")).lower() != "true" for row in passed):
        raise ValueError("passed factor file contains a non-passing row")
    if len({row.get("factor") for row in passed}) != len(passed):
        raise ValueError("passed factor file contains duplicate names")
    summary_names = summary.get("passed_factors")
    if summary_names is not None and set(summary_names) != {
        row["factor"] for row in passed
    }:
        raise ValueError("validation summary and passed factor names disagree")

    path = Path(library_path).expanduser().resolve()
    current = load_library(path)
    by_name = {row["factor"]: row for row in current["factors"]}
    source_run = run.name
    evidence_hash = _sha256(passed_path)
    try:
        evidence_file = passed_path.relative_to(path.parent.parent).as_posix()
    except ValueError:
        evidence_file = str(passed_path)
    for row in passed:
        factor = row["factor"]
        is_ic = float(row["is_official_best_ic"])
        selected_period = int(float(row["oos_period"]))
        registered_periods = {
            int(value) for value in str(row["registered_horizons"]).split("|")
            if value
        }
        approved_text = str(row.get("approved_periods", "") or "")
        approved_periods = sorted({
            int(value) for value in approved_text.split("|") if value
        }) or [selected_period]
        if (
            selected_period not in approved_periods
            or not set(approved_periods).issubset(registered_periods)
        ):
            raise ValueError(f"invalid approved periods for factor {factor!r}")
        by_name[factor] = {
            "factor": factor,
            "status": "effective",
            "frequency": "daily_intraday",
            "registered_horizons": row["registered_horizons"],
            "selected_period": selected_period,
            # The standard run emits no multi-period field and therefore
            # approves only its IS-selected/OOS-tested horizon.
            "approved_periods": approved_periods,
            "direction": 1 if is_ic >= 0.0 else -1,
            "admitted_at": admitted_at,
            "source_run": source_run,
            "research_cutoff": summary.get("research_cutoff"),
            "is_ic": is_ic,
            "is_t": float(row["is_official_best_t"]),
            "is_q": float(row["is_official_best_q"]),
            "oos_ic": float(row["oos_ic"]),
            "oos_ic_hac_t": float(row["oos_ic_hac_t"]),
            "oos_ols_hac_t": float(row["oos_ols_hac_t"]),
            "oos_days": int(float(row["oos_days"])),
            "evidence_file": evidence_file,
            "evidence_sha256": evidence_hash,
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": admitted_at,
        "source_run": source_run,
        "factors": [by_name[name] for name in sorted(by_name)],
    }
    _write_json_atomic(path, payload)
    export_current_csv(path)
    return payload


def validate_effective_factor_periods(
    library_path: str | Path,
    assignments: dict[int, list[str]],
) -> None:
    """Fail when a portfolio uses a non-effective or unapproved factor horizon."""
    factors = {
        row["factor"]: row
        for row in load_library(library_path)["factors"]
        if row.get("status") == "effective"
    }
    errors: list[str] = []
    for period, names in assignments.items():
        for name in names:
            record = factors.get(name)
            if record is None:
                errors.append(f"{name}: not in effective library")
                continue
            approved = record.get("approved_periods")
            if approved is None:  # schema-v1 libraries written before this field
                approved = [record.get("selected_period")]
            if int(period) not in {int(value) for value in approved}:
                errors.append(
                    f"{name}: period {period} not approved; approved={approved}"
                )
    if errors:
        raise ValueError("effective factor period validation failed: " + "; ".join(errors))
