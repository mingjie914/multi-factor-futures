"""Version-controlled effective-factor library backed by formal run evidence."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from research.artifacts import sha256_file


SCHEMA_VERSION = 2


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
        "factor", "family", "status", "frequency", "registered_horizons",
        "selected_period", "approved_periods", "direction", "admitted_at", "source_run",
        "research_start", "research_cutoff", "ic", "t", "q", "evidence_sha256",
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
    """Apply one formal run: merge a batch or replace from a full-pool rebuild."""
    run = Path(run_dir).expanduser().resolve()
    passed_path = run / "passed_factors.csv"
    full_path = run / "factor_validation_full.csv"
    summary_path = run / "validation_summary.json"
    contract_path = run / "run_contract.json"
    if not all(path.is_file() for path in (passed_path, summary_path, contract_path)):
        raise FileNotFoundError(
            "validation run is missing passed results, summary, or contract"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("run_id") != run.name:
        raise ValueError("validation contract run_id does not match directory")
    if (
        int(contract.get("schema_version", 0)) != 2
        or int(summary.get("schema_version", 0)) != 2
        or contract.get("workflow") != "factor-admission-validation"
        or contract.get("admission_eligible") is not True
        or summary.get("workflow") != "factor_admission"
        or summary.get("admission_policy") != "hierarchical_fdr_post_gates"
        or summary.get("horizon_mode") != "registered_contract"
    ):
        raise ValueError("validation run is not eligible for effective-factor admission")
    manifest_path = run / "artifacts" / "manifest.json"
    if not full_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "formal validation run is missing full results or artifact manifest"
        )
    scope = str(summary.get("scope", ""))
    if scope not in {"explicit_batch", "all_registered_intraday"}:
        raise ValueError("validation summary has an invalid admission scope")
    if str(contract.get("scope", "")) != scope:
        raise ValueError("validation contract and summary admission scopes disagree")
    contract_files = contract.get("files", {})
    required_files = {
        "factor_validation_full.csv",
        "passed_factors.csv",
        "validation_summary.json",
        "artifacts/manifest.json",
    }
    if not required_files.issubset(contract_files):
        raise ValueError("validation contract does not bind every required artifact")
    summary_hash = contract.get("validation_summary_sha256") or (
        contract_files.get("validation_summary.json", {}).get("sha256")
    )
    passed_hash = contract.get("passed_results_sha256") or (
        contract_files.get("passed_factors.csv", {}).get("sha256")
    )
    if summary_hash != sha256_file(summary_path):
        raise ValueError("validation summary hash does not match run contract")
    if passed_hash != sha256_file(passed_path):
        raise ValueError("passed factor hash does not match run contract")
    for filename, metadata in contract_files.items():
        artifact = run / filename
        expected_hash = metadata.get("sha256") if isinstance(metadata, dict) else None
        if not artifact.is_file() or not expected_hash:
            raise ValueError(f"validation contract artifact is missing: {filename}")
        if sha256_file(artifact) != expected_hash:
            raise ValueError(f"validation artifact hash does not match: {filename}")
    research_window = summary.get("research_window", [])
    if len(research_window) != 2:
        raise ValueError("validation summary is missing the formal research window")
    with passed_path.open(encoding="utf-8-sig", newline="") as handle:
        passed = list(csv.DictReader(handle))
    if len(passed) != int(summary.get("final_pass_count", -1)):
        raise ValueError("validation summary and passed factor rows disagree")
    with full_path.open(encoding="utf-8-sig", newline="") as handle:
        full = list(csv.DictReader(handle))
    if len(full) != int(summary.get("factor_count", -1)):
        raise ValueError("validation summary and full factor rows disagree")
    full_names = [row.get("factor") for row in full]
    submitted_names = contract.get("factors")
    if (
        not isinstance(submitted_names, list)
        or len(set(full_names)) != len(full_names)
        or full_names != submitted_names
    ):
        raise ValueError("validation contract and submitted factor rows disagree")
    if any(str(row.get("final_pass", "")).lower() != "true" for row in passed):
        raise ValueError("passed factor file contains a non-passing row")
    if len({row.get("factor") for row in passed}) != len(passed):
        raise ValueError("passed factor file contains duplicate names")
    summary_names = summary.get("passed_factors")
    if summary_names is not None and set(summary_names) != {
        row["factor"] for row in passed
    }:
        raise ValueError("validation summary and passed factor names disagree")
    full_passed = {
        str(row["factor"])
        for row in full
        if str(row.get("final_pass", "")).lower() == "true"
    }
    if full_passed != {row["factor"] for row in passed}:
        raise ValueError("full validation decisions and passed factor rows disagree")

    path = Path(library_path).expanduser().resolve()
    current = load_library(path)
    by_name = (
        {}
        if scope == "all_registered_intraday"
        else {row["factor"]: row for row in current["factors"]}
    )
    source_run = run.name
    evidence_hash = sha256_file(passed_path)
    try:
        evidence_file = passed_path.relative_to(path.parent.parent).as_posix()
    except ValueError:
        evidence_file = str(passed_path)
    for row in passed:
        factor = row["factor"]
        ic = float(row["ic"])
        selected_period = int(float(row["selected_period"]))
        registered_periods = {
            int(value) for value in str(row["registered_horizons"]).split("|")
            if value
        }
        approved_periods = [selected_period]
        if (
            selected_period not in approved_periods
            or not set(approved_periods).issubset(registered_periods)
        ):
            raise ValueError(f"invalid approved periods for factor {factor!r}")
        by_name[factor] = {
            "factor": factor,
            "family": str(row.get("family", "") or ""),
            "status": "effective",
            "frequency": "daily",
            "registered_horizons": row["registered_horizons"],
            "selected_period": selected_period,
            "approved_periods": approved_periods,
            "direction": 1 if ic >= 0.0 else -1,
            "admitted_at": admitted_at,
            "source_run": source_run,
            "research_start": research_window[0],
            "research_cutoff": summary.get("research_cutoff"),
            "ic": ic,
            "t": float(row["ols_hac_t"]),
            "q": float(row["local_q_value"]),
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
