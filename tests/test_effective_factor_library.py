from __future__ import annotations

import csv
import hashlib
import json
from types import SimpleNamespace

from research.effective_factor_library import (
    admit_validation_run,
    effective_factor_names,
    load_library,
    validate_effective_factor_periods,
)
from pipeline.runner import PipelineRunner


def test_validation_run_admission_creates_structured_library(tmp_path):
    run = tmp_path / "run-1"
    run.mkdir()
    row = {
        "factor": "intraday_probe",
        "family": "microstructure",
        "registered_horizons": "5|10|20",
        "selected_period": "10",
        "selected_variant": "raw",
        "ic": "0.03",
        "ols_hac_t": "2.5",
        "local_q_value": "0.04",
        "final_pass": "True",
    }
    with (run / "passed_factors.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    with (run / "factor_validation_full.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    (run / "artifacts").mkdir()
    (run / "artifacts" / "manifest.json").write_text("{}", encoding="utf-8")
    (run / "validation_summary.json").write_text(
        json.dumps({
            "schema_version": 2,
            "workflow": "factor_admission",
            "admission_policy": "hierarchical_fdr_post_gates",
            "scope": "explicit_batch",
            "final_pass_count": 1,
            "factor_count": 1,
            "research_cutoff": "2026-05-15",
            "research_window": ["2025-01-01", "2026-05-15"],
            "horizon_mode": "registered_contract",
        }),
        encoding="utf-8",
    )
    summary = run / "validation_summary.json"
    passed = run / "passed_factors.csv"
    full = run / "factor_validation_full.csv"
    manifest = run / "artifacts" / "manifest.json"
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    (run / "run_contract.json").write_text(
        json.dumps({
            "schema_version": 2,
            "run_id": run.name,
            "workflow": "factor-admission-validation",
            "admission_eligible": True,
            "scope": "explicit_batch",
            "factors": ["intraday_probe"],
            "validation_summary_sha256": digest(summary),
            "passed_results_sha256": digest(passed),
            "files": {
                "factor_validation_full.csv": {"sha256": digest(full)},
                "passed_factors.csv": {"sha256": digest(passed)},
                "validation_summary.json": {"sha256": digest(summary)},
                "artifacts/manifest.json": {"sha256": digest(manifest)},
            },
        }),
        encoding="utf-8",
    )
    library = tmp_path / "factor_library" / "library.json"

    payload = admit_validation_run(
        run, library, admitted_at="2026-05-15"
    )

    assert payload["factors"][0]["factor"] == "intraday_probe"
    assert payload["factors"][0]["selected_period"] == 10
    assert payload["factors"][0]["approved_periods"] == [10]
    assert payload["factors"][0]["family"] == "microstructure"
    assert payload["factors"][0]["direction"] == 1
    assert payload["factors"][0]["research_cutoff"] == "2026-05-15"
    assert effective_factor_names(library) == ["intraday_probe"]
    assert load_library(library)["source_run"] == "run-1"
    assert library.with_name("current.csv").is_file()

    validate_effective_factor_periods(library, {10: ["intraday_probe"]})


def test_period_validation_rejects_unapproved_and_unknown_factors(tmp_path):
    library = tmp_path / "library.json"
    library.write_text(json.dumps({
        "schema_version": 2,
        "factors": [{
            "factor": "approved_factor",
            "status": "effective",
            "selected_period": 5,
            "approved_periods": [5, 10],
        }],
    }), encoding="utf-8")

    validate_effective_factor_periods(library, {5: ["approved_factor"]})
    validate_effective_factor_periods(library, {10: ["approved_factor"]})
    try:
        validate_effective_factor_periods(
            library, {20: ["approved_factor"], 5: ["unknown_factor"]}
        )
    except ValueError as exc:
        message = str(exc)
        assert "period 20 not approved" in message
        assert "unknown_factor: not in effective library" in message
    else:
        raise AssertionError("invalid effective-factor assignments were accepted")


def test_admission_rejects_single_split_observation_run(tmp_path):
    run = tmp_path / "observation"
    run.mkdir()
    passed = run / "passed_factors.csv"
    passed.write_text("factor,final_pass\nprobe,True\n", encoding="utf-8")
    summary = run / "validation_summary.json"
    summary.write_text(json.dumps({
        "workflow": "single_split_observation",
        "final_pass_count": 1,
    }), encoding="utf-8")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    (run / "run_contract.json").write_text(json.dumps({
        "run_id": run.name,
        "workflow": "factor-validation-observation",
        "admission_eligible": False,
        "validation_summary_sha256": digest(summary),
        "passed_results_sha256": digest(passed),
    }), encoding="utf-8")

    try:
        admit_validation_run(
            run, tmp_path / "library.json", admitted_at="2026-09-06"
        )
    except ValueError as exc:
        assert "not eligible" in str(exc)
    else:
        raise AssertionError("observation run was admitted as effective")


def test_pipeline_period_gate_is_explicit_and_reuses_library_validator(tmp_path):
    library = tmp_path / "library.json"
    library.write_text(json.dumps({
        "schema_version": 2,
        "factors": [{
            "factor": "approved_factor",
            "status": "effective",
            "selected_period": 5,
            "approved_periods": [5],
        }],
    }), encoding="utf-8")
    runner = PipelineRunner.__new__(PipelineRunner)
    runner.config = SimpleNamespace(factor_library=SimpleNamespace(
        path=str(library), enforce_portfolio_periods=True
    ))

    runner._validate_effective_factor_periods({5: ["approved_factor"]})
    try:
        runner._validate_effective_factor_periods({10: ["approved_factor"]})
    except ValueError as exc:
        assert "period 10 not approved" in str(exc)
    else:
        raise AssertionError("pipeline accepted an unapproved factor period")


def test_admission_merges_new_passes_without_retiring_existing_factors(tmp_path):
    library = tmp_path / "factor_library" / "library.json"
    library.parent.mkdir()
    existing = [{
        "factor": f"old_{index:03d}",
        "status": "effective",
        "selected_period": 5,
        "approved_periods": [5],
    } for index in range(75)]
    library.write_text(json.dumps({
        "schema_version": 2, "factors": existing
    }), encoding="utf-8")

    run = tmp_path / "run-688"
    run.mkdir()
    full = run / "factor_validation_full.csv"
    with full.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["factor", "final_pass"])
        writer.writeheader()
        writer.writerows({
            "factor": (
                f"new_{index:03d}" if index < 25 else f"factor_{index:03d}"
            ),
            "final_pass": index < 25,
        } for index in range(688))
    rows = [{
        "factor": f"new_{index:03d}",
        "family": "test",
        "registered_horizons": "5|10|20",
        "selected_period": "10",
        "selected_variant": "raw",
        "ic": "0.03",
        "ols_hac_t": "2.5",
        "local_q_value": "0.04",
        "final_pass": "True",
    } for index in range(25)]
    with (run / "passed_factors.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = run / "validation_summary.json"
    summary.write_text(json.dumps({
        "schema_version": 2,
        "workflow": "factor_admission",
        "admission_policy": "hierarchical_fdr_post_gates",
        "scope": "explicit_batch",
        "final_pass_count": 25,
        "passed_factors": [row["factor"] for row in rows],
        "research_cutoff": "2026-05-15",
        "research_window": ["2025-01-01", "2026-05-15"],
        "horizon_mode": "registered_contract",
        "factor_count": 688,
    }), encoding="utf-8")
    passed = run / "passed_factors.csv"
    (run / "artifacts").mkdir()
    manifest = run / "artifacts" / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    submitted = [
        f"new_{index:03d}" if index < 25 else f"factor_{index:03d}"
        for index in range(688)
    ]
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    (run / "run_contract.json").write_text(json.dumps({
        "schema_version": 2,
        "run_id": run.name,
        "workflow": "factor-admission-validation",
        "admission_eligible": True,
        "scope": "explicit_batch",
        "factors": submitted,
        "validation_summary_sha256": digest(summary),
        "passed_results_sha256": digest(passed),
        "files": {
            "factor_validation_full.csv": {"sha256": digest(full)},
            "passed_factors.csv": {"sha256": digest(passed)},
            "validation_summary.json": {"sha256": digest(summary)},
            "artifacts/manifest.json": {"sha256": digest(manifest)},
        },
    }), encoding="utf-8")

    payload = admit_validation_run(run, library, admitted_at="2026-08-24")

    assert len(payload["factors"]) == 100
    assert len([row for row in payload["factors"] if row["factor"].startswith("new_")]) == 25
    assert len([row for row in payload["factors"] if row["factor"].startswith("old_")]) == 75


def test_full_pool_admission_replaces_instead_of_merging_old_effective_rows(tmp_path):
    library = tmp_path / "factor_library" / "library.json"
    library.parent.mkdir()
    library.write_text(json.dumps({
        "schema_version": 2,
        "factors": [{
            "factor": "stale_factor",
            "status": "effective",
            "selected_period": 5,
            "approved_periods": [5],
        }],
    }), encoding="utf-8")

    run = tmp_path / "full-rebuild"
    run.mkdir()
    row = {
        "factor": "fresh_factor",
        "family": "intraday",
        "registered_horizons": "3|5|10",
        "selected_period": "3",
        "selected_variant": "raw",
        "ic": "0.03",
        "ols_hac_t": "2.5",
        "local_q_value": "0.04",
        "final_pass": "True",
    }
    passed = run / "passed_factors.csv"
    with passed.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    full = run / "factor_validation_full.csv"
    with full.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    (run / "artifacts").mkdir()
    manifest = run / "artifacts" / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    summary = run / "validation_summary.json"
    summary.write_text(json.dumps({
        "schema_version": 2,
        "workflow": "factor_admission",
        "admission_policy": "hierarchical_fdr_post_gates",
        "scope": "all_registered_intraday",
        "final_pass_count": 1,
        "factor_count": 1,
        "research_cutoff": "2026-05-15",
        "research_window": ["2025-01-01", "2026-05-15"],
        "horizon_mode": "registered_contract",
    }), encoding="utf-8")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    (run / "run_contract.json").write_text(json.dumps({
        "schema_version": 2,
        "run_id": run.name,
        "workflow": "factor-admission-validation",
        "admission_eligible": True,
        "scope": "all_registered_intraday",
        "factors": ["fresh_factor"],
        "validation_summary_sha256": digest(summary),
        "passed_results_sha256": digest(passed),
        "files": {
            "factor_validation_full.csv": {"sha256": digest(full)},
            "passed_factors.csv": {"sha256": digest(passed)},
            "validation_summary.json": {"sha256": digest(summary)},
            "artifacts/manifest.json": {"sha256": digest(manifest)},
        },
    }), encoding="utf-8")

    payload = admit_validation_run(run, library, admitted_at="2026-09-06")

    assert [record["factor"] for record in payload["factors"]] == ["fresh_factor"]
    assert payload["factors"][0]["frequency"] == "daily"
