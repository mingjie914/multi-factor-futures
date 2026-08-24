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
        "registered_horizons": "5|10|20",
        "oos_period": "10",
        "approved_periods": "5|10",
        "is_official_best_ic": "0.03",
        "is_official_best_t": "2.5",
        "is_official_best_q": "0.04",
        "oos_ic": "0.02",
        "oos_ic_hac_t": "1.5",
        "oos_ols_hac_t": "1.4",
        "oos_days": "32",
        "final_pass": "True",
    }
    with (run / "passed_factors.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    (run / "validation_summary.json").write_text(
        json.dumps({
            "final_pass_count": 1,
            "research_cutoff": "2026-05-15",
            "is": ["a", "b", 126],
            "oos": ["c", "d", 42],
        }),
        encoding="utf-8",
    )
    summary = run / "validation_summary.json"
    passed = run / "passed_factors.csv"
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    (run / "run_contract.json").write_text(
        json.dumps({
            "run_id": run.name,
            "validation_summary_sha256": digest(summary),
            "passed_results_sha256": digest(passed),
        }),
        encoding="utf-8",
    )
    library = tmp_path / "factor_library" / "library.json"

    payload = admit_validation_run(
        run, library, admitted_at="2026-05-15"
    )

    assert payload["factors"][0]["factor"] == "intraday_probe"
    assert payload["factors"][0]["selected_period"] == 10
    assert payload["factors"][0]["approved_periods"] == [5, 10]
    assert payload["factors"][0]["direction"] == 1
    assert payload["factors"][0]["research_cutoff"] == "2026-05-15"
    assert effective_factor_names(library) == ["intraday_probe"]
    assert load_library(library)["source_run"] == "run-1"
    assert library.with_name("current.csv").is_file()

    validate_effective_factor_periods(library, {10: ["intraday_probe"]})


def test_period_validation_rejects_unapproved_and_unknown_factors(tmp_path):
    library = tmp_path / "library.json"
    library.write_text(json.dumps({
        "schema_version": 1,
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


def test_pipeline_period_gate_is_explicit_and_reuses_library_validator(tmp_path):
    library = tmp_path / "library.json"
    library.write_text(json.dumps({
        "schema_version": 1,
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
        "schema_version": 1, "factors": existing
    }), encoding="utf-8")

    run = tmp_path / "run-688"
    run.mkdir()
    full = run / "factor_validation_full.csv"
    with full.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["factor", "final_pass"])
        writer.writeheader()
        writer.writerows({
            "factor": f"factor_{index:03d}",
            "final_pass": index < 25,
        } for index in range(688))
    rows = [{
        "factor": f"new_{index:03d}",
        "registered_horizons": "5|10|20",
        "oos_period": "10",
        "is_official_best_ic": "0.03",
        "is_official_best_t": "2.5",
        "is_official_best_q": "0.04",
        "oos_ic": "0.02",
        "oos_ic_hac_t": "1.5",
        "oos_ols_hac_t": "1.4",
        "oos_days": "32",
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
        "final_pass_count": 25,
        "passed_factors": [row["factor"] for row in rows],
        "research_cutoff": "2026-05-15",
        "is": ["a", "b", 126],
        "oos": ["c", "d", 42],
        "factor_count": 688,
    }), encoding="utf-8")
    passed = run / "passed_factors.csv"
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    (run / "run_contract.json").write_text(json.dumps({
        "run_id": run.name,
        "validation_summary_sha256": digest(summary),
        "passed_results_sha256": digest(passed),
        "files": {
            "factor_validation_full.csv": {"sha256": digest(full)},
            "passed_factors.csv": {"sha256": digest(passed)},
            "validation_summary.json": {"sha256": digest(summary)},
        },
    }), encoding="utf-8")

    payload = admit_validation_run(run, library, admitted_at="2026-08-24")

    assert len(payload["factors"]) == 100
    assert len([row for row in payload["factors"] if row["factor"].startswith("new_")]) == 25
    assert len([row for row in payload["factors"] if row["factor"].startswith("old_")]) == 75
