from pathlib import Path
from inspect import signature

import run_factor_workflow as ide
from workflows.factor_validation import run_default_factor_validation


def test_ide_workflow_has_no_full_history_branch():
    assert {item.value for item in ide.FactorWorkflow} == {
        "validate_all_intraday",
        "admit_completed_run",
    }
    assert ide.WORKFLOW is ide.FactorWorkflow.VALIDATE_ALL_INTRADAY
    assert "admit" not in signature(run_default_factor_validation).parameters


def test_ide_default_routes_to_standard_validation(monkeypatch):
    called = {}

    def fake_validation(**kwargs):
        called.update(kwargs)
        return Path("unused")

    monkeypatch.setattr(ide, "run_default_factor_validation", fake_validation)
    monkeypatch.setattr(ide, "VALIDATION_RUN_ID", "ide_test")
    monkeypatch.setattr(ide, "WORKFLOW", ide.FactorWorkflow.VALIDATE_ALL_INTRADAY)
    ide.main()

    assert called == {
        "run_id": "ide_test",
        "config_path": "config/default.yaml",
    }


def test_ide_admission_requires_explicit_evidence(monkeypatch):
    monkeypatch.setattr(ide, "WORKFLOW", ide.FactorWorkflow.ADMIT_COMPLETED_RUN)
    monkeypatch.setattr(ide, "ADMISSION_RUN_DIR", None)
    monkeypatch.setattr(ide, "ADMITTED_AT", None)

    try:
        ide.main()
    except ValueError as exc:
        assert "ADMISSION_RUN_DIR and ADMITTED_AT" in str(exc)
    else:
        raise AssertionError("admission must fail closed without explicit evidence")


def test_ide_admission_routes_only_to_library_update(monkeypatch, tmp_path):
    called = {}

    def fake_admission(run_dir, library_path, *, admitted_at):
        called.update({
            "run_dir": run_dir,
            "library_path": library_path,
            "admitted_at": admitted_at,
        })
        return {"factors": [{"factor": "probe"}]}

    monkeypatch.setattr(ide, "WORKFLOW", ide.FactorWorkflow.ADMIT_COMPLETED_RUN)
    monkeypatch.setattr(ide, "ADMISSION_RUN_DIR", "runs/factor_validation/probe")
    monkeypatch.setattr(ide, "ADMITTED_AT", "2026-08-24")
    monkeypatch.setattr(ide, "_library_path", lambda _: tmp_path / "library.json")
    monkeypatch.setattr(ide, "admit_validation_run", fake_admission)

    ide.main()

    assert called == {
        "run_dir": "runs/factor_validation/probe",
        "library_path": tmp_path / "library.json",
        "admitted_at": "2026-08-24",
    }
