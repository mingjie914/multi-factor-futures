from inspect import signature
from pathlib import Path

import pytest

import run_factor_workflow as ide
from workflows.factor_validation import run_default_factor_validation


def test_ide_workflow_names_have_one_default_and_no_one_off_horizon_branches():
    assert {item.value for item in ide.FactorWorkflow} == {
        "validate_factor_batch",
        "validate_all_intraday",
        "observe_common_horizon",
        "admit_completed_run",
        "select_effective_subsets",
    }
    assert ide.WORKFLOW is ide.FactorWorkflow.VALIDATE_FACTOR_BATCH
    assert "admit" not in signature(run_default_factor_validation).parameters


def test_ide_batch_validation_fails_closed_without_factor_names(monkeypatch):
    monkeypatch.setattr(ide, "WORKFLOW", ide.FactorWorkflow.VALIDATE_FACTOR_BATCH)
    monkeypatch.setattr(ide, "FACTOR_NAMES", ())
    with pytest.raises(ValueError, match="FACTOR_NAMES"):
        ide.main()


def test_ide_batch_routes_to_standard_admission_validation(monkeypatch):
    called = {}

    def fake_validation(**kwargs):
        called.update(kwargs)
        return Path("unused")

    monkeypatch.setattr(ide, "run_default_factor_validation", fake_validation)
    monkeypatch.setattr(ide, "VALIDATION_RUN_ID", "ide_test")
    monkeypatch.setattr(ide, "FACTOR_NAMES", ("factor_b", "factor_a"))
    monkeypatch.setattr(ide, "WORKFLOW", ide.FactorWorkflow.VALIDATE_FACTOR_BATCH)
    ide.main()

    assert called == {
        "run_id": "ide_test",
        "config_path": "config/default.yaml",
        "factor_names": ("factor_b", "factor_a"),
    }


def test_ide_full_pool_route_is_explicit(monkeypatch):
    called = {}

    def fake_validation(**kwargs):
        called.update(kwargs)
        return Path("unused")

    monkeypatch.setattr(ide, "run_default_factor_validation", fake_validation)
    monkeypatch.setattr(ide, "VALIDATION_RUN_ID", "full_test")
    monkeypatch.setattr(ide, "WORKFLOW", ide.FactorWorkflow.VALIDATE_ALL_INTRADAY)
    ide.main()

    assert called == {
        "run_id": "full_test",
        "config_path": "config/default.yaml",
        "all_registered": True,
    }


def test_ide_common_horizon_is_one_explicit_observation_route(monkeypatch):
    called = {}

    def fake_validation(**kwargs):
        called.update(kwargs)
        return Path("unused")

    monkeypatch.setattr(ide, "run_common_horizon_factor_validation", fake_validation)
    monkeypatch.setattr(ide, "VALIDATION_RUN_ID", "common_probe")
    monkeypatch.setattr(ide, "COMMON_HORIZON", 5)
    monkeypatch.setattr(ide, "WORKFLOW", ide.FactorWorkflow.OBSERVE_COMMON_HORIZON)
    ide.main()

    assert called == {
        "run_id": "common_probe",
        "common_horizon": 5,
        "config_path": "config/default.yaml",
    }


def test_ide_admission_requires_explicit_evidence(monkeypatch):
    monkeypatch.setattr(ide, "WORKFLOW", ide.FactorWorkflow.ADMIT_COMPLETED_RUN)
    monkeypatch.setattr(ide, "ADMISSION_RUN_DIR", None)
    monkeypatch.setattr(ide, "ADMITTED_AT", None)

    with pytest.raises(ValueError, match="ADMISSION_RUN_DIR and ADMITTED_AT"):
        ide.main()


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


def test_ide_selection_routes_to_effective_library_workflow(monkeypatch):
    called = {}

    def fake_selection(**kwargs):
        called.update(kwargs)
        return Path("runs/factor_selection/probe")

    monkeypatch.setattr(ide, "WORKFLOW", ide.FactorWorkflow.SELECT_EFFECTIVE_SUBSETS)
    monkeypatch.setattr(ide, "SELECTION_RUN_ID", "selection_probe")
    monkeypatch.setattr(ide, "run_effective_factor_selection", fake_selection)

    ide.main()

    assert called == {
        "run_id": "selection_probe",
        "config_path": "config/default.yaml",
    }
