import json
from types import SimpleNamespace

import pandas as pd
import pytest

from workflows.factor_validation import (
    _admission_result_rows,
    _resolve_candidate_names,
    _validate_admission_screening_contract,
)


def test_candidate_scope_is_explicit_and_batch_order_is_stable(monkeypatch):
    registry = {
        "factor_b": SimpleNamespace(
            frequency="daily", __module__="factors.library.intraday"
        ),
        "factor_a": SimpleNamespace(
            frequency="daily", __module__="factors.library.intraday"
        ),
        "other": SimpleNamespace(
            frequency="daily", __module__="factors.library.other"
        ),
    }
    monkeypatch.setattr(
        "workflows.factor_validation.list_registered",
        lambda kind: {"factor": registry},
    )

    names, scope = _resolve_candidate_names(
        ("factor_b", "factor_a", "factor_b"),
        all_registered=False,
        module_prefix="factors.library.intraday",
    )
    assert names == ["factor_b", "factor_a"]
    assert scope == "explicit_batch"

    names, scope = _resolve_candidate_names(
        None,
        all_registered=True,
        module_prefix="factors.library.intraday",
    )
    assert names == ["factor_a", "factor_b"]
    assert scope == "all_registered_intraday"

    with pytest.raises(ValueError, match="exactly one"):
        _resolve_candidate_names(
            ("factor_a",),
            all_registered=True,
            module_prefix="factors.library.intraday",
        )
    with pytest.raises(ValueError, match="exactly one"):
        _resolve_candidate_names(
            None,
            all_registered=False,
            module_prefix="factors.library.intraday",
        )
    with pytest.raises(ValueError, match="not registered intraday"):
        _resolve_candidate_names(
            ("other",),
            all_registered=False,
            module_prefix="factors.library.intraday",
        )


def test_admission_rows_use_old_final_factors_without_oos_or_correlation(monkeypatch):
    registry = {
        "passed": SimpleNamespace(
            validation_horizons=(5, 10, 20), category="microstructure"
        ),
        "failed": SimpleNamespace(
            validation_horizons=(5, 10, 20), category="liquidity"
        ),
    }
    monkeypatch.setattr(
        "workflows.factor_validation.list_registered",
        lambda kind: {"factor": registry},
    )
    screening = {
        "significant_factors": [{
            "name": "passed",
            "best_period": 10,
            "best_variant": "raw",
            "best_ic": -0.04,
            "best_t": -2.8,
            "best_q_value": 0.03,
            "sample_sufficient": True,
            "passes_robustness": True,
            "observation_channel": False,
            "observation_reasons": [],
        }],
        "final_factors": ["passed"],
        "all_results": [
            {
                "name": "passed",
                "best_period": 10,
                "best_variant": "raw",
                "best_ic": -0.04,
                "best_t": -2.8,
                "best_q_value": 0.03,
                "all_periods": {
                    "raw_period_10": {
                        "period": 10,
                        "preprocessing_variant": "raw",
                        "estimable": True,
                        "ic": -0.04,
                        "ic_hac_t": -2.3,
                        "ic_p_value": 0.01,
                        "ols_hac_t": -2.8,
                        "ols_p_value": 0.01,
                        "factor_q_value": 0.03,
                        "local_q_value": 0.02,
                        "factor_fdr_significant": True,
                        "hierarchical_fdr_significant": True,
                        "fwer_significant": False,
                        "evidence_level": "FDR",
                    }
                },
            },
            {
                "name": "failed",
                "best_period": 0,
                "best_variant": "",
                "best_ic": 0.0,
                "best_t": 0.0,
                "best_q_value": 1.0,
                "all_periods": {
                    "neutralized_period_5": {
                        "period": 5,
                        "preprocessing_variant": "neutralized",
                        "estimable": True,
                        "hierarchical_fdr_significant": False,
                    }
                },
            },
        ],
    }

    rows = {row["factor"]: row for row in _admission_result_rows(screening)}
    assert rows["passed"]["final_pass"] is True
    assert rows["passed"]["decision_reason"] == "passed_formal_admission"
    assert rows["passed"]["best_period"] == 10
    assert rows["passed"]["family"] == "microstructure"
    assert "oos_ic" not in rows["passed"]
    assert "cluster_id" not in rows["passed"]
    assert rows["failed"]["final_pass"] is False
    assert rows["failed"]["decision_reason"] == "hierarchical_fdr_not_passed"


def test_formal_run_writes_admissible_full_history_contract(tmp_path, monkeypatch):
    import workflows.factor_validation as module

    project = tmp_path / "project"
    fake_file = project / "workflows" / "factor_validation.py"
    fake_file.parent.mkdir(parents=True)
    monkeypatch.setattr(module, "__file__", str(fake_file))

    factor_class = SimpleNamespace(
        frequency="daily",
        __module__="factors.library.intraday",
        validation_horizons=(5, 10, 20),
        category="microstructure",
    )
    monkeypatch.setattr(
        module,
        "list_registered",
        lambda kind: {"factor": {"probe": factor_class}},
    )
    config = SimpleNamespace(
        date_range=SimpleNamespace(start="2017-01-01", end="2026-05-15"),
        date_policy=SimpleNamespace(
            factor_admission_start="2025-01-01",
            research_cutoff="2026-05-15",
        ),
        validation_policy=SimpleNamespace(
            warmup_days_by_frequency={"daily": 252}
        ),
        data=SimpleNamespace(source="test"),
    )
    monkeypatch.setattr(module, "load_config", lambda path: config)

    cleared = []
    runner = SimpleNamespace(
        config=config,
        factor_engine=SimpleNamespace(clear_cache=lambda: cleared.append(True)),
    )
    monkeypatch.setattr(module, "PipelineRunner", lambda config: runner)

    def fake_screening(*args, output_dir, **kwargs):
        artifacts = module.Path(output_dir)
        (artifacts / "ic_by_window_period.json").write_text("{}", encoding="utf-8")
        (artifacts / "validation_funnel.json").write_text("{}", encoding="utf-8")
        (artifacts / "performance.json").write_text("{}", encoding="utf-8")
        result = {
            "name": "probe",
            "best_period": 10,
            "best_variant": "raw",
            "best_ic": 0.03,
            "best_t": 2.5,
            "best_q_value": 0.04,
            "all_periods": {"raw_period_10": {
                "period": 10,
                "preprocessing_variant": "raw",
                "estimable": True,
                "ic": 0.03,
                "ols_hac_t": 2.5,
                "local_q_value": 0.04,
                "factor_fdr_significant": True,
                "hierarchical_fdr_significant": True,
            }},
        }
        metadata = dict(result)
        metadata.update({
            "sample_sufficient": True,
            "passes_robustness": True,
            "observation_channel": False,
            "observation_reasons": [],
        })
        return {
            "all_results": [result],
            "significant_factors": [metadata],
            "final_factors": ["probe"],
            "discovery_audit": {"total_hypotheses": 3},
            "research_contract": {
                "data_sha256": "a" * 64,
                "config_sha256": "b" * 64,
                "code_sha256": "c" * 64,
            },
            "performance": {},
        }

    monkeypatch.setattr(module, "_run_multi_period_screening", fake_screening)

    run = module.run_default_factor_validation(
        run_id="formal_probe", factor_names=("probe",)
    )

    summary = json.loads((run / "validation_summary.json").read_text(encoding="utf-8"))
    contract = json.loads((run / "run_contract.json").read_text(encoding="utf-8"))
    assert summary["research_window"] == ["2025-01-01", "2026-05-15"]
    assert summary["backtest_start"] == "2017-01-01"
    assert summary["final_pass_count"] == 1
    assert contract["admission_eligible"] is True
    assert contract["window_policy"] == "frozen_factor_admission_window"
    assert "artifacts/performance.json" in contract["files"]
    assert contract["scope"] == "explicit_batch"
    assert not (run / "oos_factor_ic.json").exists()
    assert cleared == [True]


def test_resumed_screening_contract_cannot_change_batch_dates_or_semantics():
    names = ["factor_b", "factor_a"]
    names_hash = __import__("hashlib").sha256(json.dumps(
        names, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    screening = {"research_contract": {
        "factor_count": 2,
        "factor_names_sha256": names_hash,
        "factor_start": "2024-04-24T00:00:00",
        "ic_start": "2025-01-01T00:00:00",
        "ic_end": "2026-05-15T00:00:00",
        "frequency": "daily",
        "horizon_mode": "registered_contract",
        "research_role": "factor_admission",
    }}

    _validate_admission_screening_contract(
        screening,
        names=names,
        factor_start=pd.Timestamp("2024-04-24"),
        ic_start=pd.Timestamp("2025-01-01"),
        ic_end=pd.Timestamp("2026-05-15"),
    )
    with pytest.raises(ValueError, match="does not match"):
        _validate_admission_screening_contract(
            screening,
            names=list(reversed(names)),
            factor_start=pd.Timestamp("2024-04-24"),
            ic_start=pd.Timestamp("2025-01-01"),
            ic_end=pd.Timestamp("2026-05-15"),
        )
