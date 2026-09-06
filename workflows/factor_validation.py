"""Formal factor admission plus explicit, non-admissible split observations."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import re
from pathlib import Path

import pandas as pd

from core.config import load_config
from core.date_policy import factor_validation_window, research_cutoff
from core.registry import list_registered
from factors.processor import build_processing_context
from pipeline.runner import PipelineRunner
from research.artifacts import sha256_file
from workflows.research import (
    _joint_ic_ols_statistics,
    _run_multi_period_screening,
)


def _resolve_candidate_names(
    factor_names,
    *,
    all_registered: bool,
    module_prefix: str,
) -> tuple[list[str], str]:
    """Resolve one explicit hypothesis family; never guess batch versus full pool."""
    if bool(factor_names) == bool(all_registered):
        raise ValueError(
            "select exactly one validation scope: factor_names or all_registered"
        )
    registry = list_registered("factor").get("factor", {})
    eligible = {
        name for name, factor_class in registry.items()
        if factor_class.__module__.startswith(module_prefix)
        and str(getattr(factor_class, "frequency", "daily")).lower() == "daily"
    }
    if all_registered:
        return sorted(eligible), "all_registered_intraday"
    names = list(dict.fromkeys(str(name).strip() for name in factor_names if str(name).strip()))
    invalid = sorted(set(names) - eligible)
    if invalid:
        raise ValueError(f"not registered intraday factors: {invalid}")
    if not names:
        raise ValueError("factor_names must contain at least one factor")
    return names, "explicit_batch"


def _admission_result_rows(screening: dict) -> list[dict]:
    """Flatten the old formal discovery/post-gate decision without OOS or clusters."""
    registry = list_registered("factor").get("factor", {})
    significant = {
        row["name"]: row for row in screening.get("significant_factors", [])
    }
    final = set(screening.get("final_factors", []))
    rows: list[dict] = []
    for result in screening.get("all_results", []):
        name = result["name"]
        selected_period = int(result.get("best_period", 0) or 0)
        selected_variant = str(result.get("best_variant", "") or "")
        local = [
            values for values in result.get("all_periods", {}).values()
            if values.get("estimable")
        ]
        selected = next((
            values for values in local
            if int(values.get("period", 0) or 0) == selected_period
            and str(values.get("preprocessing_variant", "") or "") == selected_variant
        ), {})
        if not selected and selected_period:
            selected = next((
                values for values in local
                if int(values.get("period", 0) or 0) == selected_period
            ), {})
        metadata = significant.get(name, {})
        passed = name in final
        if passed:
            reason = "passed_formal_admission"
        elif name in significant:
            reason = "robustness_sample_or_cost_gate_not_passed"
        elif result.get("hierarchical_fdr_significant", False):
            reason = "ic_t_or_direction_gate_not_passed"
        elif not local:
            reason = "not_estimable"
        else:
            reason = "hierarchical_fdr_not_passed"
        factor_class = registry.get(name)
        registered_horizons = getattr(factor_class, "validation_horizons", ())
        rows.append({
            "factor": name,
            "family": str(getattr(factor_class, "category", "") or ""),
            "registered_horizons": "|".join(map(str, registered_horizons)),
            "selected_period": selected_period or "",
            "selected_variant": selected_variant,
            "ic": selected.get("ic", result.get("best_ic")),
            "ic_hac_t": selected.get("ic_hac_t"),
            "ols_hac_t": selected.get("ols_hac_t", result.get("best_t")),
            "p_value": selected.get("ols_p_value", result.get("best_p_value")),
            "factor_q_value": selected.get("factor_q_value"),
            "local_q_value": selected.get("local_q_value", result.get("best_q_value")),
            "factor_fdr_pass": bool(selected.get("factor_fdr_significant", False)),
            "hierarchical_fdr_pass": bool(
                selected.get("hierarchical_fdr_significant", False)
            ),
            "fwer_pass": bool(selected.get("fwer_significant", False)),
            "evidence_level": selected.get("evidence_level", ""),
            "passes_robustness": metadata.get("passes_robustness"),
            "sample_sufficient": metadata.get("sample_sufficient"),
            "observation_channel": metadata.get("observation_channel"),
            "observation_reasons": "|".join(
                map(str, metadata.get("observation_reasons", []))
            ),
            "final_pass": passed,
            "decision_reason": reason,
        })
    return rows


def _write_csv(
    path: Path,
    rows: list[dict],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    if not rows and not fieldnames:
        raise ValueError(f"refusing to write empty validation table: {path}")
    columns = list(fieldnames or rows[0])
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _validate_admission_screening_contract(
    screening: dict,
    *,
    names: list[str],
    factor_start: pd.Timestamp,
    ic_start: pd.Timestamp,
    ic_end: pd.Timestamp,
) -> None:
    """Fail closed before finalizing an already-computed formal screening."""
    contract = dict(screening.get("research_contract", {}))
    names_hash = hashlib.sha256(json.dumps(
        names, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    expected = {
        "factor_count": len(names),
        "factor_names_sha256": names_hash,
        "factor_start": factor_start.isoformat(),
        "ic_start": ic_start.isoformat(),
        "ic_end": ic_end.isoformat(),
        "frequency": "daily",
        "horizon_mode": "registered_contract",
        "research_role": "factor_admission",
    }
    mismatches = {
        key: (contract.get(key), value)
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "existing screening artifact does not match formal admission contract: "
            f"{mismatches}"
        )


def _evaluate_oos(config, screening: dict, window, *, frequency: str) -> dict:
    final = set(screening.get("final_factors", []))
    specs = {
        row["name"]: row
        for row in screening.get("all_results", [])
        if row.get("name") in final
    }
    if not specs:
        return {}
    oos_config = copy.deepcopy(config)
    oos_config.date_range.start = window.oos_start.date().isoformat()
    oos_config.date_range.end = window.oos_end.date().isoformat()
    oos_config.factors = sorted(specs)
    runner = PipelineRunner(config=oos_config)
    calendar = pd.DatetimeIndex(runner.data_manager.get_calendar(
        window.oos_start - pd.Timedelta(days=window.warmup_calendar_days),
        window.oos_end,
    ))
    universe = pd.Index(oos_config.universe)
    context = build_processing_context(
        runner.data_manager,
        calendar,
        universe,
        oos_config.universe_selection,
    )
    names = sorted(specs)
    computed = runner.factor_engine.compute_factors(
        names, calendar, universe, parallel=False, chunk_size=64
    )
    processed = runner.processor.process_batch(computed, context)
    periods = sorted({
        int(row["best_period"])
        for row in specs.values()
        if int(row.get("best_period", 0)) > 0
    })
    returns = {
        period: runner.data_manager.get_forward_returns(
            calendar, universe, period=period
        )
        for period in periods
    }
    output = {}
    for name in names:
        row = specs[name]
        period = int(row.get("best_period", 0))
        if name not in computed or period not in returns:
            continue
        matrix = (
            runner.processor.process_excluding(
                computed[name], context, {"neutralize"}
            )
            if row.get("best_variant") == "raw"
            else processed[name]
        )
        stats = _joint_ic_ols_statistics(
            matrix.loc[window.oos_start:window.oos_end],
            returns[period].loc[window.oos_start:window.oos_end],
            forward_period=period,
            min_stocks=10,
        )
        train_ic = float(row["best_ic"])
        oos_ic = float(stats["ic"])
        orientation = 1.0 if train_ic >= 0.0 else -1.0
        output[name] = {
            "period": period,
            "preprocessing_variant": row.get("best_variant", "neutralized"),
            "oos_ic": oos_ic,
            "oos_ic_hac_t": float(stats["ic_hac_t"]),
            "oos_ols_beta": float(stats["ols_beta"]),
            "oos_ols_hac_t": float(stats["ols_hac_t"]),
            "oos_ir_nw": float(stats["ir_nw"]),
            "oos_ic_pos_ratio": float(stats["ic_pos_ratio"]),
            "oos_ic_n": int(stats["ic_n"]),
            "oos_days": int(stats["ols_days"]),
            "oriented_oos_ic": oos_ic * orientation,
            "same_direction": bool(oos_ic * orientation > 0.0),
        }
    runner.factor_engine.clear_cache()
    return output


def _result_rows(screening: dict, oos: dict) -> list[dict]:
    registry = list_registered("factor").get("factor", {})
    significant = {
        row["name"]: row for row in screening.get("significant_factors", [])
    }
    final = set(screening.get("final_factors", []))
    rows = []
    for row in screening.get("all_results", []):
        name = row["name"]
        local = [
            values for values in row.get("all_periods", {}).values()
            if values.get("estimable")
        ]
        report = max(
            local,
            key=lambda values: (
                abs(float(values.get("ols_hac_t", 0.0))),
                -int(values.get("period", 0)),
                str(values.get("preprocessing_variant", "")),
            ),
            default={},
        )
        holdout = oos.get(name, {})
        passed = bool(name in final and holdout.get("same_direction", False))
        if passed:
            reason = "passed_is_and_oos_direction"
        elif name in final:
            reason = "oos_direction_reversed_or_zero"
        elif name in significant:
            reason = "is_post_discovery_gate_not_passed"
        elif not local:
            reason = "is_not_estimable"
        else:
            reason = "is_hierarchical_fdr_not_passed"
        metadata = significant.get(name, {})
        rows.append({
            "factor": name,
            "registered_horizons": "|".join(
                map(str, registry[name].validation_horizons)
            ),
            "is_report_period": report.get("period"),
            "is_report_variant": report.get("preprocessing_variant"),
            "is_ic": report.get("ic"),
            "is_ic_hac_t": report.get("ic_hac_t"),
            "is_ols_beta": report.get("ols_beta"),
            "is_ols_hac_t": report.get("ols_hac_t"),
            "is_p_value": report.get("ols_p_value"),
            "is_ir_nw": report.get("ir_nw"),
            "is_ic_pos_ratio": report.get("ic_pos_ratio"),
            "is_n": report.get("n"),
            "is_days": report.get("ols_days"),
            "is_factor_q_value": report.get("factor_q_value"),
            "is_local_q_value": report.get("local_q_value"),
            "is_evidence_level": report.get("evidence_level"),
            "is_factor_fdr_pass": bool(report.get("factor_fdr_significant", False)),
            "is_hierarchical_fdr_pass": bool(
                report.get("hierarchical_fdr_significant", False)
            ),
            "is_fwer_pass": bool(report.get("fwer_significant", False)),
            "is_official_best_period": row.get("best_period"),
            "is_official_best_variant": row.get("best_variant"),
            "is_official_best_ic": row.get("best_ic"),
            "is_official_best_t": row.get("best_t"),
            "is_official_best_q": row.get("best_q_value"),
            "is_discovered": name in significant,
            "is_final_candidate": name in final,
            "is_observation_channel": metadata.get("observation_channel"),
            "is_observation_reasons": "|".join(
                map(str, metadata.get("observation_reasons", []))
            ),
            "oos_period": holdout.get("period"),
            "oos_variant": holdout.get("preprocessing_variant"),
            "oos_ic": holdout.get("oos_ic"),
            "oos_ic_hac_t": holdout.get("oos_ic_hac_t"),
            "oos_ols_beta": holdout.get("oos_ols_beta"),
            "oos_ols_hac_t": holdout.get("oos_ols_hac_t"),
            "oos_ir_nw": holdout.get("oos_ir_nw"),
            "oos_ic_pos_ratio": holdout.get("oos_ic_pos_ratio"),
            "oos_ic_n": holdout.get("oos_ic_n"),
            "oos_days": holdout.get("oos_days"),
            "oriented_oos_ic": holdout.get("oriented_oos_ic"),
            "oos_same_direction": holdout.get("same_direction"),
            "final_pass": passed,
            "decision_reason": reason,
        })
    return rows


def run_single_split_factor_observation(
    *,
    run_id: str,
    config_path: str = "config/default.yaml",
    module_prefix: str = "factors.library.intraday",
    common_horizon: int | None = None,
) -> Path:
    """Run the later 126/42 split as observation evidence, never admission."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("run_id 仅允许字母、数字、点、下划线和连字符")
    project = Path(__file__).resolve().parents[1]
    run_dir = project / "runs" / "factor_validation" / run_id
    if run_dir.exists():
        if (run_dir / "run_contract.json").exists():
            raise FileExistsError(
                f"validation run already finalized: {run_dir}"
            )
        checkpoint = run_dir / "artifacts" / ".multi_period_checkpoint.json"
        screening_artifact = run_dir / "artifacts" / "ic_by_window_period.json"
        if not checkpoint.exists() and not screening_artifact.exists():
            raise FileExistsError(
                f"validation run exists without a resumable research artifact: {run_dir}"
            )
        # A failed finalization (for example, zero passed factors) may be
        # resumed from its immutable research checkpoint without recomputing
        # the 588-factor scan or overwriting a finalized run.
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(exist_ok=True)
    config = load_config(config_path)
    runner = PipelineRunner(config=config)
    window = factor_validation_window(
        config, runner.data_manager, frequency="daily_intraday"
    )
    names, _ = _resolve_candidate_names(
        None, all_registered=True, module_prefix=module_prefix
    )
    screening_path = artifacts / "ic_by_window_period.json"
    if screening_path.exists():
        screening = json.loads(screening_path.read_text(encoding="utf-8"))
        contract = dict(screening.get("research_contract", {}))
        expected_mode = (
            "common_horizon" if common_horizon is not None
            else "registered_contract"
        )
        if contract.get("horizon_mode") != expected_mode or (
            common_horizon is not None
            and int(contract.get("common_horizon", 0)) != int(common_horizon)
        ):
            raise ValueError(
                "existing factor screening artifact does not match the requested "
                "horizon contract"
            )
        # The expensive screening phase is already immutable.  This path is
        # only a post-screen finalization/resume and never recomputes or
        # changes its factor statistics.
    else:
        screening = _run_multi_period_screening(
            runner,
            names,
            config_path,
            1.96,
            window.factor_start,
            window.is_start,
            window.is_end,
            periods_override=None,
            frequency="daily_intraday",
            output_dir=str(artifacts),
            adaptivity_file=None,
            research_role="factor_validation_is",
            common_horizon=common_horizon,
        )
    oos = _evaluate_oos(config, screening, window, frequency="daily_intraday")
    rows = _result_rows(screening, oos)
    passed = sorted(
        (row for row in rows if row["final_pass"]),
        key=lambda row: (
            -abs(float(row.get("oos_ic_hac_t") or 0.0)), row["factor"]
        ),
    )
    _write_csv(run_dir / "factor_validation_full.csv", rows)
    _write_csv(
        run_dir / "passed_factors.csv",
        passed,
        fieldnames=list(rows[0]) if rows else ["factor", "final_pass"],
    )
    (run_dir / "oos_factor_ic.json").write_text(
        json.dumps(oos, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "data_source": config.data.source,
        "research_cutoff": window.oos_end.date().isoformat(),
        "warmup": [
            window.factor_start.date().isoformat(),
            (window.is_start - pd.Timedelta(days=1)).date().isoformat(),
        ],
        "is": [
            window.is_start.date().isoformat(),
            window.is_end.date().isoformat(),
            window.is_bars,
        ],
        "oos": [
            window.oos_start.date().isoformat(),
            window.oos_end.date().isoformat(),
            window.oos_bars,
        ],
        "factor_count": len(rows),
        "estimable_factor_count": sum(
            row["decision_reason"] != "is_not_estimable" for row in rows
        ),
        "hierarchical_fdr_discoveries": len(screening.get("significant_factors", [])),
        "is_final_candidates": len(screening.get("final_factors", [])),
        "oos_evaluated": len(oos),
        "final_pass_count": len(passed),
        "final_gate": "IS final candidate AND oriented OOS IC > 0",
        "horizon_mode": (
            "common_horizon" if common_horizon is not None else "registered_contract"
        ),
        "common_horizon": common_horizon,
        "passed_factors": [row["factor"] for row in passed],
    }
    (run_dir / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # Bind the durable screening JSON and funnel to an artifact manifest.  A
    # post-screen resume may already have these files but must never overwrite
    # an existing manifest.
    manifest_path = artifacts / "manifest.json"
    if not manifest_path.exists():
        from research.artifacts import ResearchArtifactBundle

        research_contract = dict(screening.get("research_contract", {}))
        ResearchArtifactBundle.create(
            artifacts,
            artifact_id=f"{run_id}:screening",
            train_start=window.is_start,
            train_end=window.is_end,
            data_sha256=str(research_contract["data_sha256"]),
            config_sha256=str(research_contract["config_sha256"]),
            code_sha256=str(research_contract["code_sha256"]),
            files={
                "ic_by_window_period.json": artifacts / "ic_by_window_period.json",
                "validation_funnel.json": artifacts / "validation_funnel.json",
            },
            metadata={
                "workflow": "single-split-factor-observation",
                "horizon_mode": summary["horizon_mode"],
                "common_horizon": common_horizon,
            },
        )
    contract = {
        "schema_version": 1,
        "run_id": run_id,
        "workflow": "single-split-factor-observation",
        "admission_eligible": False,
        "window_policy": "default_warmup_plus_126_is_plus_42_oos",
        "horizon_policy": {
            "mode": (
                "common_horizon" if common_horizon is not None
                else "registered_contract"
            ),
            "common_horizon": common_horizon,
        },
        "research_contract": screening.get("research_contract", {}),
        "oos_start": window.oos_start.date().isoformat(),
        "oos_end": window.oos_end.date().isoformat(),
        "files": {
            name: {"sha256": sha256_file(run_dir / name)}
            for name in (
                "factor_validation_full.csv",
                "passed_factors.csv",
                "validation_summary.json",
                "oos_factor_ic.json",
                "artifacts/manifest.json",
            )
        },
    }
    (run_dir / "run_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"全量明细: {run_dir / 'factor_validation_full.csv'}")
    print(f"观察期同方向因子: {len(passed)}/{len(rows)}")
    return run_dir


def run_default_factor_validation(
    *,
    run_id: str,
    config_path: str = "config/default.yaml",
    factor_names=None,
    all_registered: bool = False,
    module_prefix: str = "factors.library.intraday",
) -> Path:
    """Run the restored formal admission policy through the frozen cutoff."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("run_id 仅允许字母、数字、点、下划线和连字符")
    names, scope = _resolve_candidate_names(
        factor_names,
        all_registered=all_registered,
        module_prefix=module_prefix,
    )
    project = Path(__file__).resolve().parents[1]
    run_dir = project / "runs" / "factor_validation" / run_id
    if run_dir.exists():
        if (run_dir / "run_contract.json").exists():
            raise FileExistsError(f"validation run already finalized: {run_dir}")
        checkpoint = run_dir / "artifacts" / ".multi_period_checkpoint.json"
        screening_artifact = run_dir / "artifacts" / "ic_by_window_period.json"
        if not checkpoint.exists() and not screening_artifact.exists():
            raise FileExistsError(
                f"validation run exists without a resumable research artifact: {run_dir}"
            )
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(exist_ok=True)

    config = load_config(config_path)
    runner = PipelineRunner(config=config)
    ic_start = pd.Timestamp(config.date_range.start).normalize()
    ic_end = research_cutoff(config)
    warmup_days = int(config.validation_policy.warmup_days_by_frequency["daily"])
    factor_start = ic_start - pd.Timedelta(days=warmup_days)
    runner.config.date_range.start = ic_start.date().isoformat()
    runner.config.date_range.end = ic_end.date().isoformat()
    screening_path = artifacts / "ic_by_window_period.json"
    if screening_path.exists():
        screening = json.loads(screening_path.read_text(encoding="utf-8"))
        _validate_admission_screening_contract(
            screening,
            names=names,
            factor_start=factor_start,
            ic_start=ic_start,
            ic_end=ic_end,
        )
    else:
        screening = _run_multi_period_screening(
            runner,
            names,
            config_path,
            1.96,
            factor_start,
            ic_start,
            ic_end,
            periods_override=None,
            frequency="daily",
            output_dir=str(artifacts),
            adaptivity_file=None,
            research_role="factor_admission",
            common_horizon=None,
        )
    rows = _admission_result_rows(screening)
    passed = sorted(
        (row for row in rows if row["final_pass"]),
        key=lambda row: (-abs(float(row.get("ols_hac_t") or 0.0)), row["factor"]),
    )
    _write_csv(run_dir / "factor_validation_full.csv", rows)
    _write_csv(
        run_dir / "passed_factors.csv",
        passed,
        fieldnames=list(rows[0]) if rows else ["factor", "final_pass"],
    )
    summary = {
        "schema_version": 2,
        "workflow": "factor_admission",
        "admission_policy": "hierarchical_fdr_post_gates",
        "scope": scope,
        "data_source": config.data.source,
        "research_window": [ic_start.date().isoformat(), ic_end.date().isoformat()],
        "research_cutoff": ic_end.date().isoformat(),
        "warmup_start": factor_start.date().isoformat(),
        "factor_count": len(rows),
        "hypothesis_count": int(
            screening.get("discovery_audit", {}).get("total_hypotheses", 0)
        ),
        "hierarchical_fdr_discoveries": len(
            screening.get("significant_factors", [])
        ),
        "final_pass_count": len(passed),
        "final_gate": "hierarchical FDR + IC/t/direction/robustness/sample/cost",
        "horizon_mode": "registered_contract",
        "passed_factors": [row["factor"] for row in passed],
    }
    (run_dir / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    from research.artifacts import ResearchArtifactBundle

    research_contract = dict(screening.get("research_contract", {}))
    if not (artifacts / "manifest.json").exists():
        ResearchArtifactBundle.create(
            artifacts,
            artifact_id=f"{run_id}:screening",
            train_start=ic_start,
            train_end=ic_end,
            data_sha256=str(research_contract["data_sha256"]),
            config_sha256=str(research_contract["config_sha256"]),
            code_sha256=str(research_contract["code_sha256"]),
            files={
                "ic_by_window_period.json": artifacts / "ic_by_window_period.json",
                "validation_funnel.json": artifacts / "validation_funnel.json",
            },
            metadata={
                "workflow": "factor-admission-validation",
                "scope": scope,
                "horizon_mode": "registered_contract",
            },
        )
    contract = {
        "schema_version": 2,
        "run_id": run_id,
        "workflow": "factor-admission-validation",
        "admission_eligible": True,
        "window_policy": "full_history_through_research_cutoff",
        "scope": scope,
        "factors": names,
        "horizon_policy": {"mode": "registered_contract"},
        "research_contract": research_contract,
        "files": {
            name: {"sha256": sha256_file(run_dir / name)}
            for name in (
                "factor_validation_full.csv",
                "passed_factors.csv",
                "validation_summary.json",
                "artifacts/manifest.json",
            )
        },
    }
    (run_dir / "run_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    runner.factor_engine.clear_cache()
    print(f"正式准入明细: {run_dir / 'factor_validation_full.csv'}")
    print(f"统计通过因子: {len(passed)}/{len(rows)}")
    return run_dir


def run_common_horizon_factor_validation(
    *,
    run_id: str,
    common_horizon: int,
    config_path: str = "config/default.yaml",
    module_prefix: str = "factors.library.intraday",
) -> Path:
    """Run a governed, non-default common-horizon comparison.

    The factor registry remains unchanged.  The result is an independent
    research bundle and cannot be admitted by changing the current library
    implicitly.
    """
    horizon = int(common_horizon)
    if horizon < 1:
        raise ValueError("common_horizon must be positive")
    return run_single_split_factor_observation(
        run_id=run_id,
        config_path=config_path,
        module_prefix=module_prefix,
        common_horizon=horizon,
    )


def main() -> None:
    """Compatibility CLI; the IDE entrypoint is ``run_factor_workflow.py``."""
    parser = argparse.ArgumentParser(description="正式因子准入或显式共同周期观察")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--run-id", required=True)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--factors", help="本次冻结批次，使用逗号分隔的因子注册名")
    scope.add_argument("--all", action="store_true", help="显式重建全部日内因子")
    scope.add_argument(
        "--common-horizon", type=int,
        help="非准入共同周期观察；结果不能写入有效因子库",
    )
    parser.add_argument(
        "--module-prefix", default="factors.library.intraday",
        help="注册因子模块前缀",
    )
    args = parser.parse_args()
    try:
        if args.common_horizon is not None:
            run_common_horizon_factor_validation(
                run_id=args.run_id,
                common_horizon=args.common_horizon,
                config_path=args.config,
                module_prefix=args.module_prefix,
            )
        else:
            run_default_factor_validation(
                run_id=args.run_id,
                config_path=args.config,
                factor_names=(
                    tuple(name.strip() for name in args.factors.split(","))
                    if args.factors else None
                ),
                all_registered=args.all,
                module_prefix=args.module_prefix,
            )
    except ValueError as exc:
        parser.error(str(exc))
