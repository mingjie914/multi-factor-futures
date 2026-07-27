"""Command line entry point for the standalone mining plugin."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys

import pandas as pd

from factor_mining.api import (
    FeatureConfig,
    MiningRunSpec,
    RunMode,
    TargetSpec,
    canonical_json,
    content_hash,
)
from factor_mining.data import LocalParquetData, LocalParquetSpec, make_synthetic_panels
from factor_mining.features import FeatureEngine
from factor_mining.gp import GPConfig, GPSearch
from factor_mining.repository import CandidateRepository
from factor_mining.screening import ScreeningConfig, screen_candidates
from factor_mining.validation import PreparedTarget, ValidationConfig
from core.sectors import sector_for


DEFAULT_REPOSITORY = Path("runs/factor_mining/candidates.sqlite3")
CURVE_FIELDS = (
    "curve_total_oi", "curve_top2_oi", "curve_total_volume",
    "curve_contract_count", "curve_oi_breadth", "curve_oi_concentration",
    "curve_oi_hhi",
)


def _csv_strings(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("value cannot be empty")
    return result


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("bar counts must be positive")
    return result


def _safe_id_fragment(value: str) -> str:
    result = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9_]+", result):
        raise argparse.ArgumentTypeError(
            "candidate prefix may contain only letters, digits, and underscores"
        )
    return result


def _repository(args) -> CandidateRepository:
    return CandidateRepository(args.repository)


def _feature_config(args) -> FeatureConfig:
    raw_fields = list(FeatureConfig().raw_fields)
    if getattr(args, "include_curve", False):
        raw_fields.extend(CURVE_FIELDS)
    return FeatureConfig(
        source_frequency=args.frequency,
        decision_frequency=args.frequency,
        feature_horizons=args.feature_horizons,
        lag_steps=args.lag_steps,
        rolling_windows=args.rolling_windows,
        raw_fields=tuple(dict.fromkeys(raw_fields)),
        include_technicals=not args.no_technicals,
        include_distribution=not args.no_distribution,
        max_feature_memory_mb=args.feature_memory_mb,
    )


def _run_search(args, panels, run_id: str, repository: CandidateRepository | None):
    feature_config = _feature_config(args)
    target_spec = TargetSpec(
        name=f"forward_{args.horizon_bars}p",
        decision_frequency=args.frequency,
        horizon_bars=args.horizon_bars,
        entry_delay_bars=args.entry_delay_bars,
        cost_bps=args.cost_bps,
    )
    features = FeatureEngine(feature_config).build(panels)
    close = panels["close"].reindex(index=features.index, columns=features.symbols)
    target = PreparedTarget.from_close(close, target_spec)
    validation = ValidationConfig(
        decision_lag_bars=args.decision_lag_bars,
        min_cross_section=args.min_cross_section,
        min_time_observations=args.min_time_observations,
        neutralize_volatility=not args.no_volatility_neutralization,
        time_segments=args.time_segments,
        turnover_penalty=args.turnover_penalty,
        complexity_penalty=args.complexity_penalty,
        coverage_penalty=args.coverage_penalty,
        segment_floor_weight=args.segment_floor_weight,
    )
    gp_config = GPConfig(
        population_size=args.population,
        generations=args.generations,
        elite_size=min(args.elite_size, args.population - 1),
        max_depth=args.max_depth,
        max_complexity=args.max_complexity,
        n_jobs=args.jobs,
        evaluator_cache_mb=args.evaluator_cache_mb,
        max_candidates=args.max_candidates,
        min_abs_ic=args.min_abs_ic,
        windows=args.gp_windows,
        operators=args.operators or GPConfig().operators,
        allow_conditionals=args.allow_conditionals,
        seed=args.seed,
    )
    outcome = GPSearch(
        features,
        target,
        feature_config=feature_config,
        validation_config=validation,
        gp_config=gp_config,
        run_id=run_id,
        group_labels=(
            [sector_for(str(symbol)) for symbol in features.symbols]
            if args.sector_neutralization else None
        ),
    ).run()
    if args.candidate_prefix:
        namespaced = []
        for candidate in outcome.candidates:
            expression_hash = str(candidate.payload["expression_sha256"])
            candidate_id = (
                f"gp_{args.candidate_prefix}_h{args.horizon_bars}_"
                f"{expression_hash[:12]}"
            )
            namespaced.append(replace(
                candidate,
                candidate_id=candidate_id,
                framework_name=f"mined_{candidate_id}",
                lineage={
                    **candidate.lineage,
                    "profile": args.candidate_prefix,
                },
                content_sha256="",
            ))
        outcome = replace(outcome, candidates=tuple(namespaced))
    if repository is not None:
        repository.add_candidates(outcome.candidates, run_id=run_id)
    print(canonical_json({
        "run_id": run_id,
        "features": len(features.values),
        "shape": list(features.shape),
        "evaluated_expressions": outcome.evaluated_expressions,
        "candidate_count": len(outcome.candidates),
        "best_generation_fitness": [item.best_fitness for item in outcome.generation_stats],
        "candidates": [
            {
                "candidate_id": item.candidate_id,
                "framework_name": item.framework_name,
                "mean_rank_ic": item.metrics.get("mean_rank_ic"),
                "search_fitness": item.metrics.get("search_fitness"),
            }
            for item in outcome.candidates
        ],
        "warning": "diagnostic mining metrics are not formal HAC/audited evidence",
    }))
    return outcome


def _mine(args) -> int:
    data_root = args.data_root or os.environ.get("MF_PARQUET_ROOT")
    if not data_root:
        raise ValueError("set --data-root or MF_PARQUET_ROOT")
    feature_config = _feature_config(args)
    run_payload = {
        "start": args.start,
        "end": args.end,
        "universe": list(args.universe),
        "target_horizon": args.horizon_bars,
        "feature_config": asdict(feature_config),
        "seed": args.seed,
        "gp_windows": list(args.gp_windows),
        "population": args.population,
        "generations": args.generations,
        "operators": list(args.operators or GPConfig().operators),
    }
    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("mine_%Y%m%dT%H%M%SZ_")
        + content_hash(run_payload)[:8]
    )
    target = TargetSpec(
        name=f"forward_{args.horizon_bars}p",
        decision_frequency=args.frequency,
        horizon_bars=args.horizon_bars,
        entry_delay_bars=args.entry_delay_bars,
        cost_bps=args.cost_bps,
    )
    run_spec = MiningRunSpec(
        run_id=run_id,
        mode=RunMode.MINE,
        seed=args.seed,
        start=args.start,
        end=args.end,
        universe=args.universe,
        target=target,
        feature_config=feature_config,
        metadata={
            "population": args.population,
            "generations": args.generations,
            "elite_size": min(args.elite_size, args.population - 1),
            "max_depth": args.max_depth,
            "max_complexity": args.max_complexity,
            "max_candidates": args.max_candidates,
            "min_abs_ic": args.min_abs_ic,
            "gp_windows": list(args.gp_windows),
            "operators": list(args.operators or GPConfig().operators),
            "allow_conditionals": args.allow_conditionals,
            "sector_neutralization": args.sector_neutralization,
            "validation_config": {
                "time_segments": args.time_segments,
                "turnover_penalty": args.turnover_penalty,
                "complexity_penalty": args.complexity_penalty,
                "coverage_penalty": args.coverage_penalty,
                "segment_floor_weight": args.segment_floor_weight,
            },
        },
    )
    repository = _repository(args)
    repository.add_run(run_spec)
    panels = LocalParquetData(LocalParquetSpec(Path(data_root))).load_panels(
        args.universe, args.start, args.end, feature_config
    )
    _run_search(args, panels, run_id, repository)
    return 0


def _dev_smoke(args) -> int:
    panels = make_synthetic_panels(
        periods=args.periods, symbols=args.symbols, frequency=args.frequency, seed=args.seed
    )
    _run_search(args, panels, "synthetic_dev", None)
    return 0


def _pool_list(args) -> int:
    statuses = args.status if args.status else None
    candidates = _repository(args).list_candidates(statuses=statuses, limit=args.limit)
    print(canonical_json({
        "count": len(candidates),
        "candidates": [
            {
                "candidate_id": item.candidate_id,
                "framework_name": item.framework_name,
                "status": item.status,
                "frequency": item.frequency,
                "target": item.target.name,
                "mean_rank_ic": item.metrics.get("mean_rank_ic"),
                "search_fitness": item.metrics.get("search_fitness"),
                "dependencies": list(item.dependencies),
            }
            for item in candidates
        ],
    }))
    return 0


def _snapshot(args) -> int:
    path = _repository(args).write_snapshot(
        args.output,
        candidate_ids=args.candidate_ids,
        statuses=args.statuses,
        refuse_existing=True,
    )
    print(str(path))
    return 0


def _promote(args) -> int:
    evidence = json.loads(Path(args.evidence_json).read_text(encoding="utf-8"))
    candidate = _repository(args).promote(
        args.candidate_id,
        args.status,
        evidence=evidence,
        run_id=args.run_id,
    )
    print(canonical_json({"candidate_id": candidate.candidate_id, "status": candidate.status}))
    return 0


def _candidate_ids_file(path: Path) -> tuple[str, ...]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        if "candidate_id" not in (rows.fieldnames or ()):
            raise ValueError("candidate file must contain a candidate_id column")
        result = tuple(
            row["candidate_id"].strip() for row in rows
            if row.get("candidate_id", "").strip()
        )
    if not result:
        raise ValueError("candidate file contains no candidate ids")
    if len(result) != len(set(result)):
        raise ValueError("candidate file contains duplicate candidate ids")
    return result


def _record_prescreen_outcome(
    repository: CandidateRepository,
    candidate_id: str,
    result: dict,
    evidence: dict,
    *,
    run_id: str,
) -> None:
    """Persist diagnostics and terminally reject mechanical failures."""

    candidate_evidence = {
        **evidence,
        "valid": bool(result["hard_pass"]),
        "hard_reasons": result["hard_reasons"],
    }
    repository.record_evaluation(
        candidate_id,
        stage="mining_prescreen",
        metrics=result,
        evidence=candidate_evidence,
        run_id=run_id,
    )
    if not result["hard_pass"]:
        current = repository.get_candidate(candidate_id)
        if current.status == "mined_candidate":
            repository.promote(
                candidate_id,
                "rejected",
                evidence=candidate_evidence,
                run_id=run_id,
            )
        elif current.status != "rejected":
            raise ValueError(
                f"cannot apply prescreen rejection to {current.status}: "
                f"{candidate_id}"
            )


def _screen(args) -> int:
    data_root = args.data_root or os.environ.get("MF_PARQUET_ROOT")
    if not data_root:
        raise ValueError("set --data-root or MF_PARQUET_ROOT")
    repository = _repository(args)
    candidate_ids = (
        args.candidate_ids
        if args.candidate_ids
        else (
            _candidate_ids_file(args.candidate_file)
            if args.candidate_file
            else tuple(
                item.candidate_id
                for item in repository.list_candidates(run_ids=args.candidate_run_ids)
            )
        )
    )
    if not candidate_ids:
        raise ValueError("candidate selection returned no candidates")
    candidates = tuple(repository.get_candidate(item) for item in candidate_ids)
    feature_config = candidates[0].feature_config
    universe = args.universe
    panels = LocalParquetData(LocalParquetSpec(Path(data_root))).load_panels(
        universe, args.start, args.end, feature_config
    )
    outcome = screen_candidates(
        candidates,
        panels,
        config=ScreeningConfig(
            min_coverage=args.min_coverage,
            min_cross_section=args.min_cross_section,
            min_time_observations=args.min_time_observations,
            min_variable_row_fraction=args.min_variable_row_fraction,
            correlation_threshold=args.correlation_threshold,
            max_correlation_observations=args.max_correlation_observations,
            evaluator_cache_mb=args.evaluator_cache_mb,
        ),
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    result_frame = pd.DataFrame(outcome.results)
    result_frame["hard_reasons"] = result_frame["hard_reasons"].map(
        lambda values: "|".join(values)
    )
    result_frame["soft_flags"] = result_frame["soft_flags"].map(
        lambda values: "|".join(values)
    )
    result_frame["dependencies"] = result_frame["dependencies"].map(canonical_json)
    result_frame["formula"] = result_frame["formula"].map(canonical_json)
    result_frame.to_csv(output_dir / "prescreen_results.csv", index=False)
    outcome.correlation.to_parquet(output_dir / "signal_correlation.parquet")
    summary = {
        **outcome.summary(),
        "screen_id": args.screen_id,
        "start": args.start,
        "end": args.end,
        "frequency": feature_config.decision_frequency,
        "universe": list(universe),
        "candidate_source": str(args.candidate_file or "explicit_ids"),
        "warning": "same-sample mining pre-screen; not formal HAC evidence",
    }
    summary_path = output_dir / "prescreen_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    snapshot_path = repository.write_snapshot(
        output_dir / "prescreen_candidates.snapshot.json",
        candidate_ids=outcome.passed_candidate_ids,
    )
    evidence = {
        "scope": "mining_prescreen_not_formal_evidence",
        "screen_id": args.screen_id,
        "summary_path": str(summary_path),
        "snapshot_path": str(snapshot_path),
    }
    by_id = {result["candidate_id"]: result for result in outcome.results}
    for candidate_id in candidate_ids:
        _record_prescreen_outcome(
            repository,
            candidate_id,
            by_id[candidate_id],
            evidence,
            run_id=args.screen_id,
        )
    print(canonical_json({
        **outcome.summary(),
        "output_dir": str(output_dir),
        "snapshot": str(snapshot_path),
    }))
    return 0


def _add_search_arguments(parser: argparse.ArgumentParser, *, synthetic: bool) -> None:
    parser.add_argument("--frequency", default="1min", choices=("1min", "5min", "15min"))
    parser.add_argument("--horizon-bars", type=int, default=15)
    parser.add_argument("--entry-delay-bars", type=int, default=1)
    parser.add_argument("--decision-lag-bars", type=int, default=1)
    parser.add_argument("--cost-bps", type=float, default=1.0)
    parser.add_argument(
        "--feature-horizons",
        type=_csv_ints,
        default=(1, 2, 3, 5, 10, 15, 30, 60, 120, 240),
    )
    parser.add_argument(
        "--lag-steps", type=_csv_ints, default=(1, 2, 3, 5, 10, 15, 30, 60)
    )
    parser.add_argument(
        "--rolling-windows",
        type=_csv_ints,
        default=(3, 5, 10, 15, 30, 60, 120, 240),
    )
    parser.add_argument("--include-curve", action="store_true")
    parser.add_argument("--no-technicals", action="store_true")
    parser.add_argument("--no-distribution", action="store_true")
    parser.add_argument("--no-volatility-neutralization", action="store_true")
    parser.add_argument(
        "--sector-neutralization",
        action="store_true",
        help="remove canonical futures-sector means during search and runtime",
    )
    parser.add_argument("--population", type=int, default=160)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--elite-size", type=int, default=12)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--max-complexity", type=int, default=24)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--min-abs-ic", type=float, default=0.01)
    parser.add_argument(
        "--gp-windows", type=_csv_ints, default=GPConfig().windows,
        help="rolling windows available to GP operators, in decision bars",
    )
    parser.add_argument(
        "--operators", type=_csv_strings, default=None,
        help="explicit GP operator vocabulary; defaults to the fast core set",
    )
    parser.add_argument("--allow-conditionals", action="store_true")
    parser.add_argument(
        "--candidate-prefix", type=_safe_id_fragment,
        default=None,
        help="namespace candidate ids across campaign profiles",
    )
    parser.add_argument("--min-cross-section", type=int, default=4)
    parser.add_argument("--min-time-observations", type=int, default=30)
    parser.add_argument("--time-segments", type=int, default=4)
    parser.add_argument("--turnover-penalty", type=float, default=0.002)
    parser.add_argument("--complexity-penalty", type=float, default=0.0005)
    parser.add_argument("--coverage-penalty", type=float, default=0.0)
    parser.add_argument("--segment-floor-weight", type=float, default=0.0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--feature-memory-mb", type=int, default=4096)
    parser.add_argument("--evaluator-cache-mb", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    if synthetic:
        parser.add_argument("--periods", type=int, default=600)
        parser.add_argument("--symbols", type=int, default=12)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m factor_mining",
        description="Local-only futures factor discovery plugin",
    )
    parser.add_argument("--repository", type=Path, default=DEFAULT_REPOSITORY)
    commands = parser.add_subparsers(dest="command", required=True)

    mine = commands.add_parser("mine", help="run GP against local Parquet bars")
    mine.add_argument("--data-root", type=Path, default=None)
    mine.add_argument("--universe", type=_csv_strings, required=True)
    mine.add_argument("--start", required=True)
    mine.add_argument("--end", required=True)
    mine.add_argument("--run-id", default=None)
    _add_search_arguments(mine, synthetic=False)
    mine.set_defaults(handler=_mine)

    dev = commands.add_parser("dev-smoke", help="run a synthetic end-to-end smoke search")
    _add_search_arguments(dev, synthetic=True)
    dev.set_defaults(handler=_dev_smoke)

    pool = commands.add_parser("pool-list", help="list candidate metadata")
    pool.add_argument("--status", action="append", default=[])
    pool.add_argument("--limit", type=int, default=100)
    pool.set_defaults(handler=_pool_list)

    screen = commands.add_parser(
        "screen", help="pre-screen mined candidates on local bars"
    )
    screen.add_argument("--data-root", type=Path, default=None)
    screen.add_argument("--universe", type=_csv_strings, required=True)
    screen.add_argument("--start", required=True)
    screen.add_argument("--end", required=True)
    screen.add_argument("--screen-id", required=True)
    screen.add_argument("--output-dir", type=Path, required=True)
    candidate_selection = screen.add_mutually_exclusive_group(required=True)
    candidate_selection.add_argument("--candidate-ids", type=_csv_strings)
    candidate_selection.add_argument("--candidate-file", type=Path)
    candidate_selection.add_argument("--candidate-run-ids", type=_csv_strings)
    screen.add_argument("--min-coverage", type=float, default=0.50)
    screen.add_argument("--min-cross-section", type=int, default=4)
    screen.add_argument("--min-time-observations", type=int, default=30)
    screen.add_argument("--min-variable-row-fraction", type=float, default=0.05)
    screen.add_argument("--correlation-threshold", type=float, default=0.85)
    screen.add_argument("--max-correlation-observations", type=int, default=100_000)
    screen.add_argument("--evaluator-cache-mb", type=int, default=256)
    screen.set_defaults(handler=_screen)

    snapshot = commands.add_parser("snapshot", help="freeze candidates for framework loading")
    snapshot.add_argument("--output", type=Path, required=True)
    selection = snapshot.add_mutually_exclusive_group(required=True)
    selection.add_argument("--candidate-ids", type=_csv_strings)
    selection.add_argument("--statuses", type=_csv_strings)
    snapshot.set_defaults(handler=_snapshot)

    promote = commands.add_parser("promote", help="record an audited status transition")
    promote.add_argument("candidate_id")
    promote.add_argument("--status", required=True, choices=(
        "development_candidate", "historical_candidate", "oos_validated", "rejected"
    ))
    promote.add_argument("--evidence-json", type=Path, required=True)
    promote.add_argument("--run-id", default=None)
    promote.set_defaults(handler=_promote)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        FileExistsError, FileNotFoundError, KeyError, MemoryError, RuntimeError, ValueError
    ) as exc:
        print(f"factor_mining error: {exc}", file=sys.stderr)
        return 2
