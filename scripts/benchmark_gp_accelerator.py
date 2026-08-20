from __future__ import annotations

import argparse
import cProfile
import csv
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import pstats
import threading
import time

import numpy as np

from factor_mining.api import FeatureConfig, TargetSpec
from factor_mining.data import make_synthetic_panels
from factor_mining.features import FeatureEngine
from factor_mining.gp import GPConfig, GPSearch
from factor_mining.operators import ExpressionEvaluator
from factor_mining.runtime.static_context import StaticResearchContext
from factor_mining.validation import PreparedTarget, ValidationConfig


def _rss_bytes() -> int:
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        get_memory.restype = wintypes.BOOL
        if not get_memory(
            handle, ctypes.byref(counters), counters.cb
        ):
            return 0
        return int(counters.WorkingSetSize)
    except (AttributeError, OSError):
        return 0


class PeakRSS:
    def __init__(self):
        self.start = _rss_bytes()
        self.peak = self.start
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self._stop.wait(0.02):
            self.peak = max(self.peak, _rss_bytes())

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, _rss_bytes())


def _profile_groups(profile: cProfile.Profile) -> dict[str, float]:
    groups = {
        "expression": 0.0,
        "rolling": 0.0,
        "mad": 0.0,
        "neutralization": 0.0,
        "rank_ic": 0.0,
        "portfolio": 0.0,
    }
    if not profile.getstats():
        return groups
    stats = pstats.Stats(profile).stats
    for (filename, _line, function), values in stats.items():
        self_time = float(values[2])
        key = f"{filename}:{function}".lower()
        if "operators.py" in key and (
            ":evaluate" in key or ":_apply" in key or ":_eval" in key
        ):
            groups["expression"] += self_time
        if "rolling" in key or "_fast_rolling" in key:
            groups["rolling"] += self_time
        if "mad_winsorize" in key:
            groups["mad"] += self_time
        if "neutralize" in key:
            groups["neutralization"] += self_time
        if "rank_ic" in key or "rank_rows" in key or "rankdata" in key:
            groups["rank_ic"] += self_time
        if "portfolio" in key or "_add_portfolio_block" in key:
            groups["portfolio"] += self_time
    return groups


def _assert_equivalent(
    baseline_scores,
    scores,
    candidate_ids,
) -> None:
    if [score.direction for score in scores] != [
        score.direction for score in baseline_scores
    ]:
        raise AssertionError("accelerator direction mismatch")
    for old, new in zip(baseline_scores, scores):
        if np.isnan(old.mean_ic):
            if not np.isnan(new.mean_ic):
                raise AssertionError("accelerator IC NaN mismatch")
        elif abs(old.mean_ic - new.mean_ic) >= 1e-10:
            raise AssertionError("accelerator IC error exceeds 1e-10")
        if np.isneginf(old.fitness):
            if not np.isneginf(new.fitness):
                raise AssertionError("accelerator rejected-fitness mismatch")
        elif abs(old.fitness - new.fitness) >= 1e-10:
            raise AssertionError("accelerator fitness error exceeds 1e-10")

    def ordered(items):
        return [
            candidate_ids[index]
            for index in sorted(
                range(len(items)),
                key=lambda index: items[index].fitness,
                reverse=True,
            )
        ]

    if set(candidate_ids) != set(ordered(scores)):
        raise AssertionError("accelerator candidate ID set mismatch")
    if ordered(scores) != ordered(baseline_scores):
        raise AssertionError("accelerator candidate ordering mismatch")


def _csv_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--periods", type=int, default=3000)
    parser.add_argument("--symbols", type=int, default=47)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--block-rows", type=int, default=512)
    parser.add_argument("--workers", type=_csv_ints, default=(2, 4, 8))
    parser.add_argument("--chunk-sizes", type=_csv_ints, default=(50, 75, 100))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--skip-raw-check", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)

    panels = make_synthetic_panels(
        periods=args.periods, symbols=args.symbols, seed=107
    )
    feature_config = FeatureConfig(
        feature_horizons=(1, 2, 5, 10, 15, 30),
        lag_steps=(1, 2, 3, 5, 10),
        rolling_windows=(3, 5, 10, 15, 30),
    )
    started = time.perf_counter()
    features = FeatureEngine(feature_config).build_all_terminals(panels)
    feature_seconds = time.perf_counter() - started
    target = PreparedTarget.from_close(
        panels["close"],
        TargetSpec(name="forward_15p", horizon_bars=15, cost_bps=2.0),
    )
    labels = tuple(
        f"group_{index % 8}" for index in range(args.symbols)
    )
    context = StaticResearchContext.create(
        output_dir / "terminal_snapshot",
        features=features,
        target=target,
        feature_config=feature_config,
        source_fingerprint=(
            f"synthetic:{args.periods}:{args.symbols}:107"
        ),
        taxonomy={"group_labels": list(labels)},
        volatility=features.values["realized_vol_30p"],
        group_labels=labels,
        decision_lag_bars=1,
        block_rows=args.block_rows,
    )
    validation = ValidationConfig(
        min_time_observations=30,
        rebalance_every_bars=15,
        coverage_penalty=0.02,
        segment_floor_weight=0.1,
    )
    base_config = GPConfig(
        population_size=args.population,
        generations=1,
        elite_size=max(1, min(12, args.population - 1)),
        max_depth=4,
        max_complexity=14,
        windows=(2, 3, 5, 10, 15),
        operators=(
            "add", "sub", "mul", "div", "min", "max", "abs", "neg",
            "signed_sqrt", "ts_mean", "ts_std", "ts_min", "ts_max",
            "ts_zscore",
        ),
        seed=109,
    )
    baseline_search = GPSearch(
        features,
        target,
        feature_config=feature_config,
        validation_config=validation,
        gp_config=base_config,
        group_labels=labels,
    )
    expressions = baseline_search._initial_population()
    (output_dir / "fixed_population.json").write_text(
        json.dumps(
            [expression.to_dict() for expression in expressions],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    candidate_ids = [
        f"gp_{expression.sha256[:16]}" for expression in expressions
    ]
    raw_baseline = None
    if not args.skip_raw_check:
        raw_path = output_dir / "fixed_factor_values.npy"
        raw_baseline = np.stack([
            ExpressionEvaluator(features).evaluate(expression)
            for expression in expressions
        ])
        np.save(raw_path, raw_baseline)
        del raw_baseline
        raw_baseline = np.load(raw_path, mmap_mode="r")

    report = {
        "schema_version": 2,
        "shape": list(features.shape),
        "feature_count": len(features.values),
        "factor_count": len(expressions),
        "feature_build_seconds": feature_seconds,
        "gp_config": asdict(base_config),
        "validation_config": asdict(validation),
        "variants": {},
        "factor_chunk_tuning": [],
    }

    def run_variant(
        name: str,
        mode: str,
        *,
        workers: int,
        chunk_size: int,
        baseline_scores=None,
        repeats: int | None = None,
    ):
        config = replace(
            base_config,
            accelerator_mode=mode,
            accelerator_block_rows=args.block_rows,
            accelerator_chunk_size=chunk_size,
            use_fast_rolling=mode != "off",
            n_jobs=workers,
        )
        variant_features = features if mode == "off" else context.features
        variant_target = target if mode == "off" else context.target
        variant_context = None if mode == "off" else context

        def make_search():
            return GPSearch(
                variant_features,
                variant_target,
                feature_config=feature_config,
                validation_config=validation,
                gp_config=config,
                group_labels=labels,
                context=variant_context,
            )

        warmup = make_search()._score_population(expressions)
        if baseline_scores is not None:
            _assert_equivalent(baseline_scores, warmup, candidate_ids)
        samples = []
        scores = warmup
        sample_count = args.repeats if repeats is None else repeats
        for _ in range(sample_count):
            search = make_search()
            with PeakRSS() as memory:
                started = time.perf_counter()
                scores = search._score_population(expressions)
                elapsed = time.perf_counter() - started
            if baseline_scores is not None:
                _assert_equivalent(baseline_scores, scores, candidate_ids)
            samples.append({
                "wall_seconds": elapsed,
                "rss_start_bytes": memory.start,
                "peak_rss_bytes": memory.peak,
                "peak_rss_delta_bytes": max(0, memory.peak - memory.start),
                "executor_stats": dict(search._last_accelerator_stats),
            })

        median_index = int(np.argsort(
            [sample["wall_seconds"] for sample in samples]
        )[len(samples) // 2])
        representative = samples[median_index]
        profile = cProfile.Profile()
        if mode == "off" and not args.skip_profile:
            profile_search = make_search()
            profile.enable()
            profile_search._score_population(expressions)
            profile.disable()
        if raw_baseline is not None:
            evaluator = ExpressionEvaluator(
                variant_features,
                rolling_backend=("pandas" if mode == "off" else "fast"),
            )
            raw = np.stack([
                evaluator.evaluate(expression) for expression in expressions
            ])
            if not np.array_equal(np.isnan(raw), np.isnan(raw_baseline)):
                raise AssertionError(f"{name} factor NaN mask mismatch")
            np.testing.assert_allclose(
                raw, raw_baseline, rtol=1e-6, atol=1e-7, equal_nan=True
            )
            del raw, evaluator
            gc.collect()
        executor_stats = representative["executor_stats"]
        profile_groups = _profile_groups(profile)
        result = {
            "name": name,
            "mode": mode,
            "wall_seconds": float(np.median([
                sample["wall_seconds"] for sample in samples
            ])),
            "wall_seconds_min": min(
                sample["wall_seconds"] for sample in samples
            ),
            "wall_seconds_samples": [
                sample["wall_seconds"] for sample in samples
            ],
            "evaluations": len(expressions),
            "chunk_size": chunk_size if mode in {"chunk", "v2-lite"} else 0,
            "requested_worker_count": (
                workers if mode in {"chunk", "v2-lite"} else 1
            ),
            "worker_count": int(executor_stats.get("worker_count", 1)),
            "chunk_count": int(executor_stats.get("chunk_count", 1)),
            "expression_seconds": float(
                executor_stats.get(
                    "expression_seconds", profile_groups["expression"]
                )
            ),
            "mad_seconds": float(
                executor_stats.get("mad_seconds", profile_groups["mad"])
            ),
            "neutralization_seconds": float(executor_stats.get(
                "neutralization_seconds", profile_groups["neutralization"]
            )),
            "rank_ic_seconds": float(
                executor_stats.get("rank_ic_seconds", profile_groups["rank_ic"])
            ),
            "portfolio_seconds": float(executor_stats.get(
                "portfolio_seconds", profile_groups["portfolio"]
            )),
            "peak_rss_bytes": max(
                sample["peak_rss_bytes"] for sample in samples
            ),
            "peak_rss_delta_bytes": max(
                sample["peak_rss_delta_bytes"] for sample in samples
            ),
            "fallback_count": int(
                executor_stats.get("fallback_count", 0)
            ),
            "fallback_reasons": executor_stats.get("fallback_reasons", {}),
            "dag_reuse_rate": executor_stats.get("dag_reuse_rate"),
            "profile_self_seconds": profile_groups,
            "factor_value_assertion": raw_baseline is not None,
            "hard_assertions_passed": True,
        }
        return scores, result

    baseline_scores, baseline_result = run_variant(
        "baseline", "off", workers=1, chunk_size=75
    )
    report["variants"]["baseline"] = baseline_result
    _, v1_result = run_variant(
        "accelerator_v1", "dag", workers=1, chunk_size=75,
        baseline_scores=baseline_scores,
    )
    report["variants"]["accelerator_v1"] = v1_result

    best = None
    for workers in args.workers:
        for chunk_size in args.chunk_sizes:
            _, tuning = run_variant(
                f"chunk_w{workers}_c{chunk_size}",
                "chunk",
                workers=workers,
                chunk_size=chunk_size,
                baseline_scores=baseline_scores,
                repeats=1,
            )
            report["factor_chunk_tuning"].append(tuning)
            if best is None or tuning["wall_seconds"] < best["wall_seconds"]:
                best = tuning
    if best is None:
        raise RuntimeError("factor chunk tuning produced no result")
    best_workers = int(best["worker_count"])
    best_chunk_size = int(best["chunk_size"])
    _, chunk_result = run_variant(
        "v1_factor_chunk",
        "chunk",
        workers=best_workers,
        chunk_size=best_chunk_size,
        baseline_scores=baseline_scores,
    )
    report["variants"]["v1_factor_chunk"] = chunk_result
    _, v2_result = run_variant(
        "v2_lite",
        "v2-lite",
        workers=best_workers,
        chunk_size=best_chunk_size,
        baseline_scores=baseline_scores,
    )
    report["variants"]["v2_lite"] = v2_result
    report["selected_factor_chunk"] = {
        "worker_count": best_workers,
        "chunk_size": best_chunk_size,
    }

    baseline_seconds = report["variants"]["baseline"]["wall_seconds"]
    for values in report["variants"].values():
        values["speedup_vs_baseline"] = (
            baseline_seconds / values["wall_seconds"]
            if values["wall_seconds"] > 0 else None
        )
    (output_dir / "benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    csv_rows = [
        {"result_type": "variant", **values}
        for values in report["variants"].values()
    ] + [
        {"result_type": "tuning", **values}
        for values in report["factor_chunk_tuning"]
    ]
    scalar_keys = sorted({
        key for row in csv_rows for key, value in row.items()
        if not isinstance(value, (dict, list))
    })
    with (output_dir / "benchmark.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerows([
            {key: row.get(key) for key in scalar_keys} for row in csv_rows
        ])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
