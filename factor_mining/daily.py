"""Explicit daily export of recovered GP formulas; no default registration."""
from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import re
import shutil
import sqlite3
import time

import numpy as np
import pandas as pd

from factor_mining.api import CandidateSpec, content_hash
from factor_mining.bridge import compute_symbolic_candidate, _REGISTERED_EXPECTED_DIRECTIONS
from factor_mining.features import FeatureEngine
from factor_mining.operators import Expr, ExpressionEvaluator
from research.artifacts import sha256_file

ROOT = Path(__file__).resolve().parents[1]
MODULE = "factor_mining.daily"


def _save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def daily_name(candidate):
    return "gp_daily_" + candidate.calculated_hash()[:16]


def recovered_candidates(root):
    """Read every candidate, including prescreen rejects, without editing SQLite."""
    found = {}
    for path in sorted(Path(root).rglob("*.sqlite3")):
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            if connection.execute("pragma integrity_check").fetchone()[0] != "ok":
                raise ValueError(f"invalid candidate database: {path}")
            for (payload,) in connection.execute("select candidate_json from candidates"):
                candidate = CandidateSpec.from_dict(json.loads(payload))
                key = candidate.calculated_hash()
                if key in found and found[key].metrics != candidate.metrics:
                    raise ValueError(f"conflicting training evidence: {candidate.candidate_id}")
                found[key] = candidate
        finally:
            connection.close()
    if not found:
        raise ValueError("no recovered candidates")
    return [found[key] for key in sorted(found)]


def daily_end_locations(close, session_index):
    """Compute quote-dependent row positions once, shared by all expressions."""
    days = pd.DatetimeIndex(session_index).normalize()
    unique = days.unique().sort_values()
    codes = unique.get_indexer(days)
    locations = np.full((len(unique), len(close.columns)), -1, dtype=int)
    for col in range(len(close.columns)):
        rows = np.flatnonzero(np.isfinite(close.iloc[:, col].to_numpy(float)))
        last = np.full(len(unique), -1, dtype=int)
        np.maximum.at(last, codes[rows], rows)
        locations[:, col] = last
    return unique, locations


def daily_end(signal, close, session_index, *, locations=None):
    """Select each instrument's actual last quoted bar; never skip a NaN signal."""
    unique, rows = daily_end_locations(close, session_index) if locations is None else locations
    values = signal.reindex(index=close.index, columns=close.columns).to_numpy(float)
    result = np.full(rows.shape, np.nan)
    present = rows >= 0
    columns = np.broadcast_to(np.arange(len(close.columns)), rows.shape)
    result[present] = values[rows[present], columns[present]]
    return pd.DataFrame(result, index=unique, columns=close.columns)


def compute_daily_batch(candidates, manager, dates, universe):
    """Share real minute panels, features and expression caches within each config."""
    from data.manager import FrequencyDataProvider
    groups = defaultdict(list)
    for candidate in candidates:
        groups[content_hash(asdict(candidate.feature_config))].append(candidate)
    output, failures, timings = {}, {}, []
    for group in groups.values():
        tick = time.perf_counter()
        load_seconds = feature_seconds = 0.
        config = group[0].feature_config
        provider = FrequencyDataProvider(manager, config.decision_frequency,
                                         dates[0], dates[-1], universe)
        try:
            bars = provider.get_calendar()
            fields = set(config.raw_fields) | {"close"}
            panels = {}
            field_errors = {}
            for field in sorted(fields):
                try:
                    panels[field] = provider.get(field, bars, universe)
                except (KeyError, ValueError, NotImplementedError) as exc:
                    field_errors[field] = str(exc)
            needed = set()
            for candidate in group:
                needed.update(Expr.from_dict(candidate.payload["expression"]).terminals())
                post = candidate.payload.get("postprocess", {})
                if post.get("neutralize_volatility") and post.get("volatility_feature"):
                    needed.add(post["volatility_feature"])
            load_seconds = time.perf_counter() - tick
            feature_tick = time.perf_counter()
            features = FeatureEngine(config).build(panels, required_features=needed)
            feature_seconds = time.perf_counter() - feature_tick
            eligibility = getattr(manager, "_factor_eligibility", None)
            sessions = manager.source.trading_session_index(bars)
            locations = daily_end_locations(panels["close"], sessions)
            if eligibility is not None:
                mapped = eligibility.reindex(index=sessions.normalize(), columns=universe, fill_value=False)
                mapped.index = bars
                provider._factor_eligibility = mapped
                mask = mapped.to_numpy(bool)
            else:
                mask = None
            evaluator = ExpressionEvaluator(features, cross_section_mask=mask, rolling_backend="fast")
            for candidate in group:
                name = daily_name(candidate)
                try:
                    missing = set(candidate.dependencies) & set(field_errors)
                    if missing:
                        raise ValueError(f"unsupported dependencies: {sorted(missing)}")
                    signal = compute_symbolic_candidate(candidate, provider, bars, universe,
                                                        features=features, evaluator=evaluator)
                    output[name] = daily_end(signal, panels["close"], sessions, locations=locations).reindex(
                        index=dates, columns=universe)
                except (KeyError, ValueError, NotImplementedError) as exc:
                    failures[name] = str(exc)
        except (KeyError, ValueError, NotImplementedError) as exc:
            failures.update({daily_name(c): str(exc) for c in group})
        for candidate in group:
            name = daily_name(candidate)
            if name not in output:
                output[name] = pd.DataFrame(np.nan, index=dates, columns=universe)
        timings.append({"frequency": config.decision_frequency, "factors": len(group),
                        "seconds": time.perf_counter() - tick,
                        "load_seconds": load_seconds, "feature_seconds": feature_seconds})
        print(f"GP daily group {len(group)}: {timings[-1]['seconds']:.2f}s", flush=True)
    return output, failures, timings


@contextmanager
def registered_daily(candidates, qualification):
    """Register ordinary daily factors only for the explicit workflow lifetime."""
    from threading import Lock
    import factors.library
    from core.interfaces import Factor
    from core.registry import list_registered, register_factor
    registry = list_registered("factor")["factor"]
    cache = OrderedDict()
    cache_lock = Lock()
    added = []
    try:
        for candidate in candidates:
            name = daily_name(candidate)
            if name in registry:
                raise ValueError(f"factor already registered: {name}")
            evidence = qualification[name]
            def compute(self, data, dates, universe, _name=name, _evidence=evidence):
                key = (id(data), tuple(pd.DatetimeIndex(dates).asi8), tuple(universe))
                with cache_lock:
                    if key not in cache:
                        frames, errors, _ = compute_daily_batch(candidates, data, dates, universe)
                        if errors:
                            raise ValueError(f"qualified GP computation failed: {errors}")
                        cache[key] = frames
                        if len(cache) > 2:
                            cache.popitem(last=False)
                    frame = cache[key][_name]
                return frame * _evidence["direction"]
            cls = type(name, (Factor,), {
                "__module__": MODULE, "name": name, "category": candidate.category,
                "frequency": "daily", "input_bar_frequency": candidate.frequency,
                "signal_frequency": "daily", "validation_horizons": (5, 10, 20),
                "expected_direction": 1,
                "training_bars": evidence["training_days"], "training_days": evidence["training_days"],
                "training_start": candidate.metrics.get("training_start", ""),
                "training_end": candidate.metrics.get("training_end", ""),
                "requires_training_sample_contract": True,
                "description": "Frozen daily-end GP exposure",
                "dependencies": lambda self: [], "compute": compute,
            })
            register_factor(name, category=candidate.category)(cls)
            added.append(name)
            _REGISTERED_EXPECTED_DIRECTIONS[name] = 1
        yield added
    finally:
        for name in added:
            registry.pop(name, None)
            _REGISTERED_EXPECTED_DIRECTIONS.pop(name, None)
        cache.clear()


def prepare(study, recovered):
    from core.config import load_config
    from data.manager import DataManager
    from workflows.research import _research_data_fingerprint
    if study.exists():
        raise FileExistsError(study)
    candidates = recovered_candidates(recovered)
    config = load_config("config/default.yaml")
    for candidate in candidates:
        end = pd.Timestamp(candidate.metrics["training_end"])
        if end >= pd.Timestamp("2023-01-01"):
            raise ValueError("recovered training must end before 2023")
    start = min(pd.Timestamp(c.metrics["training_start"]).normalize() for c in candidates)
    start -= pd.Timedelta(days=config.validation_policy.warmup_days_by_frequency["daily"])
    manager = DataManager.from_config(config)
    code = ["factor_mining/daily.py", "factor_mining/bridge.py"]
    code += ["factor_mining/features.py", "factor_mining/operators.py",
             "factor_mining/runtime/rolling_backend.py"]
    protected = ["config/default.yaml", "config/local.yaml", "config/strategy_library.yaml",
                 "factor_library/library.json", "factors/library/intraday.py"]
    manifest = {"candidates": [c.to_dict() for c in candidates],
                "files": {p: sha256_file(ROOT / p) for p in code + protected},
                "data_start": str(start.date()), "cutoff": str(config.date_policy.research_cutoff),
                "universe": list(config.universe), "validation_horizons": [5, 10, 20],
                "daily_export": "original minute postprocess and lags; actual last quoted bar; training orientation once",
                "data_sha256": _research_data_fingerprint(manager.source, start, config.date_policy.research_cutoff)}
    study.mkdir(parents=True)
    _save(study / "gp_daily_contract.json", manifest)


def run(study, action):
    from core.config import load_config
    from pipeline.runner import PipelineRunner
    from factors.processor import build_processing_context
    from workflows.research import _joint_ic_ols_statistics, _research_data_fingerprint
    from workflows.external_factor_admission import qualification_reason
    from scripts.benchmark_gp_accelerator import PeakRSS
    manifest = json.loads((study / "gp_daily_contract.json").read_text(encoding="utf-8"))
    def check():
        for path, digest in manifest["files"].items():
            if sha256_file(ROOT / path) != digest:
                raise ValueError(f"frozen file changed: {path}; create a new study")
    check()
    candidates = [CandidateSpec.from_dict(c) for c in manifest["candidates"]]
    config = load_config("config/default.yaml")
    runner = PipelineRunner(config=config)
    manager = runner.data_manager
    def check_data():
        if _research_data_fingerprint(manager.source, manifest["data_start"], manifest["cutoff"]) != manifest["data_sha256"]:
            raise ValueError("frozen GP input data changed")
    check_data()
    tick, cpu = time.perf_counter(), time.process_time()
    with PeakRSS() as memory:
        if action == "profile":
            path = study / "qualification.json"
            if path.exists():
                raise FileExistsError(path)
            dates = manager.get_calendar(pd.Timestamp(config.date_policy.factor_admission_start) -
                pd.Timedelta(days=config.validation_policy.warmup_days_by_frequency["daily"]), manifest["cutoff"])
            train_dates = manager.get_calendar(manifest["data_start"], "2022-12-31")
            # Warmup supplies features, not training labels or observations.
            # Preflight the actual target interval before expensive minute work.
            training_start = min(pd.Timestamp(c.metrics["training_start"]).normalize() for c in candidates)
            target_dates = train_dates[train_dates >= training_start]
            returns = {h: manager.get_forward_returns(target_dates, config.universe, period=h) for h in (5, 10, 20)}
            full, errors, timing = compute_daily_batch(candidates, manager, dates, config.universe)
            prefix, prefix_errors, prefix_timing = compute_daily_batch(candidates, manager, dates[:len(dates)*3//4], config.universe)
            context = build_processing_context(manager, train_dates, config.universe, config.universe_selection)
            training, train_errors, train_timing = compute_daily_batch(candidates, manager, train_dates, config.universe)
            rows = {}
            for candidate in candidates:
                name = daily_name(candidate)
                reason = errors.get(name) or prefix_errors.get(name) or train_errors.get(name) or qualification_reason(
                    full[name], prefix[name], config.date_policy.factor_admission_start)
                frame = training[name].loc[pd.Timestamp(candidate.metrics["training_start"]).normalize():]
                count = int(frame.notna().sum().median())
                try:
                    processed = runner.processor.process(training[name], context).reindex(frame.index)
                    means = [_joint_ic_ols_statistics(processed, returns[h], forward_period=h)["ic"] for h in (5, 10, 20)]
                    score = float(np.mean(means))
                    if not np.isfinite(score) or score == 0:
                        reason = reason or "daily training direction undefined"
                except (ValueError, RuntimeError) as exc:
                    score, means = 0., []
                    reason = reason or str(exc)
                rows[name] = {"eligible": not reason, "reason": reason, "direction": 1 if score >= 0 else -1,
                              "training_days": count, "training_ic": means}
            check_data()
            _save(path, {"contract_sha256": sha256_file(study / "gp_daily_contract.json"), "rows": rows,
                         "full_timings": timing, "prefix_timings": prefix_timing, "training_timings": train_timing})
        else:
            q = json.loads((study / "qualification.json").read_text(encoding="utf-8"))
            if q["contract_sha256"] != sha256_file(study / "gp_daily_contract.json"):
                raise ValueError("qualification contract mismatch")
            chosen = [c for c in candidates if q["rows"][daily_name(c)]["eligible"]]
            if not chosen:
                raise ValueError("no qualified daily GP candidates")
            with registered_daily(chosen, q["rows"]) as names:
                if action == "validate":
                    from workflows.factor_validation import run_default_factor_validation
                    run_default_factor_validation(run_id=study.name, factor_names=names, module_prefix=MODULE)
                elif action == "admit":
                    import tempfile
                    from research.effective_factor_library import admit_validation_run
                    target = study / "factor_library"
                    if target.exists():
                        raise FileExistsError(target)
                    with tempfile.TemporaryDirectory(dir=study) as temporary:
                        library = Path(temporary) / "library.json"
                        shutil.copyfile(config.factor_library.path, library)
                        admit_validation_run(ROOT / "runs/factor_validation" / study.name, library,
                                             admitted_at=pd.Timestamp.now().date().isoformat())
                        Path(temporary).rename(target)
                else:
                    raise ValueError(action)
            check_data()
    check()
    performance = study / "execution_profile.json"
    records = json.loads(performance.read_text()) if performance.exists() else []
    records.append({"action": action, "wall_seconds": time.perf_counter()-tick,
                    "cpu_seconds": time.process_time()-cpu, "sampled_peak_rss_bytes": memory.peak})
    _save(performance, records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "profile", "validate", "admit"))
    parser.add_argument("--study", required=True)
    parser.add_argument("--recovered", default="runs/factor_mining/recovered_20260910")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.study):
        parser.error("invalid study ID")
    study = ROOT / "runs/factor_research" / args.study
    if args.action == "prepare":
        prepare(study, args.recovered)
    else:
        run(study, args.action)


if __name__ == "__main__":
    main()
