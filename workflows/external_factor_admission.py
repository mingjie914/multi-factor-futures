"""Opt-in daily adapters; existing admission, library and selection stay unchanged.

Run with ``python -m workflows.external_factor_admission --help``. Importing this
module never registers factors. All outputs belong to one explicit study.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

import numpy as np

MODULE = "workflows.external_factor_admission"
PREFIX = "external_daily__"
ROOT = Path(__file__).resolve().parents[1]


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _io_bytes():
    """Process I/O transfer counters on Windows; unavailable is not zero."""
    import ctypes
    try:
        counters = (ctypes.c_ulonglong * 6)()
        fn = ctypes.windll.kernel32.GetProcessIoCounters
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(type(counters))]
        if fn(ctypes.c_void_p(-1), ctypes.byref(counters)):
            return int(counters[3]), int(counters[4])
    except (AttributeError, OSError):
        pass
    return None


def baseline_files():
    """Protect tracked baseline files plus local data-routing configuration."""
    names = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode("utf-8").split("\0")
    names += ["config/local.yaml"]
    return {n: _hash(ROOT / n) for n in names if n and (ROOT / n).is_file()}


def assert_baseline(manifest):
    changed = [name for name, digest in manifest["baseline_files"].items()
               if not (ROOT / name).is_file() or _hash(ROOT / name) != digest]
    if changed:
        raise ValueError(f"protected baseline changed: {changed}")


def inventory():
    from factors.library import load_research_factor_catalog
    load_research_factor_catalog()
    from core.registry import list_registered
    from factors.specs import SPEC_BY_SLUG
    from factors.user.ta_cn_formula_library import FACTOR_SPECS

    unavailable = {s.slug: s.unavailable_reason for s in FACTOR_SPECS if not s.available}
    rows = []
    for name, cls in sorted(list_registered("factor")["factor"].items()):
        if cls.__module__.startswith("factors.library.intraday") or name.startswith(PREFIX):
            continue
        spec = SPEC_BY_SLUG.get(name)
        reason = unavailable.get(name, "")
        if getattr(cls, "requires_training_sample_contract", False):
            reason = "trained candidate requires an independently reviewed frozen snapshot"
        elif cls.frequency != "daily" and spec is None:
            reason = "no reviewed daily aggregation adapter"
        mixed = cls.__module__ == "factors.library.cross_frequency"
        if mixed:
            reason = "unverified mixed-frequency derived-field contract"
        factor_spec = getattr(cls, "factor_spec", None)
        rows.append({
            "source_factor": name, "factor": PREFIX + name,
            "source_module": cls.__module__, "spec": copy.deepcopy(spec),
            "input_bar_frequency": "unresolved" if mixed else str(getattr(cls, "input_bar_frequency", cls.frequency)),
            "signal_frequency": "daily", "validation_horizons": list(cls.validation_horizons),
            "horizon_policy": "predeclared slug-window policy for new daily exposure; not a bar-duration conversion",
            "history_years": int(factor_spec.years) + 1 if cls.__module__ == "factors.user.calendar_seasonality" else 0,
            "history_months": int(factor_spec.window) + 6 if cls.__module__ == "factors.user.macro_beta" else 0,
            "category": cls.category, "dependencies": list(cls().dependencies()),
            "blocked_reason": reason,
        })
    return rows


@contextmanager
def registered_external(rows):
    """Add aliases only; never modify original classes or default SPEC entries."""
    from factors.library import load_research_factor_catalog
    load_research_factor_catalog()
    from core.registry import list_registered, register_factor
    from factors.specs import SPEC_BY_SLUG

    registry = list_registered("factor")["factor"]
    selected = [r for r in rows if not r["blocked_reason"]]
    for row in selected:
        if row["factor"] in registry or row["factor"] in SPEC_BY_SLUG:
            raise ValueError(f"external alias already exists: {row['factor']}")
    added = []
    try:
        for row in selected:
            source = registry[row["source_factor"]]
            alias = row["factor"]
            # Call the original instance to preserve any use of self.name/spec.
            def compute(self, data, dates, universe, _source=source, _row=row):
                request = dates
                if _row.get("history_years") or _row.get("history_months"):
                    request = data.get_calendar(_history_start(dates[0], _row), dates[-1])
                return _source().compute(data, request, universe).reindex(index=dates, columns=universe)
            cls = type(alias, (source,), {
                "__module__": MODULE, "name": alias, "frequency": "daily",
                "input_bar_frequency": row["input_bar_frequency"],
                "signal_frequency": "daily",
                "validation_horizons": tuple(row["validation_horizons"]),
                "compute": compute,
            })
            register_factor(alias, category=row["category"])(cls)
            added.append(alias)
            if row["spec"] is not None:
                # Existing engine recognizes slugs, retaining shared base/transform work.
                spec = copy.deepcopy(row["spec"])
                spec["slug"] = alias
                SPEC_BY_SLUG[alias] = spec
        yield [r["factor"] for r in selected]
    finally:
        for alias in added:
            registry.pop(alias, None)
            SPEC_BY_SLUG.pop(alias, None)


def prepare(study):
    from core.config import load_config
    from research.artifacts import canonical_config_hash
    if study.exists():
        raise FileExistsError(f"study already exists: {study}")
    config = load_config("config/default.yaml")
    rows = inventory()
    study.mkdir(parents=True)
    manifest = {
        "schema_version": 1, "baseline_files": baseline_files(),
        "baseline_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "adapter_sha256": _hash(Path(__file__)),
        "config_sha256": canonical_config_hash(config),
        "rows": rows, "gp_status": "not loaded; no candidate snapshot supplied",
    }
    _write(study / "external_contract.json", manifest)
    print(f"Frozen {len(rows)} external candidates in {study}")


def _history_start(start, row, *, include_internal=False):
    import pandas as pd
    # Macro source additionally requests window+6 months before the provided dates.
    return pd.Timestamp(start) - pd.DateOffset(
        years=int(row.get("history_years", 0)),
        months=int(row.get("history_months", 0)) * (2 if include_internal else 1),
    )


def qualification_reason(frame, prefix_frame, start):
    values = frame.loc[start:].to_numpy(dtype=float)
    if not any(np.unique(v[np.isfinite(v)]).size > 1 for v in values):
        return "no finite cross-sectional variation"
    prefix = prefix_frame.loc[start:]
    if prefix.empty:
        return "no prefix observations for timing verification"
    original = frame.reindex(index=prefix.index, columns=prefix.columns).to_numpy(dtype=float)
    truncated = prefix.to_numpy(dtype=float)
    if not np.array_equal(np.isfinite(original), np.isfinite(truncated)) or not np.allclose(
        original, truncated, rtol=1e-10, atol=1e-12, equal_nan=True
    ):
        return "prefix changed when future inputs were removed"
    return ""


def profile(study, manifest):
    """Whole-window mechanical qualification, without reading forward returns."""
    import pandas as pd
    from core.config import load_config
    from data.manager import DataManager
    from factors.engine import FactorEngine
    from workflows.research import _research_data_fingerprint

    output = study / "qualification.json"
    if output.exists():
        raise FileExistsError(output)
    config = load_config("config/default.yaml")
    start = pd.Timestamp(config.date_policy.factor_admission_start)
    end = pd.Timestamp(config.date_policy.research_cutoff)
    warmup = start - pd.Timedelta(days=config.validation_policy.warmup_days_by_frequency["daily"])
    manager = DataManager.from_config(config)
    dates = manager.get_calendar(warmup, end)
    prefix_dates = dates[:len(dates) * 3 // 4]
    t0 = time.perf_counter()
    data_start = min(_history_start(warmup, row, include_internal=True) for row in manifest["rows"])
    fingerprint = _research_data_fingerprint(manager.source, data_start, end)
    rows, batches = [], []
    with registered_external(manifest["rows"]) as names:
        for offset in range(0, len(names), 64):
            batch = names[offset:offset + 64]
            engine = FactorEngine(manager, tolerant=True, log_failures=False)
            tick = time.perf_counter()
            frames = engine.compute_factors(batch, dates, config.universe, parallel=False)
            full_elapsed = time.perf_counter() - tick
            prefix_engine = FactorEngine(manager, tolerant=True, log_failures=False)
            prefix_frames = prefix_engine.compute_factors(batch, prefix_dates, config.universe, parallel=False)
            elapsed = time.perf_counter() - tick
            batches.append({"factors": batch, "seconds": elapsed,
                            "full_seconds": full_elapsed,
                            "prefix_seconds": elapsed - full_elapsed,
                            "factor_timings": list(engine.computation_timings)})
            for name in batch:
                values = frames[name].loc[start:end].to_numpy(dtype=float)
                finite = np.isfinite(values)
                reason = qualification_reason(frames[name].loc[start:end], prefix_frames[name], start)
                errors = [f for f in engine.failures if f["factor"] in (name, "*batch*", "*spec_batch*")]
                rows.append({"factor": name, "eligible": not reason, "reason": reason,
                             "finite_fraction": float(finite.mean()), "errors": errors})
            print(f"Qualification {offset + len(batch)}/{len(names)}: {elapsed:.2f}s", flush=True)
            del frames, engine, prefix_frames, prefix_engine
    if fingerprint != _research_data_fingerprint(manager.source, data_start, end):
        raise ValueError("source data changed during qualification")
    result = {"contract_sha256": _hash(study / "external_contract.json"),
              "data_sha256": fingerprint, "data_start": str(data_start.date()), "batches": batches,
              "rows": rows, "wall_seconds": time.perf_counter() - t0}
    _write(output, result)


def execute(study, action):
    from scripts.benchmark_gp_accelerator import PeakRSS
    from core.config import load_config
    from workflows.factor_validation import run_default_factor_validation
    from research.effective_factor_library import admit_validation_run

    manifest = json.loads((study / "external_contract.json").read_text(encoding="utf-8"))
    assert_baseline(manifest)
    if manifest["adapter_sha256"] != _hash(Path(__file__)):
        raise ValueError("adapter changed after freezing; create a new study")
    cpu0, io0, t0 = time.process_time(), _io_bytes(), time.perf_counter()
    memory = PeakRSS()
    memory.__enter__()
    status = "failed"
    try:
        if action == "profile":
            result = profile(study, manifest)
            status = "complete"
            return result
        qualification = json.loads((study / "qualification.json").read_text(encoding="utf-8"))
        if qualification["contract_sha256"] != _hash(study / "external_contract.json"):
            raise ValueError("qualification contract mismatch")
        names = [r["factor"] for r in qualification["rows"] if r["eligible"]]
        if not names:
            raise ValueError("no mechanically eligible external factors")
        with registered_external(manifest["rows"]):
            if action == "validate":
                import pandas as pd
                from data.manager import DataManager
                from workflows.research import _research_data_fingerprint
                config = load_config("config/default.yaml")
                start = pd.Timestamp(qualification["data_start"])
                end = pd.Timestamp(config.date_policy.research_cutoff)
                manager = DataManager.from_config(config)
                if qualification["data_sha256"] != _research_data_fingerprint(manager.source, start, end):
                    raise ValueError("source changed after qualification; start a new study")
                result = run_default_factor_validation(
                    run_id=study.name, config_path="config/default.yaml",
                    factor_names=names, module_prefix=MODULE,
                )
                if qualification["data_sha256"] != _research_data_fingerprint(manager.source, start, end):
                    raise ValueError("source changed during validation; do not admit this run")
                status = "complete"
                return result
            library = study / "factor_library" / "library.json"
            if action == "admit":
                if library.exists():
                    raise FileExistsError(library)
                config = load_config("config/default.yaml")
                # Publish the isolated directory only after native admission succeeds.
                with tempfile.TemporaryDirectory(prefix=".admit-", dir=study) as temp:
                    staged = Path(temp) / "library.json"
                    shutil.copyfile(ROOT / config.factor_library.path, staged)
                    result = admit_validation_run(
                        ROOT / "runs" / "factor_validation" / study.name,
                        library_path=staged, admitted_at=__import__("datetime").date.today().isoformat(),
                    )
                    Path(temp).rename(library.parent)
                    status = "complete"
                    return result
            if action == "select":
                import yaml
                from workflows.factor_selection import run_effective_factor_selection
                if not library.is_file():
                    raise FileNotFoundError("admit the external run into its isolated library first")
                config_path = study / "selection.yaml"
                config = yaml.safe_load((ROOT / "config/default.yaml").read_text(encoding="utf-8"))
                config["factor_library"]["path"] = str(library)
                text = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
                if config_path.exists() and config_path.read_text(encoding="utf-8") != text:
                    raise ValueError("isolated selection config differs from the frozen default")
                config_path.write_text(text, encoding="utf-8")
                result = run_effective_factor_selection(
                    run_id=study.name, config_path=str(config_path), selection_start="2016-03-31",
                )
                status = "complete"
                return result
            raise ValueError(f"unknown external action: {action}")
    finally:
        memory.__exit__(None, None, None)
        cpu1, io1 = time.process_time(), _io_bytes()
        path = study / "execution_profile.json"
        records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        records.append({"action": action, "status": status,
                        "wall_seconds": time.perf_counter() - t0,
                        "cpu_seconds": cpu1 - cpu0,
                        "sampled_peak_rss_bytes": memory.peak,
                        "read_bytes": io1[0] - io0[0] if io0 and io1 else None,
                        "write_bytes": io1[1] - io0[1] if io0 and io1 else None})
        _write(path, records)
        assert_baseline(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "profile", "validate", "admit", "select"))
    parser.add_argument("--study", required=True, help="new immutable study ID")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.study):
        parser.error("study ID must contain only letters, digits, underscore or hyphen")
    study = ROOT / "runs" / "factor_research" / args.study
    if args.action == "prepare":
        prepare(study)
    else:
        execute(study, args.action)


if __name__ == "__main__":
    main()
