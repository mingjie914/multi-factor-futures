from __future__ import annotations

from contextlib import contextmanager
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.registry import list_registered
from factors.specs import SPEC_BY_SLUG
from workflows import external_factor_admission as ext


def _row(source_name: str) -> dict:
    from factors.library import load_research_factor_catalog
    load_research_factor_catalog()
    cls = list_registered("factor")["factor"][source_name]
    return {
        "source_factor": source_name,
        "factor": ext.PREFIX + source_name,
        "source_module": cls.__module__,
        "spec": copy.deepcopy(SPEC_BY_SLUG.get(source_name)),
        "input_bar_frequency": getattr(cls, "input_bar_frequency", cls.frequency),
        "signal_frequency": "daily",
        "validation_horizons": list(cls.validation_horizons),
        "category": cls.category,
        "dependencies": list(cls().dependencies()),
        "blocked_reason": "",
    }


def test_import_is_registry_neutral(monkeypatch):
    before = dict(list_registered("factor")["factor"])
    monkeypatch.syspath_prepend(str(Path(ext.__file__).parent.parent))
    import importlib

    importlib.reload(ext)
    assert dict(list_registered("factor")["factor"]) == before


def test_inventory_excludes_intraday_and_records_unavailable_stock_formulas():
    rows = ext.inventory()
    assert len(rows) == 3460
    assert "volume_price_corr_20d" not in {r["source_factor"] for r in rows}
    assert all(not r["source_module"].startswith("factors.library.intraday") for r in rows)
    blocked = {r["source_factor"]: r["blocked_reason"] for r in rows if r["blocked_reason"]}
    assert {n for n in blocked if n.startswith(("wq101", "gtja191"))} == {
        "wq101_alpha048", "wq101_alpha056", "wq101_alpha067", "wq101_alpha090",
        "wq101_alpha100", "gtja191_alpha030", "gtja191_alpha165", "gtja191_alpha183",
    }
    mixed = [r for r in rows if r["source_module"] == "factors.library.cross_frequency"]
    assert len(mixed) == 16
    assert all(r["blocked_reason"] and r["input_bar_frequency"] == "unresolved" for r in mixed)


def test_registration_context_restores_registry_and_specs_on_exception():
    source = "return_5d_z"
    row = _row(source)
    registry = list_registered("factor")["factor"]
    original_class = registry[source]
    original_spec = copy.deepcopy(SPEC_BY_SLUG[source])
    with pytest.raises(RuntimeError):
        with ext.registered_external([row]):
            assert row["factor"] in registry
            raise RuntimeError("probe")
    assert registry[source] is original_class
    assert SPEC_BY_SLUG[source] == original_spec
    assert row["factor"] not in registry
    assert row["factor"] not in SPEC_BY_SLUG


class _FakeDaily:
    frequency = "daily"

    def prefetch(self, factors, dates, universe):
        self.prefetch_count = getattr(self, "prefetch_count", 0) + 1

    def get(self, field, dates, universe):
        t = np.arange(len(dates), dtype=float)
        close = 100.0 + t + 0.03 * t * t
        values = np.column_stack((close, close * 1.01 + 2.0))
        if field == "open":
            values = values - 0.5
        elif field == "high":
            values = values + 1.0
        elif field == "low":
            values = values - 1.0
        elif field == "volume":
            values = 1000.0 + 7.0 * t[:, None] + np.array([[0.0, 11.0]])
        return pd.DataFrame(values, index=dates, columns=universe)


def test_daily_spec_alias_uses_existing_batch_and_matches_source(monkeypatch):
    row = _row("return_5d_z")
    alias = row["factor"]
    import factors.spec_factor as spec_factor
    from factors.engine import FactorEngine

    calls = []
    original_batch = spec_factor.compute_spec_factors_batch

    def wrapped(specs, *args, **kwargs):
        calls.append([s["slug"] for s in specs])
        return original_batch(specs, *args, **kwargs)

    monkeypatch.setattr(spec_factor, "compute_spec_factors_batch", wrapped)
    dates = pd.date_range("2024-01-02", periods=40, freq="B")
    universe = pd.Index(["A", "B"])
    data = _FakeDaily()
    source_cls = list_registered("factor")["factor"][row["source_factor"]]
    with ext.registered_external([row]):
        actual = FactorEngine(data).compute_factors([alias], dates, universe)
        expected = source_cls().compute(data, dates, universe)
        pd.testing.assert_frame_equal(actual[alias], expected)
        assert calls == [[alias]]


def test_15min_spec_alias_is_daily_signal_with_explicit_input_metadata():
    row = _row("return_16p_z")
    registry = list_registered("factor")["factor"]
    original_class = registry[row["source_factor"]]
    original_spec = copy.deepcopy(SPEC_BY_SLUG[row["source_factor"]])
    with ext.registered_external([row]):
        alias_class = registry[row["factor"]]
        assert alias_class.frequency == "daily"
        assert alias_class.input_bar_frequency == "15min"
        assert alias_class.signal_frequency == "daily"
        assert SPEC_BY_SLUG[row["source_factor"]] == original_spec
    assert registry[row["source_factor"]] is original_class
    assert row["factor"] not in registry
    assert row["factor"] not in SPEC_BY_SLUG


def test_baseline_guard_detects_modified_protected_file(tmp_path, monkeypatch):
    protected = tmp_path / "protected.txt"
    protected.write_text("before", encoding="utf-8")
    monkeypatch.setattr(ext, "ROOT", tmp_path)
    monkeypatch.setattr(
        ext.subprocess, "check_output", lambda *args, **kwargs: b"protected.txt\0"
    )
    manifest = {"baseline_files": ext.baseline_files()}
    protected.write_text("after", encoding="utf-8")
    with pytest.raises(ValueError, match="protected baseline changed"):
        ext.assert_baseline(manifest)


def test_admit_failure_does_not_leave_isolated_library(tmp_path, monkeypatch):
    study = tmp_path / "study"
    study.mkdir()
    protected = tmp_path / "protected.txt"
    protected.write_text("ok", encoding="utf-8")
    source_library = tmp_path / "source-library.json"
    source_library.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ext, "ROOT", tmp_path)
    row = {"factor": "external_daily__probe", "blocked_reason": ""}
    manifest = {
        "baseline_files": {"protected.txt": ext._hash(protected)},
        "adapter_sha256": ext._hash(Path(ext.__file__)),
        "rows": [row],
    }
    contract = study / "external_contract.json"
    contract.write_text(json.dumps(manifest), encoding="utf-8")
    (study / "qualification.json").write_text(
        json.dumps({"contract_sha256": ext._hash(contract), "rows": [{"factor": row["factor"], "eligible": True}]}),
        encoding="utf-8",
    )

    @contextmanager
    def fake_registered(_rows):
        yield [row["factor"]]

    monkeypatch.setattr(ext, "registered_external", fake_registered)
    from types import SimpleNamespace
    import core.config
    import research.effective_factor_library

    monkeypatch.setattr(
        core.config, "load_config",
        lambda _path: SimpleNamespace(factor_library=SimpleNamespace(path="source-library.json")),
    )

    def reject(*args, **kwargs):
        raise ValueError("validation run is not admissible")

    monkeypatch.setattr(research.effective_factor_library, "admit_validation_run", reject)
    with pytest.raises(ValueError, match="not admissible"):
        ext.execute(study, "admit")
    assert not (study / "factor_library" / "library.json").exists()


def test_qualification_reason_accepts_causal_prefix():
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    frame = pd.DataFrame([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], index=dates)
    assert ext.qualification_reason(frame, frame.iloc[:2], dates[0]) == ""


def test_qualification_reason_rejects_prefix_leakage_and_mask_change():
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    frame = pd.DataFrame([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], index=dates)
    leaked = frame.iloc[:2].copy()
    leaked.iloc[0, 0] = 0.5
    assert "prefix" in ext.qualification_reason(frame, leaked, dates[0]).lower()
    masked = frame.iloc[:2].copy()
    masked.iloc[0, 0] = np.nan
    assert ext.qualification_reason(frame, masked, dates[0])


def test_qualification_reason_rejects_constant_signal():
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    frame = pd.DataFrame(1.0, index=dates, columns=["A", "B"])
    assert "variation" in ext.qualification_reason(frame, frame.iloc[:2], dates[0]).lower()


def test_external_history_does_not_change_requested_output_dates():
    from core.registry import register_factor
    from core.interfaces import Factor
    seen = []

    class Probe(Factor):
        name = "external_history_test_source"
        def dependencies(self):
            return ["close"]
        def compute(self, data, dates, universe):
            seen.append(dates[0])
            return pd.DataFrame(1., index=dates, columns=universe)

    class Data:
        def get_calendar(self, start, end):
            return pd.date_range(start, end, freq="B")

    register_factor(Probe.name)(Probe)
    try:
        row = _row(Probe.name)
        row["history_years"] = 6
        dates = pd.date_range("2024-05-01", periods=5, freq="B")
        with ext.registered_external([row]):
            cls = list_registered("factor")["factor"][row["factor"]]
            result = cls().compute(Data(), dates, ["A"])
        assert result.index.equals(dates)
        assert seen[0] < pd.Timestamp("2019-01-01")
    finally:
        list_registered("factor")["factor"].pop(Probe.name)
