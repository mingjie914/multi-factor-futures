from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import run_portfolio_workflow as ide
from backtest.engine import BacktestResult
from core.config import load_config, load_strategy_library
from core.sectors import FRAMEWORK_UNIVERSE


def _write_config(tmp_path, *, approved_period=5, holding_period=5):
    library = tmp_path / "library.json"
    library.write_text(json.dumps({
        "schema_version": 1,
        "factors": [{
            "factor": "factor_a",
            "status": "effective",
            "selected_period": approved_period,
            "approved_periods": [approved_period],
        }],
    }), encoding="utf-8")
    config = tmp_path / "strategy.yaml"
    config.write_text(
        "date_range:\n"
        f"  start: '{ide.COMPARISON_START}'\n"
        "  end: latest_available\n"
        f"universe: {json.dumps(list(FRAMEWORK_UNIVERSE))}\n"
        "factors: [factor_a]\n"
        f"factor_library:\n  path: '{library.as_posix()}'\n"
        "  enforce_portfolio_periods: true\n"
        f"backtest:\n  holding_period: {holding_period}\n",
        encoding="utf-8",
    )
    return config


def _write_catalog(tmp_path, config, *, plot=False):
    catalog = tmp_path / "strategy_library.yaml"
    catalog.write_text(
        "schema_version: 1\n"
        f"effective_factor_library: '{(tmp_path / 'library.json').as_posix()}'\n"
        f"output_root: '{(tmp_path / 'runs').as_posix()}'\n"
        f"plot: {str(plot).lower()}\n"
        "factor_sets:\n"
        "  - id: subset_a\n"
        "    status: active\n"
        "    description: test subset\n"
        "    factors: [factor_a]\n"
        "    selection_context: {purpose: test}\n"
        "strategies:\n"
        "  - id: probe\n"
        "    status: preferred\n"
        "    source: effective_library\n"
        "    factor_set_id: subset_a\n"
        f"    config_path: '{config.as_posix()}'\n"
        "    mode: single\n",
        encoding="utf-8",
    )
    return catalog


def test_ide_strategy_validation_enforces_library_periods(tmp_path, monkeypatch):
    config = _write_config(tmp_path, approved_period=5, holding_period=10)
    catalog = _write_catalog(tmp_path, config)
    monkeypatch.setattr(ide, "CATALOG_PATH", str(catalog))

    try:
        ide._validated_specs()
    except ValueError as exc:
        assert "period 10 not approved" in str(exc)
    else:
        raise AssertionError("IDE workflow accepted an unapproved factor period")


def test_catalog_rejects_unknown_factor_even_before_strategy_use(tmp_path, monkeypatch):
    config = _write_config(tmp_path)
    catalog = _write_catalog(tmp_path, config)
    text = catalog.read_text(encoding="utf-8").replace(
        "factors: [factor_a]", "factors: [factor_a, unknown_factor]"
    )
    catalog.write_text(text, encoding="utf-8")
    monkeypatch.setattr(ide, "CATALOG_PATH", str(catalog))

    try:
        ide._validated_specs()
    except ValueError as exc:
        assert "contains non-effective factors" in str(exc)
    else:
        raise AssertionError("catalog accepted a non-effective factor")


def test_legacy_peer_comparison_scope_is_in_memory_only(tmp_path, monkeypatch):
    config = _write_config(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            f"start: '{ide.COMPARISON_START}'", "start: '2017-01-01'"
        ),
        encoding="utf-8",
    )
    catalog = _write_catalog(tmp_path, config)
    text = catalog.read_text(encoding="utf-8").replace(
        "source: effective_library", "source: legacy_observation"
    ).replace("factor_set_id: subset_a", "factor_set_id: ''")
    catalog.write_text(text, encoding="utf-8")
    monkeypatch.setattr(ide, "CATALOG_PATH", str(catalog))

    _catalog_path, _loaded, specs = ide._validated_specs()
    strategy, _path, resolved = specs[0]
    assert strategy.source == "legacy_observation"
    assert resolved.date_range.start == ide.COMPARISON_START
    assert not any(step.type == "fillna" for step in resolved.processing)
    assert "2017-01-01" in config.read_text(encoding="utf-8")


def test_catalog_allows_only_one_preferred_strategy(tmp_path):
    config = _write_config(tmp_path)
    catalog = _write_catalog(tmp_path, config)
    text = catalog.read_text(encoding="utf-8")
    duplicate = text[text.index("  - id: probe\n"):].replace(
        "  - id: probe\n", "  - id: probe_2\n", 1
    )
    catalog.write_text(text + duplicate, encoding="utf-8")

    try:
        load_strategy_library(catalog)
    except ValueError as exc:
        assert "multiple preferred strategies" in str(exc)
    else:
        raise AssertionError("catalog accepted multiple preferred strategies")


def test_config_kinds_cannot_be_routed_through_the_wrong_loader():
    root = Path(__file__).resolve().parents[1]
    wrong_pairs = (
        (load_config, root / "config" / "strategy_library.yaml"),
        (load_config, root / "config" / "target_publication.yaml"),
        (load_strategy_library, root / "config" / "default.yaml"),
    )
    for loader, path in wrong_pairs:
        try:
            loader(path)
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"{path.name} was accepted by the wrong config loader")


def test_ide_comparison_persists_results_and_contract(tmp_path, monkeypatch):
    config = _write_config(tmp_path)
    catalog = _write_catalog(tmp_path, config)
    monkeypatch.setattr(ide, "CATALOG_PATH", str(catalog))
    monkeypatch.setattr(ide, "RUN_ID", "run-1")

    class FakeRunner:
        def __init__(self, config):
            self.config = config
            self.config.date_range.end = "2026-08-24"

        def run_full_pipeline(self):
            dates = pd.date_range("2026-08-20", periods=3)
            return BacktestResult(
                nav=pd.Series([1.0, 1.01, 1.02], index=dates),
                weights_history=pd.DataFrame(),
                metrics={"sharpe": 1.0},
            )

    monkeypatch.setattr(ide, "PipelineRunner", FakeRunner)
    output = ide.run_and_compare()

    assert (output / "probe" / "metrics.json").is_file()
    assert (output / "comparison.csv").is_file()
    assert (output / "nav_comparison.csv").is_file()
    assert (output / "run_contract.json").is_file()
    assert (output / "performance.json").is_file()
    contract = json.loads((output / "run_contract.json").read_text(encoding="utf-8"))
    assert contract["status"] == "complete"
    assert contract["strategy_library"]["snapshot"]["strategies"][0]["status"] == "preferred"
    assert contract["artifacts"]["performance.json"]["sha256"]


def test_all_strategy_branch_selects_current_and_archived_peers(monkeypatch):
    monkeypatch.setattr(
        ide, "WORKFLOW", ide.PortfolioWorkflow.RUN_AND_COMPARE_ALL
    )
    _catalog_path, _catalog, specs = ide._validated_specs()
    assert [strategy.id for strategy, _path, _config in specs] == list(
        ide.ALL_STRATEGY_IDS
    )


def test_shared_production_panel_computes_union_once(monkeypatch):
    calls = []

    class FakePanelRunner:
        def __init__(self, factors, **kwargs):
            calls.append((list(factors), kwargs))
            self.raw_ranks = {name: object() for name in factors}

    monkeypatch.setattr(
        "research.portfolio_experiment_support.FactorPanelRunner",
        FakePanelRunner,
    )
    monkeypatch.setattr(
        "research.portfolio_experiment_support.latest_local_date",
        lambda: "2026-08-28",
    )
    specs = [
        (object(), Path("a.yaml"), SimpleNamespace(
            factors=["factor_a", "shared"],
            date_range=SimpleNamespace(end="latest_available"),
        )),
        (object(), Path("b.yaml"), SimpleNamespace(
            factors=["shared", "factor_b"],
            date_range=SimpleNamespace(end="2026-08-20"),
        )),
    ]

    runner = ide._build_shared_production_panel(specs)

    assert list(runner.raw_ranks) == ["factor_a", "shared", "factor_b"]
    assert len(calls) == 1
    assert calls[0][1]["end"] == pd.Timestamp("2026-08-28")


def test_common_h5_branch_routes_without_catalog_admission(monkeypatch):
    called = {}

    def fake_compare(**kwargs):
        called.update(kwargs)
        return Path("runs/portfolio_backtest/common_h5_probe")

    monkeypatch.setattr(
        ide, "WORKFLOW", ide.PortfolioWorkflow.RUN_AND_COMPARE_COMMON_H5_MATCHED
    )
    monkeypatch.setattr(ide, "run_common_h5_compare", fake_compare)

    ide.main()

    assert called == {"ic_horizon": 5}
