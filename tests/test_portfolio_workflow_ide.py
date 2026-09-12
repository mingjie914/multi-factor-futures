from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import run_portfolio_workflow as ide
from backtest.engine import BacktestResult
from core.config import load_config, load_strategy_library
from core.sectors import FRAMEWORK_UNIVERSE


def test_frozen_json_definition_loads_without_executing_python(tmp_path):
    definition = tmp_path / "members.json"
    definition.write_text(json.dumps({"FACTORS": {"a": 1, "b": -1}}), encoding="utf-8")
    loaded = ide._load_factor_definition(definition)
    assert loaded["factors"] == ["a", "b"]
    assert loaded["directions"] == {"a": 1, "b": -1}
    assert len(loaded["sha256"]) == 64


def test_shipped_legacy_definitions_are_portable():
    root = Path(ide.__file__).resolve().parent
    catalog = load_strategy_library(root / "config/strategy_library.yaml")
    for entry in catalog.strategies:
        if entry.factor_definition_path:
            definition = root / entry.factor_definition_path
            assert definition.is_relative_to(root / "config/factor_sets")
            loaded = ide._load_factor_definition(definition)
            assert set(loaded["directions"].values()) <= {-1, 1}
    assert len([s for s in catalog.strategies if s.status != "archived"]) == 13
    assert [s.id for s in catalog.strategies if s.status == "preferred"] == ["multi_source_balanced"]


def test_comparison_plot_uses_actual_dates_and_nine_distinct_colors(tmp_path, monkeypatch):
    import matplotlib.figure
    dates = pd.date_range("2016-03-31", periods=3)
    nav = pd.DataFrame({f"候选{i}": [1., 1.01, 1.02+i*.001] for i in range(9)}, index=dates)
    captured = {}
    original = matplotlib.figure.Figure.savefig
    def save(fig, *args, **kwargs):
        captured["title"] = fig.axes[0].get_title()
        captured["colors"] = [line.get_color() for line in fig.axes[0].lines]
        return original(fig, *args, **kwargs)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", save)
    ide._write_comparison_plot(tmp_path, nav, [{"strategy": n} for n in nav])
    assert "2016-03-31至2016-04-02" in captured["title"]
    assert "最新" not in captured["title"]
    assert len(set(captured["colors"])) == 9
    assert (tmp_path / "nav_comparison.png").stat().st_size > 10000


def _write_config(tmp_path, *, approved_period=5, holding_period=5):
    library = tmp_path / "library.json"
    library.write_text(json.dumps({
        "schema_version": 3,
        "factors": [{
            "factor": "factor_a",
            "status": "effective",
            "best_period": approved_period,
            "signal_frequency": "daily",
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
        "  enforce_effective_membership: true\n"
        f"backtest:\n  holding_period: {holding_period}\n",
        encoding="utf-8",
    )
    return config


def test_comparison_plot_marks_forward_observation(tmp_path, monkeypatch):
    import matplotlib.figure
    dates = pd.to_datetime(["2026-05-15", "2026-05-18", "2026-05-19"])
    nav = pd.DataFrame({"候选1": [1., 1.01, 1.02]}, index=dates)
    captured = {}
    original = matplotlib.figure.Figure.savefig
    def save(fig, *args, **kwargs):
        captured["labels"] = fig.axes[0].get_legend_handles_labels()[1]
        return original(fig, *args, **kwargs)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", save)
    ide._write_comparison_plot(tmp_path, nav, [{"strategy": "候选1"}], cutoff=dates[0])
    assert "研究截止日 2026-05-15" in captured["labels"]
    assert "模拟实盘观察期（截止日之后）" in captured["labels"]


def test_comparison_default_start_inherits_framework():
    assert ide.COMPARISON_START == load_config("config/default.yaml").date_range.start


def test_segment_report_does_not_hardcode_recipe_parameters(tmp_path):
    dates = pd.bdate_range("2026-05-14", periods=5)
    strategy = SimpleNamespace(id="probe", status="observing", factor_set_id="probe")
    combined = SimpleNamespace(nav=pd.Series([1., 1.01, 1.02, 1.01, 1.03], index=dates))
    for production, horizon in ((True, 1), (True, 5), (False, 1)):
        ide._write_segment_report(
            tmp_path, [(strategy, combined, None)], pd.Timestamp("2026-05-15"),
            production_method_compare=production, ic_horizon=horizon,
        )
        report = (tmp_path / "portfolio_report.md").read_text(encoding="utf-8")
        assert "参数以运行合同为准" in report
        assert "ICIR + Top10" not in report
        assert "总敞口2" not in report


def test_comparison_streams_every_background_batch(tmp_path):
    from PIL import Image
    dates = pd.bdate_range("2026-05-14", periods=4)
    nav = pd.DataFrame({"基准": [1.0, 1.02, 1.01, 1.04]}, index=dates)
    consumed = []
    def batches():
        for index in range(3):
            consumed.append(index)
            yield nav * (1.0 + index * .01)
    ide._write_comparison_plot(
        tmp_path, nav, [{"strategy": "基准"}], cutoff=pd.Timestamp("2026-05-15"),
        background_batches=batches(), background_limits=(1.0, 1.1))
    assert consumed == [0, 1, 2]
    with Image.open(tmp_path / "nav_comparison.png") as rendered:
        assert rendered.size == (2250, 1500)


def test_saved_thirteen_candidates_and_default_retain_frozen_members(monkeypatch):
    monkeypatch.setattr(ide, "CATALOG_PATH", "config/strategy_library.yaml")
    monkeypatch.setattr(ide, "STRATEGY_IDS", ())
    monkeypatch.setattr(ide, "WORKFLOW", ide.PortfolioWorkflow.RUN_PREFERRED)
    _, catalog, selected = ide._validated_specs()
    assert len(selected) == 1
    assert selected[0][0].id == "multi_source_balanced"
    assert selected[0][0].name == "多源稳衡"
    assert len(selected[0][2].factors) == 17
    assert selected[0][2].factor_library.enforce_effective_membership
    monkeypatch.setattr(ide, "WORKFLOW", ide.PortfolioWorkflow.RUN_AND_COMPARE)
    _, _, peers = ide._validated_specs()
    assert len(peers) == 13
    assert len({s.name for s, _, _ in peers}) == 13
    assert sum(s.status == "observing" for s, _, _ in peers) == 12
    new_sets = {
        "sector_volume_position": 15, "seat_price_structure": 15,
        "compact_sector": 10, "volatility_follow": 16, "structure_fusion": 20,
    }
    for strategy, _, config in peers:
        if strategy.id in new_sets:
            assert strategy.status == "observing"
            assert strategy.source == "effective_library"
            assert len(config.factors) == new_sets[strategy.id]
    assert "snapshot_6f_icir" not in {s.id for s, _, _ in peers}
    subsets = {s.id: s for s in catalog.factor_sets}
    for strategy, _, config in peers:
        if strategy.source == "effective_library":
            subset = subsets[strategy.factor_set_id]
            assert config.factors == subset.factors
            assert ide._effective_factor_directions(config, config.factors) == subset.selection_context["directions"]
            assert subset.selection_context["production_approved"] is False
        else:
            definition = ide._load_factor_definition(ide._resolve(strategy.factor_definition_path))
            assert config.factors == definition["factors"]


def test_default_rejects_silent_direction_change(monkeypatch):
    import pytest
    monkeypatch.setattr(ide, "WORKFLOW", ide.PortfolioWorkflow.RUN_PREFERRED)
    monkeypatch.setattr(ide, "STRATEGY_IDS", ())
    monkeypatch.setattr(ide, "_effective_factor_directions", lambda config, factors: {name: 1 for name in factors})
    with pytest.raises(ValueError, match="frozen directions differ"):
        ide._validated_specs()


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


def test_ide_strategy_best_period_does_not_constrain_portfolio_holding(tmp_path, monkeypatch):
    config = _write_config(tmp_path, approved_period=5, holding_period=10)
    catalog = _write_catalog(tmp_path, config)
    monkeypatch.setattr(ide, "CATALOG_PATH", str(catalog))

    specs = ide._validated_specs()
    assert len(specs[2]) == 1


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
            f"start: '{ide.COMPARISON_START}'", "start: '2018-01-01'"
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
    assert "2018-01-01" in config.read_text(encoding="utf-8")


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
    monkeypatch.setattr(ide, "WORKFLOW", ide.PortfolioWorkflow.RUN_AND_COMPARE_CONFIGURED)
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
    performance = json.loads(
        (output / "performance.json").read_text(encoding="utf-8")
    )
    assert "process_peak_working_set_mib" in performance


def test_all_strategy_branch_selects_thirteen_active_peers_not_retired_six(monkeypatch):
    monkeypatch.setattr(
        ide, "WORKFLOW", ide.PortfolioWorkflow.RUN_AND_COMPARE_ALL
    )
    _catalog_path, catalog, specs = ide._validated_specs()
    assert [strategy.id for strategy, _path, _config in specs] == [
        entry.id for entry in catalog.strategies if entry.status != "archived"
    ]
    assert len(specs) == 13
    assert all(strategy.id != "snapshot_6f_icir" for strategy, _, _ in specs)


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


def test_common_h5_combination_compare_is_explicit_and_horizon_matched(monkeypatch):
    called = {}

    def fake_compare(**kwargs):
        called.update(kwargs)
        return Path("runs/portfolio_backtest/common_h5_combination_probe")

    monkeypatch.setattr(
        ide, "WORKFLOW", ide.PortfolioWorkflow.RUN_AND_COMPARE_COMMON_H5_SEARCH
    )
    monkeypatch.setattr(ide, "run_common_h5_compare", fake_compare)
    monkeypatch.setattr(ide, "COMMON_H5_SEARCH_COMPARISON_RUN_ID", "probe")
    monkeypatch.setattr(ide, "COMMON_H5_SEARCH_IC_HORIZON", 5)

    ide.main()

    assert called == {
        "ic_horizon": 5,
        "run_id_override": "probe",
        "factor_search_run_dir": ide.COMMON_H5_FACTOR_SEARCH_RUN_DIR,
    }
