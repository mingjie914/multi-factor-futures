from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import run_portfolio_workflow as ide
from backtest.engine import BacktestResult
from core.config import (
    ProductionPortfolioConfig,
    StrategyLibraryEntry,
    load_config,
    load_strategy_library,
)
from core.sectors import FRAMEWORK_UNIVERSE

RETAINED_BUFFERS = {
    "compact_sector": 1, "multi_source_resilient": 1,
    "price_volume_balanced": 0, "curve_essence": 1,
    "curve_volume_balanced": 2, "curve_position": 0,
    "sector_quality_balanced": 0, "structure_quality_balanced": 0,
    "compact_quality_balanced": 0, "compact_liquidity_stable": 1,
}
SLEEVE_CANDIDATES = {"curve_essence_sleeve_equal", "curve_essence_sleeve_erc"}


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
    assert len([s for s in catalog.strategies if s.status != "archived" and s.comparison_group == "neutral_futures"]) == 12
    historical = {"current_single_baseline", "snapshot_8f_icir", "snapshot_13f_icir"}
    for entry in catalog.strategies:
        if entry.id in historical:
            assert entry.status == "archived"
            assert entry.name.endswith("*")
    assert [s.id for s in catalog.strategies if s.status == "preferred"] == ["multi_source_resilient"]


def test_retained_catalog_has_frozen_members_and_adopted_buffers():
    catalog = load_strategy_library("config/strategy_library.yaml")
    sets = {s.id: s for s in catalog.factor_sets}
    expected = {
        "compact_sector": 10, "multi_source_resilient": 18,
        "price_volume_balanced": 16, "curve_essence": 9,
        "curve_volume_balanced": 8, "curve_position": 11,
        "sector_quality_balanced": 16, "structure_quality_balanced": 21,
        "compact_quality_balanced": 11, "compact_liquidity_stable": 10,
    }
    active = [s for s in catalog.strategies if s.status != "archived" and s.comparison_group == "neutral_futures"]
    assert {s.id for s in active} == set(expected) | SLEEVE_CANDIDATES
    active = [s for s in active if s.id in expected]
    assert [s.id for s in active if s.formal] == ["multi_source_resilient"]
    library = json.loads(Path(catalog.effective_factor_library).read_text(encoding="utf-8"))
    effective = {r["factor"]: r for r in library["factors"] if r["status"] == "effective"}
    union = set()
    for strategy in active:
        subset = sets[strategy.factor_set_id]
        context = subset.selection_context
        assert subset.status == "active"
        assert strategy.name.endswith(f"({expected[strategy.id]})")
        assert len(subset.factors) == expected[strategy.id]
        assert context["directions"] == {n: effective[n]["direction"] for n in subset.factors}
        assert context["research_cutoff"] == "2026-05-15"
        assert context["production_approved"] is False
        assert len(context["source_results_sha256"]) == 64
        assert strategy.rank_exit_buffer == RETAINED_BUFFERS[strategy.id]
        union.update(subset.factors)
    assert len(union) == 29
    assert load_config("config/default.yaml").production_portfolio.rank_exit_buffer == 0
    assert sets["multi_source_balanced"].status == "archived"
    assert len(sets["multi_source_balanced"].factors) == 17


def test_position_count_research_is_opt_in_and_shares_formal_members(monkeypatch):
    from trading.weights import strategy_contracts

    sid = "multi_source_resilient_p12"
    monkeypatch.setattr(ide, "WORKFLOW", ide.PortfolioWorkflow.RUN_AND_COMPARE)
    monkeypatch.setattr(ide, "STRATEGY_IDS", ())
    _, _, peers = ide._validated_specs()
    assert {s.id for s, _, _ in peers} == set(RETAINED_BUFFERS) | SLEEVE_CANDIDATES
    parent = next(c for s, _, c in peers if s.id == "multi_source_resilient")
    monkeypatch.setattr(ide, "STRATEGY_IDS", (sid,))
    _, catalog, selected = ide._validated_specs()
    assert len(selected) == 1
    entry, _, config = selected[0]
    assert entry.name == "多源精择12品种(18)"
    assert entry.status == "observing" and not entry.formal
    assert entry.comparison_group == "neutral_position_count_research"
    assert entry.factor_set_id == "multi_source_resilient"
    assert sid not in {s.id for s in catalog.factor_sets}
    assert config.factors == parent.factors and len(config.factors) == 18
    assert ide._effective_factor_directions(config, config.factors) == ide._effective_factor_directions(parent, parent.factors)
    assert config.production_portfolio.top_n_per_side == 6
    assert config.production_portfolio.rank_exit_buffer == 1
    expected = parent.production_portfolio.model_dump()
    expected["top_n_per_side"] = 6
    assert config.production_portfolio.model_dump() == expected
    assert ide._legacy_recipe(config).top_n == 6
    # Explicit contract resolution must not silently recover the parent's 20 names.
    resolved = strategy_contracts([sid])[0]
    expected.pop("factor_sleeves")
    assert resolved["recipe"] == expected and not resolved["formal"]
    assert resolved["directions"] == ide._effective_factor_directions(config, config.factors)
    formal = strategy_contracts(["@formal"])[0]
    assert formal["catalog_strategy_id"] == "multi_source_resilient"
    assert formal["recipe"]["top_n_per_side"] == 10
    assert formal["recipe"]["rank_exit_buffer"] == 1
    monkeypatch.setattr(ide, "STRATEGY_IDS", ())
    monkeypatch.setattr(ide, "WORKFLOW", ide.PortfolioWorkflow.RUN_PREFERRED)
    assert [s.id for s, _, _ in ide._validated_specs()[2]] == ["multi_source_resilient"]
    assert load_config("config/default.yaml").production_portfolio.top_n_per_side == 10


@pytest.mark.parametrize("value", [0, -1, True, False, 6.0, "6"])
def test_catalog_position_count_requires_a_positive_integer(value):
    with pytest.raises(ValueError):
        StrategyLibraryEntry(id="probe", config_path="config/default.yaml", top_n_per_side=value)


@pytest.mark.parametrize("override,expected", [(None, 10), (6, 6)])
def test_catalog_position_count_override_preserves_other_parameters(override, expected):
    base = ProductionPortfolioConfig()
    resolved = base.model_copy(deep=True)
    entry = StrategyLibraryEntry(id="probe", config_path="config/default.yaml", top_n_per_side=override, rank_exit_buffer=1)
    entry.apply_portfolio_overrides(resolved)
    assert resolved.top_n_per_side == expected and resolved.rank_exit_buffer == 1
    original = base.model_dump()
    original.update(top_n_per_side=expected, rank_exit_buffer=1)
    assert resolved.model_dump() == original
    assert base.top_n_per_side == 10 and base.rank_exit_buffer == 0


def test_replaced_strategy_remains_available_only_for_explicit_audit(monkeypatch):
    monkeypatch.setattr(ide, "WORKFLOW", ide.PortfolioWorkflow.RUN_AND_COMPARE_SNAPSHOT_AUDIT)
    monkeypatch.setattr(ide, "SNAPSHOT_AUDIT_IDS", ("multi_source_balanced",))
    _, _, selected = ide._validated_specs()
    assert len(selected) == 1
    strategy, _, config = selected[0]
    assert strategy.id == "multi_source_balanced"
    assert strategy.status == "archived"
    assert len(config.factors) == 17
    assert config.production_portfolio.rank_exit_buffer == 0


@pytest.mark.parametrize("count", [9, 20])
def test_comparison_plot_uses_actual_dates_and_distinct_styles(tmp_path, monkeypatch, count):
    import matplotlib.figure
    dates = pd.date_range("2016-03-31", periods=3)
    nav = pd.DataFrame({f"候选{i}": [1., 1.01, 1.02+i*.001] for i in range(count)}, index=dates)
    captured = {}
    original = matplotlib.figure.Figure.savefig
    def save(fig, *args, **kwargs):
        captured["title"] = fig.axes[0].get_title()
        captured["colors"] = [line.get_color() for line in fig.axes[0].lines]
        captured["styles"] = [(line.get_color(), line.get_linestyle()) for line in fig.axes[0].lines]
        captured["table_text"] = [cell.get_text().get_text() for ax in fig.axes
                                  for table in ax.tables for cell in table.get_celld().values()]
        return original(fig, *args, **kwargs)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", save)
    ide._write_comparison_plot(tmp_path, nav, [{"strategy": n} for n in nav])
    assert "2016-03-31至2016-04-02" in captured["title"]
    assert "最新" not in captured["title"]
    assert len(set(captured["colors"])) == min(count, 13)
    assert len(set(captured["styles"])) == count
    assert all(style == "-" for _, style in captured["styles"][:13])
    assert "年化换手(倍/年)" in captured["table_text"]
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


def test_rank_exit_buffer_defaults_are_distinct_by_config_scope():
    config = load_config("config/default.yaml")
    assert config.production_portfolio.rank_exit_buffer == 0

    entry = StrategyLibraryEntry(id="probe", config_path="strategy.yaml")
    assert entry.rank_exit_buffer is None


@pytest.mark.parametrize("value", [0, 3])
def test_rank_exit_buffer_accepts_nonnegative_integers(value):
    assert ProductionPortfolioConfig(rank_exit_buffer=value).rank_exit_buffer == value
    entry = StrategyLibraryEntry(
        id="probe", config_path="strategy.yaml", rank_exit_buffer=value
    )
    assert entry.rank_exit_buffer == value


@pytest.mark.parametrize("value", [True, False, 0.0, 1.0, -1, "0", "1"])
def test_rank_exit_buffer_rejects_non_strict_values(value):
    with pytest.raises(ValueError):
        ProductionPortfolioConfig(rank_exit_buffer=value)
    with pytest.raises(ValueError):
        StrategyLibraryEntry(
            id="probe", config_path="strategy.yaml", rank_exit_buffer=value
        )


def test_segment_metrics_include_first_post_cutoff_return_and_trade():
    import pytest
    dates = pd.bdate_range("2026-05-14", periods=4)
    nav = pd.Series([1., 1., 1.1, 1.1], index=dates)
    turnover = pd.Series([0., 2., 4., 0.], index=dates)
    rows = ide._segment_rows("probe", "observing", "probe", nav, turnover, dates[1])
    full, before, after = rows
    assert after["total_return"] == pytest.approx(.1)
    assert after["total_turnover"] == 4.
    assert after["annualized_turnover"] == 504.
    assert after["anchor_date"] == "2026-05-15"
    assert after["return_intervals"] == 2
    assert full["total_turnover"] == before["total_turnover"] + after["total_turnover"]
    single = ide._segment_rows("probe", "observing", "probe", nav.iloc[:3], turnover.iloc[:3], dates[1])
    assert single[-1]["total_return"] == pytest.approx(.1)


def test_segment_report_does_not_hardcode_recipe_parameters(tmp_path):
    dates = pd.bdate_range("2026-05-14", periods=5)
    strategy = SimpleNamespace(id="probe", status="observing", factor_set_id="probe")
    combined = SimpleNamespace(nav=pd.Series([1., 1.01, 1.02, 1.01, 1.03], index=dates))
    config = load_config("config/default.yaml")
    config.production_portfolio.rank_exit_buffer = 1
    for production, horizon in ((True, 1), (True, 5), (False, 1)):
        ide._write_segment_report(
            tmp_path, [(strategy, combined, config)], pd.Timestamp("2026-05-15"),
            production_method_compare=production, ic_horizon=horizon,
        )
        report = (tmp_path / "portfolio_report.md").read_text(encoding="utf-8")
        assert "参数以运行合同为准" in report
        assert "probe=B1" in report
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


def test_saved_ten_candidates_and_default_retain_frozen_members(monkeypatch):
    monkeypatch.setattr(ide, "CATALOG_PATH", "config/strategy_library.yaml")
    monkeypatch.setattr(ide, "STRATEGY_IDS", ())
    monkeypatch.setattr(ide, "WORKFLOW", ide.PortfolioWorkflow.RUN_PREFERRED)
    _, catalog, selected = ide._validated_specs()
    assert len(selected) == 1
    assert selected[0][0].id == "multi_source_resilient"
    assert selected[0][0].name == "多源韧衡(18)"
    assert len(selected[0][2].factors) == 18
    assert selected[0][0].formal is True
    assert selected[0][2].production_portfolio.rank_exit_buffer == 1
    assert selected[0][2].factor_library.enforce_effective_membership
    monkeypatch.setattr(ide, "WORKFLOW", ide.PortfolioWorkflow.RUN_AND_COMPARE)
    _, _, peers = ide._validated_specs()
    assert len(peers) == 12
    assert len({s.name for s, _, _ in peers}) == 12
    assert sum(s.status == "observing" for s, _, _ in peers) == 11
    assert all(s.source == "effective_library" for s, _, _ in peers)
    assert "snapshot_6f_icir" not in {s.id for s, _, _ in peers}
    subsets = {s.id: s for s in catalog.factor_sets}
    for strategy, _, config in peers:
        if strategy.id in SLEEVE_CANDIDATES:
            continue  # Independently tested native factor-sleeve recipes.
        assert config.production_portfolio.rank_exit_buffer == RETAINED_BUFFERS[strategy.id]
        recipe = ide._legacy_recipe(config).to_dict()
        assert recipe.pop("rank_exit_buffer") == RETAINED_BUFFERS[strategy.id]
        base = ide._legacy_recipe(load_config("config/default.yaml")).to_dict()
        base.pop("rank_exit_buffer")
        recipe.pop("name")
        base.pop("name")
        assert recipe == base
        assert ide._strategy_label(strategy.id).endswith(f"[B{RETAINED_BUFFERS[strategy.id]}]")
        if strategy.source == "effective_library":
            subset = subsets[strategy.factor_set_id]
            assert strategy.name.endswith(f"({len(subset.factors)})")
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


@pytest.mark.parametrize("global_value,override,expected", [(0, None, 0), (2, None, 2), (2, 0, 0), (0, 1, 1)])
def test_strategy_rank_buffer_resolves_once_without_rewriting_config(tmp_path, monkeypatch, global_value, override, expected):
    path = _write_config(tmp_path)
    catalog_path = _write_catalog(tmp_path, path)
    catalog = load_strategy_library(catalog_path)
    catalog.strategies[0].rank_exit_buffer = override
    cfg = load_config(path)
    cfg.production_portfolio.rank_exit_buffer = global_value
    monkeypatch.setattr(ide, "load_strategy_library", lambda _: catalog)
    monkeypatch.setattr(ide, "load_config", lambda _: cfg.model_copy(deep=True))
    monkeypatch.setattr(ide, "WORKFLOW", ide.PortfolioWorkflow.RUN_PREFERRED)
    _, _, specs = ide._validated_specs()
    resolved = specs[0][2]
    assert resolved.production_portfolio.rank_exit_buffer == expected
    assert ide._legacy_recipe(resolved).rank_exit_buffer == expected
    assert cfg.production_portfolio.rank_exit_buffer == global_value


def test_configured_comparison_rejects_rank_buffer_route_change(tmp_path, monkeypatch):
    path = _write_config(tmp_path)
    catalog_path = _write_catalog(tmp_path, path)
    catalog = load_strategy_library(catalog_path)
    catalog.strategies[0].rank_exit_buffer = 1
    monkeypatch.setattr(ide, "load_strategy_library", lambda _: catalog)
    monkeypatch.setattr(ide, "WORKFLOW", ide.PortfolioWorkflow.RUN_AND_COMPARE_CONFIGURED)
    with pytest.raises(ValueError, match="rank_exit_buffer|production_portfolio"):
        ide._validated_specs()


@pytest.mark.parametrize("value", [0, 1, "true", "false", None])
def test_formal_designation_requires_explicit_boolean(value):
    with pytest.raises(ValueError):
        StrategyLibraryEntry(id="probe", config_path="strategy.yaml", formal=value)


def test_formal_designation_is_not_publication_approval(tmp_path):
    import yaml
    assert StrategyLibraryEntry(id="probe", config_path="strategy.yaml").formal is False
    assert yaml.safe_load(Path("config/target_publication.yaml").read_text(encoding="utf-8"))["enabled"] is False
    config = _write_config(tmp_path)
    path = _write_catalog(tmp_path, config)
    path.write_text(path.read_text(encoding="utf-8").replace(
        "status: preferred", "status: archived\n    formal: true"), encoding="utf-8")
    with pytest.raises(ValueError, match="formal strategy"):
        load_strategy_library(path)


def test_native_comparison_still_rejects_other_recipe_changes(monkeypatch):
    original = ide.load_config
    def changed(path):
        config = original(path)
        config.production_portfolio.top_n_per_side = 5
        return config
    # Freeze the true common recipe before simulating a differing strategy YAML.
    default = original("config/default.yaml")
    monkeypatch.setattr(ide, "_default_production_config", lambda: default)
    monkeypatch.setattr(ide, "load_config", changed)
    monkeypatch.setattr(ide, "WORKFLOW", ide.PortfolioWorkflow.RUN_AND_COMPARE)
    monkeypatch.setattr(ide, "STRATEGY_IDS", ())
    with pytest.raises(ValueError, match="overrides production_portfolio"):
        ide._validated_specs()


def test_rank_buffer_rejects_other_buffer_layers_and_generic_pipeline():
    from core.config import validate_rank_buffer_route
    cfg = load_config("config/default.yaml")
    cfg.production_portfolio.rank_exit_buffer = 1
    validate_rank_buffer_route(cfg, actual_holdings_supported=True)
    with pytest.raises(ValueError, match="actual holdings"):
        validate_rank_buffer_route(cfg, actual_holdings_supported=False)
    for field in ("asset_selection", "universe_selection"):
        getattr(cfg, field).enabled = True
        with pytest.raises(ValueError, match="cannot combine"):
            validate_rank_buffer_route(cfg, actual_holdings_supported=True)
        getattr(cfg, field).enabled = False


@pytest.mark.parametrize("buffer", [0, 1, 2])
def test_native_portfolio_exports_actual_ledger_and_diagnoses_costs_once(tmp_path, monkeypatch, buffer):
    import numpy as np
    import optimization.costs as costs
    import research.historical_portfolio_search as research
    import research.portfolio_experiment_support as support
    from backtest.research_ledger import build_close_marked_ledger
    dates = pd.bdate_range(ide.COMPARISON_START, periods=3)
    targets = pd.DataFrame({"AU": [1., .8, .6], "CU": [-1., -.8, -.6]}, index=dates)
    returns = pd.DataFrame({"AU": [0., .01, -.02], "CU": [0., -.01, .01]}, index=dates)
    native = build_close_marked_ledger(targets, returns, trade_cost_rate=.0002,
        decision_target=(lambda date, desired, actual: desired * .5) if buffer else None)
    native.metadata = {**native.metadata, "rank_buffer_diagnostics": {"decisions": 2 if buffer else 0}}
    cfg = load_config("config/default.yaml")
    cfg.factors = ["probe"]
    cfg.production_portfolio.rank_exit_buffer = buffer
    panel = SimpleNamespace(cal=dates, daily_ret=returns, u=list(returns),
                            env=SimpleNamespace(sector_of={}), get_contract_schedule=lambda: None)
    panel.for_factors = lambda *args, **kwargs: panel
    class Evaluator:
        def __init__(self, *args, **kwargs):
            pass
        def run(self, factors, recipe):
            assert factors == ["probe"] and recipe.rank_exit_buffer == buffer
            return native
    monkeypatch.setattr(research, "PortfolioEvaluator", Evaluator)
    monkeypatch.setattr(research, "CausalEligibilityEnvironment", lambda *args: SimpleNamespace(sector_of={}))
    monkeypatch.setattr(support, "latest_local_date", lambda: dates[-1])
    calls = []
    original = costs.evaluate_cost_sensitivity
    def diagnose(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(costs, "evaluate_cost_sensitivity", diagnose)
    result, _ = ide._run_production_portfolio(cfg, tmp_path, factor_directions={"probe": 1},
        factor_direction_source={"fixture": True}, panel_runner_override=panel)
    assert result.research_ledger is native
    assert result.weights_history is native.target_weights
    assert len(calls) == 1
    for filename, frame in {
        "weights.csv": native.target_weights,
        "research_target_weights.csv": native.target_weights,
        "research_effective_weights.csv": native.effective_weights,
        "research_asset_returns.csv": native.asset_returns,
        "research_return_contributions.csv": native.contributions,
        "research_return_ledger.csv": native.daily,
    }.items():
        saved = pd.read_csv(tmp_path / filename, index_col=0, float_precision="round_trip")
        np.testing.assert_array_equal(saved.to_numpy(), frame.to_numpy())


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


def test_all_strategy_branch_explicitly_includes_all_active_groups_not_archives(monkeypatch):
    monkeypatch.setattr(
        ide, "WORKFLOW", ide.PortfolioWorkflow.RUN_AND_COMPARE_ALL
    )
    _catalog_path, catalog, specs = ide._validated_specs()
    assert [strategy.id for strategy, _path, _config in specs] == [
        entry.id for entry in catalog.strategies if entry.status != "archived"
    ]
    assert any(s.comparison_group == "long_only_trend_9" for s, _, _ in specs)
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
            date_range=SimpleNamespace(start=ide.COMPARISON_START, end="latest_available"),
            production_portfolio=ProductionPortfolioConfig(), universe=FRAMEWORK_UNIVERSE,
        )),
        (object(), Path("b.yaml"), SimpleNamespace(
            factors=["shared", "factor_b"],
            date_range=SimpleNamespace(start=ide.COMPARISON_START, end="2026-08-20"),
            production_portfolio=ProductionPortfolioConfig(), universe=FRAMEWORK_UNIVERSE,
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
