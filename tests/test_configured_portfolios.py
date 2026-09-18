"""Configured strategies share selection, search and close-marked accounting."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core.config import load_config, validate_portfolio_config
from core.sectors import FRAMEWORK_UNIVERSE
from optimization.portfolio_construction import allocate_ranked_portfolio
from research.historical_portfolio_search import PortfolioEvaluator, PortfolioRecipe
from research.portfolio_experiment_support import FactorPanelRunner, configured_futures_cost_model


def case():
    config = load_config("config/strategy_trend_allocation.yaml")
    recipe = PortfolioRecipe.from_config(config.production_portfolio)
    dates = pd.bdate_range("2024-01-01", periods=100)
    rng = np.random.default_rng(5)
    u = list(recipe.investment_universe)
    values = {n: pd.DataFrame(rng.normal(size=(100, 9)), index=dates, columns=u)
              for n in ("a", "b", "c")}
    returns = pd.DataFrame(rng.normal(0.0003, .008, (100, 9)), index=dates, columns=u)
    ranks = {n: f.rank(axis=1, pct=True) for n, f in values.items()}
    runner = SimpleNamespace(cal=dates, u=u, factor_values=values, factor_directions=dict.fromkeys(values, 1),
        ranks=ranks, ic=pd.DataFrame(rng.normal(.03, .1, (100, 3)), index=dates, columns=values),
        daily_ret=returns, close_tradable=returns.notna(), contract_schedule=None,
        reference_universe=FRAMEWORK_UNIVERSE, env=SimpleNamespace(sector_of=dict.fromkeys(u, "all")))
    evaluator = PortfolioEvaluator(runner, start=dates[40], end=dates[-1], cost_model=configured_futures_cost_model(config))
    return config, recipe, runner, evaluator


def test_sparse_config_preserves_global_contract_and_can_override(tmp_path):
    default = load_config("config/default.yaml")
    config = load_config("config/strategy_trend_allocation.yaml")
    assert config.universe == default.universe == list(FRAMEWORK_UNIVERSE)
    assert config.production_portfolio.investment_universe == ["IF", "IC", "T", "TF", "AU", "CU", "SC", "RB", "M"]
    assert default.production_portfolio.position_mode == "long_short"
    assert default.production_portfolio.risk_history_timing == "prior_close"
    assert configured_futures_cost_model(default).accounting_method == "close_marked"
    assert config.production_portfolio.risk_history_timing == "decision_close"
    assert configured_futures_cost_model(config).accounting_method == "index_formula"
    profile = Path("config/strategy_trend_allocation.yaml").resolve().as_posix()
    child = tmp_path / "child.yaml"
    child.write_text(f"extends: '{profile}'\nproduction_portfolio:\n  target_volatility: 0.03\n", encoding="utf-8")
    assert load_config(child).production_portfolio.target_volatility == .03
    child.write_text("extends: child.yaml", encoding="utf-8")
    with pytest.raises(ValueError, match="cyclic"):
        load_config(child)


@pytest.mark.parametrize("change", [
    {"investment_universe": ["IF", "IF"]}, {"investment_universe": ["BAD"]},
    {"asset_weight_method": "typo"}, {"position_limits": {"AG": .2}},
    {"selection_fraction": 0.01}, {"volatility_window": 4}, {"rank_exit_buffer": 1},
    {"correlation_multiplier_cap": float("nan")}, {"position_mode": "long_short"},
])
def test_invalid_recipe_is_not_silently_accepted(change):
    config, _, _, _ = case()
    for key, value in change.items():
        setattr(config.production_portfolio, key, value)
    with pytest.raises(ValueError):
        validate_portfolio_config(config)


def test_document_weights_match_independent_sample_formula():
    _, recipe, runner, _ = case()
    row = pd.Series(range(9), index=runner.u, dtype=float)
    history = runner.daily_ret.iloc[:30]
    result = allocate_ranked_portfolio(row, history, recipe)
    selected = row.nlargest(5).index
    data = history[selected].iloc[-20:].to_numpy()
    std = np.std(data, axis=0, ddof=1) * np.sqrt(242)
    correlation = np.corrcoef(data.T)
    rho = correlation[np.triu_indices(5, 1)].mean()
    cf = min(np.sqrt(5 / (1 + 4 * rho)), 2)
    np.testing.assert_allclose(result[selected], .04 / 5 / std * cf, rtol=1e-13)
    assert (result > 0).sum() == 5
    assert result.drop(selected).eq(0).all()
    assert not np.isclose(result.sum(), 1)
    plain = allocate_ranked_portfolio(row, history, replace(recipe, correlation_window=None))
    np.testing.assert_allclose(plain[selected], .04 / 5 / std)


def test_correlation_window_is_independent_and_cf_is_capped():
    _, recipe, runner, _ = case()
    row = pd.Series(range(9), index=runner.u)
    history = runner.daily_ret.iloc[:60].copy()
    altered = history.copy()
    altered.iloc[:40] = np.arange(40)[:, None] * np.arange(1, 10)[None, :]
    long = replace(recipe, correlation_window=60)
    assert not np.allclose(allocate_ranked_portfolio(row, history, long), allocate_ranked_portfolio(row, altered, long))
    np.testing.assert_allclose(allocate_ranked_portfolio(row, history, recipe), allocate_ranked_portfolio(row, altered, recipe))
    capped = allocate_ranked_portfolio(row, history, replace(recipe, correlation_multiplier_cap=1))
    plain = allocate_ranked_portfolio(row, history, replace(recipe, correlation_window=None))
    np.testing.assert_allclose(capped, plain)


def test_parameter_average_then_factor_average_then_caps():
    _, recipe, runner, evaluator = case()
    uncapped = replace(recipe, position_limits=())
    parts = {n: evaluator.weights([n], uncapped) for n in runner.ranks}
    grouped = replace(uncapped, factor_groups=(("alpha", ("a", "b")),))
    expected = ((parts["a"] + parts["b"]) / 2 + parts["c"]) / 2
    pd.testing.assert_frame_equal(evaluator.weights(["a", "b", "c"], grouped), expected)
    flat = evaluator.weights(["a", "b", "c"], replace(grouped, parameter_aggregation="flat"))
    np.testing.assert_allclose(flat, (parts["a"] + parts["b"] + parts["c"]) / 3)
    assert not np.allclose(flat, expected)
    limited = evaluator.weights(["a", "b", "c"], replace(grouped, position_limits=(("IF", .005),)))
    pd.testing.assert_frame_equal(limited.drop(columns="IF"), expected.drop(columns="IF"))
    pd.testing.assert_series_equal(limited["IF"], expected["IF"].clip(upper=.005))
    with pytest.raises(ValueError, match="remain complete"):
        evaluator.weights(["a", "c"], grouped)


@pytest.mark.parametrize("timing", ["prior_close", "decision_close"])
def test_rounding_direction_ties_and_future_causality(timing):
    _, recipe, runner, evaluator = case()
    recipe = replace(recipe, risk_history_timing=timing)
    runner.factor_values["a"].loc[:, :] = 1.0
    runner.factor_values["a"]["TF"] = 1 + 1e-12
    rounded = evaluator.weights(["a"], recipe)
    assert set(rounded.iloc[0][rounded.iloc[0] > 0].index) == set(sorted(runner.u)[:5])
    assert "TF" in evaluator.weights(["a"], replace(recipe, score_decimals=None)).iloc[0].loc[lambda s: s > 0].index
    cutoff = evaluator.dates[20]
    expected = evaluator.weights(["b"], recipe)
    mask = runner.daily_ret.index > cutoff if timing == "decision_close" else runner.daily_ret.index >= cutoff
    runner.daily_ret.loc[mask] *= 100
    runner.factor_values["b"].loc[runner.cal > cutoff] *= -100
    evaluator._factor_portfolio_cache.clear()
    pd.testing.assert_frame_equal(evaluator.weights(["b"], recipe).loc[:cutoff], expected.loc[:cutoff])


def test_close_risk_window_matches_formula_and_changes_only_next_bar_holdings():
    _, recipe, runner, evaluator = case()
    uncapped = replace(recipe, position_limits=())
    date = evaluator.dates[10]
    expected = allocate_ranked_portfolio(runner.factor_values["a"].loc[date].round(10),
        runner.daily_ret.loc[:date].tail(20), uncapped)
    actual = evaluator.weights(["a"], uncapped)
    np.testing.assert_allclose(actual.loc[date], expected)
    old = evaluator.weights(["a"], replace(uncapped, risk_history_timing="prior_close"))
    assert not np.allclose(old.loc[date], actual.loc[date])
    ledger = evaluator.run(["a"], uncapped)
    np.testing.assert_allclose(ledger.effective_weights.iloc[1:], actual.iloc[:-1])
    previous = actual.shift().fillna(0.)
    prices = runner.daily_ret.reindex(actual.index)
    literal = (previous * prices).sum(axis=1) - .0002 * (actual - previous * (1 + prices)).abs().sum(axis=1) - .001/242
    literal.iloc[0] = 0.
    np.testing.assert_allclose(ledger.daily.net_return, literal, atol=1e-15)


def test_cost_option_is_explicit_and_cannot_fall_through_generic_engine():
    from optimization.costs import SimpleFuturesCost
    native = SimpleFuturesCost(annual_roll_cost=0.)
    alternative = SimpleFuturesCost(annual_roll_cost=0., accounting_method="index_formula")
    assert "accounting_method" not in native.ledger_parameters()
    assert alternative.ledger_parameters()["accounting_method"] == "index_formula"
    for kwargs in ({"accounting_method": "typo"}, {"accounting_method": "index_formula"}):
        with pytest.raises(ValueError):
            SimpleFuturesCost(**kwargs)
    weights = pd.Series({"A": .3})
    with pytest.raises(ValueError, match="configured daily research ledger"):
        alternative.estimate_cost(weights, weights, pd.Timestamp("2025-01-01"))
    with pytest.raises(ValueError, match="configured daily research ledger"):
        alternative.estimate_holding_cost(weights, pd.Timestamp("2025-01-01"))


def test_switching_cost_method_cannot_reuse_cached_net_returns():
    _, recipe, _, evaluator = case()
    document = evaluator.ledger(["a"], recipe)
    evaluator.cost_model.accounting_method = "close_marked"
    native = evaluator.ledger(["a"], recipe)
    assert not np.allclose(document.net_return, native.net_return)
    np.testing.assert_allclose(document.gross_return, native.gross_return)
    pd.testing.assert_frame_equal(native, evaluator.run(["a"], recipe).daily.assign(nav=native.nav))


def test_no_pool_shrink_and_missing_after_start_fails():
    _, recipe, runner, evaluator = case()
    history = runner.daily_ret.iloc[:30]
    row = runner.factor_values["a"].iloc[40].copy()
    row.iloc[:5] = np.nan
    assert allocate_ranked_portfolio(row, history, recipe).eq(0).all()
    runner.factor_values["a"].loc[evaluator.dates[10]] = row
    with pytest.raises(RuntimeError, match="after portfolio start"):
        evaluator.weights(["a"], recipe)


@pytest.mark.parametrize("mode", ["long_only", "short_only", "long_short"])
@pytest.mark.parametrize("method", ["equal", "inverse_volatility", "erc"])
def test_existing_allocators_work_with_another_fixed_pool(mode, method):
    _, recipe, runner, _ = case()
    other = ("IF", "IC", "T", "AU")
    changed = replace(recipe, investment_universe=other, position_mode=mode, selection_fraction=.5,
                      asset_weight=method, allocation_scale="fixed_gross", position_limits=())
    result = allocate_ranked_portfolio(runner.factor_values["a"].iloc[40], runner.daily_ret.iloc[:40], changed)
    assert tuple(result.index) == other
    assert result.abs().sum() == pytest.approx(1)
    if mode == "long_only":
        assert result.ge(0).all()
    elif mode == "short_only":
        assert result.le(0).all()
    else:
        assert result.sum() == pytest.approx(0)


def test_shared_ledger_search_and_single_date_agree(tmp_path):
    _, recipe, runner, evaluator = case()
    target = evaluator.weights(["a", "b"], recipe)
    result = evaluator.run(["a", "b"], recipe)
    pd.testing.assert_frame_equal(result.target_weights, target)
    pd.testing.assert_frame_equal(evaluator.ledger(["a", "b"], recipe), evaluator.ledger_from_weights(target))
    date = evaluator.dates[-1]
    pd.testing.assert_series_equal(evaluator.target_for_holdings(["a", "b"], recipe, date, pd.Series(0., index=runner.u)), target.loc[date])
    np.testing.assert_allclose(result.daily["gross_return"].iloc[1], (target.iloc[0] * runner.daily_ret.loc[evaluator.dates[1]]).sum())
    assert result.metadata["periods_per_year"] == 242
    assert result.metadata["factor_reference_universe"] == list(FRAMEWORK_UNIVERSE)
    result.save(tmp_path)
    assert (tmp_path / "research_return_ledger_metadata.json").is_file()


def test_raw_panel_strategy_view_reranks_without_mutating_reference():
    _, _, runner, _ = case()
    panel = object.__new__(FactorPanelRunner)
    panel.__dict__.update(runner.__dict__)
    panel.raw_ranks = {n: f.rank(axis=1, pct=True) for n, f in panel.factor_values.items()}
    panel._ic_returns = runner.daily_ret
    panel._contract_schedule_loaded = True
    panel._contract_schedule = None
    panel.factor_directions["a"] = -1
    view = panel.for_factors(["a"], universe=runner.u[:4], score_decimals=10)
    expected = 1 - panel.factor_values["a"][runner.u[:4]].round(10).rank(axis=1, pct=True)
    pd.testing.assert_frame_equal(view.ranks["a"], expected)
    assert panel.u == runner.u and len(panel.raw_ranks["a"].columns) == 9
    assert view.factor_directions["a"] == -1


def test_search_cache_separates_recipes_and_reuses_same_contract(tmp_path):
    import duckdb
    from workflows.factor_selection import _run_budgeted_pool_search
    _, recipe, runner, evaluator = case()
    kwargs = dict(evaluator=evaluator, portfolio_ic=runner.ic, pool=list(runner.ranks),
                  segments=[(evaluator.start, evaluator.end)], clusters={"a": 1, "b": 2, "c": 3},
                  output=tmp_path, budget=4, beam_width=1, round_width=2, cache_namespace="data-code-v1")
    with duckdb.connect(":memory:") as db:
        first = _run_budgeted_pool_search(recipe=recipe, db=db, **kwargs)
        count = db.execute("SELECT count(*) FROM pool_results").fetchone()[0]
        assert count > 0 and all(r["status"] == "evaluated" for r in first[1])
        assert _run_budgeted_pool_search(recipe=recipe, db=db, **kwargs) == first
        _run_budgeted_pool_search(recipe=replace(recipe, target_volatility=.03), db=db, **kwargs)
        assert db.execute("SELECT count(*) FROM pool_results").fetchone()[0] > count
        count = db.execute("SELECT count(*) FROM pool_results").fetchone()[0]
        evaluator.ic_window = 90
        _run_budgeted_pool_search(recipe=recipe, db=db, **kwargs)
        assert db.execute("SELECT count(*) FROM pool_results").fetchone()[0] > count
        count = db.execute("SELECT count(*) FROM pool_results").fetchone()[0]
        evaluator.cost_model.accounting_method = "close_marked"
        _run_budgeted_pool_search(recipe=recipe, db=db, **kwargs)
        assert db.execute("SELECT count(*) FROM pool_results").fetchone()[0] > count
        count = db.execute("SELECT count(*) FROM pool_results").fetchone()[0]
        _run_budgeted_pool_search(recipe=replace(recipe, risk_history_timing="prior_close"), db=db, **kwargs)
        assert db.execute("SELECT count(*) FROM pool_results").fetchone()[0] > count


def test_portfolio_first_search_uses_net_proxy_and_rejects_coverage_before_pairs(tmp_path):
    import duckdb
    from workflows.factor_selection import _run_budgeted_pool_search, _write_csv, _daily_spearman_ic
    _, recipe, runner, evaluator = case()
    evaluator = evaluator.bounded(runner.cal[40], runner.cal[79])
    runner.factor_values["a"].loc[evaluator.start] = np.nan
    kwargs = dict(pool=list(runner.ranks), recipe=recipe,
                  segments=[(evaluator.start, evaluator.end)], clusters={"a": 1, "b": 2, "c": 3},
                  budget=7, beam_width=2, round_width=4, cache_namespace="frozen")
    def run(folder, ic):
        folder.mkdir()
        with duckdb.connect(":memory:") as db:
            return _run_budgeted_pool_search(evaluator=evaluator, portfolio_ic=ic,
                db=db, output=folder, **kwargs)[1]
    first = run(tmp_path / "first", runner.ic)
    assert first[0]["status"] == "rejected_runtime" and "coverage below 5" in first[0]["error"]
    assert {r["factors"][0] for r in first if r["phase"] == "single_factor_eligibility"} == {"a", "b", "c"}
    assert all("a" not in r["factors"] for r in first if r["factor_count"] > 1)
    assert any(r["factors"] == ["b", "c"] for r in first)
    # Rejected first rows must not truncate the richer success-result schema.
    _write_csv(tmp_path / "results.csv", first)
    assert "full_sharpe" in pd.read_csv(tmp_path / "results.csv")
    # Poison all IC and post-search market/factor data. Neither may drive a
    # portfolio-first/equal proposal or escape into exact selection returns.
    runner.ic.loc[:, :] = -1000
    runner.daily_ret.loc[runner.cal[80]:] = 100
    for frame in runner.factor_values.values():
        frame.loc[runner.cal[80]:] = np.nan
    second = run(tmp_path / "second", runner.ic)
    assert [(r["factors"], r["status"], r.get("full_sharpe")) for r in second] == [
        (r["factors"], r["status"], r.get("full_sharpe")) for r in first]
    frame = runner.ranks["b"].iloc[:3]
    assert _daily_spearman_ic(frame, frame, frame.index).empty
    assert _daily_spearman_ic(frame, frame, frame.index, minimum_cross_section=9).eq(1).all()


def test_configured_selection_freezes_plan_and_reuses_canonical_factor_runner(tmp_path, monkeypatch):
    import workflows.factor_selection as selection
    config = load_config("config/strategy_trend_allocation.yaml")
    library = tmp_path / "library.json"
    library.write_text("{}", encoding="utf-8")
    rows = [dict(factor=n, best_period=1, direction=1, signal_frequency="daily") for n in ("a", "b")]
    monkeypatch.setattr(selection, "_load_library", lambda *a, **k: (library, rows))
    monkeypatch.setattr(selection, "load_config", lambda *a: config)
    for name in ("workflows/factor_selection.py", "research/historical_portfolio_search.py",
                 "optimization/portfolio_construction.py", "backtest/research_ledger.py",
                 "optimization/costs.py", "optimization/factor_weighting.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen", encoding="utf-8")
    monkeypatch.setattr(selection, "__file__", str(tmp_path / "workflows/factor_selection.py"))
    manager = SimpleNamespace(source=SimpleNamespace(checkpoint_source_fingerprint=lambda *a: "data"),
                              get_calendar=lambda a, b: pd.bdate_range(a, b))
    monkeypatch.setattr(selection, "PipelineRunner", lambda **k: SimpleNamespace(data_manager=manager))
    calls = []
    class Panel:
        _source_tree_fingerprint = staticmethod(lambda: "factor-code")
        def __init__(self, names, **kwargs):
            calls.append((names, kwargs))
            raise RuntimeError("panel sentinel")
    monkeypatch.setattr(selection, "FactorPanelRunner", Panel)
    kwargs = dict(run_id="frozen", selection_start="2019-01-01", search_end="2024-12-31")
    for resume in (False, True):
        with pytest.raises(RuntimeError, match="panel sentinel"):
            selection.run_effective_factor_selection(**kwargs, resume=resume)
    assert len(calls) == 2 and calls[0][0] == ["a", "b"]
    assert calls[0][1]["universe"] == list(FRAMEWORK_UNIVERSE)
    assert calls[0][1]["retain_values"] and calls[0][1]["end"] == pd.Timestamp("2026-05-15")
    with pytest.raises(RuntimeError, match="frozen selection plan changed"):
        selection.run_effective_factor_selection(**kwargs, resume=True, portfolio_search_budget=513)
    import factors.numerics as numerics
    monkeypatch.setattr(numerics, "factor_kernel_contract", lambda: {"factor_kernel_mode": "changed-test-runtime"})
    with pytest.raises(RuntimeError, match="frozen selection plan changed"):
        selection.run_effective_factor_selection(**kwargs, resume=True)
    with pytest.raises(ValueError, match="search_end must"):
        selection.run_effective_factor_selection(run_id="invalid", selection_start="2019-01-01", search_end="2026-05-16")
    size_kwargs = dict(run_id="size_frozen", selection_start="2019-01-01", search_end="2024-12-31", search_sizes=(2,))
    with pytest.raises(RuntimeError, match="panel sentinel"):
        selection.run_effective_factor_selection(**size_kwargs)
    with pytest.raises(RuntimeError, match="frozen selection plan changed"):
        selection.run_effective_factor_selection(**size_kwargs, resume=True, search_seed=20260916)


def test_size_search_reserves_large_portfolios_replays_and_excludes_future(tmp_path):
    import duckdb
    from collections import Counter
    from workflows.factor_selection import _run_budgeted_pool_search
    _, recipe, runner, evaluator = case()
    rng = np.random.default_rng(23)
    names = [f"factor_{i}" for i in range(9)]
    runner.factor_values = {n: pd.DataFrame(rng.normal(size=(100, 9)), index=runner.cal, columns=runner.u) for n in names}
    runner.ranks = {n: f.rank(axis=1, pct=True) for n, f in runner.factor_values.items()}
    runner.factor_directions = dict.fromkeys(names, 1)
    runner.ic = pd.DataFrame(rng.normal(size=(100, 9)), index=runner.cal, columns=names)
    evaluator = evaluator.bounded(runner.cal[40], runner.cal[79])
    kwargs = dict(evaluator=evaluator, portfolio_ic=runner.ic, pool=names, recipe=recipe,
                  segments=[(evaluator.start, evaluator.end)], clusters={n: i % 3 for i, n in enumerate(names)},
                  budget=63, search_sizes=(2, 4, 8), initial_candidates=[names, names[:2]],
                  cache_namespace="frozen-size-test", search_seed=17)
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first_dir.mkdir(); second_dir.mkdir()
    with duckdb.connect(":memory:") as db:
        first = _run_budgeted_pool_search(db=db, output=first_dir, **kwargs)
        assert _run_budgeted_pool_search(db=db, output=first_dir, **kwargs) == first
    rows = first[1]
    assert len(rows) == 63 and first[2] == "exact_backtest_budget_exhausted"
    assert all(r["status"] == "evaluated" for r in rows)
    assert Counter(r["search_size_target"] for r in rows if "search_size_target" in r) == {2: 18, 4: 18, 8: 18}
    assert set(range(1, 10)) == {r["factor_count"] for r in rows}
    assert any(r["phase"] == "prior_locked_seed" and r["factor_count"] == 9 for r in rows)
    assert any(r["phase"] == "incumbent_size_seed" and r["factor_count"] > 2 and set(names[:2]) <= set(r["factors"]) for r in rows)
    progress = json.loads((first_dir / "size_search.json").read_text())
    assert progress["attempted_by_size"] == progress["quotas"]
    runner.ic.loc[:, :] = -1000
    runner.daily_ret.loc[runner.cal[80]:] = 100
    for values in runner.factor_values.values():
        values.loc[runner.cal[80]:] = np.nan
    with duckdb.connect(":memory:") as db:
        # Recompute under poisoned later data instead of masking it with cache.
        clean = evaluator.bounded(runner.cal[40], runner.cal[79])
        second = _run_budgeted_pool_search(db=db, output=second_dir, **{**kwargs, "evaluator": clean})
    canonical = lambda rr: [{k: v for k, v in r.items() if k not in ("seconds", "cpu_seconds")} for r in rr]
    assert canonical(rows) == canonical(second[1])


@pytest.mark.parametrize("failure", ["rounded_flat", "long_short_coverage"])
def test_search_rejects_uninformative_representation_and_counts_both_sides(tmp_path, failure):
    import duckdb
    from workflows.factor_selection import _run_budgeted_pool_search
    _, recipe, runner, evaluator = case()
    if failure == "rounded_flat":
        runner.factor_values["a"] *= 1e-14
        assert runner.factor_values["a"].nunique(axis=1).gt(1).all()
        message = "no cross-sectional information after configured rounding"
    else:
        recipe = replace(recipe, position_mode="long_short", selection_fraction=None,
                         top_n=2, asset_weight="equal", allocation_scale="fixed_gross")
        runner.factor_values["a"].loc[evaluator.start, runner.u[3:]] = np.nan
        message = "fixed-pool coverage below 4"
    with duckdb.connect(":memory:") as db:
        _, rows, _ = _run_budgeted_pool_search(evaluator=evaluator, portfolio_ic=runner.ic,
            pool=list(runner.ranks), recipe=recipe, segments=[(evaluator.start, evaluator.end)],
            clusters={"a": 1, "b": 2, "c": 3}, db=db, output=tmp_path, budget=3, cache_namespace="frozen")
    assert rows[0]["status"] == "rejected_runtime" and message in rows[0]["error"]
    assert all(r["status"] == "evaluated" for r in rows[1:])


def test_configured_catalog_and_search_use_same_panel_start(monkeypatch):
    import run_portfolio_workflow as ide
    import research.portfolio_experiment_support as support
    config = load_config("config/strategy_trend_allocation.yaml")
    config.factors = ["a", "b"]
    monkeypatch.setattr(support, "latest_local_date", lambda: pd.Timestamp("2026-05-15"))
    monkeypatch.setattr(support, "FactorPanelRunner", lambda *args, **kwargs: kwargs)
    result = ide._build_shared_production_panel([(None, None, config)])
    assert result["start"] == pd.Timestamp("2019-01-01") - pd.Timedelta(days=252)
    assert result["end"] == pd.Timestamp("2026-05-15")
    assert result["retain_values"]
    neutral = load_config("config/default.yaml")
    result = ide._build_shared_production_panel([(None, None, neutral)])
    assert result["start"] == pd.Timestamp(neutral.date_range.start) - pd.Timedelta(days=ide.LEGACY_PANEL_BUFFER_DAYS)


def test_shared_catalog_has_separate_peer_groups_and_keeps_preferred(monkeypatch):
    import run_portfolio_workflow as ide
    monkeypatch.setattr(ide, "WORKFLOW", ide.PortfolioWorkflow.RUN_AND_COMPARE)
    monkeypatch.setattr(ide, "STRATEGY_IDS", ())
    _, _, peers = ide._validated_specs()
    assert len(peers) == 12 and all(s.comparison_group == "neutral_futures" for s, _, _ in peers)
    monkeypatch.setattr(ide, "STRATEGY_IDS", ("trend_allocation_robust_pair",))
    _, catalog, selected = ide._validated_specs()
    entry, _, config = selected[0]
    assert entry.recipe_source == "strategy_config" and entry.status == "observing" and not entry.formal
    assert len(config.factors) == 2 and config.production_portfolio.position_mode == "long_only"
    assert [s.id for s in catalog.strategies if s.status == "preferred"] == ["multi_source_resilient"]
    monkeypatch.setattr(ide, "WORKFLOW", ide.PortfolioWorkflow.VALIDATE_CONFIGURATIONS)
    monkeypatch.setattr(ide, "STRATEGY_IDS", ())
    assert {s.id for s, _, _ in ide._validated_specs()[2]} == {
        s.id for s in catalog.strategies if s.status != "archived"}


def test_groups_bind_after_catalog_factors_without_copying_members_to_method_config(tmp_path):
    profile = Path("config/strategy_trend_allocation.yaml").resolve().as_posix()
    child = tmp_path / "group.yaml"
    child.write_text(f"extends: '{profile}'\nproduction_portfolio:\n  factor_groups:\n    alpha: [a, b]\n", encoding="utf-8")
    config = load_config(child)
    validate_portfolio_config(config, factor_names=["a", "b", "c"])
    with pytest.raises(ValueError, match="configured factor names"):
        validate_portfolio_config(config, factor_names=["a", "c"])


def test_shared_panel_rejects_different_data_sources(monkeypatch):
    import run_portfolio_workflow as ide
    config, _, _, _ = case()
    other = config.model_copy(deep=True)
    other.data.source = "parquet_futures"
    monkeypatch.setattr("research.portfolio_experiment_support.latest_local_date", lambda: "2026-05-15")
    with pytest.raises(ValueError, match="identical data"):
        ide._build_shared_production_panel([(None, None, config), (None, None, other)])


def test_unavailable_factor_cannot_silently_become_cash():
    _, recipe, runner, evaluator = case()
    runner.factor_values["a"].loc[:, :] = np.nan
    with pytest.raises(RuntimeError, match="no constructible"):
        evaluator.weights(["a", "b"], recipe)


def test_no_implicit_cost_annualization_change():
    config, _, _, _ = case()
    config.costs.periods_per_year = 252
    with pytest.raises(ValueError, match="annualization"):
        validate_portfolio_config(config)


def test_neutral_recipe_grid_cannot_silently_change_a_configured_search():
    from research.historical_portfolio_search import portfolio_recipe_grid
    _, recipe, _, _ = case()
    with pytest.raises(ValueError, match="neutral grid"):
        portfolio_recipe_grid(recipe)
