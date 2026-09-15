from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from external_strategies.guosen_trend_index.production_compare import (
    _run_production_weights,
)
from workflows.experiments import historical_portfolio_search as workflow

from research.historical_portfolio_search import (
    PortfolioEvaluator,
    PortfolioRecipe,
    beam_factor_sets,
    cluster_factors,
    exhaustive_factor_set_shortlist,
    factor_weights,
    performance_metrics,
    prepare_complete_history,
    select_pool,
    training_factor_diagnostics,
)
from optimization.factor_weighting import (
    combine_available_factor_scores,
    rank_information_coefficients,
)
from factors.engine import FactorComputationError
from research.portfolio_experiment_support import FactorPanelRunner


def test_declared_recipe_grid_is_full_cartesian_and_preserves_constraints():
    from itertools import product
    from research.historical_portfolio_search import portfolio_recipe_grid
    base = PortfolioRecipe(asset_min_fraction=0.01, asset_max_fraction=0.30,
                           gross_exposure=1.5, sector_weight_caps=(("metals", 0.4),))
    recipes = portfolio_recipe_grid(base)
    assert len(recipes) == len(set(recipes)) == 96
    assert {(r.factor_weight, r.top_n, r.sector_cap, r.asset_weight) for r in recipes} == set(product(
        ("equal", "diag_icir", "lw_abs", "lw_positive"), (5, 8, 10, 12),
        (0, 3), ("equal", "inverse_volatility", "erc"),
    ))
    assert base in recipes
    assert all((r.asset_min_fraction, r.asset_max_fraction, r.gross_exposure,
                r.sector_weight_caps) == (0.01, 0.30, 1.5, (("metals", 0.4),))
               for r in recipes)


def test_neighborhood_matches_independent_powerset_oracle():
    from itertools import combinations
    from research.historical_portfolio_search import factor_neighborhood
    old, new = set("abc"), set("def")
    expected = set()
    for size in range(2, 7):
        for values in combinations(sorted(old | new), size):
            members = set(values)
            if len(old - members) <= 2 and len(members - old) <= 2:
                expected.add(values)
    rows = list(factor_neighborhood(sorted(old), sorted(new)))
    assert {row[0] for row in rows} == expected
    assert len(rows) == len(expected)
    for members, removed, added in rows:
        assert set(members) == (old - set(removed)) | set(added)
    assert rows == list(factor_neighborhood("cbac", "fedaa"))


def test_top5_with_existing_twenty_percent_cap_has_equal_final_weights():
    from optimization.portfolio_construction import PortfolioConstraints, allocate_sleeve
    rng = np.random.default_rng(47)
    history = pd.DataFrame(rng.normal(size=(90, 5)) * np.arange(1, 6),
                           columns=list("ABCDE"))
    constraints = PortfolioConstraints(top_n_per_side=5)
    for method in ("equal", "inverse_volatility", "erc"):
        weights = allocate_sleeve(history, method=method, constraints=constraints,
                                  sector_of={name: name for name in history})
        np.testing.assert_allclose(weights.to_numpy(), np.full(5, 0.2), atol=1e-9)


def test_neighborhood_full_declared_counts_and_invalid_limits():
    from research.historical_portfolio_search import factor_neighborhood
    new = [f"new_{i}" for i in range(14)]
    counts = [sum(1 for _ in factor_neighborhood([f"old_{i}" for i in range(n)], new))
              for n in (10, 8, 13, 17, 17, 18, 13, 10)]
    assert sum(counts) == 86178
    assert sum(counts) * 96 == 8273088
    with np.testing.assert_raises(ValueError):
        list(factor_neighborhood(["a", "b"], new, max_add=-1))


def test_joint_neighborhood_generator_preserves_signs_and_originals_first():
    from research.historical_portfolio_search import unique_factor_neighborhoods
    seeds = [{"id": "one", "directions": {"a": 1, "b": -1}},
             {"id": "two", "directions": {"a": 1, "b": 1}}]
    candidates = {"c": {"direction": -1}, "d": {"direction": 1}}
    rows = list(unique_factor_neighborhoods(seeds, candidates))
    assert [row["seed"] for row in rows[:2]] == ["one", "two"]
    assert all(row["original"] for row in rows[:2])
    assert not any(row["original"] for row in rows[2:])
    identities = [tuple(row["directions"].items()) for row in rows]
    assert len(identities) == len(set(identities))
    assert (("a", 1), ("b", -1)) in identities
    assert (("a", 1), ("b", 1)) in identities
    assert identities.count((("c", -1), ("d", 1))) == 1
    assert rows == list(unique_factor_neighborhoods(seeds, candidates))


def test_batched_scores_are_bitwise_equal_to_daily_public_combiner():
    from optimization.factor_weighting import causal_history
    rng = np.random.default_rng(17)
    dates = pd.bdate_range("2024-01-02", periods=100)
    names, universe = ["a", "b", "c", "d", "e"], ["Y", "X", "Z"]
    ranks = {name: pd.DataFrame(rng.uniform(size=(100, 3)), index=dates, columns=universe)
             for name in names}
    ranks["a"].iloc[70, 1] = np.nan
    ranks["b"].iloc[65, 2] = np.inf
    ranks["c"].iloc[80, :] = np.nan
    ic = pd.DataFrame(rng.normal(0.5, 0.02, size=(100, len(names))), index=dates, columns=names)
    ic.iloc[15, 1] = np.nan
    runner = SimpleNamespace(cal=dates, u=universe, ranks=ranks, ic=ic)
    evaluator = PortfolioEvaluator(runner, start=dates[0], end=dates[-1])
    for method in ("equal", "diag_icir", "lw_abs", "lw_positive"):
        expected = pd.DataFrame(np.nan, index=dates, columns=universe)
        for date in dates:
            if method == "equal":
                weights = pd.Series(1 / len(names), index=names)
            else:
                history = prepare_complete_history(causal_history(ic, date, 60), minimum_observations=30)
                weights = factor_weights(history, method)
            expected.loc[date] = combine_available_factor_scores(
                {name: ranks[name].loc[date] for name in names}, weights, universe)
        pd.testing.assert_frame_equal(evaluator._score_matrix(names, method), expected, check_exact=True)


def test_neighborhood_pilot_keeps_frozen_recipe_and_records_failed_jobs(monkeypatch, tmp_path):
    base = PortfolioRecipe(asset_min_fraction=0.01, asset_max_fraction=0.3,
                           gross_exposure=1.5, sector_weight_caps=(("metals", 0.4),))
    contract = {
        "input_sha256": {}, "factor_source_sha256": "frozen", "source_fingerprint": "market",
        "seeds": [{"id": "small", "directions": {"a": 1, "b": -1}},
                  {"id": "large", "directions": {"a": 1, "b": -1, "c": 1}}],
        "panel_start": "2015-04-01", "performance_start": "2016-03-31",
        "end": "2016-04-01", "baseline_recipe": base.to_dict(),
        "frozen_runtime": {"ic_window": 60, "risk_lookback_calendar_days": 90},
    }
    (tmp_path / "search_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    class FakeRunner:
        _source_tree_fingerprint = staticmethod(lambda: "frozen")
        def __init__(self, *args, **kwargs):
            assert kwargs["ic_horizon"] == 1
        def get_contract_schedule(self):
            return None
        def performance_profile(self):
            return {}
        def for_factors(self, factors, *, factor_directions):
            assert factor_directions["b"] == -1
            return self
    class FakeEvaluator:
        def __init__(self, *args, **kwargs):
            assert kwargs["start"] == "2016-03-31"
        def clear_transient_caches(self):
            pass
        def ledger(self, factors, recipe):
            assert recipe.gross_exposure == 1.5
            assert recipe.sector_weight_caps == (("metals", 0.4),)
            if recipe.factor_weight == "equal":
                raise RuntimeError("test unavailable history")
            return pd.DataFrame({"net_return": [0.0, 0.01]},
                                index=pd.bdate_range("2016-03-31", periods=2))
    monkeypatch.setattr(workflow, "Runner", FakeRunner)
    monkeypatch.setattr(workflow, "PortfolioEvaluator", FakeEvaluator)
    monkeypatch.setattr(workflow, "_search_source_snapshot", lambda *_args: {"source_fingerprint": "market"})
    result = workflow.benchmark_neighborhood_search(tmp_path)
    assert len(result["jobs"]) == 10
    assert sum(row["status"] == "failed" for row in result["jobs"]) == 2
    assert result["new_factor_compute_included"] is False
    assert {row["recipe"]["top_n"] for row in result["jobs"]} == {5, 10}
    with np.testing.assert_raises(FileExistsError):
        workflow.benchmark_neighborhood_search(tmp_path)


def test_native_batch_resumes_without_repeating_or_dropping_failures(monkeypatch, tmp_path):
    import duckdb
    dates = pd.bdate_range("2016-03-31", periods=3)
    recipes = [PortfolioRecipe("equal", 5, 3, "equal"),
               PortfolioRecipe("diag_icir", 5, 3, "equal")]
    contract = {
        "input_sha256": {}, "factor_source_sha256": "code", "source_fingerprint": "market",
        "panel_start": "2015-04-01", "performance_start": "2016-03-31", "end": "2016-04-04",
        "selection_end": "2016-04-01", "observation_start": "2016-04-02",
        "seeds": [{"id": "baseline", "directions": {"a": 1, "b": -1}}],
        "candidates": {}, "recipes": [r.to_dict() for r in recipes],
        "baseline_recipe": recipes[0].to_dict(),
        "historical_folds": [{"fold": "one", "test_start": "2016-04-01", "test_end": "2016-04-04"}],
        "fold_role": "historical_stability_not_independent_factor_discovery_oos",
        "counts": {"jobs_after_dedup": 2},
        "frozen_runtime": {"ic_horizon": 1, "ic_window": 60, "risk_lookback_calendar_days": 90},
    }
    (tmp_path / "search_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    (tmp_path / "pilot_profile.json").write_text(json.dumps({
        "role": "cost_measurement_not_selection", "jobs": [{}] * 10}), encoding="utf-8")
    (tmp_path / "ledger_parity.json").write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    checkpoint = tmp_path / "factor_panel_checkpoint"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text(json.dumps({"contract": {"factors": ["a", "b"]},
        "completed": {"a": "a.parquet", "b": "b.parquet"}}), encoding="utf-8")
    for name in ("a", "b"):
        pd.DataFrame({"X": [1, 2, 3]}, index=dates).to_parquet(checkpoint / f"{name}.parquet")
    calls = []
    class FakeRunner:
        _source_tree_fingerprint = staticmethod(lambda: "code")
        def __init__(self, names, **kwargs):
            assert names == ["a", "b"]
            self.raw_ranks = {}
            self.cal = dates
            self.daily_ret = pd.DataFrame({"X": 0.0}, index=dates)
            self.env = SimpleNamespace(sector_of={"X": "X"})
        def _load_factor_checkpoint(self, *args):
            return {n: pd.DataFrame({"X": [1, 2, 3]}, index=dates) for n in ["a", "b"]}, None
        def for_factors(self, *args, **kwargs):
            assert kwargs["factor_directions"] == {"a": 1, "b": -1}
            return self
        def get_contract_schedule(self):
            pass
        def performance_profile(self):
            return {}
    class FakeEvaluator:
        def __init__(self, *args, **kwargs):
            self._ledger_cache = {}
        def ledger(self, factors, recipe):
            calls.append(recipe.factor_weight)
            if recipe.factor_weight == "diag_icir":
                raise RuntimeError("insufficient historical sample")
            return pd.DataFrame({"net_return": [0.0, 0.01, -0.02], "nav": [1000., 1010., 989.8]}, index=dates)
        def clear_transient_caches(self):
            pass
    monkeypatch.setattr(workflow, "Runner", FakeRunner)
    monkeypatch.setattr(workflow, "PortfolioEvaluator", FakeEvaluator)
    monkeypatch.setattr(workflow, "_search_source_snapshot", lambda *args: {"source_fingerprint": "market"})
    first = workflow.run_neighborhood_batch(tmp_path, max_jobs=1)
    assert workflow.report_neighborhood_search(tmp_path)["complete"] is False
    assert "部分完成" in (tmp_path / "conclusion_report.md").read_text(encoding="utf-8")
    second = workflow.run_neighborhood_batch(tmp_path, max_jobs=1)
    third = workflow.run_neighborhood_batch(tmp_path, max_jobs=1)
    assert calls == ["equal", "diag_icir"]
    assert first["full_search_complete"] is False
    assert second["counts"] == {"completed": 1, "failed": 1}
    assert second["full_search_complete"] is True
    assert third["performed_this_batch"] == 0
    with duckdb.connect(str(tmp_path / "search_results.duckdb"), read_only=True) as db:
        assert db.execute("select id from results order by id").fetchall() == [(0,), (1,)]
    report = workflow.report_neighborhood_search(tmp_path)
    assert report["complete"] is True
    assert report["counts"] == {"completed": 1, "failed": 1}
    assert len(pd.read_parquet(tmp_path / "all_performance.parquet")) == 2
    assert (tmp_path / "nav_comparison.png").stat().st_size > 10000
    assert "失败 1" in (tmp_path / "conclusion_report.md").read_text(encoding="utf-8")


def test_historical_period_protocol_is_frozen_and_copy_safe():
    from research.validation import (
        LONG_HISTORY_REPLAY_END,
        OOS_END,
        SIMULATED_LIVE_START,
        expanding_window_folds,
        historical_experiment_period_snapshot,
    )

    folds = expanding_window_folds()
    assert workflow.OUTER_FOLDS == folds
    assert folds[-1]["test_end"] == OOS_END == "2026-05-14"
    assert SIMULATED_LIVE_START == "2026-05-15"
    assert LONG_HISTORY_REPLAY_END == "2026-08-20"
    assert historical_experiment_period_snapshot()["long_history_replay"]["role"] == "frozen_control_only"
    folds[-1]["test_end"] = "2099-12-31"
    assert expanding_window_folds()[-1]["test_end"] == "2026-05-14"


def test_factor_manifest_loads_only_estimable_factors(tmp_path):
    path = tmp_path / "ic_by_window_period.json"
    path.write_text(json.dumps({"all_results": [
        {"name": "a", "all_periods": {"p1": {"estimable": True}}},
        {"name": "b", "all_periods": {"p1": {"estimable": False}}},
        {"name": "a", "all_periods": {"p2": {"estimable": True}}},
    ]}), encoding="utf-8")

    factors, digest = workflow._load_estimable_factor_manifest(path)

    assert factors == ["a"]
    assert len(digest) == 64


def test_simulated_live_uses_frozen_fold_four_decision(monkeypatch, tmp_path):
    dates = pd.bdate_range("2026-05-14", "2026-05-20")
    runner = SimpleNamespace(
        cal=dates,
        daily_ret=pd.DataFrame({"A": 0.0}, index=dates),
    )

    class FakeEvaluator:
        def __init__(self, *_args, **_kwargs):
            pass

        def weights(self, factors, recipe):
            assert factors == ["factor_a"]
            assert recipe.name == "equal__top1_bottom1__capnone__equal"
            return pd.DataFrame({"A": 1.0}, index=dates)

        def ledger_from_weights(self, weights):
            return pd.DataFrame({"net_return": [0.0, 0.01, -0.01, 0.02, 0.0]}, index=weights.index)

        def clear_transient_caches(self):
            pass

    monkeypatch.setattr(workflow, "PortfolioEvaluator", FakeEvaluator)
    metrics = workflow._evaluate_simulated_live(runner, {
        "selected_candidate": "beam_1",
        "selected_factors": ["factor_a"],
        "selected_recipe": PortfolioRecipe("equal", 1, 0, "equal").to_dict(),
    }, tmp_path)

    assert metrics["start"] == "2026-05-15"
    assert metrics["end"] == "2026-05-20"
    assert (tmp_path / "simulated_live_ledger.csv").is_file()


def test_factor_panel_schedule_is_frozen_before_data_environment_is_detached():
    dates = pd.bdate_range("2024-01-02", periods=2)
    expected = pd.DataFrame({"A": ["A2401", "A2405"]}, index=dates)

    class _Source:
        def fetch_contract_schedule(self, tickers, start, end):
            assert tickers == ["A"]
            assert start == dates.min()
            assert end == dates.max()
            return expected

    runner = FactorPanelRunner.__new__(FactorPanelRunner)
    runner.env = SimpleNamespace(
        data_manager=SimpleNamespace(source=_Source())
    )
    runner.u = ["A"]
    runner.cal = dates
    runner._contract_schedule = None
    runner._contract_schedule_loaded = False

    assert runner.get_contract_schedule() is expected
    runner.env = SimpleNamespace()
    assert runner.get_contract_schedule() is expected


def test_factor_panel_strategy_views_reuse_values_and_keep_directions_independent():
    dates = pd.bdate_range("2024-01-02", periods=4)
    raw_rank = pd.DataFrame(
        [[1 / 3, 2 / 3, 1.0]] * len(dates),
        index=dates,
        columns=["A", "B", "C"],
    )
    returns = pd.DataFrame(
        [[0.01, 0.02, 0.03]] * len(dates),
        index=dates,
        columns=raw_rank.columns,
    )
    runner = FactorPanelRunner.__new__(FactorPanelRunner)
    runner.raw_ranks = {"factor_a": raw_rank}
    runner._ic_returns = returns

    positive = runner.for_factors(
        ["factor_a"], factor_directions={"factor_a": 1}
    )
    negative = runner.for_factors(
        ["factor_a"], factor_directions={"factor_a": -1}
    )

    assert positive.raw_ranks is runner.raw_ranks
    pd.testing.assert_frame_equal(positive.ranks["factor_a"], raw_rank)
    pd.testing.assert_frame_equal(negative.ranks["factor_a"], 1 - raw_rank)
    pd.testing.assert_frame_equal(runner.raw_ranks["factor_a"], raw_rank)
    assert positive.ic["factor_a"].dropna().gt(0).all()
    assert negative.ic["factor_a"].dropna().lt(0).all()


def test_factor_panel_checkpoint_resumes_and_rejects_contract_drift(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=3)
    values = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        index=dates,
        columns=["A", "B"],
    )

    def runner(namespace):
        item = FactorPanelRunner.__new__(FactorPanelRunner)
        item.cal = dates
        item.u = ["A", "B"]
        item.env = SimpleNamespace(data_manager=SimpleNamespace(
            source=SimpleNamespace(cache_namespace=namespace)
        ))
        return item

    first = runner("release-a")
    loaded, manifest = first._load_factor_checkpoint(tmp_path, ["factor_a"])
    assert loaded == {}
    first._save_factor_checkpoint(
        tmp_path, manifest, {"factor_a": values}
    )

    resumed, _ = runner("release-a")._load_factor_checkpoint(
        tmp_path, ["factor_a"]
    )
    pd.testing.assert_frame_equal(resumed["factor_a"], values)

    with np.testing.assert_raises_regex(RuntimeError, "contract changed"):
        runner("release-b")._load_factor_checkpoint(tmp_path, ["factor_a"])


def test_factor_panel_runner_resumes_after_completed_batch(tmp_path, monkeypatch):
    import research.portfolio_experiment_support as support

    dates = pd.bdate_range("2024-01-02", periods=4)
    universe = ["A", "B", "C"]
    calls = []
    state = {"fail": True}

    class Engine:
        def compute_factors(self, names, request_dates, requested_universe, parallel=False):
            del parallel
            calls.append(tuple(names))
            assert list(requested_universe) == universe
            if state["fail"] and "factor_10" in names:
                raise RuntimeError("injected interruption")
            return {
                name: pd.DataFrame(
                    np.tile([1.0, 2.0, 3.0], (len(request_dates), 1)),
                    index=request_dates,
                    columns=universe,
                )
                for name in names
            }

    class Environment:
        def __init__(self, factors, *, start, end):
            del factors, start, end
            self.cal = dates
            self.u = universe
            self.engine = Engine()
            self.daily_ret = pd.DataFrame(
                np.tile([0.01, 0.02, 0.03], (len(dates), 1)),
                index=dates,
                columns=universe,
            )
            self.close_tradable = self.daily_ret.notna()
            self.data_manager = SimpleNamespace(source=SimpleNamespace(
                cache_namespace="release-a",
                checkpoint_source_fingerprint=lambda start, end: "slice-a",
            ))
            self.sector_of = {}

    monkeypatch.setattr(support, "ExperimentEnvironment", Environment)
    monkeypatch.setattr(
        FactorPanelRunner, "_source_tree_fingerprint", staticmethod(lambda: "code-a")
    )
    factors = [f"factor_{index:02d}" for index in range(11)]
    with np.testing.assert_raises_regex(RuntimeError, "injected interruption"):
        FactorPanelRunner(factors, checkpoint_dir=tmp_path)

    first_run_calls = list(calls)
    assert first_run_calls[0] == tuple(factors[:10])
    state["fail"] = False
    calls.clear()
    resumed = FactorPanelRunner(factors, checkpoint_dir=tmp_path)

    assert calls == [("factor_10",)]
    assert resumed.checkpoint_loaded_factor_count == 10
    assert resumed.computed_factor_count == 1
    assert resumed.performance_profile() == {
        "factor_totals": [],
        "chunk_count": 1,
        "chunk_seconds": resumed.factor_chunk_timings[0]["seconds"],
        "factor_seconds": 0,
        "shared_overhead_seconds": resumed.factor_chunk_timings[0]["seconds"],
        "chunks": resumed.factor_chunk_timings,
    }


def test_factor_panel_declared_history_does_not_change_ordinary_chunk_boundaries():
    history = pd.bdate_range("2012-01-02", periods=1600)
    calendar = history[600:]
    ordinary = list(FactorPanelRunner._iter_factor_chunks(calendar))
    extended = list(FactorPanelRunner._iter_factor_chunks(
        calendar, history_calendar=history, history_days=253,
    ))
    from core.period import iter_overlapping_chunks
    old = list(iter_overlapping_chunks(calendar, 400, 128))
    for (target, request), (old_target, old_request), (long_target, long_request) in zip(ordinary, old, extended):
        assert target.equals(old_target) and request.equals(old_request)
        assert long_target.equals(target)
        assert long_request[0] == history[history.searchsorted(target[0]) - 253]
        assert long_request[-1] == request[-1]


def test_incremental_trade_jobs_keep_axes_separate_and_do_not_mutate_seeds():
    peers = [{"id": "s1", "name": "策略一", "directions": {"a": 1, "b": -1}},
             {"id": "s2", "name": "策略二", "directions": {"b": -1, "d": 1}}]
    before = json.dumps(peers, sort_keys=True)
    candidates = {"c": {"direction": -1}, "e": {"direction": 1}}
    jobs = workflow.incremental_trade_jobs(peers, candidates, {"a": 1, "b": 2, "c": 1, "d": 1, "e": 3})
    assert [j["kind"] for j in jobs[:2]] == ["baseline", "baseline"]
    assert len(jobs) == 12  # 2 baselines + 4 buffers + 4 additions + 2 replacements.
    assert len({j["id"] for j in jobs}) == len(jobs)
    assert sum(j["kind"] == "replace" for j in jobs) == 2
    assert all(not j["buffer"] for j in jobs if j["added"])
    assert json.dumps(peers, sort_keys=True) == before


def test_trade_turnover_partition_includes_contract_roll_once():
    from backtest.research_ledger import build_close_marked_ledger
    dates = pd.bdate_range("2025-01-02", periods=5)
    weights = pd.DataFrame({"A": [1., 1., -1., 0., 0.], "B": [-1., -1., 1., 0., 0.]}, index=dates)
    returns = pd.DataFrame(0., index=dates, columns=weights.columns)
    contracts = pd.DataFrame({"A": ["A1", "A1", "A2", "A2", "A2"], "B": ["B1"] * 5}, index=dates)
    result = build_close_marked_ledger(weights, returns, contract_schedule=contracts, trade_cost_rate=0.)
    components = workflow._trade_turnover_components(result)
    np.testing.assert_allclose(components.sum(axis=1), result.daily.executed_traded_notional, atol=1e-12)
    assert components.roll_extra.sum() == 2.
    assert components.entry_exit.sum() == 4.
    assert components.reversal.sum() == 4.


def test_incremental_trade_integration_preserves_baseline_and_cutoff():
    rng = np.random.default_rng(915)
    dates = pd.bdate_range("2025-01-02", periods=100)
    symbols = [f"S{i:02}" for i in range(38)]
    names = ["f1", "f2"]
    returns = pd.DataFrame(rng.normal(0, .01, (100, 38)), index=dates, columns=symbols)
    ranks = {name: pd.DataFrame(rng.random((100, 38)), index=dates, columns=symbols) for name in names}
    runner = SimpleNamespace(cal=dates, u=symbols, daily_ret=returns, ranks=ranks,
        ic=pd.DataFrame(rng.normal(.01, .02, (100, 2)), index=dates, columns=names),
        close_tradable=returns.notna(), get_contract_schedule=lambda: None,
        env=SimpleNamespace(sector_of={s: str(i % 6) for i, s in enumerate(symbols)}))
    recipe = PortfolioRecipe("lw_abs", 10, 3, "erc")
    evaluator = PortfolioEvaluator(runner, start=dates[60], end=dates[-1])
    job = {"directions": {n: 1 for n in names}, "buffer": 0}
    actual, _ = workflow._incremental_trade_result(evaluator, job, recipe)
    expected = evaluator.ledger_from_weights(evaluator.weights(names, recipe))
    pd.testing.assert_frame_equal(actual.daily, expected[actual.daily.columns], check_exact=True)
    buffered, diag = workflow._incremental_trade_result(evaluator, {**job, "buffer": 1}, recipe)
    assert diag["decisions"] == len(buffered.daily) - 1
    assert (buffered.effective_weights.iloc[1:].gt(0).sum(axis=1) == 10).all()
    assert (buffered.effective_weights.iloc[1:].lt(0).sum(axis=1) == 10).all()
    cutoff = dates[85]
    first = workflow._incremental_trade_metrics(buffered, cutoff, dates[60])
    changed_runner = SimpleNamespace(**vars(runner))
    changed_runner.daily_ret = returns.copy()
    changed_runner.daily_ret.loc[dates > cutoff] += .01
    changed_evaluator = PortfolioEvaluator(changed_runner, start=dates[60], end=dates[-1])
    changed, _ = workflow._incremental_trade_result(changed_evaluator, {**job, "buffer": 1}, recipe)
    second = workflow._incremental_trade_metrics(changed, cutoff, dates[60])
    assert first["selection"] == second["selection"]
    assert first["robustness"] == second["robustness"]
    assert first["observation"] != second["observation"]
    assert first["full"]["annualized_turnover"] > 0


@pytest.mark.parametrize("value", [-1, True, 1.0, "1"])
def test_recipe_rejects_non_integer_rank_buffer(value):
    with pytest.raises(ValueError, match="rank_exit_buffer"):
        PortfolioRecipe(rank_exit_buffer=value)


def test_risk_eligibility_batch_matches_causal_window_for_all_subsets(monkeypatch):
    from optimization.portfolio_construction import causal_risk_window
    dates = pd.bdate_range("2024-01-01", periods=130)
    rng = np.random.default_rng(917)
    values = rng.normal(size=(130, 9))
    values[rng.random(values.shape) < .2] = np.nan
    values[50:65, 2] = np.inf
    returns = pd.DataFrame(values, index=dates, columns=list("ABCDEFGHI"))
    runner = SimpleNamespace(cal=dates, u=list(returns), daily_ret=returns)
    for mode in ("reference", "native", "shadow"):
        monkeypatch.setenv("MF_FACTOR_KERNEL_MODE", mode)
        evaluator = PortfolioEvaluator(runner, start=dates[0], end=dates[-1])
        for date in dates[::7]:
            for columns in (list(returns), ["C", "A", "UNKNOWN", "F"]):
                expected = causal_risk_window(returns, date, 90).reindex(columns=columns)
                counts = np.isfinite(expected.to_numpy()).sum(axis=0)
                assert evaluator._risk_eligible(date, columns, 10) == [c for c, n in zip(columns, counts) if n >= 10]
        assert evaluator.risk_count_backend == ("polars" if mode == "reference" else mode)


def test_shared_buffer_run_exports_real_targets_and_separates_cache(monkeypatch):
    from dataclasses import replace
    dates = pd.bdate_range("2025-01-01", periods=30)
    symbols = list("ABCDEF")
    returns = pd.DataFrame(0., index=dates, columns=symbols)
    scores = pd.DataFrame([list(range(6, 0, -1))] * 30, index=dates, columns=symbols, dtype=float)
    scores.loc[dates[21]:, "C"] = 7.
    runner = SimpleNamespace(cal=dates, u=symbols, daily_ret=returns, ranks={"f": scores},
                             env=SimpleNamespace(sector_of={}))
    evaluator = PortfolioEvaluator(runner, start=dates[20], end=dates[-1])
    recipe = PortfolioRecipe("equal", 2, 0, "equal", asset_max_fraction=1.)
    buffered = replace(recipe, rank_exit_buffer=1)
    assert recipe.name != buffered.name
    assert recipe.to_dict()["rank_exit_buffer"] == 0
    with pytest.raises(ValueError, match="actual holdings"):
        evaluator.weights(["f"], buffered)
    old_targets = evaluator.weights(["f"], recipe)
    old = evaluator.ledger_from_weights(old_targets)
    plain = evaluator.run(["f"], recipe)
    pd.testing.assert_frame_equal(plain.target_weights, old_targets, check_exact=True)
    pd.testing.assert_frame_equal(plain.daily, old[plain.daily.columns], check_exact=True)
    result = evaluator.run(["f"], buffered)
    # New rank leader C must not replace actual A/B while both stay in N+B.
    assert result.target_weights.loc[dates[-1], "C"] == 0.
    assert old_targets.loc[dates[-1], "C"] > 0.
    assert result.target_weights.notna().all().all()
    pd.testing.assert_frame_equal(result.effective_weights.iloc[1:].reset_index(drop=True),
                                  result.target_weights.iloc[:-1].reset_index(drop=True), check_exact=True)
    assert result.daily.iloc[-1].decision_turnover == 0.
    for policy in (recipe, buffered):
        direct = evaluator.run(["f"], policy)
        pd.testing.assert_frame_equal(evaluator.ledger(["f"], policy)[direct.daily.columns],
                                      direct.daily, check_exact=True)
    assert len(evaluator._ledger_cache) == 2
    with pytest.raises(ValueError, match="conflicting"):
        workflow._incremental_trade_result(evaluator, {"directions": {"f": 1}, "buffer": 2}, buffered)


def test_factor_profile_cache_resume_preserves_measured_work(tmp_path):
    measured = SimpleNamespace(computed_factor_count=2, checkpoint_loaded_factor_count=0,
                               performance_profile=lambda: {"factor_totals": [{"seconds": 12.}], "chunks": [1]})
    cached = SimpleNamespace(computed_factor_count=0, checkpoint_loaded_factor_count=2,
                             performance_profile=lambda: {"factor_totals": [], "chunks": []})
    path = tmp_path / "profile.json"
    workflow._write_factor_profile(path, measured, preparation_seconds=13.)
    original = path.read_bytes()
    workflow._write_factor_profile(path, cached, preparation_seconds=.01)
    assert path.read_bytes() == original
    fresh = tmp_path / "cached.json"
    workflow._write_factor_profile(fresh, cached, preparation_seconds=.01)
    assert json.loads(fresh.read_text())["computed"] == 0
    assert json.loads(fresh.read_text())["checkpoint_loaded"] == 2


def test_trade_interaction_native_controls_resume_and_report(tmp_path, monkeypatch):
    import duckdb
    rng = np.random.default_rng(916)
    dates = pd.bdate_range("2025-01-02", periods=80)
    symbols = [f"S{i:02}" for i in range(12)]
    returns = pd.DataFrame(rng.normal(0, .01, (80, 12)), index=dates, columns=symbols)
    ranks = {n: pd.DataFrame(rng.random((80, 12)), index=dates, columns=symbols) for n in ("f1", "f2")}
    runner = SimpleNamespace(cal=dates, u=symbols, daily_ret=returns, ranks=ranks,
        ic=pd.DataFrame(rng.normal(.01, .02, (80, 2)), index=dates, columns=list(ranks)),
        close_tradable=returns.notna(), get_contract_schedule=lambda: None,
        env=SimpleNamespace(sector_of={s: str(i % 3) for i, s in enumerate(symbols)}))
    recipe = PortfolioRecipe("lw_abs", 2, 2, "erc", asset_max_fraction=1.)
    evaluator = PortfolioEvaluator(runner, start=dates[60], end=dates[-1])
    jobs = [{"id": f"seed__B{b}", "seed": "seed", "name": "测试策略(2)", "buffer": b,
             "kind": "baseline" if b == 0 else "buffer", "directions": {"f1": 1, "f2": 1},
             "added": "", "removed": "", "reference_job": "prior_add"} for b in (0, 1, 2)]
    reference, output = tmp_path / "reference", tmp_path / "study"
    reference.mkdir(); output.mkdir()
    control, _ = workflow._incremental_trade_result(evaluator, jobs[0], recipe)
    with duckdb.connect(str(reference / "results.duckdb")) as db:
        frame = control.daily.reset_index(names="date").assign(job="prior_add")
        db.register("control", frame)
        db.execute("CREATE TABLE daily AS SELECT * FROM control")
    source = {"performance_start": str(dates[60].date()), "selection_end": str(dates[70].date())}
    first = workflow._run_incremental_jobs(output, reference, evaluator, recipe, jobs, source, max_jobs=1)
    assert first == {"completed": 1, "planned": 3, "finished": False}
    final = workflow._run_incremental_jobs(output, reference, evaluator, recipe, jobs, source)
    assert final == {"completed": 3, "planned": 3, "finished": True}
    assert workflow._run_incremental_jobs(output, reference, evaluator, recipe, jobs, source) == final
    with duckdb.connect(str(output / "results.duckdb"), read_only=True) as db:
        results = [json.loads(row[0]) for row in db.execute("SELECT payload FROM results").fetchall()]
        assert db.execute("SELECT count(*) FROM daily").fetchone()[0] == 3*20
        assert db.execute("SELECT count(*) FROM asset_daily").fetchone()[0] == 3*20*12
    assert all(r["status"] == "completed" for r in results)
    assert next(r for r in results if r["kind"] == "baseline")["baseline_exact"]
    peers = [{"id": "seed", "name": "测试策略(2)"}]
    workflow._json_dump(output / "contract.json", {"peers": peers})
    lines = workflow._trade_interaction_report_lines(output, {"interaction_parent": str(output), "peers": peers}, results)
    assert any("交互差" in line for line in lines)
    assert (output / "original_turnover_sources.png").stat().st_size > 1000
    assert (output / "original_cost_curves.png").stat().st_size > 1000
    # The same native ledgers feed the repeated-selection report; reporting
    # must not rerank membership when observation returns are available.
    from core import registry
    monkeypatch.setattr(registry, "get", lambda *a: SimpleNamespace(description="测试因子"))
    declared = [(dates[60], dates[65]), (dates[66], dates[70])]
    metrics = workflow._incremental_trade_metrics(control, dates[70], dates[60], segments=declared)
    assert [(r["start"], r["end"]) for r in metrics["segments"]] == [
        (str(a.date()), str(b.date())) for a,b in declared]
    contract = {"peers": [{**peers[0], "directions": {"f1": 1, "f2": 1}}],
        "effective_library_count": 2,
        "source": {**source, "panel_start": str(dates[0].date()), "end": str(dates[-1].date())},
        "selection_segments": [(r["start"],r["end"]) for r in results[0]["metrics"]["segments"]],
        "candidates": {n: {"input_bar_frequency": "1min", "best_period": 5} for n in ranks}}
    workflow._json_dump(output / "contract.json", contract)
    workflow._json_dump(output / "jobs.json", jobs)
    workflow._json_dump(output / "clusters.json", {"clusters": {"f1": 1, "f2": 2}})
    workflow._json_dump(output / "selection_manifest.json", {"selected": ["frozen"], "coverage": {
        "A": {"attempts": 1, "successful": 1, "attempted_factors": 2, "successful_factors": 2}}})
    workflow._json_dump(output / "performance.json", {"wall_seconds": 1., "cpu_seconds": 1.})
    with duckdb.connect(str(output / "results.duckdb")) as db:
        row = next(r for r in results if r["buffer"] == 1)
        row.update(kind="candidate", methods=["A"])
        db.execute("UPDATE results SET payload=? WHERE id=?", [json.dumps(row), row["id"]])
    assert workflow.report_repeated_library_search(output)["promoted"] is False
    assert (output / "nav_comparison.png").stat().st_size > 1000
    assert (output / "selection_segments.png").stat().st_size > 1000
    assert "测试因子" in (output / "report.md").read_text(encoding="utf-8")


def test_repeated_selection_rejects_mixed_modes_before_execution(monkeypatch, tmp_path):
    import sys
    import pytest
    for extra in (["--incremental-trade-study"], ["--max-jobs", "1"],
                  ["--repeat-selection-report"], ["--reference-run", str(tmp_path)]):
        monkeypatch.setattr(sys, "argv", ["study", "--output", str(tmp_path),
            "--repeat-selection-parent", str(tmp_path), *extra])
        with pytest.raises(SystemExit) as exc:
            workflow.main()
        assert exc.value.code == 2


def test_incremental_diagnostic_plots_allow_uncomputed_cells(tmp_path, monkeypatch):
    from core import registry
    monkeypatch.setattr(registry, "get", lambda *args: SimpleNamespace(description="测试因子"))
    contract = {"peers": [{"id": "seed", "name": "测试策略(1)"}], "candidates": {"f": {}},
                "source": {"selection_end": "2026-05-15", "end": "2026-09-11"}}
    workflow._plot_incremental_trade_diagnostics(tmp_path, contract, [], {})
    for name in ("buffer_nav.png", "buffer_effects.png", "factor_increment_effects.png"):
        assert (tmp_path / name).stat().st_size > 1000


def test_incremental_report_shortlist_ignores_observation_and_waits_for_all_jobs(tmp_path, monkeypatch):
    import duckdb
    import run_portfolio_workflow
    from core import registry
    monkeypatch.setattr(registry, "get", lambda *args: SimpleNamespace(description="测试因子"))
    monkeypatch.setattr(run_portfolio_workflow, "_write_comparison_plot", lambda *a, **k: None)
    monkeypatch.setattr(workflow, "_plot_incremental_trade_diagnostics", lambda *a: None)
    contract = {"source": {"selection_end": "2026-05-15", "panel_start": "2015-04-01",
                           "performance_start": "2016-03-31", "end": "2026-09-11"},
                "peers": [], "candidates": {"f": {"input_bar_frequency": "1min", "best_period": 5, "direction": 1}},
                "shortlist_rule": "frozen"}
    workflow._json_dump(tmp_path / "contract.json", contract)
    workflow._json_dump(tmp_path / "clusters.json", {"clusters": {"f": 1}})
    workflow._json_dump(tmp_path / "jobs.json", ["base", "add", "pending"])
    period = {"annual_return": .1, "sharpe": 1., "max_drawdown": -.1,
              "annualized_turnover": 200., "gross_annual_return": .15,
              "total_return": 1., "annual_volatility": .1,
              "cost_sensitivity": {"breakeven_trade_cost_rate": .0006,
                  "scenarios": [{"trade_cost_rate": bp/10000, "metrics": {"annual_return": .01}} for bp in (4, 8)]}}
    robust = {"positive_segment_ratio": 1., "worst_sharpe": .5, "median_sharpe": 1.,
              "median_annual_return": .1, "worst_drawdown": -.1}
    base = {"id": "base", "seed": "seed", "name": "测试策略(1)", "kind": "baseline", "buffer": 0,
            "status": "completed", "directions": {"old": 1}, "added": "", "removed": "",
            "first_holding_date": "2016-04-01", "wall_seconds": 1.,
            "metrics": {"selection": period, "observation": period, "full": period, "robustness": robust}}
    add = json.loads(json.dumps(base))
    add.update(id="add", kind="add", added="f", directions={"old": 1, "f": 1})
    add["metrics"]["selection"]["annual_return"] = .11
    with duckdb.connect(str(tmp_path / "results.duckdb")) as db:
        db.execute("CREATE TABLE results(id VARCHAR, payload VARCHAR)")
        db.executemany("INSERT INTO results VALUES (?, ?)", [(r["id"], json.dumps(r)) for r in (base, add)])
        db.execute("CREATE TABLE daily(job VARCHAR, date DATE, nav_after DOUBLE)")
        db.executemany("INSERT INTO daily VALUES (?, ?, ?)", [(r, d, n) for r in ("base", "add")
            for d, n in (("2016-03-31", 1.), ("2026-09-11", 2.))])
    assert workflow.report_incremental_trade_study(tmp_path)["shortlist"] == 0
    workflow._json_dump(tmp_path / "jobs.json", ["base", "add"])
    assert workflow.report_incremental_trade_study(tmp_path)["shortlist"] == 1
    add["metrics"]["observation"].update(annual_return=-.9, sharpe=-10., total_return=-.9)
    add["metrics"]["full"].update(annual_return=-.9, sharpe=-10., total_return=-.9)
    with duckdb.connect(str(tmp_path / "results.duckdb")) as db:
        db.execute("UPDATE results SET payload=? WHERE id='add'", [json.dumps(add)])
    assert workflow.report_incremental_trade_study(tmp_path)["shortlist"] == 1
    assert json.loads((tmp_path / "shortlist.json").read_text())["promoted"] is False


def test_factor_checkpoint_fingerprints_declared_history(monkeypatch):
    calls = []
    runner = FactorPanelRunner.__new__(FactorPanelRunner)
    runner.cal = pd.bdate_range("2020-01-02", periods=10)
    runner.u = ["A"]
    runner._factor_history_days = {"long_factor": 253}
    runner._history_calendar = pd.bdate_range("2018-01-02", runner.cal[-1])
    runner.env = SimpleNamespace(data_manager=SimpleNamespace(source=SimpleNamespace(
        checkpoint_source_fingerprint=lambda left, right: calls.append((left, right)) or "slice",
    )))
    monkeypatch.setattr(runner, "_source_tree_fingerprint", lambda: "code")
    ordinary = runner._checkpoint_contract(["ordinary"])
    assert "data_history_start" not in ordinary
    assert calls[-1][0] == runner.cal[0]
    long = runner._checkpoint_contract(["long_factor"])
    assert long["factor_history_trading_days"] == {"long_factor": 253}
    assert calls[-1][0] == runner._history_calendar[0]


def test_return_metrics_include_drawdown_from_initial_capital():
    metrics = performance_metrics(pd.Series([-0.50, 0.0]))

    assert metrics["max_drawdown"] == -0.50


def test_return_metrics_exclude_declared_nav_anchor():
    metrics = performance_metrics(
        pd.Series([0.0, 0.10, -0.10]),
        periods_per_year=2,
        initial_anchor=True,
    )

    assert metrics["observations"] == 2
    assert np.isclose(metrics["annual_return"], -0.01)


def test_factor_panel_schedule_fails_if_environment_was_detached_too_early():
    runner = FactorPanelRunner.__new__(FactorPanelRunner)
    runner.env = SimpleNamespace()
    runner.u = ["A"]
    runner.cal = pd.bdate_range("2024-01-02", periods=2)
    runner._contract_schedule = None
    runner._contract_schedule_loaded = False

    with np.testing.assert_raises_regex(RuntimeError, "before detaching"):
        runner.get_contract_schedule()


def test_factor_panel_allows_only_leading_empty_chunks_for_late_start_factors():
    dates = pd.bdate_range("2019-01-02", periods=2)
    finite = pd.DataFrame({"A": [1.0, 2.0]}, index=dates)

    class _Engine:
        late_available = False

        def compute_factors(self, names, request_dates, universe, parallel=False):
            del request_dates, universe, parallel
            if len(names) > 1:
                raise FactorComputationError("retry individually")
            name = names[0]
            if name == "late" and not self.late_available:
                try:
                    raise ValueError("factor output contains no finite values")
                except ValueError as exc:
                    raise FactorComputationError("late empty") from exc
            return {name: finite}

    engine = _Engine()
    first = FactorPanelRunner._compute_part(
        engine, ["early", "late"], dates, ["A"], {}
    )
    assert list(first) == ["early"]

    engine.late_available = True
    second = FactorPanelRunner._compute_part(
        engine, ["early", "late"], dates, ["A"], {"early": finite}
    )
    assert list(second) == ["early", "late"]

    engine.late_available = False
    with np.testing.assert_raises(FactorComputationError):
        FactorPanelRunner._compute_part(
            engine,
            ["late"],
            dates,
            ["A"],
            {"late": finite},
        )


def test_factor_panel_chunk_overlap_preserves_120_day_boundary_values():
    calendar = pd.bdate_range("2020-01-02", periods=900)
    raw = pd.Series(np.arange(len(calendar), dtype=float), index=calendar)
    expected = raw.rolling(120, min_periods=120).mean()
    actual = pd.Series(np.nan, index=calendar)

    for target_dates, request_dates in FactorPanelRunner._iter_factor_chunks(calendar):
        part = raw.loc[request_dates].rolling(120, min_periods=120).mean()
        actual.loc[target_dates] = part.reindex(target_dates)

    pd.testing.assert_series_equal(actual, expected)


def test_factor_weight_methods_are_positive_normalized_and_causal_ready():
    history = pd.DataFrame({
        "a": np.linspace(0.01, 0.03, 40),
        "b": np.linspace(0.03, 0.01, 40),
        "c": np.sin(np.linspace(0.0, 3.0, 40)) * 0.01 + 0.005,
    })

    for method in ("equal", "diag_icir", "lw_abs", "lw_positive"):
        weights = factor_weights(history, method)
        assert list(weights.index) == list(history.columns)
        assert weights.ge(0.0).all()
        assert np.isclose(weights.sum(), 1.0)
    assert factor_weights(history, "lw_positive").max() <= 0.35 + 1e-12
    expected_icir = history.mean() / history.std(ddof=1)
    expected_icir = expected_icir.clip(lower=0.0) / expected_icir.clip(lower=0.0).sum()
    pd.testing.assert_series_equal(
        factor_weights(history, "diag_icir"), expected_icir
    )


def test_complete_history_waits_for_late_factor():
    history = pd.DataFrame({
        "a": np.linspace(-0.1, 0.1, 59),
        "b": np.linspace(0.2, -0.2, 59),
        "late": [np.nan] * 30 + list(np.linspace(0.0, 0.1, 29)),
    })

    clean = prepare_complete_history(history, minimum_observations=30)

    assert list(clean.columns) == ["a", "b"]
    assert len(clean) == 59


def test_complete_history_waits_for_global_minimum():
    history = pd.DataFrame({
        "a": np.linspace(-0.1, 0.1, 29),
        "b": np.linspace(0.2, -0.2, 29),
    })

    clean = prepare_complete_history(history, minimum_observations=30)

    assert clean.empty
    assert list(clean.columns) == []


def test_complete_history_rechecks_joint_overlap_after_column_admission():
    history = pd.DataFrame({
        "early": [1.0] * 30 + [np.nan] * 29,
        "late": [np.nan] * 29 + [1.0] * 30,
    })

    clean = prepare_complete_history(history, minimum_observations=30)

    assert clean.empty
    assert list(clean.columns) == []


def test_non_equal_factor_weighting_does_not_fall_back_during_warmup():
    history = pd.DataFrame({
        "a": np.linspace(0.01, 0.02, 29),
        "b": np.linspace(0.02, 0.01, 29),
    })

    assert factor_weights(history, "diag_icir").empty
    with np.testing.assert_raises_regex(ValueError, "unknown factor-weight method"):
        factor_weights(history, "not-a-method")


def test_equal_factor_scores_do_not_require_ic_history():
    dates = pd.bdate_range("2024-01-02", periods=2)
    ranks = {
        "a": pd.DataFrame([[0.2, 0.8], [0.4, 0.6]], index=dates, columns=["A", "B"]),
        "b": pd.DataFrame([[0.6, 0.4], [0.8, 0.2]], index=dates, columns=["A", "B"]),
    }
    runner = SimpleNamespace(cal=dates, u=["A", "B"], ranks=ranks,
                             ic=pd.DataFrame(np.nan, index=dates, columns=["a", "b"]))
    score = PortfolioEvaluator(runner, start=dates[0], end=dates[-1])._score_matrix(
        ["a", "b"], "equal"
    )
    pd.testing.assert_frame_equal(score, (ranks["a"] + ranks["b"]) / 2.0)


def test_rank_ic_requires_three_common_instruments():
    dates = pd.bdate_range("2024-01-02", periods=2)
    rank = pd.DataFrame([[0.2, 0.8, np.nan], [0.4, 0.6, 1.0]], index=dates,
                        columns=["A", "B", "C"])
    returns = pd.DataFrame([[0.0, 0.0, 0.0], [0.1, -0.1, 0.2]], index=dates,
                           columns=rank.columns)
    ic = rank_information_coefficients({"factor": rank}, returns)
    assert pd.isna(ic.loc[dates[0], "factor"])


def test_factor_scores_renormalize_over_available_factors_per_asset():
    score = combine_available_factor_scores(
        {
            "f1": pd.Series({"A": 1.0, "B": 0.5, "C": 0.0}),
            "f2": pd.Series({"A": np.nan, "B": 0.5, "C": 1.0}),
        },
        pd.Series({"f1": 0.75, "f2": 0.25}),
        ["A", "B", "C"],
    )

    pd.testing.assert_series_equal(
        score,
        pd.Series({"A": 1.0, "B": 0.5, "C": 0.25}),
    )


def test_select_pool_honours_requested_size_and_sector_cap():
    score = pd.Series({"A": 6.0, "B": 5.0, "C": 4.0, "D": 3.0, "E": 2.0})
    sectors = {"A": "x", "B": "x", "C": "x", "D": "y", "E": "z"}

    picks = select_pool(
        score,
        eligible=list(score.index),
        sector_of=sectors,
        top_n=3,
        sector_cap=1,
        ascending=False,
    )

    assert picks == ["A", "D", "E"]


def test_training_clustering_and_beam_search_use_training_slice_only():
    dates = pd.bdate_range("2018-01-01", periods=500)
    base = np.sin(np.linspace(0.0, 20.0, len(dates))) * 0.01 + 0.002
    ic = pd.DataFrame({
        "a": base,
        "a_clone": base * 0.95 + 0.0001,
        "b": np.cos(np.linspace(0.0, 15.0, len(dates))) * 0.006 + 0.002,
        "c": np.linspace(0.001, 0.003, len(dates)),
        "d": np.linspace(0.003, 0.001, len(dates)),
        "e": np.full(len(dates), 0.002),
    }, index=dates)
    start, end = dates[0], dates[-1]
    diagnostics = training_factor_diagnostics(ic, start, end, minimum_coverage=0.5)
    clusters = cluster_factors(ic.loc[start:end], list(ic), correlation_threshold=0.65)

    assert clusters["a"] == clusters["a_clone"]
    candidates = beam_factor_sets(
        ic,
        diagnostics,
        clusters,
        start=start,
        end=end,
        minimum_size=3,
        maximum_size=4,
        beam_width=10,
        output_limit=5,
        segments=[(dates[0], dates[249]), (dates[250], dates[-1])],
    )
    assert candidates
    for candidate in candidates:
        labels = [clusters[name] for name in candidate["factors"]]
        assert len(labels) == len(set(labels))


def test_exhaustive_shortlist_covers_every_subset_and_ignores_future_data():
    dates = pd.bdate_range("2019-01-01", periods=520)
    train_end = dates[399]
    ic = pd.DataFrame({
        "a": 0.003 + np.sin(np.arange(len(dates)) / 13.0) * 0.002,
        "b": 0.002 + np.cos(np.arange(len(dates)) / 17.0) * 0.002,
        "c": 0.001 + np.sin(np.arange(len(dates)) / 19.0) * 0.003,
        "d": -0.001 + np.cos(np.arange(len(dates)) / 11.0) * 0.003,
    }, index=dates)

    first, audit = exhaustive_factor_set_shortlist(
        ic,
        list(ic),
        start=dates[0],
        end=train_end,
        minimum_size=2,
        per_size_limit=1,
        global_limit=2,
        batch_size=3,
    )
    changed = ic.copy()
    changed.loc[changed.index > train_end] = 1000.0
    second, changed_audit = exhaustive_factor_set_shortlist(
        changed,
        list(changed),
        start=dates[0],
        end=train_end,
        minimum_size=2,
        per_size_limit=1,
        global_limit=2,
        batch_size=4,
    )

    assert audit["examined_subsets"] == 11  # C(4,2) + C(4,3) + C(4,4)
    assert changed_audit == audit
    assert {row["factor_count"] for row in first} == {2, 3, 4}
    assert first == second


class _FakeEnv:
    def __init__(self, dates, returns):
        self._dates = dates
        self._returns = returns
        self.sector_of = {"A": "x", "B": "y", "C": "z"}


def test_evaluator_risk_history_excludes_decision_date_return():
    dates = pd.bdate_range("2024-01-01", periods=45)
    returns = pd.DataFrame({
        "A": np.linspace(-0.01, 0.01, len(dates)),
        "B": np.linspace(0.02, -0.02, len(dates)),
        "C": np.nan,
    }, index=dates)
    ranks = {
        "f1": pd.DataFrame(0.6, index=dates, columns=returns.columns),
        "f2": pd.DataFrame(0.4, index=dates, columns=returns.columns),
    }
    runner = SimpleNamespace(
        cal=dates,
        u=list(returns.columns),
        daily_ret=returns,
        ranks=ranks,
        ic=pd.DataFrame({"f1": 0.01, "f2": 0.02}, index=dates),
        env=_FakeEnv(dates, returns),
    )
    evaluator = PortfolioEvaluator(runner, start=dates[0], end=dates[-1])
    decision = dates[-1]

    history = evaluator._risk_history(decision, ["A", "B", "C"])

    assert history.index.max() < decision
    assert list(history.columns) == ["A", "B", "C"]
    assert evaluator._risk_eligible(decision, ["A", "B", "C"], 10) == ["A", "B"]
    recipe = PortfolioRecipe("equal", 1, 0, "equal")
    assert recipe.name == "equal__top1_bottom1__capnone__equal"


def test_method_coordinate_search_keeps_each_fold_independent(monkeypatch, tmp_path):
    folds = [
        {"fold": "early", "train_start": "2016-01-01", "train_end": "2017-12-31"},
        {"fold": "late", "train_start": "2016-01-01", "train_end": "2019-12-31"},
    ]
    calls = []

    def fake_rank(_evaluator, recipes, _factor_sets, fold, *, stage):
        calls.append((fold["fold"], stage, list(recipes)))
        scores = (
            {"equal": 2.0, "diag_icir": 1.0}
            if fold["fold"] == "early"
            else {"lw_positive": 2.0, "lw_abs": 1.0}
        )
        rows = []
        for recipe in recipes:
            row = {
                **recipe.to_dict(),
                "stage": stage,
                "fold": fold["fold"],
                "segment_count": 1,
                "positive_segment_ratio": 1.0,
                "worst_sharpe": scores.get(recipe.factor_weight, 0.0),
                "median_sharpe": scores.get(recipe.factor_weight, 0.0),
                "median_annual_return": 0.1,
                "worst_drawdown": -0.1,
            }
            rows.append(row)
        return sorted(rows, key=workflow.robustness_key, reverse=True)

    monkeypatch.setattr(workflow, "OUTER_FOLDS", folds)
    monkeypatch.setattr(workflow, "_rank_recipes_for_fold", fake_rank)
    evaluator = SimpleNamespace(
        clear_transient_caches=lambda: None,
        bounded=lambda _start, _end: evaluator,
    )

    workflow._stage_method_search(evaluator, tmp_path)

    early_selection = next(
        recipes for fold, stage, recipes in calls
        if fold == "early" and stage == "selection"
    )
    late_selection = next(
        recipes for fold, stage, recipes in calls
        if fold == "late" and stage == "selection"
    )
    assert {recipe.factor_weight for recipe in early_selection} <= {"equal", "diag_icir"}
    assert {recipe.factor_weight for recipe in late_selection} <= {"lw_positive", "lw_abs"}


def test_fixed_recipe_ranking_uses_only_bounded_training_evaluator():
    fold = {
        "train_start": "2016-01-01",
        "train_end": "2019-12-31",
        "test_start": "2020-01-01",
        "test_end": "2021-12-31",
    }
    recipes = {
        "production": PortfolioRecipe("lw_abs", 10, 3, "erc"),
        "alternative": PortfolioRecipe("equal", 12, 0, "inverse_volatility"),
    }
    dates = pd.bdate_range("2016-01-01", "2021-12-31")

    class TrainingOnlyEvaluator:
        def ledger(self, _factors, recipe):
            assert dates.max() > pd.Timestamp(fold["train_end"])
            values = pd.Series(0.0, index=dates)
            train = values.index <= pd.Timestamp(fold["train_end"])
            wave = np.sin(np.arange(int(train.sum())) / 8.0)
            values.loc[train] = (
                0.001 + 0.0002 * wave
                if recipe == recipes["alternative"]
                else 0.0002 + 0.0010 * wave
            )
            values.loc[~train] = -0.50 if recipe == recipes["alternative"] else 0.50
            values.iloc[0] = 0.0
            return pd.DataFrame(
                {
                    "net_return": values,
                    "turnover": 0.1,
                    "executed_traded_notional": 0.1,
                },
                index=dates,
            )

    ranked = workflow._rank_fixed_recipes_for_fold(
        TrainingOnlyEvaluator(), ["f1", "f2"], recipes, fold
    )

    assert ranked[0]["challenger"] == "alternative"


def test_current_lw_signal_and_top10_match_existing_production_runner():
    dates = pd.bdate_range("2023-01-02", periods=70)
    symbols = [f"S{index:02d}" for index in range(25)]
    factors = ["f1", "f2", "f3"]
    columns = np.arange(len(symbols), dtype=float)
    ranks = {
        name: pd.DataFrame(
            np.tile((columns + offset) % len(symbols), (len(dates), 1)),
            index=dates,
            columns=symbols,
        ).rank(axis=1, pct=True)
        for offset, name in enumerate(factors)
    }
    ic = pd.DataFrame({
        name: 0.01 + 0.002 * np.sin(np.arange(len(dates)) / (5.0 + offset))
        for offset, name in enumerate(factors)
    }, index=dates)
    returns = pd.DataFrame(
        np.tile(np.linspace(-0.01, 0.01, len(dates))[:, None], (1, len(symbols))),
        index=dates,
        columns=symbols,
    )

    class EqualRiskEnv:
        sector_of = {symbol: symbol for symbol in symbols}

    runner = SimpleNamespace(
        cal=dates,
        u=symbols,
        daily_ret=returns,
        ranks=ranks,
        ic=ic,
        env=EqualRiskEnv(),
    )
    existing = _run_production_weights(runner, factors, dates[-1])
    evaluator = PortfolioEvaluator(runner, start=dates[0], end=dates[-1])
    candidate = evaluator.weights(
        factors, PortfolioRecipe("lw_abs", 10, 3, "equal")
    )

    pd.testing.assert_frame_equal(candidate, existing)

    for name in factors:
        runner.ranks[name].loc[dates[-1]] = np.nan
    interrupted = PortfolioEvaluator(runner, start=dates[0], end=dates[-1])
    with np.testing.assert_raises_regex(RuntimeError, "after portfolio start"):
        interrupted.weights(
            factors, PortfolioRecipe("lw_abs", 10, 3, "equal")
        )


def test_risk_lookback_calendar_days_is_explicit_and_causal():
    dates = pd.bdate_range("2024-01-02", periods=120)
    returns = pd.DataFrame({"A": 0.001, "B": -0.001}, index=dates)
    runner = SimpleNamespace(cal=dates, daily_ret=returns)
    short = PortfolioEvaluator(
        runner,
        start=dates[0],
        end=dates[-1],
        risk_lookback_calendar_days=60,
    )
    long = PortfolioEvaluator(
        runner,
        start=dates[0],
        end=dates[-1],
        risk_lookback_calendar_days=120,
    )

    short_history = short._risk_history(dates[-1], ["A", "B"])
    long_history = long._risk_history(dates[-1], ["A", "B"])

    assert len(short_history) < len(long_history)
    assert short_history.index.max() < dates[-1]
    assert long_history.index.max() < dates[-1]


def test_attribution_compounds_costs_sides_and_subintervals_exactly():
    from workflows.experiments.portfolio_attribution import attributed_period
    from backtest.research_ledger import build_close_marked_ledger
    dates = pd.bdate_range("2026-01-01", periods=5)
    targets = pd.DataFrame({"A": [0.5, 0.5, -0.5, -0.5, 0], "B": [-0.5]*5}, index=dates)
    returns = pd.DataFrame({"A": [0, .02, -.01, .03, -.02], "B": [0, -.01, .01, -.02, .01]}, index=dates)
    result = build_close_marked_ledger(targets, returns, trade_cost_rate=.001, annual_fee=.01)
    for frame in (result.daily.iloc[1:], result.daily.iloc[3:]):
        row = attributed_period(frame, result.contributions, result.effective_weights,
                                {"A": "same", "B": "same"}, {"A": "x", "B": "y"})
        expected = (1 + frame.net_return).prod()-1
        assert abs(row["total_return"]-expected) < 1e-14
        assert abs(row["reconciliation_error"]) < 1e-14
        gross = sum(x["contribution"] for x in row["assets"])
        assert abs(gross-row["long_contribution"]-row["short_contribution"]) < 1e-14
        assert row["trade_cost_contribution"] <= 0
        assert row["holding_cost_contribution"] <= 0
        assert abs(sum(row["sector"].values())-gross) < 1e-14
    import pytest
    bad = result.contributions.copy()
    bad.iloc[2, 0] += .01
    with pytest.raises(AssertionError):
        attributed_period(result.daily, bad, result.effective_weights,
                          {"A": "s", "B": "s"}, {"A": "s", "B": "s"})


def test_report_selection_excludes_archives_and_rejects_changed_members():
    import pytest
    from types import SimpleNamespace as NS
    from workflows.experiments.portfolio_attribution import _report_peers
    peer = {"name": "current", "directions": {"a": 1}}
    subset = NS(id="current", factors=["a"], selection_context={"directions": {"a": 1}})
    active = NS(id="current", name="current", status="observing", source="effective_library", factor_set_id="current")
    archived = NS(id="old", name="old*", status="archived", source="legacy_observation")
    catalog = NS(strategies=[archived, active], factor_sets=[subset])
    assert _report_peers({"peers": [peer]}, catalog) == [peer]
    subset.selection_context["directions"] = {"a": -1}
    with pytest.raises(ValueError, match="members/directions differ"):
        _report_peers({"peers": [peer]}, catalog)


def test_filtered_report_cannot_overwrite_source_evidence(tmp_path):
    import pytest
    from workflows.experiments.portfolio_attribution import report
    with pytest.raises(ValueError, match="preserve its source"):
        report(tmp_path, source_dir=tmp_path, catalog_path="config/strategy_library.yaml")


def test_deletions_cover_factors_clusters_and_preserve_direction_identity():
    from workflows.experiments.portfolio_attribution import deletion_jobs
    peers = [{"name": "first", "directions": {"a": 1, "b": -1, "c": 1}},
             {"name": "second", "directions": {"a": -1, "b": -1, "c": 1}}]
    jobs = deletion_jobs(peers, {"a": 1, "b": 2, "c": 2})
    uses = [use for job in jobs.values() for use in job["uses"]]
    assert len(uses) == 10
    assert len(jobs) == 7
    assert all(set(job["directions"]) | set(use["removed"]) == {"a", "b", "c"}
               for job in jobs.values() for use in job["uses"])
    assert any(job["directions"] == {"a": 1} for job in jobs.values())
    assert any(job["directions"] == {"a": -1} for job in jobs.values())


def test_attribution_periods_do_not_use_observation_for_selection_metrics():
    from workflows.experiments.portfolio_attribution import period_metrics
    dates = pd.bdate_range("2016-03-31", "2026-09-08")
    returns = .0003 + np.sin(np.arange(len(dates))) * .003
    returns[0] = 0
    daily = pd.DataFrame({"net_return": returns, "trade_cost": .0001, "holding_cost": .00002}, index=dates)
    daily.iloc[0] = 0
    cutoff = pd.Timestamp("2026-05-15")
    first = period_metrics(daily, str(dates[0].date()), cutoff)
    daily.loc[daily.index > cutoff, "net_return"] = -.02
    second = period_metrics(daily, str(dates[0].date()), cutoff)
    assert first["selection"] == second["selection"]
    assert first["robustness"] == second["robustness"]
    assert first["observation"] != second["observation"]
    assert first["selection"]["static_cost_2x"]["annual_return"] < first["selection"]["base"]["annual_return"]


def test_history_optimization_preserves_recipe_grid_targets(monkeypatch):
    from itertools import product
    import research.historical_portfolio_search as hp

    rng = np.random.default_rng(915)
    dates = pd.bdate_range("2024-01-02", periods=80)
    symbols = [f"S{i:02}" for i in range(38)]
    names = [f"f{i}" for i in range(20)]
    returns = pd.DataFrame(rng.normal(0, .01, (80, 38)), index=dates, columns=symbols)
    ranks = {name: pd.DataFrame(rng.normal(size=(80, 38)), index=dates, columns=symbols).rank(axis=1, pct=True) for name in names}
    ic = pd.DataFrame(rng.normal(.01, .02, (80, 20)), index=dates, columns=names)
    ic.iloc[:5, :3] = np.nan
    runner = SimpleNamespace(cal=dates, u=symbols, daily_ret=returns, ranks=ranks, ic=ic,
                             env=SimpleNamespace(sector_of={s: str(i % 6) for i, s in enumerate(symbols)}))

    def old_history(frame, date, window):
        return frame.loc[frame.index < pd.Timestamp(date)].tail(int(window))

    def old_risk(frame, date, lookback):
        index = pd.DatetimeIndex(frame.index)
        return frame.loc[(index >= date - pd.Timedelta(days=lookback)) & (index < date)]

    for method, allocation, (count, top) in product(
        ("equal", "diag_icir", "lw_abs", "lw_positive"),
        ("equal", "inverse_volatility", "erc"), ((4, 5), (8, 10), (20, 12)),
    ):
        recipe = PortfolioRecipe(method, top, 3 if top != 12 else 0, allocation)
        actual = PortfolioEvaluator(runner, start=dates[-3], end=dates[-1]).weights(names[:count], recipe)
        with monkeypatch.context() as patch:
            patch.setattr(hp, "causal_history", old_history)
            patch.setattr(hp, "causal_risk_window", old_risk)
            expected = PortfolioEvaluator(runner, start=dates[-3], end=dates[-1]).weights(names[:count], recipe)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_trade_replay_inherits_parent_reporting_segments(monkeypatch, tmp_path):
    import duckdb
    import json
    import research.artifacts as artifacts
    import workflows.experiments.historical_portfolio_search as workflow
    from workflows.factor_selection import _production_recipe, _configured_cost_model

    config = workflow.load_config("config/default.yaml")
    source = {"panel_start": "2016-01-01", "end": "2026-09-11",
              "source_fingerprint": "fixed", "recipe": _production_recipe(config).to_dict()}
    runtime = {"costs": _configured_cost_model(config).ledger_parameters(),
               "ic_window": config.production_portfolio.ic_window,
               "risk_lookback_calendar_days": config.production_portfolio.risk_lookback_calendar_days}
    parent = tmp_path / "parent"
    parent.mkdir()
    row = dict(id="control", seed="control", name="control", buffer=0, directions={},
               added=[], removed=[], kind="baseline", status="completed")
    with duckdb.connect(str(parent / "results.duckdb")) as db:
        db.execute("CREATE TABLE results AS SELECT ? AS payload", [json.dumps(row)])
    (parent / "acceptance.json").write_text(json.dumps({"review_complete": True,
        "computation_contract_sha256": "sealed", "results_sha256": "sealed"}), encoding="utf-8")
    monkeypatch.setattr(artifacts, "sha256_file", lambda _: "sealed")
    monkeypatch.setattr(workflow, "_search_source_snapshot", lambda *args: {
        "end": source["end"], "source_fingerprint": source["source_fingerprint"]})
    for explicit in (False, True):
        previous = dict(source=source, runtime=runtime, input_sha256={}, code_sha256={})
        if explicit:
            previous["selection_segments"] = [[f"{year}-01-01", f"{year+1}-12-31"]
                                               for year in range(2016, 2026, 2)]
        encoded = json.dumps(previous)
        (parent / "contract.json").write_text(encoded, encoding="utf-8")
        output = tmp_path / str(explicit)
        assert workflow.replay_completed_trade_study(output, parent, max_jobs=0)["status"] == "frozen"
        frozen = json.loads((output / "contract.json").read_text(encoding="utf-8"))
        assert frozen["source"].get("selection_segments") == previous.get("selection_segments")
        assert (parent / "contract.json").read_text(encoding="utf-8") == encoded
        assert "selection_segments" not in source


def test_repair_audit_rejects_unverified_provenance_before_loading_data(monkeypatch, tmp_path):
    import workflows.experiments.portfolio_attribution as audit
    reference, output, baseline = tmp_path / "search", tmp_path / "repair", tmp_path / "old"
    old = {"factor_source_sha256": "old", "input_sha256": {}}
    valid = {"baseline_audit": str(baseline), "before_factor_source_sha256": "old",
             "after_factor_source_sha256": "new", "unchanged_module_ast_exact": True,
             "old_tree_reconstructed_exact": True, "recomputed_factors": ["factor"],
             "engineering_parity": "engineering.json"}
    monkeypatch.setattr(audit.Runner, "_source_tree_fingerprint", staticmethod(lambda: "new"))
    for field, value in (("after_factor_source_sha256", "different"),
                         ("before_factor_source_sha256", "different"),
                         ("unchanged_module_ast_exact", False),
                         ("old_tree_reconstructed_exact", False),
                         ("baseline_audit", str(tmp_path / "wrong"))):
        documents = {str(reference / "pool_contract.json"): old,
                     str(output / "repair_provenance.json"): {**valid, field: value},
                     "engineering.json": {"all_13_exact": True},
                     str(baseline / "contract.json"): {"source": old}}
        monkeypatch.setattr(audit, "read_json", lambda p: documents[str(p)])
        with np.testing.assert_raises_regex(ValueError, "repair provenance"):
            audit.run(reference, output, repair_reference=baseline)
    assert not output.exists()
