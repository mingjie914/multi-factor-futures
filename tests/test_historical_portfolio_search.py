from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
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
