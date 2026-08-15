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
    factor_weights,
    prepare_complete_history,
    select_pool,
    training_factor_diagnostics,
)


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


class _FakeEnv:
    def __init__(self, dates, returns):
        self._dates = dates
        self._returns = returns
        self.sector_of = {"A": "x", "B": "y", "C": "z"}

    def eligible_symbols(self, date):
        history = self._returns.loc[self._returns.index < pd.Timestamp(date)]
        return history.columns[history.notna().sum().ge(10)].tolist()


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
    assert list(history.columns) == ["A", "B"]
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
            return pd.DataFrame({"net_return": values, "turnover": 0.1}, index=dates)

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

        def eligible_symbols(self, _date):
            return symbols

        def capped(self, row, ascending=False, date=None):
            del date
            return row.sort_values(ascending=ascending).index[:10].tolist()

        def erc_w(self, pool, _date):
            if int(dates.get_loc(_date)) < 10:
                return None
            return {symbol: 1.0 / len(pool) for symbol in pool}

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
