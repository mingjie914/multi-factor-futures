from __future__ import annotations

from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
import pytest


def test_factor_synthesizer_preserves_cross_sectional_missingness():
    from factors.synthesizer import FactorSynthesizer

    frame = pd.DataFrame(
        [[1.0, np.nan, np.nan], [1.0, 1.0, np.nan]],
        index=pd.date_range("2025-01-02", periods=2, freq="B"),
        columns=["A", "B", "C"],
    )

    result = FactorSynthesizer._cross_section_zscore(frame)

    assert result.loc[:, "C"].isna().all()
    assert np.isnan(result.iloc[0]["B"])
    assert result.iloc[0]["A"] == 0.0


def test_source_tree_hash_covers_rust_sources(tmp_path):
    from research.artifacts import source_tree_hash

    source = tmp_path / "kernel.rs"
    source.write_text("fn value() -> i32 { 1 }", encoding="utf-8")
    before = source_tree_hash(tmp_path)
    source.write_text("fn value() -> i32 { 2 }", encoding="utf-8")
    assert source_tree_hash(tmp_path) != before


def test_factor_processor_rejects_all_unavailable_final_output():
    from factors.processor import FactorProcessor

    dates = pd.date_range("2025-01-02", periods=2, freq="B")
    factor = pd.DataFrame({"A": [1.0, 2.0]}, index=dates)
    context = SimpleNamespace(
        eligibility=pd.DataFrame(False, index=dates, columns=["A"])
    )
    with pytest.raises(ValueError, match="no finite values"):
        FactorProcessor([]).process(factor, context)


def test_regression_hac_handles_near_sample_length_lag_without_warning():
    from testing.regression import _newey_west_t_stat

    values = pd.Series([0.01, -0.02, 0.03, 0.005, -0.01, 0.02])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        t_stat = _newey_west_t_stat(values, forward_period=5)

    assert np.isfinite(t_stat)


class _RiskData:
    def __init__(self, dates: pd.DatetimeIndex, assets: pd.Index, returns: pd.DataFrame):
        self.dates = dates
        self.assets = assets
        self.close = 100.0 * (1.0 + returns.fillna(0.0)).cumprod()
        date_scale = np.linspace(1.0, 2.0, len(dates))[:, None]
        asset_scale = np.linspace(0.8, 1.4, len(assets))[None, :]
        self.volume = pd.DataFrame(
            1000.0 * date_scale * asset_scale,
            index=dates,
            columns=assets,
        )
        sector_names = np.array(["metal", "energy", "agri", "financial"])
        labels = sector_names[np.arange(len(assets)) % len(sector_names)]
        self.industry = pd.DataFrame(
            np.tile(labels, (len(dates), 1)), index=dates, columns=assets
        )

    def get(self, field, dates, universe):
        source = self.close if field == "close" else self.volume
        return source.reindex(index=dates, columns=universe)

    def get_industry(self, dates, universe):
        return self.industry.reindex(index=dates, columns=universe)

    def get_contract_pair(self, field, dates, universe):
        near = self.close.reindex(index=dates, columns=universe)
        asset_slope = pd.Series(
            np.linspace(0.99, 1.02, len(universe)), index=universe
        )
        far = near.mul(asset_slope, axis=1)
        return {"near": near, "far": far}


def _risk_fixture(seed: int = 7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=120, freq="B")
    assets = pd.Index([f"A{i:02d}" for i in range(16)])
    common = rng.normal(0.0, 0.007, size=(len(dates), 1))
    returns = pd.DataFrame(
        common + rng.normal(0.0, 0.01, size=(len(dates), len(assets))),
        index=dates,
        columns=assets,
    )
    alpha = {
        f"alpha_{i}": pd.DataFrame(
            rng.normal(size=returns.shape), index=dates, columns=assets
        )
        for i in range(30)
    }
    return dates, assets, returns, alpha


def test_factor_lookup_is_point_in_time():
    from backtest.engine import Backtester

    factor = pd.DataFrame(
        {"A": [1.0, 2.0, 999.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-10"]),
    )
    row = Backtester._factor_frame_asof(factor, pd.Timestamp("2024-01-05"))
    assert row is not None
    assert row.index.equals(pd.DatetimeIndex(["2024-01-05"]))
    assert row.loc[pd.Timestamp("2024-01-05"), "A"] == 2.0
    assert Backtester._factor_frame_asof(factor, pd.Timestamp("2023-12-31")) is None


def test_backtester_fails_closed_on_empty_calendar():
    from backtest.engine import Backtester

    data = SimpleNamespace(
        get_calendar=lambda start, end: pd.DatetimeIndex([]),
    )
    dates = pd.bdate_range("2024-01-02", periods=2)

    with pytest.raises(RuntimeError, match="交易日历为空"):
        Backtester()._prepare_backtest_data(
            data,
            SimpleNamespace(compute_factors=lambda *args, **kwargs: {}),
            SimpleNamespace(process_batch=lambda *args, **kwargs: {}),
            [],
            {dates[0]: pd.Index(["A"])},
            dates,
            SimpleNamespace(),
        )


def test_backtester_fails_closed_on_empty_close():
    from backtest.engine import Backtester

    dates = pd.bdate_range("2024-01-02", periods=20)
    universe = pd.Index(["A"])
    data = SimpleNamespace(
        get_calendar=lambda start, end: dates,
        get_forward_returns=lambda requested, assets, period=1: pd.DataFrame(
            0.0, index=requested, columns=assets
        ),
        get=lambda field, requested, assets: pd.DataFrame(columns=assets),
    )

    def compute_factors(*args, **kwargs):
        assert hasattr(data, "_factor_eligibility")
        return {}

    with pytest.raises(RuntimeError, match="close行情为空"):
        Backtester()._prepare_backtest_data(
            data,
            SimpleNamespace(compute_factors=compute_factors),
            SimpleNamespace(process_batch=lambda *args, **kwargs: {}),
            [],
            {dates[0]: universe},
            pd.DatetimeIndex([dates[0], dates[-1]]),
            SimpleNamespace(),
        )


def test_backtester_rebalances_only_on_explicit_dates(monkeypatch):
    from backtest.engine import Backtester

    dates = pd.bdate_range("2024-01-02", periods=15)
    rebalance_dates = pd.DatetimeIndex([dates[0], dates[-1]])
    universe = pd.Index(["A"])
    factors = {"f": pd.DataFrame(1.0, index=dates, columns=universe)}
    daily_returns = pd.DataFrame(0.0, index=dates, columns=universe)
    daily_returns.loc[dates[2], "A"] = -0.10
    forward_returns = pd.DataFrame(0.0, index=dates, columns=universe)
    backtester = Backtester(cost_model=None)
    calls = []

    monkeypatch.setattr(
        backtester,
        "_prepare_backtest_data",
        lambda *args, **kwargs: (
            rebalance_dates,
            universe,
            dates,
            factors,
            forward_returns,
            daily_returns,
        ),
    )

    def refit(*args, **kwargs):
        backtester._last_fit_factors = {"f"}
        return args[3], 0

    monkeypatch.setattr(backtester, "_refit_alpha", refit)
    monkeypatch.setattr(
        backtester,
        "_predict_returns",
        lambda *args, **kwargs: pd.Series(1.0, index=universe),
    )

    def optimize(*args, **kwargs):
        calls.append(pd.Timestamp(args[5]))
        return pd.Series(1.0, index=universe)

    monkeypatch.setattr(backtester, "_optimize_period", optimize)

    backtester.run(
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        ["f"],
        {date: universe for date in rebalance_dates},
        rebalance_dates,
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        [],
    )

    assert calls == list(rebalance_dates)


def test_prediction_and_optimizer_failures_abort_formal_backtest():
    from backtest.engine import Backtester

    class BrokenAlpha:
        def predict(self, factors, universe, date):
            raise RuntimeError("prediction failed")

    class BrokenOptimizer:
        def optimize(self, *args, **kwargs):
            raise RuntimeError("optimization failed")

    dates = pd.date_range("2024-01-01", periods=2, freq="B")
    universe = pd.Index(["A", "B"])
    factors = {"f": pd.DataFrame([[1.0, -1.0], [2.0, -2.0]], index=dates, columns=universe)}
    previous = pd.Series([0.2, -0.1], index=universe)
    backtester = Backtester()
    backtester._last_fit_factors = {"f"}

    with pytest.raises(RuntimeError, match="alpha prediction failed"):
        backtester._predict_returns(BrokenAlpha(), factors, dates[-1], universe)
    with pytest.raises(RuntimeError, match="portfolio optimization failed"):
        backtester._optimize_period(
            pd.Series([0.01, -0.01], index=universe),
            SimpleNamespace(),
            previous,
            [],
            BrokenOptimizer(),
            dates[-1],
            universe,
            0.1,
        )


def test_solver_rejects_inaccurate_status_by_default(monkeypatch):
    import cvxpy as cp

    from optimization.solver_utils import SolverValidationError, solve_validated

    variable = SimpleNamespace(value=np.array([1.0]))

    class FakeProblem:
        constraints = []
        status = None
        value = 1.0

        def solve(self, **kwargs):
            self.status = cp.OPTIMAL_INACCURATE

    monkeypatch.setattr(cp, "installed_solvers", lambda: ["FAKE"])
    with pytest.raises(SolverValidationError, match="optimal_inaccurate"):
        solve_validated(FakeProblem(), variable, ["FAKE"])


def test_erc_matches_requested_risk_contributions():
    from optimization.risk_budgeting import RiskBudgetingOptimizer

    covariance = np.array(
        [[0.04, 0.006, 0.002], [0.006, 0.09, 0.004], [0.002, 0.004, 0.16]]
    )
    budget = np.array([0.2, 0.3, 0.5])
    weights = RiskBudgetingOptimizer._erc_weights(covariance, budget)
    contributions = weights * (covariance @ weights)
    contribution_share = contributions / contributions.sum()
    np.testing.assert_allclose(contribution_share, budget, atol=5e-4)
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights > 0)


def test_covariance_validation_rejects_materially_indefinite_matrix():
    from optimization.solver_utils import validated_psd_covariance

    with pytest.raises(ValueError, match="positive semidefinite"):
        validated_psd_covariance(np.array([[1.0, 2.0], [2.0, 1.0]]))


@pytest.mark.parametrize("budget", [np.array([0.0, 0.0]), np.array([1.0, -0.1])])
def test_erc_rejects_invalid_risk_budget(budget):
    from optimization.risk_budgeting import RiskBudgetingOptimizer

    with pytest.raises(ValueError, match="risk budget"):
        RiskBudgetingOptimizer._erc_weights(np.eye(2), budget)


def test_optimizers_reject_missing_expected_returns():
    from optimization.mean_variance import MeanVarianceOptimizer
    from optimization.risk_budgeting import RiskBudgetingOptimizer

    class IdentityRisk:
        def covariance(self, date, universe):
            return pd.DataFrame(np.eye(len(universe)), index=universe, columns=universe)

    universe = pd.Index(["A", "B"])
    for optimizer in (MeanVarianceOptimizer(), RiskBudgetingOptimizer()):
        with pytest.raises(ValueError, match="expected returns"):
            optimizer.optimize(
                pd.Series({"A": 0.1}),
                IdentityRisk(),
                pd.Series(dtype=float),
                [],
                None,
                pd.Timestamp("2025-01-02"),
                universe,
            )


def test_meta_target_volatility_survives_relative_weight_normalisation(monkeypatch):
    from optimization.meta_optimizer import MetaOptimizer

    rng = np.random.default_rng(11)
    dates = pd.date_range("2023-01-02", periods=80, freq="B")
    returns = pd.DataFrame(
        rng.normal(0.0, 0.025, size=(len(dates), 3)),
        index=dates,
        columns=["short", "mid", "long"],
    )
    optimizer = MetaOptimizer(method="min_variance", target_volatility=0.10)
    monkeypatch.setattr(
        optimizer, "_min_variance", lambda covariance, n: np.ones(n) / n
    )
    relative = optimizer.optimize(returns, date=dates[-1] + pd.Timedelta(days=1))
    assert relative.sum() == pytest.approx(1.0)
    assert optimizer.last_capital_scale < 1.0
    effective = relative * optimizer.last_capital_scale
    assert effective.sum() == pytest.approx(optimizer.last_capital_scale)


def test_meta_projection_enforces_aggregate_underlying_limits():
    from optimization.meta_optimizer import UnderlyingExposureController

    controller = UnderlyingExposureController(
        [
            {"type": "net_exposure", "lower": -0.25, "upper": 0.25},
            {"type": "leverage", "limit": 0.80},
            {"type": "position_limit", "lower": -0.40, "upper": 0.40},
            {"type": "sector_exposure", "limit": 0.40},
        ],
        min_weight=0.0,
        max_weight=1.0,
        sector_map={"A": "metal", "B": "energy"},
    )
    # Sleeve limits alone would allow this target, but their common +A exposure
    # breaches both the instrument and sector limits after aggregation.
    exposure_matrix = np.array([[0.8, 0.8], [0.3, -0.3]])
    target = np.array([0.5, 0.5])
    applied, diagnostics = controller.apply(target, exposure_matrix, ["A", "B"])
    aggregate = exposure_matrix @ applied

    assert diagnostics["constraint_adjusted"] is True
    assert diagnostics["feasible"] is True
    assert applied.sum() <= target.sum() + 1e-8
    assert np.abs(aggregate).sum() <= 0.80 + 1e-7
    assert np.max(np.abs(aggregate)) <= 0.40 + 1e-7
    assert abs(aggregate.sum()) <= 0.25 + 1e-7


def test_meta_projection_enforces_sleeve_bounds_on_fast_path():
    from optimization.meta_optimizer import UnderlyingExposureController

    controller = UnderlyingExposureController([], min_weight=0.4, max_weight=0.6)
    applied, diagnostics = controller.apply(
        np.array([0.9, 0.1]), np.eye(2), ["A", "B"]
    )

    assert diagnostics["constraint_adjusted"] is True
    np.testing.assert_allclose(applied.sum(), 1.0, atol=1e-6)
    assert np.all(applied >= 0.4 - 1e-6)
    assert np.all(applied <= 0.6 + 1e-6)


@pytest.mark.parametrize(
    "spec, message",
    [
        ({"type": "turnover", "limit": 0.3}, "unsupported aggregate"),
        ({"type": "leverage", "limit": -1.0}, "invalid leverage"),
        (
            {"type": "net_exposure", "lower": 0.5, "upper": -0.5},
            "invalid net_exposure",
        ),
    ],
)
def test_meta_projection_rejects_invalid_aggregate_constraints(spec, message):
    from optimization.meta_optimizer import UnderlyingExposureController

    with pytest.raises(ValueError, match=message):
        UnderlyingExposureController([spec])


def test_meta_optimizer_rejects_unsorted_return_dates():
    from optimization.meta_optimizer import MetaOptimizer

    dates = pd.DatetimeIndex(["2025-01-03", "2025-01-02"])
    returns = pd.DataFrame([[0.1, 0.0], [0.0, 0.1]], index=dates, columns=["a", "b"])
    with pytest.raises(ValueError, match="unique and sorted"):
        MetaOptimizer().optimize(returns)


def test_subportfolio_cube_rejects_missing_effective_ledger():
    from pipeline.runner import PipelineRunner

    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    result = SimpleNamespace(
        weights_history=pd.DataFrame({"A": [1.0]}, index=[dates[0]]),
        costs=pd.Series(dtype=float),
    )
    raw = [{"config": SimpleNamespace(name="sleeve"), "result": result}]
    with pytest.raises(ValueError, match="audited effective-weight ledger"):
        PipelineRunner._build_effective_exposure_cube(raw, dates, ["sleeve"])


def test_subportfolio_cube_prefers_drifted_ledger_exposure():
    from pipeline.runner import PipelineRunner

    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    ledger_weights = pd.DataFrame({"A": [0.0, 0.5, 0.6]}, index=dates)
    result = SimpleNamespace(
        weights_history=pd.DataFrame({"A": [0.5]}, index=[dates[0]]),
        research_ledger=SimpleNamespace(
            effective_weights=ledger_weights,
            daily=pd.DataFrame(
                {
                    "trade_cost": 0.0,
                    "holding_cost": 0.0,
                    "executed_traded_notional": 0.0,
                },
                index=dates,
            ),
        ),
        costs=pd.Series(dtype=float),
    )
    raw = [{"config": SimpleNamespace(name="sleeve"), "result": result}]

    cube, instruments, *_ = PipelineRunner._build_effective_exposure_cube(
        raw, dates, ["sleeve"]
    )

    assert instruments == ["A"]
    np.testing.assert_allclose(cube[:, 0, 0], [0.0, 0.5, 0.6])


def test_turnover_budget_reserves_mandatory_dynamic_universe_exits():
    from optimization.constraints import TurnoverConstraint, turnover_transition
    from optimization.hierarchical_asset_risk_parity import (
        HierarchicalAssetRiskParityOptimizer,
    )

    current = pd.Series({"A": 0.10, "OLD": 0.20})
    previous, forced_exit = turnover_transition(current, ["A"])
    assert previous.to_dict() == {"A": 0.10}
    assert forced_exit == pytest.approx(0.20)

    optimizer = HierarchicalAssetRiskParityOptimizer()
    constrained = optimizer._apply_hard_constraints(
        pd.Series({"A": 0.50}),
        current,
        [TurnoverConstraint(limit=0.30)],
        current_drawdown=0.0,
    )
    assert constrained["A"] == pytest.approx(0.20)
    assert forced_exit + abs(constrained["A"] - current["A"]) == pytest.approx(0.30)


def test_meta_combination_keeps_audited_sleeve_costs_without_false_drift_netting():
    from optimization.costs import SimpleFuturesCost
    from pipeline.runner import PipelineRunner

    dates = pd.date_range("2024-01-02", periods=2, freq="B")
    decision_date = dates[0] - pd.offsets.BDay(1)
    sub_results = []
    for name, exposure in [("long", 1.0), ("short", -1.0)]:
        result = SimpleNamespace(
            weights_history=pd.DataFrame({"A": [exposure]}, index=[decision_date]),
            research_ledger=SimpleNamespace(
                effective_weights=pd.DataFrame(
                    {"A": [exposure, exposure]}, index=dates
                ),
                daily=pd.DataFrame(
                    {
                        "trade_cost": [0.001, 0.0],
                        "holding_cost": [0.0, 0.0],
                        "executed_traded_notional": [1.0, 0.0],
                    },
                    index=dates,
                ),
            ),
            costs=pd.Series([0.001, 0.0], index=dates),
            nav=pd.Series(
                1.0, index=pd.DatetimeIndex([decision_date]).append(dates)
            ),
        )
        sub_results.append({"config": SimpleNamespace(name=name), "result": result})

    runner = PipelineRunner.__new__(PipelineRunner)
    runner.cost_model = SimpleFuturesCost()
    runner.config = SimpleNamespace(
        # Turnover is enforced inside each sleeve and must not be inherited by
        # the aggregate underlying-exposure controller.
        optimization=SimpleNamespace(
            constraints=[{"type": "turnover", "limit": 0.30}]
        )
    )
    meta_cfg = SimpleNamespace(
        underlying_constraints=[],
        enforce_underlying_constraints=True,
        min_weight=0.0,
        max_weight=1.0,
    )
    # This research framework has no order/fill engine, so it preserves each
    # sleeve's audited cost instead of claiming exact cross-sleeve execution netting.
    returns = pd.DataFrame(
        {"long": [-0.001, 0.0], "short": [-0.001, 0.0]}, index=dates
    )
    desired = np.full((2, 2), 0.5)
    nav, _ = runner._combine_sleeve_path(
        returns, desired, sub_results, meta_cfg, 1.0
    )

    np.testing.assert_allclose(nav.to_numpy(), [1.0, 0.999, 0.999])
    np.testing.assert_allclose(runner._meta_cost_history.to_numpy(), [0.001, 0.0])
    np.testing.assert_allclose(
        runner._meta_underlying_weights_history["A"].to_numpy(), [0.0, 0.0]
    )


def test_dynamic_constraints_require_explicit_runtime_context():
    import cvxpy as cp

    from optimization.constraints import (
        DrawdownControlConstraint,
        LeverageConstraint,
    )

    variables = {"w": cp.Variable(2)}
    with pytest.raises(ValueError, match="vol-target.*context"):
        LeverageConstraint(limit=2.0, vol_target=0.10).apply(None, variables, None)
    with pytest.raises(ValueError, match="drawdown-control.*context"):
        DrawdownControlConstraint().apply(None, variables, None)


def test_factor_turnover_is_scale_invariant_and_position_based():
    from testing.turnover import TurnoverTest

    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    factor = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0], [1.0, 3.0, 2.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
        index=dates,
        columns=list("ABCD"),
    )
    original = TurnoverTest().run(factor)
    rescaled = TurnoverTest().run(factor * 1000.0 + 17.0)

    pd.testing.assert_series_equal(
        original.turnover_series, rescaled.turnover_series
    )
    assert original.turnover_series.between(0.0, 1.0).all()


def test_regression_wls_requires_ex_ante_weights_and_clips_them():
    from testing.regression import RegressionTest

    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    assets = pd.Index([f"A{i:02d}" for i in range(12)])
    factor = pd.DataFrame(
        np.tile(np.linspace(-1.0, 1.0, len(assets)), (len(dates), 1)),
        index=dates,
        columns=assets,
    )
    returns = factor * 0.02

    with pytest.raises(ValueError, match="sample_weights are required"):
        RegressionTest(weighted=True).run(factor, returns)

    extreme = pd.Series(np.geomspace(1e-9, 1e9, len(assets)), index=assets)
    clipped = RegressionTest._weights_for_date(extreme, dates[0], assets)
    assert clipped.mean() == pytest.approx(1.0)
    assert clipped.max() / clipped.min() <= 100.0 + 1e-8

    weighted = RegressionTest(weighted=True).run(
        factor, returns, sample_weights=extreme
    )
    np.testing.assert_allclose(weighted.factor_returns["factor_return"], 0.02)


def test_ridge_alpha_selection_is_time_ordered_and_coefficients_shrink():
    from alpha.ols import (
        RidgeModel,
        _fit_linear_coefficients,
        _select_ridge_alpha_time_series,
    )

    rng = np.random.default_rng(31)
    dates = pd.date_range("2022-01-03", periods=60, freq="B")
    assets = pd.Index([f"A{i:02d}" for i in range(12)])
    factor = pd.DataFrame(
        rng.normal(size=(len(dates), len(assets))), index=dates, columns=assets
    )
    second = factor + pd.DataFrame(
        rng.normal(scale=0.05, size=factor.shape), index=dates, columns=assets
    )
    returns = 0.02 * factor - 0.01 * second

    model = RidgeModel(ridge_alphas=[0.01, 0.1, 1.0], ridge_cv_folds=3)
    model.fit({"factor": factor, "second": second}, returns)
    assert model.selected_alpha_ in {0.01, 0.1, 1.0}

    X = np.column_stack([factor.to_numpy().ravel(), second.to_numpy().ravel()])
    y = returns.to_numpy().ravel()
    ols_coef, _ = _fit_linear_coefficients(
        X, y, fit_intercept=True, ridge_alpha=0.0
    )
    ridge_coef, _ = _fit_linear_coefficients(
        X, y, fit_intercept=True, ridge_alpha=10.0
    )
    assert np.linalg.norm(ridge_coef) < np.linalg.norm(ols_coef)

    row_dates = np.repeat(dates.to_numpy(), len(assets))
    selected_before_perturbation = _select_ridge_alpha_time_series(
        X, y, row_dates, [0.01, 0.1, 1.0], fit_intercept=True, n_folds=3
    )
    assert selected_before_perturbation in {0.01, 0.1, 1.0}


def test_lasso_is_sparse_sector_model_and_default_remains_ridge():
    from pathlib import Path

    from alpha.ols import (
        SectorGroupedLassoModel,
        _fit_lasso_coefficients,
        _select_lasso_alpha_time_series,
    )
    from core.config import load_config
    from core.registry import create

    rng = np.random.default_rng(77)
    X = rng.normal(size=(500, 4))
    y = 0.4 + 1.5 * X[:, 0] - 0.8 * X[:, 2] + rng.normal(scale=0.03, size=500)
    coef, intercept = _fit_lasso_coefficients(
        X, y, fit_intercept=True, lasso_alpha=0.05
    )
    assert coef[1] == pytest.approx(0.0, abs=1e-12)
    assert coef[3] == pytest.approx(0.0, abs=1e-12)
    assert coef[0] > 1.3 and coef[2] < -0.6
    assert intercept == pytest.approx(0.4, abs=0.02)

    dates = pd.date_range("2024-01-02", periods=40, freq="B")
    row_dates = np.repeat(dates.to_numpy(), 12)
    selected = _select_lasso_alpha_time_series(
        X[:480], y[:480], row_dates,
        [1e-4, 1e-3, 1e-2], fit_intercept=True, n_folds=3,
    )
    assert selected in {1e-4, 1e-3, 1e-2}

    factor = pd.DataFrame(
        np.tile(np.asarray([-1.0, 1.0]), (len(dates), 1)),
        index=dates,
        columns=["RB", "HC"],
    )
    model = create(
        "return_model",
        "sector_grouped_lasso",
        min_samples_per_sector=1,
        lasso_alphas=[1e-6, 1e-5],
    )
    assert isinstance(model, SectorGroupedLassoModel)
    model.fit({"signal": factor}, factor * 0.02)
    predicted = model.predict({"signal": factor}, pd.Index(factor.columns), dates[-1])
    assert np.sign(predicted.loc["RB"]) == -1
    assert np.sign(predicted.loc["HC"]) == 1
    assert model.selected_alpha_["global"] in {1e-6, 1e-5}

    root = Path(__file__).resolve().parents[1]
    config = load_config(str(root / "config" / "default.yaml"))
    assert config.alpha.type == "sector_grouped_ridge"


def test_meta_robust_baselines_return_bounded_diversified_weights():
    from optimization.meta_optimizer import MetaOptimizer

    covariance = np.array(
        [[0.01, 0.002, 0.001], [0.002, 0.04, 0.006], [0.001, 0.006, 0.09]]
    )
    optimizer = MetaOptimizer(min_weight=0.05, max_weight=0.8)
    inverse_vol = optimizer._inverse_volatility(covariance, 3)
    hrp = optimizer._hrp(covariance, 3)
    shrunk = optimizer._shrink_covariance(covariance)

    for weights in (inverse_vol, hrp):
        assert weights.sum() == pytest.approx(1.0)
        assert np.all(weights >= 0.05 - 1e-8)
        assert np.all(weights <= 0.8 + 1e-8)
    assert inverse_vol[0] > inverse_vol[1] > inverse_vol[2]
    assert np.allclose(shrunk, shrunk.T)
    assert np.linalg.eigvalsh(shrunk).min() > 0


def test_factor_family_governance_caps_only_within_each_family():
    from research.governance import select_candidates_by_family

    candidates = [
        {"name": "momentum_5d", "best_t": 5.0},
        {"name": "momentum_20d", "best_t": 4.0},
        {"name": "roll_yield_20d", "best_t": 3.0},
        {"name": "basis_ratio_20d", "best_t": 2.0},
    ]
    selected, audit = select_candidates_by_family(
        candidates,
        default_cap=1,
        family_caps={"carry": 2},
    )
    names = {row["name"] for row in selected}

    assert names == {"momentum_5d", "roll_yield_20d", "basis_ratio_20d"}
    assert audit["selected_by_family"] == {"carry": 2, "trend": 1}
    assert audit["rejected"][0]["name"] == "momentum_20d"


def test_family_equal_weight_alpha_balances_families_not_factor_counts():
    from alpha.family import FamilyEqualWeightModel

    dates = pd.date_range("2023-01-02", periods=30, freq="B")
    assets = pd.Index([f"A{i:02d}" for i in range(12)])
    base = pd.DataFrame(
        np.tile(np.linspace(-1.0, 1.0, len(assets)), (len(dates), 1)),
        index=dates,
        columns=assets,
    )
    factors = {"trend_one": base, "trend_two": base, "carry_one": -base}
    returns = base * 0.01
    model = FamilyEqualWeightModel(
        family_map={
            "trend_one": "trend",
            "trend_two": "trend",
            "carry_one": "carry",
        }
    ).fit(factors, returns)
    current = {name: frame.loc[[dates[-1]]] for name, frame in factors.items()}
    prediction = model.predict(current, assets, dates[-1])

    assert np.isfinite(prediction).all()
    assert prediction.corr(base.loc[dates[-1]]) > 0.99


def test_sector_top_n_hysteresis_retains_boundary_name():
    from optimization.asset_selection import SectorForecastSelector

    selector = SectorForecastSelector(
        mode="hysteresis_top_n",
        top_n_per_side=1,
        exit_buffer=1,
        sector_map={name: "sector" for name in list("ABCD")},
    )
    first = selector.apply(pd.Series({"A": 4.0, "B": 3.0, "C": -2.0, "D": -1.0}))
    second = selector.apply(pd.Series({"A": 3.5, "B": 4.0, "C": -1.0, "D": -2.0}))

    assert set(first[first != 0].index) == {"A", "C"}
    # New leaders enter while prior leaders remain inside Top-(N+buffer).
    assert set(second[second != 0].index) == {"A", "B", "C", "D"}


def test_soft_sector_quota_equalizes_forecast_gross_by_sector():
    from optimization.asset_selection import SectorForecastSelector

    selector = SectorForecastSelector(
        mode="soft_quota",
        sector_map={"A": "one", "B": "one", "C": "two"},
    )
    selected = selector.apply(pd.Series({"A": 10.0, "B": -5.0, "C": 1.0}))
    assert selected[["A", "B"]].abs().sum() == pytest.approx(0.5)
    assert selected[["C"]].abs().sum() == pytest.approx(0.5)


def test_rebalance_dates_use_actual_last_trading_day():
    from pipeline.runner import PipelineRunner

    calendar = pd.date_range("2024-03-01", "2024-04-05", freq="B")
    weekly = PipelineRunner._rebalance_dates_from_calendar(calendar, "weekly")
    monthly = PipelineRunner._rebalance_dates_from_calendar(calendar, "monthly")

    assert all(date.weekday() < 5 for date in weekly)
    assert pd.Timestamp("2024-03-29") in monthly
    assert pd.Timestamp("2024-03-31") not in monthly
    with pytest.raises(ValueError, match="unsupported rebalance frequency"):
        PipelineRunner._rebalance_dates_from_calendar(calendar, "monthyl")


def test_pipeline_runner_rejects_intraday_frequency_before_initialization():
    from core.config import load_config
    from pipeline.runner import PipelineRunner

    with pytest.raises(ValueError, match="daily-only"):
        PipelineRunner(config=load_config("config/default.yaml"), frequency="15min")


def test_pipeline_runner_rejects_a_separate_universe_contract():
    from core.config import FrameworkConfig
    from pipeline.runner import PipelineRunner

    with pytest.raises(ValueError, match="FRAMEWORK_UNIVERSE"):
        PipelineRunner(config=FrameworkConfig(universe=["RB"]))


def test_parquet_factory_rejects_stale_generic_range_cache():
    from core.config import load_config
    from data.manager import DataManager

    config = load_config("config/default.yaml")
    config.data.cache = {"enabled": True, "backend": "parquet", "path": "./cache"}
    with pytest.raises(ValueError, match="not source-fingerprinted"):
        DataManager.from_config(config)


def test_horizon_ensemble_adds_only_configured_neighbour():
    from core.config import load_config
    from workflows.walkforward import _horizon_targets

    config = load_config("config/default.yaml")
    config.horizon_ensemble.enabled = True
    config.horizon_ensemble.neighbor_count = 1
    config.horizon_ensemble.max_log_distance = 1.0
    targets = _horizon_targets(config, 10)

    assert [target.name for target in targets] == ["mid_term", "short_term"]


def test_canonical_sector_map_is_shared_by_research_alpha_and_execution():
    from core.sectors import SECTOR_MAP, SECTOR_NAMES, instruments_in_sectors
    from alpha.ols import SectorGroupedOLSModel
    from factors.library.cross_commodity import SECTOR_MAP as factor_sector_map
    from factors.library.intraday import _SECTOR_MAP as intraday_sector_map
    from strategies.combined import SECTOR_OF as production_selection_groups
    from workflows.factor_adaptivity import SECTOR_MAP as research_sector_map

    assert SECTOR_MAP == SectorGroupedOLSModel._SECTOR_MAP
    assert SECTOR_MAP == factor_sector_map == intraday_sector_map == research_sector_map
    assert SECTOR_MAP["CU"] == "nonferrous"
    assert SECTOR_MAP["AU"] == "precious"
    assert SECTOR_MAP["IF"] == "stock_index"
    assert SECTOR_MAP["T"] == "bond"
    assert SECTOR_MAP["SA"] == "energy"
    assert production_selection_groups["SA"] == "能化"
    assert production_selection_groups["AU"] == "有色"
    assert production_selection_groups["T"] == "金融"
    assert {"stock_index", "bond", "nonferrous", "precious"}.issubset(
        SECTOR_NAMES
    )
    assert {"financial", "metal"}.isdisjoint(SECTOR_NAMES)
    instruments = ["CU", "AU", "IF", "T", "RB"]
    assert instruments_in_sectors(instruments, ["nonferrous"]) == ["CU"]
    assert instruments_in_sectors(instruments, ["precious"]) == ["AU"]
    assert instruments_in_sectors(instruments, ["stock_index"]) == ["IF"]
    assert instruments_in_sectors(instruments, ["bond"]) == ["T"]


def test_valid_period_ensemble_uses_only_approved_horizons():
    from core.period import approved_horizon_ensemble

    selected = approved_horizon_ensemble(
        20, "5|20|40", enabled=True, neighbor_count=2,
        max_log_distance=1.5,
    )
    assert selected == [20, 40, 5]
    assert 10 not in selected


def test_valid_sector_universe_gate_excludes_unconfirmed_precious_assets():
    from pipeline.runner import PipelineRunner

    schedule = {
        pd.Timestamp("2025-01-01"): pd.Index(["CU", "AU", "M", "IF"])
    }
    filtered = PipelineRunner._restrict_universe_schedule_to_sectors(
        schedule, {"nonferrous", "agri"}
    )
    assert list(filtered[pd.Timestamp("2025-01-01")]) == ["CU", "M"]


def test_asset_selector_hard_gates_optimizer_universe():
    from backtest.engine import Backtester

    class CapturingOptimizer:
        def __init__(self):
            self.universe = None

        def optimize(self, expected, risk, current, constraints, costs, date,
                     universe, **kwargs):
            self.universe = pd.Index(universe)
            return pd.Series(0.1, index=universe)

    backtester = Backtester(cost_model=None)
    backtester.asset_selector = object()
    optimizer = CapturingOptimizer()
    universe = pd.Index(["CU", "AU", "M"])
    target = backtester._optimize_period(
        pd.Series({"CU": 0.2, "AU": 0.0, "M": -0.1}),
        risk_model=object(),
        current_weights=pd.Series(0.0, index=universe),
        constraints=[],
        optimizer=optimizer,
        date=pd.Timestamp("2025-01-02"),
        universe=universe,
        realized_vol=0.1,
    )
    assert list(optimizer.universe) == ["CU", "M"]
    assert target["AU"] == 0.0


def test_mean_variance_receives_declared_marginal_turnover_cost():
    from optimization.costs import SimpleFuturesCost, marginal_turnover_cost_rate
    from optimization.mean_variance import MeanVarianceOptimizer

    class DiagonalRisk:
        def covariance(self, date, universe):
            return pd.DataFrame(
                np.eye(len(universe)) * 0.001,
                index=universe,
                columns=universe,
            )

    universe = pd.Index(["A", "B"])
    costs = SimpleFuturesCost()
    assert marginal_turnover_cost_rate(
        costs, universe, pd.Timestamp("2025-01-01")
    ) == pytest.approx(0.0002)

    weights = MeanVarianceOptimizer(cost_penalty=0.5).optimize(
        pd.Series({"A": 0.01, "B": -0.01}),
        DiagonalRisk(),
        pd.Series(0.0, index=universe),
        [],
        costs,
        pd.Timestamp("2025-01-01"),
        universe,
    )
    assert weights["A"] > 1e-3
    assert weights["B"] < -1e-3


def test_risk_model_is_low_dimensional_psd_ordered_and_point_in_time():
    from risk.barra_futures import BarraFuturesModel

    dates, assets, returns, alpha = _risk_fixture()
    cutoff = dates[79]
    data = _RiskData(dates, assets, returns)

    model = BarraFuturesModel(estimation_window=252).estimate(data, alpha, returns)
    covariance = model.covariance(cutoff, assets[::-1])

    assert covariance.index.equals(assets[::-1])
    assert covariance.columns.equals(assets[::-1])
    assert np.isfinite(covariance.to_numpy()).all()
    assert np.linalg.eigvalsh(covariance.to_numpy()).min() >= -1e-10
    assert len(model._risk_factor_names) < len(assets)
    assert not set(alpha).intersection(model._risk_factor_names)
    assert set(model._style_factors).issubset(
        {"carry", "momentum", "volatility", "skewness", "liquidity"}
    )

    perturbed = returns.copy()
    perturbed.loc[perturbed.index > cutoff] *= 100.0
    model_perturbed = BarraFuturesModel(estimation_window=252).estimate(
        data, alpha, perturbed
    )
    covariance_perturbed = model_perturbed.covariance(cutoff, assets[::-1])
    np.testing.assert_allclose(
        covariance.to_numpy(), covariance_perturbed.to_numpy(), rtol=1e-10, atol=1e-12
    )


def test_risk_model_documented_asset_covariance_path_and_annualisation():
    from risk.barra_futures import BarraFuturesModel

    dates, assets, returns, _ = _risk_fixture(seed=13)
    # A valid momentum style but too few assets for the cross-sectional model
    # selects the documented asset-covariance path without hiding missing data.
    small_assets = assets[:3]
    small_returns = returns[small_assets]

    class EmptyData:
        def get(self, field, dates, universe):
            return pd.DataFrame(index=dates, columns=universe, dtype=float)

        def get_industry(self, dates, universe):
            return pd.DataFrame("other", index=dates, columns=universe, dtype=object)

        def get_contract_pair(self, field, dates, universe):
            return {"near": pd.DataFrame(), "far": pd.DataFrame()}

    model = BarraFuturesModel(style_factors=["momentum"]).estimate(
        EmptyData(), {}, small_returns
    )
    covariance = model.covariance(dates[-1], small_assets)
    assert model.last_covariance_mode == "asset_shrinkage"
    assert np.isfinite(covariance.to_numpy()).all()
    assert np.linalg.eigvalsh(covariance.to_numpy()).min() >= -1e-10
    assert np.count_nonzero(np.triu(covariance.to_numpy(), 1)) > 0

    weights = pd.Series([0.5, 0.3, 0.2], index=small_assets)
    expected = np.sqrt(float(weights @ covariance @ weights) * 252.0)
    assert model.portfolio_risk(weights, dates[-1]) == pytest.approx(expected)


def test_research_artifact_bundle_validates_time_config_and_file_hash(tmp_path):
    from research.artifacts import ResearchArtifactBundle, canonical_config_hash

    artifact_file = tmp_path / "factor_adaptivity_summary.csv"
    artifact_file.write_text(
        "factor,valid_sectors\nmomentum,energy|metal\n", encoding="utf-8"
    )
    config = {"market": "futures", "factors": ["momentum"], "date_range": {"start": "2024-01-01"}}
    config_hash = canonical_config_hash(config)
    bundle = ResearchArtifactBundle.create(
        tmp_path,
        artifact_id="fold_001",
        train_start="2021-01-01",
        train_end="2023-12-29",
        data_sha256="1" * 64,
        config_sha256=config_hash,
        code_sha256="2" * 64,
        files={"factor_adaptivity_summary": artifact_file},
    )
    loaded = ResearchArtifactBundle.load(
        tmp_path,
        decision_date="2024-01-02",
        expected_config_hash=config_hash,
    )
    assert loaded.artifact_id == bundle.artifact_id
    assert loaded.read_csv("factor_adaptivity_summary").iloc[0]["factor"] == "momentum"

    with pytest.raises(ValueError, match="must be before decision date"):
        ResearchArtifactBundle.load(tmp_path, decision_date="2023-12-29")
    with pytest.raises(ValueError, match="config hash"):
        ResearchArtifactBundle.load(tmp_path, expected_config_hash="3" * 64)

    artifact_file.write_text("factor,valid_sectors\nmomentum,agri\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        ResearchArtifactBundle.load(tmp_path)


def test_research_config_hash_ignores_only_runtime_report_directory():
    from research.artifacts import canonical_config_hash

    left = {
        "market": "futures",
        "backtest": {"report_dir": "runs/a", "rebalance_freq": 5},
    }
    right = {
        "market": "futures",
        "backtest": {"report_dir": "runs/b", "rebalance_freq": 5},
    }
    changed = {
        "market": "futures",
        "backtest": {"report_dir": "runs/b", "rebalance_freq": 10},
    }

    assert canonical_config_hash(left) == canonical_config_hash(right)
    assert canonical_config_hash(left) != canonical_config_hash(changed)
    assert left["backtest"]["report_dir"] == "runs/a"


def test_streamed_dataframe_hash_collection_matches_materialized_collection():
    from research.artifacts import (
        dataframe_collection_sha256,
        dataframe_hash_collection_sha256,
        dataframe_sha256,
    )

    frames = {
        "factor:alpha": pd.DataFrame(
            [[1.0, np.nan], [2.0, 3.0]],
            index=pd.date_range("2024-01-01", periods=2),
            columns=["RB", "CU"],
        ),
        "returns:5": pd.DataFrame(
            [[0.01, -0.02], [0.03, 0.04]],
            index=pd.date_range("2024-01-01", periods=2),
            columns=["RB", "CU"],
        ),
    }
    frame_hashes = {
        name: dataframe_sha256(frame) for name, frame in frames.items()
    }

    assert dataframe_hash_collection_sha256(frame_hashes) == (
        dataframe_collection_sha256(frames)
    )


def test_parallel_factor_engine_preserves_alignment_and_values(monkeypatch):
    import factors.engine as engine_module

    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    universe = pd.Index(["RB", "CU"])

    class Data:
        def prefetch(self, *args, **kwargs):
            return None

    def factor_class(name, offset):
        class ExampleFactor:
            def __init__(self):
                self.name = name

            def dependencies(self):
                return ["close"]

            def compute(self, data, requested_dates, requested_universe):
                values = np.arange(8, dtype=float).reshape(4, 2) + offset
                return pd.DataFrame(
                    values,
                    index=requested_dates[:4][::-1],
                    columns=requested_universe[::-1],
                )

        return ExampleFactor

    factors = {
        "parallel_alpha_a": factor_class("parallel_alpha_a", 0.0),
        "parallel_alpha_b": factor_class("parallel_alpha_b", 10.0),
    }
    monkeypatch.setattr(
        engine_module,
        "registry_get",
        lambda kind, name: factors[name],
    )

    sequential = engine_module.FactorEngine(Data()).compute_factors(
        list(factors), dates, universe, parallel=False
    )
    parallel = engine_module.FactorEngine(Data()).compute_factors(
        list(factors), dates, universe, parallel=True, max_workers=2
    )

    for name in factors:
        assert parallel[name].index.equals(dates)
        assert parallel[name].columns.equals(universe)
        pd.testing.assert_frame_equal(parallel[name], sequential[name])


def test_factor_engine_is_strict_by_default_and_tolerant_only_when_explicit(
    monkeypatch,
):
    import factors.engine as engine_module

    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    universe = pd.Index(["RB"])

    class Data:
        def prefetch(self, *args, **kwargs):
            return None

    class BrokenFactor:
        name = "broken_factor"

        def dependencies(self):
            return ["close"]

        def compute(self, data, requested_dates, requested_universe):
            raise RuntimeError("source exploded")

    monkeypatch.setattr(
        engine_module,
        "registry_get",
        lambda kind, name: BrokenFactor,
    )

    with pytest.raises(engine_module.FactorComputationError, match="broken_factor"):
        engine_module.FactorEngine(Data()).compute_factors(
            ["broken_factor"], dates, universe
        )

    tolerant = engine_module.FactorEngine(Data(), tolerant=True)
    result = tolerant.compute_factors(["broken_factor"], dates, universe)
    assert result["broken_factor"].isna().all().all()
    assert tolerant.failures
    assert tolerant.failures[-1]["factor"] == "broken_factor"

    class MissingFactor(BrokenFactor):
        name = "missing_factor"

        def compute(self, data, requested_dates, requested_universe):
            return pd.DataFrame(
                np.nan, index=requested_dates, columns=requested_universe
            )

    monkeypatch.setattr(
        engine_module,
        "registry_get",
        lambda kind, name: MissingFactor,
    )
    with pytest.raises(engine_module.FactorComputationError, match="no finite values"):
        engine_module.FactorEngine(Data()).compute_factors(
            ["missing_factor"], dates, universe
        )


def test_factor_engine_validates_optimized_spec_batch_outputs(monkeypatch):
    import factors.engine as engine_module
    import factors.spec_factor as spec_factor_module
    import factors.specs as specs_module

    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    universe = pd.Index(["RB"])
    monkeypatch.setattr(
        specs_module,
        "SPEC_BY_SLUG",
        {"spec_nan": {"slug": "spec_nan"}},
    )
    monkeypatch.setattr(
        spec_factor_module,
        "compute_spec_factors_batch",
        lambda *args, **kwargs: {
            "spec_nan": pd.DataFrame(np.nan, index=dates, columns=universe)
        },
    )

    with pytest.raises(engine_module.FactorComputationError, match="no finite values"):
        engine_module.FactorEngine(SimpleNamespace())._compute_spec_factors_optimized(
            ["spec_nan"], dates, universe
        )


def test_default_config_does_not_load_legacy_report_artifacts():
    from core.config import load_config

    config = load_config("config/default.yaml")
    assert config.research_artifacts.enabled is False
    assert config.research_artifacts.path == ""


def test_framework_config_rejects_unknown_keys_but_costs_remain_extensible():
    from core.config import FrameworkConfig

    with pytest.raises(ValueError, match="extra_forbidden|extra fields"):
        FrameworkConfig(backtset={})
    with pytest.raises(ValueError, match="extra_forbidden|extra fields"):
        FrameworkConfig(data={"sorce": "random"})

    config = FrameworkConfig(costs={"type": "custom", "custom_bps": 1.5})
    assert config.costs.custom_bps == 1.5


def test_benjamini_hochberg_controls_global_false_discovery_rate():
    from research.statistics import benjamini_hochberg

    q_values, rejected = benjamini_hochberg([0.001, 0.01, 0.04, 0.9], alpha=0.05)
    np.testing.assert_allclose(q_values, [0.004, 0.02, 0.053333333333, 0.9])
    np.testing.assert_array_equal(rejected, [True, True, False, False])


def test_vectorized_raw_ols_matches_daily_lstsq():
    from testing.regression import _vectorized_univariate_ols

    rng = np.random.default_rng(20260726)
    dates = pd.date_range("2024-01-01", periods=8, freq="B")
    assets = [f"A{i}" for i in range(12)]
    factor = pd.DataFrame(
        rng.normal(size=(len(dates), len(assets))), index=dates, columns=assets
    )
    returns = 0.002 + 0.03 * factor + pd.DataFrame(
        rng.normal(scale=0.001, size=factor.shape), index=dates, columns=assets
    )
    factor.iloc[2, :3] = np.nan
    returns.iloc[5, :2] = np.nan

    actual = _vectorized_univariate_ols(factor, returns, min_stocks=8)
    expected = []
    for date in dates:
        valid = factor.loc[date].notna() & returns.loc[date].notna()
        x = factor.loc[date, valid].to_numpy(dtype=float)
        y = returns.loc[date, valid].to_numpy(dtype=float)
        coefficient, _, _, _ = np.linalg.lstsq(
            np.column_stack([np.ones(len(x)), x]), y, rcond=None
        )
        expected.append(coefficient[1])
    np.testing.assert_allclose(actual.to_numpy(), expected, rtol=1e-12, atol=1e-12)


def test_global_bonferroni_selects_horizon_only_from_approved_tests():
    from workflows.research import _apply_global_bonferroni

    def period(p_value, t_stat):
        return {
            "ols_p_value": p_value,
            "ols_hac_t": t_stat,
            "ols_n": 100,
            "ols_beta": 0.001,
            "ic_hac_t": t_stat,
            "ic": 0.03,
            "ir_nw": 0.6,
            "ic_pos_ratio": 0.55,
        }

    results = [
        {
            "name": "approved",
            "all_periods": {
                "period_3": period(0.01, 2.6),
                "period_5": period(0.001, 3.4),
            },
        },
        {
            "name": "rejected",
            "all_periods": {
                "period_3": period(0.02, 2.4),
                "period_5": period(0.9, 0.1),
            },
        },
    ]

    n_hypotheses, cutoff = _apply_global_bonferroni(results)

    assert n_hypotheses == 4
    assert cutoff == pytest.approx(0.0125)
    assert results[0]["bonferroni_significant"] is True
    assert results[0]["best_period"] == 5
    assert results[1]["bonferroni_significant"] is False
    assert results[1]["best_period"] == 0
    assert results[1]["best_p_value"] == pytest.approx(1.0)


def test_bonferroni_gate_uses_raw_p_values_before_local_thresholds():
    from workflows.factor_adaptivity import _apply_multiple_testing

    def entry(p_value, passes=True):
        return {
            "p_value": p_value,
            "passes_thresholds": passes,
            "t": 4.0,
            "ic": 0.03,
            "ir": 0.2,
            "n_obs": 100,
        }

    results = {
        "signal": {
            "sectors": {
                "energy": {
                    "5": entry(0.01),
                    "10": entry(0.001, passes=False),
                }
            }
        },
        "noise": {
            "sectors": {"agri": {"5": entry(0.02), "10": entry(0.9)}}
        },
    }
    assert _apply_multiple_testing(
        results, method="bonferroni", alpha=0.05
    ) == 4
    # The raw cutoff is 0.0125. Statistical significance is computed first,
    # while economic/robustness thresholds remain a separate admission gate.
    energy = results["signal"]["sectors"]["energy"]
    assert energy["5"]["fdr_significant"] is True
    assert energy["5"]["is_valid"] is True
    assert energy["10"]["fdr_significant"] is True
    assert energy["10"]["is_valid"] is False
    assert results["noise"]["valid_sectors"] == []


def test_simes_combines_consistent_local_evidence():
    from research.statistics import simes_p_value

    assert simes_p_value([]) == 1.0
    assert simes_p_value([0.001, 0.002, 0.9, 1.0]) == pytest.approx(0.004)


def test_hierarchical_fdr_requires_factor_and_local_approval():
    from workflows.factor_adaptivity import _apply_multiple_testing

    def entry(p_value, passes=True):
        return {"p_value": p_value, "passes_thresholds": passes}

    results = {
        "strong": {
            "sectors": {"energy": {"5": entry(1e-6), "10": entry(2e-6)}}
        },
        "noise": {
            "sectors": {"energy": {"5": entry(0.4), "10": entry(0.8)}}
        },
    }
    assert _apply_multiple_testing(results, method="hierarchical") == 4
    assert results["strong"]["factor_fdr_significant"] is True
    assert results["strong"]["sectors"]["energy"]["5"]["is_valid"] is True
    assert results["noise"]["factor_fdr_significant"] is False
    assert results["noise"]["valid_sectors"] == []


def test_adaptivity_optimum_uses_only_fdr_approved_hypotheses():
    from workflows.factor_adaptivity import _apply_multiple_testing

    def entry(p_value, t_value, passes=True):
        return {
            "p_value": p_value, "passes_thresholds": passes,
            "t": t_value, "ic": t_value / 100.0, "ir": 0.2,
            "n_obs": 100,
        }

    results = {
        "signal": {
            "sectors": {
                "agri": {
                    "5": entry(1e-6, 4.0),
                    "40": entry(1e-9, 20.0, passes=False),
                },
                "energy": {"10": entry(2e-6, -3.0)},
            }
        }
    }
    _apply_multiple_testing(results, method="hierarchical")
    signal = results["signal"]
    assert signal["best_sector"] == "agri"
    assert signal["best_period"] == 5
    assert {row["sector"]: row["best_period"] for row in signal["sector_optima"]} == {
        "agri": 5, "energy": 10,
    }


def test_sector_model_can_zero_unvalidated_sectors():
    from alpha.ols import SectorGroupedOLSModel

    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    values = np.linspace(-1.0, 1.0, len(dates))
    factor = pd.DataFrame({"RB": values, "CU": values[::-1]}, index=dates)
    returns = factor * 0.01
    model = SectorGroupedOLSModel(
        min_samples_per_sector=1,
        sector_factor_map={"ferrous": ["signal"]},
        unmapped_sector_policy="zero",
    )
    model.fit({"signal": factor}, returns, pd.Index(["RB", "CU"]))

    prediction = model.predict(
        {"signal": factor}, pd.Index(["RB", "CU"]), dates[-1]
    )
    assert prediction["CU"] == pytest.approx(0.0)
    assert prediction["RB"] != pytest.approx(0.0)


def test_adaptivity_newey_west_lag_uses_tested_horizon(monkeypatch):
    import workflows.factor_adaptivity as adaptivity

    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    columns = pd.Index(["RB", "I", "FG", "JM"])
    rng = np.random.default_rng(21)
    factor = pd.DataFrame(rng.normal(size=(30, 4)), index=dates, columns=columns)
    returns = factor * 0.01 + pd.DataFrame(
        rng.normal(scale=0.001, size=(30, 4)), index=dates, columns=columns
    )
    observed = []

    def fake_newey_west(series, forward_period):
        observed.append(forward_period)
        return 1.0, 2.0

    monkeypatch.setattr(adaptivity, "_newey_west_ir", fake_newey_west)
    result = adaptivity._compute_ic_by_sector(
        factor,
        returns,
        {column: "ferrous" for column in columns},
        min_stocks=3,
        forward_period=20,
    )
    assert result
    assert observed == [20]


def test_ic_hac_correction_does_not_inflate_significance():
    from testing.ic_test import _newey_west_ir

    values = pd.Series(np.tile([0.20, -0.10], 100) + 0.01)
    _, hac_t = _newey_west_ir(values, forward_period=40)
    centered = values.to_numpy() - values.mean()
    iid_variance = float(centered @ centered / len(values))
    iid_t = float(values.mean() / np.sqrt(iid_variance / len(values)))

    assert np.isfinite(hac_t)
    assert abs(hac_t) <= abs(iid_t) + 1e-12


def test_nested_walkforward_assigns_only_training_approved_factors():
    from core.config import load_config
    from workflows.walkforward import _assign_fold_factors

    config = load_config("config/default.yaml")

    class Bundle:
        def has(self, logical_name):
            return False

        def read_csv(self, logical_name, **kwargs):
            assert logical_name == "factor_adaptivity_summary"
            return pd.DataFrame(
                {
                    "factor": ["f_short", "f_mid", "f_long", "rejected"],
                    "best_period": [5, 10, 40, 10],
                    "n_valid_sectors": [2, 1, 3, 0],
                    "best_q": [0.01, 0.02, 0.03, 0.001],
                    "best_t": [3.0, 2.5, -2.2, 10.0],
                }
            )

    counts = _assign_fold_factors(config, Bundle())
    selected = {factor for sub in config.sub_portfolios for factor in sub.factors}
    assert selected == {"f_short", "f_mid", "f_long"}
    assert "rejected" not in selected
    assert sum(counts.values()) == 3


def test_nested_walkforward_routes_factor_by_each_sector_horizon():
    from core.config import load_config
    from workflows.walkforward import _assign_fold_factors

    config = load_config("config/default.yaml")

    class Bundle:
        def has(self, logical_name):
            return logical_name == "factor_sector_selection"

        def read_csv(self, logical_name, **kwargs):
            if logical_name == "factor_adaptivity_summary":
                return pd.DataFrame({
                    "factor": ["multi_horizon"], "best_period": [5],
                    "n_valid_sectors": [2], "best_q": [0.01], "best_t": [3.0],
                })
            return pd.DataFrame({
                "factor": ["multi_horizon", "multi_horizon", "multi_horizon"],
                "sector": ["agri", "energy", "nonferrous"],
                "best_period": [5, 40, 20],
            })

    _assign_fold_factors(config, Bundle(), drop_empty_sleeves=True)
    selected = {sub.name: sub.factors for sub in config.sub_portfolios}
    assert "multi_horizon" in selected["short_term"]
    assert "multi_horizon" in selected["long_term"]
    exact = [sub for sub in config.sub_portfolios if sub.holding_period == 20]
    assert len(exact) == 1
    assert exact[0].factors == ["multi_horizon"]


def test_practical_profile_routes_valid_horizon_ensemble_and_shortens_retraining():
    from core.config import load_config
    from workflows.walkforward import (
        _apply_practical_profile,
        _assign_fold_factors,
    )

    config = load_config("config/default.yaml")
    _apply_practical_profile(config)

    class Bundle:
        def has(self, logical_name):
            return logical_name == "factor_sector_selection"

        def read_csv(self, logical_name, **kwargs):
            if logical_name == "factor_adaptivity_summary":
                return pd.DataFrame({
                    "factor": ["stable_vol"], "best_period": [20],
                    "n_valid_sectors": [1], "best_q": [0.01], "best_t": [4.0],
                })
            return pd.DataFrame({
                "factor": ["stable_vol"], "sector": ["agri"],
                "best_period": [20], "valid_periods": ["5|20|40"],
            })

    _assign_fold_factors(config, Bundle(), drop_empty_sleeves=True)
    by_period = {int(sub.holding_period): sub for sub in config.sub_portfolios}
    assert set(by_period) == {5, 20, 40}
    assert all(sub.factors == ["stable_vol"] for sub in by_period.values())
    assert by_period[20].retrain_freq == 5
    assert config.asset_selection.restrict_to_valid_sectors is True


def test_nested_walkforward_cluster_deduplication_selects_best_member():
    from core.config import load_config
    from workflows.walkforward import _assign_fold_factors

    config = load_config("config/default.yaml")

    class Bundle:
        def read_csv(self, logical_name, **kwargs):
            assert logical_name == "factor_adaptivity_summary"
            return pd.DataFrame(
                {
                    "factor": ["f_short_a", "f_short_b", "f_mid", "f_long"],
                    "best_period": [5, 5, 10, 40],
                    "n_valid_sectors": [1, 2, 1, 1],
                    "best_q": [0.02, 0.01, 0.03, 0.04],
                    "best_t": [4.0, 3.0, 2.5, -2.2],
                }
            )

        def has(self, logical_name):
            return logical_name == "factor_correlation"

        def read_json(self, logical_name):
            assert logical_name == "factor_correlation"
            return {
                "clusters": [
                    {
                        "factors": [
                            {"name": "f_short_a"},
                            {"name": "f_short_b"},
                        ]
                    }
                ]
            }

    _assign_fold_factors(config, Bundle(), deduplicate_clusters=True)
    selected = {factor for sub in config.sub_portfolios for factor in sub.factors}
    assert selected == {"f_short_b", "f_mid", "f_long"}
    assert "f_short_a" not in selected


def test_walkforward_coverage_grace_uses_supplied_exchange_calendar():
    from workflows.walkforward import _calendar_coverage_bounds

    calendar = pd.bdate_range("2023-07-01", "2024-06-30").difference(
        pd.DatetimeIndex(["2023-07-05", "2024-06-20"])
    )
    latest_start, earliest_end = _calendar_coverage_bounds(
        calendar, "2023-07-01", "2024-06-30", grace_bars=5
    )
    assert latest_start == calendar[5]
    assert earliest_end == calendar[-6]


def test_nested_walkforward_fold_test_ranges_are_unique(monkeypatch, tmp_path):
    """Rolling WF must produce non-overlapping test ranges."""
    import workflows.walkforward as validation
    from core.config import load_config

    observed = []

    def fail_after_record(*args, **kwargs):
        observed.append(kwargs["name"])
        raise RuntimeError("stop test")

    monkeypatch.setattr(validation, "_build_fold_bundle", fail_after_record)
    # Force sample assessment to pass so the mock gets called
    from unittest.mock import MagicMock
    mock_assessment = MagicMock()
    mock_assessment.sufficient = True
    monkeypatch.setattr(
        "research.sample_policy.assess_sample_counts",
        lambda *a, **kw: mock_assessment,
    )
    results = validation.rolling_walk_forward(
        load_config("config/default.yaml"),
        run_root=tmp_path / "run",
        candidate_factors=["momentum_20d"],
        build_correlation=False,
        calendar=pd.bdate_range("2018-01-01", "2024-12-31"),
    )
    assert len(observed) > 0, "expected at least one fold to pass sampling"
    # Extract fold names: "xxx_折N" -> extract "折N"
    fold_names = [name.rsplit("_", 1)[-1] for name in observed]
    assert len(fold_names) == len(set(fold_names)), "fold names must be unique"
    assert len(results) == len(observed)


def test_roll_adjustment_uses_latest_common_close_and_fails_without_overlap():
    from data.continuous_contract import (
        RolloverAdjustmentError,
        _compute_rollover_ratio,
    )

    old = pd.DataFrame(
        {"close": [100.0, 101.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    new = pd.DataFrame(
        {"close": [200.0]}, index=pd.to_datetime(["2024-01-02"])
    )
    ratio = _compute_rollover_ratio(
        {"OLD": old, "NEW": new}, "OLD", "NEW", pd.Timestamp("2024-01-03")
    )
    assert ratio == pytest.approx(0.5)

    disjoint = pd.DataFrame(
        {"close": [200.0]}, index=pd.to_datetime(["2024-01-04"])
    )
    with pytest.raises(RolloverAdjustmentError, match="no common close"):
        _compute_rollover_ratio(
            {"OLD": old, "NEW": disjoint},
            "OLD", "NEW", pd.Timestamp("2024-01-04"),
        )


def test_walkforward_assignment_excludes_insufficient_sample_factor():
    from core.config import load_config
    from workflows.walkforward import _assign_fold_factors

    config = load_config("config/default.yaml")

    class Bundle:
        def has(self, logical_name):
            return False

        def read_csv(self, logical_name, **kwargs):
            return pd.DataFrame({
                "factor": ["eligible", "observation"],
                "best_period": [5, 5],
                "n_valid_sectors": [1, 1],
                "best_q": [0.01, 0.001],
                "best_t": [3.0, 10.0],
                "sample_sufficient": [True, False],
            })

    _assign_fold_factors(config, Bundle(), drop_empty_sleeves=True)
    selected = {factor for sub in config.sub_portfolios for factor in sub.factors}
    assert selected == {"eligible"}
