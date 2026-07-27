from __future__ import annotations

from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
import pytest


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


def test_prediction_and_optimizer_failures_hold_previous_weights():
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

    prediction = backtester._predict_returns(BrokenAlpha(), factors, dates[-1], universe)
    assert prediction is None
    result = backtester._optimize_period(
        pd.Series([0.01, -0.01], index=universe),
        SimpleNamespace(),
        previous,
        [],
        BrokenOptimizer(),
        dates[-1],
        universe,
        0.1,
    )
    pd.testing.assert_series_equal(result, previous)
    assert {entry["stage"] for entry in backtester._failure_ledger} == {
        "prediction",
        "optimization",
    }


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


def test_subportfolio_weight_history_becomes_effective_next_trading_day():
    from pipeline.runner import PipelineRunner

    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    decision_dates = pd.DatetimeIndex([dates[0] - pd.offsets.BDay(1), dates[1]])
    result = SimpleNamespace(
        weights_history=pd.DataFrame(
            {"A": [1.0, 2.0]}, index=decision_dates
        ),
        costs=pd.Series(dtype=float),
    )
    raw = [{"config": SimpleNamespace(name="sleeve"), "result": result}]
    cube, instruments, _ = PipelineRunner._build_effective_exposure_cube(
        raw, dates, ["sleeve"]
    )

    assert instruments == ["A"]
    np.testing.assert_allclose(cube[:, 0, 0], [1.0, 1.0, 2.0])


def test_meta_combination_nets_opposite_bottom_trades_and_costs():
    from optimization.costs import SimpleFuturesCost
    from pipeline.runner import PipelineRunner

    dates = pd.date_range("2024-01-02", periods=2, freq="B")
    decision_date = dates[0] - pd.offsets.BDay(1)
    sub_results = []
    for name, exposure in [("long", 1.0), ("short", -1.0)]:
        result = SimpleNamespace(
            weights_history=pd.DataFrame({"A": [exposure]}, index=[decision_date]),
            costs=pd.Series([0.001, 0.0], index=dates),
        )
        sub_results.append({"config": SimpleNamespace(name=name), "result": result})

    runner = PipelineRunner.__new__(PipelineRunner)
    runner.cost_model = SimpleFuturesCost(commission_rate=0.001, slippage=0.0)
    runner.config = SimpleNamespace(
        optimization=SimpleNamespace(constraints=[])
    )
    meta_cfg = SimpleNamespace(
        underlying_constraints=[],
        enforce_underlying_constraints=True,
        net_underlying_costs=True,
        min_weight=0.0,
        max_weight=1.0,
    )
    # Each standalone sleeve return contains its own opening cost. Their equal
    # and opposite bottom positions net to zero in the aggregate portfolio.
    returns = pd.DataFrame(
        {"long": [-0.001, 0.0], "short": [-0.001, 0.0]}, index=dates
    )
    desired = np.full((2, 2), 0.5)
    nav, _ = runner._combine_sleeve_path(
        returns, desired, sub_results, meta_cfg, 1.0
    )

    np.testing.assert_allclose(nav.to_numpy(), [1.0, 1.0])
    np.testing.assert_allclose(runner._meta_cost_history.to_numpy(), [0.0, 0.0])
    np.testing.assert_allclose(
        runner._meta_underlying_weights_history["A"].to_numpy(), [0.0, 0.0]
    )


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

    ols = RegressionTest(weighted=False).run(factor, returns)
    weighted_without_input = RegressionTest(weighted=True).run(factor, returns)
    pd.testing.assert_frame_equal(
        ols.factor_returns, weighted_without_input.factor_returns
    )

    extreme = pd.Series(np.geomspace(1e-9, 1e9, len(assets)), index=assets)
    clipped = RegressionTest._weights_for_date(extreme, dates[0], assets)
    assert clipped.mean() == pytest.approx(1.0)
    assert clipped.max() / clipped.min() <= 100.0 + 1e-8


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
    from signals.selection import SectorForecastSelector

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
    from signals.selection import SectorForecastSelector

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
    from workflows.factor_adaptivity import SECTOR_MAP as research_sector_map

    assert SECTOR_MAP == SectorGroupedOLSModel._SECTOR_MAP
    assert SECTOR_MAP == factor_sector_map == research_sector_map
    assert SECTOR_MAP["CU"] == "nonferrous"
    assert SECTOR_MAP["AU"] == "precious"
    assert SECTOR_MAP["IF"] == "stock_index"
    assert SECTOR_MAP["T"] == "bond"
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


def test_mean_variance_uses_transaction_cost_in_return_units():
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
    costs = SimpleFuturesCost(commission_rate=0.0001, slippage=0.001)
    assert marginal_turnover_cost_rate(
        costs, universe, pd.Timestamp("2025-01-01")
    ) == pytest.approx(0.0011)

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


def test_risk_model_asset_covariance_fallback_and_annualisation():
    from risk.barra_futures import BarraFuturesModel

    dates, assets, returns, _ = _risk_fixture(seed=13)
    # No sector/style data and fewer assets than factor-regression requirements
    # force the asset-covariance path without producing a constant diagonal.
    small_assets = assets[:3]
    small_returns = returns[small_assets]

    class EmptyData:
        def get(self, field, dates, universe):
            return pd.DataFrame(index=dates, columns=universe, dtype=float)

        def get_industry(self, dates, universe):
            return pd.DataFrame(index=dates, columns=universe, dtype=object)

        def get_contract_pair(self, field, dates, universe):
            return {"near": pd.DataFrame(), "far": pd.DataFrame()}

    model = BarraFuturesModel(style_factors=["carry"]).estimate(
        EmptyData(), {}, small_returns
    )
    covariance = model.covariance(dates[-1], small_assets)
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


def test_walkforward_coverage_grace_starts_from_first_business_day():
    from workflows.walkforward import _business_day_coverage_bounds

    latest_start, earliest_end = _business_day_coverage_bounds(
        "2023-07-01", "2024-06-30", grace_days=5
    )
    assert latest_start == pd.Timestamp("2023-07-10")
    assert earliest_end == pd.Timestamp("2024-06-21")


def test_nested_walkforward_fold_test_ranges_are_unique(monkeypatch, tmp_path):
    # Failure before artifact construction would indicate a duplicate fold range.
    import workflows.walkforward as validation
    from core.config import load_config

    observed = []

    def fail_after_record(*args, **kwargs):
        observed.append(kwargs["name"])
        raise RuntimeError("stop test")

    monkeypatch.setattr(validation, "_build_fold_bundle", fail_after_record)
    results = validation.walk_forward_4fold(
        load_config("config/default.yaml"),
        run_root=tmp_path / "run",
        candidate_factors=["momentum_20d"],
        build_correlation=False,
    )
    assert [name.rsplit("_", 1)[-1] for name in observed] == [
        "段1", "段2", "段3", "段4"
    ]
    assert len(results) == 4


def test_ddb_dominant_contract_is_t_minus_one_and_roll_is_continuous():
    from data.ddb_source import DDBSource

    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    contracts = ["RB2401.SHF", "RB2405.SHF"]
    volume = pd.DataFrame(
        [[100, 10], [20, 200], [10, 300], [5, 400]],
        index=dates,
        columns=contracts,
    )
    close = pd.DataFrame(
        [[100, 200], [101, 202], [102, 204], [103, 206]],
        index=dates,
        columns=contracts,
    )
    source = DDBSource({"dominant_lag_days": 1})
    schedule = source._build_daily_dominant_contract(volume)["RB"]
    assert pd.isna(schedule.iloc[0])
    assert schedule.iloc[1] == "RB2401.SHF"
    assert schedule.iloc[2] == "RB2405.SHF"

    scales = source._build_roll_scales(close, {"RB": schedule})
    continuous = source._aggregate_to_root_by_daily_volume(
        close,
        {"RB": schedule},
        roll_scales=scales,
        adjust_prices=True,
    )["RB"]
    # Roll-day return equals the new contract's own return from t-1 to t.
    assert continuous.iloc[2] / continuous.iloc[1] - 1 == pytest.approx(204 / 202 - 1)


def test_ddb_minute_bars_filter_to_one_t_minus_one_contract(monkeypatch):
    from data.ddb_source import DDBSource

    source = DDBSource({"dominant_lag_days": 1})
    rows = []
    for day, volumes in [
        ("2024-01-01", {"RB2401": 100, "RB2405": 10}),
        ("2024-01-02", {"RB2401": 5, "RB2405": 500}),
    ]:
        for contract, volume in volumes.items():
            for minute, price in [("14:59:00", 100.0), ("15:00:00", 101.0)]:
                rows.append(
                    {
                        "TradeDate": day,
                        "InstrumentID": contract,
                        "Time": minute,
                        "OpenPrice": price,
                        "HighPrice": price,
                        "LowPrice": price,
                        "ClosePrice": price,
                        "Volume": volume,
                        "Turnover": volume * price,
                    }
                )
    monkeypatch.setattr(source, "_query", lambda script: pd.DataFrame(rows))
    bars = source.fetch_minute_bars(
        ["RB"], "2024-01-02", "2024-01-02", frequency="1min"
    )
    assert len(bars) == 2
    assert bars.index.get_level_values("root").unique().tolist() == ["RB"]
    # Previous day selected RB2401, whose requested-day volume is 5 per bar.
    assert bars["volume"].tolist() == [5, 5]


def test_intraday_features_use_amount_and_actual_thirty_minutes():
    from data.ddb_source import DDBSource

    datetimes = pd.to_datetime(
        ["2024-01-02 14:00", "2024-01-02 14:30", "2024-01-02 15:00"]
    )
    index = pd.MultiIndex.from_arrays(
        [datetimes, ["RB", "RB", "RB"]], names=["datetime", "root"]
    )
    bars = pd.DataFrame(
        {
            "open": [90.0, 100.0, 130.0],
            "high": [90.0, 100.0, 130.0],
            "low": [90.0, 100.0, 130.0],
            "close": [90.0, 100.0, 130.0],
            "volume": [1.0, 1.0, 1.0],
            "amount": [90.0, 100.0, 130.0],
        },
        index=index,
    )
    features = DDBSource._compute_intraday_features_from_bars(
        bars, ["tail_momentum", "amihud_illiquidity"]
    )
    assert features["tail_momentum"].iloc[0, 0] == pytest.approx((130 - 100) / 130)
    expected_amihud = (abs(100 / 90 - 1) + abs(130 / 100 - 1)) / (90 + 100 + 130)
    assert features["amihud_illiquidity"].iloc[0, 0] == pytest.approx(expected_amihud)


def test_mysql_endpoint_failover_uses_ordered_fallback(monkeypatch):
    import data.mysql_source as mysql_module

    class Connection:
        def __init__(self, fails):
            self.fails = fails

        def __enter__(self):
            if self.fails:
                raise OSError("endpoint unavailable")
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, statement):
            return 1

    class Engine:
        def __init__(self, fails):
            self.fails = fails
            self.disposed = False

        def connect(self):
            return Connection(self.fails)

        def dispose(self):
            self.disposed = True

    created_urls = []

    def fake_create_engine(url, **kwargs):
        created_urls.append(url)
        return Engine("wind-primary" in url)

    monkeypatch.setattr(mysql_module, "create_engine", fake_create_engine)
    source = mysql_module.MySQLSource(
        {
            "endpoints": [
                {
                    "name": "wind",
                    "host": "wind-primary",
                    "user": "user",
                    "password": "pw",
                    "database": "wind",
                }
            ],
            "fallbacks": [
                {
                    "name": "aliyun-rds",
                    "host": "rds-fallback",
                    "user": "user",
                    "password": "pw",
                    "database": "wind",
                }
            ],
        }
    )
    assert source.engine is not None
    assert source.active_endpoint_name == "aliyun-rds"
    assert "wind-primary" in created_urls[0]
    assert "rds-fallback" in created_urls[1]


def test_ddb_source_constructs_through_framework_config():
    from core.config import load_config
    from data.manager import DataManager

    config = load_config("config/default.yaml")
    config.data.source = "ddb_futures"
    manager = DataManager.from_config(config)
    assert manager.source.__class__.__name__ == "DDBSource"
