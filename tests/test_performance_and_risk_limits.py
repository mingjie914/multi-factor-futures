from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _reference_stack(factors, returns):
    names = sorted(factors)
    panels = []
    for name in names:
        series = factors[name].stack()
        series.name = name
        panels.append(series)
    y = returns.stack()
    y.name = "fwd_ret"
    return pd.concat(panels, axis=1).join(y, how="inner").dropna().sort_index(level=0)


def test_fast_factor_stack_is_numerically_identical_with_missing_values():
    from factors.utils import stack_factors_and_returns

    rng = np.random.default_rng(10)
    dates = pd.date_range("2024-01-02", periods=25, freq="B")
    assets = pd.Index(["C", "A", "B"])
    factors = {
        name: pd.DataFrame(rng.normal(size=(25, 3)), index=dates, columns=assets)
        for name in ("z", "a", "m")
    }
    returns = pd.DataFrame(rng.normal(size=(25, 3)), index=dates, columns=assets)
    factors["m"].iloc[3, 1] = np.nan
    returns.iloc[7, 2] = np.nan

    merged, names, X, y, codes = stack_factors_and_returns(factors, returns)
    expected = _reference_stack(factors, returns)
    pd.testing.assert_frame_equal(merged, expected)
    np.testing.assert_array_equal(X, expected[names].to_numpy(dtype=float))
    np.testing.assert_array_equal(y, expected["fwd_ret"].to_numpy(dtype=float))
    _, expected_codes = np.unique(
        expected.index.get_level_values(0).values, return_inverse=True
    )
    np.testing.assert_array_equal(codes, expected_codes)


def test_incremental_ic_matches_full_window_recomputation():
    from alpha.ic_monitor import ICMonitor

    rng = np.random.default_rng(11)
    dates = pd.date_range("2022-01-03", periods=90, freq="B")
    assets = pd.Index([f"A{i}" for i in range(12)])
    factors = {
        name: pd.DataFrame(rng.normal(size=(90, 12)), index=dates, columns=assets)
        for name in ("f1", "f2", "f3")
    }
    returns = pd.DataFrame(rng.normal(size=(90, 12)), index=dates, columns=assets)
    monitor = ICMonitor(window=20)
    monitor.update({k: v.iloc[:60] for k, v in factors.items()}, returns.iloc[:60])
    monitor.update({k: v.iloc[5:70] for k, v in factors.items()}, returns.iloc[5:70])

    for name, factor in factors.items():
        expected = monitor._compute_daily_ic(factor.iloc[:70], returns.iloc[:70])
        pd.testing.assert_series_equal(monitor._ic_history[name], expected)


def test_batched_ic_preserves_ties_and_missing_value_semantics():
    from alpha.ic_monitor import ICMonitor

    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    columns = list("ABCDE")
    factors = {
        "f1": pd.DataFrame(
            [[1, 1, 2, 3, np.nan], [3, 2, 2, 1, 0], [np.nan] * 5],
            index=dates,
            columns=columns,
        ),
        "f2": pd.DataFrame(
            [[5, 4, 3, 2, 1], [1, np.nan, 1, 2, 3], [1, 2, 3, 4, 5]],
            index=dates,
            columns=columns,
        ),
    }
    returns = pd.DataFrame(
        [[1, 2, 2, 4, 5], [5, 4, np.nan, 2, 1], [2, 1, 3, 5, 4]],
        index=dates,
        columns=columns,
    )
    batch = ICMonitor._compute_daily_ic_batch(factors, returns)
    monitor = ICMonitor()
    for name, frame in factors.items():
        expected = monitor._compute_daily_ic(frame, returns)
        pd.testing.assert_series_equal(batch[name], expected)


def test_ic_uses_only_pairwise_complete_cross_section():
    from scipy.stats import spearmanr
    from alpha.ic_monitor import ICMonitor

    date = pd.Timestamp("2024-01-02")
    columns = list("ABCDE")
    factor = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, np.nan]], index=[date], columns=columns)
    returns = pd.DataFrame([[np.nan, 4.0, 3.0, 2.0, 1.0]], index=[date], columns=columns)

    actual = ICMonitor()._compute_daily_ic(factor, returns).loc[date]
    expected = spearmanr([2.0, 3.0, 4.0], [4.0, 3.0, 2.0]).statistic

    assert actual == pytest.approx(expected)


def test_ic_requires_three_pairwise_observations_and_inactivates_unavailable_factor():
    from alpha.ic_monitor import ICMonitor

    date = pd.Timestamp("2024-01-02")
    factor = pd.DataFrame([[1.0, 2.0, np.nan]], index=[date], columns=list("ABC"))
    returns = pd.DataFrame([[2.0, 1.0, 0.0]], index=[date], columns=list("ABC"))
    monitor = ICMonitor(decay_tolerance=2)

    monitor.update({"factor": factor}, returns)

    assert np.isnan(monitor._ic_history["factor"].loc[date])
    assert monitor.inactive_factors == {"factor"}


def test_vectorized_layered_backtest_matches_qcut_reference():
    from testing.layered import LayeredBacktest

    rng = np.random.default_rng(27)
    dates = pd.date_range("2024-01-02", periods=18, freq="B")
    assets = pd.Index([f"A{i}" for i in range(17)])
    factor = pd.DataFrame(
        rng.normal(size=(18, 17)), index=dates, columns=assets
    )
    returns = pd.DataFrame(
        rng.normal(size=(18, 17)), index=dates, columns=assets
    )
    factor.iloc[3, :4] = np.nan
    returns.iloc[5, :3] = np.nan

    actual = LayeredBacktest(n_groups=5).run(factor, returns)
    expected = {f"Q{i + 1}": [] for i in range(5)}
    valid_dates = []
    for date in dates:
        factor_row = factor.loc[date].dropna()
        return_row = returns.loc[date]
        if len(factor_row) < 10:
            continue
        labels = pd.qcut(
            factor_row.rank(method="first"), 5, labels=list(expected)
        )
        grouped = return_row[factor_row.index].groupby(
            labels, observed=False
        ).mean()
        for label in expected:
            expected[label].append(float(grouped.get(label, 0.0)))
        valid_dates.append(date)

    for label, values in expected.items():
        pd.testing.assert_series_equal(
            actual.group_returns[label],
            pd.Series(values, index=pd.DatetimeIndex(valid_dates)),
            check_freq=False,
        )


def test_layered_rejects_invalid_annualization_inputs():
    from testing.layered import LayeredBacktest

    with pytest.raises(ValueError, match="n_groups"):
        LayeredBacktest(n_groups=0)
    with pytest.raises(ValueError, match="holding_period"):
        LayeredBacktest(holding_period=0)
    with pytest.raises(ValueError, match="periods_per_year"):
        LayeredBacktest().run(
            pd.DataFrame([[1.0, 2.0]], columns=["A", "B"]),
            pd.DataFrame([[0.1, 0.2]], columns=["A", "B"]),
            periods_per_year=0,
        )


def test_layered_does_not_turn_missing_group_returns_into_zero():
    from testing.layered import LayeredBacktest

    factor = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], columns=list("ABCD"))
    returns = pd.DataFrame([[0.1, 0.2, np.nan, np.nan]], columns=list("ABCD"))

    result = LayeredBacktest(n_groups=2).run(factor, returns)

    assert result.group_returns["Q1"].iloc[0] == pytest.approx(0.15)
    assert pd.isna(result.group_returns["Q2"].iloc[0])
    assert pd.isna(result.group_returns["long_short"].iloc[0])


def _reference_ridge_choice(X, y, dates, alphas, fit_intercept, n_folds):
    from alpha.ols import _fit_linear_coefficients

    candidates = sorted(set(alphas))
    date_values = pd.DatetimeIndex(dates)
    unique_dates = date_values.unique().sort_values()
    initial = max(int(len(unique_dates) * 0.5), 10)
    fold_size = max((len(unique_dates) - initial) // n_folds, 1)
    losses = {alpha: [] for alpha in candidates}
    for fold in range(n_folds):
        train_end = initial + fold * fold_size
        valid_end = len(unique_dates) if fold == n_folds - 1 else min(
            train_end + fold_size, len(unique_dates)
        )
        if train_end >= valid_end:
            continue
        train = date_values <= unique_dates[train_end - 1]
        valid = (date_values > unique_dates[train_end - 1]) & (
            date_values <= unique_dates[valid_end - 1]
        )
        for alpha in candidates:
            coef, intercept = _fit_linear_coefficients(
                X[train], y[train], fit_intercept=fit_intercept, ridge_alpha=alpha
            )
            error = y[valid] - (X[valid] @ coef + intercept)
            losses[alpha].append(float(np.mean(error**2)))
    scores = {key: float(np.mean(value)) for key, value in losses.items() if value}
    return min(scores, key=lambda alpha: (scores[alpha], -alpha))


def test_ridge_cv_sufficient_statistics_preserve_selection():
    from alpha.ols import _select_ridge_alpha_time_series

    rng = np.random.default_rng(12)
    dates = pd.date_range("2021-01-04", periods=80, freq="B")
    row_dates = np.repeat(dates.to_numpy(), 10)
    X = rng.normal(size=(800, 18))
    y = X[:, :3] @ np.array([0.2, -0.1, 0.05]) + rng.normal(size=800)
    alphas = [0.0, 0.01, 0.1, 1.0, 10.0]
    expected = _reference_ridge_choice(X, y, row_dates, alphas, True, 3)
    actual = _select_ridge_alpha_time_series(
        X, y, row_dates, alphas, fit_intercept=True, n_folds=3
    )
    assert actual == expected


def test_dynamic_risk_caps_clip_asset_then_sector_without_renormalizing():
    from optimization.risk_limits import VolatilityRiskCapController

    names = pd.Index(["A", "B", "C"])
    covariance = pd.DataFrame(
        np.diag(np.array([0.20, 0.10, 0.25]) ** 2 / 252.0),
        index=names,
        columns=names,
    )
    controller = VolatilityRiskCapController(
        asset_vol_budget=0.025,
        sector_vol_budget=0.03,
        hard_asset_cap=1.0,
        gross_cap=2.0,
        net_cap=0.5,
        sector_map={"A": "metal", "B": "metal", "C": "agri"},
    )
    limited, diagnostics = controller.apply(
        pd.Series([0.40, 0.40, -0.40], index=names), covariance
    )
    assert limited["A"] <= 0.125 + 1e-10
    assert limited["B"] <= 0.25 + 1e-10
    assert abs(limited["C"]) <= 0.10 + 1e-10
    assert diagnostics["max_asset_vol_proxy"] <= 0.025 + 1e-10
    assert diagnostics["max_sector_standalone_vol"] <= 0.03 + 1e-10
    assert limited.abs().sum() < 1.20


def test_aggregate_dynamic_risk_scale_preserves_sleeve_mix():
    from optimization.risk_limits import VolatilityRiskCapController

    names = ["A", "B"]
    annual_covariance = np.diag([0.20**2, 0.10**2])
    controller = VolatilityRiskCapController(
        asset_vol_budget=0.03,
        sector_vol_budget=1.0,
        hard_asset_cap=1.0,
        sector_map={"A": "s1", "B": "s2"},
    )
    scale, diagnostics = controller.scale_for_aggregate(
        np.array([0.30, 0.20]), annual_covariance, names
    )
    assert scale == pytest.approx(0.5)
    assert diagnostics["max_asset_vol_proxy"] <= 0.03 + 1e-12


@pytest.mark.parametrize(
    "overrides",
    [
        {"hard_asset_cap": np.nan},
        {"gross_cap": -1.0},
        {"periods_per_year": 0.0},
        {"atr_window": 1},
    ],
)
def test_dynamic_risk_limits_reject_invalid_parameters(overrides):
    from optimization.risk_limits import VolatilityRiskCapController

    params = {"asset_vol_budget": 0.03, "sector_vol_budget": 0.06, **overrides}
    with pytest.raises(ValueError):
        VolatilityRiskCapController(**params)


def test_dynamic_risk_limits_reject_missing_weight_alignment():
    from optimization.risk_limits import VolatilityRiskCapController

    controller = VolatilityRiskCapController(
        asset_vol_budget=0.03, sector_vol_budget=0.06
    )
    covariance = pd.DataFrame(np.eye(2), index=["A", "B"], columns=["A", "B"])
    with pytest.raises(ValueError, match="missing or non-finite"):
        controller.apply(pd.Series({"A": 0.1}), covariance, ["A", "B"])


def test_barra_prefetch_reuses_slow_market_fields():
    from risk.barra_futures import BarraFuturesModel

    rng = np.random.default_rng(13)
    dates = pd.date_range("2023-01-02", periods=80, freq="B")
    assets = pd.Index([f"A{i}" for i in range(12)])
    returns = pd.DataFrame(
        rng.normal(scale=0.01, size=(80, 12)), index=dates, columns=assets
    )

    class CountingData:
        def __init__(self):
            self.get_calls = 0
            self.pair_calls = 0
            self.industry_calls = 0

        def get(self, field, requested_dates, universe):
            self.get_calls += 1
            return pd.DataFrame(100.0, index=requested_dates, columns=universe)

        def get_contract_pair(self, field, requested_dates, universe):
            self.pair_calls += 1
            near = pd.DataFrame(100.0, index=requested_dates, columns=universe)
            return {"near": near, "far": near * 1.01}

        def get_industry(self, requested_dates, universe):
            self.industry_calls += 1
            values = np.tile(
                np.where(np.arange(len(universe)) % 2, "s1", "s2"),
                (len(requested_dates), 1),
            )
            return pd.DataFrame(values, index=requested_dates, columns=universe)

    data = CountingData()
    model = BarraFuturesModel().prepare_data(data, dates, assets)
    model.estimate(data, {}, returns.iloc[:60])
    model.estimate(data, {}, returns.iloc[5:70])
    assert data.get_calls == 2
    assert data.pair_calls == 1
    assert data.industry_calls == 1


def test_barra_rejects_duplicate_return_dates():
    from risk.barra_futures import BarraFuturesModel

    dates = pd.DatetimeIndex(["2025-01-02", "2025-01-02"])
    returns = pd.DataFrame({"A": [0.01, -0.02]}, index=dates)
    with pytest.raises(ValueError, match="duplicate dates"):
        BarraFuturesModel()._clean_returns(returns)


def test_barra_rejects_missing_industry_classification():
    from risk.barra_futures import BarraFuturesModel

    dates = pd.date_range("2025-01-02", periods=2, freq="B")
    assets = pd.Index(["A", "B"])

    class MissingIndustry:
        @staticmethod
        def get_industry(requested_dates, universe):
            return pd.DataFrame(np.nan, index=requested_dates, columns=universe)

    with pytest.raises(RuntimeError, match="classification is missing"):
        BarraFuturesModel()._sector_exposures(MissingIndustry(), dates, assets)


def test_cost_configuration_charges_turnover_and_prorates_roll_allowance():
    from optimization.costs import SimpleAShareCost, SimpleFuturesCost

    model = SimpleFuturesCost(
        turnover_cost_rate=0.0002,
        annual_roll_cost=0.00105,
    )
    target = pd.Series({"A": 0.6, "B": -0.4})
    current = pd.Series({"A": 0.1, "B": -0.2})
    date = pd.Timestamp("2025-01-02")
    assert model.estimate_cost(target, current, date) == pytest.approx(0.7 * 0.0002)
    assert model.estimate_holding_cost(target, date) == pytest.approx(
        0.00105 / 252.0
    )

    with pytest.raises(TypeError):
        SimpleFuturesCost(commission_rate=0.0002)
    with pytest.raises(ValueError, match="finite"):
        model.estimate_cost(
            pd.Series({"A": np.nan}), pd.Series({"A": 0.0}), date
        )

    equity = SimpleAShareCost(
        commission=0.0003,
        stamp_duty_sell=0.001,
        slippage=0.0,
    )
    assert equity.estimate_cost(
        pd.Series({"A": 0.0}),
        pd.Series({"A": 0.0, "B": 0.5}),
        date,
    ) == pytest.approx(0.5 * (0.0003 + 0.001))


def test_default_config_registers_costs_and_dynamic_risk_limits():
    from core.config import load_config

    config = load_config("config/default.yaml")
    assert config.costs.turnover_cost_rate == pytest.approx(0.0002)
    assert config.costs.annual_fee == pytest.approx(0.0)
    assert config.costs.annual_roll_cost == pytest.approx(0.00105)
    assert config.costs.cost_stage == "post_screen_backtest"
    assert config.optimization.dynamic_risk_limits.enabled is True
    assert config.meta_optimizer.underlying_dynamic_risk_limits.enabled is True


def test_meta_combination_charges_management_fee_once_after_netting():
    from optimization.costs import SimpleFuturesCost
    from pipeline.runner import PipelineRunner

    dates = pd.date_range("2025-01-02", periods=2, freq="B")
    decision = dates[0] - pd.offsets.BDay(1)
    daily_fee = 0.00105 / 252.0
    sub_results = []
    for name, exposure in (("long", 1.0), ("short", -1.0)):
        result = SimpleNamespace(
            weights_history=pd.DataFrame({"A": [exposure]}, index=[decision]),
            research_ledger=SimpleNamespace(
                effective_weights=pd.DataFrame(
                    {"A": [exposure, exposure]}, index=dates
                ),
                daily=pd.DataFrame(
                    {
                        "trade_cost": [0.0, 0.0],
                        "holding_cost": [0.0, daily_fee],
                        "executed_traded_notional": [0.0, 0.0],
                    },
                    index=dates,
                ),
            ),
            costs=pd.Series([0.0, daily_fee], index=dates),
            nav=pd.Series(
                1.0, index=pd.DatetimeIndex([decision]).append(dates)
            ),
        )
        sub_results.append({"config": SimpleNamespace(name=name), "result": result})

    runner = PipelineRunner.__new__(PipelineRunner)
    runner.cost_model = SimpleFuturesCost(annual_fee=0.00105)
    runner.config = SimpleNamespace(optimization=SimpleNamespace(constraints=[]))
    meta_cfg = SimpleNamespace(
        underlying_constraints=[],
        enforce_underlying_constraints=True,
        min_weight=0.0,
        max_weight=1.0,
        underlying_dynamic_risk_limits=SimpleNamespace(enabled=False),
    )
    returns = pd.DataFrame(
        {"long": [0.0, -daily_fee], "short": [0.0, -daily_fee]}, index=dates
    )
    nav, _ = runner._combine_sleeve_path(
        returns, np.full((2, 2), 0.5), sub_results, meta_cfg, 1.0
    )
    assert runner._meta_holding_cost_history.sum() == pytest.approx(daily_fee)
    assert runner._meta_trade_cost_history.sum() == pytest.approx(0.0)
    assert nav.iloc[-1] == pytest.approx(1.0 - daily_fee)


def test_performance_metrics_record_zero_risk_free_rate():
    from backtest.metrics import compute_all_metrics, compute_sharpe
    from research.statistics import deflated_sharpe_ratio

    returns = pd.Series([0.0, 0.01, -0.004, 0.006, 0.002, -0.001])
    nav = (1.0 + returns).cumprod()
    metrics = compute_all_metrics(nav, returns=returns)
    expected = returns.mean() / returns.std() * np.sqrt(252.0)

    assert compute_sharpe(returns) == pytest.approx(expected)
    assert metrics["risk_free_rate"] == pytest.approx(0.0)
    assert metrics["sharpe"] == pytest.approx(expected)
    dsr = deflated_sharpe_ratio(returns, n_trials=3)
    assert dsr["risk_free_rate"] == pytest.approx(0.0)


def test_numpy_pearson_ic_matches_dataframe_reference():
    from testing.ic_test import _vectorized_pearson_ic

    rng = np.random.default_rng(22)
    dates = pd.date_range("2020-01-02", periods=120, freq="B")
    columns = pd.Index([f"A{i}" for i in range(18)])
    factor = pd.DataFrame(
        rng.normal(size=(120, 18)), index=dates, columns=columns
    )
    returns = pd.DataFrame(
        rng.normal(size=(120, 18)), index=dates, columns=columns
    )
    factor.mask(rng.random(factor.shape) < 0.12, inplace=True)
    returns.mask(rng.random(returns.shape) < 0.08, inplace=True)
    factor.iloc[5] = 1.0

    valid = factor.notna() & returns.notna()
    counts = valid.sum(axis=1)
    factor_filled = factor.where(valid, 0.0)
    returns_filled = returns.where(valid, 0.0)
    divisor = counts.where(counts > 0, 1).to_numpy()
    factor_centered = (
        factor_filled.sub((factor_filled * valid).sum(axis=1) / divisor, axis=0)
        * valid
    )
    returns_centered = (
        returns_filled.sub((returns_filled * valid).sum(axis=1) / divisor, axis=0)
        * valid
    )
    expected = (
        (factor_centered * returns_centered).sum(axis=1)
        / np.sqrt(
            (factor_centered**2).sum(axis=1)
            * (returns_centered**2).sum(axis=1)
        ).replace(0.0, np.nan)
    ).where(counts >= 10).dropna()

    actual, date_mask = _vectorized_pearson_ic(factor, returns, min_stocks=10)
    pd.testing.assert_series_equal(actual, expected, rtol=1e-13, atol=1e-15)
    pd.testing.assert_series_equal(date_mask, counts >= 10)


def test_ic_test_skips_unrequested_spearman_and_decay(monkeypatch):
    import testing.ic_test as ic_module

    dates = pd.date_range("2024-01-02", periods=30, freq="B")
    columns = pd.Index([f"A{i}" for i in range(12)])
    values = np.arange(360, dtype=float).reshape(30, 12)
    factor = pd.DataFrame(values, index=dates, columns=columns)
    returns = factor * 0.001

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Spearman should not run in Pearson-only mode")

    monkeypatch.setattr(ic_module, "_vectorized_spearman_ic", fail_if_called)
    result = ic_module.ICTest(
        methods=["pearson"], decay_periods=[], forward_period=5
    ).run(factor, returns)
    assert result.n_obs == len(dates)
    assert result.rank_ic_series.empty
    assert result.ic_decay == {}


def test_ic_test_reports_primary_spearman_positive_ratio():
    from testing.ic_test import ICTest

    dates = pd.date_range("2024-01-02", periods=12, freq="B")
    columns = pd.Index([f"A{i}" for i in range(12)])
    factor = pd.DataFrame(np.tile(np.arange(12, dtype=float), (12, 1)), index=dates, columns=columns)
    returns = factor.copy()

    result = ICTest(methods=["spearman"]).run(factor, returns)

    assert result.ic_mean == pytest.approx(1.0)
    assert result.to_dict()["ic_pos_ratio"] == pytest.approx(1.0)


def test_weighted_regression_requires_ex_ante_weights():
    from testing.regression import RegressionTest

    factor = pd.DataFrame([[1.0] * 10])
    returns = pd.DataFrame([[0.1] * 10])
    with pytest.raises(ValueError, match="sample_weights"):
        RegressionTest(weighted=True).run(factor, returns)


def test_data_manager_prefetch_reuses_positive_and_negative_fields():
    from data.manager import DataManager

    dates = pd.date_range("2024-01-02", periods=8, freq="B")
    universe = pd.Index(["A", "B"])

    class Source:
        def __init__(self):
            self.calls = []

        def fetch_price(self, tickers, start, end, fields):
            field = fields[0]
            self.calls.append(field)
            if field == "missing":
                return {field: pd.DataFrame()}
            return {
                field: pd.DataFrame(1.0, index=dates, columns=universe)
            }

    class Factor:
        def dependencies(self):
            return ["close", "missing"]

    source = Source()
    manager = DataManager(
        source=source,
        cache=None,
        config={"cache": {"enabled": False}},
    )
    manager.prefetch([Factor()], dates, universe)
    manager.get("close", dates, universe)
    manager.get("close", dates, universe)
    manager.get("missing", dates, universe)
    manager.get("missing", dates, universe)
    assert source.calls.count("close") == 1
    assert source.calls.count("missing") == 1


def test_data_manager_does_not_write_when_cache_is_disabled():
    from data.manager import DataManager

    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    universe = pd.Index(["A"])

    class Source:
        def fetch_price(self, tickers, start, end, fields):
            return {
                fields[0]: pd.DataFrame(1.0, index=dates, columns=universe)
            }

    class DisabledCache:
        def get(self, *args, **kwargs):
            raise AssertionError("disabled cache must not be read")

        def put(self, *args, **kwargs):
            raise AssertionError("disabled cache must not be written")

    manager = DataManager(
        source=Source(),
        cache=DisabledCache(),
        config={"cache": {"enabled": False}},
    )
    result = manager.get("close", dates, universe)
    assert result.shape == (3, 1)


def test_cache_covering_lookup_loads_only_selected_parquet(monkeypatch, tmp_path):
    from data.cache import Cache

    cache = Cache(tmp_path)
    full_dates = pd.date_range("2024-01-01", periods=30, freq="B")
    tickers = pd.Index(["A", "B", "C"])
    cache.put(
        "futures", "Source", "close", tickers, full_dates[0], full_dates[-1],
        pd.DataFrame(1.0, index=full_dates, columns=tickers),
    )
    wider_dates = pd.date_range("2023-12-01", periods=60, freq="B")
    cache.put(
        "futures", "Source", "close", tickers, wider_dates[0], wider_dates[-1],
        pd.DataFrame(2.0, index=wider_dates, columns=tickers),
    )

    original = pd.read_parquet
    calls = []

    def counted(path, *args, **kwargs):
        calls.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", counted)
    files_before = set(tmp_path.glob("*.parquet"))
    result = cache.get(
        "futures", "Source", "close", ["A", "B"],
        full_dates[5], full_dates[20],
    )
    assert len(calls) == 1
    assert result.shape == (16, 2)
    assert result.to_numpy().mean() == pytest.approx(1.0)
    assert set(tmp_path.glob("*.parquet")) == files_before
