from __future__ import annotations

import warnings

import numpy as np

from factor_mining.api import FeatureConfig, TargetSpec
from factor_mining.data import make_synthetic_panels
from factor_mining.features import FeatureEngine
from factor_mining.gp import GPConfig, GPSearch, expression_lookback
from factor_mining.operators import Expr, ExpressionEvaluator
from factor_mining.validation import (
    PreparedTarget,
    ValidationConfig,
    _layer_diagnostics,
    _portfolio_diagnostics,
    mad_winsorize,
)


def _small_feature_set():
    panels = make_synthetic_panels(periods=160, symbols=8, seed=11)
    config = FeatureConfig(
        feature_horizons=(1, 2, 5, 10),
        lag_steps=(1, 2, 5),
        rolling_windows=(3, 5, 10, 15),
    )
    return panels, config, FeatureEngine(config).build(panels)


def test_expression_roundtrip_protected_division_and_lookback():
    _, _, features = _small_feature_set()
    expression = Expr.operation(
        "ts_mean",
        Expr.operation("div", Expr.terminal("return_1p"), Expr.constant(0.0)),
        window=5,
    )

    restored = Expr.from_dict(expression.to_dict())
    result = ExpressionEvaluator(features).evaluate(restored)

    assert restored.sha256 == expression.sha256
    assert expression_lookback(expression, features) == 6
    assert result.shape == features.shape
    assert not np.isinf(result).any()


def test_protected_division_preserves_missing_observations():
    _, _, features = _small_feature_set()
    expression = Expr.operation(
        "div", Expr.terminal("return_1p"), Expr.constant(0.0)
    )

    result = ExpressionEvaluator(features).evaluate(expression)

    assert np.isnan(result[0]).all()
    assert np.all(result[1:] == 0.0)


def test_gp_search_is_deterministic_and_emits_bridgeable_candidates():
    panels, feature_config, features = _small_feature_set()
    target_spec = TargetSpec(name="forward_5p", horizon_bars=5, cost_bps=1.0)
    target = PreparedTarget.from_close(panels["close"], target_spec)
    gp_config = GPConfig(
        population_size=24,
        generations=2,
        elite_size=4,
        max_candidates=4,
        max_depth=4,
        max_complexity=16,
        windows=(2, 3, 5),
        seed=23,
    )
    validation = ValidationConfig(
        min_time_observations=20, neutralize_volatility=True
    )

    first = GPSearch(
        features, target, feature_config=feature_config,
        validation_config=validation, gp_config=gp_config, run_id="test",
    ).run()
    second = GPSearch(
        features, target, feature_config=feature_config,
        validation_config=validation, gp_config=gp_config, run_id="test",
    ).run()

    assert first.candidates
    assert [item.candidate_id for item in first.candidates] == [
        item.candidate_id for item in second.candidates
    ]
    for candidate in first.candidates:
        assert candidate.kind == "symbolic"
        assert candidate.frequency == "1min"
        assert "close" in candidate.dependencies
        assert candidate.payload["decision_lag_bars"] == 1
        assert candidate.content_sha256 == ""
        candidate.validated()


def test_vectorized_candidate_diagnostics_match_row_reference():
    rng = np.random.default_rng(29)
    signal = rng.normal(size=(31, 11))
    target = rng.normal(size=(31, 11))
    signal[2, :8] = np.nan
    target[7, 3:9] = np.nan
    signal[9, 2:5] = 0.25

    expected_returns = np.full(len(signal), np.nan)
    expected_weights = np.zeros_like(signal, dtype=np.float32)
    layer_rows = [[] for _ in range(5)]
    for row in range(len(signal)):
        valid = np.isfinite(signal[row]) & np.isfinite(target[row])
        count = int(valid.sum())
        if count < 4:
            continue
        selected = np.flatnonzero(valid)
        order = selected[np.argsort(signal[row, selected], kind="stable")]
        side = max(1, int(np.floor(count * 0.2)))
        expected_weights[row, order[:side]] = -0.5 / side
        expected_weights[row, order[-side:]] = 0.5 / side
        expected_returns[row] = np.nansum(expected_weights[row] * target[row])
        if count >= 5:
            for layer, indices in enumerate(np.array_split(order, 5)):
                layer_rows[layer].append(float(np.mean(target[row, indices])))

    actual_returns, actual_turnover = _portfolio_diagnostics(
        signal, target, fraction=0.2, minimum=4
    )
    expected_turnover = np.full(len(signal), np.nan)
    expected_turnover[1:] = 0.5 * np.abs(np.diff(expected_weights, axis=0)).sum(
        axis=1
    )
    np.testing.assert_allclose(actual_returns, expected_returns, equal_nan=True)
    np.testing.assert_allclose(actual_turnover, expected_turnover, equal_nan=True)

    actual_layers, _ = _layer_diagnostics(signal, target, layers=5, minimum=4)
    expected_layers = tuple(float(np.mean(values)) for values in layer_rows)
    np.testing.assert_allclose(actual_layers, expected_layers)


def test_mad_winsorize_handles_all_missing_rows_without_warning():
    signal = np.array([[np.nan, np.nan], [1.0, 100.0]], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = mad_winsorize(signal, clip=5.0)
    assert np.isnan(result[0]).all()
    assert np.isfinite(result[1]).all()
