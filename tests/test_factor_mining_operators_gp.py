from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from factor_mining.api import FeatureConfig, TargetSpec
from factor_mining.data import make_synthetic_panels
from factor_mining.features import FeatureEngine
from factor_mining.gp import (
    GPConfig,
    GPSearch,
    expression_lookback,
    infer_economic_category,
)
from factor_mining.operators import Expr, ExpressionEvaluator
from factor_mining.validation import (
    PreparedTarget,
    ValidationConfig,
    evaluate_candidate,
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


def test_cross_section_operators_use_point_in_time_mask():
    _, _, features = _small_feature_set()
    mask = np.ones(features.shape, dtype=bool)
    mask[:, -2:] = False
    expression = Expr.operation(
        "square", Expr.operation("cs_zscore", Expr.terminal("return_1p"))
    )

    baseline = ExpressionEvaluator(
        features, cross_section_mask=mask
    ).evaluate(expression)
    values = dict(features.values)
    changed = values["return_1p"].copy()
    changed[:, -2:] = 1_000_000.0
    values["return_1p"] = changed
    perturbed_features = type(features)(
        index=features.index,
        symbols=features.symbols,
        values=values,
        raw_dependencies=features.raw_dependencies,
        lookbacks=features.lookbacks,
    )
    result = ExpressionEvaluator(
        perturbed_features, cross_section_mask=mask
    ).evaluate(expression)

    np.testing.assert_allclose(result[:, :-2], baseline[:, :-2], equal_nan=True)
    assert np.isnan(result[:, -2:]).all()


def test_protected_division_preserves_missing_observations():
    _, _, features = _small_feature_set()
    expression = Expr.operation(
        "div", Expr.terminal("return_1p"), Expr.constant(0.0)
    )

    result = ExpressionEvaluator(features).evaluate(expression)

    assert np.isnan(result[0]).all()
    assert np.all(result[1:] == 0.0)


def test_fast_rolling_rank_and_decay_match_reference_implementations():
    _, _, features = _small_feature_set()
    source = features.values["return_1p"]
    frame = pd.DataFrame(source)
    evaluator = ExpressionEvaluator(features)

    rank = evaluator.evaluate(
        Expr.operation("ts_rank", Expr.terminal("return_1p"), window=5)
    )
    expected_rank = frame.rolling(5, min_periods=2).apply(
        lambda values: pd.Series(values).rank(pct=True).iloc[-1], raw=False
    ).to_numpy()
    np.testing.assert_allclose(rank, expected_rank, equal_nan=True, rtol=1e-6)

    decay = evaluator.evaluate(
        Expr.operation("decay_linear", Expr.terminal("return_1p"), window=5)
    )
    weights = np.arange(1, 6, dtype=float)
    weights /= weights.sum()
    expected_decay = frame.rolling(5, min_periods=5).apply(
        lambda values: float(np.dot(values, weights)), raw=True
    ).to_numpy()
    np.testing.assert_allclose(decay, expected_decay, equal_nan=True, rtol=1e-6)


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


def test_gp_can_require_a_terminal_family():
    panels, feature_config, features = _small_feature_set()
    target = PreparedTarget.from_close(
        panels["close"], TargetSpec(name="forward_5p", horizon_bars=5)
    )
    outcome = GPSearch(
        features,
        target,
        feature_config=feature_config,
        validation_config=ValidationConfig(
            min_time_observations=20, neutralize_volatility=True
        ),
        gp_config=GPConfig(
            population_size=20,
            generations=1,
            elite_size=4,
            max_candidates=4,
            required_terminal_prefixes=("volume_",),
            seed=31,
        ),
    ).run()

    assert outcome.candidates
    for candidate in outcome.candidates:
        expression = Expr.from_dict(candidate.payload["expression"])
        assert any(name.startswith("volume_") for name in expression.terminals())


def test_gp_required_terminal_family_fails_when_unavailable():
    panels, feature_config, features = _small_feature_set()
    target = PreparedTarget.from_close(
        panels["close"], TargetSpec(name="forward_5p", horizon_bars=5)
    )

    with pytest.raises(ValueError, match="no usable terminals match"):
        GPSearch(
            features,
            target,
            feature_config=feature_config,
            gp_config=GPConfig(required_terminal_prefixes=("curve_",)),
        )


def test_economic_category_uses_actual_terminals():
    assert infer_economic_category(("curve_oi_hhi", "macd_diff_12_26_9")) == "term_structure"
    assert infer_economic_category(("macd_diff_12_26_9",)) == "momentum"
    assert infer_economic_category(("oi_change_15p",)) == "volume_oi"


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


def test_robust_fitness_rewards_coverage_and_worst_segment():
    panels = make_synthetic_panels(periods=120, symbols=8, seed=41)
    target = PreparedTarget.from_close(
        panels["close"], TargetSpec(name="forward_5p", horizon_bars=5)
    )
    signal = panels["close"].pct_change(fill_method=None).to_numpy()
    sparse = signal.copy()
    sparse[:, :4] = np.nan
    base = ValidationConfig(
        min_time_observations=20,
        neutralize_volatility=False,
        coverage_penalty=0.02,
        segment_floor_weight=0.5,
    )

    dense_result = evaluate_candidate(signal, target, base)
    sparse_result = evaluate_candidate(sparse, target, base)

    assert dense_result.coverage > sparse_result.coverage
    assert "oriented_segment_floor_ic" in dense_result.metrics
    assert dense_result.fitness > sparse_result.fitness


def test_coverage_uses_required_volatility_control_as_eligibility():
    panels = make_synthetic_panels(periods=120, symbols=8, seed=43)
    target = PreparedTarget.from_close(
        panels["close"], TargetSpec(name="forward_5p", horizon_bars=5)
    )
    signal = panels["close"].pct_change(fill_method=None).to_numpy()
    volatility = np.abs(signal)
    volatility[:, :4] = np.nan

    result = evaluate_candidate(
        signal,
        target,
        ValidationConfig(min_time_observations=20, neutralize_volatility=True),
        volatility=volatility,
    )

    assert result.coverage > 0.95
    assert result.metrics["coverage_denominator"] == "target_and_volatility_control"


def test_gp_window_vocabulary_is_positive_and_duplicate_free():
    from factor_mining.gp import GPConfig

    with pytest.raises(ValueError, match="positive"):
        GPConfig(windows=(0, 5))
    with pytest.raises(ValueError, match="duplicates"):
        GPConfig(windows=(5, 5))
