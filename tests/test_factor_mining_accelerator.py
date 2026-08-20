from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from factor_mining.api import FeatureConfig, TargetSpec
from factor_mining.data import make_synthetic_panels
from factor_mining.features import FeatureEngine
from factor_mining.gp import GPConfig, GPSearch
from factor_mining.operators import Expr, ExpressionEvaluator
from factor_mining.runtime.batch_fitness import (
    batch_prepare_shifted_signal,
    batch_rank_rows,
)
from factor_mining.runtime.population_executor import PopulationExecutor
import factor_mining.runtime.rolling_backend as rolling_backend_module
from factor_mining.runtime.rolling_backend import rolling
from factor_mining.runtime.static_context import StaticResearchContext
from factor_mining.validation import (
    PreparedTarget,
    ValidationConfig,
    prepare_signal,
    shift_signal,
)


def _fixture(tmp_path, *, periods=220, symbols=8):
    panels = make_synthetic_panels(periods=periods, symbols=symbols, seed=71)
    feature_config = FeatureConfig(
        feature_horizons=(1, 2, 5, 10),
        lag_steps=(1, 2, 5),
        rolling_windows=(3, 5, 10, 15),
    )
    features = FeatureEngine(feature_config).build_all_terminals(panels)
    target = PreparedTarget.from_close(
        panels["close"],
        TargetSpec(name="forward_5p", horizon_bars=5, cost_bps=1.0),
    )
    volatility = features.values["realized_vol_15p"]
    labels = tuple("left" if index < symbols // 2 else "right"
                   for index in range(symbols))
    taxonomy = {"group_labels": list(labels)}
    context = StaticResearchContext.create(
        tmp_path / "snapshot",
        features=features,
        target=target,
        feature_config=feature_config,
        source_fingerprint="fixed-test-fingerprint",
        taxonomy=taxonomy,
        volatility=volatility,
        group_labels=labels,
        decision_lag_bars=1,
        block_rows=64,
    )
    return panels, feature_config, features, target, volatility, labels, context


def test_static_context_is_read_only_zero_copy_and_hash_bound(tmp_path):
    _, feature_config, features, target, _, labels, context = _fixture(tmp_path)

    assert isinstance(context.terminal_storage, np.memmap)
    assert context.terminal_storage.flags.writeable is False
    assert context.terminal_matrix.shape == (
        *features.shape, len(features.values)
    )
    close = context["close"]
    assert close.shape == features.shape
    assert np.shares_memory(close, context.terminal_storage)
    np.testing.assert_array_equal(close, features.values["close"])
    assert context.group_labels == labels
    assert context.coverage_denominator > 0

    with pytest.raises(ValueError, match="TargetSpec hash mismatch"):
        StaticResearchContext.load(
            context.cache_dir,
            expected_feature_config=feature_config,
            expected_target_spec=replace(target.spec, horizon_bars=6),
            expected_taxonomy={"group_labels": list(labels)},
            expected_source_fingerprint="fixed-test-fingerprint",
        )


@pytest.mark.parametrize("method", ["mean", "std", "min", "max"])
def test_optional_fast_rolling_matches_pandas_or_falls_back(method):
    rng = np.random.default_rng(73)
    value = rng.normal(size=(91, 9)).astype(np.float32)
    value[::7, 2] = np.nan
    expected = rolling(value, 15, method, backend="pandas")
    actual = rolling(value, 15, method, backend="fast")

    assert np.array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_allclose(
        actual, expected, rtol=1e-6, atol=1e-7, equal_nan=True
    )


def test_missing_bottleneck_uses_pandas_fallback(monkeypatch):
    rng = np.random.default_rng(74)
    value = rng.normal(size=(51, 7)).astype(np.float32)
    expected = rolling(value, 10, "mean", backend="pandas")
    monkeypatch.setattr(rolling_backend_module, "_bottleneck", None)
    rolling_backend_module._backend_is_compatible.cache_clear()

    actual = rolling(value, 10, "mean", backend="fast")

    np.testing.assert_array_equal(actual, expected)


def test_batch_prepare_and_rank_match_factor_loop(tmp_path):
    _, _, features, _, volatility, labels, context = _fixture(tmp_path)
    rng = np.random.default_rng(79)
    raw = rng.normal(size=(100, *features.shape)).astype(np.float32)
    raw[rng.random(raw.shape) < 0.08] = np.nan
    config = ValidationConfig(min_time_observations=20)
    expected = np.stack([
        prepare_signal(
            factor,
            config,
            volatility=volatility,
            group_labels=labels,
        )
        for factor in raw
    ])
    shifted = np.stack([
        shift_signal(factor, config.decision_lag_bars) for factor in raw
    ])
    actual = batch_prepare_shifted_signal(
        shifted,
        config,
        shifted_volatility=context.shifted_volatility,
        industry_group_indices=context.industry_group_indices,
    )

    assert np.array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_allclose(
        actual, expected, rtol=1e-6, atol=1e-7, equal_nan=True
    )
    legacy_rank = np.stack([
        pd.DataFrame(factor).rank(
            axis=1, method="average", pct=True
        ).to_numpy()
        for factor in expected
    ])
    accelerated_rank = batch_rank_rows(actual)
    np.testing.assert_array_equal(
        np.isnan(accelerated_rank), np.isnan(legacy_rank)
    )
    np.testing.assert_allclose(
        accelerated_rank, legacy_rank, rtol=0.0, atol=0.0, equal_nan=True
    )


def test_same_100_asts_preserve_values_ic_direction_and_order(tmp_path):
    (
        _,
        feature_config,
        features,
        target,
        _,
        labels,
        context,
    ) = _fixture(tmp_path, periods=260, symbols=10)
    validation = ValidationConfig(
        min_time_observations=20,
        time_segments=4,
        rebalance_every_bars=5,
        coverage_penalty=0.02,
        segment_floor_weight=0.1,
    )
    base_config = GPConfig(
        population_size=100,
        generations=1,
        elite_size=10,
        max_depth=4,
        max_complexity=14,
        windows=(2, 3, 5, 10),
        operators=(
            "add", "sub", "mul", "div", "min", "max", "abs", "neg",
            "signed_sqrt", "ts_mean", "ts_std", "ts_min", "ts_max",
            "ts_zscore",
        ),
        seed=83,
    )
    baseline = GPSearch(
        features,
        target,
        feature_config=feature_config,
        validation_config=validation,
        gp_config=base_config,
        group_labels=labels,
    )
    expressions = baseline._initial_population()
    assert len(expressions) == 100
    baseline_scores = baseline._score_population(expressions)

    dag_executor = PopulationExecutor(
        context,
        validation,
        block_rows=64,
        use_dag=True,
        legacy_score=baseline._score,
    )
    dag_scores = dag_executor.run(expressions)
    batch_executor = PopulationExecutor(
        context,
        validation,
        block_rows=64,
        use_dag=False,
        legacy_score=baseline._score,
    )
    batch_scores = batch_executor.run(expressions)
    chunk_scores = PopulationExecutor(
        context,
        validation,
        block_rows=64,
        use_dag=True,
        factor_chunk_size=50,
        factor_workers=2,
        legacy_score=baseline._score,
    ).run(expressions)
    online_scores = PopulationExecutor(
        context,
        validation,
        block_rows=64,
        use_dag=True,
        factor_chunk_size=50,
        factor_workers=2,
        online_fitness=True,
        legacy_score=baseline._score,
    ).run(expressions)

    legacy_evaluator = ExpressionEvaluator(features)
    raw_legacy = np.stack([
        legacy_evaluator.evaluate(expression) for expression in expressions
    ])
    raw_blocks = []
    lookback = max(
        expression.max_window * expression.depth for expression in expressions
    )
    for start in range(0, features.shape[0], 64):
        stop = min(features.shape[0], start + 64)
        raw_blocks.append(
            dag_executor._evaluate_dag_block(
                expressions, start, stop, lookback
            )
        )
    raw_dag = np.concatenate(raw_blocks, axis=1)
    chunk_executor = PopulationExecutor(
        context,
        validation,
        block_rows=64,
        use_dag=True,
        factor_chunk_size=50,
        factor_workers=2,
        legacy_score=baseline._score,
    )
    raw_chunk_blocks = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for start in range(0, features.shape[0], 64):
            stop = min(features.shape[0], start + 64)
            raw_chunk_blocks.append(
                chunk_executor._evaluate_chunked_block(
                    expressions, start, stop, lookback, pool
                )
            )
    raw_chunk = np.concatenate(raw_chunk_blocks, axis=1)

    assert np.array_equal(np.isnan(raw_dag), np.isnan(raw_legacy))
    assert np.array_equal(np.isnan(raw_chunk), np.isnan(raw_legacy))
    np.testing.assert_allclose(
        raw_dag, raw_legacy, rtol=1e-6, atol=1e-7, equal_nan=True
    )
    np.testing.assert_allclose(
        raw_chunk, raw_legacy, rtol=1e-6, atol=1e-7, equal_nan=True
    )
    candidate_ids = [f"gp_{expression.sha256[:16]}" for expression in expressions]
    assert len(set(candidate_ids)) == 100

    def ordered(scores):
        return [
            candidate_ids[index]
            for index in sorted(
                range(len(scores)),
                key=lambda index: scores[index].fitness,
                reverse=True,
            )
        ]

    # DAG/分块归约顺序不同会产生约1e-9的浮点噪声；方向与完整排序仍须完全一致。
    score_atol = 2e-9
    for accelerated in (
        batch_scores, dag_scores, chunk_scores, online_scores,
    ):
        assert [score.direction for score in accelerated] == [
            score.direction for score in baseline_scores
        ]
        for old, new in zip(baseline_scores, accelerated):
            if np.isnan(old.mean_ic):
                assert np.isnan(new.mean_ic)
            else:
                assert abs(old.mean_ic - new.mean_ic) < score_atol
            if np.isneginf(old.fitness):
                assert np.isneginf(new.fitness)
            else:
                assert abs(old.fitness - new.fitness) < score_atol
        assert ordered(accelerated) == ordered(baseline_scores)


def test_stateful_ema_expression_uses_legacy_fallback(tmp_path):
    _, _, features, target, _, labels, context = _fixture(tmp_path)
    expression = Expr.operation(
        "ts_ema", Expr.terminal("return_1p"), window=5
    )
    validation = ValidationConfig(min_time_observations=20)
    legacy = GPSearch(
        features,
        target,
        feature_config=FeatureConfig(
            feature_horizons=(1, 2, 5, 10),
            lag_steps=(1, 2, 5),
            rolling_windows=(3, 5, 10, 15),
        ),
        validation_config=validation,
        gp_config=GPConfig(
            population_size=4, generations=1, elite_size=1
        ),
        group_labels=labels,
    )
    calls = []

    def legacy_score(item):
        calls.append(item.sha256)
        return legacy._score(item)

    result = PopulationExecutor(
        context,
        validation,
        block_rows=32,
        use_dag=True,
        legacy_score=legacy_score,
    ).run([expression])

    assert calls == [expression.sha256]
    expected = legacy._score(expression)
    assert result[0].direction == expected.direction
    assert result[0].mean_ic == expected.mean_ic
    assert result[0].fitness == expected.fitness


@pytest.mark.parametrize("operator", ["ts_ema", "ts_corr", "ts_cov"])
def test_v2_operator_whitelist_routes_unsupported_to_legacy(
    tmp_path, operator
):
    _, _, features, target, _, labels, context = _fixture(tmp_path)
    left = Expr.terminal("return_1p")
    expression = (
        Expr.operation(operator, left, window=5)
        if operator == "ts_ema"
        else Expr.operation(
            operator, left, Expr.terminal("volume_z_10p"), window=5
        )
    )
    validation = ValidationConfig(min_time_observations=20)
    legacy = GPSearch(
        features,
        target,
        feature_config=FeatureConfig(
            feature_horizons=(1, 2, 5, 10),
            lag_steps=(1, 2, 5),
            rolling_windows=(3, 5, 10, 15),
        ),
        validation_config=validation,
        gp_config=GPConfig(
            population_size=4, generations=1, elite_size=1
        ),
        group_labels=labels,
    )
    calls = []

    def legacy_score(item):
        calls.append(item.sha256)
        return legacy._score(item)

    executor = PopulationExecutor(
        context,
        validation,
        block_rows=32,
        use_dag=True,
        factor_chunk_size=50,
        factor_workers=2,
        online_fitness=True,
        legacy_score=legacy_score,
    )
    result = executor.run([expression])

    assert calls == [expression.sha256]
    assert result[0] == legacy._score(expression)
    assert executor.stats["fallback_count"] == 1
    assert executor.stats["fallback_reasons"] == {
        f"unsupported_operator:{operator}": 1
    }


def test_full_gp_run_preserves_candidate_set_and_order(tmp_path):
    _, feature_config, features, target, _, labels, context = _fixture(tmp_path)
    validation = ValidationConfig(
        min_time_observations=20,
        rebalance_every_bars=5,
    )
    base = GPConfig(
        population_size=24,
        generations=2,
        elite_size=4,
        max_candidates=6,
        max_depth=4,
        max_complexity=12,
        windows=(2, 3, 5),
        operators=(
            "add", "sub", "mul", "div", "min", "max", "abs", "neg",
            "signed_sqrt", "ts_mean", "ts_min", "ts_max", "ts_zscore",
        ),
        seed=97,
    )
    baseline = GPSearch(
        features,
        target,
        feature_config=feature_config,
        validation_config=validation,
        gp_config=base,
        group_labels=labels,
        run_id="equivalence",
    ).run()
    accelerated = GPSearch(
        context.features,
        context.target,
        feature_config=feature_config,
        validation_config=validation,
        gp_config=replace(
            base,
            accelerator_mode="dag",
            accelerator_block_rows=64,
            use_fast_rolling=True,
        ),
        group_labels=labels,
        context=context,
        run_id="equivalence",
    ).run()
    accelerated_v2 = GPSearch(
        context.features,
        context.target,
        feature_config=feature_config,
        validation_config=validation,
        gp_config=replace(
            base,
            accelerator_mode="v2-lite",
            accelerator_block_rows=64,
            accelerator_chunk_size=12,
            n_jobs=2,
            use_fast_rolling=True,
        ),
        group_labels=labels,
        context=context,
        run_id="equivalence",
    ).run()

    for actual in (accelerated, accelerated_v2):
        assert [item.candidate_id for item in actual.candidates] == [
            item.candidate_id for item in baseline.candidates
        ]
        assert [
            item.metrics["search_fitness"] for item in actual.candidates
        ] == [
            item.metrics["search_fitness"] for item in baseline.candidates
        ]
