from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import math
import time
import warnings
from typing import Callable, Sequence

import numpy as np

from factor_mining.features import FeatureSet
from factor_mining.operators import Expr, ExpressionEvaluator, _sanitize
from factor_mining.runtime.batch_fitness import (
    BatchFitnessAccumulator,
    OnlineFitnessAccumulator,
)
from factor_mining.runtime.static_context import StaticResearchContext
from factor_mining.validation import CandidateResult, ValidationConfig


STATEFUL_OPERATORS = frozenset({"ts_ema"})
BLOCK_NUMERICALLY_UNSTABLE_OPERATORS = frozenset({
    "cs_rank", "cs_zscore", "cs_demean",
})
LEGACY_BLOCK_OPERATORS = (
    STATEFUL_OPERATORS | BLOCK_NUMERICALLY_UNSTABLE_OPERATORS
)
V2_SAFE_OPERATORS = frozenset({
    "terminal", "constant",
    "add", "sub", "mul", "div", "avg", "max", "min",
    "neg", "abs", "signed_sqrt", "signed_log1p",
    "delay", "delta", "ts_mean", "ts_std", "ts_min", "ts_max",
    "ts_zscore",
})


def contains_stateful_operator(expression: Expr) -> bool:
    return (
        expression.op in STATEFUL_OPERATORS
        or any(contains_stateful_operator(argument) for argument in expression.args)
    )


def requires_legacy_executor(expression: Expr) -> bool:
    return (
        expression.op in LEGACY_BLOCK_OPERATORS
        or any(requires_legacy_executor(argument) for argument in expression.args)
    )


def v2_legacy_reason(expression: Expr) -> str | None:
    if expression.op not in V2_SAFE_OPERATORS:
        return f"unsupported_operator:{expression.op}"
    for argument in expression.args:
        reason = v2_legacy_reason(argument)
        if reason is not None:
            return reason
    return None


def _runtime_lookback(expression: Expr) -> int:
    """Lookback introduced by expression operators, excluding built terminals."""

    child = max(
        (_runtime_lookback(argument) for argument in expression.args),
        default=0,
    )
    if expression.op in {
        "delay", "delta", "ts_sum", "ts_mean", "ts_std", "ts_min",
        "ts_max", "ts_median", "ts_rank", "ts_zscore", "ts_skew",
        "ts_kurt", "ts_corr", "ts_cov", "decay_linear",
    }:
        return child + int(expression.window)
    return child


def _topological_nodes(expressions: Sequence[Expr]) -> tuple[Expr, ...]:
    visited: set[Expr] = set()
    ordered: list[Expr] = []

    def visit(node: Expr) -> None:
        if node in visited:
            return
        for argument in node.args:
            visit(argument)
        visited.add(node)
        ordered.append(node)

    for expression in expressions:
        visit(expression)
    return tuple(ordered)


def _tree_node_count(expression: Expr) -> int:
    return 1 + sum(_tree_node_count(argument) for argument in expression.args)


class PopulationExecutor:
    """Block-wise population tensor evaluator for opt-in GP acceleration."""

    def __init__(
        self,
        context: StaticResearchContext,
        validation_config: ValidationConfig,
        *,
        block_rows: int = 2500,
        use_dag: bool = True,
        use_fast_rolling: bool = False,
        evaluator_cache_mb: int = 128,
        legacy_score: Callable[[Expr], CandidateResult] | None = None,
        factor_chunk_size: int = 0,
        factor_workers: int = 1,
        online_fitness: bool = False,
    ):
        if block_rows < 1:
            raise ValueError("accelerator block_rows must be positive")
        if factor_chunk_size < 0:
            raise ValueError("accelerator factor_chunk_size cannot be negative")
        if factor_workers < 1:
            raise ValueError("accelerator factor_workers must be positive")
        self.context = context
        self.validation_config = validation_config
        self.block_rows = int(block_rows)
        self.use_dag = bool(use_dag)
        self.use_fast_rolling = bool(use_fast_rolling)
        self.evaluator_cache_bytes = int(evaluator_cache_mb) * 1024 * 1024
        self.legacy_score = legacy_score
        self.factor_chunk_size = int(factor_chunk_size)
        self.factor_workers = int(factor_workers)
        self.online_fitness = bool(online_fitness)
        self.stats: dict[str, object] = {}

    def run(self, expressions: Sequence[Expr]) -> list[CandidateResult]:
        started = time.perf_counter()
        expressions = tuple(expressions)
        if not expressions:
            return []
        reasons = [
            (
                v2_legacy_reason(expression)
                if self.factor_chunk_size else
                (
                    "legacy_block_operator"
                    if requires_legacy_executor(expression) else None
                )
            )
            for expression in expressions
        ]
        accelerated_indices = [
            index for index, reason in enumerate(reasons) if reason is None
        ]
        legacy_indices = [
            index for index, reason in enumerate(reasons) if reason is not None
        ]
        fallback_reasons = Counter(
            reason for reason in reasons if reason is not None
        )
        runtime_fallback_count = 0
        results: list[CandidateResult | None] = [None] * len(expressions)
        if accelerated_indices:
            accelerated = tuple(expressions[index] for index in accelerated_indices)
            try:
                batch_results = self._run_accelerated(accelerated)
            except Exception as exc:
                if self.legacy_score is None:
                    raise
                fallback_reasons[
                    f"runtime_error:{type(exc).__name__}"
                ] += len(accelerated)
                runtime_fallback_count += len(accelerated)
                warnings.warn(
                    "GP accelerator failed; falling back to legacy evaluator: "
                    f"{type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                batch_results = [
                    self.legacy_score(expression) for expression in accelerated
                ]
            for index, result in zip(accelerated_indices, batch_results):
                results[index] = result
        if legacy_indices:
            if self.legacy_score is None:
                raise ValueError(
                    "stateful expressions require a legacy_score fallback"
                )
            for index in legacy_indices:
                results[index] = self.legacy_score(expressions[index])
        if any(result is None for result in results):
            raise RuntimeError("population executor did not produce every result")
        self.stats.setdefault("expression_seconds", 0.0)
        self.stats.update({
            "evaluations": len(expressions),
            "accelerated_evaluations": len(accelerated_indices),
            "fallback_count": len(legacy_indices) + runtime_fallback_count,
            "fallback_reasons": dict(sorted(fallback_reasons.items())),
            "worker_count": (
                min(
                    self.factor_workers,
                    max(
                        1,
                        math.ceil(
                            len(accelerated_indices)
                            / max(1, self.factor_chunk_size)
                        ),
                    ),
                )
                if self.factor_chunk_size else 1
            ),
            "total_seconds": time.perf_counter() - started,
        })
        return [result for result in results if result is not None]

    def _run_accelerated(
        self, expressions: Sequence[Expr]
    ) -> list[CandidateResult]:
        accumulator_type = (
            OnlineFitnessAccumulator
            if self.online_fitness else BatchFitnessAccumulator
        )
        expression_chunks = (
            [
                tuple(expressions[start:start + self.factor_chunk_size])
                for start in range(0, len(expressions), self.factor_chunk_size)
            ]
            if self.factor_chunk_size else [tuple(expressions)]
        )
        accumulators = [
            accumulator_type(
                self.context,
                self.validation_config,
                [expression.complexity for expression in chunk],
            )
            for chunk in expression_chunks
        ]
        rows = self.context.features.shape[0]
        lookback = max(map(_runtime_lookback, expressions), default=0)
        expression_seconds = 0.0
        chunk_count = len(expression_chunks)
        workers = min(self.factor_workers, chunk_count)
        pool = (
            ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="factor-gp-chunk",
            )
            if self.factor_chunk_size and workers > 1 else None
        )
        try:
            for start in range(0, rows, self.block_rows):
                stop = min(rows, start + self.block_rows)
                if self.factor_chunk_size:
                    durations = []
                    for chunk_index, tensor, duration in (
                        self._evaluate_chunked_parts(
                            expression_chunks, start, stop, lookback, pool
                        )
                    ):
                        durations.append(duration)
                        accumulators[chunk_index].add_block(start, tensor)
                        del tensor
                    expression_seconds += max(durations, default=0.0)
                elif self.use_dag:
                    expression_started = time.perf_counter()
                    tensor = self._evaluate_dag_block(
                        expressions, start, stop, lookback
                    )
                    expression_seconds += (
                        time.perf_counter() - expression_started
                    )
                    accumulators[0].add_block(start, tensor)
                    del tensor
                else:
                    expression_started = time.perf_counter()
                    tensor = self._evaluate_individual_block(
                        expressions, start, stop, lookback
                    )
                    expression_seconds += (
                        time.perf_counter() - expression_started
                    )
                    accumulators[0].add_block(start, tensor)
                    del tensor
        finally:
            if pool is not None:
                pool.shutdown(wait=True)
        results = [
            result
            for accumulator in accumulators
            for result in accumulator.finalize()
        ]
        total_nodes = sum(map(_tree_node_count, expressions))
        if self.factor_chunk_size:
            unique_nodes = sum(
                len(_topological_nodes(
                    expressions[start:start + self.factor_chunk_size]
                ))
                for start in range(0, len(expressions), self.factor_chunk_size)
            )
        else:
            unique_nodes = len(_topological_nodes(expressions))
        timing_totals = {
            key: sum(
                accumulator.timings[key] for accumulator in accumulators
            )
            for key in accumulators[0].timings
        }
        self.stats = {
            "expression_seconds": expression_seconds,
            "chunk_count": chunk_count,
            "block_count": math.ceil(rows / self.block_rows),
            "worker_count": workers,
            "total_tree_nodes": total_nodes,
            "unique_dag_nodes": unique_nodes,
            "dag_reuse_rate": (
                1.0 - unique_nodes / total_nodes if total_nodes else 0.0
            ),
            **timing_totals,
        }
        return results

    def _evaluate_chunked_parts(
        self,
        expression_chunks: Sequence[Sequence[Expr]],
        start: int,
        stop: int,
        lookback: int,
        pool: ThreadPoolExecutor | None,
    ):
        expanded_start = max(0, start - lookback)
        features = self._block_features(expanded_start, stop)
        offset = start - expanded_start
        if pool is None:
            for chunk_index, chunk in enumerate(expression_chunks):
                value, duration = self._timed_feature_chunk(
                    chunk, features, offset
                )
                yield chunk_index, value, duration
            return
        futures = {
            pool.submit(
                self._timed_feature_chunk, chunk, features, offset
            ): chunk_index
            for chunk_index, chunk in enumerate(expression_chunks)
        }
        for future in as_completed(futures):
            value, duration = future.result()
            yield futures[future], value, duration

    def _evaluate_chunked_block(
        self,
        expressions: Sequence[Expr],
        start: int,
        stop: int,
        lookback: int,
        pool: ThreadPoolExecutor | None,
    ) -> np.ndarray:
        tensor = np.empty(
            (len(expressions), stop - start, self.context.features.shape[1]),
            dtype=np.float32,
        )
        chunks = [
            tuple(expressions[index:index + self.factor_chunk_size])
            for index in range(0, len(expressions), self.factor_chunk_size)
        ]
        for chunk_index, value, _ in self._evaluate_chunked_parts(
            chunks, start, stop, lookback, pool
        ):
            index = chunk_index * self.factor_chunk_size
            tensor[index:index + len(value)] = value
        return tensor

    def _timed_feature_chunk(
        self,
        expressions: Sequence[Expr],
        features: FeatureSet,
        offset: int,
    ) -> tuple[np.ndarray, float]:
        started = time.perf_counter()
        value = self._evaluate_feature_chunk(expressions, features, offset)
        return value, time.perf_counter() - started

    def _evaluate_feature_chunk(
        self,
        expressions: Sequence[Expr],
        features: FeatureSet,
        offset: int,
    ) -> np.ndarray:
        if self.use_dag:
            return self._evaluate_dag_features(expressions, features, offset)
        evaluator = self._evaluator(
            features, cache_bytes=self.evaluator_cache_bytes
        )
        return np.stack([
            evaluator.evaluate(expression, copy=False)[offset:]
            for expression in expressions
        ]).astype(np.float32, copy=False)

    def _block_features(self, start: int, stop: int) -> FeatureSet:
        features = self.context.features
        return FeatureSet(
            index=features.index[start:stop],
            symbols=features.symbols,
            values={
                name: value[start:stop]
                for name, value in features.values.items()
            },
            raw_dependencies=features.raw_dependencies,
            lookbacks=features.lookbacks,
            dtype=features.dtype,
        )

    def _evaluator(self, features: FeatureSet, *, cache_bytes: int):
        return ExpressionEvaluator(
            features,
            cache_max_bytes=cache_bytes,
            rolling_backend=("fast" if self.use_fast_rolling else "pandas"),
        )

    def _evaluate_individual_block(
        self,
        expressions: Sequence[Expr],
        start: int,
        stop: int,
        lookback: int,
    ) -> np.ndarray:
        expanded_start = max(0, start - lookback)
        features = self._block_features(expanded_start, stop)
        evaluator = self._evaluator(
            features, cache_bytes=self.evaluator_cache_bytes
        )
        offset = start - expanded_start
        return np.stack([
            evaluator.evaluate(expression, copy=False)[offset:]
            for expression in expressions
        ]).astype(np.float32, copy=False)

    def _evaluate_dag_block(
        self,
        expressions: Sequence[Expr],
        start: int,
        stop: int,
        lookback: int,
    ) -> np.ndarray:
        expanded_start = max(0, start - lookback)
        features = self._block_features(expanded_start, stop)
        return self._evaluate_dag_features(
            expressions, features, start - expanded_start
        )

    def _evaluate_dag_features(
        self,
        expressions: Sequence[Expr],
        features: FeatureSet,
        offset: int,
    ) -> np.ndarray:
        evaluator = self._evaluator(features, cache_bytes=0)
        cache: dict[Expr, np.ndarray] = {}
        for node in _topological_nodes(expressions):
            if node.op == "terminal":
                if node.name not in features.values:
                    raise KeyError(f"missing terminal feature: {node.name}")
                cache[node] = np.asarray(
                    features.values[node.name], dtype=np.float32
                )
                continue
            if node.op == "constant":
                value = np.full(
                    features.shape, node.value, dtype=np.float32
                )
            else:
                value = evaluator._apply(
                    node.op,
                    [cache[argument] for argument in node.args],
                    node.window,
                )
            value = _sanitize(value)
            value.setflags(write=False)
            cache[node] = value
        return np.stack([
            cache[expression][offset:] for expression in expressions
        ]).astype(np.float32, copy=False)
