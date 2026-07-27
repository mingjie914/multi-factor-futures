"""Deterministic, dependency-light genetic programming search."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import os
import threading
from typing import Sequence

import numpy as np

from factor_mining.api import CandidateSpec, FeatureConfig
from factor_mining.features import FeatureSet
from factor_mining.operators import (
    BOOLEAN,
    NUMERIC,
    OPERATOR_SPECS,
    Expr,
    ExpressionEvaluator,
)
from factor_mining.validation import (
    CandidateResult,
    PreparedTarget,
    ValidationConfig,
    evaluate_candidate,
    prepare_signal,
)


DEFAULT_OPERATORS = (
    "add", "sub", "mul", "div", "avg", "max", "min",
    "neg", "abs", "signed_sqrt", "signed_log1p",
    "delay", "delta", "ts_mean", "ts_ema", "ts_std", "ts_min", "ts_max",
    "ts_zscore", "ts_corr", "cs_rank", "cs_zscore", "cs_demean",
)

SLOW_OPTIONAL_OPERATORS = (
    "ts_median", "ts_rank", "ts_skew", "ts_kurt", "ts_cov", "decay_linear",
)


@dataclass(frozen=True)
class GPConfig:
    population_size: int = 160
    generations: int = 8
    tournament_size: int = 5
    elite_size: int = 12
    max_depth: int = 5
    max_complexity: int = 24
    initialization_depth: int = 3
    crossover_probability: float = 0.65
    mutation_probability: float = 0.25
    reproduction_probability: float = 0.10
    terminal_probability: float = 0.28
    constant_probability: float = 0.04
    windows: tuple[int, ...] = (2, 3, 5, 10, 15, 30, 60)
    constants: tuple[float, ...] = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
    operators: tuple[str, ...] = DEFAULT_OPERATORS
    allow_conditionals: bool = False
    n_jobs: int = 1
    evaluator_cache_mb: int = 128
    max_candidates: int = 30
    min_abs_ic: float = 0.0
    correlation_limit: float = 0.85
    seed: int = 17

    def __post_init__(self) -> None:
        if self.population_size < 4 or self.generations < 1:
            raise ValueError("GP population/generation settings are too small")
        if not 1 <= self.elite_size < self.population_size:
            raise ValueError("elite_size must be between 1 and population_size - 1")
        if self.max_depth < 2 or self.max_complexity < 3:
            raise ValueError("GP tree limits are too small")
        if self.n_jobs == 0 or self.n_jobs < -1:
            raise ValueError("n_jobs must be -1 or a positive integer")
        if self.evaluator_cache_mb < 0:
            raise ValueError("evaluator_cache_mb cannot be negative")
        probabilities = (
            self.crossover_probability
            + self.mutation_probability
            + self.reproduction_probability
        )
        if not math.isclose(probabilities, 1.0, abs_tol=1e-9):
            raise ValueError("crossover/mutation/reproduction probabilities must sum to one")
        unknown = sorted(set(self.operators) - set(OPERATOR_SPECS))
        if unknown:
            raise ValueError(f"unknown GP operators: {unknown}")


@dataclass(frozen=True)
class GenerationStats:
    generation: int
    unique_expressions: int
    finite_fitness: int
    best_fitness: float
    median_fitness: float


@dataclass(frozen=True)
class SearchOutcome:
    candidates: tuple[CandidateSpec, ...]
    generation_stats: tuple[GenerationStats, ...]
    evaluated_expressions: int


def expression_lookback(expression: Expr, features: FeatureSet) -> int:
    if expression.op == "terminal":
        return int(features.lookbacks.get(expression.name, 0))
    if expression.op == "constant":
        return 0
    child = max((expression_lookback(arg, features) for arg in expression.args), default=0)
    if OPERATOR_SPECS[expression.op].uses_window:
        return child + expression.window
    return child


class GPSearch:
    """Search symbolic expressions without adding a DEAP/gplearn dependency."""

    def __init__(
        self,
        features: FeatureSet,
        target: PreparedTarget,
        *,
        feature_config: FeatureConfig,
        validation_config: ValidationConfig | None = None,
        gp_config: GPConfig | None = None,
        run_id: str = "gp_run",
        group_labels: Sequence[str] | None = None,
    ):
        if features.shape != target.values.shape:
            raise ValueError("feature and target shapes differ")
        if feature_config.decision_frequency != target.spec.decision_frequency:
            raise ValueError("feature and target frequencies differ")
        self.features = features
        self.target = target
        self.feature_config = feature_config
        self.validation_config = validation_config or ValidationConfig()
        self.config = gp_config or GPConfig()
        self.run_id = str(run_id)
        self.group_labels = group_labels
        if group_labels is not None and len(group_labels) != len(features.symbols):
            raise ValueError("group_labels must contain one label per feature symbol")
        self.rng = np.random.default_rng(self.config.seed)
        self.terminals = tuple(
            name for name in features.feature_names
            if np.isfinite(features.values[name]).sum() >= self.validation_config.min_cross_section
        )
        if not self.terminals:
            raise ValueError("no usable terminal features")
        self._numeric_ops = tuple(
            op for op in self.config.operators
            if OPERATOR_SPECS[op].output_type == NUMERIC
            and (self.config.allow_conditionals or op != "if_else")
        )
        self._boolean_ops = tuple(
            op for op in self.config.operators
            if OPERATOR_SPECS[op].output_type == BOOLEAN
        )
        self._fitness_cache: dict[str, CandidateResult] = {}
        self._thread_local = threading.local()
        self._volatility_name, self._volatility = self._select_volatility_control()
        if self.validation_config.neutralize_volatility and self._volatility is None:
            raise ValueError(
                "volatility neutralization requires a 15/30/60-bar realized-vol feature"
            )

    def _select_volatility_control(self) -> tuple[str | None, np.ndarray | None]:
        if not self.validation_config.neutralize_volatility:
            return None, None
        for name in ("realized_vol_60p", "realized_vol_30p", "realized_vol_15p"):
            if name in self.features.values:
                return name, self.features.values[name]
        return None, None

    def _random_terminal(self) -> Expr:
        if self.rng.random() < self.config.constant_probability:
            return Expr.constant(float(self.rng.choice(self.config.constants)))
        return Expr.terminal(str(self.rng.choice(self.terminals)))

    def _random_expression(self, depth: int, output_type: str = NUMERIC) -> Expr:
        if depth <= 1 or (
            output_type == NUMERIC and self.rng.random() < self.config.terminal_probability
        ):
            if output_type == BOOLEAN:
                left = self._random_expression(1, NUMERIC)
                right = self._random_expression(1, NUMERIC)
                return Expr.operation(str(self.rng.choice(("gt", "lt"))), left, right)
            return self._random_terminal()

        choices = self._numeric_ops if output_type == NUMERIC else self._boolean_ops
        if not choices:
            return self._random_expression(1, output_type)
        op = str(self.rng.choice(choices))
        spec = OPERATOR_SPECS[op]
        expected = spec.input_types or (NUMERIC,) * spec.arity
        args = tuple(self._random_expression(depth - 1, kind) for kind in expected)
        window = int(self.rng.choice(self.config.windows)) if spec.uses_window else 0
        return Expr.operation(op, *args, window=window)

    def _valid(self, expression: Expr) -> bool:
        return (
            expression.output_type == NUMERIC
            and expression.depth <= self.config.max_depth
            and expression.complexity <= self.config.max_complexity
            and bool(expression.terminals())
        )

    def _evaluator(self) -> ExpressionEvaluator:
        evaluator = getattr(self._thread_local, "evaluator", None)
        if evaluator is None:
            evaluator = ExpressionEvaluator(
                self.features,
                cache_max_bytes=(
                    self.config.evaluator_cache_mb * 1024 * 1024
                    // self._worker_count()
                ),
            )
            self._thread_local.evaluator = evaluator
        return evaluator

    def _score(self, expression: Expr) -> CandidateResult:
        cached = self._fitness_cache.get(expression.sha256)
        if cached is not None:
            return cached
        try:
            signal = self._evaluator().evaluate(expression, copy=False)
            result = evaluate_candidate(
                signal,
                self.target,
                self.validation_config,
                complexity=expression.complexity,
                volatility=self._volatility,
                group_labels=self.group_labels,
                full_diagnostics=False,
            )
        except (FloatingPointError, KeyError, TypeError, ValueError):
            result = CandidateResult(
                fitness=float("-inf"), direction=1, mean_ic=float("nan"),
                ic_ir=float("nan"), ic_hit_rate=float("nan"), coverage=0.0,
                observations=0, gross_long_short_mean=float("nan"),
                net_long_short_mean=float("nan"), turnover_mean=float("nan"),
                monotonicity=float("nan"), layer_returns=(), segment_ic=(),
                metrics={"rejection_reason": "evaluation_error"},
            )
        self._fitness_cache[expression.sha256] = result
        return result

    def _score_population(self, population: Sequence[Expr]) -> list[CandidateResult]:
        workers = self._worker_count()
        if workers == 1:
            return [self._score(expression) for expression in population]
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="factor-gp") as pool:
            return list(pool.map(self._score, population))

    def _worker_count(self) -> int:
        if self.config.n_jobs == -1:
            return min(32, max(1, os.cpu_count() or 1))
        return int(self.config.n_jobs)

    def _tournament(self, population: Sequence[Expr], scores: Sequence[CandidateResult]) -> Expr:
        picks = self.rng.integers(0, len(population), size=self.config.tournament_size)
        winner = max(picks, key=lambda index: scores[int(index)].fitness)
        return population[int(winner)]

    def _crossover(self, left: Expr, right: Expr) -> Expr:
        left_paths = list(left.paths())
        self.rng.shuffle(left_paths)
        right_paths = list(right.paths())
        self.rng.shuffle(right_paths)
        for left_path in left_paths:
            left_node = left.subtree(left_path)
            compatible = [
                path for path in right_paths
                if right.subtree(path).output_type == left_node.output_type
            ]
            if compatible:
                child = left.replace(left_path, right.subtree(compatible[0]))
                if self._valid(child):
                    return child
        return left

    def _mutate(self, expression: Expr) -> Expr:
        paths = list(expression.paths())
        self.rng.shuffle(paths)
        for path in paths:
            node = expression.subtree(path)
            replacement = self._random_expression(
                max(1, min(3, self.config.max_depth - len(path))),
                node.output_type,
            )
            child = expression.replace(path, replacement)
            if self._valid(child):
                return child
        return expression

    def _initial_population(self) -> list[Expr]:
        population: dict[str, Expr] = {}
        attempts = 0
        limit = self.config.population_size * 30
        while len(population) < self.config.population_size and attempts < limit:
            expression = self._random_expression(self.config.initialization_depth)
            attempts += 1
            if self._valid(expression):
                population.setdefault(expression.sha256, expression)
        if len(population) < self.config.population_size:
            raise RuntimeError("could not create a sufficiently diverse GP population")
        return list(population.values())

    def _next_population(
        self, population: Sequence[Expr], scores: Sequence[CandidateResult]
    ) -> list[Expr]:
        ranking = sorted(
            range(len(population)), key=lambda index: scores[index].fitness, reverse=True
        )
        next_by_hash = {
            population[index].sha256: population[index]
            for index in ranking[: self.config.elite_size]
        }
        attempts = 0
        limit = self.config.population_size * 40
        while len(next_by_hash) < self.config.population_size and attempts < limit:
            attempts += 1
            draw = self.rng.random()
            parent = self._tournament(population, scores)
            if draw < self.config.crossover_probability:
                other = self._tournament(population, scores)
                child = self._crossover(parent, other)
            elif draw < self.config.crossover_probability + self.config.mutation_probability:
                child = self._mutate(parent)
            else:
                child = parent
            if self._valid(child):
                next_by_hash.setdefault(child.sha256, child)
        while len(next_by_hash) < self.config.population_size:
            child = self._random_expression(self.config.initialization_depth)
            if self._valid(child):
                next_by_hash.setdefault(child.sha256, child)
        return list(next_by_hash.values())

    def _deduplicate_by_signal(
        self, ranked: Sequence[tuple[Expr, CandidateResult]]
    ) -> list[tuple[Expr, CandidateResult]]:
        selected: list[tuple[Expr, CandidateResult]] = []
        signals: list[np.ndarray] = []
        for expression, score in ranked:
            raw = self._evaluator().evaluate(expression, copy=False)
            signal = prepare_signal(
                score.direction * raw,
                self.validation_config,
                volatility=self._volatility,
                group_labels=self.group_labels,
            )
            flat = signal.ravel()
            step = max(1, math.ceil(len(flat) / 100_000))
            flat = np.array(flat[::step], copy=True)
            redundant = False
            for previous in signals:
                valid = np.isfinite(flat) & np.isfinite(previous)
                if valid.sum() < 20:
                    continue
                correlation = np.corrcoef(flat[valid], previous[valid])[0, 1]
                if np.isfinite(correlation) and abs(correlation) >= self.config.correlation_limit:
                    redundant = True
                    break
            if not redundant:
                selected.append((expression, score))
                signals.append(flat)
            if len(selected) >= self.config.max_candidates:
                break
        return selected

    def run(self) -> SearchOutcome:
        population = self._initial_population()
        hall: dict[str, tuple[Expr, CandidateResult, int]] = {}
        stats: list[GenerationStats] = []
        for generation in range(self.config.generations):
            scores = self._score_population(population)
            finite = [score.fitness for score in scores if np.isfinite(score.fitness)]
            for expression, score in zip(population, scores):
                if np.isfinite(score.fitness):
                    prior = hall.get(expression.sha256)
                    if prior is None or score.fitness > prior[1].fitness:
                        hall[expression.sha256] = (expression, score, generation)
            stats.append(GenerationStats(
                generation=generation,
                unique_expressions=len(population),
                finite_fitness=len(finite),
                best_fitness=max(finite, default=float("-inf")),
                median_fitness=float(np.median(finite)) if finite else float("-inf"),
            ))
            if generation + 1 < self.config.generations:
                population = self._next_population(population, scores)
            if self.config.n_jobs == 1:
                self._evaluator().clear()

        ranked = sorted(
            ((expr, score) for expr, score, _ in hall.values()
             if abs(score.mean_ic) >= self.config.min_abs_ic),
            key=lambda item: item[1].fitness,
            reverse=True,
        )
        selected = self._deduplicate_by_signal(ranked)
        candidates: list[CandidateSpec] = []
        for expression, score in selected:
            generation = hall[expression.sha256][2]
            full_score = evaluate_candidate(
                self._evaluator().evaluate(expression, copy=False),
                self.target,
                self.validation_config,
                complexity=expression.complexity,
                volatility=self._volatility,
                group_labels=self.group_labels,
                full_diagnostics=True,
            )
            dependencies = tuple(sorted({
                "close", *self.features.dependencies_for(expression.terminals())
            }))
            candidate_id = f"gp_{expression.sha256[:16]}"
            payload = {
                "expression": expression.to_dict(),
                "expression_sha256": expression.sha256,
                "decision_lag_bars": self.validation_config.decision_lag_bars,
                "postprocess": {
                    "mad_clip": self.validation_config.mad_clip,
                    "cross_sectional_demean": True,
                    "neutralize_volatility": self.validation_config.neutralize_volatility,
                    "volatility_feature": self._volatility_name,
                },
                "group_labels": (
                    dict(zip(map(str, self.features.symbols), map(str, self.group_labels)))
                    if self.group_labels is not None else None
                ),
            }
            candidates.append(CandidateSpec(
                candidate_id=candidate_id,
                framework_name=f"mined_{candidate_id}",
                kind="symbolic",
                category="auto_mined",
                frequency=self.feature_config.decision_frequency,
                target=self.target.spec,
                dependencies=dependencies,
                lookback_bars=(
                    expression_lookback(expression, self.features)
                    + self.validation_config.decision_lag_bars
                ),
                payload=payload,
                feature_config=self.feature_config,
                metrics={
                    **full_score.to_metrics(),
                    "search_fitness": score.fitness,
                },
                lineage={"engine": "gp", "run_id": self.run_id, "generation": generation},
                status="mined_candidate",
                expected_direction=full_score.direction,
            ))
        return SearchOutcome(
            candidates=tuple(candidates),
            generation_stats=tuple(stats),
            evaluated_expressions=len(self._fitness_cache),
        )
