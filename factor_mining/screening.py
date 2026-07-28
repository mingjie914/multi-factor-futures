"""Deterministic same-sample pre-screening for mined symbolic candidates.

The pre-screen is a mining-stage quality-control pass.  It can reject only
mechanically invalid candidates; economic diagnostics and peer correlations
are annotations for the later framework study, not formal evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from factor_mining.api import CandidateSpec, TargetSpec, content_hash
from factor_mining.features import FeatureEngine
from factor_mining.gp import expression_lookback
from factor_mining.operators import Expr, ExpressionEvaluator
from factor_mining.validation import (
    PreparedTarget,
    ValidationConfig,
    evaluate_candidate,
    predictive_ic_decay,
    prepare_signal,
    signal_rank_persistence,
)


@dataclass(frozen=True)
class ScreeningConfig:
    """Conservative hard gates plus non-binding diagnostic thresholds."""

    min_coverage: float = 0.50
    min_cross_section: int = 4
    min_time_observations: int = 30
    min_variable_row_fraction: float = 0.05
    correlation_threshold: float = 0.85
    max_correlation_observations: int = 100_000
    evaluator_cache_mb: int = 256
    clipped_value: float = 0.99e8
    diagnostic_horizons: tuple[int, ...] = (1, 3, 5, 10, 20, 40)
    persistence_lags: tuple[int, ...] = (1, 3, 5, 10, 20, 40)

    def __post_init__(self) -> None:
        for name, value in (
            ("min_coverage", self.min_coverage),
            ("min_variable_row_fraction", self.min_variable_row_fraction),
            ("correlation_threshold", self.correlation_threshold),
        ):
            if not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.min_cross_section < 2 or self.min_time_observations < 2:
            raise ValueError("screening minimums must be at least two")
        if self.max_correlation_observations < 100:
            raise ValueError("max_correlation_observations must be at least 100")
        if self.evaluator_cache_mb < 0:
            raise ValueError("evaluator_cache_mb cannot be negative")
        if any(value < 1 for value in self.diagnostic_horizons):
            raise ValueError("diagnostic_horizons must be positive bar counts")
        if any(value < 1 for value in self.persistence_lags):
            raise ValueError("persistence_lags must be positive bar counts")


@dataclass(frozen=True)
class ScreeningOutcome:
    results: tuple[dict, ...]
    passed_candidate_ids: tuple[str, ...]
    correlation: pd.DataFrame
    config: ScreeningConfig
    feature_count: int
    shape: tuple[int, int]

    @property
    def input_count(self) -> int:
        return len(self.results)

    @property
    def rejected_count(self) -> int:
        return self.input_count - len(self.passed_candidate_ids)

    def summary(self) -> dict:
        reason_counts: dict[str, int] = {}
        flag_counts: dict[str, int] = {}
        for result in self.results:
            for reason in result["hard_reasons"]:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            for flag in result["soft_flags"]:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
        payload = {
            "stage": "mining_prescreen_not_formal_evidence",
            "input_count": self.input_count,
            "hard_pass_count": len(self.passed_candidate_ids),
            "hard_reject_count": self.rejected_count,
            "hard_reason_counts": dict(sorted(reason_counts.items())),
            "soft_flag_counts": dict(sorted(flag_counts.items())),
            "feature_count": self.feature_count,
            "matrix_shape": list(self.shape),
            "config": asdict(self.config),
            "passed_candidate_ids_sha256": content_hash(
                list(self.passed_candidate_ids)
            ),
        }
        return payload


def _common_feature_config(candidates: Sequence[CandidateSpec]):
    first = candidates[0].feature_config
    if any(candidate.feature_config != first for candidate in candidates[1:]):
        raise ValueError("screening batch must use one feature configuration")
    if any(candidate.frequency != first.decision_frequency for candidate in candidates):
        raise ValueError("candidate frequencies do not match the feature configuration")
    return first


def _group_labels(candidate: CandidateSpec, symbols: pd.Index) -> list[str] | None:
    mapping = candidate.payload.get("group_labels") or {}
    if not mapping:
        return None
    missing = [str(symbol) for symbol in symbols if str(symbol) not in mapping]
    if missing:
        raise ValueError(f"group labels are missing symbols: {missing}")
    return [str(mapping[str(symbol)]) for symbol in symbols]


def _variable_row_fraction(signal_rank: np.ndarray, minimum: int) -> float:
    finite = np.isfinite(signal_rank)
    usable = finite.sum(axis=1) >= minimum
    if not usable.any():
        return 0.0
    values = signal_rank[usable]
    row_min = np.nanmin(values, axis=1)
    row_max = np.nanmax(values, axis=1)
    return float(np.mean((row_max - row_min) > 1e-7))


def _sample_rank_signal(
    rank: np.ndarray, max_observations: int, minimum: int
) -> tuple[np.ndarray, float]:
    flat = rank.ravel()
    step = max(1, int(np.ceil(len(flat) / max_observations)))
    sample = np.asarray(flat[::step], dtype=np.float32)
    return sample, _variable_row_fraction(rank, minimum=minimum)


def _correlation_matrix(
    names: Sequence[str], samples: Sequence[np.ndarray | None]
) -> pd.DataFrame:
    if not names:
        return pd.DataFrame(dtype=float)
    width = max((len(sample) for sample in samples if sample is not None), default=0)
    values = np.full((width, len(names)), np.nan, dtype=np.float32)
    for column, sample in enumerate(samples):
        if sample is not None:
            values[: len(sample), column] = sample
    return pd.DataFrame(values, columns=names).corr(min_periods=20)


def _correlation_clusters(
    correlation: pd.DataFrame, threshold: float
) -> tuple[dict[str, str], dict[str, int], dict[str, tuple[str, float]]]:
    names = list(correlation.columns)
    parent = list(range(len(names)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    nearest: dict[str, tuple[str, float]] = {}
    values = correlation.to_numpy(dtype=float)
    for left in range(len(names)):
        best_name, best_value = "", float("nan")
        for right in range(len(names)):
            if left == right or not np.isfinite(values[left, right]):
                continue
            absolute = abs(float(values[left, right]))
            if not np.isfinite(best_value) or absolute > abs(best_value):
                best_name, best_value = names[right], float(values[left, right])
            if right > left and absolute >= threshold:
                union(left, right)
        nearest[names[left]] = (best_name, best_value)

    groups: dict[int, list[str]] = {}
    for index, name in enumerate(names):
        groups.setdefault(root(index), []).append(name)
    ordered = sorted(groups.values(), key=lambda members: min(names.index(x) for x in members))
    cluster_by_name: dict[str, str] = {}
    size_by_name: dict[str, int] = {}
    for number, members in enumerate(ordered, start=1):
        cluster = f"corr_{number:03d}"
        for name in members:
            cluster_by_name[name] = cluster
            size_by_name[name] = len(members)
    return cluster_by_name, size_by_name, nearest


def _dependency_family(candidate: CandidateSpec) -> str:
    dependencies = set(candidate.dependencies)
    if any(name.startswith("curve_") for name in dependencies):
        return "curve"
    if "amount" in dependencies:
        return "amount_liquidity"
    if "volume" in dependencies:
        return "volume_liquidity"
    if "oi" in dependencies or "oi_change" in dependencies:
        return "positioning"
    return "price"


def screen_candidates(
    candidates: Sequence[CandidateSpec],
    panels: Mapping[str, pd.DataFrame],
    *,
    config: ScreeningConfig | None = None,
    eligibility: pd.DataFrame | None = None,
    processing_eligibility: pd.DataFrame | None = None,
) -> ScreeningOutcome:
    """Evaluate all candidates without selecting on an arbitrary quota.

    ``eligibility`` selects evaluation observations. ``processing_eligibility``
    selects the point-in-time cross-section used while computing the signal.
    Keeping them separate retains warm-up history without admitting warm-up
    rows into screening statistics.
    """
    if not candidates:
        raise ValueError("screening candidates cannot be empty")
    screening = config or ScreeningConfig()
    feature_config = _common_feature_config(candidates)

    expressions: dict[str, Expr] = {}
    required_features: set[str] = set()
    for candidate in candidates:
        candidate.validated()
        expression = Expr.from_dict(candidate.payload["expression"])
        expressions[candidate.candidate_id] = expression
        required_features.update(expression.terminals())
        volatility_name = (candidate.payload.get("postprocess") or {}).get(
            "volatility_feature"
        )
        if volatility_name:
            required_features.add(str(volatility_name))

    features = FeatureEngine(feature_config).build(
        panels, required_features=required_features
    )
    eligibility_values = None
    if eligibility is not None:
        aligned_eligibility = eligibility.reindex(
            index=features.index,
            columns=features.symbols,
            fill_value=False,
        ).fillna(False)
        eligibility_values = aligned_eligibility.to_numpy(dtype=bool)
    processing_eligibility_values = eligibility_values
    if processing_eligibility is not None:
        aligned_processing_eligibility = processing_eligibility.reindex(
            index=features.index,
            columns=features.symbols,
            fill_value=False,
        ).fillna(False)
        processing_eligibility_values = (
            aligned_processing_eligibility.to_numpy(dtype=bool)
        )
    close = panels["close"].reindex(index=features.index, columns=features.symbols)
    targets = {
        target: PreparedTarget.from_close(close, target)
        for target in {candidate.target for candidate in candidates}
    }
    decay_targets = {}
    for candidate in candidates:
        key = (
            candidate.target.entry_delay_bars,
            candidate.target.entry_price,
            candidate.target.session_policy,
        )
        if key in decay_targets:
            continue
        decay_targets[key] = {
            int(horizon): PreparedTarget.from_close(
                close,
                TargetSpec(
                    name=f"diagnostic_forward_{int(horizon)}p",
                    decision_frequency=candidate.frequency,
                    horizon_bars=int(horizon),
                    entry_delay_bars=candidate.target.entry_delay_bars,
                    entry_price=candidate.target.entry_price,
                    session_policy=candidate.target.session_policy,
                    cost_bps=candidate.target.cost_bps,
                ),
            )
            for horizon in screening.diagnostic_horizons
        }
    evaluator = ExpressionEvaluator(
        features,
        cache_max_bytes=screening.evaluator_cache_mb * 1024 * 1024,
        cross_section_mask=processing_eligibility_values,
    )
    results: list[dict] = []
    samples: list[np.ndarray | None] = []
    seen_expression_horizon: set[tuple[str, int]] = set()

    for candidate in candidates:
        expression = expressions[candidate.candidate_id]
        hard_reasons: list[str] = []
        soft_flags: list[str] = []
        metrics: dict[str, object] = {}
        sample: np.ndarray | None = None
        key = (expression.sha256, candidate.target.horizon_bars)
        if expression.sha256 != candidate.payload.get("expression_sha256"):
            hard_reasons.append("expression_hash_mismatch")
        if key in seen_expression_horizon:
            hard_reasons.append("duplicate_expression_same_horizon")
        seen_expression_horizon.add(key)
        if not expression.terminals():
            hard_reasons.append("no_terminal_dependency")
        expected_lookback = expression_lookback(expression, features) + int(
            candidate.payload.get("decision_lag_bars", 1)
        )
        if candidate.lookback_bars < expected_lookback:
            hard_reasons.append("declared_lookback_too_short")
        missing_dependencies = sorted(
            dependency for dependency in candidate.dependencies
            if dependency not in panels
            or panels[dependency].reindex(
                index=features.index, columns=features.symbols
            ).isna().all().all()
        )
        if missing_dependencies:
            hard_reasons.append("missing_raw_dependency")
            metrics["missing_dependencies"] = missing_dependencies

        postprocess = candidate.payload.get("postprocess") or {}
        validation = ValidationConfig(
            decision_lag_bars=int(candidate.payload.get("decision_lag_bars", 1)),
            min_cross_section=screening.min_cross_section,
            min_time_observations=screening.min_time_observations,
            mad_clip=float(postprocess.get("mad_clip", 5.0)),
            neutralize_volatility=bool(
                postprocess.get("neutralize_volatility", False)
            ),
            rebalance_every_bars=int(
                candidate.payload.get(
                    "rebalance_every_bars", 1
                )
            ),
            economic_fitness_weight=float(
                candidate.payload.get("economic_fitness_weight", 0.0)
            ),
        )
        volatility_name = postprocess.get("volatility_feature")
        volatility = (
            features.values.get(str(volatility_name)) if volatility_name else None
        )
        try:
            raw = evaluator.evaluate(expression, copy=False)
            signal = candidate.expected_direction * raw
            signal_for_processing = signal
            volatility_for_processing = volatility
            if processing_eligibility_values is not None:
                signal_for_processing = np.where(
                    processing_eligibility_values, signal, np.nan
                )
                if volatility is not None:
                    volatility_for_processing = np.where(
                        processing_eligibility_values, volatility, np.nan
                    )
            prepared = prepare_signal(
                signal_for_processing,
                validation,
                volatility=volatility_for_processing,
                group_labels=_group_labels(candidate, features.symbols),
            )
            if eligibility_values is not None:
                prepared = np.where(eligibility_values, prepared, np.nan)
            prepared_rank = pd.DataFrame(prepared).rank(
                axis=1, method="average", pct=True
            ).to_numpy(dtype=np.float32)
            diagnostic = evaluate_candidate(
                signal,
                targets[candidate.target],
                validation,
                complexity=expression.complexity,
                volatility=volatility,
                group_labels=_group_labels(candidate, features.symbols),
                eligibility=eligibility_values,
                processing_eligibility=processing_eligibility_values,
                full_diagnostics=True,
            )
            sample, variable_fraction = _sample_rank_signal(
                prepared_rank,
                screening.max_correlation_observations,
                validation.min_cross_section,
            )
            metrics.update(diagnostic.to_metrics())
            decay_key = (
                candidate.target.entry_delay_bars,
                candidate.target.entry_price,
                candidate.target.session_policy,
            )
            metrics["predictive_ic_decay"] = predictive_ic_decay(
                prepared,
                decay_targets[decay_key],
                min_cross_section=validation.min_cross_section,
                signal_rank=prepared_rank,
            )
            metrics["signal_rank_persistence"] = signal_rank_persistence(
                prepared,
                lags=screening.persistence_lags,
                min_cross_section=validation.min_cross_section,
                signal_rank=prepared_rank,
            )
            metrics.update({
                "expected_direction": candidate.expected_direction,
                "diagnostic_universe_policy": (
                    "point_in_time_eligibility"
                    if eligibility_values is not None
                    else "static_declared_universe"
                ),
                "economic_search_contract_frozen": bool(
                    "rebalance_every_bars" in candidate.payload
                    and "economic_fitness_weight" in candidate.payload
                ),
                "direction_stable": bool(diagnostic.mean_ic >= 0),
                "variable_row_fraction": variable_fraction,
                "expression_complexity": expression.complexity,
                "expression_depth": expression.depth,
                "computed_lookback_bars": expected_lookback,
                "clipped_output_fraction": float(
                    np.mean(np.isfinite(raw) & (np.abs(raw) >= screening.clipped_value))
                ),
            })
            if diagnostic.observations < screening.min_time_observations:
                hard_reasons.append("insufficient_time_observations")
            if diagnostic.coverage < screening.min_coverage:
                hard_reasons.append("insufficient_coverage")
            if variable_fraction < screening.min_variable_row_fraction:
                hard_reasons.append("insufficient_cross_sectional_variation")
            if not bool(metrics["direction_stable"]):
                soft_flags.append("direction_unstable_in_prescreen")
            if diagnostic.net_long_short_mean <= 0:
                soft_flags.append("nonpositive_cost_adjusted_return")
            segments = np.asarray(diagnostic.segment_ic, dtype=float)
            if len(segments) and np.mean(segments > 0) < 0.75:
                soft_flags.append("weak_segment_sign_stability")
            if float(metrics["clipped_output_fraction"]) > 0.01:
                soft_flags.append("frequent_operator_clipping")
        except (FloatingPointError, KeyError, TypeError, ValueError) as exc:
            hard_reasons.append("evaluation_error")
            metrics["evaluation_error"] = f"{type(exc).__name__}: {exc}"

        if hard_reasons:
            sample = None
        samples.append(sample)
        results.append({
            "candidate_id": candidate.candidate_id,
            "framework_name": candidate.framework_name,
            "target_horizon_bars": candidate.target.horizon_bars,
            "dependency_family": _dependency_family(candidate),
            "dependencies": list(candidate.dependencies),
            "formula": expression.to_dict(),
            "expression_sha256": expression.sha256,
            "hard_pass": not hard_reasons,
            "hard_reasons": sorted(set(hard_reasons)),
            "soft_flags": sorted(set(soft_flags)),
            **metrics,
        })

    names = [result["candidate_id"] for result in results]
    correlation = _correlation_matrix(names, samples)
    clusters, sizes, nearest = _correlation_clusters(
        correlation, screening.correlation_threshold
    )
    for result in results:
        name = result["candidate_id"]
        peer, peer_correlation = nearest.get(name, ("", float("nan")))
        result.update({
            "correlation_cluster": clusters.get(name, ""),
            "correlation_cluster_size": sizes.get(name, 0),
            "nearest_peer": peer,
            "nearest_peer_correlation": peer_correlation,
        })
        if sizes.get(name, 0) > 1:
            result["soft_flags"] = sorted(
                set(result["soft_flags"]) | {"high_peer_correlation"}
            )

    passed = tuple(
        result["candidate_id"] for result in results if result["hard_pass"]
    )
    return ScreeningOutcome(
        results=tuple(results),
        passed_candidate_ids=passed,
        correlation=correlation,
        config=screening,
        feature_count=len(features.values),
        shape=features.shape,
    )
