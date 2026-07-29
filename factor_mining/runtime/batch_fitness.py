from __future__ import annotations

from functools import lru_cache
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from factor_mining.runtime.static_context import StaticResearchContext
from factor_mining.validation import (
    CandidateResult,
    ValidationConfig,
    _fixed_annual_cost_per_target,
    _rank_ic_from_ranks,
)


_EPS = 1e-12


def _row_nanmean_tensor(value: np.ndarray) -> np.ndarray:
    finite = np.isfinite(value)
    total = np.where(finite, value, 0.0).sum(axis=-1, keepdims=True)
    count = finite.sum(axis=-1, keepdims=True)
    return np.divide(
        total,
        count,
        out=np.full_like(total, np.nan, dtype=float),
        where=count > 0,
    )


def _nanmedian_last(value: np.ndarray) -> np.ndarray:
    """NaN median on the last axis without NumPy masked-array machinery."""

    finite = np.isfinite(value)
    count = finite.sum(axis=-1)
    ordered = np.sort(np.where(finite, value, np.inf), axis=-1)
    lower_index = np.maximum(0, (count - 1) // 2)
    upper_index = np.maximum(0, count // 2)
    lower = np.take_along_axis(
        ordered, lower_index[..., None], axis=-1
    )[..., 0]
    upper = np.take_along_axis(
        ordered, upper_index[..., None], axis=-1
    )[..., 0]
    median = 0.5 * (lower + upper)
    median[count == 0] = np.nan
    return median[..., None]


def batch_mad_winsorize(signal: np.ndarray, clip: float) -> np.ndarray:
    """Apply the legacy cross-sectional MAD rule to an ``(M,B,N)`` tensor."""

    value = np.asarray(signal, dtype=float)
    median = _nanmedian_last(value)
    mad = _nanmedian_last(np.abs(value - median))
    scale = 1.4826 * mad
    lower = median - clip * scale
    upper = median + clip * scale
    scalable = np.isfinite(scale) & (scale > _EPS)
    clipped = np.where(
        scalable, np.minimum(np.maximum(value, lower), upper), value
    )
    return clipped.astype(np.float32)


def batch_neutralize_signal(
    signal: np.ndarray,
    *,
    shifted_volatility: np.ndarray | None,
    industry_group_indices: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Batch the existing group demean and one-control residualization."""

    result = np.asarray(signal, dtype=float).copy()
    if industry_group_indices:
        codes = np.empty(result.shape[2], dtype=np.intp)
        design = np.zeros(
            (result.shape[2], len(industry_group_indices)), dtype=float
        )
        for group, columns in enumerate(industry_group_indices.values()):
            codes[columns] = group
            design[columns, group] = 1.0
        finite = np.isfinite(result)
        group_sum = np.einsum(
            "mbn,ng->mbg",
            np.where(finite, result, 0.0),
            design,
            optimize=True,
        )
        group_count = np.einsum(
            "mbn,ng->mbg",
            finite,
            design,
            optimize=True,
        )
        group_mean = np.divide(
            group_sum,
            group_count,
            out=np.full_like(group_sum, np.nan, dtype=float),
            where=group_count > 0,
        )
        result -= group_mean[:, :, codes]
    result -= _row_nanmean_tensor(result)
    if shifted_volatility is not None:
        control = np.asarray(shifted_volatility, dtype=float)
        if control.shape != result.shape[1:]:
            raise ValueError("shifted volatility shape differs from signal block")
        control = control - _row_nanmean_tensor(control[None, :, :])[0]
        valid = np.isfinite(result) & np.isfinite(control[None, :, :])
        masked_result = np.where(valid, result, 0.0)
        masked_control = np.where(valid, control[None, :, :], 0.0)
        covariance = np.einsum(
            "mbn,mbn->mb",
            masked_result,
            masked_control,
            optimize=True,
        )[:, :, None]
        variance = np.einsum(
            "mbn,mbn->mb",
            masked_control,
            masked_control,
            optimize=True,
        )[:, :, None]
        beta = np.divide(
            covariance,
            variance,
            out=np.zeros_like(covariance),
            where=variance > _EPS,
        )
        result -= beta * control[None, :, :]
    return result.astype(np.float32)


def batch_prepare_shifted_signal(
    shifted_signal: np.ndarray,
    config: ValidationConfig,
    *,
    shifted_volatility: np.ndarray | None,
    industry_group_indices: Mapping[str, np.ndarray],
) -> np.ndarray:
    winsorized = batch_mad_winsorize(shifted_signal, config.mad_clip)
    return batch_neutralize_signal(
        winsorized,
        shifted_volatility=(
            shifted_volatility if config.neutralize_volatility else None
        ),
        industry_group_indices=industry_group_indices,
    )


@lru_cache(maxsize=1)
def _scipy_rank_matches_pandas() -> bool:
    probe = np.array([
        [np.nan, np.nan, np.nan, np.nan],
        [1.0, 1.0, 3.0, np.nan],
        [4.0, -2.0, 0.0, 9.0],
        [0.0, -0.0, 0.0, 1.0],
    ])
    expected = pd.DataFrame(probe).rank(
        axis=1, method="average", pct=True
    ).to_numpy()
    count = np.isfinite(probe).sum(axis=1, keepdims=True)
    actual = rankdata(
        probe, axis=-1, method="average", nan_policy="omit"
    )
    actual = np.divide(
        actual,
        count,
        out=np.full_like(actual, np.nan, dtype=float),
        where=count > 0,
    )
    return bool(
        np.array_equal(np.isnan(actual), np.isnan(expected))
        and np.allclose(actual, expected, rtol=0.0, atol=0.0, equal_nan=True)
    )


def batch_rank_rows(value: np.ndarray) -> np.ndarray:
    """Rank the symbol axis with Pandas-compatible average ties and NaNs."""

    source = np.asarray(value)
    if _scipy_rank_matches_pandas():
        count = np.isfinite(source).sum(axis=-1, keepdims=True)
        ranked = rankdata(
            source, axis=-1, method="average", nan_policy="omit"
        )
        return np.divide(
            ranked,
            count,
            out=np.full_like(ranked, np.nan, dtype=float),
            where=count > 0,
        )
    flat = source.reshape(-1, source.shape[-1])
    ranked = pd.DataFrame(flat).rank(
        axis=1, method="average", pct=True
    ).to_numpy()
    return ranked.reshape(source.shape)


def batch_rank_ic(
    signal_rank: np.ndarray,
    target_rank: np.ndarray,
    minimum: int,
) -> np.ndarray:
    factors, rows, symbols = signal_rank.shape
    flat_signal = np.asarray(signal_rank, dtype=float).reshape(
        factors * rows, symbols
    )
    flat_target = np.broadcast_to(
        np.asarray(target_rank)[None, :, :],
        (factors, rows, symbols),
    ).reshape(factors * rows, symbols)
    return _rank_ic_from_ranks(
        flat_signal, flat_target, minimum
    ).reshape(factors, rows)


class BatchFitnessAccumulator:
    """Block-wise batch implementation of the mining fitness pipeline."""

    def __init__(
        self,
        context: StaticResearchContext,
        config: ValidationConfig,
        complexities: Sequence[int],
        *,
        _store_ic: bool = True,
    ):
        self.context = context
        self.config = config
        self.complexities = np.asarray(complexities, dtype=int)
        self.factor_count = len(self.complexities)
        rows, columns = context.features.shape
        self.rows = rows
        self.columns = columns
        self.ic = (
            np.full((self.factor_count, rows), np.nan, dtype=float)
            if _store_ic else None
        )
        self.coverage_numerator = np.zeros(self.factor_count, dtype=np.int64)
        self._gross_chunks: list[np.ndarray] = []
        self._turnover_sum = np.zeros(self.factor_count, dtype=float)
        self._turnover_count = np.zeros(self.factor_count, dtype=np.int64)
        self._previous_decision_weights: np.ndarray | None = None
        self.timings = {
            "mad_seconds": 0.0,
            "neutralization_seconds": 0.0,
            "rank_ic_seconds": 0.0,
            "portfolio_seconds": 0.0,
        }
        lag = int(config.decision_lag_bars)
        self._raw_tail = np.full(
            (self.factor_count, lag, columns), np.nan, dtype=np.float32
        )

    def _shift_block(self, raw_signal: np.ndarray) -> np.ndarray:
        value = np.asarray(raw_signal, dtype=np.float32)
        if value.shape[0] != self.factor_count or value.shape[2] != self.columns:
            raise ValueError("population tensor shape differs from context")
        lag = int(self.config.decision_lag_bars)
        combined = np.concatenate((self._raw_tail, value), axis=1)
        shifted = combined[:, :value.shape[1], :]
        self._raw_tail = combined[:, -lag:, :].copy()
        return shifted

    def _process_block(
        self, start: int, raw_signal: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        block_rows = raw_signal.shape[1]
        stop = int(start) + int(block_rows)
        if start < 0 or stop > self.rows:
            raise ValueError("population block is outside context range")
        shifted = self._shift_block(raw_signal)
        shifted_volatility = (
            None if self.context.shifted_volatility is None
            else self.context.shifted_volatility[start:stop]
        )
        started = time.perf_counter()
        winsorized = batch_mad_winsorize(shifted, self.config.mad_clip)
        self.timings["mad_seconds"] += time.perf_counter() - started
        started = time.perf_counter()
        prepared = batch_neutralize_signal(
            winsorized,
            shifted_volatility=(
                shifted_volatility
                if self.config.neutralize_volatility else None
            ),
            industry_group_indices=self.context.industry_group_indices,
        )
        self.timings["neutralization_seconds"] += (
            time.perf_counter() - started
        )
        coverage_mask = np.asarray(
            self.context.coverage_mask[start:stop], dtype=bool
        )
        self.coverage_numerator += (
            np.isfinite(prepared) & coverage_mask[None, :, :]
        ).sum(axis=(1, 2))

        started = time.perf_counter()
        signal_rank = batch_rank_rows(prepared)
        ic = batch_rank_ic(
            signal_rank,
            self.context.target.rank_values[start:stop],
            self.config.min_cross_section,
        )
        self.timings["rank_ic_seconds"] += time.perf_counter() - started
        return signal_rank, ic

    def add_block(self, start: int, raw_signal: np.ndarray) -> None:
        signal_rank, ic = self._process_block(start, raw_signal)
        self.ic[:, start:start + raw_signal.shape[1]] = ic
        started = time.perf_counter()
        self._add_portfolio_block(start, signal_rank)
        self.timings["portfolio_seconds"] += time.perf_counter() - started

    def _add_portfolio_block(
        self, start: int, signal_rank: np.ndarray
    ) -> None:
        block_rows = signal_rank.shape[1]
        global_rows = np.arange(start, start + block_rows)
        selected = np.flatnonzero(
            global_rows % int(self.config.rebalance_every_bars) == 0
        )
        if not len(selected):
            return
        ranks = np.asarray(signal_rank[:, selected, :], dtype=float)
        valid_signal = np.isfinite(ranks)
        counts = valid_signal.sum(axis=2)
        centered = ranks - _row_nanmean_tensor(ranks)
        gross_exposure = np.nansum(
            np.abs(centered), axis=2, keepdims=True
        )
        weights = np.divide(
            centered,
            gross_exposure,
            out=np.zeros_like(centered),
            where=(gross_exposure > _EPS) & valid_signal,
        )
        target = np.asarray(
            self.context.target.values[start:start + block_rows][selected],
            dtype=float,
        )
        joint_counts = (
            np.isfinite(target)[None, :, :] & valid_signal
        ).sum(axis=2)
        usable = (
            (counts >= self.config.min_cross_section)
            & (joint_counts >= self.config.min_cross_section)
        )
        gross = np.full(usable.shape, np.nan, dtype=float)
        gross[usable] = np.nansum(
            weights[usable] * np.broadcast_to(target, weights.shape)[usable],
            axis=1,
        )
        self._gross_chunks.append(gross)

        if self._previous_decision_weights is None:
            differences = np.diff(weights, axis=1)
        else:
            differences = np.diff(
                np.concatenate(
                    (self._previous_decision_weights[:, None, :], weights),
                    axis=1,
                ),
                axis=1,
            )
        if differences.shape[1]:
            turnover = 0.5 * np.abs(differences).sum(axis=2)
            self._turnover_sum += turnover.sum(axis=1)
            self._turnover_count += turnover.shape[1]
        self._previous_decision_weights = weights[:, -1, :].copy()

    def finalize(self) -> list[CandidateResult]:
        gross = (
            np.concatenate(self._gross_chunks, axis=1)
            if self._gross_chunks
            else np.empty((self.factor_count, 0), dtype=float)
        )
        results = []
        fixed_cost = _fixed_annual_cost_per_target(self.context.target.spec)
        for factor in range(self.factor_count):
            ic = self.ic[factor]
            valid_ic = ic[np.isfinite(ic)]
            observations = int(len(valid_ic))
            if observations < self.config.min_time_observations:
                results.append(CandidateResult(
                    fitness=float("-inf"), direction=1, mean_ic=float("nan"),
                    ic_ir=float("nan"), ic_hit_rate=float("nan"), coverage=0.0,
                    observations=observations,
                    gross_long_short_mean=float("nan"),
                    net_long_short_mean=float("nan"), turnover_mean=float("nan"),
                    monotonicity=float("nan"), layer_returns=(), segment_ic=(),
                    metrics={"rejection_reason": "insufficient_time_observations"},
                ))
                continue
            mean_ic = float(np.mean(valid_ic))
            direction = 1 if mean_ic >= 0 else -1
            ic_std = (
                float(np.std(valid_ic, ddof=1))
                if observations > 1 else np.nan
            )
            ic_ir = (
                mean_ic / ic_std
                if np.isfinite(ic_std) and ic_std > _EPS else 0.0
            )
            hit_rate = float(np.mean(np.sign(valid_ic) == direction))
            coverage = float(
                self.coverage_numerator[factor]
                / max(1, self.context.coverage_denominator)
            )
            base_gross = gross[factor]
            oriented_gross = direction * base_gross
            rank_gross_mean = (
                float(np.nanmean(oriented_gross))
                if np.isfinite(oriented_gross).any() else 0.0
            )
            rank_net = np.where(
                np.isfinite(oriented_gross),
                oriented_gross - fixed_cost,
                np.nan,
            )
            rank_net_mean = (
                float(np.nanmean(rank_net))
                if np.isfinite(rank_net).any() else 0.0
            )
            target_dispersion = float(self.context.target_dispersion)
            cost_adjusted_return_score = (
                float(np.clip(
                    rank_net_mean / target_dispersion, -1.0, 1.0
                ))
                if target_dispersion > _EPS else 0.0
            )
            turnover_mean = (
                float(
                    self._turnover_sum[factor]
                    / self._turnover_count[factor]
                )
                if self._turnover_count[factor] else 0.0
            )
            segments = tuple(
                float(np.nanmean(part))
                for part in np.array_split(ic, self.config.time_segments)
                if np.isfinite(part).any()
            )
            stable_fraction = (
                float(np.mean(np.sign(segments) == direction))
                if segments else 0.0
            )
            oriented_segment_floor = (
                float(np.min(direction * np.asarray(segments, dtype=float)))
                if segments else 0.0
            )
            bounded_ir = min(abs(ic_ir), 3.0)
            economic_weight = float(self.config.economic_fitness_weight)
            complexity = int(self.complexities[factor])
            fitness = (
                (1.0 - economic_weight) * abs(mean_ic)
                + economic_weight * cost_adjusted_return_score
                + 0.02 * bounded_ir
                + 0.01 * hit_rate
                + 0.01 * stable_fraction
                + self.config.segment_floor_weight * oriented_segment_floor
                - self.config.coverage_penalty * (1.0 - coverage)
                - self.config.complexity_penalty * max(0, complexity - 1)
            )
            results.append(CandidateResult(
                fitness=float(fitness),
                direction=direction,
                mean_ic=mean_ic,
                ic_ir=float(ic_ir),
                ic_hit_rate=hit_rate,
                coverage=coverage,
                observations=observations,
                gross_long_short_mean=float("nan"),
                net_long_short_mean=float("nan"),
                turnover_mean=turnover_mean,
                monotonicity=float("nan"),
                layer_returns=(),
                segment_ic=segments,
                metrics={
                    "complexity": complexity,
                    "stable_segment_fraction": stable_fraction,
                    "oriented_segment_floor_ic": oriented_segment_floor,
                    "coverage_denominator": (
                        "target_and_volatility_control"
                        if self.config.neutralize_volatility
                        and self.context.volatility is not None
                        else "target"
                    ),
                    "rebalance_every_bars": int(
                        self.config.rebalance_every_bars
                    ),
                    "rank_weight_gross_mean": rank_gross_mean,
                    "rank_weight_net_mean": rank_net_mean,
                    "rank_weight_turnover_mean": turnover_mean,
                    "target_cross_sectional_dispersion": target_dispersion,
                    "cost_adjusted_return_score": cost_adjusted_return_score,
                    "economic_fitness_weight": economic_weight,
                    "mining_cost_definition": (
                        "fixed_annual_cost_prorated_by_target_holding_bars"
                    ),
                    "annual_transaction_cost_bps": float(
                        self.context.target.spec.cost_bps
                    ),
                    "cost_per_target_observation": float(fixed_cost),
                    "cost_uses_turnover": False,
                },
            ))
        return results


class OnlineFitnessAccumulator(BatchFitnessAccumulator):
    """Memory-bounded block statistics with the same mining fitness formula."""

    def __init__(
        self,
        context: StaticResearchContext,
        config: ValidationConfig,
        complexities: Sequence[int],
    ):
        super().__init__(
            context,
            config,
            complexities,
            _store_ic=False,
        )
        self._gross_chunks = []
        self._ic_sum = np.zeros(self.factor_count, dtype=float)
        self._ic_squared_sum = np.zeros(self.factor_count, dtype=float)
        self._ic_count = np.zeros(self.factor_count, dtype=np.int64)
        self._positive_ic = np.zeros(self.factor_count, dtype=np.int64)
        self._negative_ic = np.zeros(self.factor_count, dtype=np.int64)
        self._gross_sum = np.zeros(self.factor_count, dtype=float)
        self._gross_count = np.zeros(self.factor_count, dtype=np.int64)
        parts = np.array_split(
            np.arange(self.rows, dtype=np.int64), self.config.time_segments
        )
        self._segment_bounds = tuple(
            (int(part[0]), int(part[-1]) + 1)
            for part in parts if len(part)
        )
        self._segment_sum = np.zeros(
            (self.factor_count, len(self._segment_bounds)), dtype=float
        )
        self._segment_count = np.zeros(
            (self.factor_count, len(self._segment_bounds)), dtype=np.int64
        )

    def add_block(self, start: int, raw_signal: np.ndarray) -> None:
        signal_rank, ic = self._process_block(start, raw_signal)
        valid = np.isfinite(ic)
        safe = np.where(valid, ic, 0.0)
        self._ic_sum += safe.sum(axis=1)
        self._ic_squared_sum += np.square(safe).sum(axis=1)
        self._ic_count += valid.sum(axis=1)
        self._positive_ic += (ic > 0.0).sum(axis=1)
        self._negative_ic += (ic < 0.0).sum(axis=1)
        stop = start + ic.shape[1]
        for segment, (segment_start, segment_stop) in enumerate(
            self._segment_bounds
        ):
            overlap_start = max(start, segment_start)
            overlap_stop = min(stop, segment_stop)
            if overlap_start >= overlap_stop:
                continue
            local = ic[:, overlap_start - start:overlap_stop - start]
            local_valid = np.isfinite(local)
            self._segment_sum[:, segment] += np.where(
                local_valid, local, 0.0
            ).sum(axis=1)
            self._segment_count[:, segment] += local_valid.sum(axis=1)
        started = time.perf_counter()
        self._add_online_portfolio_block(start, signal_rank)
        self.timings["portfolio_seconds"] += time.perf_counter() - started

    def _add_online_portfolio_block(
        self, start: int, signal_rank: np.ndarray
    ) -> None:
        block_rows = signal_rank.shape[1]
        global_rows = np.arange(start, start + block_rows)
        selected = np.flatnonzero(
            global_rows % int(self.config.rebalance_every_bars) == 0
        )
        if not len(selected):
            return
        ranks = np.asarray(signal_rank[:, selected, :], dtype=float)
        valid_signal = np.isfinite(ranks)
        counts = valid_signal.sum(axis=2)
        centered = ranks - _row_nanmean_tensor(ranks)
        gross_exposure = np.nansum(
            np.abs(centered), axis=2, keepdims=True
        )
        weights = np.divide(
            centered,
            gross_exposure,
            out=np.zeros_like(centered),
            where=(gross_exposure > _EPS) & valid_signal,
        )
        target = np.asarray(
            self.context.target.values[start:start + block_rows][selected],
            dtype=float,
        )
        joint_counts = (
            np.isfinite(target)[None, :, :] & valid_signal
        ).sum(axis=2)
        usable = (
            (counts >= self.config.min_cross_section)
            & (joint_counts >= self.config.min_cross_section)
        )
        gross = np.full(usable.shape, np.nan, dtype=float)
        gross[usable] = np.nansum(
            weights[usable] * np.broadcast_to(target, weights.shape)[usable],
            axis=1,
        )
        valid_gross = np.isfinite(gross)
        self._gross_sum += np.where(valid_gross, gross, 0.0).sum(axis=1)
        self._gross_count += valid_gross.sum(axis=1)

        if self._previous_decision_weights is None:
            differences = np.diff(weights, axis=1)
        else:
            differences = np.diff(
                np.concatenate(
                    (self._previous_decision_weights[:, None, :], weights),
                    axis=1,
                ),
                axis=1,
            )
        if differences.shape[1]:
            turnover = 0.5 * np.abs(differences).sum(axis=2)
            self._turnover_sum += turnover.sum(axis=1)
            self._turnover_count += turnover.shape[1]
        self._previous_decision_weights = weights[:, -1, :].copy()

    def finalize(self) -> list[CandidateResult]:
        results = []
        fixed_cost = _fixed_annual_cost_per_target(self.context.target.spec)
        for factor in range(self.factor_count):
            observations = int(self._ic_count[factor])
            if observations < self.config.min_time_observations:
                results.append(CandidateResult(
                    fitness=float("-inf"), direction=1, mean_ic=float("nan"),
                    ic_ir=float("nan"), ic_hit_rate=float("nan"), coverage=0.0,
                    observations=observations,
                    gross_long_short_mean=float("nan"),
                    net_long_short_mean=float("nan"), turnover_mean=float("nan"),
                    monotonicity=float("nan"), layer_returns=(), segment_ic=(),
                    metrics={"rejection_reason": "insufficient_time_observations"},
                ))
                continue
            mean_ic = float(self._ic_sum[factor] / observations)
            direction = 1 if mean_ic >= 0 else -1
            if observations > 1:
                centered_sum = (
                    self._ic_squared_sum[factor]
                    - observations * mean_ic * mean_ic
                )
                ic_std = float(np.sqrt(max(0.0, centered_sum / (observations - 1))))
            else:
                ic_std = np.nan
            ic_ir = (
                mean_ic / ic_std
                if np.isfinite(ic_std) and ic_std > _EPS else 0.0
            )
            directional_hits = (
                self._positive_ic[factor]
                if direction > 0 else self._negative_ic[factor]
            )
            hit_rate = float(directional_hits / observations)
            coverage = float(
                self.coverage_numerator[factor]
                / max(1, self.context.coverage_denominator)
            )
            if self._gross_count[factor]:
                base_gross_mean = float(
                    self._gross_sum[factor] / self._gross_count[factor]
                )
                rank_gross_mean = direction * base_gross_mean
                rank_net_mean = rank_gross_mean - fixed_cost
            else:
                rank_gross_mean = 0.0
                rank_net_mean = 0.0
            target_dispersion = float(self.context.target_dispersion)
            cost_adjusted_return_score = (
                float(np.clip(
                    rank_net_mean / target_dispersion, -1.0, 1.0
                ))
                if target_dispersion > _EPS else 0.0
            )
            turnover_mean = (
                float(
                    self._turnover_sum[factor]
                    / self._turnover_count[factor]
                )
                if self._turnover_count[factor] else 0.0
            )
            segments = tuple(
                float(
                    self._segment_sum[factor, segment]
                    / self._segment_count[factor, segment]
                )
                for segment in range(len(self._segment_bounds))
                if self._segment_count[factor, segment]
            )
            stable_fraction = (
                float(np.mean(np.sign(segments) == direction))
                if segments else 0.0
            )
            oriented_segment_floor = (
                float(np.min(direction * np.asarray(segments, dtype=float)))
                if segments else 0.0
            )
            bounded_ir = min(abs(ic_ir), 3.0)
            economic_weight = float(self.config.economic_fitness_weight)
            complexity = int(self.complexities[factor])
            fitness = (
                (1.0 - economic_weight) * abs(mean_ic)
                + economic_weight * cost_adjusted_return_score
                + 0.02 * bounded_ir
                + 0.01 * hit_rate
                + 0.01 * stable_fraction
                + self.config.segment_floor_weight * oriented_segment_floor
                - self.config.coverage_penalty * (1.0 - coverage)
                - self.config.complexity_penalty * max(0, complexity - 1)
            )
            results.append(CandidateResult(
                fitness=float(fitness),
                direction=direction,
                mean_ic=mean_ic,
                ic_ir=float(ic_ir),
                ic_hit_rate=hit_rate,
                coverage=coverage,
                observations=observations,
                gross_long_short_mean=float("nan"),
                net_long_short_mean=float("nan"),
                turnover_mean=turnover_mean,
                monotonicity=float("nan"),
                layer_returns=(),
                segment_ic=segments,
                metrics={
                    "complexity": complexity,
                    "stable_segment_fraction": stable_fraction,
                    "oriented_segment_floor_ic": oriented_segment_floor,
                    "coverage_denominator": (
                        "target_and_volatility_control"
                        if self.config.neutralize_volatility
                        and self.context.volatility is not None
                        else "target"
                    ),
                    "rebalance_every_bars": int(
                        self.config.rebalance_every_bars
                    ),
                    "rank_weight_gross_mean": rank_gross_mean,
                    "rank_weight_net_mean": rank_net_mean,
                    "rank_weight_turnover_mean": turnover_mean,
                    "target_cross_sectional_dispersion": target_dispersion,
                    "cost_adjusted_return_score": cost_adjusted_return_score,
                    "economic_fitness_weight": economic_weight,
                    "mining_cost_definition": (
                        "fixed_annual_cost_prorated_by_target_holding_bars"
                    ),
                    "annual_transaction_cost_bps": float(
                        self.context.target.spec.cost_bps
                    ),
                    "cost_per_target_observation": float(fixed_cost),
                    "cost_uses_turnover": False,
                },
            ))
        return results
