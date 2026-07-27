"""Fast development-stage diagnostics for candidate factor signals.

These metrics guide search and debugging.  They are deliberately not a
replacement for the framework's protocol-bound HAC and layered tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from factor_mining.api import TargetSpec


_EPS = 1e-12


@dataclass(frozen=True)
class PreparedTarget:
    index: pd.DatetimeIndex
    symbols: pd.Index
    values: np.ndarray
    rank_values: np.ndarray
    spec: TargetSpec

    @classmethod
    def from_close(cls, close: pd.DataFrame, spec: TargetSpec) -> "PreparedTarget":
        if close is None or close.empty:
            raise ValueError("close panel is required for the target")
        if spec.session_policy != "allow_cross_session":
            raise ValueError(
                "the first release supports session_policy='allow_cross_session' only"
            )
        entry = close.shift(-spec.entry_delay_bars)
        exit_price = close.shift(-(spec.entry_delay_bars + spec.horizon_bars))
        target = exit_price.divide(entry.where(entry.abs() > _EPS)) - 1.0
        values = target.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
        values.setflags(write=False)
        rank_values = _rank_rows(values).astype(np.float32)
        rank_values.setflags(write=False)
        return cls(
            index=pd.DatetimeIndex(close.index),
            symbols=pd.Index(close.columns),
            values=values,
            rank_values=rank_values,
            spec=spec,
        )


@dataclass(frozen=True)
class ValidationConfig:
    decision_lag_bars: int = 1
    min_cross_section: int = 4
    min_time_observations: int = 30
    mad_clip: float = 5.0
    neutralize_volatility: bool = True
    long_short_fraction: float = 0.2
    layer_count: int = 5
    time_segments: int = 4
    turnover_penalty: float = 0.002
    complexity_penalty: float = 0.0005

    def __post_init__(self) -> None:
        if self.decision_lag_bars < 1:
            raise ValueError("decision_lag_bars must be at least one")
        if self.min_cross_section < 2 or self.min_time_observations < 2:
            raise ValueError("validation minimums must be at least two")
        if not 0.0 < self.long_short_fraction <= 0.5:
            raise ValueError("long_short_fraction must be in (0, 0.5]")
        if self.layer_count < 2 or self.time_segments < 1:
            raise ValueError("layer_count/time_segments are invalid")


@dataclass(frozen=True)
class CandidateResult:
    fitness: float
    direction: int
    mean_ic: float
    ic_ir: float
    ic_hit_rate: float
    coverage: float
    observations: int
    gross_long_short_mean: float
    net_long_short_mean: float
    turnover_mean: float
    monotonicity: float
    layer_returns: tuple[float, ...]
    segment_ic: tuple[float, ...]
    metrics: Mapping[str, object] = field(default_factory=dict)

    def to_metrics(self) -> dict:
        return {
            "diagnostic_fitness": self.fitness,
            "mean_rank_ic": self.mean_ic,
            "rank_ic_ir": self.ic_ir,
            "ic_directional_hit_rate": self.ic_hit_rate,
            "coverage": self.coverage,
            "time_observations": self.observations,
            "gross_long_short_mean": self.gross_long_short_mean,
            "net_long_short_mean": self.net_long_short_mean,
            "turnover_mean": self.turnover_mean,
            "layer_monotonicity": self.monotonicity,
            "layer_returns": list(self.layer_returns),
            "segment_mean_ic": list(self.segment_ic),
            "metric_scope": "mining_diagnostic_not_formal_hac",
            **dict(self.metrics),
        }


def shift_signal(signal: np.ndarray, bars: int) -> np.ndarray:
    value = np.asarray(signal, dtype=np.float32)
    result = np.full(value.shape, np.nan, dtype=np.float32)
    if bars < len(value):
        result[bars:] = value[:-bars]
    return result


def mad_winsorize(signal: np.ndarray, clip: float) -> np.ndarray:
    value = np.asarray(signal, dtype=float)
    usable_rows = np.isfinite(value).any(axis=1)
    median = np.full((len(value), 1), np.nan, dtype=float)
    mad = np.full((len(value), 1), np.nan, dtype=float)
    if usable_rows.any():
        usable = value[usable_rows]
        usable_median = np.nanmedian(usable, axis=1, keepdims=True)
        median[usable_rows] = usable_median
        mad[usable_rows] = np.nanmedian(
            np.abs(usable - usable_median), axis=1, keepdims=True
        )
    scale = 1.4826 * mad
    lower = median - clip * scale
    upper = median + clip * scale
    usable = np.isfinite(scale) & (scale > _EPS)
    clipped = np.where(usable, np.minimum(np.maximum(value, lower), upper), value)
    return clipped.astype(np.float32)


def neutralize_signal(
    signal: np.ndarray,
    *,
    volatility: np.ndarray | None = None,
    group_labels: Sequence[str] | None = None,
) -> np.ndarray:
    """Cross-sectionally remove group means and one volatility control."""
    result = np.asarray(signal, dtype=float).copy()
    if group_labels is not None:
        labels = np.asarray(group_labels)
        if labels.shape != (result.shape[1],):
            raise ValueError("group_labels must contain one label per symbol")
        for label in np.unique(labels):
            columns = labels == label
            result[:, columns] -= _row_nanmean(result[:, columns])

    row_mean = _row_nanmean(result)
    result -= row_mean
    if volatility is not None:
        control = np.asarray(volatility, dtype=float)
        if control.shape != result.shape:
            raise ValueError("volatility control shape does not match signal")
        control = control - _row_nanmean(control)
        valid = np.isfinite(result) & np.isfinite(control)
        covariance = np.nansum(np.where(valid, result * control, np.nan), axis=1, keepdims=True)
        variance = np.nansum(np.where(valid, control * control, np.nan), axis=1, keepdims=True)
        beta = np.divide(
            covariance,
            variance,
            out=np.zeros_like(covariance),
            where=variance > _EPS,
        )
        result = result - beta * control
    return result.astype(np.float32)


def prepare_signal(
    raw_signal: np.ndarray,
    config: ValidationConfig,
    *,
    volatility: np.ndarray | None = None,
    group_labels: Sequence[str] | None = None,
) -> np.ndarray:
    signal = shift_signal(raw_signal, config.decision_lag_bars)
    signal = mad_winsorize(signal, config.mad_clip)
    aligned_volatility = None
    if volatility is not None:
        aligned_volatility = shift_signal(volatility, config.decision_lag_bars)
    return neutralize_signal(
        signal,
        volatility=(
            aligned_volatility if config.neutralize_volatility else None
        ),
        group_labels=group_labels,
    )


def _rank_rows(value: np.ndarray) -> np.ndarray:
    return pd.DataFrame(value).rank(axis=1, method="average", pct=True).to_numpy()


def _row_nanmean(value: np.ndarray) -> np.ndarray:
    finite = np.isfinite(value)
    total = np.where(finite, value, 0.0).sum(axis=1, keepdims=True)
    count = finite.sum(axis=1, keepdims=True)
    return np.divide(
        total,
        count,
        out=np.full_like(total, np.nan, dtype=float),
        where=count > 0,
    )


def _rank_ic(
    signal: np.ndarray, target_rank: np.ndarray, minimum: int
) -> tuple[np.ndarray, np.ndarray]:
    signal_rank = _rank_rows(signal)
    valid = np.isfinite(signal_rank) & np.isfinite(target_rank)
    count = valid.sum(axis=1)
    x = np.where(valid, signal_rank, np.nan)
    y = np.where(valid, target_rank, np.nan)
    x -= _row_nanmean(x)
    y -= _row_nanmean(y)
    numerator = np.nansum(x * y, axis=1)
    denominator = np.sqrt(np.nansum(x * x, axis=1) * np.nansum(y * y, axis=1))
    result = np.divide(
        numerator,
        denominator,
        out=np.full(len(signal), np.nan),
        where=(count >= minimum) & (denominator > _EPS),
    )
    return result, signal_rank


def _portfolio_diagnostics(
    signal: np.ndarray,
    target: np.ndarray,
    fraction: float,
    minimum: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = signal.shape
    valid = np.isfinite(signal) & np.isfinite(target)
    counts = valid.sum(axis=1)
    usable = counts >= minimum
    order = np.argsort(
        np.where(valid, signal, np.inf), axis=1, kind="stable"
    )
    positions = np.arange(columns)[None, :]
    sides = np.maximum(1, np.floor(counts * fraction).astype(int))
    ordered_weights = np.zeros((rows, columns), dtype=np.float32)
    short = usable[:, None] & (positions < sides[:, None])
    long = (
        usable[:, None]
        & (positions >= (counts - sides)[:, None])
        & (positions < counts[:, None])
    )
    scale = np.divide(
        0.5,
        sides,
        out=np.zeros(rows, dtype=np.float32),
        where=sides > 0,
    )
    ordered_weights[short] = np.broadcast_to(-scale[:, None], short.shape)[short]
    ordered_weights[long] = np.broadcast_to(scale[:, None], long.shape)[long]
    weights = np.zeros_like(ordered_weights)
    np.put_along_axis(weights, order, ordered_weights, axis=1)
    returns = np.full(rows, np.nan)
    returns[usable] = np.nansum(
        weights[usable] * target[usable], axis=1, dtype=float
    )
    turnover = np.full(rows, np.nan)
    if rows > 1:
        turnover[1:] = 0.5 * np.abs(np.diff(weights, axis=0)).sum(axis=1)
    return returns, turnover


def _layer_diagnostics(
    signal: np.ndarray, target: np.ndarray, layers: int, minimum: int
) -> tuple[tuple[float, ...], float]:
    rows, columns = signal.shape
    valid = np.isfinite(signal) & np.isfinite(target)
    counts = valid.sum(axis=1)
    usable = counts >= max(minimum, layers)
    order = np.argsort(
        np.where(valid, signal, np.inf), axis=1, kind="stable"
    )
    ordered_target = np.take_along_axis(target, order, axis=1)
    positions = np.arange(columns)[None, :]
    quotient, remainder = np.divmod(counts, layers)
    layer_means = np.full((rows, layers), np.nan, dtype=float)
    for layer in range(layers):
        start = layer * quotient + np.minimum(layer, remainder)
        end = (layer + 1) * quotient + np.minimum(layer + 1, remainder)
        selected = (
            usable[:, None]
            & (positions >= start[:, None])
            & (positions < end[:, None])
        )
        selected_count = selected.sum(axis=1)
        selected_sum = np.where(selected, ordered_target, 0.0).sum(
            axis=1, dtype=float
        )
        layer_means[:, layer] = np.divide(
            selected_sum,
            selected_count,
            out=np.full(rows, np.nan, dtype=float),
            where=selected_count > 0,
        )
    means = np.nanmean(layer_means[usable], axis=0) if usable.any() else np.full(
        layers, np.nan
    )
    finite = np.isfinite(means)
    monotonicity = np.nan
    if finite.sum() >= 2:
        monotonicity = float(
            pd.Series(np.arange(layers)[finite]).corr(
                pd.Series(means[finite]), method="spearman"
            )
        )
    return tuple(float(value) for value in means), monotonicity


def evaluate_candidate(
    signal: np.ndarray,
    target: PreparedTarget,
    config: ValidationConfig,
    *,
    complexity: int = 1,
    volatility: np.ndarray | None = None,
    group_labels: Sequence[str] | None = None,
    full_diagnostics: bool = True,
) -> CandidateResult:
    if np.asarray(signal).shape != target.values.shape:
        raise ValueError("signal and target shapes differ")
    prepared = prepare_signal(
        signal,
        config,
        volatility=volatility,
        group_labels=group_labels,
    )
    ic, signal_rank = _rank_ic(
        prepared, target.rank_values, config.min_cross_section
    )
    valid_ic = ic[np.isfinite(ic)]
    observations = int(len(valid_ic))
    if observations < config.min_time_observations:
        return CandidateResult(
            fitness=float("-inf"), direction=1, mean_ic=float("nan"),
            ic_ir=float("nan"), ic_hit_rate=float("nan"), coverage=0.0,
            observations=observations, gross_long_short_mean=float("nan"),
            net_long_short_mean=float("nan"), turnover_mean=float("nan"),
            monotonicity=float("nan"), layer_returns=(), segment_ic=(),
            metrics={"rejection_reason": "insufficient_time_observations"},
        )

    mean_ic = float(np.mean(valid_ic))
    direction = 1 if mean_ic >= 0 else -1
    ic_std = float(np.std(valid_ic, ddof=1)) if observations > 1 else np.nan
    ic_ir = mean_ic / ic_std if np.isfinite(ic_std) and ic_std > _EPS else 0.0
    hit_rate = float(np.mean(np.sign(valid_ic) == direction))
    finite_target = np.isfinite(target.values)
    coverage = float((np.isfinite(prepared) & finite_target).sum() / max(1, finite_target.sum()))

    if full_diagnostics:
        gross, turnover = _portfolio_diagnostics(
            direction * prepared,
            target.values,
            config.long_short_fraction,
            config.min_cross_section,
        )
        net = gross - np.nan_to_num(turnover) * (
            target.spec.cost_bps / 10_000.0
        )
        layers, monotonicity = _layer_diagnostics(
            direction * prepared,
            target.values,
            config.layer_count,
            config.min_cross_section,
        )
        gross_mean = float(np.nanmean(gross))
        net_mean = float(np.nanmean(net))
        turnover_mean = (
            float(np.nanmean(turnover)) if np.isfinite(turnover).any() else 0.0
        )
    else:
        changes = np.abs(np.diff(signal_rank, axis=0))
        turnover_mean = (
            float(np.nanmean(changes)) if np.isfinite(changes).any() else 0.0
        )
        gross_mean = float("nan")
        net_mean = float("nan")
        layers = ()
        monotonicity = float("nan")
    segments = tuple(
        float(np.nanmean(part))
        for part in np.array_split(ic, config.time_segments)
        if np.isfinite(part).any()
    )
    stable_fraction = float(np.mean(np.sign(segments) == direction)) if segments else 0.0
    bounded_ir = min(abs(ic_ir), 3.0)
    fitness = (
        abs(mean_ic)
        + 0.02 * bounded_ir
        + 0.01 * hit_rate
        + 0.01 * stable_fraction
        - config.turnover_penalty * turnover_mean
        - config.complexity_penalty * max(0, complexity - 1)
    )
    return CandidateResult(
        fitness=float(fitness),
        direction=direction,
        mean_ic=mean_ic,
        ic_ir=float(ic_ir),
        ic_hit_rate=hit_rate,
        coverage=coverage,
        observations=observations,
        gross_long_short_mean=gross_mean,
        net_long_short_mean=net_mean,
        turnover_mean=turnover_mean,
        monotonicity=float(monotonicity),
        layer_returns=layers,
        segment_ic=segments,
        metrics={"complexity": int(complexity)},
    )
