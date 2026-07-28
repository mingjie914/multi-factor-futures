"""Fast development-stage diagnostics for candidate factor signals.

These metrics guide search and debugging.  They are deliberately not a
replacement for the framework's protocol-bound HAC and layered tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from core.period import PeriodContext
from factor_mining.api import TargetSpec


_EPS = 1e-12


def _fixed_annual_cost_per_target(spec: TargetSpec) -> float:
    """Prorate the fixed annual validation cost over one target holding span."""
    bars_per_year = PeriodContext.from_string(
        spec.decision_frequency
    ).bars_per_year
    return (
        float(spec.cost_bps)
        / 10_000.0
        * float(spec.horizon_bars)
        / float(bars_per_year)
    )


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
        # Shift each contract on its own valid-bar calendar.  A matrix-wide
        # shift would turn a missing quote for one contract into a different
        # economic horizon from the formal framework's return label.
        close_values = close.to_numpy(dtype=float, copy=False)
        values = np.full(close_values.shape, np.nan, dtype=np.float32)
        offset = int(spec.entry_delay_bars + spec.horizon_bars)
        for column in range(close_values.shape[1]):
            valid_rows = np.flatnonzero(np.isfinite(close_values[:, column]))
            if len(valid_rows) <= offset:
                continue
            series = close_values[valid_rows, column]
            output_rows = valid_rows[:-offset]
            entry = series[
                spec.entry_delay_bars:len(series) - spec.horizon_bars
            ]
            exit_price = series[offset:]
            usable = np.abs(entry) > _EPS
            values[output_rows[usable], column] = (
                exit_price[usable] / entry[usable] - 1.0
            ).astype(np.float32)
        values[~np.isfinite(values)] = np.nan
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
    complexity_penalty: float = 0.0005
    coverage_penalty: float = 0.0
    segment_floor_weight: float = 0.0
    rebalance_every_bars: int = 1
    economic_fitness_weight: float = 0.50

    def __post_init__(self) -> None:
        if self.decision_lag_bars < 1:
            raise ValueError("decision_lag_bars must be at least one")
        if self.min_cross_section < 2 or self.min_time_observations < 2:
            raise ValueError("validation minimums must be at least two")
        if not 0.0 < self.long_short_fraction <= 0.5:
            raise ValueError("long_short_fraction must be in (0, 0.5]")
        if self.layer_count < 2 or self.time_segments < 1:
            raise ValueError("layer_count/time_segments are invalid")
        if self.rebalance_every_bars < 1:
            raise ValueError("rebalance_every_bars must be at least one")
        if not 0.0 <= self.economic_fitness_weight <= 1.0:
            raise ValueError("economic_fitness_weight must be in [0, 1]")
        if any(value < 0.0 for value in (
            self.complexity_penalty,
            self.coverage_penalty,
            self.segment_floor_weight,
        )):
            raise ValueError("validation fitness penalties/weights cannot be negative")


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


def _row_nanstd(value: np.ndarray) -> np.ndarray:
    finite = np.isfinite(value)
    count = finite.sum(axis=1)
    mean = _row_nanmean(value)
    centered = np.where(finite, value - mean, 0.0)
    variance = np.divide(
        np.einsum("ij,ij->i", centered, centered),
        count,
        out=np.full(len(value), np.nan, dtype=float),
        where=count > 0,
    )
    return np.sqrt(variance)


def _rank_ic(
    signal: np.ndarray, target_rank: np.ndarray, minimum: int
) -> tuple[np.ndarray, np.ndarray]:
    signal_rank = _rank_rows(signal)
    return _rank_ic_from_ranks(signal_rank, target_rank, minimum), signal_rank


def _rank_ic_from_ranks(
    signal_rank: np.ndarray, target_rank: np.ndarray, minimum: int
) -> np.ndarray:
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
        out=np.full(len(signal_rank), np.nan),
        where=(count >= minimum) & (denominator > _EPS),
    )
    return result


def _mean_rank_correlation(
    left: np.ndarray,
    right: np.ndarray,
    minimum: int,
    *,
    chunk_rows: int = 8192,
) -> tuple[float, int]:
    """Memory-bounded mean row correlation for large minute panels."""

    total = 0.0
    observations = 0
    for start in range(0, len(left), chunk_rows):
        stop = min(start + chunk_rows, len(left))
        x_raw = left[start:stop]
        y_raw = right[start:stop]
        valid = np.isfinite(x_raw) & np.isfinite(y_raw)
        counts = valid.sum(axis=1)
        x = np.where(valid, x_raw, np.nan)
        y = np.where(valid, y_raw, np.nan)
        x -= _row_nanmean(x)
        y -= _row_nanmean(y)
        numerator = np.nansum(x * y, axis=1)
        denominator = np.sqrt(
            np.nansum(x * x, axis=1) * np.nansum(y * y, axis=1)
        )
        usable = (counts >= minimum) & (denominator > _EPS)
        correlations = np.divide(
            numerator,
            denominator,
            out=np.full(len(x), np.nan, dtype=float),
            where=usable,
        )[usable]
        total += float(np.sum(correlations))
        observations += int(len(correlations))
    return (
        float(total / observations) if observations else 0.0,
        observations,
    )


def _rank_weight_portfolio(
    signal_rank: np.ndarray,
    target: np.ndarray,
    *,
    direction: int,
    rebalance_every_bars: int,
    minimum: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast rank-weight portfolio used only by the mining fitness.

    The weighting convention matches the formal turnover test: demean ranks,
    normalize gross exposure to one, and charge half-turnover only when the
    declared schedule rebalances.  It avoids an argsort in every GP fitness
    evaluation.
    """

    oriented = float(direction) * np.asarray(signal_rank, dtype=float)
    valid_signal = np.isfinite(oriented)
    counts = valid_signal.sum(axis=1)
    centered = oriented - _row_nanmean(oriented)
    gross_exposure = np.nansum(np.abs(centered), axis=1, keepdims=True)
    weights = np.divide(
        centered,
        gross_exposure,
        out=np.zeros_like(centered),
        where=(gross_exposure > _EPS) & valid_signal,
    )
    decision_rows = np.arange(0, len(weights), int(rebalance_every_bars))
    decision_weights = weights[decision_rows]
    decision_target = np.asarray(target, dtype=float)[decision_rows]
    joint_counts = (
        np.isfinite(decision_target) & valid_signal[decision_rows]
    ).sum(axis=1)
    usable = (counts[decision_rows] >= minimum) & (joint_counts >= minimum)
    gross = np.full(len(decision_rows), np.nan, dtype=float)
    gross[usable] = np.nansum(
        decision_weights[usable] * decision_target[usable], axis=1
    )
    turnover = np.full(len(decision_rows), np.nan, dtype=float)
    if len(decision_rows) > 1:
        turnover[1:] = 0.5 * np.abs(np.diff(decision_weights, axis=0)).sum(axis=1)
    return gross, turnover


def predictive_ic_decay(
    prepared_signal: np.ndarray,
    targets: Mapping[int, PreparedTarget],
    *,
    min_cross_section: int,
    signal_rank: np.ndarray | None = None,
) -> dict:
    """Return a true forward-horizon rank-IC curve in decision bars."""

    curve = {}
    signal_rank = (
        _rank_rows(prepared_signal) if signal_rank is None else signal_rank
    )
    for horizon in sorted(int(value) for value in targets):
        mean_ic, observations = _mean_rank_correlation(
            signal_rank,
            targets[horizon].rank_values,
            min_cross_section,
        )
        curve[str(horizon)] = {
            "mean_rank_ic": mean_ic,
            "n_observations": observations,
        }
    if curve:
        eligible_horizons = [
            key for key, value in curve.items()
            if int(value["n_observations"]) > 0
        ]
        peak = max(
            eligible_horizons,
            key=lambda key: (
                abs(float(curve[key]["mean_rank_ic"])),
                -int(key),
            ),
        ) if eligible_horizons else ""
    else:
        peak = ""
    return {
        "horizon_unit": "decision_bars",
        "curve": curve,
        "peak_horizon_bars": int(peak) if peak else None,
        "peak_abs_rank_ic": (
            abs(float(curve[peak]["mean_rank_ic"])) if peak else 0.0
        ),
    }


def signal_rank_persistence(
    prepared_signal: np.ndarray,
    *,
    lags: Sequence[int],
    min_cross_section: int,
    half_life_threshold: float = 0.50,
    signal_rank: np.ndarray | None = None,
) -> dict:
    """Measure persistence of cross-sectional signal ranks across bar lags."""

    ranks = (
        _rank_rows(np.asarray(prepared_signal, dtype=float))
        if signal_rank is None
        else signal_rank
    )
    curve = {}
    for lag in sorted(set(int(value) for value in lags)):
        if lag < 1:
            raise ValueError("signal persistence lags must be positive")
        if lag >= len(ranks):
            curve[str(lag)] = {"mean_rank_autocorrelation": 0.0, "n_observations": 0}
            continue
        mean_correlation, observations = _mean_rank_correlation(
            ranks[:-lag], ranks[lag:], min_cross_section
        )
        curve[str(lag)] = {
            "mean_rank_autocorrelation": mean_correlation,
            "n_observations": observations,
        }
    half_life = next(
        (
            int(lag)
            for lag in sorted(curve, key=int)
            if int(curve[lag]["n_observations"]) > 0
            and float(curve[lag]["mean_rank_autocorrelation"]) < half_life_threshold
        ),
        None,
    )
    return {
        "lag_unit": "decision_bars",
        "definition": "mean_cross_sectional_rank_correlation_t_vs_t_plus_lag",
        "curve": curve,
        "half_life_threshold": float(half_life_threshold),
        "half_life_bars": half_life,
        "half_life_censored": half_life is None,
    }


def _portfolio_diagnostics(
    signal: np.ndarray,
    target: np.ndarray,
    fraction: float,
    minimum: int,
    rebalance_every_bars: int = 1,
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
    decision_rows = np.arange(0, rows, int(rebalance_every_bars))
    decision_usable = usable[decision_rows]
    returns = np.full(rows, np.nan)
    selected_rows = decision_rows[decision_usable]
    returns[selected_rows] = np.nansum(
        weights[selected_rows] * target[selected_rows], axis=1, dtype=float
    )
    turnover = np.full(rows, np.nan)
    if len(decision_rows) > 1:
        turnover[decision_rows[1:]] = 0.5 * np.abs(
            np.diff(weights[decision_rows], axis=0)
        ).sum(axis=1)
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
    eligibility: np.ndarray | None = None,
    processing_eligibility: np.ndarray | None = None,
    full_diagnostics: bool = True,
) -> CandidateResult:
    raw_signal = np.asarray(signal)
    if raw_signal.shape != target.values.shape:
        raise ValueError("signal and target shapes differ")
    eligible = None
    processing_eligible = None
    prepared_volatility = volatility
    if eligibility is not None:
        eligible = np.asarray(eligibility)
        if eligible.shape != raw_signal.shape:
            raise ValueError("eligibility shape differs from candidate signal")
        if eligible.dtype != np.bool_:
            raise TypeError("eligibility must be a boolean array")
    if processing_eligibility is not None:
        processing_eligible = np.asarray(processing_eligibility)
        if processing_eligible.shape != raw_signal.shape:
            raise ValueError(
                "processing_eligibility shape differs from candidate signal"
            )
        if processing_eligible.dtype != np.bool_:
            raise TypeError("processing_eligibility must be a boolean array")
    elif eligible is not None:
        processing_eligible = eligible
    if processing_eligible is not None:
        # Use the full computation-period universe before cross-sectional
        # transforms.  The evaluation mask is applied separately below so
        # warm-up observations remain available to lag/rolling operations.
        raw_signal = np.where(processing_eligible, raw_signal, np.nan)
        if volatility is not None:
            prepared_volatility = np.where(
                processing_eligible, volatility, np.nan
            )
    prepared = prepare_signal(
        raw_signal,
        config,
        volatility=prepared_volatility,
        group_labels=group_labels,
    )
    if eligible is not None:
        prepared = np.where(eligible, prepared, np.nan)
    elif processing_eligible is not None:
        prepared = np.where(processing_eligible, prepared, np.nan)
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
    eligible_target = finite_target.copy()
    if eligible is not None:
        eligible_target &= eligible
    elif processing_eligible is not None:
        eligible_target &= processing_eligible
    if config.neutralize_volatility and volatility is not None:
        eligible_target &= np.isfinite(
            shift_signal(volatility, config.decision_lag_bars)
        )
    coverage = float(
        (np.isfinite(prepared) & eligible_target).sum()
        / max(1, eligible_target.sum())
    )

    rank_gross, rank_turnover = _rank_weight_portfolio(
        signal_rank,
        target.values,
        direction=direction,
        rebalance_every_bars=config.rebalance_every_bars,
        minimum=config.min_cross_section,
    )
    fixed_cost_per_target = _fixed_annual_cost_per_target(target.spec)
    rank_net = np.where(
        np.isfinite(rank_gross),
        rank_gross - fixed_cost_per_target,
        np.nan,
    )
    rank_net_mean = (
        float(np.nanmean(rank_net)) if np.isfinite(rank_net).any() else 0.0
    )
    row_dispersion = _row_nanstd(target.values)
    target_dispersion = (
        float(np.nanmean(row_dispersion))
        if np.isfinite(row_dispersion).any()
        else 0.0
    )
    cost_adjusted_return_score = (
        float(np.clip(rank_net_mean / target_dispersion, -1.0, 1.0))
        if target_dispersion > _EPS
        else 0.0
    )

    if full_diagnostics:
        gross, turnover = _portfolio_diagnostics(
            direction * prepared,
            target.values,
            config.long_short_fraction,
            config.min_cross_section,
            rebalance_every_bars=config.rebalance_every_bars,
        )
        net = np.where(
            np.isfinite(gross),
            gross - fixed_cost_per_target,
            np.nan,
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
        turnover_mean = (
            float(np.nanmean(rank_turnover))
            if np.isfinite(rank_turnover).any()
            else 0.0
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
    oriented_segment_floor = (
        float(np.min(direction * np.asarray(segments, dtype=float)))
        if segments else 0.0
    )
    bounded_ir = min(abs(ic_ir), 3.0)
    economic_weight = float(config.economic_fitness_weight)
    fitness = (
        (1.0 - economic_weight) * abs(mean_ic)
        + economic_weight * cost_adjusted_return_score
        + 0.02 * bounded_ir
        + 0.01 * hit_rate
        + 0.01 * stable_fraction
        + config.segment_floor_weight * oriented_segment_floor
        - config.coverage_penalty * (1.0 - coverage)
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
        metrics={
            "complexity": int(complexity),
            "stable_segment_fraction": stable_fraction,
            "oriented_segment_floor_ic": oriented_segment_floor,
            "coverage_denominator": (
                "target_and_volatility_control"
                if config.neutralize_volatility and volatility is not None
                else "target"
            ),
            "rebalance_every_bars": int(config.rebalance_every_bars),
            "rank_weight_gross_mean": (
                float(np.nanmean(rank_gross))
                if np.isfinite(rank_gross).any()
                else 0.0
            ),
            "rank_weight_net_mean": rank_net_mean,
            "rank_weight_turnover_mean": (
                float(np.nanmean(rank_turnover))
                if np.isfinite(rank_turnover).any()
                else 0.0
            ),
            "target_cross_sectional_dispersion": target_dispersion,
            "cost_adjusted_return_score": cost_adjusted_return_score,
            "economic_fitness_weight": economic_weight,
            "mining_cost_definition": (
                "fixed_annual_cost_prorated_by_target_holding_bars"
            ),
            "annual_transaction_cost_bps": float(target.spec.cost_bps),
            "cost_per_target_observation": float(fixed_cost_per_target),
            "cost_uses_turnover": False,
        },
    )
