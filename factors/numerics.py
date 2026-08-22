"""Small array kernels shared by factor implementations."""

from __future__ import annotations

import numpy as np


def count_isolated_peaks(valid, jumps, high, low) -> np.ndarray:
    """Count isolated overlapping-range peaks by column; -1 means <10 bars."""

    arrays = [
        np.asarray(valid, dtype=bool),
        np.asarray(jumps, dtype=bool),
        np.asarray(high, dtype=float),
        np.asarray(low, dtype=float),
    ]
    shape = arrays[0].shape
    if len(shape) != 2 or any(value.shape != shape for value in arrays[1:]):
        raise ValueError("kernel inputs must be same-shaped two-dimensional arrays")
    valid_array, jump_array, high_array, low_array = arrays
    result = np.full(shape[1], -1, dtype=np.int64)
    for column in range(shape[1]):
        observed = np.flatnonzero(valid_array[:, column])
        if observed.size < 10:
            continue
        previous, current, following = observed[:-2], observed[1:-1], observed[2:]
        eligible = jump_array[current, column] & ~(
            jump_array[previous, column] & jump_array[following, column]
        )
        bounds = np.column_stack((
            high_array[previous, column],
            low_array[previous, column],
            high_array[following, column],
            low_array[following, column],
        ))
        overlap = np.maximum(bounds[:, 1], bounds[:, 3]) <= np.minimum(
            bounds[:, 0], bounds[:, 2]
        )
        result[column] = np.count_nonzero(
            eligible & ~np.isnan(bounds).any(axis=1) & overlap
        )
    return result


def histogram_window_l1_stability(values, block_size: int = 5) -> float:
    """Match repeated window-vs-rest histogram distances without rebuilding rest."""
    array = np.asarray(values, dtype=float)
    total, _ = np.histogram(array, bins=10, range=(-4, 4))
    distances = []
    for start in range(0, len(array), block_size):
        window = array[start:start + block_size]
        if len(window) < 3 or len(array) - len(window) < 10:
            continue
        inside, _ = np.histogram(window, bins=10, range=(-4, 4))
        outside = total - inside
        inside_total, outside_total = inside.sum(), outside.sum()
        if inside_total == 0 or outside_total == 0:
            distances.append(np.nan)
        else:
            distances.append(float(np.abs(
                inside / inside_total - outside / outside_total
            ).sum()))
    return float(np.std(distances, ddof=0)) if distances else 0.0


def rolling_split_sum_difference(
    returns, scores, window: int = 20, min_periods: int = 15
) -> np.ndarray:
    """Split prior-window returns at the score median and subtract group sums."""

    return_array = np.asarray(returns, dtype=float)
    score_array = np.asarray(scores, dtype=float)
    if return_array.ndim != 2 or return_array.shape != score_array.shape:
        raise ValueError("returns and scores must be same-shaped two-dimensional arrays")
    window = int(window)
    min_periods = int(min_periods)
    if window < 1 or min_periods < 1 or min_periods > window:
        raise ValueError("require 1 <= min_periods <= window")
    result = np.full(return_array.shape, np.nan, dtype=float)
    for row in range(window, len(return_array)):
        row_returns = return_array[row - window:row]
        row_scores = score_array[row - window:row]
        for column in range(return_array.shape[1]):
            valid = ~np.isnan(row_returns[:, column])
            if np.count_nonzero(valid) < min_periods:
                continue
            observed_returns = row_returns[valid, column]
            observed_scores = row_scores[valid, column]
            high = observed_scores > np.median(observed_scores)
            result[row, column] = (
                np.sum(observed_returns[high]) - np.sum(observed_returns[~high])
            )
    return result


def rolling_linear_slope(
    values, window: int = 20, min_periods: int = 8
) -> np.ndarray:
    """OLS slope for fixed-width rolling columns without Python callbacks."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("values must be a two-dimensional array")
    window, min_periods = int(window), int(min_periods)
    if window < 2 or min_periods < 2 or min_periods > window:
        raise ValueError("require 2 <= min_periods <= window")

    rows = array.shape[0]
    result = np.full(array.shape, np.nan, dtype=float)
    for end in range(min_periods - 1, rows):
        start = max(0, end - window + 1)
        segment = array[start:end + 1]
        complete = np.isfinite(segment).all(axis=0)
        x = np.arange(len(segment), dtype=float)
        x -= x.mean()
        observed = segment[:, complete]
        observed = observed - observed.mean(axis=0)
        result[end, complete] = x @ observed / np.dot(x, x)
    return result


def variance_ratio(values, q: int) -> float:
    """Match the established autocorrelation-form variance-ratio statistic."""

    array = np.asarray(values, dtype=float)
    horizon = int(q)
    if array.ndim != 1 or horizon < 2:
        raise ValueError("variance ratio requires a one-dimensional array and horizon >= 2")
    if len(array) < 5 * horizon:
        return 0.0
    mean = array.mean()
    variance = np.sum((array[1:] - mean) ** 2) / (len(array) - 1)
    if variance < 1e-12:
        return 0.0
    result = 1.0
    for lag in range(1, horizon):
        current, previous = array[lag:], array[:-lag]
        current = current - current.mean()
        previous = previous - previous.mean()
        current_ss = np.dot(current, current)
        previous_ss = np.dot(previous, previous)
        if (
            np.sqrt(current_ss / len(current)) > 1e-12
            and np.sqrt(previous_ss / len(previous)) > 1e-12
        ):
            correlation = np.dot(current, previous) / np.sqrt(
                current_ss * previous_ss
            )
            result += 2.0 * (1.0 - lag / horizon) * correlation
    return float(result - 1.0)
