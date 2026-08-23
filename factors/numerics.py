"""Small array kernels shared by factor implementations."""

from __future__ import annotations

import importlib.util
from importlib.metadata import PackageNotFoundError, version
import os

import numpy as np


_RETURN_STAT_FIELDS = (
    "count", "sum", "mean", "std", "abs_mean", "max", "positive_count",
    "negative_count", "raw_m2", "raw_m4", "skew", "excess_kurtosis",
)
_UNARY_STAT_FIELDS = (
    "count", "sum", "mean", "std", "sample_std", "min", "max", "first",
    "last", "skew",
)
_PAIR_STAT_FIELDS = (
    "count", "left_mean", "right_mean", "left_std", "right_std",
    "covariance", "correlation", "slope", "residual_std", "product_mean",
)
_LAGGED_PAIR_STAT_FIELDS = ("count", "left_std", "right_std", "correlation")


def factor_kernel_mode() -> str:
    configured = os.environ.get("MF_FACTOR_KERNEL_MODE")
    mode = (
        configured.strip().lower()
        if configured is not None
        else ("native" if importlib.util.find_spec("_mf_factor_kernels") else "reference")
    )
    if mode not in {"reference", "shadow", "native"}:
        raise ValueError("MF_FACTOR_KERNEL_MODE must be reference, shadow, or native")
    return mode


def factor_kernel_contract() -> dict[str, str | None]:
    """Return the active numeric runtime identity for research manifests."""
    mode = factor_kernel_mode()
    native_version = None
    if mode in {"shadow", "native"}:
        try:
            native_version = version("mf-factor-kernels")
        except PackageNotFoundError:
            native_version = "unavailable"
    return {"factor_kernel_mode": mode, "factor_kernel_version": native_version}


def _load_native_module():
    import _mf_factor_kernels

    return _mf_factor_kernels


def native_array_kernel(name: str, *args) -> np.ndarray:
    try:
        native_module = _load_native_module()
    except ImportError as exc:
        raise RuntimeError("native factor kernels are not installed") from exc
    try:
        kernel = getattr(native_module, name)
    except AttributeError as exc:
        raise RuntimeError(f"native factor kernel {name!r} is unavailable") from exc
    return np.asarray(kernel(*args), dtype=float)


def assert_native_equal(reference, candidate, name: str) -> None:
    if not (
        np.array_equal(np.isnan(reference), np.isnan(candidate))
        and np.allclose(reference, candidate, rtol=1e-12, atol=1e-12, equal_nan=True)
    ):
        raise RuntimeError(f"native {name} differs from reference")


def daily_unary_statistics(values, day_offsets) -> dict[str, np.ndarray]:
    array = np.asarray(values, dtype=np.float64, order="C")
    offsets = np.asarray(day_offsets, dtype=np.int64)
    mode = factor_kernel_mode()
    reference = None
    if mode != "native":
        reference = np.full((len(offsets) - 1, array.shape[1], 10), np.nan)
        for day, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            for column in range(array.shape[1]):
                observed = array[start:end, column]
                observed = observed[~np.isnan(observed)]
                if observed.size == 0:
                    continue
                mean = observed.mean()
                std = observed.std(ddof=0)
                reference[day, column, :9] = (
                    observed.size, observed.sum(), mean, std,
                    observed.std(ddof=1) if observed.size >= 2 else np.nan,
                    observed.min(), observed.max(), observed[0], observed[-1],
                )
                if observed.size >= 3:
                    reference[day, column, 9] = pd_series_skew(observed, mean, std)
    native = native_array_kernel("daily_unary_stats", array, offsets) if mode != "reference" else None
    if mode == "shadow":
        assert_native_equal(reference, native, "daily_unary_stats")
    output = native if mode == "native" else reference
    return dict(zip(_UNARY_STAT_FIELDS, np.moveaxis(output, 2, 0)))


def daily_pair_statistics(left, right, day_offsets) -> dict[str, np.ndarray]:
    left_array = np.asarray(left, dtype=np.float64, order="C")
    right_array = np.asarray(right, dtype=np.float64, order="C")
    if left_array.shape != right_array.shape:
        raise ValueError("pair inputs must have the same shape")
    offsets = np.asarray(day_offsets, dtype=np.int64)
    mode = factor_kernel_mode()
    reference = None
    if mode != "native":
        reference = np.full((len(offsets) - 1, left_array.shape[1], 10), np.nan)
        for day, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            for column in range(left_array.shape[1]):
                x = left_array[start:end, column]
                y = right_array[start:end, column]
                valid = ~(np.isnan(x) | np.isnan(y))
                x, y = x[valid], y[valid]
                if x.size == 0:
                    continue
                x_mean, y_mean = x.mean(), y.mean()
                x_centered, y_centered = x - x_mean, y - y_mean
                x_variance = np.mean(x_centered ** 2)
                y_variance = np.mean(y_centered ** 2)
                covariance = np.mean(x_centered * y_centered)
                x_std, y_std = np.sqrt(x_variance), np.sqrt(y_variance)
                reference[day, column, :6] = (
                    x.size, x_mean, y_mean, x_std, y_std, covariance,
                )
                if x_std > 0 and y_std > 0:
                    reference[day, column, 6] = covariance / (x_std * y_std)
                if x_variance > 0:
                    slope = covariance / x_variance
                    reference[day, column, 7] = slope
                    reference[day, column, 8] = np.std(
                        y_centered - slope * x_centered, ddof=0
                    )
                reference[day, column, 9] = np.mean(x * y)
    native = native_array_kernel(
        "daily_pair_stats", left_array, right_array, offsets
    ) if mode != "reference" else None
    if mode == "shadow":
        assert_native_equal(reference, native, "daily_pair_stats")
    output = native if mode == "native" else reference
    return dict(zip(_PAIR_STAT_FIELDS, np.moveaxis(output, 2, 0)))


def daily_lagged_pair_statistics(
    left, right, day_offsets, lag: int = 1
) -> dict[str, np.ndarray]:
    left_array = np.asarray(left, dtype=np.float64, order="C")
    right_array = np.asarray(right, dtype=np.float64, order="C")
    if left_array.shape != right_array.shape or lag < 1:
        raise ValueError("pair inputs must have the same shape and lag must be positive")
    offsets = np.asarray(day_offsets, dtype=np.int64)
    mode = factor_kernel_mode()
    reference = None
    if mode != "native":
        reference = np.full((len(offsets) - 1, left_array.shape[1], 4), np.nan)
        for day, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            for column in range(left_array.shape[1]):
                x = left_array[start:end, column]
                y = right_array[start:end, column]
                valid = ~(np.isnan(x) | np.isnan(y))
                pairs = np.column_stack((x[valid], y[valid]))
                if len(pairs) <= lag:
                    continue
                x_lagged, y_lagged = pairs[:-lag, 0], pairs[lag:, 1]
                reference[day, column, 0] = len(x_lagged)
                x_std = x_lagged.std(ddof=0)
                y_std = y_lagged.std(ddof=0)
                reference[day, column, 1:3] = (x_std, y_std)
                if x_std > 0 and y_std > 0:
                    reference[day, column, 3] = np.corrcoef(
                        x_lagged, y_lagged
                    )[0, 1]
    native = native_array_kernel(
        "daily_lagged_pair_stats", left_array, right_array, offsets, lag
    ) if mode != "reference" else None
    if mode == "shadow":
        assert_native_equal(reference, native, "daily_lagged_pair_stats")
    output = native if mode == "native" else reference
    return dict(zip(_LAGGED_PAIR_STAT_FIELDS, np.moveaxis(output, 2, 0)))


def daily_tail_statistics(values, day_offsets, windows) -> dict[str, np.ndarray]:
    array = np.asarray(values, dtype=np.float64, order="C")
    offsets = np.asarray(day_offsets, dtype=np.int64)
    windows = np.asarray(windows, dtype=np.int64)
    if windows.ndim != 1 or np.any(windows < 1):
        raise ValueError("windows must be a one-dimensional positive array")
    mode = factor_kernel_mode()
    reference = None
    if mode != "native":
        reference = np.full(
            (len(offsets) - 1, array.shape[1], len(windows) + 2), np.nan
        )
        for day, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            for column in range(array.shape[1]):
                observed = array[start:end, column]
                observed = observed[~np.isnan(observed)]
                if observed.size == 0:
                    continue
                reference[day, column, :2] = (observed.size, observed[-1])
                for index, window in enumerate(windows):
                    if observed.size >= window + 1:
                        reference[day, column, index + 2] = observed[
                            -window - 1:-1
                        ].mean()
    native = native_array_kernel(
        "daily_tail_means", array, offsets, windows
    ) if mode != "reference" else None
    if mode == "shadow":
        assert_native_equal(reference, native, "daily_tail_means")
    output = native if mode == "native" else reference
    return {
        "count": output[:, :, 0],
        "last": output[:, :, 1],
        "means": output[:, :, 2:],
    }


def daily_breakout_statistics(high, low, close, day_offsets) -> dict[str, np.ndarray]:
    high_array = np.asarray(high, dtype=np.float64, order="C")
    low_array = np.asarray(low, dtype=np.float64, order="C")
    close_array = np.asarray(close, dtype=np.float64, order="C")
    if high_array.shape != low_array.shape or high_array.shape != close_array.shape:
        raise ValueError("OHLC inputs must have the same shape")
    offsets = np.asarray(day_offsets, dtype=np.int64)
    mode = factor_kernel_mode()
    reference = None
    if mode != "native":
        reference = np.full((len(offsets) - 1, close_array.shape[1], 2), np.nan)
        for day, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            if end - start < 30:
                continue
            for column in range(close_array.shape[1]):
                observed = np.column_stack((
                    high_array[start:end, column],
                    low_array[start:end, column],
                    close_array[start:end, column],
                ))
                observed = observed[~np.isnan(observed).any(axis=1)]
                if len(observed) < 30:
                    continue
                atr = np.mean(observed[-20:, 0] - observed[-20:, 1])
                if atr < 1e-12:
                    reference[day, column] = 0.0
                    continue
                retraces = []
                breaks = holds = 0
                for index in range(20, len(observed)):
                    high_max = observed[index - 20:index, 0].max()
                    low_min = observed[index - 20:index, 1].min()
                    price = observed[index, 2]
                    close_after = observed[min(index + 4, len(observed) - 1), 2]
                    if price >= high_max:
                        retraces.append(
                            (observed[index:index + 5, 0].max() - close_after) / atr
                        )
                    elif price <= low_min:
                        retraces.append(
                            (close_after - observed[index:index + 5, 1].min()) / atr
                        )
                    if price >= high_max and high_max - observed[index - 1, 0] > 0:
                        breaks += 1
                        holds += int(close_after >= price - 0.5 * atr)
                    elif price <= low_min and observed[index - 1, 1] - low_min > 0:
                        breaks += 1
                        holds += int(close_after <= price + 0.5 * atr)
                reference[day, column, 0] = np.mean(retraces) if retraces else 0.0
                reference[day, column, 1] = holds / breaks if breaks else 0.5
    native = native_array_kernel(
        "daily_breakout_features", high_array, low_array, close_array, offsets
    ) if mode != "reference" else None
    if mode == "shadow":
        assert_native_equal(reference, native, "daily_breakout_features")
    output = native if mode == "native" else reference
    return {"false_retrace": output[:, :, 0], "quality": output[:, :, 1]}


def daily_smart_money_statistics(close, volume, day_offsets) -> dict[str, np.ndarray]:
    close_array = np.asarray(close, dtype=np.float64, order="C")
    volume_array = np.asarray(volume, dtype=np.float64, order="C")
    if close_array.shape != volume_array.shape:
        raise ValueError("close and volume must have the same shape")
    offsets = np.asarray(day_offsets, dtype=np.int64)
    mode = factor_kernel_mode()
    reference = None

    def markov_state(prices):
        returns = np.r_[0.0, prices[1:] / prices[:-1] - 1.0]
        states = (returns > 0).astype(int)
        transitions = np.zeros((2, 2))
        np.add.at(transitions, (states[:-1], states[1:]), 1.0)
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        transitions /= row_sums
        probabilities = np.empty(len(states))
        state_probability = np.array([0.5, 0.5])
        for index, state in enumerate(states):
            if index:
                state_probability = state_probability @ transitions
            probabilities[index] = state_probability[state]
        return returns, probabilities < np.percentile(probabilities, 5)

    if mode != "native":
        reference = np.full((len(offsets) - 1, close_array.shape[1], 2), np.nan)
        for day, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            if end - start < 30:
                continue
            for column in range(close_array.shape[1]):
                prices = close_array[start:end, column]
                quantities = volume_array[start:end, column]
                common = ~(np.isnan(prices) | np.isnan(quantities))
                paired_prices, paired_quantities = prices[common], quantities[common]
                if len(paired_prices) >= 30:
                    _, anomaly = markov_state(paired_prices)
                    rest_volume = paired_quantities[~anomaly].sum()
                    if anomaly.sum() >= 3 and rest_volume >= 1e-12:
                        anomaly_value = (
                            paired_prices[anomaly] * paired_quantities[anomaly]
                        ).sum() / paired_quantities[anomaly].sum()
                        rest_value = (
                            paired_prices[~anomaly] * paired_quantities[~anomaly]
                        ).sum() / rest_volume
                        if rest_value >= 1e-12:
                            reference[day, column, 0] = anomaly_value / rest_value
                prices = prices[~np.isnan(prices)]
                if len(prices) >= 30:
                    returns, anomaly = markov_state(prices)
                    all_std = returns.std(ddof=0)
                    reference[day, column, 1] = (
                        0.0 if all_std < 1e-12 or anomaly.sum() < 3
                        else returns[anomaly].std(ddof=0) / all_std
                    )
    native = native_array_kernel(
        "daily_smart_money_v4", close_array, volume_array, offsets
    ) if mode != "reference" else None
    if mode == "shadow":
        assert_native_equal(reference, native, "daily_smart_money_v4")
    output = native if mode == "native" else reference
    return {"price_ratio": output[:, :, 0], "volatility_ratio": output[:, :, 1]}


_OI_FEATURE_FIELDS = (
    "blowup_position", "torrent", "herding", "peak_count",
    "range_position", "volume_oi_divergence", "volume_oi_price_confirm",
)


def daily_oi_statistics(high, low, close, volume, position, day_offsets):
    arrays = [
        np.asarray(value, dtype=np.float64, order="C")
        for value in (high, low, close, volume, position)
    ]
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("OHLCV and position inputs must have the same shape")
    high, low, close, volume, position = arrays
    offsets = np.asarray(day_offsets, dtype=np.int64)
    mode = factor_kernel_mode()
    reference = None
    if mode != "native":
        reference = np.full((len(offsets) - 1, close.shape[1], 7), np.nan)
        close_returns = close / np.vstack((np.full((1, close.shape[1]), np.nan), close[:-1])) - 1.0
        oi_change = position - np.vstack((
            np.full((1, position.shape[1]), np.nan), position[:-1]
        ))
        for day, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            rows = end - start
            for column in range(close.shape[1]):
                shape_values = np.column_stack((
                    high[start:end, column], low[start:end, column],
                    close[start:end, column], oi_change[start:end, column],
                ))
                shape_values = shape_values[~np.isnan(shape_values).any(axis=1)]
                if rows >= 30 and len(shape_values) >= 30:
                    width = shape_values[:, 0].max() - shape_values[:, 1].min()
                    changes = np.abs(shape_values[:, 3])
                    sigma = changes.std(ddof=0)
                    if width < 1e-12 or sigma < 1e-12:
                        reference[day, column, 0] = 0.5
                    else:
                        selected = changes > changes.mean() + 2.0 * sigma
                        reference[day, column, 0] = (
                            0.5 if selected.sum() < 2 else
                            ((shape_values[selected, 2] - shape_values[:, 1].min()) / width).mean()
                        )
                    if width < 1e-12:
                        reference[day, column, 4] = 0.5
                    else:
                        additions = shape_values[:, 3] > 0
                        weights = shape_values[additions, 3]
                        reference[day, column, 4] = (
                            0.5 if additions.sum() < 5 or weights.sum() <= 1e-12 else
                            np.average(
                                (shape_values[additions, 2] - shape_values[:, 1].min()) / width,
                                weights=weights,
                            )
                        )
                torrent_values = np.column_stack((
                    close_returns[start:end, column], volume[start:end, column],
                    oi_change[start:end, column],
                ))
                torrent_values = torrent_values[~np.isnan(torrent_values).any(axis=1)]
                if rows >= 30 and len(torrent_values) >= 30:
                    volume_mean = torrent_values[:, 1].mean()
                    if volume_mean >= 1e-12:
                        selected = (
                            (torrent_values[:, 0] < 0)
                            & (torrent_values[:, 1] > volume_mean)
                            & (torrent_values[:, 2] > 0)
                        )
                        reference[day, column, 1] = (
                            0.0 if selected.sum() < 2
                            else -torrent_values[selected, 0].mean()
                        )
                herding_values = np.column_stack((
                    close_returns[start:end, column], oi_change[start:end, column]
                ))
                herding_values = herding_values[~np.isnan(herding_values).any(axis=1)]
                if rows >= 20 and len(herding_values) >= 20:
                    valid = (herding_values[:, 0] != 0) & (herding_values[:, 1] != 0)
                    reference[day, column, 2] = (
                        0.0 if valid.sum() < 10 else np.mean(
                            (herding_values[valid, 0] > 0)
                            == (herding_values[valid, 1] > 0)
                        )
                    )
                peaks = np.abs(oi_change[start:end, column])
                peaks = peaks[~np.isnan(peaks)]
                if rows >= 30 and len(peaks) >= 30:
                    sigma = peaks.std(ddof=0)
                    if sigma < 1e-12:
                        reference[day, column, 3] = 0.0
                    else:
                        jumps = peaks > peaks.mean() + sigma
                        reference[day, column, 3] = np.sum(
                            jumps[1:-1] & ~(jumps[:-2] & jumps[2:])
                        )

                pairs = np.column_stack((
                    position[start:end, column], volume[start:end, column]
                ))
                pairs = pairs[~np.isnan(pairs).any(axis=1)]
                if rows >= 20 and len(pairs) >= 20 and len(pairs) - 1 >= 10:
                    differences = np.diff(pairs, axis=0)
                    accumulate = (differences[:, 1] < 0) & (differences[:, 0] > 0)
                    sell_off = (differences[:, 1] > 0) & (differences[:, 0] < 0)
                    reference[day, column, 5] = (
                        accumulate.sum() - sell_off.sum()
                    ) / len(differences)
                triples = np.column_stack((
                    position[start:end, column], volume[start:end, column],
                    close[start:end, column],
                ))
                triples = triples[~np.isnan(triples).any(axis=1)]
                if rows >= 20 and len(triples) >= 20:
                    scores = np.prod(np.sign(np.diff(triples, axis=0)), axis=1)
                    scores = scores[scores != 0]
                    reference[day, column, 6] = (
                        0.0 if len(scores) < 5 else scores.mean()
                    )
    native = native_array_kernel(
        "daily_oi_features", high, low, close, volume, position, offsets
    ) if mode != "reference" else None
    if mode == "shadow":
        assert_native_equal(reference, native, "daily_oi_features")
    output = native if mode == "native" else reference
    return dict(zip(_OI_FEATURE_FIELDS, np.moveaxis(output, 2, 0)))


def daily_return_path_statistics(close, day_offsets) -> dict[str, np.ndarray]:
    close = np.asarray(close, dtype=np.float64, order="C")
    offsets = np.asarray(day_offsets, dtype=np.int64)
    mode = factor_kernel_mode()
    reference = None
    if mode != "native":
        reference = np.full((len(offsets) - 1, close.shape[1], 2), np.nan)
        returns = close / np.vstack((np.full((1, close.shape[1]), np.nan), close[:-1])) - 1.0
        for day, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            if end - start < 30:
                continue
            for column in range(close.shape[1]):
                observed = returns[start:end, column]
                observed = observed[~np.isnan(observed)]
                if len(observed) >= 30:
                    patterns = {}
                    for index in range(len(observed) - 2):
                        pattern = tuple(np.argsort(observed[index:index + 3]))
                        patterns[pattern] = patterns.get(pattern, 0) + 1
                    total = float(sum(patterns.values()))
                    entropy = -sum(
                        (count / total) * np.log(count / total)
                        for count in patterns.values()
                    )
                    reference[day, column, 0] = -entropy / np.log(6.0)
                if end - start >= 60 and len(observed) >= 60:
                    vol5 = [
                        observed[index:index + 5].std(ddof=0)
                        for index in range(0, len(observed) - 4, 5)
                    ]
                    vol30 = [
                        observed[index:index + 30].std(ddof=0)
                        for index in range(0, len(observed) - 29, 30)
                    ]
                    mean5 = np.mean(vol5) if vol5 else 0.0
                    mean30 = np.mean(vol30) if vol30 else 0.0
                    reference[day, column, 1] = (
                        mean5 / mean30 if mean30 > 1e-12 else 1.0
                    )
    native = native_array_kernel(
        "daily_return_path_features", close, offsets
    ) if mode != "reference" else None
    if mode == "shadow":
        assert_native_equal(reference, native, "daily_return_path_features")
    output = native if mode == "native" else reference
    return {"permutation_entropy": output[:, :, 0], "vol_ratio_5_30": output[:, :, 1]}


def daily_volume_shock_statistics(close, volume, amount, day_offsets):
    close, volume, amount = (
        np.asarray(value, dtype=np.float64, order="C")
        for value in (close, volume, amount)
    )
    if close.shape != volume.shape or close.shape != amount.shape:
        raise ValueError("close, volume, and amount must have the same shape")
    offsets = np.asarray(day_offsets, dtype=np.int64)
    mode = factor_kernel_mode()
    reference = None
    if mode != "native":
        reference = np.full((len(offsets) - 1, close.shape[1], 3), np.nan)
        global_returns = close / np.vstack((
            np.full((1, close.shape[1]), np.nan), close[:-1]
        )) - 1.0
        for day, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            if end - start < 30:
                continue
            for column in range(close.shape[1]):
                pairs = np.column_stack((
                    global_returns[start:end, column], volume[start:end, column]
                ))
                pairs = pairs[~np.isnan(pairs).any(axis=1)]
                if len(pairs) >= 30:
                    threshold = pairs[:, 1].mean() + 2.0 * pairs[:, 1].std(ddof=0)
                    spikes = np.flatnonzero(pairs[:, 1] > threshold)
                    impacts = []
                    for index in spikes:
                        forward = min(5, len(pairs) - index - 1)
                        if forward >= 1:
                            impacts.append(np.sign(pairs[index + 1:index + 1 + forward, 0].sum()))
                    reference[day, column, 0] = np.mean(impacts) if impacts else 0.0

                price_volume = np.column_stack((
                    close[start:end, column], volume[start:end, column]
                ))
                price_volume = price_volume[~np.isnan(price_volume).any(axis=1)]
                if len(price_volume) >= 30:
                    threshold = (
                        price_volume[:, 1].mean()
                        + 2.0 * price_volume[:, 1].std(ddof=0)
                    )
                    spikes = np.flatnonzero(price_volume[:, 1] > threshold)
                    returns = np.r_[
                        np.nan, price_volume[1:, 0] / price_volume[:-1, 0] - 1.0
                    ]
                    shock = np.abs(returns[spikes])
                    recoveries = [
                        abs(returns[index + 1:index + 6].sum())
                        for index in spikes if index + 5 < len(returns)
                    ]
                    if recoveries and np.isfinite(shock).any():
                        reference[day, column, 1] = (
                            np.mean(recoveries) / max(np.nanmean(shock), 1e-9)
                        )

                surge = np.column_stack((
                    np.abs(global_returns[start:end, column]),
                    amount[start:end, column],
                ))
                surge = surge[~np.isnan(surge).any(axis=1)]
                if end - start >= 60 and len(surge) >= 60:
                    amihud = surge[:, 0] / (surge[:, 1] + 1e-12)
                    sigma = surge[:, 0].std(ddof=0)
                    if sigma > 0:
                        surge_rows = np.flatnonzero(
                            surge[:, 0] > surge[:, 0].mean() + 2.0 * sigma
                        )
                        before = [
                            amihud[max(0, index - 20):index].mean()
                            for index in surge_rows if index - max(0, index - 20) > 5
                        ]
                        if before:
                            reference[day, column, 2] = np.mean(before)
    native = native_array_kernel(
        "daily_volume_shock_features", close, volume, amount, offsets
    ) if mode != "reference" else None
    if mode == "shadow":
        assert_native_equal(reference, native, "daily_volume_shock_features")
    output = native if mode == "native" else reference
    return {
        "large_order_impact": output[:, :, 0],
        "liquidity_elasticity": output[:, :, 1],
        "vol_surge_liq_before": output[:, :, 2],
    }


def daily_candle_path_statistics(open_, high, low, close, day_offsets):
    arrays = [
        np.asarray(value, dtype=np.float64, order="C")
        for value in (open_, high, low, close)
    ]
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("OHLC inputs must have the same shape")
    open_, high, low, close = arrays
    offsets = np.asarray(day_offsets, dtype=np.int64)
    mode = factor_kernel_mode()
    reference = None
    if mode != "native":
        reference = np.full((len(offsets) - 1, close.shape[1], 2), np.nan)
        for day, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            if end - start < 30:
                continue
            for column in range(close.shape[1]):
                high_low = np.column_stack((
                    high[start:end, column], low[start:end, column]
                ))
                high_low = high_low[~np.isnan(high_low).any(axis=1)]
                if len(high_low) >= 30:
                    up = np.maximum(np.diff(high_low[:, 0]), 0.0)
                    down = np.maximum(-np.diff(high_low[:, 1]), 0.0)
                    plus = np.where((up > down) & (up > 0), up, 0.0).sum()
                    minus = np.where((down > up) & (down > 0), down, 0.0).sum()
                    denominator = plus + minus
                    reference[day, column, 0] = (
                        100.0 * abs(plus - minus) / denominator
                        if denominator > 1e-12 else 0.0
                    )
                open_close = np.column_stack((
                    open_[start:end, column], close[start:end, column]
                ))
                open_close = open_close[~np.isnan(open_close).any(axis=1)]
                if len(open_close) >= 30:
                    count = 0
                    for index in range(1, len(open_close) - 1):
                        previous = open_close[index - 1, 1] - open_close[index - 1, 0]
                        current = abs(open_close[index, 1] - open_close[index, 0])
                        following = open_close[index + 1, 1] - open_close[index + 1, 0]
                        body1 = previous / max(abs(previous), 1e-12)
                        body2 = current / max(current + abs(previous) + 1e-12, 1e-12)
                        body3 = following / max(abs(following), 1e-12)
                        count += int(body1 < -0.5 and body2 < 0.3 and body3 > 0.5)
                    reference[day, column, 1] = count / len(open_close)
    native = native_array_kernel(
        "daily_candle_path_features", open_, high, low, close, offsets
    ) if mode != "reference" else None
    if mode == "shadow":
        assert_native_equal(reference, native, "daily_candle_path_features")
    output = native if mode == "native" else reference
    return {"adx": output[:, :, 0], "morning_star": output[:, :, 1]}


_PRICE_VOLUME_FIELDS = (
    "volume_time_centroid", "close_position", "reversal_intensity",
    "upper_lower_amount_ratio", "ret_vol_coupling", "price_volume_elasticity",
    "up_down_volume_asymmetry", "signed_volume_ratio", "vwap_deviation",
    "depth_proxy", "new_high_volume", "obv_slope",
)


def daily_price_volume_statistics(high, low, close, volume, amount, day_offsets):
    arrays = [
        np.asarray(value, dtype=np.float64, order="C")
        for value in (high, low, close, volume, amount)
    ]
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("price-volume inputs must have the same shape")
    high, low, close, volume, amount = arrays
    offsets = np.asarray(day_offsets, dtype=np.int64)
    mode = factor_kernel_mode()
    reference = None
    if mode != "native":
        reference = np.full((len(offsets) - 1, close.shape[1], 12), np.nan)
        for day, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            if end - start < 20:
                continue
            for column in range(close.shape[1]):
                observed_amount = amount[start:end, column]
                observed_amount = observed_amount[~np.isnan(observed_amount)]
                if observed_amount.size >= 20:
                    total = observed_amount.sum()
                    reference[day, column, 0] = (
                        np.dot(
                            np.arange(1, observed_amount.size + 1) / observed_amount.size,
                            observed_amount,
                        ) / total if total > 0 else 0.5
                    )

                observed_close = close[start:end, column]
                observed_close = observed_close[~np.isnan(observed_close)]
                if observed_close.size >= 20:
                    observed_high = high[start:end, column]
                    observed_high = observed_high[~np.isnan(observed_high)]
                    observed_low = low[start:end, column]
                    observed_low = observed_low[~np.isnan(observed_low)]
                    maximum = observed_high.max() if observed_high.size else np.nan
                    minimum = observed_low.min() if observed_low.size else np.nan
                    width = maximum - minimum
                    reference[day, column, 1] = (
                        (observed_close[-1] - minimum) / width if width > 1e-12 else 0.5
                    )

                previous = close[start - 1, column] if start else np.nan
                returns = []
                return_volume = []
                for row in range(start, end):
                    value = close[row, column]
                    if not np.isnan(value) and not np.isnan(previous):
                        ret = value / previous - 1.0
                        if not np.isnan(ret):
                            returns.append(ret)
                            if not np.isnan(volume[row, column]):
                                return_volume.append((ret, volume[row, column]))
                    previous = value
                returns = np.asarray(returns)
                pairs = np.asarray(return_volume)
                if returns.size >= 20:
                    reference[day, column, 2] = (
                        np.count_nonzero(np.diff(np.sign(returns)) != 0)
                        / (returns.size - 1)
                    )
                if len(pairs) >= 20:
                    ret, vol = pairs[:, 0], pairs[:, 1]
                    vol_mean = vol.mean()
                    if vol_mean >= 1e-12:
                        selected = ret[vol > vol_mean]
                        if selected.size >= 3:
                            reference[day, column, 4] = max(
                                np.count_nonzero(selected > 0),
                                np.count_nonzero(selected < 0),
                            ) / selected.size
                        reference[day, column, 5] = np.mean(
                            np.abs(ret) / (vol / vol_mean + 1e-12)
                        )
                    up, down = vol[ret > 0], vol[ret < 0]
                    up_mean = up.mean() if up.size else 0.0
                    down_mean = down.mean() if down.size else 0.0
                    reference[day, column, 6] = (
                        up_mean / down_mean if down_mean > 1e-12
                        else (2.0 if up_mean > 1e-12 else 1.0)
                    )
                    total = up.sum() + down.sum()
                    reference[day, column, 7] = (
                        (up.sum() - down.sum()) / total if total > 1e-12 else 0.0
                    )
                    if end - start >= 30 and len(pairs) >= 30:
                        obv = np.cumsum(np.sign(ret) * vol)
                        time = np.arange(len(obv)) / max(1, len(obv) - 1)
                        base = vol.sum()
                        reference[day, column, 11] = (
                            np.polyfit(time, obv, 1)[0] / base if base > 1e-12 else 0.0
                        )

                close_amount = np.column_stack((
                    close[start:end, column], amount[start:end, column]
                ))
                close_amount = close_amount[~np.isnan(close_amount).any(axis=1)]
                if len(close_amount) >= 20:
                    middle = (close_amount[:, 0].max() + close_amount[:, 0].min()) / 2.0
                    upper = close_amount[close_amount[:, 0] > middle, 1].sum()
                    lower_amount = close_amount[close_amount[:, 0] < middle, 1].sum()
                    reference[day, column, 3] = (
                        upper / lower_amount if lower_amount > 1e-12 else 1.0
                    )

                close_volume = np.column_stack((
                    close[start:end, column], volume[start:end, column]
                ))
                close_volume = close_volume[~np.isnan(close_volume).any(axis=1)]
                if len(close_volume) >= 20:
                    volume_sum = close_volume[:, 1].sum()
                    if volume_sum >= 1e-12:
                        vwap = np.dot(close_volume[:, 0], close_volume[:, 1]) / volume_sum
                        reference[day, column, 8] = np.std(close_volume[:, 0] - vwap)

                depth = np.column_stack((
                    amount[start:end, column],
                    high[start:end, column] - low[start:end, column],
                ))
                depth = depth[~np.isnan(depth).any(axis=1)]
                if len(depth) >= 20:
                    nonzero = depth[depth[:, 1] != 0, 1]
                    if len(nonzero) >= 10:
                        reference[day, column, 9] = (
                            depth[:, 0].mean() / nonzero.mean()
                            if nonzero.mean() > 1e-12 else 0.0
                        )

                high_volume = np.column_stack((
                    high[start:end, column], volume[start:end, column]
                ))
                high_volume = high_volume[~np.isnan(high_volume).any(axis=1)]
                if len(high_volume) >= 20:
                    volume_mean = high_volume[:, 1].mean()
                    if volume_mean >= 1e-12:
                        new_high = np.r_[True, high_volume[1:, 0] > np.maximum.accumulate(high_volume[:-1, 0])]
                        reference[day, column, 10] = (
                            high_volume[new_high, 1].mean() / volume_mean
                            if new_high.sum() >= 2 else 1.0
                        )
    native = native_array_kernel(
        "daily_price_volume_features", high, low, close, volume, amount, offsets
    ) if mode != "reference" else None
    if mode == "shadow":
        assert_native_equal(reference, native, "daily_price_volume_features")
    output = native if mode == "native" else reference
    return dict(zip(_PRICE_VOLUME_FIELDS, np.moveaxis(output, 2, 0)))


_PRICE_PATH_FIELDS = (
    "parkinson_vol_ratio", "choppiness", "range_crossing", "mid_line_time",
    "vwap_band_retention", "path_bandwidth", "edge_touch_ratio",
    "midline_direction", "open_drive", "vol_extreme_magnitude",
    "vol_ratio_trend", "risk_adj_momentum",
)


def _segmented_ohlc_volatility(
    close, high, low, window, *, positive_only=False
):
    values = []
    for start in range(0, len(close) - window + 1, window):
        segment = np.concatenate((
            close[start:start + window], high[start:start + window],
            low[start:start + window],
        ))
        mean = segment.mean()
        if mean > 0:
            value = segment.std(ddof=0) / mean
            if not positive_only or value > 0:
                values.append(value)
    return np.asarray(values)


def daily_price_path_statistics(open_, high, low, close, volume, day_offsets):
    arrays = [
        np.asarray(value, dtype=np.float64, order="C")
        for value in (open_, high, low, close, volume)
    ]
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("price-path inputs must have the same shape")
    open_, high, low, close, volume = arrays
    offsets = np.asarray(day_offsets, dtype=np.int64)
    mode = factor_kernel_mode()
    reference = None
    if mode != "native":
        reference = np.full((len(offsets) - 1, close.shape[1], 12), np.nan)
        for day, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            if end - start < 20:
                continue
            open_window = max(10, min(30, (end - start) // 4))
            for column in range(close.shape[1]):
                hlc = np.column_stack((
                    high[start:end, column], low[start:end, column],
                    close[start:end, column],
                ))
                hlc = hlc[~np.isnan(hlc).any(axis=1)]
                closes = close[start:end, column]
                closes = closes[~np.isnan(closes)]
                close_returns = np.diff(closes) / closes[:-1] if len(closes) > 1 else np.array([])
                log_ranges = np.log(
                    high[start:end, column] / low[start:end, column]
                )
                log_ranges = log_ranges[np.isfinite(log_ranges)]
                high_low_count = np.count_nonzero(~(
                    np.isnan(high[start:end, column]) | np.isnan(low[start:end, column])
                ))
                if high_low_count >= 20 and len(log_ranges) >= 10:
                    parkinson = np.sqrt(
                        np.square(log_ranges).sum()
                        / (4.0 * len(log_ranges) * np.log(2.0))
                    )
                    volatility = np.std(close_returns) if len(close_returns) else np.nan
                    reference[day, column, 0] = (
                        0.0 if volatility < 1e-12 else -parkinson / volatility
                    )

                previous = close[start - 1, column] if start else np.nan
                hlc_return, open_close_return = [], []
                for row in range(start, end):
                    value = close[row, column]
                    ret = value / previous - 1.0 if not (
                        np.isnan(value) or np.isnan(previous)
                    ) else np.nan
                    if not np.isnan(high[row, column]) and not np.isnan(low[row, column]) and not np.isnan(ret):
                        hlc_return.append((high[row, column], low[row, column], ret))
                    if not np.isnan(open_[row, column]) and not np.isnan(value) and not np.isnan(ret):
                        open_close_return.append((open_[row, column], value, ret))
                    previous = value
                hlc_return = np.asarray(hlc_return)
                open_close_return = np.asarray(open_close_return)
                if len(hlc_return) >= 20:
                    width = hlc_return[:, 0].max() - hlc_return[:, 1].min()
                    absolute_sum = np.abs(hlc_return[:, 2]).sum()
                    reference[day, column, 1] = (
                        0.0 if width < 1e-12 or absolute_sum <= 1e-12
                        else -np.log10(absolute_sum / width) / np.log10(len(hlc_return))
                    )
                if len(open_close_return) >= 20:
                    first_open = open_close_return[0, 0]
                    volatility = np.std(open_close_return[:, 2])
                    reference[day, column, 11] = (
                        np.nan if first_open < 1e-12 else (
                            (open_close_return[-1, 1] / first_open - 1.0) / volatility
                            if volatility > 1e-12 else 0.0
                        )
                    )

                if len(hlc) >= 20:
                    maximum, minimum = hlc[:, 0].max(), hlc[:, 1].min()
                    middle = (maximum + minimum) / 2.0
                    above = hlc[:, 2] > middle
                    difference = np.diff(above.astype(int))
                    reference[day, column, 3] = above.mean()
                    if end - start >= 30 and len(hlc) >= 30:
                        reference[day, column, 2] = -np.count_nonzero(difference)
                        up, down = np.count_nonzero(difference == 1), np.count_nonzero(difference == -1)
                        reference[day, column, 7] = up / (up + down) if up + down else 0.5
                        width = maximum - minimum
                        if width < 1e-12:
                            reference[day, column, 5:7] = 0.0
                        else:
                            p10, p90 = np.percentile(hlc[:, 2], (10, 90))
                            reference[day, column, 5] = (p90 - p10) / width
                            position = (hlc[:, 2] - minimum) / width
                            reference[day, column, 6] = -np.mean(
                                (position <= 0.1) | (position >= 0.9)
                            )

                close_volume = np.column_stack((
                    close[start:end, column], volume[start:end, column]
                ))
                close_volume = close_volume[~np.isnan(close_volume).any(axis=1)]
                if len(close_volume) >= 30:
                    sigma = close_volume[:, 0].std(ddof=0)
                    if sigma < 1e-12:
                        reference[day, column, 4] = 0.0
                    else:
                        total = close_volume[:, 1].sum()
                        anchor = (
                            np.dot(close_volume[:, 0], close_volume[:, 1]) / total
                            if total > 1e-12 else close_volume[:, 0].mean()
                        )
                        reference[day, column, 4] = np.mean(
                            np.abs(close_volume[:, 0] - anchor) <= sigma
                        )

                ohlc = np.column_stack((
                    open_[start:end, column], high[start:end, column],
                    low[start:end, column], close[start:end, column],
                ))
                ohlc = ohlc[~np.isnan(ohlc).any(axis=1)]
                if end - start >= 30 and len(ohlc) >= 30:
                    width = ohlc[:, 1].max() - ohlc[:, 2].min()
                    if ohlc[0, 0] >= 1e-12 and width >= 1e-12:
                        reference[day, column, 8] = (
                            ohlc[open_window - 1, 3] / ohlc[0, 0] - 1.0
                        ) / width

                if len(hlc) >= 40:
                    vol5_all = _segmented_ohlc_volatility(
                        hlc[:, 2], hlc[:, 0], hlc[:, 1], 5
                    )
                    vol5_positive = _segmented_ohlc_volatility(
                        hlc[:, 2], hlc[:, 0], hlc[:, 1], 5,
                        positive_only=True,
                    )
                    vol30 = _segmented_ohlc_volatility(
                        hlc[:, 2], hlc[:, 0], hlc[:, 1], 30
                    )
                    if len(vol5_positive) < 5 or len(vol30) < 3:
                        reference[day, column, 9] = 0.0
                    else:
                        threshold = np.percentile(vol5_positive, 95)
                        extreme = vol30[vol30 > threshold]
                        reference[day, column, 9] = (
                            1.0 if abs(threshold) < 1e-12 else (
                                extreme.mean() / threshold if len(extreme) else 0.0
                            )
                        )
                    reference[day, column, 10] = (
                        0.0 if len(vol5_all) < 5 or len(vol30) < 3
                        else np.mean(vol30 > np.percentile(vol5_all, 95))
                    )
    native = native_array_kernel(
        "daily_price_path_features", open_, high, low, close, volume, offsets
    ) if mode != "reference" else None
    if mode == "shadow":
        assert_native_equal(reference, native, "daily_price_path_features")
    output = native if mode == "native" else reference
    return dict(zip(_PRICE_PATH_FIELDS, np.moveaxis(output, 2, 0)))


def pd_series_skew(observed, mean=None, std=None) -> float:
    """Pandas-compatible Fisher unbiased skew for one finite NumPy vector."""

    array = np.asarray(observed, dtype=float)
    mean = array.mean() if mean is None else mean
    std = array.std(ddof=0) if std is None else std
    if array.size < 3:
        return np.nan
    if std == 0:
        return 0.0
    biased = np.mean((array - mean) ** 3) / std ** 3
    return float(np.sqrt(array.size * (array.size - 1)) / (array.size - 2) * biased)


def _reference_daily_return_stats(values, day_offsets) -> np.ndarray:
    """Reference for adjacent-row minute returns grouped by trading day."""

    array = np.asarray(values, dtype=np.float64, order="C")
    offsets = np.asarray(day_offsets, dtype=np.int64)
    output = np.full((len(offsets) - 1, array.shape[1], 12), np.nan)
    returns = array / np.vstack((np.full((1, array.shape[1]), np.nan), array[:-1])) - 1.0
    for day, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        group = returns[start:end]
        for column in range(array.shape[1]):
            observed = group[:, column]
            observed = observed[~np.isnan(observed)]
            if observed.size == 0:
                continue
            mean = observed.mean()
            std = observed.std(ddof=0)
            centered = observed - mean
            output[day, column, :10] = (
                observed.size, observed.sum(), mean, std,
                np.abs(observed).mean(), observed.max(),
                np.count_nonzero(observed > 0), np.count_nonzero(observed < 0),
                np.mean(observed ** 2), np.mean(observed ** 4),
            )
            if observed.size >= 3:
                if std == 0:
                    output[day, column, 10:] = 0.0
                elif np.isfinite(std):
                    skew = np.mean(centered ** 3) / std ** 3
                    output[day, column, 10] = (
                        np.sqrt(observed.size * (observed.size - 1))
                        / (observed.size - 2) * skew
                    )
                    output[day, column, 11] = np.mean(centered ** 4) / std ** 4 - 3.0
    return output


def daily_return_statistics(values, day_offsets) -> dict[str, np.ndarray]:
    """Daily minute-return primitives with opt-in native/shadow execution."""

    mode = factor_kernel_mode()
    array = np.asarray(values, dtype=np.float64, order="C")
    offsets = np.asarray(day_offsets, dtype=np.int64)
    reference = None
    if mode in {"reference", "shadow"}:
        reference = _reference_daily_return_stats(array, offsets)
    if mode in {"shadow", "native"}:
        native = native_array_kernel("daily_return_stats", array, offsets)
        if mode == "shadow":
            assert_native_equal(reference, native, "daily_return_stats")
        values_out = native if mode == "native" else reference
    else:
        values_out = reference
    return dict(zip(_RETURN_STAT_FIELDS, np.moveaxis(values_out, 2, 0)))


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
