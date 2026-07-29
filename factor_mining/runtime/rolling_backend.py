from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

try:  # Optional acceleration only; never a framework dependency.
    import bottleneck as _bottleneck
except ImportError:  # pragma: no cover - exercised in dependency-minimal installs
    _bottleneck = None


FAST_METHODS = frozenset({"mean", "min", "max"})


def minimum_periods(window: int) -> int:
    return max(2, min(int(window), max(3, int(window) // 2)))


def pandas_rolling(
    value: np.ndarray, window: int, method: str
) -> np.ndarray:
    frame = pd.DataFrame(np.asarray(value, dtype=float))
    rolling = frame.rolling(
        int(window), min_periods=minimum_periods(window)
    )
    if method == "sum":
        return rolling.sum().to_numpy()
    if method == "mean":
        return rolling.mean().to_numpy()
    if method == "std":
        return rolling.std().to_numpy()
    if method == "min":
        return rolling.min().to_numpy()
    if method == "max":
        return rolling.max().to_numpy()
    if method == "median":
        return rolling.median().to_numpy()
    if method == "skew":
        return rolling.skew().to_numpy()
    if method == "kurt":
        return rolling.kurt().to_numpy()
    raise ValueError(method)


def _fast_rolling(
    value: np.ndarray, window: int, method: str
) -> np.ndarray:
    if _bottleneck is None:
        raise RuntimeError("bottleneck is unavailable")
    source = np.asarray(value, dtype=float)
    kwargs = {
        "window": int(window),
        "min_count": minimum_periods(window),
        "axis": 0,
    }
    if method == "mean":
        return _bottleneck.move_mean(source, **kwargs)
    if method == "std":
        return _bottleneck.move_std(source, ddof=1, **kwargs)
    if method == "min":
        return _bottleneck.move_min(source, **kwargs)
    if method == "max":
        return _bottleneck.move_max(source, **kwargs)
    raise ValueError(method)


@lru_cache(maxsize=128)
def _backend_is_compatible(method: str, window: int) -> bool:
    # Bottleneck move_std can be locally close to Pandas yet exceed the factor
    # tolerance after division by a near-zero rolling standard deviation in
    # ts_zscore.  Keep std on the exact legacy path until a guarded
    # implementation can prove end-to-end equivalence.
    if method == "std":
        return False
    if _bottleneck is None or method not in FAST_METHODS:
        return False
    rows = max(32, int(window) * 3)
    rng = np.random.default_rng(731 + int(window))
    probe = rng.normal(size=(rows, 7))
    probe[::7, 1] = np.nan
    probe[3::11, 4] = np.nan
    probe[:, 6] = 2.0
    expected = pandas_rolling(probe, window, method)
    try:
        actual = _fast_rolling(probe, window, method)
    except (MemoryError, RuntimeError, TypeError, ValueError):
        return False
    return bool(
        np.array_equal(np.isnan(actual), np.isnan(expected))
        and np.allclose(
            actual, expected, rtol=1e-6, atol=1e-7, equal_nan=True
        )
    )


def rolling(
    value: np.ndarray,
    window: int,
    method: str,
    *,
    backend: str = "pandas",
) -> np.ndarray:
    if backend not in {"pandas", "fast"}:
        raise ValueError(f"unknown rolling backend: {backend}")
    if (
        backend == "fast"
        and method in FAST_METHODS
        and _backend_is_compatible(method, int(window))
    ):
        try:
            return _fast_rolling(value, window, method)
        except (MemoryError, RuntimeError, TypeError, ValueError):
            pass
    return pandas_rolling(value, window, method)


def bottleneck_available() -> bool:
    return _bottleneck is not None
