from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm


def benjamini_hochberg(
    p_values: Iterable[float], alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """Return BH-FDR adjusted q-values and rejection flags."""
    values = np.asarray(list(p_values), dtype=float)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if values.size == 0:
        return values.copy(), np.zeros(0, dtype=bool)
    finite = np.isfinite(values)
    clipped = np.ones_like(values)
    clipped[finite] = np.clip(values[finite], 0.0, 1.0)
    order = np.argsort(clipped, kind="stable")
    ranked = clipped[order]
    scale = len(values) / np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate((ranked * scale)[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    adjusted[~finite] = 1.0
    return adjusted, adjusted <= alpha


def simes_p_value(p_values: Iterable[float]) -> float:
    """Combine positively dependent local tests into one global-null p-value."""
    values = np.asarray(list(p_values), dtype=float)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if values.size == 0:
        return 1.0
    values = np.where(np.isfinite(values), np.clip(values, 0.0, 1.0), 1.0)
    ranked = np.sort(values, kind="stable")
    adjusted = ranked * values.size / np.arange(1, values.size + 1)
    return float(np.clip(np.min(adjusted), 0.0, 1.0))


def deflated_sharpe_ratio(
    returns: pd.Series,
    *,
    n_trials: int,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> dict:
    """Estimate the probability that excess-return Sharpe beats selection noise."""
    series = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    series = series - float(risk_free_rate) / periods_per_year
    n = len(series)
    if n < 3 or n_trials < 1 or float(series.std(ddof=1)) <= 0:
        return {
            "sharpe": 0.0,
            "expected_max_sharpe": 0.0,
            "probability": 0.0,
            "n_obs": n,
            "n_trials": int(max(n_trials, 0)),
            "risk_free_rate": float(risk_free_rate),
        }
    daily_sharpe = float(series.mean() / series.std(ddof=1))
    annual_sharpe = daily_sharpe * np.sqrt(periods_per_year)
    skewness = float(series.skew())
    kurtosis = float(series.kurt() + 3.0)
    sharpe_std = np.sqrt(
        max(
            1.0 - skewness * daily_sharpe
            + ((kurtosis - 1.0) / 4.0) * daily_sharpe**2,
            1e-12,
        )
        / (n - 1)
    )
    if n_trials == 1:
        expected_max_daily = 0.0
    else:
        euler_gamma = 0.5772156649015329
        z1 = norm.ppf(1.0 - 1.0 / n_trials)
        z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        expected_max_daily = sharpe_std * ((1.0 - euler_gamma) * z1 + euler_gamma * z2)
    probability = float(norm.cdf((daily_sharpe - expected_max_daily) / sharpe_std))
    return {
        "sharpe": annual_sharpe,
        "expected_max_sharpe": float(expected_max_daily * np.sqrt(periods_per_year)),
        "probability": probability,
        "n_obs": n,
        "n_trials": int(n_trials),
        "risk_free_rate": float(risk_free_rate),
    }


def probability_backtest_overfitting(
    strategy_returns: pd.DataFrame,
    *,
    n_partitions: int = 8,
    max_combinations: int = 5000,
) -> dict:
    """Estimate PBO with combinatorially symmetric cross-validation (CSCV)."""
    frame = strategy_returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    frame = frame.dropna(axis=1, how="all").fillna(0.0)
    if len(frame) < n_partitions * 2 or frame.shape[1] < 2:
        return {"pbo": float("nan"), "n_splits": 0, "logits": []}
    n_partitions = int(n_partitions)
    if n_partitions < 4 or n_partitions % 2:
        raise ValueError("n_partitions must be an even integer >= 4")
    blocks = [block for block in np.array_split(np.arange(len(frame)), n_partitions) if len(block)]
    half = n_partitions // 2
    total = comb(n_partitions, half)
    candidates = combinations(range(n_partitions), half)
    if total > max_combinations:
        stride = max(total // max_combinations, 1)
        candidates = (item for index, item in enumerate(candidates) if index % stride == 0)

    values = frame.to_numpy(dtype=float)
    logits = []
    for selected in candidates:
        train_blocks = set(selected)
        train_idx = np.concatenate([blocks[i] for i in train_blocks])
        test_idx = np.concatenate([blocks[i] for i in range(n_partitions) if i not in train_blocks])
        train = values[train_idx]
        test = values[test_idx]
        train_std = train.std(axis=0, ddof=1)
        train_score = np.divide(
            train.mean(axis=0), train_std, out=np.full(frame.shape[1], -np.inf), where=train_std > 0
        )
        selected_strategy = int(np.argmax(train_score))
        test_std = test.std(axis=0, ddof=1)
        test_score = np.divide(
            test.mean(axis=0), test_std, out=np.full(frame.shape[1], -np.inf), where=test_std > 0
        )
        selected_score = test_score[selected_strategy]
        rank = int(np.sum(test_score <= selected_score))
        relative_rank = np.clip(rank / (frame.shape[1] + 1.0), 1e-12, 1.0 - 1e-12)
        logits.append(float(np.log(relative_rank / (1.0 - relative_rank))))
        if len(logits) >= max_combinations:
            break
    if not logits:
        return {"pbo": float("nan"), "n_splits": 0, "logits": []}
    return {
        "pbo": float(np.mean(np.asarray(logits) <= 0.0)),
        "n_splits": len(logits),
        "logits": logits,
    }
