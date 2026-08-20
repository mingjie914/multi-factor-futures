"""Causal factor-combination helpers shared by production and research."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def causal_history(
    frame: pd.DataFrame,
    date: pd.Timestamp,
    window: int,
) -> pd.DataFrame:
    """Return up to the latest ``window`` rows strictly before ``date``."""
    if int(window) <= 0:
        raise ValueError("history window must be positive")
    return frame.loc[frame.index < pd.Timestamp(date)].tail(int(window))


def prepare_complete_history(
    history: pd.DataFrame,
    minimum_observations: int = 30,
) -> pd.DataFrame:
    """Admit columns with enough finite observations, then align their rows."""
    clean = history.replace([np.inf, -np.inf], np.nan)
    required = int(minimum_observations)
    if required <= 0:
        raise ValueError("minimum_observations must be positive")
    if len(clean) < required:
        return clean.iloc[0:0, 0:0]
    columns = clean.columns[clean.notna().sum(axis=0).ge(required)]
    aligned = clean.loc[:, columns].dropna(axis=0, how="any")
    if len(aligned) < required:
        return aligned.iloc[0:0, 0:0]
    return aligned


def rank_information_coefficients(
    ranks: Mapping[str, pd.DataFrame],
    next_bar_returns: pd.DataFrame,
    *,
    minimum_cross_section: int = 3,
) -> pd.DataFrame:
    """Compute rank IC for T decisions against T+1 returns."""
    minimum = int(minimum_cross_section)
    if minimum < 2:
        raise ValueError("minimum_cross_section must be at least 2")
    forward_rank = next_bar_returns.shift(-1).rank(axis=1)
    columns = {}
    for name, rank in ranks.items():
        aligned = rank.reindex(
            index=forward_rank.index, columns=forward_rank.columns
        )
        common = (aligned.notna() & forward_rank.notna()).sum(axis=1)
        columns[str(name)] = aligned.corrwith(forward_rank, axis=1).where(
            common >= minimum
        )
    return pd.DataFrame(columns, index=forward_rank.index)


def ledoit_wolf_covariance(ic_matrix: pd.DataFrame) -> np.ndarray:
    """Return the project's constant-correlation shrinkage covariance."""
    values = ic_matrix.to_numpy(dtype=float)
    rows, columns = values.shape
    if rows < 2 or columns < 2:
        raise ValueError("Ledoit-Wolf covariance requires at least 2x2 observations")
    sample_cov = np.cov(values, rowvar=False, ddof=1)
    sample_corr = np.corrcoef(values, rowvar=False)
    if not np.isfinite(sample_cov).all():
        raise ValueError("IC history produced a non-finite sample covariance")
    upper = sample_corr[np.triu_indices(columns, k=1)]
    average_corr = float(np.nanmean(upper)) if len(upper) else 0.0
    if not np.isfinite(average_corr):
        raise ValueError("IC history produced a non-finite average correlation")
    target_corr = (
        np.eye(columns) * (1.0 - average_corr)
        + np.ones((columns, columns)) * average_corr
    )
    standard_deviation = np.std(values, axis=0, ddof=1)
    if not np.isfinite(standard_deviation).all() or np.any(standard_deviation <= 0.0):
        raise ValueError("IC history contains a constant or invalid factor")
    target_cov = np.outer(standard_deviation, standard_deviation) * target_corr
    centered = values - np.mean(values, axis=0)
    pi = sum(
        np.sum((np.outer(row, row) - sample_cov) ** 2)
        for row in centered
    ) / rows
    gamma = float(np.sum((target_cov - sample_cov) ** 2))
    shrinkage = float(np.clip(pi / gamma, 0.0, 1.0)) if gamma > 0.0 else 0.5
    covariance = shrinkage * target_cov + (1.0 - shrinkage) * sample_cov
    if not np.isfinite(covariance).all():
        raise ValueError("Ledoit-Wolf covariance contains NaN/Inf")
    return covariance


def _normalise_positive(raw: np.ndarray, columns: Sequence[str]) -> pd.Series:
    values = np.asarray(raw, dtype=float)
    values = np.where(np.isfinite(values) & (values > 0.0), values, 0.0)
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("factor-weight estimate has no positive finite mass")
    return pd.Series(values / total, index=columns, dtype=float)


def _cap_simplex(weights: pd.Series, cap: float) -> pd.Series:
    """Project positive normalized factor weights onto a capped simplex."""
    if weights.empty:
        return weights
    effective_cap = float(cap)
    if effective_cap <= 0.0 or effective_cap * len(weights) < 1.0 - 1e-12:
        raise ValueError("factor-weight cap is infeasible")
    result = weights.clip(lower=0.0).astype(float)
    if result.sum() <= 0.0:
        raise ValueError("factor weights contain no positive mass")
    result /= result.sum()
    for _ in range(len(result) + 2):
        high = result > effective_cap + 1e-12
        if not high.any():
            break
        result.loc[high] = effective_cap
        low = ~high
        residual = 1.0 - float(result.loc[high].sum())
        if not low.any() or residual <= 0.0:
            break
        base = result.loc[low]
        result.loc[low] = (
            residual / int(low.sum())
            if base.sum() <= 0.0
            else base / base.sum() * residual
        )
    return result / result.sum()


def factor_weights(history: pd.DataFrame, method: str) -> pd.Series:
    """Estimate causal, non-short factor weights from complete IC history."""
    supported = {"equal", "diag_icir", "lw_abs", "lw_positive"}
    if method not in supported:
        raise ValueError(f"unknown factor-weight method: {method}")
    clean = history.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if clean.shape[1] < 2 or len(clean) == 0:
        return pd.Series(dtype=float)
    columns = list(clean.columns)
    if method == "equal":
        return pd.Series(1.0 / len(columns), index=columns, dtype=float)
    if len(clean) < 30:
        return pd.Series(dtype=float)

    mean_ic = clean.mean().to_numpy(dtype=float)
    if method == "diag_icir":
        volatility = clean.std(ddof=1).replace(0.0, np.nan).to_numpy(dtype=float)
        return _normalise_positive(mean_ic / volatility, columns)

    covariance = ledoit_wolf_covariance(clean)
    if method == "lw_abs":
        try:
            raw = np.linalg.solve(covariance, mean_ic)
        except np.linalg.LinAlgError as exc:
            raise ValueError("lw_abs factor-weight solve failed") from exc
        return _normalise_positive(np.abs(raw), columns)
    if method == "lw_positive":
        ridge = max(float(np.trace(covariance)) / len(columns), 1e-12) * 1e-6
        try:
            raw = np.linalg.solve(
                covariance + ridge * np.eye(len(columns)), mean_ic
            )
        except np.linalg.LinAlgError as exc:
            raise ValueError("lw_positive factor-weight solve failed") from exc
        return _cap_simplex(
            _normalise_positive(np.maximum(raw, 0.0), columns), 0.35
        )
    raise AssertionError("unreachable factor-weight method")


def combine_available_factor_scores(
    ranks: Mapping[str, pd.Series],
    weights: pd.Series,
    universe: Sequence[str],
) -> pd.Series:
    """Combine one cross-section and renormalize over factors available per asset."""
    if weights.empty:
        return pd.Series(np.nan, index=list(universe), dtype=float)
    values = weights.astype(float)
    if (
        not np.isfinite(values.to_numpy()).all()
        or (values < 0.0).any()
        or float(values.sum()) <= 0.0
    ):
        raise ValueError("factor weights must contain non-negative finite mass")
    missing = [str(name) for name in values.index if str(name) not in ranks]
    if missing:
        raise KeyError(f"factor ranks are missing: {missing}")

    index = list(universe)
    numerator = pd.Series(0.0, index=index, dtype=float)
    denominator = pd.Series(0.0, index=index, dtype=float)
    for name, weight in values.items():
        factor_rank = ranks[str(name)].reindex(index).astype(float)
        factor_rank = factor_rank.replace([np.inf, -np.inf], np.nan)
        available = factor_rank.notna()
        numerator = numerator.add(factor_rank.fillna(0.0) * weight, fill_value=0.0)
        denominator = denominator.add(available.astype(float) * weight, fill_value=0.0)
    return numerator.div(denominator.where(denominator > 0.0))


__all__ = [
    "causal_history",
    "combine_available_factor_scores",
    "factor_weights",
    "ledoit_wolf_covariance",
    "prepare_complete_history",
    "rank_information_coefficients",
]
