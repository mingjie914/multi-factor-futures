from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from .strategy import GuosenTrendIndexSpec


def _sharpe(returns: np.ndarray, periods_per_year: int) -> float:
    values = returns[np.isfinite(returns)]
    if values.size < 2:
        return 0.0
    volatility = float(values.std(ddof=1) * np.sqrt(periods_per_year))
    return float(values.mean() * periods_per_year / volatility) if volatility > 0 else 0.0


def _max_drawdown(returns: np.ndarray) -> float:
    values = np.where(np.isfinite(returns), returns, 0.0)
    nav = np.cumprod(1.0 + values)
    peaks = np.maximum.accumulate(nav)
    return float(np.min(nav / peaks - 1.0)) if len(nav) else 0.0


def _project_to_gross(
    weights: np.ndarray,
    caps: np.ndarray,
    target_gross: float,
) -> np.ndarray:
    """Vectorized capped proportional projection for all dates."""
    result = np.zeros_like(weights)
    active = weights > 0.0
    has_positions = active.any(axis=1)
    capacity = (active * caps[None, :]).sum(axis=1)
    infeasible = has_positions & (capacity < target_gross - 1e-12)
    if infeasible.any():
        row = int(np.flatnonzero(infeasible)[0])
        raise ValueError(f"gross projection infeasible at row {row}")
    remaining = np.where(has_positions, target_gross, 0.0)
    unfrozen = active.copy()
    for _ in range(weights.shape[1]):
        denominator = (weights * unfrozen).sum(axis=1)
        scale = np.divide(
            remaining,
            denominator,
            out=np.zeros_like(remaining),
            where=denominator > 0.0,
        )
        proposed = weights * scale[:, None]
        over_cap = unfrozen & (proposed > caps[None, :] + 1e-12)
        can_finish = unfrozen.any(axis=1) & ~over_cap.any(axis=1)
        if can_finish.any():
            result[can_finish] += proposed[can_finish] * unfrozen[can_finish]
            remaining[can_finish] = 0.0
            unfrozen[can_finish] = False
        if over_cap.any():
            result[over_cap] = np.broadcast_to(caps, weights.shape)[over_cap]
            remaining -= (over_cap * caps[None, :]).sum(axis=1)
            unfrozen &= ~over_cap
        if not unfrozen.any():
            break
    if np.max(np.abs(remaining)) > 1e-10:
        raise ValueError("gross projection did not converge")
    return result


def exhaustive_subset_search(
    portfolios: Mapping[str, pd.DataFrame],
    close: pd.DataFrame,
    spec: GuosenTrendIndexSpec,
    start: pd.Timestamp,
    end: pd.Timestamp,
    baseline_sets: Mapping[str, set[str]],
    output: str | Path,
    target_gross: float = 1.0,
) -> pd.DataFrame:
    """Evaluate all subsets without recomputing factors or risk portfolios.

    Candidate ranking uses only three development segments ending in 2024.
    The 2025+ segment is reported as a holdout diagnostic and is not part of
    ``development_score``.
    """
    if spec.execution_lag_days != 0:
        raise ValueError("subset cache currently requires execution_lag_days=0")
    names = list(portfolios)
    if not names:
        raise ValueError("no factor portfolios supplied")
    dates = pd.DatetimeIndex(close.index)
    evaluation = dates[(dates >= start) & (dates <= end)]
    components = np.stack(
        [
            portfolios[name].reindex(index=evaluation, columns=spec.universe)
            .fillna(0.0).to_numpy(dtype=float)
            for name in names
        ],
        axis=0,
    )
    asset_returns = close.reindex(columns=spec.universe).pct_change(
        fill_method=None
    ).reindex(evaluation).fillna(0.0).to_numpy(dtype=float)
    caps = pd.Series(spec.asset_caps).reindex(spec.universe).to_numpy(dtype=float)
    fee = spec.annual_management_fee / spec.periods_per_year
    baseline_lookup = {
        frozenset(factors): label for label, factors in baseline_sets.items()
    }
    segments = {
        "dev_2016_2019": (pd.Timestamp("2016-03-31"), pd.Timestamp("2019-12-31")),
        "dev_2020_2022": (pd.Timestamp("2020-01-01"), pd.Timestamp("2022-12-31")),
        "dev_2023_2024": (pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")),
        "holdout_2025_2026": (pd.Timestamp("2025-01-01"), end),
    }
    segment_masks = {
        label: (evaluation >= left) & (evaluation <= right)
        for label, (left, right) in segments.items()
    }
    pre_holdout = evaluation < pd.Timestamp("2025-01-01")
    records = []
    total = (1 << len(names)) - 1
    for mask in range(1, total + 1):
        indices = [index for index in range(len(names)) if mask & (1 << index)]
        selected_names = [names[index] for index in indices]
        selected = components[indices]
        active = selected.sum(axis=2) > 0.0
        denominator = active.sum(axis=0)
        weights = np.divide(
            selected.sum(axis=0),
            denominator[:, None],
            out=np.zeros_like(selected[0]),
            where=denominator[:, None] > 0,
        )
        weights = np.minimum(np.maximum(weights, 0.0), caps[None, :])
        weights = _project_to_gross(weights, caps, target_gross)
        gross_returns = (weights * asset_returns).sum(axis=1)
        turnover = np.abs(np.diff(weights, axis=0, prepend=weights[[0]])).sum(axis=1)
        net_returns = gross_returns - turnover * spec.transaction_cost_rate - fee
        if len(net_returns):
            net_returns[0] = 0.0
            turnover[0] = 0.0
        segment_sharpes = {
            label: _sharpe(net_returns[segment_mask], spec.periods_per_year)
            for label, segment_mask in segment_masks.items()
        }
        development = np.array(
            [segment_sharpes[label] for label in list(segments)[:3]], dtype=float
        )
        pre_returns = net_returns[pre_holdout]
        full_sharpe = _sharpe(net_returns, spec.periods_per_year)
        pre_holdout_sharpe = _sharpe(pre_returns, spec.periods_per_year)
        development_score = (
            0.30 * pre_holdout_sharpe
            + 0.35 * float(np.median(development))
            + 0.35 * float(np.min(development))
            - 0.01 * len(indices)
        )
        factor_key = frozenset(selected_names)
        records.append({
            "mask": mask,
            "factor_count": len(indices),
            "factors": "|".join(selected_names),
            "baseline_label": baseline_lookup.get(factor_key, ""),
            "development_score": development_score,
            "pre_2025_sharpe": pre_holdout_sharpe,
            "full_sharpe": full_sharpe,
            "full_max_drawdown": _max_drawdown(net_returns),
            "annual_turnover": float(turnover.mean() * spec.periods_per_year),
            "positive_development_segments": int((development > 0.0).sum()),
            "worst_development_sharpe": float(np.min(development)),
            "median_development_sharpe": float(np.median(development)),
            **segment_sharpes,
        })
        if mask % 2048 == 0 or mask == total:
            print(f"subset search: {mask}/{total}", flush=True)
    result = pd.DataFrame.from_records(records).sort_values(
        ["development_score", "worst_development_sharpe", "factor_count"],
        ascending=[False, False, True],
    )
    result.insert(0, "development_rank", np.arange(1, len(result) + 1))
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path / "factor_subset_search_all.csv", index=False)
    result.head(50).to_csv(output_path / "factor_subset_search_top50.csv", index=False)
    result.loc[result["baseline_label"].ne("")].to_csv(
        output_path / "factor_subset_search_baselines.csv", index=False
    )
    return result
