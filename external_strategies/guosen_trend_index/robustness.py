from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from data.market_quality import prepare_close_data
from research.validation import OOS_END, OOS_START, SIMULATED_LIVE_START
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


def _project_batch_to_gross(
    weights: np.ndarray,
    caps: np.ndarray,
    target_gross: float,
) -> np.ndarray:
    """Project ``(candidate, date, asset)`` weights onto capped gross."""
    weights = np.asarray(weights, dtype=float)
    caps = np.asarray(caps, dtype=float)
    if weights.ndim != 3 or caps.ndim != 1 or weights.shape[-1] != len(caps):
        raise ValueError("invalid batch weight or cap shape")
    if (
        not np.isfinite(caps).all()
        or np.any(caps < 0.0)
        or not np.isfinite(target_gross)
        or target_gross < 0.0
    ):
        raise ValueError("gross caps and target must be finite and non-negative")

    active = weights > 0.0
    has_positions = active.any(axis=2)
    capacity = (active * caps[None, None, :]).sum(axis=2)
    infeasible = has_positions & (capacity < target_gross - 1e-12)
    if infeasible.any():
        candidate, date = np.argwhere(infeasible)[0]
        raise ValueError(
            f"gross projection infeasible at batch row {candidate}, date offset {date}"
        )

    result = np.zeros_like(weights)
    remaining = np.where(has_positions, target_gross, 0.0)
    unfrozen = active.copy()
    for _ in range(weights.shape[-1]):
        denominator = (weights * unfrozen).sum(axis=2)
        proposed = np.divide(
            weights * remaining[:, :, None],
            denominator[:, :, None],
            out=np.zeros_like(weights),
            where=denominator[:, :, None] > 0.0,
        )
        over_cap = unfrozen & (
            proposed > caps[None, None, :] + 1e-12
        )
        can_finish = unfrozen.any(axis=2) & ~over_cap.any(axis=2)
        if can_finish.any():
            result[can_finish] += (
                proposed[can_finish] * unfrozen[can_finish]
            )
            remaining[can_finish] = 0.0
            unfrozen[can_finish] = False
        if over_cap.any():
            result = np.where(
                over_cap, caps[None, None, :], result
            )
            remaining -= (over_cap * caps[None, None, :]).sum(axis=2)
            unfrozen &= ~over_cap
        if not unfrozen.any():
            break

    if np.max(np.abs(remaining)) > 1e-10:
        raise ValueError("gross projection did not converge")
    return result


def _prepare_contract_schedule(
    contract_schedule: pd.DataFrame | None,
    evaluation: pd.DatetimeIndex,
    universe: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Align a root-to-concrete-contract schedule once for the batch search."""
    if contract_schedule is None:
        return None
    schedule = contract_schedule.copy()
    schedule.index = pd.DatetimeIndex(schedule.index)
    if schedule.index.has_duplicates or schedule.columns.has_duplicates:
        raise ValueError("contract_schedule must have unique dates and roots")
    schedule = schedule.reindex(index=evaluation, columns=universe)
    strings = schedule.astype("string")
    valid = schedule.notna() & strings.ne("")
    return strings.fillna("").to_numpy(dtype=object), valid.to_numpy(dtype=bool)


def _simulate_batch(
    target_weights: np.ndarray,
    asset_returns: np.ndarray,
    spec: GuosenTrendIndexSpec,
    *,
    contract_schedule: tuple[np.ndarray, np.ndarray] | None = None,
    decision_tradable: np.ndarray | None = None,
    dates: pd.DatetimeIndex | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a candidate batch with close-marked, drifting exposures.

    ``target_weights[:, t]`` is observed at close ``t`` and therefore becomes
    effective on return row ``t + 1``.  The turnover computed after row ``t``
    is charged on row ``t + 1``.  The two-dimensional return/turnover outputs
    keep candidates batched while the date recursion preserves exact drift.
    """
    targets = np.asarray(target_weights, dtype=float)
    returns = np.asarray(asset_returns, dtype=float)
    if targets.ndim != 3 or returns.ndim != 2:
        raise ValueError("invalid batch simulation shape")
    candidates, n_dates, n_assets = targets.shape
    if returns.shape != (n_dates, n_assets):
        raise ValueError("target weights and returns have incompatible shapes")
    if not np.isfinite(targets).all():
        raise ValueError("target weights contain NaN or infinity")
    if contract_schedule is not None:
        symbols, valid_contract = contract_schedule
        if symbols.shape != returns.shape or valid_contract.shape != returns.shape:
            raise ValueError("contract schedule has incompatible shape")
        symbols = np.where(valid_contract, symbols, None)
    else:
        symbols = valid_contract = None
    if decision_tradable is None:
        tradable = np.ones_like(returns, dtype=bool)
    else:
        tradable = np.asarray(decision_tradable, dtype=bool)
        if tradable.shape != returns.shape:
            raise ValueError("decision tradability has incompatible shape")

    rate = float(spec.transaction_cost_rate)
    fee = float(spec.annual_management_fee) / float(spec.periods_per_year)
    if not np.isfinite([rate, fee]).all() or rate < 0.0 or fee < 0.0:
        raise ValueError("spec costs must be finite and non-negative")

    net_returns = np.zeros((candidates, n_dates), dtype=float)
    turnover = np.zeros((candidates, n_dates), dtype=float)
    current = np.zeros((candidates, n_assets), dtype=float)
    held_symbols = (
        np.full((candidates, n_assets), None, dtype=object)
        if symbols is not None else None
    )
    pending_traded_notional = np.zeros(candidates, dtype=float)
    tolerance = 1e-12

    for date_position in range(n_dates):
        current_active = np.abs(current) > tolerance
        marked_returns = returns[date_position]
        missing_returns = current_active & ~np.isfinite(marked_returns)[None, :]
        if missing_returns.any():
            candidate, asset = np.argwhere(missing_returns)[0]
            date_label = (
                dates[date_position].strftime("%Y-%m-%d")
                if dates is not None else str(date_position)
            )
            raise ValueError(
                "active asset return is missing at "
                f"{date_label}: candidate={candidate}, asset={asset}"
            )
        if symbols is not None and (
            current_active & pd.isna(held_symbols)
        ).any():
            candidate, asset = np.argwhere(
                current_active & pd.isna(held_symbols)
            )[0]
            raise ValueError(
                "active contract is missing at "
                f"{dates[date_position] if dates is not None else date_position}: "
                f"candidate={candidate}, asset={asset}"
            )

        safe_returns = np.where(np.isfinite(marked_returns), marked_returns, 0.0)
        if (current_active & (safe_returns[None, :] <= -1.0)).any():
            candidate, asset = np.argwhere(
                current_active & (safe_returns[None, :] <= -1.0)
            )[0]
            raise ValueError(
                f"active asset return is <= -100%: candidate={candidate}, asset={asset}"
            )

        gross = (current * safe_returns[None, :]).sum(axis=1)
        net = gross - pending_traded_notional * rate - fee
        if date_position == 0:
            # There is no prior target and therefore no holding bar on row 0.
            net[:] = 0.0
        if (~np.isfinite(net)).any() or (1.0 + net <= 0.0).any():
            candidate = int(np.flatnonzero((~np.isfinite(net)) | (1.0 + net <= 0.0))[0])
            raise ValueError(f"candidate NAV is exhausted at row {candidate}")
        net_returns[:, date_position] = net

        end_weights = current * (1.0 + safe_returns[None, :]) / (1.0 + net)[:, None]
        if date_position == n_dates - 1:
            continue

        desired_target = targets[:, date_position, :]
        transition_tradable = (
            tradable[date_position] & tradable[date_position + 1]
        )
        target = np.where(
            transition_tradable[None, :], desired_target, end_weights
        )
        target_active = np.abs(target) > tolerance
        target_symbols = None
        if symbols is not None:
            target_symbols = np.broadcast_to(
                symbols[date_position + 1][None, :], (candidates, n_assets)
            ).astype(object, copy=True)
            target_symbols = np.where(
                transition_tradable[None, :], target_symbols, held_symbols
            )
            missing_target_contract = target_active & pd.isna(target_symbols)
            if missing_target_contract.any():
                candidate, asset = np.argwhere(missing_target_contract)[0]
                raise ValueError(
                    "active target contract is missing at "
                    f"{dates[date_position + 1] if dates is not None else date_position + 1}: "
                    f"candidate={candidate}, asset={asset}"
                )

        if symbols is None:
            traded = np.abs(target - end_weights).sum(axis=1)
        else:
            rollover = (
                current_active
                & target_active
                & (held_symbols != target_symbols)
            )
            traded_by_asset = np.where(
                rollover,
                np.abs(end_weights) + np.abs(target),
                np.abs(target - end_weights),
            )
            traded = traded_by_asset.sum(axis=1)
        turnover[:, date_position] = traded
        pending_traded_notional = traded
        current = target
        if target_symbols is not None:
            held_symbols = np.where(target_active, target_symbols, None)

    return net_returns, turnover


def exhaustive_subset_search(
    portfolios: Mapping[str, pd.DataFrame],
    close: pd.DataFrame,
    spec: GuosenTrendIndexSpec,
    start: pd.Timestamp,
    end: pd.Timestamp,
    baseline_sets: Mapping[str, set[str]],
    output: str | Path,
    target_gross: float = 1.0,
    contract_schedule: pd.DataFrame | None = None,
    audited_nontrading_closes=None,
) -> pd.DataFrame:
    """Evaluate all subsets without recomputing factors or risk portfolios.

    Candidate ranking uses only three development segments ending in 2024.
    Fixed OOS and simulated-live segments are reported separately and neither
    is part of ``development_score``.
    """
    if spec.execution_lag_days != 0:
        raise ValueError("subset cache currently requires execution_lag_days=0")
    names = list(portfolios)
    if not names:
        raise ValueError("no factor portfolios supplied")
    close = close.reindex(columns=spec.universe).sort_index()
    dates = pd.DatetimeIndex(close.index)
    evaluation = dates[(dates >= start) & (dates <= end)]
    if evaluation.empty:
        raise ValueError("evaluation window contains no close dates")
    components = np.stack(
        [
            portfolios[name].reindex(index=evaluation, columns=spec.universe)
            .fillna(0.0).to_numpy(dtype=float)
            for name in names
        ],
        axis=0,
    )
    marked_returns, close_tradable = prepare_close_data(
        close, audited_nontrading_closes
    )
    asset_returns = marked_returns.reindex(evaluation).to_numpy(dtype=float)
    decision_tradable = close_tradable.reindex(evaluation).to_numpy(dtype=bool)
    caps = pd.Series(spec.asset_caps).reindex(spec.universe).to_numpy(dtype=float)
    if not np.isfinite(caps).all() or np.any(caps < 0.0):
        raise ValueError("asset caps must be finite, non-negative and cover the universe")
    schedule = _prepare_contract_schedule(contract_schedule, evaluation, spec.universe)
    baseline_lookup = {
        frozenset(factors): label for label, factors in baseline_sets.items()
    }
    segments = {
        "dev_2016_2019": (pd.Timestamp("2016-03-31"), pd.Timestamp("2019-12-31")),
        "dev_2020_2022": (pd.Timestamp("2020-01-01"), pd.Timestamp("2022-12-31")),
        "dev_2023_2024": (pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")),
        "oos_2025_20260514": (pd.Timestamp(OOS_START), pd.Timestamp(OOS_END)),
        "simulated_live": (pd.Timestamp(SIMULATED_LIVE_START), end),
    }
    segment_masks = {
        label: (evaluation >= left) & (evaluation <= right)
        for label, (left, right) in segments.items()
    }
    pre_oos = evaluation < pd.Timestamp(OOS_START)
    records = []
    total = (1 << len(names)) - 1
    # The recursion is exact in time but candidates stay in a NumPy batch.
    # 128 keeps the peak temporary array bounded for the 14-factor 16,383 run.
    batch_size = 128
    factor_activity = (components.sum(axis=2) > 0.0).astype(float)
    masks = np.arange(1, total + 1, dtype=np.uint32)
    factor_bits = np.arange(len(names), dtype=np.uint32)
    for batch_start in range(0, len(masks), batch_size):
        batch_masks = masks[batch_start:batch_start + batch_size]
        bits = ((batch_masks[:, None] >> factor_bits[None, :]) & 1).astype(float)
        selected_sum = np.einsum(
            "bf,fda->bda", bits, components, optimize=True
        )
        active_factor_count = bits @ factor_activity
        weights = np.divide(
            selected_sum,
            active_factor_count[:, :, None],
            out=np.zeros_like(selected_sum),
            where=active_factor_count[:, :, None] > 0.0,
        )
        weights = np.clip(weights, 0.0, caps[None, None, :])
        weights = _project_batch_to_gross(weights, caps, target_gross)
        net_returns, turnover = _simulate_batch(
            weights,
            asset_returns,
            spec,
            contract_schedule=schedule,
            decision_tradable=decision_tradable,
            dates=evaluation,
        )

        for row, mask in enumerate(batch_masks.tolist()):
            indices = [index for index in range(len(names)) if mask & (1 << index)]
            selected_names = [names[index] for index in indices]
            candidate_returns = net_returns[row]
            candidate_turnover = turnover[row]
            segment_sharpes = {
                label: _sharpe(candidate_returns[segment_mask], spec.periods_per_year)
                for label, segment_mask in segment_masks.items()
            }
            development = np.array(
                [segment_sharpes[label] for label in list(segments)[:3]], dtype=float
            )
            pre_returns = candidate_returns[pre_oos]
            full_sharpe = _sharpe(candidate_returns, spec.periods_per_year)
            pre_oos_sharpe = _sharpe(pre_returns, spec.periods_per_year)
            development_score = (
                0.30 * pre_oos_sharpe
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
                "pre_2025_sharpe": pre_oos_sharpe,
                "full_sharpe": full_sharpe,
                "full_max_drawdown": _max_drawdown(candidate_returns),
                "annual_turnover": float(
                    candidate_turnover.mean() * spec.periods_per_year
                ),
                "positive_development_segments": int((development > 0.0).sum()),
                "worst_development_sharpe": float(np.min(development)),
                "median_development_sharpe": float(np.median(development)),
                **segment_sharpes,
            })
        last_mask = int(batch_masks[-1])
        if last_mask % 2048 == 0 or last_mask == total:
            print(f"subset search: {last_mask}/{total}", flush=True)
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
