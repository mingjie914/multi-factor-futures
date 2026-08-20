"""Shared, fail-closed construction for production-style futures sleeves."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


class PortfolioConstructionError(ValueError):
    """Raised when a requested portfolio cannot satisfy its declared rules."""


@dataclass(frozen=True)
class PortfolioConstraints:
    """Constraints expressed as fractions of each long or short sleeve."""

    top_n_per_side: int = 10
    sector_count_cap: int = 3
    asset_min_fraction: float = 0.005
    asset_max_fraction: float = 0.20
    asset_max_overrides: Mapping[str, float] = field(default_factory=dict)
    sector_weight_caps: Mapping[str, float] = field(default_factory=dict)
    gross_exposure: float = 2.0
    covariance_shrinkage: float = 0.30
    minimum_risk_observations: int = 10

    def __post_init__(self) -> None:
        if int(self.top_n_per_side) <= 0:
            raise PortfolioConstructionError("top_n_per_side must be positive")
        if int(self.sector_count_cap) < 0:
            raise PortfolioConstructionError("sector_count_cap cannot be negative")
        if not 0.0 <= float(self.asset_min_fraction) <= 1.0:
            raise PortfolioConstructionError("asset_min_fraction must be in [0, 1]")
        if not 0.0 < float(self.asset_max_fraction) <= 1.0:
            raise PortfolioConstructionError("asset_max_fraction must be in (0, 1]")
        if float(self.asset_min_fraction) > float(self.asset_max_fraction):
            raise PortfolioConstructionError(
                "asset_min_fraction cannot exceed asset_max_fraction"
            )
        if not np.isfinite(float(self.gross_exposure)) or self.gross_exposure <= 0.0:
            raise PortfolioConstructionError("gross_exposure must be positive")
        if not 0.0 <= float(self.covariance_shrinkage) <= 1.0:
            raise PortfolioConstructionError("covariance_shrinkage must be in [0, 1]")
        if int(self.minimum_risk_observations) < 2:
            raise PortfolioConstructionError(
                "minimum_risk_observations must be at least 2"
            )
        for symbol, value in self.asset_max_overrides.items():
            if not str(symbol).strip() or not 0.0 < float(value) <= 1.0:
                raise PortfolioConstructionError(
                    f"invalid asset maximum override: {symbol!r}={value!r}"
                )
        for sector, value in self.sector_weight_caps.items():
            if not str(sector).strip() or not 0.0 < float(value) <= 1.0:
                raise PortfolioConstructionError(
                    f"invalid sector weight cap: {sector!r}={value!r}"
                )

    @classmethod
    def from_config(cls, config, *, top_n: int | None = None) -> "PortfolioConstraints":
        return cls(
            top_n_per_side=(
                int(top_n) if top_n is not None else int(config.top_n_per_side)
            ),
            sector_count_cap=int(config.sector_count_cap),
            asset_min_fraction=float(config.asset_min_fraction),
            asset_max_fraction=float(config.asset_max_fraction),
            asset_max_overrides={
                str(key).upper(): float(value)
                for key, value in config.asset_max_overrides.items()
            },
            sector_weight_caps={
                str(key): float(value)
                for key, value in config.sector_weight_caps.items()
            },
            gross_exposure=float(config.gross_exposure),
            covariance_shrinkage=float(config.covariance_shrinkage),
            minimum_risk_observations=int(config.minimum_risk_observations),
        )


def _ordered_symbols(score: pd.Series, *, ascending: bool) -> list[str]:
    ranked = pd.DataFrame({
        "symbol": score.index.astype(str),
        "score": score.to_numpy(dtype=float),
    })
    ranked = ranked.sort_values(
        ["score", "symbol"],
        ascending=[ascending, True],
        kind="mergesort",
    )
    return ranked["symbol"].tolist()


def select_pool(
    score: pd.Series,
    *,
    eligible: Sequence[str],
    sector_of: Mapping[str, str],
    top_n: int,
    sector_count_cap: int,
    ascending: bool,
    excluded: Sequence[str] = (),
) -> list[str]:
    """Select one deterministic side and require its exact requested size."""
    eligible_set = {str(symbol) for symbol in eligible}
    excluded_set = {str(symbol) for symbol in excluded}
    clean = pd.Series(
        score.to_numpy(dtype=float),
        index=score.index.astype(str),
        dtype=float,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.index.has_duplicates:
        duplicates = clean.index[clean.index.duplicated()].unique().tolist()
        raise PortfolioConstructionError(
            f"duplicate symbols after normalization: {duplicates[:5]}"
        )
    clean = clean.loc[
        [
            symbol for symbol in clean.index
            if symbol in eligible_set and symbol not in excluded_set
        ]
    ]
    picks: list[str] = []
    sector_counts: dict[str, int] = {}
    for symbol in _ordered_symbols(clean, ascending=ascending):
        sector = str(sector_of.get(symbol, "其他"))
        if (
            int(sector_count_cap) > 0
            and sector_counts.get(sector, 0) >= int(sector_count_cap)
        ):
            continue
        picks.append(symbol)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(picks) == int(top_n):
            return picks
    raise PortfolioConstructionError(
        f"cannot select {top_n} assets with sector_count_cap={sector_count_cap}; "
        f"selected={len(picks)}, eligible={len(clean)}"
    )


def select_long_short_pools(
    score: pd.Series,
    *,
    eligible: Sequence[str],
    sector_of: Mapping[str, str],
    constraints: PortfolioConstraints,
) -> tuple[list[str], list[str]]:
    """Select disjoint Top/Bottom pools using the same rules everywhere."""
    long_pool = select_pool(
        score,
        eligible=eligible,
        sector_of=sector_of,
        top_n=constraints.top_n_per_side,
        sector_count_cap=constraints.sector_count_cap,
        ascending=False,
    )
    short_pool = select_pool(
        score,
        eligible=eligible,
        sector_of=sector_of,
        top_n=constraints.top_n_per_side,
        sector_count_cap=constraints.sector_count_cap,
        ascending=True,
        excluded=long_pool,
    )
    return long_pool, short_pool


def prepare_risk_history(
    returns: pd.DataFrame,
    pool: Sequence[str],
    minimum_observations: int,
) -> pd.DataFrame:
    """Return complete causal observations without silently dropping assets."""
    symbols = list(dict.fromkeys(str(symbol) for symbol in pool))
    history = returns.reindex(columns=symbols).replace([np.inf, -np.inf], np.nan)
    counts = history.notna().sum(axis=0)
    missing = counts[counts < int(minimum_observations)]
    if len(missing):
        raise PortfolioConstructionError(
            "insufficient risk history: "
            + ", ".join(f"{name}={int(value)}" for name, value in missing.items())
        )
    complete = history.dropna(axis=0, how="any")
    if len(complete) < int(minimum_observations):
        raise PortfolioConstructionError(
            "insufficient complete risk rows: "
            f"{len(complete)} < {int(minimum_observations)}"
        )
    return complete


def causal_risk_window(
    returns: pd.DataFrame,
    decision_date: pd.Timestamp,
    lookback_calendar_days: int,
) -> pd.DataFrame:
    """Slice one shared ``[T-lookback, T)`` risk window."""
    frame = pd.DataFrame(returns, copy=False)
    index = pd.DatetimeIndex(frame.index)
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise PortfolioConstructionError("risk-return dates must be unique and sorted")
    decision = pd.Timestamp(decision_date)
    lookback = int(lookback_calendar_days)
    if lookback <= 0:
        raise PortfolioConstructionError("risk lookback must be positive")
    start = decision - pd.Timedelta(days=lookback)
    return frame.loc[(index >= start) & (index < decision)]


def _box_simplex_projection(
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Euclidean projection onto ``sum(w)=1`` with elementwise bounds."""
    tolerance = 1e-12
    if lower.sum() > 1.0 + tolerance or upper.sum() < 1.0 - tolerance:
        raise PortfolioConstructionError(
            "asset bounds are infeasible for a unit sleeve: "
            f"sum(lower)={lower.sum():.6f}, sum(upper)={upper.sum():.6f}"
        )
    lo = float(np.min(target - upper) - 1.0)
    hi = float(np.max(target - lower) + 1.0)
    for _ in range(80):
        middle = (lo + hi) / 2.0
        projected = np.clip(target - middle, lower, upper)
        if projected.sum() > 1.0:
            lo = middle
        else:
            hi = middle
    result = np.clip(target - (lo + hi) / 2.0, lower, upper)
    residual = 1.0 - float(result.sum())
    if abs(residual) > tolerance:
        room = upper - result if residual > 0.0 else result - lower
        candidates = np.flatnonzero(room > tolerance)
        for index in candidates:
            adjustment = np.sign(residual) * min(abs(residual), float(room[index]))
            result[index] += adjustment
            residual -= adjustment
            if abs(residual) <= tolerance:
                break
    if abs(float(result.sum()) - 1.0) > 1e-10:
        raise PortfolioConstructionError("bounded simplex projection did not converge")
    return result


def _project_with_sector_caps(
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    symbols: Sequence[str],
    sector_of: Mapping[str, str],
    sector_caps: Mapping[str, float],
) -> np.ndarray:
    """Use SLSQP only when optional sector-weight caps are configured."""
    from scipy.optimize import minimize

    initial = _box_simplex_projection(target, lower, upper)
    constraints: list[dict] = [{
        "type": "eq", "fun": lambda values: float(values.sum() - 1.0)
    }]
    for sector, cap in sector_caps.items():
        indices = np.array(
            [i for i, symbol in enumerate(symbols) if sector_of.get(symbol, "其他") == sector],
            dtype=int,
        )
        if len(indices) == 0:
            continue
        if float(lower[indices].sum()) > float(cap) + 1e-12:
            raise PortfolioConstructionError(
                f"sector cap is infeasible: {sector} lower={lower[indices].sum():.6f} "
                f"> cap={float(cap):.6f}"
            )
        constraints.append({
            "type": "ineq",
            "fun": lambda values, idx=indices, limit=float(cap): (
                limit - float(values[idx].sum())
            ),
        })
    result = minimize(
        lambda values: 0.5 * float(np.sum((values - target) ** 2)),
        initial,
        method="SLSQP",
        bounds=list(zip(lower, upper)),
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 200, "disp": False},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise PortfolioConstructionError(
            f"sector-capped weight projection failed: {result.message}"
        )
    return np.asarray(result.x, dtype=float)


def allocate_sleeve(
    history: pd.DataFrame,
    *,
    method: str,
    constraints: PortfolioConstraints,
    sector_of: Mapping[str, str],
) -> pd.Series:
    """Allocate one unit-gross sleeve, then enforce declared bounds exactly."""
    symbols = list(history.columns.astype(str))
    if len(symbols) != constraints.top_n_per_side:
        raise PortfolioConstructionError(
            f"allocation requires exactly {constraints.top_n_per_side} assets; "
            f"received={len(symbols)}"
        )
    if method == "equal":
        raw = np.full(len(symbols), 1.0 / len(symbols), dtype=float)
    elif method == "inverse_volatility":
        inverse = 1.0 / history.std(ddof=0).replace(0.0, np.nan)
        if not np.isfinite(inverse.to_numpy(dtype=float)).all():
            raise PortfolioConstructionError("inverse-volatility input is invalid")
        raw = inverse.to_numpy(dtype=float) / float(inverse.sum())
    elif method == "erc":
        from optimization.risk_budgeting import RiskBudgetingOptimizer

        covariance_raw = history.cov().to_numpy(dtype=float)
        diagonal = np.diag(np.diag(covariance_raw))
        shrinkage = float(constraints.covariance_shrinkage)
        covariance = (1.0 - shrinkage) * covariance_raw + shrinkage * diagonal
        try:
            raw = RiskBudgetingOptimizer._erc_weights(
                covariance, np.ones(len(symbols), dtype=float)
            )
        except (RuntimeError, ValueError) as exc:
            raise PortfolioConstructionError("ERC allocation failed") from exc
    else:
        raise PortfolioConstructionError(f"unknown asset-weight method: {method}")

    raw = np.asarray(raw, dtype=float)
    if not np.isfinite(raw).all() or float(raw.sum()) <= 0.0:
        raise PortfolioConstructionError("asset allocator returned invalid weights")
    raw = np.maximum(raw, 0.0)
    raw /= raw.sum()
    lower = np.full(len(symbols), float(constraints.asset_min_fraction))
    upper = np.array([
        float(
            constraints.asset_max_overrides.get(
                symbol, constraints.asset_max_fraction
            )
        )
        for symbol in symbols
    ])
    if np.any(lower > upper):
        raise PortfolioConstructionError("an asset maximum is below its minimum")
    if constraints.sector_weight_caps:
        projected = _project_with_sector_caps(
            raw,
            lower,
            upper,
            symbols,
            sector_of,
            constraints.sector_weight_caps,
        )
    else:
        projected = _box_simplex_projection(raw, lower, upper)
    tolerance = 1e-9
    if (
        not np.isfinite(projected).all()
        or not np.isclose(float(projected.sum()), 1.0, atol=tolerance)
        or np.any(projected < lower - tolerance)
        or np.any(projected > upper + tolerance)
    ):
        raise PortfolioConstructionError("projected sleeve violates asset bounds")
    for sector, cap in constraints.sector_weight_caps.items():
        sector_weight = sum(
            float(projected[index])
            for index, symbol in enumerate(symbols)
            if sector_of.get(symbol, "其他") == sector
        )
        if sector_weight > float(cap) + tolerance:
            raise PortfolioConstructionError(
                f"projected sleeve violates sector cap: {sector}"
            )
    return pd.Series(projected, index=symbols, dtype=float)


def validate_long_short_weights(
    weights: pd.Series,
    *,
    long_pool: Sequence[str],
    short_pool: Sequence[str],
    constraints: PortfolioConstraints,
    sector_of: Mapping[str, str],
) -> None:
    """Validate all final production invariants after scaling and netting."""
    tolerance = 1e-9
    clean = weights.replace([np.inf, -np.inf], np.nan)
    if clean.isna().any():
        raise PortfolioConstructionError("final weights contain NaN or infinity")
    long = clean[clean > tolerance]
    short = -clean[clean < -tolerance]
    expected = int(constraints.top_n_per_side)
    if len(long) != expected or len(short) != expected:
        raise PortfolioConstructionError(
            f"final holdings must be {expected} long/{expected} short; "
            f"received={len(long)}/{len(short)}"
        )
    if set(long.index) != set(long_pool) or set(short.index) != set(short_pool):
        raise PortfolioConstructionError("final holdings differ from selected pools")
    side_gross = float(constraints.gross_exposure) / 2.0
    if not np.isclose(long.sum(), side_gross, atol=tolerance):
        raise PortfolioConstructionError("long sleeve gross exposure is invalid")
    if not np.isclose(short.sum(), side_gross, atol=tolerance):
        raise PortfolioConstructionError("short sleeve gross exposure is invalid")
    if not np.isclose(clean.abs().sum(), constraints.gross_exposure, atol=tolerance):
        raise PortfolioConstructionError("portfolio gross exposure is invalid")
    if not np.isclose(clean.sum(), 0.0, atol=tolerance):
        raise PortfolioConstructionError("portfolio net exposure is not neutral")

    for sleeve in (long, short):
        fractions = sleeve / side_gross
        for symbol, value in fractions.items():
            maximum = float(
                constraints.asset_max_overrides.get(
                    str(symbol), constraints.asset_max_fraction
                )
            )
            if value > maximum + tolerance or value < constraints.asset_min_fraction - tolerance:
                raise PortfolioConstructionError(
                    f"asset bound violated: {symbol}={value:.8f}, "
                    f"bounds=[{constraints.asset_min_fraction:.8f}, {maximum:.8f}]"
                )
        sector_counts: dict[str, int] = {}
        sector_weights: dict[str, float] = {}
        for symbol, value in fractions.items():
            sector = str(sector_of.get(str(symbol), "其他"))
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            sector_weights[sector] = sector_weights.get(sector, 0.0) + float(value)
        if constraints.sector_count_cap > 0 and any(
            count > constraints.sector_count_cap for count in sector_counts.values()
        ):
            raise PortfolioConstructionError("sector count cap is violated")
        for sector, cap in constraints.sector_weight_caps.items():
            if sector_weights.get(sector, 0.0) > float(cap) + tolerance:
                raise PortfolioConstructionError(
                    f"sector weight cap violated: {sector}"
                )


def combine_sleeves(
    long_weights: pd.Series,
    short_weights: pd.Series,
    *,
    universe: Sequence[str],
    long_pool: Sequence[str],
    short_pool: Sequence[str],
    constraints: PortfolioConstraints,
    sector_of: Mapping[str, str],
) -> pd.Series:
    """Scale two unit sleeves, net them, and validate the final portfolio."""
    side_gross = float(constraints.gross_exposure) / 2.0
    result = pd.Series(0.0, index=list(universe), dtype=float)
    result.loc[long_weights.index] += long_weights.to_numpy() * side_gross
    result.loc[short_weights.index] -= short_weights.to_numpy() * side_gross
    validate_long_short_weights(
        result,
        long_pool=long_pool,
        short_pool=short_pool,
        constraints=constraints,
        sector_of=sector_of,
    )
    return result[result.abs() > 1e-12]


__all__ = [
    "PortfolioConstraints",
    "PortfolioConstructionError",
    "allocate_sleeve",
    "causal_risk_window",
    "combine_sleeves",
    "prepare_risk_history",
    "select_long_short_pools",
    "select_pool",
    "validate_long_short_weights",
]
