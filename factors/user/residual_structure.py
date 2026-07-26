"""Point-in-time common-component residual factors for futures."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.interfaces import Factor
from core.sectors import sector_for
from factors.user import register_user_factor


@dataclass(frozen=True)
class ResidualStructureSpec:
    slug: str
    base: str
    window: int
    expected_direction: str
    description: str


_SPECS = (
    ("market_residual_momentum_20d", "market_momentum", 20, "positive"),
    ("market_residual_momentum_60d", "market_momentum", 60, "positive"),
    ("market_residual_momentum_120d", "market_momentum", 120, "positive"),
    ("sector_residual_momentum_20d", "sector_momentum", 20, "positive"),
    ("sector_residual_momentum_60d", "sector_momentum", 60, "positive"),
    ("sector_residual_momentum_120d", "sector_momentum", 120, "positive"),
    ("market_residual_reversal_5d", "market_reversal", 5, "positive"),
    ("market_residual_reversal_10d", "market_reversal", 10, "positive"),
    ("idiosyncratic_volatility_20d", "idio_volatility", 20, "negative"),
    ("idiosyncratic_volatility_60d", "idio_volatility", 60, "negative"),
    ("residual_max_return_20d", "residual_max", 20, "negative"),
    ("residual_max_return_60d", "residual_max", 60, "negative"),
)


FACTOR_SPECS = tuple(
    ResidualStructureSpec(
        slug=slug,
        base=base,
        window=window,
        expected_direction=direction,
        description=(
            f"{base.replace('_', ' ')} over {window} bars after removing "
            "the lagged common futures return component"
        ),
    )
    for slug, base, window, direction in _SPECS
)


def _empty(dates, universe) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=dates, columns=universe, dtype=float)


def _lagged_returns(data, dates, universe) -> pd.DataFrame | None:
    close = data.get("close", dates, universe)
    if close is None or close.empty:
        return None
    close = (
        close.reindex(index=dates, columns=universe)
        .astype(float)
        .shift(1)
        .where(lambda frame: frame > 0)
    )
    return np.log(close).diff().replace([np.inf, -np.inf], np.nan)


def _market_residual(returns: pd.DataFrame, beta_window: int = 60) -> pd.DataFrame:
    market = returns.median(axis=1, skipna=True)
    min_periods = max(20, beta_window // 2)
    mean_asset = returns.rolling(beta_window, min_periods=min_periods).mean()
    mean_market = market.rolling(beta_window, min_periods=min_periods).mean()
    mean_product = returns.mul(market, axis=0).rolling(
        beta_window, min_periods=min_periods
    ).mean()
    covariance = mean_product - mean_asset.mul(mean_market, axis=0)
    variance = market.rolling(beta_window, min_periods=min_periods).var(ddof=0)
    beta = covariance.div(variance.replace(0.0, np.nan), axis=0)
    return returns - beta.mul(market, axis=0)


def _sector_residual(returns: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    groups: dict[str, list[str]] = {}
    for instrument in returns.columns:
        groups.setdefault(sector_for(str(instrument)), []).append(instrument)
    for instruments in groups.values():
        if len(instruments) < 2:
            continue
        sector_return = returns[instruments].median(axis=1, skipna=True)
        result[instruments] = returns[instruments].sub(sector_return, axis=0)
    return result


def _compute_spec(
    spec: ResidualStructureSpec, returns: pd.DataFrame
) -> pd.DataFrame:
    market_residual = _market_residual(returns)
    min_periods = spec.window
    if spec.base == "market_momentum":
        result = market_residual.rolling(
            spec.window, min_periods=min_periods
        ).sum()
    elif spec.base == "sector_momentum":
        result = _sector_residual(returns).rolling(
            spec.window, min_periods=min_periods
        ).sum()
    elif spec.base == "market_reversal":
        result = -market_residual.rolling(
            spec.window, min_periods=min_periods
        ).sum()
    elif spec.base == "idio_volatility":
        result = -market_residual.rolling(
            spec.window, min_periods=min_periods
        ).std(ddof=0)
    elif spec.base == "residual_max":
        result = -market_residual.rolling(
            spec.window, min_periods=min_periods
        ).max()
    else:
        raise ValueError(f"unsupported residual-structure base: {spec.base}")
    return result.replace([np.inf, -np.inf], np.nan)


def _make_factor_class(spec: ResidualStructureSpec):
    class ResidualStructureFactor(Factor):
        name = spec.slug
        category = "residual_structure"
        frequency = "daily"
        description = spec.description
        factor_spec = spec

        def dependencies(self) -> list[str]:
            return ["close"]

        def compute(self, data, dates, universe):
            returns = _lagged_returns(data, dates, universe)
            if returns is None:
                return _empty(dates, universe)
            return _compute_spec(self.factor_spec, returns).reindex(
                index=dates, columns=universe
            )

    class_name = "".join(part.title() for part in spec.slug.split("_"))
    ResidualStructureFactor.__name__ = class_name
    ResidualStructureFactor.__qualname__ = class_name
    return register_user_factor(spec.slug, category="residual_structure")(
        ResidualStructureFactor
    )


for _factor_spec in FACTOR_SPECS:
    globals()[_factor_spec.slug] = _make_factor_class(_factor_spec)


__all__ = ["FACTOR_SPECS", "ResidualStructureSpec"] + [
    spec.slug for spec in FACTOR_SPECS
]
