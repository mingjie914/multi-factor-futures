"""Point-in-time factors built from the full futures contract OI curve."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.interfaces import Factor
from factors.user import register_user_factor


@dataclass(frozen=True)
class ContractCurveFactorSpec:
    slug: str
    base: str
    window: int
    scope: str
    expected_direction: str
    description: str


def _growth_specs(scope: str, field: str) -> list[ContractCurveFactorSpec]:
    label = {
        "main": "lagged dominant contract",
        "top2": "two largest concrete contracts",
        "total": "all concrete contracts",
    }[scope]
    return [
        ContractCurveFactorSpec(
            slug=f"curve_{scope}_oi_growth_{window}b",
            base="oi_growth",
            window=window,
            scope=field,
            expected_direction="to_be_estimated",
            description=f"Log OI growth over {window} bars using {label}",
        )
        for window in (1, 5, 20)
    ]


FACTOR_SPECS = tuple(
    _growth_specs("main", "oi")
    + _growth_specs("top2", "curve_top2_oi")
    + _growth_specs("total", "curve_total_oi")
    + [
        ContractCurveFactorSpec(
            slug=f"curve_total_oi_price_confirm_{window}b",
            base="price_confirmation",
            window=window,
            scope="curve_total_oi",
            expected_direction="positive",
            description=(
                f"{window}-bar price trend scaled by positive full-curve OI growth"
            ),
        )
        for window in (5, 20)
    ]
    + [
        ContractCurveFactorSpec(
            slug="curve_oi_breadth_price_confirm_5b",
            base="breadth_confirmation",
            window=5,
            scope="curve_oi_breadth",
            expected_direction="positive",
            description=(
                "Five-bar price trend confirmed when a majority of concrete "
                "contracts are adding open interest"
            ),
        )
    ]
)


def _empty(dates, universe) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=dates, columns=universe, dtype=float)


def _load(data, field, dates, universe):
    frame = data.get(field, dates, universe)
    if frame is None or frame.empty:
        return None
    aligned = (
        frame.reindex(index=dates, columns=universe)
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
    )
    result = _empty(dates, universe)
    for ticker in universe:
        series = aligned[ticker].dropna().shift(1)
        result.loc[series.index, ticker] = series
    return result


def _valid_bar_transform(frame: pd.DataFrame, transform) -> pd.DataFrame:
    result = pd.DataFrame(np.nan, index=frame.index, columns=frame.columns)
    for ticker in frame.columns:
        series = frame[ticker].dropna()
        if series.empty:
            continue
        transformed = transform(series)
        result.loc[transformed.index, ticker] = transformed
    return result


def _make_factor_class(spec: ContractCurveFactorSpec):
    class ContractCurvePositioningFactor(Factor):
        name = spec.slug
        category = "positioning_curve"
        frequency = "daily"
        description = spec.description
        factor_spec = spec

        def dependencies(self) -> list[str]:
            dependencies = [spec.scope]
            if spec.base in {"price_confirmation", "breadth_confirmation"}:
                dependencies.append("close")
            return dependencies

        def compute(self, data, dates, universe):
            exposure = _load(data, spec.scope, dates, universe)
            if exposure is None:
                return _empty(dates, universe)
            window = spec.window
            if spec.base == "oi_growth":
                positive = exposure.where(exposure > 0.0)
                result = _valid_bar_transform(
                    positive, lambda series: np.log(series).diff(window)
                )
            elif spec.base == "price_confirmation":
                close = _load(data, "close", dates, universe)
                if close is None:
                    return _empty(dates, universe)
                momentum = _valid_bar_transform(
                    close.where(close > 0.0),
                    lambda series: np.log(series).diff(window),
                )
                oi_growth = _valid_bar_transform(
                    exposure.where(exposure > 0.0),
                    lambda series: np.log(series).diff(window),
                )
                result = momentum * oi_growth.clip(lower=0.0)
            elif spec.base == "breadth_confirmation":
                close = _load(data, "close", dates, universe)
                if close is None:
                    return _empty(dates, universe)
                momentum = _valid_bar_transform(
                    close.where(close > 0.0),
                    lambda series: np.log(series).diff(window),
                )
                breadth = _valid_bar_transform(
                    exposure.clip(0.0, 1.0),
                    lambda series: series.rolling(
                        window, min_periods=window
                    ).mean(),
                )
                result = momentum * (breadth - 0.5).clip(lower=0.0)
            else:  # pragma: no cover
                raise ValueError(f"unsupported contract-curve base: {spec.base}")
            return result.replace([np.inf, -np.inf], np.nan).reindex(
                index=dates, columns=universe
            )

    class_name = "".join(part.title() for part in spec.slug.split("_"))
    ContractCurvePositioningFactor.__name__ = class_name
    ContractCurvePositioningFactor.__qualname__ = class_name
    return register_user_factor(
        spec.slug, category="positioning_curve"
    )(ContractCurvePositioningFactor)


for _factor_spec in FACTOR_SPECS:
    globals()[_factor_spec.slug] = _make_factor_class(_factor_spec)


__all__ = ["ContractCurveFactorSpec", "FACTOR_SPECS"] + [
    spec.slug for spec in FACTOR_SPECS
]
