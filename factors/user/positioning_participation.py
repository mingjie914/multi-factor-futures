"""SPEC-style futures positioning and participation factors.

All source fields are lagged by one bar before a signal is computed. This
keeps close-derived daily factors usable at the next decision point without
consuming the current close.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.interfaces import Factor
from factors.user import register_user_factor


@dataclass(frozen=True)
class PositioningFactorSpec:
    slug: str
    base: str
    window: int
    expected_direction: str
    description: str


_BASE_DESCRIPTIONS = {
    "oi_build_trend": (
        "Price trend scaled by positive aggregate open-interest growth; "
        "expected to continue when new positions confirm the move"
    ),
    "opening_flow_pressure": (
        "Return pressure weighted by the share of volume becoming new open "
        "interest; expected to continue in the pressure direction"
    ),
    "low_churn_trend": (
        "Price trend scaled by inverse volume/open-interest turnover; expected "
        "to persist when position churn is low"
    ),
    "liquidation_pressure": (
        "Return pressure weighted by the share of volume associated with open-"
        "interest contraction; expected to mean-revert"
    ),
}


FACTOR_SPECS = tuple(
    PositioningFactorSpec(
        slug=f"{base}_{window}d",
        base=base,
        window=window,
        expected_direction="negative" if base == "liquidation_pressure" else "positive",
        description=f"{description} ({window}-bar window)",
    )
    for base, description in _BASE_DESCRIPTIONS.items()
    for window in (5, 10, 20)
)


def _empty(dates, universe) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=dates, columns=universe, dtype=float)


def _load_lagged_inputs(data, dates, universe):
    frames = {}
    for field in ("close", "volume", "oi"):
        frame = data.get(field, dates, universe)
        if frame is None or frame.empty:
            return None
        frames[field] = (
            frame.reindex(index=dates, columns=universe)
            .astype(float)
            .shift(1)
            .replace([np.inf, -np.inf], np.nan)
        )
    return frames


def _compute_base(spec: PositioningFactorSpec, inputs) -> pd.DataFrame:
    close = inputs["close"].where(inputs["close"] > 0)
    volume = inputs["volume"].where(inputs["volume"] > 0)
    oi = inputs["oi"].where(inputs["oi"] > 0)
    log_close = np.log(close)
    log_oi = np.log(oi)
    daily_return = log_close.diff()
    window = spec.window

    if spec.base == "oi_build_trend":
        momentum = log_close.diff(window)
        oi_growth = log_oi.diff(window).clip(lower=-0.5, upper=1.0)
        result = momentum * oi_growth.clip(lower=0.0)
    elif spec.base == "opening_flow_pressure":
        opening_share = (oi.diff().clip(lower=0.0) / volume).clip(0.0, 1.0)
        result = (daily_return * opening_share).rolling(
            window, min_periods=window
        ).sum()
    elif spec.base == "low_churn_trend":
        momentum = log_close.diff(window)
        turnover = (volume / oi).clip(lower=1e-4, upper=10.0)
        average_turnover = turnover.rolling(window, min_periods=window).mean()
        result = momentum / np.sqrt(average_turnover)
    elif spec.base == "liquidation_pressure":
        closing_share = ((-oi.diff()).clip(lower=0.0) / volume).clip(0.0, 1.0)
        result = (daily_return * closing_share).rolling(
            window, min_periods=window
        ).sum()
    else:  # pragma: no cover - guarded by the immutable specs above
        raise ValueError(f"unsupported positioning factor base: {spec.base}")

    return result.replace([np.inf, -np.inf], np.nan)


def _make_factor_class(spec: PositioningFactorSpec):
    class PositioningParticipationFactor(Factor):
        name = spec.slug
        category = "positioning_participation"
        frequency = "daily"
        description = spec.description
        factor_spec = spec

        def dependencies(self) -> list[str]:
            return ["close", "volume", "oi"]

        def compute(self, data, dates, universe):
            inputs = _load_lagged_inputs(data, dates, universe)
            if inputs is None:
                return _empty(dates, universe)
            return _compute_base(self.factor_spec, inputs).reindex(
                index=dates, columns=universe
            )

    class_name = "".join(part.title() for part in spec.slug.split("_"))
    PositioningParticipationFactor.__name__ = class_name
    PositioningParticipationFactor.__qualname__ = class_name
    return register_user_factor(
        spec.slug, category="positioning_participation"
    )(PositioningParticipationFactor)


for _factor_spec in FACTOR_SPECS:
    globals()[_factor_spec.slug] = _make_factor_class(_factor_spec)


__all__ = ["FACTOR_SPECS", "PositioningFactorSpec"] + [
    spec.slug for spec in FACTOR_SPECS
]
