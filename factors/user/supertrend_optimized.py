"""One preregistered stability refinement of the Supertrend distance factor."""
from __future__ import annotations

import numpy as np

from core.interfaces import Factor
from factors.user import register_user_factor
from factors.user.supertrend import (
    DISTANCE_CLIP,
    _empty,
    _load_inputs,
    _supertrend_components,
)


@register_user_factor("supertrend_distance_smooth3_20_2", category="trend")
class SupertrendDistanceSmooth3(Factor):
    name = "supertrend_distance_smooth3_20_2"
    category = "trend"
    frequency = "daily"
    description = (
        "Three-bar exponentially smoothed signed distance from the active "
        "Supertrend line using Wilder ATR(20) and multiplier 2"
    )

    def dependencies(self) -> list[str]:
        return ["high", "low", "close"]

    def compute(self, data, dates, universe):
        inputs = _load_inputs(data, dates, universe)
        if inputs is None:
            return _empty(dates, universe)
        close = inputs["close"]
        direction, active_line, atr, _ = _supertrend_components(
            inputs["high"], inputs["low"], close
        )
        distance = (
            direction * (close - active_line).abs() / atr.replace(0, np.nan)
        ).clip(-DISTANCE_CLIP, DISTANCE_CLIP)
        result = distance.ewm(
            span=3,
            adjust=False,
            min_periods=3,
        ).mean().shift(1)
        return result.replace([np.inf, -np.inf], np.nan).reindex(
            index=dates, columns=universe
        )


__all__ = ["SupertrendDistanceSmooth3"]
