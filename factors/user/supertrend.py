"""Point-in-time Supertrend factors with a 20-bar Wilder ATR and multiplier 2."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.interfaces import Factor
from factors.user import register_user_factor


ATR_WINDOW = 20
ATR_MULTIPLIER = 2.0
DISTANCE_CLIP = 5.0


@dataclass(frozen=True)
class SupertrendFactorSpec:
    slug: str
    output: str
    description: str


FACTOR_SPECS = (
    SupertrendFactorSpec(
        slug="supertrend_state_20_2",
        output="state",
        description="Supertrend direction using Wilder ATR(20) and multiplier 2",
    ),
    SupertrendFactorSpec(
        slug="supertrend_distance_20_2",
        output="distance",
        description=(
            "Signed distance from the active Supertrend line, normalized by "
            "Wilder ATR(20) and clipped to five ATR"
        ),
    ),
    SupertrendFactorSpec(
        slug="supertrend_flip_20_2",
        output="flip",
        description="Supertrend direction changes: bullish +1, bearish -1, else 0",
    ),
)


def _empty(dates, universe) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=dates, columns=universe, dtype=float)


def _coherent_ohlc_columns(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    *,
    tolerance: float = 0.05,
) -> pd.Series:
    """Identify columns whose OHLC fields share a plausible price scale."""
    finite = high.notna() & low.notna() & close.notna() & (close > 0)
    outside = (
        (close > high * (1.0 + tolerance))
        | (close < low * (1.0 - tolerance))
    ) & finite
    observations = finite.sum().replace(0, np.nan)
    outside_share = outside.sum() / observations
    return outside_share.fillna(1.0) <= 0.01


def _load_inputs(data, dates, universe) -> dict[str, pd.DataFrame] | None:
    frames = {}
    for field in ("high", "low", "close"):
        frame = data.get(field, dates, universe)
        if frame is None or frame.empty:
            return None
        frames[field] = (
            frame.reindex(index=dates, columns=universe)
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
        )
    valid = (
        (frames["high"] > 0)
        & (frames["low"] > 0)
        & (frames["close"] > 0)
        & (frames["high"] >= frames["low"])
    )
    coherent = _coherent_ohlc_columns(
        frames["high"], frames["low"], frames["close"]
    )
    valid.loc[:, ~coherent] = False
    return {field: frame.where(valid) for field, frame in frames.items()}


def _wilder_atr(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame
) -> pd.DataFrame:
    previous_close = close.shift(1)
    true_range = pd.DataFrame(
        np.maximum.reduce(
            [
                (high - low).to_numpy(),
                (high - previous_close).abs().to_numpy(),
                (low - previous_close).abs().to_numpy(),
            ]
        ),
        index=close.index,
        columns=close.columns,
    )
    return true_range.ewm(
        alpha=1.0 / ATR_WINDOW,
        adjust=False,
        min_periods=ATR_WINDOW,
    ).mean()


def _supertrend_components(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    atr = _wilder_atr(high, low, close)
    midpoint = (high + low) / 2.0
    basic_upper = midpoint + ATR_MULTIPLIER * atr
    basic_lower = midpoint - ATR_MULTIPLIER * atr

    shape = close.shape
    final_upper = np.full(shape, np.nan, dtype=float)
    final_lower = np.full(shape, np.nan, dtype=float)
    direction = np.full(shape, np.nan, dtype=float)
    active_line = np.full(shape, np.nan, dtype=float)

    upper_values = basic_upper.to_numpy(dtype=float)
    lower_values = basic_lower.to_numpy(dtype=float)
    close_values = close.to_numpy(dtype=float)
    midpoint_values = midpoint.to_numpy(dtype=float)

    for column in range(shape[1]):
        previous_upper = np.nan
        previous_lower = np.nan
        previous_close = np.nan
        previous_direction = np.nan

        for row in range(shape[0]):
            upper = upper_values[row, column]
            lower = lower_values[row, column]
            price = close_values[row, column]
            if not (np.isfinite(upper) and np.isfinite(lower) and np.isfinite(price)):
                continue

            if not np.isfinite(previous_direction):
                current_upper = upper
                current_lower = lower
                current_direction = 1.0 if price >= midpoint_values[row, column] else -1.0
            else:
                current_upper = (
                    upper
                    if upper < previous_upper or previous_close > previous_upper
                    else previous_upper
                )
                current_lower = (
                    lower
                    if lower > previous_lower or previous_close < previous_lower
                    else previous_lower
                )
                if previous_direction < 0 and price > current_upper:
                    current_direction = 1.0
                elif previous_direction > 0 and price < current_lower:
                    current_direction = -1.0
                else:
                    current_direction = previous_direction

            final_upper[row, column] = current_upper
            final_lower[row, column] = current_lower
            direction[row, column] = current_direction
            active_line[row, column] = (
                current_lower if current_direction > 0 else current_upper
            )
            previous_upper = current_upper
            previous_lower = current_lower
            previous_close = price
            previous_direction = current_direction

    def frame(values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(values, index=close.index, columns=close.columns)

    return frame(direction), frame(active_line), atr, frame(final_upper)


def _compute_output(spec: SupertrendFactorSpec, inputs) -> pd.DataFrame:
    close = inputs["close"]
    direction, active_line, atr, _ = _supertrend_components(
        inputs["high"], inputs["low"], close
    )
    if spec.output == "state":
        result = direction
    elif spec.output == "distance":
        result = (
            direction * (close - active_line).abs() / atr.replace(0, np.nan)
        ).clip(-DISTANCE_CLIP, DISTANCE_CLIP)
    elif spec.output == "flip":
        result = direction.diff() / 2.0
        result = result.where(result.abs() == 1.0, 0.0).where(direction.notna())
    else:  # pragma: no cover - guarded by immutable specs
        raise ValueError(f"unsupported Supertrend output: {spec.output}")
    return result.shift(1).replace([np.inf, -np.inf], np.nan)


def _make_factor_class(spec: SupertrendFactorSpec):
    class SupertrendFactor(Factor):
        name = spec.slug
        category = "trend"
        frequency = "daily"
        description = spec.description
        factor_spec = spec

        def dependencies(self) -> list[str]:
            return ["high", "low", "close"]

        def compute(self, data, dates, universe):
            inputs = _load_inputs(data, dates, universe)
            if inputs is None:
                return _empty(dates, universe)
            return _compute_output(self.factor_spec, inputs).reindex(
                index=dates, columns=universe
            )

    class_name = "".join(part.title() for part in spec.slug.split("_"))
    SupertrendFactor.__name__ = class_name
    SupertrendFactor.__qualname__ = class_name
    return register_user_factor(spec.slug, category="trend")(SupertrendFactor)


for _factor_spec in FACTOR_SPECS:
    globals()[_factor_spec.slug] = _make_factor_class(_factor_spec)


__all__ = ["FACTOR_SPECS", "SupertrendFactorSpec"] + [
    spec.slug for spec in FACTOR_SPECS
]
