"""Low-churn futures adaptations of public WQ101 and GTJA191 formulas."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.interfaces import Factor
from factors.user import register_user_factor


@dataclass(frozen=True)
class ExternalFormulaSpec:
    slug: str
    formula: str
    source: str
    description: str


FACTOR_SPECS = (
    ExternalFormulaSpec(
        "adapted_wq002_flow_corr_20d", "wq002", "WorldQuant Alpha 101 #2",
        "Negative correlation of ranked volume change and intraday return",
    ),
    ExternalFormulaSpec(
        "adapted_wq012_signed_volume_reversal_20d", "wq012", "WorldQuant Alpha 101 #12",
        "Signed volume-change reversal accumulated over 20 bars",
    ),
    ExternalFormulaSpec(
        "adapted_wq030_direction_volume_20d", "wq030", "WorldQuant Alpha 101 #30",
        "Short-run direction persistence scaled by the 5-to-20 bar volume ratio",
    ),
    ExternalFormulaSpec(
        "adapted_wq035_triple_rank_32d", "wq035", "WorldQuant Alpha 101 #35",
        "Joint time ranks of volume, range-adjusted price and return",
    ),
    ExternalFormulaSpec(
        "adapted_wq043_relative_volume_reversal_20d", "wq043", "WorldQuant Alpha 101 #43",
        "Relative-volume time rank times medium-horizon return reversal rank",
    ),
    ExternalFormulaSpec(
        "adapted_wq055_clv_volume_corr_20d", "wq055", "WorldQuant Alpha 101 #55",
        "Negative correlation between ranked close location and ranked volume",
    ),
    ExternalFormulaSpec(
        "adapted_gtja003_true_range_pressure_20d", "gtja003", "Guotai Junan 191 #3",
        "Directional price movement accumulated relative to true range",
    ),
    ExternalFormulaSpec(
        "adapted_gtja009_range_liquidity_flow_20d", "gtja009", "Guotai Junan 191 #9",
        "Midpoint movement times range, scaled by relative liquidity",
    ),
    ExternalFormulaSpec(
        "adapted_gtja011_volume_clv_120d", "gtja011", "Guotai Junan 191 #11",
        "Long-horizon relative-volume-weighted close-location pressure",
    ),
    ExternalFormulaSpec(
        "adapted_gtja052_up_down_range_26d", "gtja052", "Guotai Junan 191 #52",
        "Log ratio of accumulated upside and downside true-range pressure",
    ),
    ExternalFormulaSpec(
        "adapted_gtja076_illiquidity_instability_60d", "gtja076", "Guotai Junan 191 #76",
        "Coefficient of variation of return per unit of relative volume",
    ),
    ExternalFormulaSpec(
        "adapted_gtja191_volume_low_corr_20d", "gtja191", "Guotai Junan 191 #191",
        "Relative-volume/low-price correlation plus normalized midpoint pressure",
    ),
)


_DEPENDENCIES = ("open", "high", "low", "close", "volume")


def _empty(dates, universe) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=dates, columns=universe, dtype=float)


def _load_inputs(data, dates, universe):
    frames = {}
    for field in _DEPENDENCIES:
        frame = data.get(field, dates, universe)
        if (
            frame is None
            or frame.empty
            or not frame.notna().to_numpy().any()
        ):
            return None
        frames[field] = (
            frame.reindex(index=dates, columns=universe)
            .astype(float)
            .shift(1)
            .replace([np.inf, -np.inf], np.nan)
        )
    for field in ("open", "high", "low", "close", "volume"):
        frames[field] = frames[field].where(frames[field] > 0)
    return frames


def _rank_pct(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, pct=True, method="average")


def _ts_rank(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=window).rank(pct=True)


def _corr(left: pd.DataFrame, right: pd.DataFrame, window: int) -> pd.DataFrame:
    return left.rolling(window, min_periods=window).corr(right)


def _true_range(inputs) -> pd.DataFrame:
    high, low, close = inputs["high"], inputs["low"], inputs["close"]
    previous = close.shift(1)
    return pd.DataFrame(
        np.maximum.reduce([
            (high - low).to_numpy(),
            (high - previous).abs().to_numpy(),
            (low - previous).abs().to_numpy(),
        ]),
        index=close.index,
        columns=close.columns,
    )


def _weekly_hold(frame: pd.DataFrame) -> pd.DataFrame:
    periods = frame.index.to_period("W-FRI")
    is_last = np.r_[periods[:-1] != periods[1:], True]
    sampled = frame.copy()
    sampled.iloc[~is_last, :] = np.nan
    return sampled.ffill()


def _low_churn(frame: pd.DataFrame) -> pd.DataFrame:
    smoothed = frame.ewm(span=20, adjust=False, min_periods=10).mean()
    return _weekly_hold(smoothed)


def _compute_formula(formula: str, inputs) -> pd.DataFrame:
    open_price = inputs["open"]
    high = inputs["high"]
    low = inputs["low"]
    close = inputs["close"]
    volume = inputs["volume"]
    log_close = np.log(close)
    returns = log_close.diff()
    log_volume = np.log(volume)
    relative_volume_20 = volume / volume.rolling(20, min_periods=20).mean()
    price_range = (high - low).where((high - low) > 0)
    clv = ((2.0 * close - high - low) / price_range).clip(-1.0, 1.0)

    if formula == "wq002":
        return -_corr(
            _rank_pct(log_volume.diff(2)),
            _rank_pct(np.log(close / open_price)),
            20,
        )
    if formula == "wq012":
        return (-returns * np.sign(log_volume.diff())).rolling(
            20, min_periods=20
        ).mean()
    if formula == "wq030":
        direction = (
            np.sign(returns) + np.sign(returns.shift(1)) + np.sign(returns.shift(2))
        )
        volume_ratio = (
            volume.rolling(5, min_periods=5).sum()
            / volume.rolling(20, min_periods=20).sum()
        )
        return (1.0 - _rank_pct(direction)) * volume_ratio
    if formula == "wq035":
        normalized_price = (close + high - low) / close
        return (
            _ts_rank(volume, 32)
            * (1.0 - _ts_rank(normalized_price, 16))
            * (1.0 - _ts_rank(returns, 32))
        )
    if formula == "wq043":
        return _ts_rank(relative_volume_20, 20) * _ts_rank(
            -log_close.diff(7), 8
        )
    if formula == "wq055":
        return -_corr(_rank_pct(clv), _rank_pct(log_volume), 20)
    if formula == "gtja003":
        previous = close.shift(1)
        down_reference = pd.DataFrame(
            np.maximum(high.to_numpy(), previous.to_numpy()),
            index=close.index,
            columns=close.columns,
        )
        up_reference = pd.DataFrame(
            np.minimum(low.to_numpy(), previous.to_numpy()),
            index=close.index,
            columns=close.columns,
        )
        movement = (close - up_reference).where(
            close > previous, close - down_reference
        ).where(close != previous, 0.0)
        numerator = movement.rolling(20, min_periods=20).sum()
        denominator = _true_range(inputs).rolling(20, min_periods=20).sum()
        return numerator / denominator.replace(0.0, np.nan)
    if formula == "gtja009":
        midpoint = (high + low) / 2.0
        midpoint_return = midpoint.diff() / close.shift(1)
        range_ratio = price_range / close
        liquidity_scale = relative_volume_20.clip(lower=1e-6).pow(0.5)
        return (midpoint_return * range_ratio / liquidity_scale).ewm(
            alpha=2.0 / 7.0, adjust=False, min_periods=7
        ).mean()
    if formula == "gtja011":
        relative_volume_60 = volume / volume.rolling(60, min_periods=60).median()
        numerator = (clv * relative_volume_60).rolling(
            120, min_periods=120
        ).sum()
        denominator = relative_volume_60.rolling(
            120, min_periods=120
        ).sum()
        return numerator / denominator.replace(0.0, np.nan)
    if formula == "gtja052":
        previous_typical = ((high + low + close) / 3.0).shift(1)
        upside = (high - previous_typical).clip(lower=0.0)
        downside = (previous_typical - low).clip(lower=0.0)
        up_sum = upside.rolling(26, min_periods=26).sum()
        down_sum = downside.rolling(26, min_periods=26).sum()
        return np.log(up_sum / down_sum.replace(0.0, np.nan))
    if formula == "gtja076":
        relative_volume_60 = volume / volume.rolling(60, min_periods=60).median()
        illiquidity = returns.abs() / relative_volume_60.replace(0.0, np.nan)
        mean = illiquidity.rolling(60, min_periods=60).mean()
        std = illiquidity.rolling(60, min_periods=60).std(ddof=0)
        return std / mean.replace(0.0, np.nan)
    if formula == "gtja191":
        low_location = low / close - 1.0
        volume_level = relative_volume_20.rolling(20, min_periods=20).mean()
        midpoint_pressure = ((high + low) / 2.0 - close) / close
        return _corr(volume_level, low_location, 20) + midpoint_pressure
    raise ValueError(f"unsupported external formula: {formula}")


def _make_factor_class(spec: ExternalFormulaSpec):
    class ExternalFormulaFactor(Factor):
        name = spec.slug
        category = "external_formula_adapted"
        frequency = "daily"
        description = f"{spec.description}; adapted from {spec.source}"
        factor_spec = spec

        def dependencies(self) -> list[str]:
            return list(_DEPENDENCIES)

        def compute(self, data, dates, universe):
            inputs = _load_inputs(data, dates, universe)
            if inputs is None:
                return _empty(dates, universe)
            raw = _compute_formula(self.factor_spec.formula, inputs)
            return _low_churn(raw).reindex(
                index=dates, columns=universe
            ).replace([np.inf, -np.inf], np.nan)

    class_name = "".join(part.title() for part in spec.slug.split("_"))
    ExternalFormulaFactor.__name__ = class_name
    ExternalFormulaFactor.__qualname__ = class_name
    return register_user_factor(
        spec.slug, category="external_formula_adapted"
    )(ExternalFormulaFactor)


for _factor_spec in FACTOR_SPECS:
    globals()[_factor_spec.slug] = _make_factor_class(_factor_spec)


__all__ = ["FACTOR_SPECS", "ExternalFormulaSpec"] + [
    spec.slug for spec in FACTOR_SPECS
]
