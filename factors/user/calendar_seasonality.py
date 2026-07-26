"""Point-in-time calendar-seasonality factors for daily futures data.

The factors use only observations from the same calendar month in prior
years. Current-year monthly observations are always excluded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.interfaces import Factor
from factors.user import register_user_factor


@dataclass(frozen=True)
class CalendarSeasonalitySpec:
    slug: str
    base: str
    years: int
    dependencies: tuple[str, ...]
    description: str


_MONTHLY_SPECS = (
    ("calendar_return_mean_3y", "return_mean", 3, ("close",)),
    ("calendar_return_mean_5y", "return_mean", 5, ("close",)),
    ("calendar_return_tstat_3y", "return_tstat", 3, ("close",)),
    ("calendar_return_tstat_5y", "return_tstat", 5, ("close",)),
    ("calendar_return_median_3y", "return_median", 3, ("close",)),
    ("calendar_return_positive_share_5y", "return_positive_share", 5, ("close",)),
    ("calendar_intraday_mean_3y", "intraday_mean", 3, ("open", "close")),
    ("calendar_intraday_mean_5y", "intraday_mean", 5, ("open", "close")),
    ("calendar_gap_mean_3y", "gap_mean", 3, ("open", "close")),
    ("calendar_clv_mean_3y", "clv_mean", 3, ("high", "low", "close")),
    ("calendar_volume_pressure_3y", "volume_pressure", 3, ("close", "volume")),
    ("calendar_volume_pressure_5y", "volume_pressure", 5, ("close", "volume")),
)


FACTOR_SPECS = tuple(
    CalendarSeasonalitySpec(
        slug=slug,
        base=base,
        years=years,
        dependencies=dependencies,
        description=(
            f"Point-in-time {base.replace('_', ' ')} for the current calendar "
            f"month estimated from the prior {years} same-month observations"
        ),
    )
    for slug, base, years, dependencies in _MONTHLY_SPECS
)


def _empty(dates, universe) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=dates, columns=universe, dtype=float)


def _load_lagged_inputs(data, dates, universe, dependencies):
    frames = {}
    for field in dependencies:
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


def _monthly_observation(base: str, inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    close = inputs["close"].where(inputs["close"] > 0)
    log_return = np.log(close).diff()
    periods = close.index.to_period("M")

    if base.startswith("return_"):
        return log_return.groupby(periods).sum(min_count=10)
    if base == "intraday_mean":
        open_price = inputs["open"].where(inputs["open"] > 0)
        intraday = np.log(close / open_price)
        return intraday.groupby(periods).mean()
    if base == "gap_mean":
        open_price = inputs["open"].where(inputs["open"] > 0)
        gap = np.log(open_price / close.shift(1))
        return gap.groupby(periods).mean()
    if base == "clv_mean":
        high = inputs["high"]
        low = inputs["low"]
        price_range = (high - low).where((high - low) > 0)
        clv = (2.0 * close - high - low) / price_range
        return clv.clip(-1.0, 1.0).groupby(periods).mean()
    if base == "volume_pressure":
        volume = inputs["volume"].where(inputs["volume"] > 0)
        signed_volume = np.sign(log_return) * volume
        numerator = signed_volume.groupby(periods).sum(min_count=10)
        denominator = volume.groupby(periods).sum(min_count=10)
        return numerator / denominator.replace(0.0, np.nan)
    raise ValueError(f"unsupported calendar-seasonality base: {base}")


def _same_month_history(
    monthly: pd.DataFrame, years: int, statistic: str
) -> pd.DataFrame:
    result = pd.DataFrame(np.nan, index=monthly.index, columns=monthly.columns)
    for month in range(1, 13):
        mask = monthly.index.month == month
        history = monthly.loc[mask].shift(1)
        rolling = history.rolling(years, min_periods=years)
        if statistic == "mean":
            values = rolling.mean()
        elif statistic == "median":
            values = rolling.median()
        elif statistic == "tstat":
            mean = rolling.mean()
            std = rolling.std(ddof=1)
            values = mean / (std / np.sqrt(float(years))).replace(0.0, np.nan)
        elif statistic == "positive_share":
            positive = history.gt(0).astype(float).where(history.notna())
            values = positive.rolling(years, min_periods=years).mean() - 0.5
        else:
            raise ValueError(f"unsupported calendar statistic: {statistic}")
        result.loc[mask] = values
    return result


def _compute_spec(
    spec: CalendarSeasonalitySpec,
    inputs: dict[str, pd.DataFrame],
    dates,
    universe,
) -> pd.DataFrame:
    monthly = _monthly_observation(spec.base, inputs)
    statistic = "mean"
    if spec.base.startswith("return_"):
        statistic = spec.base.removeprefix("return_")
    monthly_signal = _same_month_history(monthly, spec.years, statistic)
    requested_periods = pd.PeriodIndex(pd.DatetimeIndex(dates), freq="M")
    result = monthly_signal.reindex(requested_periods)
    result.index = dates
    return result.reindex(index=dates, columns=universe).replace(
        [np.inf, -np.inf], np.nan
    )


def _make_factor_class(spec: CalendarSeasonalitySpec):
    class CalendarSeasonalityFactor(Factor):
        name = spec.slug
        category = "seasonality"
        frequency = "daily"
        description = spec.description
        factor_spec = spec

        def dependencies(self) -> list[str]:
            return list(self.factor_spec.dependencies)

        def compute(self, data, dates, universe):
            inputs = _load_lagged_inputs(
                data, dates, universe, self.factor_spec.dependencies
            )
            if inputs is None:
                return _empty(dates, universe)
            return _compute_spec(
                self.factor_spec, inputs, dates, universe
            )

    class_name = "".join(part.title() for part in spec.slug.split("_"))
    CalendarSeasonalityFactor.__name__ = class_name
    CalendarSeasonalityFactor.__qualname__ = class_name
    return register_user_factor(spec.slug, category="seasonality")(
        CalendarSeasonalityFactor
    )


for _factor_spec in FACTOR_SPECS:
    globals()[_factor_spec.slug] = _make_factor_class(_factor_spec)


__all__ = ["CalendarSeasonalitySpec", "FACTOR_SPECS"] + [
    spec.slug for spec in FACTOR_SPECS
]
