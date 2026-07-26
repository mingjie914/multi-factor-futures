"""Point-in-time rebalance adapters for historically persistent SPEC factors."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.interfaces import Factor
from core.registry import get as registry_get
from factors.user import register_user_factor


@dataclass(frozen=True)
class LowChurnAdapterSpec:
    slug: str
    base_factor: str
    schedule: str
    description: str


_BASE_FACTORS = (
    "candle_pressure_120d_rank",
    "volume_weighted_clv_120d_rank",
    "vr_ratio_20d_stability",
    "directional_consistency_10d_smooth",
    "breakout_20d_z",
)


FACTOR_SPECS = tuple(
    LowChurnAdapterSpec(
        slug=f"low_churn_{base}_{schedule}",
        base_factor=base,
        schedule=schedule,
        description=(
            f"One-bar-lagged {base} held on a point-in-time {schedule} schedule"
        ),
    )
    for base in _BASE_FACTORS
    for schedule in ("weekly", "monthly")
) + tuple(
    LowChurnAdapterSpec(
        slug=f"low_churn_{base}_biweekly",
        base_factor=base,
        schedule="biweekly",
        description=(
            f"One-bar-lagged {base} held on even-ISO-week Fridays"
        ),
    )
    for base in _BASE_FACTORS[:2]
)


def _empty(dates, universe) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=dates, columns=universe, dtype=float)


def _rebalance_mask(dates: pd.DatetimeIndex, schedule: str) -> np.ndarray:
    if schedule == "weekly":
        return dates.weekday == 4
    if schedule == "biweekly":
        iso_week = dates.isocalendar().week.to_numpy(dtype=int)
        return (dates.weekday == 4) & (iso_week % 2 == 0)
    if schedule == "monthly":
        next_business_day = dates + pd.offsets.BDay(1)
        return next_business_day.month != dates.month
    raise ValueError(f"unsupported rebalance schedule: {schedule}")


def _hold(frame: pd.DataFrame, schedule: str) -> pd.DataFrame:
    sampled = frame.copy()
    sampled.iloc[~_rebalance_mask(frame.index, schedule), :] = np.nan
    return sampled.ffill()


def _make_factor_class(spec: LowChurnAdapterSpec):
    class LowChurnAdapter(Factor):
        name = spec.slug
        category = "low_churn_adapter"
        frequency = "daily"
        description = spec.description
        factor_spec = spec

        def dependencies(self) -> list[str]:
            base = registry_get("factor", self.factor_spec.base_factor)()
            return list(base.dependencies())

        def compute(self, data, dates, universe):
            if len(dates) == 0 or len(universe) == 0:
                return _empty(dates, universe)
            base = registry_get("factor", self.factor_spec.base_factor)()
            raw = base.compute(data, dates, universe)
            if raw is None or raw.empty:
                return _empty(dates, universe)
            lagged = raw.reindex(index=dates, columns=universe).shift(1)
            return _hold(lagged, self.factor_spec.schedule).replace(
                [np.inf, -np.inf], np.nan
            )

    class_name = "".join(part.title() for part in spec.slug.split("_"))
    LowChurnAdapter.__name__ = class_name
    LowChurnAdapter.__qualname__ = class_name
    return register_user_factor(spec.slug, category="low_churn_adapter")(
        LowChurnAdapter
    )


for _factor_spec in FACTOR_SPECS:
    globals()[_factor_spec.slug] = _make_factor_class(_factor_spec)


__all__ = ["FACTOR_SPECS", "LowChurnAdapterSpec"] + [
    spec.slug for spec in FACTOR_SPECS
]
