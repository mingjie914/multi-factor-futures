"""Fail-closed factor timing contracts used by formal research."""
from __future__ import annotations

from typing import Iterable

from core.period import (
    PeriodContext,
    holding_periods_for_window,
    parse_slug_window,
)


def normalise_frequency(value: object) -> str:
    """Return the canonical frequency name used by ``PeriodContext``."""
    return PeriodContext.from_string(str(value or "daily")).unit.value


def default_validation_horizons(name: str) -> tuple[int, ...]:
    """Freeze the legacy window policy into an explicit registration contract."""
    window = parse_slug_window(str(name)) or "other"
    return tuple(holding_periods_for_window(window))


def bind_factor_contract(cls: type, name: str) -> type:
    """Validate and freeze frequency/horizon metadata on a factor class."""
    cls.frequency = normalise_frequency(getattr(cls, "frequency", "daily"))
    declared = getattr(cls, "validation_horizons", ())
    if not declared:
        declared = default_validation_horizons(name)
    horizons = tuple(sorted({int(value) for value in declared}))
    if not horizons or any(value < 1 for value in horizons):
        raise ValueError(
            f"factor {name!r} validation_horizons must contain positive bars"
        )
    cls.validation_horizons = horizons
    cls.horizon_unit = "bars"
    return cls


def factor_validation_horizons(factor: object) -> tuple[int, ...]:
    horizons = tuple(
        sorted({int(value) for value in getattr(factor, "validation_horizons", ())})
    )
    if not horizons or any(value < 1 for value in horizons):
        raise ValueError(
            f"factor {getattr(factor, 'name', type(factor).__name__)!r} has no "
            "valid validation_horizons contract"
        )
    return horizons


def validate_factor_contract(
    factor: object,
    *,
    provider_frequency: object,
    requested_horizons: Iterable[int] | None = None,
) -> tuple[int, ...]:
    """Reject frequency or formal-horizon mismatches before computation."""
    factor_name = str(getattr(factor, "name", type(factor).__name__))
    factor_frequency = normalise_frequency(getattr(factor, "frequency", "daily"))
    data_frequency = normalise_frequency(provider_frequency)
    if factor_frequency != data_frequency:
        raise ValueError(
            f"factor {factor_name!r} frequency {factor_frequency!r} is incompatible "
            f"with data frequency {data_frequency!r}"
        )
    if requested_horizons is not None:
        declared = factor_validation_horizons(factor)
        requested = tuple(sorted({int(value) for value in requested_horizons}))
        if requested != declared:
            raise ValueError(
                f"factor {factor_name!r} requested horizons {requested} do not match "
                f"its frozen validation_horizons {declared}"
            )
        return declared
    return tuple(getattr(factor, "validation_horizons", ()))
