"""Lagged price-and-volume pressure consensus factors."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.interfaces import Factor
from core.sectors import sector_for
from factors.user import register_user_factor


@dataclass(frozen=True)
class FlowConsensusSpec:
    slug: str
    variant: str
    window: int
    description: str


_SPECS = (
    ("flow_consensus_equal_10d", "equal", 10),
    ("flow_consensus_equal_20d", "equal", 20),
    ("flow_consensus_equal_60d", "equal", 60),
    ("flow_consensus_price_20d", "price", 20),
    ("flow_consensus_price_60d", "price", 60),
    ("flow_consensus_volume_20d", "volume", 20),
    ("flow_consensus_volume_60d", "volume", 60),
    ("flow_consensus_agreement_20d", "agreement", 20),
    ("flow_consensus_agreement_60d", "agreement", 60),
    ("flow_consensus_sector_neutral_20d", "sector_neutral", 20),
    ("flow_consensus_low_vol_20d", "low_vol", 20),
    ("flow_consensus_dual_horizon", "dual_horizon", 20),
)


FACTOR_SPECS = tuple(
    FlowConsensusSpec(
        slug=slug,
        variant=variant,
        window=window,
        description=(
            f"Equal-weight rank consensus of lagged price and volume pressure "
            f"components ({variant}, {window}-bar anchor)"
        ),
    )
    for slug, variant, window in _SPECS
)


_DEPENDENCIES = ("open", "high", "low", "close", "volume")


def _empty(dates, universe) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=dates, columns=universe, dtype=float)


def _load_inputs(data, dates, universe):
    frames = {}
    for field in _DEPENDENCIES:
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


def _rank(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, pct=True, method="average") - 0.5


def _components(inputs, window: int) -> dict[str, pd.DataFrame]:
    open_price = inputs["open"].where(inputs["open"] > 0)
    high = inputs["high"]
    low = inputs["low"]
    close = inputs["close"].where(inputs["close"] > 0)
    volume = inputs["volume"].where(inputs["volume"] > 0)
    min_periods = window

    log_return = np.log(close).diff()
    intraday = np.log(close / open_price).rolling(
        window, min_periods=min_periods
    ).mean()
    price_range = (high - low).where((high - low) > 0)
    clv_window = max(window, 60)
    clv = ((2.0 * close - high - low) / price_range).clip(-1.0, 1.0)
    clv = clv.rolling(clv_window, min_periods=clv_window).mean()
    direction = np.sign(log_return).rolling(
        window, min_periods=min_periods
    ).mean()

    up_volume = volume.where(log_return > 0, 0.0).rolling(
        window, min_periods=min_periods
    ).sum()
    down_volume = volume.where(log_return < 0, 0.0).rolling(
        window, min_periods=min_periods
    ).sum()
    total_directional = up_volume + down_volume
    volume_balance = (up_volume - down_volume) / total_directional.replace(
        0.0, np.nan
    )
    signed_volume = (np.sign(log_return) * volume).rolling(
        window, min_periods=min_periods
    ).sum() / volume.rolling(window, min_periods=min_periods).sum().replace(
        0.0, np.nan
    )
    volume_median = volume.rolling(window, min_periods=min_periods).median()
    relative_volume = np.log(volume / volume_median.replace(0.0, np.nan)).rolling(
        max(3, window // 5), min_periods=max(3, window // 5)
    ).mean()

    return {
        "intraday": _rank(intraday),
        "clv": _rank(clv),
        "direction": _rank(direction),
        "volume_balance": _rank(volume_balance),
        "signed_volume": _rank(signed_volume),
        "relative_volume": _rank(relative_volume),
    }


def _mean_components(
    components: dict[str, pd.DataFrame], names: tuple[str, ...]
) -> pd.DataFrame:
    return sum((components[name] for name in names)) / float(len(names))


def _sector_neutral(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    groups: dict[str, list[str]] = {}
    for instrument in frame.columns:
        groups.setdefault(sector_for(str(instrument)), []).append(instrument)
    for instruments in groups.values():
        if len(instruments) < 2:
            result[instruments] = np.nan
            continue
        result[instruments] = frame[instruments].sub(
            frame[instruments].median(axis=1), axis=0
        )
    return result


def _equal_consensus(inputs, window: int) -> tuple[pd.DataFrame, dict]:
    components = _components(inputs, window)
    names = tuple(components)
    return _mean_components(components, names), components


def _compute_spec(spec: FlowConsensusSpec, inputs) -> pd.DataFrame:
    if spec.variant == "dual_horizon":
        fast, _ = _equal_consensus(inputs, 10)
        slow, _ = _equal_consensus(inputs, 60)
        return (fast + slow) / 2.0

    equal, components = _equal_consensus(inputs, spec.window)
    if spec.variant == "equal":
        return equal
    if spec.variant == "price":
        return _mean_components(
            components, ("intraday", "clv", "direction")
        )
    if spec.variant == "volume":
        return _mean_components(
            components, ("volume_balance", "signed_volume", "relative_volume")
        )
    if spec.variant == "agreement":
        signs = sum(np.sign(component) for component in components.values())
        agreement = signs.abs() / float(len(components))
        return equal * agreement
    if spec.variant == "sector_neutral":
        return _sector_neutral(equal)
    if spec.variant == "low_vol":
        close = inputs["close"].where(inputs["close"] > 0)
        realized_vol = np.log(close).diff().rolling(
            spec.window, min_periods=spec.window
        ).std(ddof=0)
        low_vol_weight = 1.5 - realized_vol.rank(axis=1, pct=True)
        return equal * low_vol_weight
    raise ValueError(f"unsupported flow-consensus variant: {spec.variant}")


def _make_factor_class(spec: FlowConsensusSpec):
    class FlowConsensusFactor(Factor):
        name = spec.slug
        category = "liquidity_flow"
        frequency = "daily"
        description = spec.description
        factor_spec = spec

        def dependencies(self) -> list[str]:
            return list(_DEPENDENCIES)

        def compute(self, data, dates, universe):
            inputs = _load_inputs(data, dates, universe)
            if inputs is None:
                return _empty(dates, universe)
            return _compute_spec(self.factor_spec, inputs).reindex(
                index=dates, columns=universe
            ).replace([np.inf, -np.inf], np.nan)

    class_name = "".join(part.title() for part in spec.slug.split("_"))
    FlowConsensusFactor.__name__ = class_name
    FlowConsensusFactor.__qualname__ = class_name
    return register_user_factor(spec.slug, category="liquidity_flow")(
        FlowConsensusFactor
    )


for _factor_spec in FACTOR_SPECS:
    globals()[_factor_spec.slug] = _make_factor_class(_factor_spec)


__all__ = ["FACTOR_SPECS", "FlowConsensusSpec"] + [
    spec.slug for spec in FACTOR_SPECS
]
