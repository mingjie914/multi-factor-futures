"""Registered futures adapters for the complete WQ101 and GTJA191 sets.

Formula implementations are imported from the MIT-licensed ``ta_cn`` package.
This module supplies point-in-time futures fields and fails closed where a
stock-specific input has no defensible futures analogue.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
import os
import re

import numpy as np
import pandas as pd

from core.interfaces import Factor
from factors.user import register_user_factor


os.environ["TA_CN_MODE"] = "WIDE"


@dataclass(frozen=True)
class TaCnFormulaSpec:
    slug: str
    library: str
    number: int
    available: bool
    unavailable_reason: str | None
    description: str


_UNAVAILABLE = {
    ("wq101", 48): "requires stock subindustry membership",
    ("wq101", 56): "requires stock market capitalization",
    ("wq101", 67): "requires stock subindustry membership",
    ("wq101", 90): "requires stock subindustry membership",
    ("wq101", 100): "requires stock subindustry membership",
    ("gtja191", 30): "requires stock MKT, SMB, and HML returns",
    ("gtja191", 165): "upstream SUMAC formula has ambiguous scalar MAX/MIN semantics",
    ("gtja191", 183): "upstream SUMAC formula has ambiguous scalar MAX/MIN semantics",
}


FACTOR_SPECS = tuple(
    TaCnFormulaSpec(
        slug=f"{library}_alpha{number:03d}",
        library=library,
        number=number,
        available=(library, number) not in _UNAVAILABLE,
        unavailable_reason=_UNAVAILABLE.get((library, number)),
        description=(
            f"Futures adaptation of {library.upper()} alpha {number:03d}; "
            "all market inputs lagged one valid bar; VWAP uses OHLC typical price"
        ),
    )
    for library, count in (("wq101", 101), ("gtja191", 191))
    for number in range(1, count + 1)
)


def _empty(dates, universe) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=dates, columns=universe, dtype=float)


def _lag_valid_bars(frame: pd.DataFrame, dates, universe) -> pd.DataFrame:
    aligned = frame.reindex(index=dates, columns=universe).astype(float)
    result = _empty(dates, universe)
    for ticker in universe:
        series = aligned[ticker].dropna().shift(1)
        result.loc[series.index, ticker] = series
    return result.replace([np.inf, -np.inf], np.nan)


def _request_key(dates, universe) -> tuple:
    index = pd.Index(dates)
    return (
        len(index),
        index[0] if len(index) else None,
        index[-1] if len(index) else None,
        hash(pd.util.hash_pandas_object(index, index=False).to_numpy().tobytes()),
        tuple(str(item) for item in universe),
    )


def _broadcast(series: pd.Series, columns) -> pd.DataFrame:
    values = np.repeat(series.to_numpy()[:, None], len(columns), axis=1)
    return pd.DataFrame(values, index=series.index, columns=columns)


def _formula_inputs(data, dates, universe) -> dict[str, pd.DataFrame]:
    key = _request_key(dates, universe)
    cache = getattr(data, "_ta_cn_formula_input_cache", None)
    if isinstance(cache, tuple) and cache[0] == key:
        return cache[1]

    raw = {}
    for field in ("open", "high", "low", "close", "volume", "amount"):
        frame = data.get(field, dates, universe)
        if frame is None or frame.empty:
            raw[field] = _empty(dates, universe)
        else:
            raw[field] = _lag_valid_bars(frame, dates, universe)

    open_price = raw["open"].where(raw["open"] > 0.0)
    high = raw["high"].where(raw["high"] > 0.0)
    low = raw["low"].where(raw["low"] > 0.0)
    close = raw["close"].where(raw["close"] > 0.0)
    volume = raw["volume"].where(raw["volume"] > 0.0)
    amount = raw["amount"]
    returns = close.pct_change(fill_method=None)
    vwap = (open_price + high + low + close) / 4.0

    inputs = {
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "returns": returns,
        "vwap": vwap,
        "OPEN": open_price,
        "HIGH": high,
        "LOW": low,
        "CLOSE": close,
        "VOLUME": volume,
        "AMOUNT": amount,
        "RET": returns,
        "VWAP": vwap,
    }
    for window in (5, 10, 15, 20, 30, 40, 50, 60, 81, 120, 150, 180):
        inputs[f"adv{window}"] = volume.rolling(
            window, min_periods=window
        ).mean()

    industry_getter = getattr(data, "get_industry", None)
    if callable(industry_getter):
        industry = industry_getter(dates, universe).reindex(
            index=dates, columns=universe
        )
    else:
        industry = pd.DataFrame(index=dates, columns=universe, dtype=object)
    industry = industry.copy()
    industry.index = pd.RangeIndex(len(industry.index))
    industry.columns = pd.RangeIndex(len(industry.columns))
    inputs["industry"] = industry
    inputs["sector"] = industry

    market_return = returns.mean(axis=1, skipna=True).fillna(0.0)
    benchmark_close = 100.0 * (1.0 + market_return).cumprod()
    prior_benchmark = benchmark_close.shift(1)
    open_ratio = (open_price / close.shift(1)).mean(axis=1, skipna=True)
    benchmark_open = prior_benchmark * open_ratio.fillna(1.0)
    inputs["BANCHMARKINDEXCLOSE"] = _broadcast(
        benchmark_close, close.columns
    )
    inputs["BANCHMARKINDEXOPEN"] = _broadcast(
        benchmark_open, close.columns
    )

    previous_open = open_price.shift(1)
    inputs["DTM"] = (high - open_price).where(
        open_price > previous_open, 0.0
    ).where(
        open_price <= previous_open,
        pd.DataFrame(
            np.maximum(
                (high - open_price).to_numpy(),
                (open_price - previous_open).to_numpy(),
            ),
            index=dates,
            columns=universe,
        ),
    )
    inputs["DBM"] = (open_price - low).where(
        open_price < previous_open, 0.0
    ).where(
        open_price >= previous_open,
        pd.DataFrame(
            np.maximum(
                (open_price - low).to_numpy(),
                (open_price - previous_open).to_numpy(),
            ),
            index=dates,
            columns=universe,
        ),
    )
    try:
        setattr(data, "_ta_cn_formula_input_cache", (key, inputs))
    except Exception:
        pass
    return inputs


def _module_and_function(spec: TaCnFormulaSpec):
    module_name = (
        "ta_cn.alphas.alpha101"
        if spec.library == "wq101"
        else "ta_cn.alphas.alpha191"
    )
    module = importlib.import_module(module_name)
    return getattr(module, f"alpha_{spec.number:03d}")


def _dependencies(spec: TaCnFormulaSpec) -> list[str]:
    if not spec.available:
        return ["close"]
    function = _module_and_function(spec)
    arguments = set(inspect.signature(function).parameters) - {"kwargs"}
    dependencies = set()
    direct = {
        "open": "open", "high": "high", "low": "low", "close": "close",
        "volume": "volume", "OPEN": "open", "HIGH": "high", "LOW": "low",
        "CLOSE": "close", "VOLUME": "volume", "AMOUNT": "amount",
    }
    for argument in arguments:
        if argument in direct:
            dependencies.add(direct[argument])
        elif argument in {"returns", "RET"}:
            dependencies.add("close")
        elif argument in {"vwap", "VWAP"}:
            dependencies.update(("open", "high", "low", "close"))
        elif re.fullmatch(r"adv\d+", argument):
            dependencies.add("volume")
        elif argument.startswith("BANCHMARKINDEX"):
            dependencies.update(("open", "close"))
        elif argument in {"DTM", "DBM"}:
            dependencies.update(("open", "high", "low"))
    return sorted(dependencies)


def _make_factor_class(spec: TaCnFormulaSpec):
    class TaCnFormulaFactor(Factor):
        name = spec.slug
        category = f"external_formula_{spec.library}"
        frequency = "daily"
        description = spec.description
        factor_spec = spec
        research_tier = "candidate" if spec.available else "unavailable"
        unavailable_reason = spec.unavailable_reason

        def dependencies(self) -> list[str]:
            return _dependencies(spec)

        def compute(self, data, dates, universe):
            if not spec.available:
                return _empty(dates, universe)
            try:
                function = _module_and_function(spec)
                inputs = _formula_inputs(data, dates, universe)
                from ta_cn.utils_wide import WArr

                arguments = {
                    name: (
                        inputs[name]
                        if name in {"industry", "sector", "subindustry"}
                        else WArr.from_obj(inputs[name], "down")
                    )
                    for name in inspect.signature(function).parameters
                    if name != "kwargs"
                }
                raw = function(**arguments)
                if isinstance(raw, pd.DataFrame):
                    result = raw
                elif isinstance(raw, pd.Series):
                    result = raw.unstack() if isinstance(raw.index, pd.MultiIndex) else raw.to_frame()
                else:
                    raw_method = getattr(raw, "raw", None)
                    values = raw_method() if callable(raw_method) else raw
                    result = pd.DataFrame(values, index=dates, columns=universe)
                return result.reindex(index=dates, columns=universe).replace(
                    [np.inf, -np.inf], np.nan
                ).astype(float)
            except Exception as exc:
                raise RuntimeError(f"{spec.slug} formula computation failed") from exc

    class_name = "".join(part.title() for part in spec.slug.split("_"))
    TaCnFormulaFactor.__name__ = class_name
    TaCnFormulaFactor.__qualname__ = class_name
    return register_user_factor(
        spec.slug, category=f"external_formula_{spec.library}"
    )(TaCnFormulaFactor)


for _factor_spec in FACTOR_SPECS:
    globals()[_factor_spec.slug] = _make_factor_class(_factor_spec)


__all__ = ["FACTOR_SPECS", "TaCnFormulaSpec"] + [
    spec.slug for spec in FACTOR_SPECS
]
