"""Vectorized feature construction over time-by-instrument market panels."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Set

import numpy as np
import pandas as pd

from factor_mining.api import FeatureConfig


_EPS = 1e-12


def _safe_div(numerator: pd.DataFrame, denominator: pd.DataFrame) -> pd.DataFrame:
    return numerator / denominator.where(denominator.abs() > _EPS)


def _minimum_periods(window: int) -> int:
    return max(2, min(int(window), max(3, int(window) // 2)))


@dataclass(frozen=True)
class FeatureSet:
    index: pd.DatetimeIndex
    symbols: pd.Index
    values: Mapping[str, np.ndarray]
    raw_dependencies: Mapping[str, frozenset[str]]
    lookbacks: Mapping[str, int]
    dtype: str = "float32"

    def __post_init__(self) -> None:
        expected = (len(self.index), len(self.symbols))
        if not self.values:
            raise ValueError("feature set cannot be empty")
        for name, value in self.values.items():
            if np.asarray(value).shape != expected:
                raise ValueError(
                    f"feature {name!r} has shape {np.asarray(value).shape}, "
                    f"expected {expected}"
                )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.values))

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.index), len(self.symbols)

    def frame(self, name: str) -> pd.DataFrame:
        if name not in self.values:
            raise KeyError(f"unknown feature: {name}")
        return pd.DataFrame(self.values[name], index=self.index, columns=self.symbols)

    def dependencies_for(self, names: Iterable[str]) -> tuple[str, ...]:
        fields: Set[str] = set()
        for name in names:
            fields.update(self.raw_dependencies.get(name, ()))
        return tuple(sorted(fields))

    def lookback_for(self, names: Iterable[str]) -> int:
        return max((int(self.lookbacks.get(name, 0)) for name in names), default=0)


class FeatureEngine:
    """Build a reusable feature vocabulary once per mining/validation batch."""

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()

    def build_all_terminals(
        self, panels: Mapping[str, pd.DataFrame]
    ) -> FeatureSet:
        """Build the complete configured terminal vocabulary for a GP run."""

        return self.build(panels, required_features=None)

    def build(
        self,
        panels: Mapping[str, pd.DataFrame],
        *,
        required_features: Iterable[str] | None = None,
    ) -> FeatureSet:
        close = panels.get("close")
        if close is None or close.empty:
            raise ValueError("close panel is required and cannot be empty")
        index = pd.DatetimeIndex(close.index)
        symbols = pd.Index(close.columns)
        aligned: Dict[str, pd.DataFrame] = {}
        for field in self.config.raw_fields:
            panel = panels.get(field)
            if panel is None or panel.empty:
                continue
            frame = panel.reindex(index=index, columns=symbols).astype(float)
            if not frame.isna().all().all():
                aligned[field] = frame
        if "close" not in aligned:
            aligned["close"] = close.reindex(index=index, columns=symbols).astype(float)

        values: Dict[str, np.ndarray] = {}
        dependencies: Dict[str, frozenset[str]] = {}
        lookbacks: Dict[str, int] = {}
        required = (
            None if required_features is None
            else frozenset(str(name) for name in required_features)
        )
        if required is not None and not required:
            raise ValueError("required_features cannot be empty")
        stored_bytes = 0
        memory_limit = int(self.config.max_feature_memory_mb) * 1024 * 1024

        def add(name: str, frame: pd.DataFrame, deps: Sequence[str], lookback: int = 0) -> None:
            nonlocal stored_bytes
            if required is not None and name not in required:
                return
            result = frame.reindex(index=index, columns=symbols).replace([np.inf, -np.inf], np.nan)
            array = np.asarray(result, dtype=self.config.dtype)
            projected = stored_bytes + array.nbytes
            if projected > memory_limit:
                raise MemoryError(
                    "feature matrix exceeds max_feature_memory_mb="
                    f"{self.config.max_feature_memory_mb}; shorten the date range, "
                    "reduce windows/features, or explicitly raise the budget"
                )
            array.setflags(write=False)
            values[name] = array
            dependencies[name] = frozenset(deps)
            lookbacks[name] = int(lookback)
            stored_bytes = projected

        for field, frame in aligned.items():
            add(field, frame, (field,), 0)

        close = aligned["close"]
        open_price = aligned.get(
            "open", pd.DataFrame(np.nan, index=index, columns=symbols)
        )
        high = aligned.get("high", pd.DataFrame(np.nan, index=index, columns=symbols))
        low = aligned.get("low", pd.DataFrame(np.nan, index=index, columns=symbols))
        previous_close = close.shift(1)
        one_return = _safe_div(close, previous_close) - 1.0
        add("return_1p", one_return, ("close",), 1)
        add("log_return_1p", np.log(close.where(close > 0)).diff(), ("close",), 1)
        add(
            "open_gap_1p",
            _safe_div(open_price, previous_close) - 1.0,
            ("open", "close"),
            1,
        )
        add(
            "intrabar_return_1p",
            _safe_div(close, open_price) - 1.0,
            ("open", "close"),
            0,
        )
        add(
            "range_1p",
            _safe_div(high - low, previous_close),
            ("high", "low", "close"),
            1,
        )
        add("body_1p", _safe_div(close - open_price, previous_close), ("open", "close"), 1)
        upper_body = pd.DataFrame(
            np.maximum(open_price, close), index=index, columns=symbols
        )
        lower_body = pd.DataFrame(
            np.minimum(open_price, close), index=index, columns=symbols
        )
        add(
            "upper_shadow_1p",
            _safe_div(high - upper_body, previous_close),
            ("open", "high", "close"),
            1,
        )
        add(
            "lower_shadow_1p",
            _safe_div(lower_body - low, previous_close),
            ("open", "low", "close"),
            1,
        )
        add(
            "close_location_1p",
            _safe_div(close - low, high - low) - 0.5,
            ("high", "low", "close"),
            0,
        )

        for field in ("volume", "amount", "oi"):
            frame = aligned.get(field)
            if frame is None:
                continue
            add(f"log1p_{field}", np.log1p(frame.clip(lower=0)), (field,), 0)
            add(f"{field}_change_1p", frame.pct_change(fill_method=None), (field,), 1)
        if "volume" in aligned and "oi" in aligned:
            add(
                "volume_oi_ratio_1p",
                _safe_div(aligned["volume"], aligned["oi"]),
                ("volume", "oi"),
                0,
            )
        if "amount" in aligned and "volume" in aligned:
            add(
                "vwap_proxy_1p",
                _safe_div(aligned["amount"], aligned["volume"]),
                ("amount", "volume"),
                0,
            )

        for horizon in sorted(set(int(value) for value in self.config.feature_horizons)):
            if horizon == 1:
                continue
            name = f"return_{horizon}p"
            if required is None or name in required:
                add(name, close.pct_change(horizon, fill_method=None), ("close",), horizon)
            for field in ("volume", "amount", "oi"):
                if field in aligned:
                    name = f"{field}_change_{horizon}p"
                    if required is None or name in required:
                        add(
                            name,
                            aligned[field].pct_change(horizon, fill_method=None),
                            (field,), horizon,
                        )

        for lag in sorted(set(int(value) for value in self.config.lag_steps)):
            for field in ("close", "volume", "amount", "oi"):
                if field in aligned:
                    name = f"{field}_lag_{lag}p"
                    if required is None or name in required:
                        add(name, aligned[field].shift(lag), (field,), lag)

        true_range = pd.DataFrame(
            np.maximum.reduce([
                (high - low).to_numpy(),
                (high - previous_close).abs().to_numpy(),
                (low - previous_close).abs().to_numpy(),
            ]),
            index=index,
            columns=symbols,
        )
        negative_return = one_return.clip(upper=0)
        positive_return = one_return.clip(lower=0)
        amihud_base = None
        if "amount" in aligned:
            amihud_base = _safe_div(one_return.abs(), aligned["amount"].abs())

        for window in sorted(set(int(value) for value in self.config.rolling_windows)):
            if required is not None and not _requires_rolling_window(required, window):
                continue
            min_periods = _minimum_periods(window)
            close_mean = close.rolling(window, min_periods=min_periods).mean()
            close_std = close.rolling(window, min_periods=min_periods).std()
            add(f"close_mean_{window}p", close_mean, ("close",), window)
            add(
                f"close_ma_gap_{window}p",
                _safe_div(close, close_mean) - 1.0,
                ("close",),
                window,
            )
            close_ema = close.ewm(
                span=window, adjust=False, min_periods=min_periods
            ).mean()
            add(
                f"close_ema_gap_{window}p",
                _safe_div(close, close_ema) - 1.0,
                ("close",),
                window,
            )
            add(
                f"boll_position_{window}p",
                _safe_div(close - close_mean, close_std),
                ("close",),
                window,
            )
            add(
                f"boll_width_{window}p",
                _safe_div(4.0 * close_std, close_mean.abs()),
                ("close",),
                window,
            )
            add(
                f"realized_vol_{window}p",
                one_return.rolling(window, min_periods=min_periods).std(),
                ("close",),
                window,
            )
            add(
                f"up_semivariance_{window}p",
                positive_return.pow(2).rolling(
                    window, min_periods=min_periods
                ).mean(),
                ("close",),
                window,
            )
            add(
                f"down_semivariance_{window}p",
                negative_return.pow(2).rolling(
                    window, min_periods=min_periods
                ).mean(),
                ("close",),
                window,
            )
            add(
                f"atr_ratio_{window}p",
                _safe_div(
                    true_range.rolling(window, min_periods=min_periods).mean(),
                    close.abs(),
                ),
                ("high", "low", "close"),
                window,
            )

            delta = close.diff()
            gain = delta.clip(lower=0).rolling(window, min_periods=min_periods).mean()
            loss = (-delta.clip(upper=0)).rolling(window, min_periods=min_periods).mean()
            rs = _safe_div(gain, loss)
            add(f"rsi_{window}p", 100.0 - 100.0 / (1.0 + rs), ("close",), window)

            for field in ("volume", "amount", "oi"):
                if field not in aligned:
                    continue
                rolling_mean = aligned[field].rolling(window, min_periods=min_periods).mean()
                add(f"{field}_mean_{window}p", rolling_mean, (field,), window)
                add(
                    f"{field}_relative_{window}p",
                    _safe_div(aligned[field], rolling_mean) - 1.0,
                    (field,),
                    window,
                )

            if "volume" in aligned:
                volume = aligned["volume"]
                rolling_volume = volume.rolling(window, min_periods=min_periods)
                add(
                    f"volume_concentration_{window}p",
                    _safe_div(rolling_volume.max(), rolling_volume.mean()),
                    ("volume",),
                    window,
                )
                add(
                    f"price_volume_corr_{window}p",
                    one_return.rolling(window, min_periods=min_periods).corr(
                        volume.pct_change(fill_method=None)
                    ),
                    ("close", "volume"),
                    window,
                )
            if amihud_base is not None:
                add(
                    f"amihud_{window}p",
                    amihud_base.rolling(window, min_periods=min_periods).mean(),
                    ("close", "amount"),
                    window,
                )
            if self.config.include_distribution:
                rolling_return = one_return.rolling(window, min_periods=min_periods)
                add(f"return_skew_{window}p", rolling_return.skew(), ("close",), window)
                add(f"return_kurt_{window}p", rolling_return.kurt(), ("close",), window)

        if self.config.include_technicals:
            if required is None or "macd_diff_12_26_9" in required:
                fast = close.ewm(span=12, adjust=False, min_periods=6).mean()
                slow = close.ewm(span=26, adjust=False, min_periods=13).mean()
                macd = fast - slow
                signal = macd.ewm(span=9, adjust=False, min_periods=5).mean()
                add(
                    "macd_diff_12_26_9",
                    _safe_div(macd - signal, close.abs()),
                    ("close",),
                    35,
                )

        if required is not None:
            missing = sorted(required - set(values))
            if missing:
                raise KeyError(f"requested features are unavailable: {missing}")

        return FeatureSet(
            index=index,
            symbols=symbols,
            values=values,
            raw_dependencies=dependencies,
            lookbacks=lookbacks,
            dtype=self.config.dtype,
        )


_ROLLING_PREFIXES = (
    "close_mean_", "close_ma_gap_", "close_ema_gap_", "boll_position_",
    "boll_width_", "realized_vol_", "up_semivariance_", "down_semivariance_",
    "atr_ratio_", "rsi_", "volume_mean_", "volume_relative_", "amount_mean_",
    "amount_relative_", "oi_mean_", "oi_relative_", "volume_concentration_",
    "price_volume_corr_", "amihud_", "return_skew_", "return_kurt_",
)


def _requires_rolling_window(required: frozenset[str], window: int) -> bool:
    suffix = f"_{window}p"
    return any(
        name.startswith(_ROLLING_PREFIXES) and name.endswith(suffix)
        for name in required
    )
