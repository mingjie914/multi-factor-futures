"""Local-only market-data adapter and deterministic synthetic fixtures."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from factor_mining.api import FeatureConfig


@dataclass(frozen=True)
class LocalParquetSpec:
    root_path: Path
    datasets: Mapping[str, str] | None = None
    dominant_lag_days: int = 1
    schedule_buffer_days: int = 45
    eager_fields: bool = False
    panel_cache_entries: int = 1
    curve_cache_enabled: bool = False
    curve_cache_path: Path | None = None
    selected_cache_enabled: bool = False
    selected_cache_path: Path | None = None


class LocalParquetData:
    """Read the framework's published partitioned-Parquet store."""

    def __init__(self, spec: LocalParquetSpec):
        root = Path(spec.root_path).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        from data.parquet_source import ParquetFuturesSource

        config = {
            "root_path": str(root),
            "datasets": dict(spec.datasets or {}),
            "dominant_lag_days": int(spec.dominant_lag_days),
            "schedule_buffer_days": int(spec.schedule_buffer_days),
            "eager_fields": bool(spec.eager_fields),
            "panel_cache_entries": int(spec.panel_cache_entries),
            "curve_cache_enabled": bool(spec.curve_cache_enabled),
            "selected_cache_enabled": bool(spec.selected_cache_enabled),
        }
        if spec.curve_cache_path is not None:
            config["curve_cache_path"] = str(spec.curve_cache_path)
        if spec.selected_cache_path is not None:
            config["selected_cache_path"] = str(spec.selected_cache_path)
        self._source = ParquetFuturesSource(parquet_config=config)

    def load_panels(
        self,
        universe: Sequence[str],
        start,
        end,
        feature_config: FeatureConfig,
    ) -> Dict[str, pd.DataFrame]:
        fields = list(dict.fromkeys(feature_config.raw_fields))
        panels = self._source.fetch_price_at_frequency(
            list(universe),
            pd.Timestamp(start),
            pd.Timestamp(end),
            fields,
            frequency=feature_config.decision_frequency,
        )
        close = panels.get("close")
        if close is None or close.empty:
            raise ValueError("local Parquet query returned no close data")
        return panels

def make_synthetic_panels(
    *,
    periods: int = 600,
    symbols: int = 12,
    frequency: str = "1min",
    seed: int = 7,
) -> Dict[str, pd.DataFrame]:
    """Create closure-safe OHLCVAOI data for unit tests and dev-mode smoke runs."""
    if periods < 20 or symbols < 2:
        raise ValueError("synthetic panels need at least 20 periods and 2 symbols")
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-02 09:00", periods=periods, freq=frequency)
    columns = pd.Index([f"S{i:02d}" for i in range(symbols)])
    common = rng.normal(0.0, 0.0004, size=(periods, 1))
    idiosyncratic = rng.normal(0.0, 0.0015, size=(periods, symbols))
    returns = common + idiosyncratic
    close = 100.0 * np.exp(np.cumsum(returns, axis=0))
    open_price = np.vstack([close[0], close[:-1]])
    spread = np.abs(rng.normal(0.0008, 0.0003, size=close.shape))
    high = np.maximum(open_price, close) * (1.0 + spread)
    low = np.minimum(open_price, close) * (1.0 - spread)
    volume = rng.lognormal(mean=8.0, sigma=0.6, size=close.shape)
    amount = volume * close
    oi = rng.lognormal(mean=9.0, sigma=0.35, size=close.shape)

    def frame(value) -> pd.DataFrame:
        return pd.DataFrame(value, index=index, columns=columns)

    return {
        "open": frame(open_price),
        "high": frame(high),
        "low": frame(low),
        "close": frame(close),
        "volume": frame(volume),
        "amount": frame(amount),
        "oi": frame(oi),
        "oi_change": frame(np.vstack([np.full((1, symbols), np.nan), np.diff(oi, axis=0)])),
    }
