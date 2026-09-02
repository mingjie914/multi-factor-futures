"""Local partitioned-Parquet futures market-data source.

The source builds a point-in-time continuous series from concrete contracts.
The contract used on trading day ``t`` is the vendor dominant contract observed
on the previous trading day. Price fields are forward ratio-adjusted at a roll
using prices that were already known at the end of ``t-1``.
"""
from __future__ import annotations

from collections import OrderedDict
from bisect import bisect_left
from datetime import date
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import polars as pl

from core.interfaces import DataSource
from core.registry import register
from core.sectors import sector_for
from data.contract_symbols import (
    CONTRACT_SYMBOL_SEMANTICS_VERSION,
    MARKET_FIELDS,
    canonicalize_contract_aliases_polars,
    contract_symbol_parts_polars,
)


logger = logging.getLogger(__name__)

_BAR_AGGREGATIONS = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "amount": "sum",
    "oi": "last",
    "position": "last",
    "settle": "last",
}

_RAW_FIELD_MAP = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "amount": "amount",
    "oi": "position",
    "settle": "settle_price",
}
_CURVE_FIELDS = (
    "curve_total_oi",
    "curve_top2_oi",
    "curve_total_volume",
    "curve_contract_count",
    "curve_oi_breadth",
    "curve_oi_concentration",
    "curve_oi_hhi",
)
_PRICE_FIELDS = {"open", "high", "low", "close", "settle"}
_DEFAULT_EAGER_FIELDS = tuple(
    field for field in _RAW_FIELD_MAP if field != "settle"
)
_CONTRACT_SELECTION_SEMANTICS = "exact_yymm_executable_previous_day_oi_epoch_v5"
_CURVE_CACHE_SCHEMA_VERSION = 4
_SELECTED_CACHE_SCHEMA_VERSION = 5
_FREQUENCY_ROUTE = {
    "daily": ("daily", None),
    "1min": ("1min", None),
    "5min": ("1min", "5min"),
    "15min": ("15min", None),
    "30min": ("15min", "30min"),
    "hourly": ("15min", "60min"),
}


def _normalise_frequency(value: str) -> str:
    aliases = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "60m": "hourly",
        "60min": "hourly",
        "1h": "hourly",
    }
    frequency = aliases.get(str(value).lower(), str(value).lower())
    if frequency not in _FREQUENCY_ROUTE:
        raise ValueError(
            f"unsupported parquet frequency {value!r}; "
            f"expected one of {sorted(_FREQUENCY_ROUTE)}"
        )
    return frequency


class MissingParquetPartitionError(RuntimeError):
    """Raised when a requested range crosses a missing internal month."""


@register("data_source", "parquet_futures")
class ParquetFuturesSource(DataSource):
    """Read futures bars from Hive-style ``year_month=YYYY-MM`` partitions."""

    market = "futures"
    cache_namespace = "ParquetFuturesSource_v5"

    def __init__(
        self,
        parquet_config: Optional[dict] = None,
        *,
        _validate_storage: bool = True,
    ) -> None:
        config = dict(parquet_config or {})
        root_text = str(config.get("root_path", "")).strip()
        if not root_text:
            raise ValueError("parquet.root_path is required")
        self.root_path = Path(root_text).expanduser().resolve()
        if _validate_storage and not self.root_path.is_dir():
            raise FileNotFoundError(self.root_path)

        datasets = dict(config.get("datasets") or {})
        self.datasets = {
            "daily": datasets.get("daily", "futureshistoryprices1d"),
            "1min": datasets.get("1min", "futureshistoryprices1m"),
            "15min": datasets.get("15min", "futureshistoryprices15m"),
        }
        if datasets.get("5min"):
            self.datasets["5min"] = datasets["5min"]
        self.seat_dataset = str(
            config.get("seat_dataset", "futuresseatdata")
        ).strip()
        if not self.seat_dataset:
            raise ValueError("parquet.seat_dataset must not be empty")
        self._frequency_routes = dict(_FREQUENCY_ROUTE)
        if "5min" in self.datasets:
            self._frequency_routes["5min"] = ("5min", None)
        # 日历缓存 (fetch_calendar 每次全量读 1d + 正则, 静态数据按 (start,end) 缓存)
        self._CALENDAR_CACHE: dict = {}
        if _validate_storage:
            for name, relative in self.datasets.items():
                path = self.root_path / str(relative)
                if not path.is_dir():
                    raise FileNotFoundError(f"parquet dataset {name!r} not found: {path}")

        self.dominant_lag_days = max(int(config.get("dominant_lag_days", 1)), 1)
        self.schedule_buffer_days = max(
            int(config.get("schedule_buffer_days", 45)), 10
        )
        active_from = dict(config.get("root_active_from") or {})
        self.root_active_from: dict[str, pd.Timestamp] = {}
        for root, value in active_from.items():
            timestamp = pd.Timestamp(value)
            if pd.isna(timestamp):
                raise ValueError(f"invalid active epoch for {root!r}: {value!r}")
            self.root_active_from[str(root).strip().upper()] = timestamp.normalize()
        self._active_epoch_config = {
            root: timestamp.date().isoformat()
            for root, timestamp in sorted(self.root_active_from.items())
        }
        self.eager_fields = bool(config.get("eager_fields", True))
        self.panel_cache_entries = max(int(config.get("panel_cache_entries", 1)), 0)
        self.curve_cache_enabled = bool(config.get("curve_cache_enabled", False))
        curve_cache_text = str(
            config.get("curve_cache_path", "./cache/curve_aggregates")
        ).strip()
        self.curve_cache_path = Path(curve_cache_text).expanduser().resolve()
        if self.curve_cache_enabled:
            self.curve_cache_path.mkdir(parents=True, exist_ok=True)
        self.selected_cache_enabled = bool(
            config.get("selected_cache_enabled", False)
        )
        selected_cache_text = str(
            config.get("selected_cache_path", "./cache/selected_contracts")
        ).strip()
        self.selected_cache_path = Path(
            selected_cache_text
        ).expanduser().resolve()
        if self.selected_cache_enabled:
            self.selected_cache_path.mkdir(parents=True, exist_ok=True)
        self._schema_cache: Dict[tuple[Path, int, int], set[str]] = {}
        self._plan_cache: OrderedDict[tuple, pl.DataFrame] = OrderedDict()
        self._panel_cache: OrderedDict[tuple, Dict[str, pd.DataFrame]] = OrderedDict()
        self._panel_lock = threading.RLock()

    @staticmethod
    def _normalise_tickers(tickers: Iterable[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(str(item).upper() for item in tickers))

    def _dataset_path(self, native_frequency: str) -> Path:
        return self.root_path / self.datasets[native_frequency]

    def _month_files(
        self, native_frequency: str, start, end
    ) -> list[Path]:
        dataset = self._dataset_path(native_frequency)
        periods = list(pd.period_range(
            pd.Timestamp(start).to_period("M"),
            pd.Timestamp(end).to_period("M"),
            freq="M",
        ))
        available: dict[pd.Period, list[Path]] = {}
        for directory in sorted(dataset.glob("year_month=*")):
            if not directory.is_dir():
                continue
            try:
                period = pd.Period(directory.name.split("=", 1)[1], freq="M")
            except (IndexError, ValueError):
                continue
            available[period] = sorted(directory.glob("*.parquet"))
        if not available:
            return []
        first, last = min(available), max(available)
        missing = [
            period for period in periods
            if first <= period <= last and not available.get(period)
        ]
        if missing:
            rendered = ", ".join(str(period) for period in missing)
            raise MissingParquetPartitionError(
                f"missing {native_frequency} parquet month(s): {rendered}"
            )
        files: list[Path] = []
        for period in periods:
            files.extend(available.get(period, []))
        return files

    def _curve_cache_source_fingerprint(
        self,
        native_frequency: str,
        month_start: pd.Timestamp,
        month_end: pd.Timestamp,
    ) -> str:
        files = self._month_files(
            native_frequency,
            month_start - pd.Timedelta(days=7),
            month_end,
        )
        files.extend(
            self._month_files(
                "daily",
                month_start - pd.Timedelta(days=15),
                month_end + pd.Timedelta(days=7),
            )
        )
        records = []
        for path in sorted(set(files), key=lambda item: str(item)):
            stat = path.stat()
            try:
                relative = path.relative_to(self.root_path).as_posix()
            except ValueError:
                relative = str(path)
            records.append((relative, int(stat.st_size), int(stat.st_mtime_ns)))
        payload = json.dumps(records, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _files_fingerprint(root_path: Path, files: Iterable[Path]) -> str:
        records = []
        for path in sorted(set(files), key=lambda item: str(item)):
            stat = path.stat()
            try:
                relative = path.relative_to(root_path).as_posix()
            except ValueError:
                relative = str(path)
            records.append((relative, int(stat.st_size), int(stat.st_mtime_ns)))
        payload = json.dumps(records, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _selected_cache_source_fingerprint(
        self,
        native_frequency: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> str:
        files = self._month_files(
            native_frequency, start.normalize() - pd.Timedelta(days=7), end
        )
        files.extend(
            self._month_files(
                "daily",
                start.normalize() - pd.Timedelta(days=self.schedule_buffer_days),
                end.normalize() + pd.Timedelta(days=7),
            )
        )
        return self._files_fingerprint(self.root_path, files)

    def checkpoint_source_fingerprint(self, start, end) -> str:
        """Fingerprint only storage that can affect the requested factor slice."""
        start_ts = pd.Timestamp(start).normalize() - pd.Timedelta(
            days=self.schedule_buffer_days
        )
        end_ts = pd.Timestamp(end).normalize() + pd.Timedelta(days=7)
        files = []
        for frequency in self.datasets:
            files.extend(self._month_files(frequency, start_ts, end_ts))
        seat_root = self.root_path / self.seat_dataset
        if seat_root.is_dir():
            first = start_ts.to_period("M")
            last = end_ts.to_period("M")
            for path in seat_root.rglob("*.parquet"):
                periods = [part.split("=", 1)[1] for part in path.parts
                           if part.startswith("year_month=")]
                if not periods:
                    continue
                try:
                    period = pd.Period(periods[-1], freq="M")
                except ValueError:
                    continue
                if first <= period <= last:
                    files.append(path)
        return self._files_fingerprint(self.root_path, files)

    def _selected_cache_files(
        self,
        frequency: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        tickers: tuple[str, ...],
        fields: tuple[str, ...],
    ) -> tuple[Path, Path]:
        payload = json.dumps(
            {
                "frequency": frequency,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "tickers": list(tickers),
                "fields": list(fields),
                "contract_semantics": _CONTRACT_SELECTION_SEMANTICS,
                "dominant_lag_days": self.dominant_lag_days,
                "root_active_from": self._active_epoch_config,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        request_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
        stem = f"selected_v{_SELECTED_CACHE_SCHEMA_VERSION}_{frequency}_{request_hash}"
        return (
            self.selected_cache_path / f"{stem}.parquet",
            self.selected_cache_path / f"{stem}.json",
        )

    def _read_selected_cache_polars(
        self,
        frequency: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        tickers: tuple[str, ...],
        fields: tuple[str, ...],
        source_fingerprint: str,
    ) -> Optional[pl.DataFrame]:
        if not self.selected_cache_enabled:
            return None
        data_path, metadata_path = self._selected_cache_files(
            frequency, start, end, tickers, fields
        )
        if not data_path.is_file() or not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected = {
                "schema_version": _SELECTED_CACHE_SCHEMA_VERSION,
                "frequency": frequency,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "tickers": list(tickers),
                "fields": list(fields),
                "source_fingerprint": source_fingerprint,
                "contract_symbol_version": CONTRACT_SYMBOL_SEMANTICS_VERSION,
                "contract_semantics": _CONTRACT_SELECTION_SEMANTICS,
                "dominant_lag_days": self.dominant_lag_days,
                "root_active_from": self._active_epoch_config,
            }
            if any(metadata.get(key) != value for key, value in expected.items()):
                return None
            frame = pl.read_parquet(data_path).with_columns(
                pl.col("trade_datetime").cast(pl.Datetime("ns"))
            )
            required = {"trade_datetime", "root", *fields}
            if not required.issubset(frame.columns) or frame.is_empty():
                return None
            if frame.select(
                pl.col("trade_datetime").is_null().any()
                | pl.col("root").is_null().any()
            ).item():
                return None
            if frame.select(
                pl.struct(["trade_datetime", "root"]).is_duplicated().any()
            ).item():
                return None
            if not set(frame["root"].cast(pl.String).unique()).issubset(tickers):
                return None
            return frame
        except Exception:
            logger.warning("selected-contract cache read failed: %s", data_path)
            return None

    def _write_selected_cache(
        self,
        frame: pd.DataFrame | pl.DataFrame,
        frequency: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        tickers: tuple[str, ...],
        fields: tuple[str, ...],
        source_fingerprint: str,
    ) -> None:
        is_empty = frame.empty if isinstance(frame, pd.DataFrame) else frame.is_empty()
        if not self.selected_cache_enabled or is_empty:
            return
        data_path, metadata_path = self._selected_cache_files(
            frequency, start, end, tickers, fields
        )
        suffix = f".{os.getpid()}.{threading.get_ident()}.tmp"
        data_temp = data_path.with_name(data_path.name + suffix)
        metadata_temp = metadata_path.with_name(metadata_path.name + suffix)
        metadata = {
            "schema_version": _SELECTED_CACHE_SCHEMA_VERSION,
            "frequency": frequency,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "tickers": list(tickers),
            "fields": list(fields),
            "source_fingerprint": source_fingerprint,
            "contract_symbol_version": CONTRACT_SYMBOL_SEMANTICS_VERSION,
            "contract_semantics": _CONTRACT_SELECTION_SEMANTICS,
            "dominant_lag_days": self.dominant_lag_days,
            "root_active_from": self._active_epoch_config,
        }
        try:
            polars_frame = (
                pl.from_pandas(frame) if isinstance(frame, pd.DataFrame) else frame
            )
            polars_frame.write_parquet(data_temp)
            metadata_temp.write_text(
                json.dumps(metadata, ensure_ascii=True, sort_keys=True),
                encoding="utf-8",
            )
            data_temp.replace(data_path)
            metadata_temp.replace(metadata_path)
        except Exception:
            logger.warning("selected-contract cache write failed: %s", data_path)
        finally:
            for temporary in (data_temp, metadata_temp):
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    logger.debug(
                        "selected-contract temporary cleanup failed: %s",
                        temporary,
                        exc_info=True,
                    )

    def _curve_cache_files(
        self,
        frequency: str,
        period: pd.Period,
        tickers: tuple[str, ...],
    ) -> tuple[Path, Path]:
        roots_text = json.dumps(
            {"tickers": tickers, "root_active_from": self._active_epoch_config},
            ensure_ascii=True,
            sort_keys=True,
        )
        roots_hash = hashlib.sha256(roots_text.encode("utf-8")).hexdigest()[:16]
        stem = f"curve_v{_CURVE_CACHE_SCHEMA_VERSION}_{frequency}_{period}_{roots_hash}"
        return (
            self.curve_cache_path / f"{stem}.parquet",
            self.curve_cache_path / f"{stem}.json",
        )

    def _read_curve_month_cache_polars(
        self,
        frequency: str,
        period: pd.Period,
        tickers: tuple[str, ...],
        source_fingerprint: str,
    ) -> Optional[pl.DataFrame]:
        if not self.curve_cache_enabled:
            return None
        data_path, metadata_path = self._curve_cache_files(
            frequency, period, tickers
        )
        if not data_path.is_file() or not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected = {
                "schema_version": _CURVE_CACHE_SCHEMA_VERSION,
                "frequency": frequency,
                "period": str(period),
                "tickers": list(tickers),
                "source_fingerprint": source_fingerprint,
                "contract_symbol_version": CONTRACT_SYMBOL_SEMANTICS_VERSION,
                "root_active_from": self._active_epoch_config,
            }
            if any(metadata.get(key) != value for key, value in expected.items()):
                return None
            frame = pl.read_parquet(data_path).with_columns(
                pl.col("trade_datetime").cast(pl.Datetime("ns"))
            )
            required = {"trade_datetime", "root", *_CURVE_FIELDS}
            if not required.issubset(frame.columns) or frame.is_empty():
                return None
            if frame.select(
                pl.col("trade_datetime").is_null().any()
                | pl.col("root").is_null().any()
            ).item():
                return None
            if frame.select(
                pl.struct(["trade_datetime", "root"]).is_duplicated().any()
            ).item():
                return None
            if not set(frame["root"].cast(pl.String).unique()).issubset(tickers):
                return None
            return frame
        except Exception:
            logger.warning("curve aggregate cache read failed: %s", data_path)
            return None

    def _write_curve_month_cache(
        self,
        frame: pd.DataFrame | pl.DataFrame,
        frequency: str,
        period: pd.Period,
        tickers: tuple[str, ...],
        source_fingerprint: str,
    ) -> None:
        is_empty = frame.empty if isinstance(frame, pd.DataFrame) else frame.is_empty()
        if not self.curve_cache_enabled or is_empty:
            return
        data_path, metadata_path = self._curve_cache_files(
            frequency, period, tickers
        )
        suffix = f".{os.getpid()}.{threading.get_ident()}.tmp"
        data_temp = data_path.with_name(data_path.name + suffix)
        metadata_temp = metadata_path.with_name(metadata_path.name + suffix)
        metadata = {
            "schema_version": _CURVE_CACHE_SCHEMA_VERSION,
            "frequency": frequency,
            "period": str(period),
            "tickers": list(tickers),
            "source_fingerprint": source_fingerprint,
            "contract_symbol_version": CONTRACT_SYMBOL_SEMANTICS_VERSION,
            "root_active_from": self._active_epoch_config,
        }
        try:
            polars_frame = (
                pl.from_pandas(frame) if isinstance(frame, pd.DataFrame) else frame
            )
            polars_frame.write_parquet(data_temp)
            metadata_temp.write_text(
                json.dumps(metadata, ensure_ascii=True, sort_keys=True),
                encoding="utf-8",
            )
            data_temp.replace(data_path)
            metadata_temp.replace(metadata_path)
        except Exception:
            logger.warning("curve aggregate cache write failed: %s", data_path)
        finally:
            for temporary in (data_temp, metadata_temp):
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    logger.debug(
                        "curve-cache temporary cleanup failed: %s",
                        temporary,
                        exc_info=True,
                    )

    def _available_columns(self, path: Path) -> set[str]:
        stat = path.stat()
        key = (path, int(stat.st_size), int(stat.st_mtime_ns))
        cached = self._schema_cache.get(key)
        if cached is not None:
            return cached
        import pyarrow.parquet as pq

        columns = set(pq.ParquetFile(path).schema_arrow.names)
        self._schema_cache[key] = columns
        return columns

    def _read_storage_partitions_polars(
        self,
        native_frequency: str,
        start,
        end,
        columns: Iterable[str],
        roots: Iterable[str] | None = None,
    ) -> tuple[pl.DataFrame, list[str], set[str]]:
        """Read raw Parquet partitions into the canonical Polars boundary."""
        del roots  # Parquet fallback preserves the existing post-read root filter.
        files = self._month_files(native_frequency, start, end)
        requested_columns = list(dict.fromkeys(columns))
        if not files:
            return pl.DataFrame(), requested_columns, set()
        schemas = [self._available_columns(path) for path in files]
        available = set.intersection(*schemas)
        present_anywhere = set.union(*schemas)
        schema_sensitive = list(requested_columns)
        if "symbol" in requested_columns:
            schema_sensitive.extend(
                ["exchange", "trade_date", "trade_datetime", *MARKET_FIELDS]
            )
        partial = sorted(
            column for column in dict.fromkeys(schema_sensitive)
            if column in present_anywhere and column not in available
        )
        if partial:
            raise RuntimeError(
                f"inconsistent {native_frequency} parquet schema across requested "
                f"partitions; partially available columns: {partial}"
            )
        requested = [column for column in requested_columns if column in available]
        if not requested:
            return pl.DataFrame(), [], available
        selected = list(requested)
        if "symbol" in selected:
            for column in ("exchange", "trade_date", "trade_datetime"):
                if column in available and column not in selected:
                    selected.append(column)
        start_date = pd.Timestamp(start).date()
        end_date = pd.Timestamp(end).date()
        frames = []
        for path in files:
            try:
                frame = pl.read_parquet(path, columns=selected).filter(
                    pl.col("trade_date").cast(pl.Date).is_between(
                        start_date, end_date, closed="both"
                    )
                )
            except Exception as exc:
                raise RuntimeError(f"failed to read market partition: {path}") from exc
            if frame.height:
                frames.append(frame)
        if not frames:
            return pl.DataFrame(), requested, available
        return pl.concat(frames, how="vertical_relaxed"), requested, available

    def _read_partitions_polars(
        self,
        native_frequency: str,
        start,
        end,
        columns: Iterable[str],
        roots: Iterable[str] | None = None,
    ) -> pl.DataFrame:
        """Apply alias semantics while retaining a Polars result."""
        combined, requested, available = self._read_storage_partitions_polars(
            native_frequency, start, end, columns, roots=roots
        )
        if not requested or combined.is_empty():
            return combined.select(requested) if requested else combined
        if "symbol" in combined.columns:
            before = combined.height
            canonical = canonicalize_contract_aliases_polars(combined)
            missing_checks = [
                field for field in MARKET_FIELDS
                if field in available and field not in combined.columns
            ]
            if canonical.height < before and missing_checks:
                validation_columns = list(combined.columns) + missing_checks
                validation, _, _ = self._read_storage_partitions_polars(
                    native_frequency,
                    start,
                    end,
                    validation_columns,
                    roots=roots,
                )
                if validation.is_empty():
                    return pl.DataFrame(schema={name: pl.Null for name in requested})
                canonical = canonicalize_contract_aliases_polars(validation)
            combined = canonical
        return combined.select(requested)

    def _annotate_symbols_polars(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Polars equivalent of symbol parsing and economic-epoch filtering."""
        if frame.is_empty() or "symbol" not in frame.columns:
            return frame
        parts = contract_symbol_parts_polars(frame["symbol"])
        result = frame.with_columns(
            *(parts[column] for column in (
                "symbol", "root", "suffix", "delivery_year",
                "delivery_month", "is_concrete",
            )),
            pl.col("trade_date").cast(pl.Date),
        ).drop_nulls(["root", "suffix", "trade_date"])
        if self.root_active_from:
            active = pl.DataFrame({
                "root": list(self.root_active_from),
                "__active_from": [
                    pd.Timestamp(value).date()
                    for value in self.root_active_from.values()
                ],
            })
            result = result.join(active, on="root", how="left").filter(
                pl.col("__active_from").is_null()
                | (pl.col("trade_date") >= pl.col("__active_from"))
            ).drop("__active_from")
        return result

    def _assign_intraday_trade_dates_polars(
        self, frame: pl.DataFrame
    ) -> pl.DataFrame:
        """Map natural timestamps to exchange dates without a pandas frame."""
        if frame.is_empty():
            return frame
        timestamps = frame["trade_datetime"].cast(pl.Datetime("ns"))
        natural = timestamps.dt.date()
        targets = pl.DataFrame({
            "natural": natural,
            "hour": timestamps.dt.hour(),
        }).select(
            pl.when(pl.col("hour") >= 18)
            .then(pl.col("natural") + pl.duration(days=1))
            .otherwise(pl.col("natural"))
            .alias("target")
        )["target"]
        target_values = targets.to_numpy().astype("datetime64[D]")
        calendar = self.fetch_calendar(
            pd.Timestamp(target_values.min()) - pd.Timedelta(days=7),
            pd.Timestamp(target_values.max()) + pd.Timedelta(days=7),
        )
        calendar_values = (
            pd.DatetimeIndex(calendar).normalize().unique().sort_values()
            .to_numpy(dtype="datetime64[D]")
        )
        if not len(calendar_values):
            raise ValueError(
                "cannot assign exchange trading dates: daily parquet calendar is empty"
            )
        locations = np.searchsorted(calendar_values, target_values, side="left")
        assigned = np.full(len(target_values), np.datetime64("NaT", "D"))
        valid = locations < len(calendar_values)
        assigned[valid] = calendar_values[locations[valid]]
        return frame.with_columns(
            pl.Series("trade_date", assigned).cast(pl.Date)
        )

    def trading_session_index(self, timestamps) -> pd.DatetimeIndex:
        """Map wall-clock bars onto an ordered exchange-trading-day clock.

        The returned timestamps are synthetic labels used only by daily
        intraday factors.  Every bar normalizes to its real exchange trading
        day, while the ordering remains evening session, after-midnight
        session, then day session.  Offsets are whole hours, so fixed-width
        intraday resampling keeps its original bar boundaries.
        """
        wall_clock = pd.DatetimeIndex(timestamps)
        if wall_clock.tz is not None:
            wall_clock = wall_clock.tz_localize(None)
        dated = self._assign_intraday_trade_dates_polars(pl.DataFrame({
            "trade_datetime": wall_clock.to_numpy(dtype="datetime64[ns]")
        }))
        trading_days = pd.DatetimeIndex(dated["trade_date"].to_numpy())
        if trading_days.isna().any():
            raise ValueError("cannot map every intraday bar to a trading day")

        time_of_day = wall_clock - wall_clock.normalize()
        evening = wall_clock.hour >= 18
        offsets = time_of_day + pd.Timedelta(hours=6)
        offsets = offsets.where(~evening, time_of_day - pd.Timedelta(hours=18))
        session_index = trading_days + offsets
        return pd.DatetimeIndex(session_index)

    @staticmethod
    def _deduplicate_polars(frame: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
        if frame.is_empty():
            return frame
        sort_columns = list(keys)
        if "sequence" in frame.columns:
            sort_columns.append("sequence")
        return frame.sort(sort_columns).unique(
            subset=keys, keep="last", maintain_order=True
        )

    def _continuous_plan_polars(
        self, tickers: tuple[str, ...], start, end
    ) -> pl.DataFrame:
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        buffer_start = start_ts - pd.Timedelta(days=self.schedule_buffer_days)
        daily_files = self._month_files("daily", buffer_start, end_ts)
        source_fingerprint = self._files_fingerprint(self.root_path, daily_files)
        key = (
            tickers, start_ts, end_ts, self.dominant_lag_days,
            tuple(self._active_epoch_config.items()), source_fingerprint,
        )
        cached = self._plan_cache.get(key)
        if cached is not None:
            self._plan_cache.move_to_end(key)
            return cached.clone()

        daily = self._read_partitions_polars(
            "daily",
            buffer_start,
            end_ts,
            ["symbol", "trade_date", "close", "volume", "position", "sequence"],
            roots=tickers,
        )
        daily = self._annotate_symbols_polars(daily)
        if daily.is_empty():
            return pl.DataFrame()
        daily = daily.filter(pl.col("root").is_in(tickers))
        concrete = self._deduplicate_polars(
            daily.filter(pl.col("is_concrete")),
            ["trade_date", "root", "symbol"],
        ).with_columns(
            pl.col("close").cast(pl.Float64, strict=False),
            pl.col("volume").cast(pl.Float64, strict=False),
            pl.col("position").cast(pl.Float64, strict=False),
        )
        calendar = concrete["trade_date"].drop_nulls().unique().sort()
        if not len(calendar):
            return pl.DataFrame()

        observed = (
            concrete.filter(
                (pl.col("volume").fill_null(0.0) > 0.0)
                & (pl.col("close") > 0.0)
                & pl.col("close").is_finite()
                & (pl.col("position") > 0.0)
                & pl.col("position").is_finite()
            )
            .sort(
                ["trade_date", "root", "position", "volume", "symbol"],
                descending=[False, False, True, True, False],
                nulls_last=True,
            )
            .unique(
                subset=["trade_date", "root"], keep="first", maintain_order=True
            )
            .select("trade_date", "root", pl.col("symbol").alias("observed"))
        )
        grid = pl.DataFrame({"root": list(tickers)}).join(
            pl.DataFrame({"trade_date": calendar}), how="cross"
        ).sort(["root", "trade_date"])
        mapping = grid.join(
            observed,
            on=["root", "trade_date"],
            how="left",
            validate="1:1",
        )

        executable = concrete.filter(
            (pl.col("volume").fill_null(0.0) > 0.0)
            & (pl.col("close") > 0.0)
            & pl.col("close").is_finite()
        )
        quoted_by_date: dict[tuple[str, date], set[str]] = {}
        for row in executable.select("root", "trade_date", "symbol").iter_rows():
            quoted_by_date.setdefault((row[0], row[1]), set()).add(row[2])

        decisions: list[str | None] = []
        current_root = None
        current_contract = None
        for root, trade_date, observed_contract in mapping.select(
            "root", "trade_date", "observed"
        ).iter_rows():
            if root != current_root:
                current_root = root
                current_contract = None
            if observed_contract is not None:
                if current_contract is None:
                    current_contract = observed_contract
                elif observed_contract != current_contract:
                    quoted = quoted_by_date.get((root, trade_date), set())
                    if current_contract in quoted and observed_contract in quoted:
                        current_contract = observed_contract
            decisions.append(current_contract)
        mapping = mapping.with_columns(
            pl.Series("decision_contract", decisions, dtype=pl.String)
        ).with_columns(
            pl.col("decision_contract")
            .shift(self.dominant_lag_days)
            .over("root")
            .alias("contract")
        )

        closes = {
            (root, trade_date, symbol): close
            for root, trade_date, symbol, close in concrete.select(
                "root", "trade_date", "symbol", "close"
            ).iter_rows()
        }
        plans = []
        current_root = None
        adjustment = 1.0
        previous_contract = None
        previous_date = None
        for root, trade_date, contract in mapping.select(
            "root", "trade_date", "contract"
        ).iter_rows():
            if root != current_root:
                current_root = root
                adjustment = 1.0
                previous_contract = None
                previous_date = None
            if contract is None:
                adjustment = 1.0
                previous_contract = None
                previous_date = None
                continue
            if previous_contract is not None and contract != previous_contract:
                old_close = closes.get((root, previous_date, previous_contract))
                new_close = closes.get((root, previous_date, contract))
                if old_close is None or new_close is None:
                    from data.continuous_contract import RolloverAdjustmentError

                    raise RolloverAdjustmentError(
                        f"no common close on {previous_date} for "
                        f"{previous_contract}->{contract}"
                    )
                if (
                    not np.isfinite(old_close) or old_close <= 0.0
                    or not np.isfinite(new_close) or new_close <= 0.0
                ):
                    raise ValueError(
                        f"invalid rollover closes for {previous_contract}->{contract}: "
                        f"old={old_close}, new={new_close}"
                    )
                adjustment *= old_close / new_close
            plans.append((trade_date, root, contract, adjustment))
            previous_contract = contract
            previous_date = trade_date

        plan = pl.DataFrame(
            plans,
            schema={
                "trade_date": pl.Date,
                "root": pl.String,
                "contract": pl.String,
                "adjustment": pl.Float64,
            },
            orient="row",
        )
        if not plan.is_empty():
            plan = plan.filter(
                pl.col("trade_date").is_between(
                    start_ts.date(), end_ts.date(), closed="both"
                )
            )
        self._plan_cache[key] = plan
        while len(self._plan_cache) > 4:
            self._plan_cache.popitem(last=False)
        return plan.clone()

    @staticmethod
    def _resample_long_polars(frame: pl.DataFrame, rule: str) -> pl.DataFrame:
        if frame.is_empty():
            return frame
        polars_rule = rule.replace("min", "m")
        work = frame.with_columns(
            pl.col("trade_datetime").dt.truncate(polars_rule)
        ).sort(["trade_datetime", "root"])
        aggregations = []
        for field, method in _BAR_AGGREGATIONS.items():
            if field not in work.columns:
                continue
            values = pl.col(field).drop_nulls()
            expression = {
                "first": values.first(),
                "last": values.last(),
                "max": pl.col(field).max(),
                "min": pl.col(field).min(),
                "sum": pl.col(field).sum(),
            }[method]
            aggregations.append(expression.alias(field))
        return work.group_by(
            ["trade_datetime", "root"], maintain_order=True
        ).agg(aggregations)

    def _selected_long_polars(
        self,
        tickers: tuple[str, ...],
        start,
        end,
        frequency: str,
        fields: tuple[str, ...],
    ) -> pl.DataFrame:
        """Build the selected-contract long table without a pandas hot path."""
        native_frequency, resample_rule = self._frequency_routes[frequency]
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        source_fingerprint = self._selected_cache_source_fingerprint(
            native_frequency, start_ts, end_ts
        )
        cached = self._read_selected_cache_polars(
            frequency,
            start_ts,
            end_ts,
            tickers,
            fields,
            source_fingerprint,
        )
        if cached is not None:
            return cached
        plan_polars = self._continuous_plan_polars(tickers, start, end)
        if plan_polars.is_empty():
            return pl.DataFrame()

        raw_fields = [
            _RAW_FIELD_MAP[field] for field in fields if field in _RAW_FIELD_MAP
        ]
        columns = ["symbol", "trade_datetime", "trade_date", "sequence", *raw_fields]
        frames = []
        for period in pd.period_range(
            pd.Timestamp(start).to_period("M"),
            pd.Timestamp(end).to_period("M"),
            freq="M",
        ):
            month_start = max(pd.Timestamp(start), period.start_time)
            month_end = min(pd.Timestamp(end), period.end_time)
            read_start = (
                month_start
                if native_frequency == "daily"
                else month_start.normalize() - pd.Timedelta(days=7)
            )
            month_plan = plan_polars.filter(
                pl.col("trade_date").is_between(
                    month_start.normalize().date(),
                    month_end.normalize().date(),
                    closed="both",
                )
            )
            raw = self._read_selected_partitions_polars(
                native_frequency,
                read_start,
                month_end,
                columns,
                tickers,
                month_plan,
            )
            if raw.is_empty():
                continue
            raw = self._annotate_symbols_polars(raw)
            if native_frequency != "daily":
                raw = self._assign_intraday_trade_dates_polars(raw)
            raw = raw.filter(
                pl.col("root").is_in(tickers) & pl.col("is_concrete")
            )
            selected = raw.join(
                month_plan,
                on=["trade_date", "root"],
                how="inner",
                validate="m:1",
            ).filter(pl.col("symbol") == pl.col("contract"))
            if selected.is_empty():
                continue
            selected = selected.with_columns(
                pl.col("trade_datetime").cast(pl.Datetime("ns"))
            )
            selected = self._deduplicate_polars(
                selected, ["trade_datetime", "root"]
            )
            aliases = []
            for field in fields:
                raw_field = _RAW_FIELD_MAP.get(field)
                if (
                    raw_field is not None
                    and raw_field in selected.columns
                    and field not in selected.columns
                ):
                    aliases.append(pl.col(raw_field).alias(field))
            if aliases:
                selected = selected.with_columns(aliases)
            prices = [
                (pl.col(field).cast(pl.Float64) * pl.col("adjustment")).alias(field)
                for field in _PRICE_FIELDS.intersection(fields)
                if field in selected.columns
            ]
            if prices:
                selected = selected.with_columns(prices)
            keep = ["trade_datetime", "root", *(
                field for field in fields if field in selected.columns
            )]
            frames.append(selected.select(keep))

        if not frames:
            return pl.DataFrame()
        result = pl.concat(frames, how="vertical_relaxed")
        result = self._deduplicate_polars(result, ["trade_datetime", "root"])
        if resample_rule:
            result = self._resample_long_polars(result, resample_rule)
        result = result.sort(["trade_datetime", "root"])
        self._write_selected_cache(
            result,
            frequency,
            start_ts,
            end_ts,
            tickers,
            fields,
            source_fingerprint,
        )
        return result

    def _read_selected_partitions_polars(
        self,
        native_frequency: str,
        start,
        end,
        columns: Iterable[str],
        roots: tuple[str, ...],
        _plan: pl.DataFrame,
    ) -> pl.DataFrame:
        """Read candidate rows; DuckDB overrides this to push down the plan."""
        return self._read_partitions_polars(
            native_frequency, start, end, columns, roots=roots
        )

    def _daily_curve_long_polars(
        self,
        tickers: tuple[str, ...],
        start,
        end,
    ) -> pl.DataFrame:
        """Aggregate daily concrete-contract curve state in Polars."""
        buffer_start = pd.Timestamp(start).normalize() - pd.Timedelta(days=10)
        raw = self._read_partitions_polars(
            "daily",
            buffer_start,
            pd.Timestamp(end).normalize(),
            [
                "symbol", "trade_datetime", "trade_date", "position",
                "volume", "sequence",
            ],
            roots=tickers,
        )
        raw = self._annotate_symbols_polars(raw)
        if raw.is_empty():
            return pl.DataFrame()
        concrete = self._deduplicate_polars(
            raw.filter(pl.col("root").is_in(tickers) & pl.col("is_concrete")),
            ["trade_date", "root", "symbol"],
        )
        if concrete.is_empty():
            return pl.DataFrame()
        concrete = concrete.with_columns(
            pl.col("position").cast(pl.Float64, strict=False).clip(0.0, None),
            pl.col("volume").cast(pl.Float64, strict=False).clip(0.0, None),
        ).sort(["root", "symbol", "trade_date"])
        previous = pl.col("position").shift(1).over(["root", "symbol"])
        concrete = concrete.with_columns(
            pl.when(previous.is_not_null())
            .then(pl.col("position") > previous)
            .otherwise(None)
            .alias("oi_increased"),
            pl.col("position").pow(2).alias("position_sq"),
        )
        keys = ["trade_date", "root"]
        aggregate = concrete.group_by(keys, maintain_order=True).agg(
            pl.col("position").sum().alias("curve_total_oi"),
            pl.col("volume").sum().alias("curve_total_volume"),
            (pl.col("position") > 0).sum().alias("curve_contract_count"),
            pl.col("oi_increased").mean().alias("curve_oi_breadth"),
            pl.col("position_sq").sum().alias("position_sq_sum"),
        )
        top2 = (
            concrete.sort(
                [*keys, "position"],
                descending=[False, False, True],
                nulls_last=True,
            )
            .group_by(keys, maintain_order=True)
            .head(2)
            .group_by(keys, maintain_order=True)
            .agg(pl.col("position").sum().alias("curve_top2_oi"))
        )
        result = aggregate.join(top2, on=keys, how="left", validate="1:1")
        denominator = pl.when(pl.col("curve_total_oi") != 0.0).then(
            pl.col("curve_total_oi")
        ).otherwise(None)
        return (
            result.with_columns(
                (pl.col("curve_top2_oi") / denominator).alias(
                    "curve_oi_concentration"
                ),
                (pl.col("position_sq_sum") / denominator.pow(2)).alias(
                    "curve_oi_hhi"
                ),
            )
            .drop("position_sq_sum")
            .rename({"trade_date": "trade_datetime"})
            .with_columns(pl.col("trade_datetime").cast(pl.Datetime("ns")))
            .filter(
                pl.col("trade_datetime").is_between(
                    pd.Timestamp(start).normalize().to_datetime64(),
                    pd.Timestamp(end).normalize().to_datetime64(),
                    closed="both",
                )
            )
            .sort(["trade_datetime", "root"])
        )

    @staticmethod
    def _aggregate_intraday_states_polars(
        raw: pl.DataFrame,
        daily: pl.DataFrame,
    ) -> pl.DataFrame:
        """Use Polars for state shaping and NumPy for dense state arithmetic."""
        if raw.is_empty():
            return pl.DataFrame()
        daily_states: dict[str, dict[date, dict[str, float]]] = {}
        for root, trade_date, symbol, position in daily.select(
            "root", "trade_date", "symbol", "position"
        ).sort(["root", "trade_date", "symbol"]).iter_rows():
            value = float(position) if position is not None else 0.0
            if not np.isfinite(value):
                value = 0.0
            daily_states.setdefault(root, {}).setdefault(trade_date, {})[
                symbol
            ] = max(value, 0.0)
        daily_dates = {
            root: sorted(by_date) for root, by_date in daily_states.items()
        }

        frames = []
        grouped = raw.sort(["root", "trade_date", "trade_datetime"]).partition_by(
            ["root", "trade_date"], maintain_order=True, as_dict=True
        )
        for key, group in grouped.items():
            root, trade_date = key
            dates = daily_dates.get(root, [])
            location = bisect_left(dates, trade_date)
            baseline = (
                daily_states[root][dates[location - 1]]
                if location > 0 else {}
            )
            observed = group.select(
                "trade_datetime", "symbol", "position"
            ).pivot(
                on="symbol",
                index="trade_datetime",
                values="position",
                aggregate_function="last",
            ).sort("trade_datetime")
            contracts = sorted(
                set(observed.columns).difference({"trade_datetime"}) | set(baseline)
            )
            if not contracts:
                continue
            missing = [contract for contract in contracts if contract not in observed]
            if missing:
                observed = observed.with_columns(
                    *(pl.lit(None).cast(pl.Float64).alias(item) for item in missing)
                )
            observed = observed.with_columns(
                *(pl.col(item).cast(pl.Float64, strict=False).fill_nan(None)
                  for item in contracts)
            ).select("trade_datetime", *contracts)
            values = observed.select(contracts).to_numpy().astype(float, copy=False)
            initial = np.array(
                [baseline.get(contract, 0.0) for contract in contracts], dtype=float
            )
            first_missing = np.isnan(values[0])
            values[0, first_missing] = initial[first_missing]
            row_numbers = np.arange(len(values))
            for column in range(values.shape[1]):
                valid = ~np.isnan(values[:, column])
                indices = np.where(valid, row_numbers, 0)
                np.maximum.accumulate(indices, out=indices)
                values[:, column] = values[indices, column]
            values = np.clip(np.nan_to_num(values, nan=0.0), 0.0, None)

            total_oi = values.sum(axis=1)
            top2_oi = (
                total_oi.copy()
                if values.shape[1] <= 2
                else np.partition(values, values.shape[1] - 2, axis=1)[:, -2:].sum(axis=1)
            )
            contract_count = (values > 0.0).sum(axis=1)
            denominator = np.where(total_oi > 0.0, total_oi, np.nan)
            previous = np.vstack([initial, values[:-1]])
            breadth = (values > previous).sum(axis=1) / values.shape[1]
            volume = {
                timestamp: value
                for timestamp, value in group.group_by(
                    "trade_datetime", maintain_order=True
                ).agg(
                    pl.col("volume").fill_nan(None).sum().alias("volume")
                ).iter_rows()
            }
            timestamps = observed["trade_datetime"].to_list()
            frames.append(pl.DataFrame({
                "trade_datetime": timestamps,
                "root": [root] * len(timestamps),
                "curve_total_oi": total_oi,
                "curve_top2_oi": top2_oi,
                "curve_total_volume": [float(volume.get(item, 0.0) or 0.0)
                                       for item in timestamps],
                "curve_contract_count": contract_count,
                "curve_oi_breadth": breadth,
                "curve_oi_concentration": top2_oi / denominator,
                "curve_oi_hhi": np.square(values).sum(axis=1)
                / np.square(denominator),
            }))
        return (
            pl.concat(frames, how="vertical_relaxed")
            if frames else pl.DataFrame()
        )

    @staticmethod
    def _resample_curve_long_polars(
        frame: pl.DataFrame, rule: str
    ) -> pl.DataFrame:
        if frame.is_empty():
            return frame
        work = frame.with_columns(
            pl.col("trade_datetime").dt.truncate(rule.replace("min", "m"))
        ).sort(["trade_datetime", "root"])
        return work.group_by(
            ["trade_datetime", "root"], maintain_order=True
        ).agg(
            pl.col("curve_total_oi").drop_nulls().last(),
            pl.col("curve_top2_oi").drop_nulls().last(),
            pl.col("curve_total_volume").sum(),
            pl.col("curve_contract_count").drop_nulls().last(),
            pl.col("curve_oi_breadth").fill_nan(None).mean(),
            pl.col("curve_oi_concentration").drop_nulls().last(),
            pl.col("curve_oi_hhi").drop_nulls().last(),
        )

    def _intraday_curve_long_polars(
        self,
        tickers: tuple[str, ...],
        start,
        end,
        frequency: str,
    ) -> pl.DataFrame:
        native_frequency, resample_rule = self._frequency_routes[frequency]
        frames = []
        for period in pd.period_range(
            pd.Timestamp(start).to_period("M"),
            pd.Timestamp(end).to_period("M"),
            freq="M",
        ):
            month_start = period.start_time
            month_end = period.end_time
            source_fingerprint = self._curve_cache_source_fingerprint(
                native_frequency, month_start, month_end
            )
            cached = self._read_curve_month_cache_polars(
                frequency, period, tickers, source_fingerprint
            )
            if cached is not None:
                frames.append(cached)
                continue
            raw = self._read_partitions_polars(
                native_frequency,
                month_start.normalize() - pd.Timedelta(days=7),
                month_end,
                ["symbol", "trade_datetime", "trade_date", "position", "volume"],
                roots=tickers,
            )
            raw = self._annotate_symbols_polars(raw)
            raw = self._assign_intraday_trade_dates_polars(raw)
            raw = raw.filter(
                pl.col("root").is_in(tickers)
                & pl.col("is_concrete")
                & pl.col("trade_date").is_between(
                    month_start.normalize().date(),
                    month_end.normalize().date(),
                    closed="both",
                )
            )
            raw = self._deduplicate_polars(
                raw, ["trade_datetime", "root", "symbol"]
            )
            if raw.is_empty():
                continue
            raw = raw.with_columns(
                pl.col("position").cast(pl.Float64, strict=False).fill_nan(None)
                .clip(0.0, None),
                pl.col("volume").cast(pl.Float64, strict=False).fill_nan(None)
                .clip(0.0, None),
            )

            daily = self._read_partitions_polars(
                "daily",
                month_start.normalize() - pd.Timedelta(days=15),
                month_end.normalize(),
                ["symbol", "trade_date", "position", "sequence"],
                roots=tickers,
            )
            daily = self._annotate_symbols_polars(daily)
            daily = self._deduplicate_polars(
                daily.filter(
                    pl.col("root").is_in(tickers) & pl.col("is_concrete")
                ),
                ["trade_date", "root", "symbol"],
            ).with_columns(
                pl.col("position").cast(pl.Float64, strict=False).fill_nan(None)
                .clip(0.0, None)
            )
            aggregate = self._aggregate_intraday_states_polars(raw, daily)
            if aggregate.is_empty():
                continue
            if resample_rule:
                aggregate = self._resample_curve_long_polars(
                    aggregate, resample_rule
                )
            self._write_curve_month_cache(
                aggregate,
                frequency,
                period,
                tickers,
                source_fingerprint,
            )
            frames.append(aggregate)
        if not frames:
            return pl.DataFrame()
        result = self._deduplicate_polars(
            pl.concat(frames, how="vertical_relaxed"),
            ["trade_datetime", "root"],
        ).with_columns(pl.col("trade_datetime").cast(pl.Datetime("ns")))
        dated = self._assign_intraday_trade_dates_polars(
            result.select("trade_datetime")
        )
        result = result.with_columns(dated["trade_date"]).filter(
            pl.col("trade_date").is_between(
                pd.Timestamp(start).normalize().date(),
                pd.Timestamp(end).normalize().date(),
                closed="both",
            )
        ).drop("trade_date")
        return result.sort(["trade_datetime", "root"])

    @staticmethod
    def _polars_long_panels(
        frame: pl.DataFrame,
        fields: Iterable[str],
        tickers: tuple[str, ...],
    ) -> Dict[str, pd.DataFrame]:
        panels = {}
        for field in fields:
            if field not in frame.columns:
                continue
            wide = frame.select("trade_datetime", "root", field).pivot(
                on="root",
                index="trade_datetime",
                values=field,
                aggregate_function="last",
            ).sort("trade_datetime")
            missing = [root for root in tickers if root not in wide.columns]
            if missing:
                wide = wide.with_columns(
                    *(pl.lit(None).cast(pl.Float64).alias(root) for root in missing)
                )
            panel = wide.select("trade_datetime", *tickers).to_pandas().set_index(
                "trade_datetime"
            )
            panel.index = pd.DatetimeIndex(panel.index)
            panel.columns.name = "root"
            panels[field] = panel
        return panels

    def _build_panels(
        self,
        tickers: tuple[str, ...],
        start,
        end,
        frequency: str,
        fields: tuple[str, ...],
    ) -> Dict[str, pd.DataFrame]:
        available_fields = tuple(field for field in fields if field in _RAW_FIELD_MAP)
        curve_fields = tuple(field for field in fields if field in _CURVE_FIELDS)
        panels: Dict[str, pd.DataFrame] = {}
        if available_fields:
            long = self._selected_long_polars(
                tickers, start, end, frequency, available_fields
            )
            panels.update(self._polars_long_panels(
                long, available_fields, tickers
            ))
        if curve_fields:
            if frequency == "daily":
                curve_long_polars = self._daily_curve_long_polars(
                    tickers, start, end
                )
                panels.update(self._polars_long_panels(
                    curve_long_polars, curve_fields, tickers
                ))
            else:
                curve_long_polars = self._intraday_curve_long_polars(
                    tickers, start, end, frequency
                )
                panels.update(self._polars_long_panels(
                    curve_long_polars, curve_fields, tickers
                ))
        if "oi_change" in fields and "oi" in panels:
            panels["oi_change"] = panels["oi"].diff()
        return panels

    def fetch_price(
        self,
        tickers,
        start,
        end,
        fields: List[str],
    ) -> Dict[str, pd.DataFrame]:
        return self.fetch_price_at_frequency(
            tickers, start, end, fields, frequency="daily"
        )

    def fetch_contract_schedule(self, tickers, start, end) -> pd.DataFrame:
        """Return the concrete contract effective for each root/trading day."""
        roots = self._normalise_tickers(tickers)
        plan = self._continuous_plan_polars(roots, start, end)
        if plan.is_empty():
            return pd.DataFrame(columns=roots, dtype=object)
        schedule = plan.pivot(
            on="root", index="trade_date", values="contract"
        ).select("trade_date", *roots).to_pandas().set_index("trade_date")
        schedule.index = pd.DatetimeIndex(schedule.index).rename("trade_date")
        schedule.columns.name = "root"
        return schedule.sort_index().reindex(columns=roots)

    def _contract_curve_polars(
        self,
        tickers,
        start,
        end,
        fields: List[str],
        frequency: str = "daily",
    ) -> pl.DataFrame:
        frequency = _normalise_frequency(frequency)
        roots = self._normalise_tickers(tickers)
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if not roots or start_ts > end_ts:
            return pl.DataFrame()

        field_map = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "position": "position",
            "settle": "settle_price",
        }
        requested = tuple(dict.fromkeys(str(field) for field in fields))
        unsupported = sorted(set(requested) - set(field_map))
        if unsupported:
            raise ValueError(
                "unsupported contract-curve fields: " + ", ".join(unsupported)
            )

        native_frequency, resample_rule = self._frequency_routes[frequency]
        read_start = (
            start_ts.normalize()
            if native_frequency == "daily"
            else start_ts.normalize() - pd.Timedelta(days=7)
        )
        raw_columns = [
            "symbol", "trade_datetime", "trade_date", "sequence",
            *(field_map[field] for field in requested),
        ]
        raw = self._read_partitions_polars(
            native_frequency, read_start, end_ts, raw_columns, roots=roots
        )
        raw = self._annotate_symbols_polars(raw)
        if raw.is_empty():
            return pl.DataFrame()
        if native_frequency != "daily":
            raw = self._assign_intraday_trade_dates_polars(raw)
        raw = raw.filter(
            pl.col("root").is_in(roots)
            & pl.col("is_concrete")
            & pl.col("trade_date").is_between(
                start_ts.normalize().date(), end_ts.normalize().date(),
                closed="both",
            )
        )
        if raw.is_empty():
            return pl.DataFrame()

        raw = self._deduplicate_polars(
            raw, ["trade_datetime", "root", "symbol"]
        )
        rename = {
            source_field: requested_field
            for requested_field, source_field in field_map.items()
            if requested_field in requested and source_field != requested_field
        }
        raw = raw.rename(rename)
        keep = [
            "trade_datetime", "trade_date", "root", "symbol",
            *requested,
        ]
        raw = raw.select(list(dict.fromkeys(keep)))

        if resample_rule and native_frequency != "daily":
            raw = raw.with_columns(
                pl.col("trade_datetime")
                .dt.truncate(resample_rule.replace("min", "m"))
                .alias("_bar_time")
            ).sort("_bar_time")
            aggregations = []
            for field in requested:
                method = _BAR_AGGREGATIONS[field]
                values = pl.col(field).drop_nulls()
                expression = {
                    "first": values.first(),
                    "last": values.last(),
                    "max": pl.col(field).max(),
                    "min": pl.col(field).min(),
                    "sum": pl.col(field).sum(),
                }[method]
                aggregations.append(expression.alias(field))
            raw = raw.group_by(
                ["_bar_time", "trade_date", "root", "symbol"],
                maintain_order=True,
            ).agg(aggregations).rename({"_bar_time": "trade_datetime"})
        return raw.sort(["trade_datetime", "root", "symbol"])

    def fetch_contract_curve_at_frequency(
        self,
        tickers,
        start,
        end,
        fields: List[str],
        frequency: str = "daily",
    ) -> pd.DataFrame:
        """Return exact, unadjusted concrete-contract rows."""
        return self._contract_curve_polars(
            tickers, start, end, fields, frequency
        ).to_pandas()

    def fetch_contract_pair_prices(
        self, tickers, start, end, field: str = "close"
    ) -> Dict[str, pd.DataFrame]:
        """Return the first two observable maturities on each trading day."""
        roots = self._normalise_tickers(tickers)
        if field not in {"open", "high", "low", "close", "settle"}:
            raise ValueError(f"unsupported contract-pair field: {field!r}")
        curve = self._contract_curve_polars(
            roots,
            start,
            end,
            [field, "position"],
            frequency="daily",
        )
        calendar = self.fetch_calendar(start, end)
        empty = pd.DataFrame(index=calendar, columns=roots, dtype=float)
        if curve.is_empty():
            return {"near": empty.copy(), "far": empty.copy()}

        curve = curve.with_columns(
            pl.col(field).cast(pl.Float64, strict=False),
            pl.col("position").cast(pl.Float64, strict=False),
        ).filter(
            pl.col(field).is_finite()
            & (pl.col(field) > 0.0)
            & (pl.col("position") > 0.0)
        )
        if curve.is_empty():
            return {"near": empty.copy(), "far": empty.copy()}
        parts = contract_symbol_parts_polars(curve["symbol"])
        curve = curve.with_columns(
            parts["delivery_year"], parts["delivery_month"]
        ).drop_nulls(["delivery_year", "delivery_month"]).sort([
            "trade_date", "root", "delivery_year", "delivery_month", "symbol"
        ]).with_columns(
            pl.int_range(1, pl.len() + 1)
            .over(["trade_date", "root"])
            .alias("_maturity_rank")
        )

        result = {}
        for label, rank in (("near", 1), ("far", 2)):
            selected = curve.filter(
                pl.col("_maturity_rank") == rank
            ).select(
                pl.col("trade_date").cast(pl.Datetime("ns")).alias(
                    "trade_datetime"
                ),
                "root",
                field,
            )
            panels = self._polars_long_panels(selected, [field], roots)
            result[label] = panels.get(field, empty).reindex(
                index=calendar, columns=roots
            )
        return result

    def fetch_price_at_frequency(
        self,
        tickers,
        start,
        end,
        fields: List[str],
        frequency: str = "daily",
    ) -> Dict[str, pd.DataFrame]:
        frequency = _normalise_frequency(frequency)
        roots = self._normalise_tickers(tickers)
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if not roots or start_ts > end_ts:
            return {}

        requested = tuple(dict.fromkeys(str(field) for field in fields))
        supported_fields = set(_RAW_FIELD_MAP) | set(_CURVE_FIELDS) | {"oi_change"}
        unsupported = sorted(set(requested) - supported_fields)
        if unsupported:
            raise ValueError(
                "unsupported parquet price fields: " + ", ".join(unsupported)
            )
        load_fields = requested
        if self.eager_fields:
            load_fields = tuple(dict.fromkeys(_DEFAULT_EAGER_FIELDS + requested))
        if any(field in _CURVE_FIELDS for field in requested):
            load_fields = tuple(dict.fromkeys(load_fields + _CURVE_FIELDS))
        if "oi_change" in requested and "oi" not in load_fields:
            load_fields += ("oi",)
        native_frequency, _ = self._frequency_routes[frequency]
        source_fingerprint = self._selected_cache_source_fingerprint(
            native_frequency, start_ts, end_ts
        )
        cache_key = (
            roots, start_ts, end_ts, frequency,
            tuple(self._active_epoch_config.items()), source_fingerprint,
        )
        with self._panel_lock:
            cached = self._panel_cache.get(cache_key)
            if cached is None:
                cached = self._build_panels(
                    roots, start_ts, end_ts, frequency, load_fields
                )
            else:
                missing_fields = tuple(
                    field for field in load_fields
                    if field in supported_fields and field not in cached
                )
                if "oi_change" in missing_fields and "oi" in cached:
                    cached["oi_change"] = cached["oi"].diff()
                    missing_fields = tuple(
                        field for field in missing_fields
                        if field != "oi_change"
                    )
                if missing_fields:
                    cached.update(
                        self._build_panels(
                            roots, start_ts, end_ts, frequency, missing_fields
                        )
                    )
            if self.panel_cache_entries:
                self._panel_cache[cache_key] = cached
                self._panel_cache.move_to_end(cache_key)
                while len(self._panel_cache) > self.panel_cache_entries:
                    self._panel_cache.popitem(last=False)
            result = {
                field: cached[field]
                for field in requested
                if field in cached
            }
            missing = sorted(set(requested) - set(result))
            if missing:
                raise RuntimeError(
                    "parquet source could not build requested fields: "
                    + ", ".join(missing)
                )
            return result

    def fetch_fundamental(self, tickers, start, end, fields: List[str]) -> dict:
        return {}

    def fetch_industry(self, tickers, date) -> pd.Series:
        roots = self._normalise_tickers(tickers)
        return pd.Series({root: sector_for(root) for root in roots}, dtype=object)

    def fetch_index_constituents(self, index_code: str, date) -> pd.Index:
        timestamp = pd.Timestamp(date)
        frame = self._read_partitions_polars(
            "daily", timestamp - pd.Timedelta(days=45), timestamp,
            ["symbol", "trade_date"],
        )
        frame = self._annotate_symbols_polars(frame)
        roots = frame.filter(pl.col("is_concrete"))["root"].unique().sort().to_list()
        return pd.Index(roots)

    def fetch_listing_dates(self, tickers) -> pd.Series:
        """Return each root's first published concrete-contract trade date."""
        roots = self._normalise_tickers(tickers)
        if not roots:
            return pd.Series(dtype="datetime64[ns]")
        files = sorted(self._dataset_path("daily").glob("year_month=*/*.parquet"))
        if not files:
            return pd.Series(index=roots, dtype="datetime64[ns]")
        missing = [
            str(path) for path in files
            if not {"symbol", "trade_date"}.issubset(self._available_columns(path))
        ]
        if missing:
            raise RuntimeError(
                "daily parquet shards lack listing-date columns: "
                + ", ".join(missing[:5])
            )
        frame = pl.concat([
            pl.read_parquet(path, columns=["symbol", "trade_date"])
            .group_by("symbol").agg(pl.col("trade_date").min())
            for path in files
        ], how="vertical_relaxed")
        listing = self._annotate_symbols_polars(frame).filter(
            pl.col("is_concrete") & pl.col("root").is_in(roots)
        ).group_by("root").agg(pl.col("trade_date").min())
        aligned = pl.DataFrame({"root": list(roots)}).join(
            listing, on="root", how="left"
        )
        result = pd.Series(
            pd.to_datetime(aligned["trade_date"].to_list()),
            index=list(roots),
            name="trade_date",
            dtype="datetime64[ns]",
        )
        result.index.name = "root"
        return result

    def fetch_latest_trade_date(self) -> pd.Timestamp:
        """Return the newest published daily trade date from physical shards."""
        dataset = self._dataset_path("daily")
        directories = []
        for directory in dataset.glob("year_month=*"):
            if not directory.is_dir():
                continue
            try:
                period = pd.Period(directory.name.split("=", 1)[1], freq="M")
            except (IndexError, ValueError):
                continue
            files = sorted(directory.glob("*.parquet"))
            if files:
                directories.append((period, directory, files))
        if not directories:
            raise FileNotFoundError(f"no daily parquet shards under {dataset}")

        _, directory, files = max(directories, key=lambda item: item[0])
        missing = [
            str(path) for path in files
            if "trade_date" not in self._available_columns(path)
        ]
        if missing:
            raise RuntimeError(
                "latest daily parquet shards lack trade_date: "
                + ", ".join(missing[:5])
            )
        latest_values = [
            pl.read_parquet(path, columns=["trade_date"])
            .select(pl.col("trade_date").cast(pl.Date).max())
            .item()
            for path in files
        ]
        latest_values = [value for value in latest_values if value is not None]
        if not latest_values:
            raise RuntimeError(
                f"latest daily parquet shards contain no valid trade_date: {directory}"
            )
        latest = max(latest_values)
        return pd.Timestamp(latest).normalize()

    def fetch_calendar(self, start, end) -> pd.DatetimeIndex:
        # 日历是静态数据: 按 (start,end) 缓存, 避免每次 erc_w/回测重复读全量 1d
        # 并重复 _annotate_symbols 正则 (6000万次 str.extract 热点)
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        files = self._month_files("daily", start_ts, end_ts)
        source_fingerprint = self._files_fingerprint(self.root_path, files)
        key = (
            start_ts, end_ts, tuple(self._active_epoch_config.items()),
            source_fingerprint,
        )
        if key in self._CALENDAR_CACHE:
            return self._CALENDAR_CACHE[key]
        frame = self._read_partitions_polars(
            "daily", start_ts, end_ts, ["symbol", "trade_date"]
        )
        frame = self._annotate_symbols_polars(frame)
        if frame.is_empty():
            res = pd.DatetimeIndex([])
        else:
            dates = frame.filter(pl.col("is_concrete")).select(
                pl.col("trade_date").drop_nulls().unique().sort()
            )["trade_date"]
            res = pd.DatetimeIndex(dates.to_numpy().astype("datetime64[ns]"))
        # LRU: 最多缓存 32 个日历 (回测窗口固定, 命中率高)
        if len(self._CALENDAR_CACHE) >= 32:
            self._CALENDAR_CACHE.pop(next(iter(self._CALENDAR_CACHE)))
        self._CALENDAR_CACHE[key] = res
        return res

__all__ = ["MissingParquetPartitionError", "ParquetFuturesSource"]
