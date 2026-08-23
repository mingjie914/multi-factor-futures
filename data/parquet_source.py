"""Local partitioned-Parquet futures market-data source.

The source builds a point-in-time continuous series from concrete contracts.
The contract used on trading day ``t`` is the vendor dominant contract observed
on the previous trading day. Price fields are forward ratio-adjusted at a roll
using prices that were already known at the end of ``t-1``.
"""
from __future__ import annotations

from collections import OrderedDict
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

from core.interfaces import DataSource
from core.registry import register
from core.sectors import sector_for
from data.contract_symbols import (
    CONTRACT_SYMBOL_SEMANTICS_VERSION,
    MARKET_FIELDS,
    canonicalize_contract_aliases,
    contract_symbol_parts,
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
        self._plan_cache: OrderedDict[tuple, pd.DataFrame] = OrderedDict()
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

    def _read_selected_cache(
        self,
        frequency: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        tickers: tuple[str, ...],
        fields: tuple[str, ...],
        source_fingerprint: str,
    ) -> Optional[pd.DataFrame]:
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
            frame = pd.read_parquet(data_path)
            required = {"trade_datetime", "root", *fields}
            frame["trade_datetime"] = pd.to_datetime(frame["trade_datetime"])
            if not self._valid_cached_long_frame(frame, required, tickers):
                return None
            return frame
        except Exception:
            logger.warning("selected-contract cache read failed: %s", data_path)
            return None

    def _write_selected_cache(
        self,
        frame: pd.DataFrame,
        frequency: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        tickers: tuple[str, ...],
        fields: tuple[str, ...],
        source_fingerprint: str,
    ) -> None:
        if not self.selected_cache_enabled or frame.empty:
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
            frame.to_parquet(data_temp, index=False)
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

    def _read_curve_month_cache(
        self,
        frequency: str,
        period: pd.Period,
        tickers: tuple[str, ...],
        source_fingerprint: str,
    ) -> Optional[pd.DataFrame]:
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
            frame = pd.read_parquet(data_path)
            required = {"trade_datetime", "root", *_CURVE_FIELDS}
            frame["trade_datetime"] = pd.to_datetime(frame["trade_datetime"])
            if not self._valid_cached_long_frame(frame, required, tickers):
                return None
            return frame
        except Exception:
            logger.warning("curve aggregate cache read failed: %s", data_path)
            return None

    @staticmethod
    def _valid_cached_long_frame(
        frame: pd.DataFrame,
        required: set[str],
        tickers: tuple[str, ...],
    ) -> bool:
        if not required.issubset(frame.columns) or frame.empty:
            return False
        if frame["trade_datetime"].isna().any() or frame["root"].isna().any():
            return False
        if frame.duplicated(["trade_datetime", "root"]).any():
            return False
        return set(frame["root"].astype(str)).issubset(tickers)

    def _write_curve_month_cache(
        self,
        frame: pd.DataFrame,
        frequency: str,
        period: pd.Period,
        tickers: tuple[str, ...],
        source_fingerprint: str,
    ) -> None:
        if not self.curve_cache_enabled or frame.empty:
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
            frame.to_parquet(data_temp, index=False)
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

    def _read_storage_partitions(
        self,
        native_frequency: str,
        start,
        end,
        columns: Iterable[str],
        roots: Iterable[str] | None = None,
    ) -> tuple[pd.DataFrame, list[str], set[str]]:
        files = self._month_files(native_frequency, start, end)
        if not files:
            requested = list(dict.fromkeys(columns))
            return pd.DataFrame(columns=requested), requested, set()
        schemas = [self._available_columns(path) for path in files]
        available = set.intersection(*schemas)
        present_anywhere = set.union(*schemas)
        requested_columns = list(dict.fromkeys(columns))
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
            return pd.DataFrame(), [], available
        selected = list(requested)
        if "symbol" in selected:
            for column in ("exchange", "trade_date", "trade_datetime"):
                if column in available and column not in selected:
                    selected.append(column)
        filters = [
            ("trade_date", ">=", pd.Timestamp(start).date()),
            ("trade_date", "<=", pd.Timestamp(end).date()),
        ]
        # 逐文件读取再 concat: 避免 pyarrow 多文件 schema merge 时
        # dictionary 编码 vs plain string 的 year_month 列类型冲突 (ArrowTypeError)
        frames = []
        for f in files:
            try:
                df = pd.read_parquet(f, columns=selected, filters=filters)
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                raise RuntimeError(f"failed to read market partition: {f}") from exc
        if not frames:
            return pd.DataFrame(columns=requested), requested, available
        combined = pd.concat(frames, ignore_index=True)
        return combined, requested, available

    def _read_partitions(
        self,
        native_frequency: str,
        start,
        end,
        columns: Iterable[str],
        roots: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        combined, requested, available = self._read_storage_partitions(
            native_frequency, start, end, columns, roots=roots
        )
        if not requested:
            return combined
        if combined.empty:
            return combined.loc[:, requested]
        if "symbol" in combined:
            before = len(combined)
            canonical = canonicalize_contract_aliases(combined)
            missing_checks = [
                field for field in MARKET_FIELDS
                if field in available and field not in combined.columns
            ]
            if len(canonical) < before and missing_checks:
                validation_columns = list(combined.columns) + missing_checks
                validation, _, _ = self._read_storage_partitions(
                    native_frequency, start, end, validation_columns, roots=roots
                )
                if validation.empty:
                    return pd.DataFrame(columns=requested)
                canonical = canonicalize_contract_aliases(validation)
            combined = canonical
        return combined.loc[:, requested]

    def _annotate_symbols(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or "symbol" not in frame:
            return frame
        result = frame.copy()
        parts = contract_symbol_parts(result["symbol"])
        for column in (
            "symbol", "root", "suffix", "delivery_year",
            "delivery_month", "is_concrete",
        ):
            result[column] = parts[column]
        result["trade_date"] = pd.to_datetime(result["trade_date"]).dt.normalize()
        result = result.dropna(subset=["root", "suffix", "trade_date"])
        if self.root_active_from:
            active_from = result["root"].map(self.root_active_from)
            result = result.loc[
                active_from.isna() | result["trade_date"].ge(active_from)
            ]
        return result

    def _assign_intraday_trade_dates(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Replace vendor natural dates with exchange trading dates."""
        if frame.empty:
            return frame
        result = frame.copy()
        timestamps = pd.to_datetime(result["trade_datetime"], errors="coerce")
        natural_dates = timestamps.dt.normalize()
        targets = natural_dates.where(
            timestamps.dt.hour.lt(18), natural_dates + pd.Timedelta(days=1)
        )
        calendar = self.fetch_calendar(
            targets.min() - pd.Timedelta(days=7),
            targets.max() + pd.Timedelta(days=7),
        )
        calendar = pd.DatetimeIndex(calendar).normalize().unique().sort_values()
        if len(calendar) == 0:
            raise ValueError(
                "cannot assign exchange trading dates: daily parquet calendar is empty"
            )
        locations = calendar.searchsorted(targets.to_numpy(), side="left")
        assigned = np.full(
            len(targets), np.datetime64("NaT", "ns"), dtype="datetime64[ns]"
        )
        valid = locations < len(calendar)
        assigned[valid] = calendar.to_numpy()[locations[valid]]
        result["trade_date"] = assigned
        return result

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
        dated = self._assign_intraday_trade_dates(
            pd.DataFrame({"trade_datetime": wall_clock})
        )
        trading_days = pd.DatetimeIndex(dated["trade_date"])
        if trading_days.isna().any():
            raise ValueError("cannot map every intraday bar to a trading day")

        time_of_day = wall_clock - wall_clock.normalize()
        evening = wall_clock.hour >= 18
        offsets = time_of_day + pd.Timedelta(hours=6)
        offsets = offsets.where(~evening, time_of_day - pd.Timedelta(hours=18))
        session_index = trading_days + offsets
        return pd.DatetimeIndex(session_index)

    @staticmethod
    def _deduplicate(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        if frame.empty:
            return frame
        sort_columns = list(keys)
        if "sequence" in frame.columns:
            sort_columns.append("sequence")
        return (
            frame.sort_values(sort_columns)
            .drop_duplicates(keys, keep="last")
            .reset_index(drop=True)
        )

    def _infer_vendor_main(
        self, frame: pd.DataFrame, tickers: tuple[str, ...]
    ) -> pd.DataFrame:
        """Choose the highest-open-interest real contract for each root/date."""
        frame = self._annotate_symbols(frame)
        frame = frame.loc[frame["root"].isin(tickers)]
        concrete = frame.loc[frame["is_concrete"]].copy()
        concrete = self._deduplicate(concrete, ["trade_date", "root", "symbol"])
        close = pd.to_numeric(concrete["close"], errors="coerce")
        position = pd.to_numeric(concrete["position"], errors="coerce")
        liquid = concrete.loc[
            concrete["volume"].fillna(0).gt(0)
            & close.gt(0)
            & np.isfinite(close)
            & position.gt(0)
            & np.isfinite(position)
        ]
        return (
            liquid.sort_values(
                ["trade_date", "root", "position", "volume", "symbol"],
                ascending=[True, True, False, False, True],
                na_position="last",
            )
            .drop_duplicates(["trade_date", "root"], keep="first")
            [["trade_date", "root", "symbol"]]
            .sort_values(["root", "trade_date"])
        )

    def _continuous_plan(
        self, tickers: tuple[str, ...], start, end
    ) -> pd.DataFrame:
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
            return cached.copy(deep=False)

        daily = self._read_partitions(
            "daily",
            buffer_start,
            end_ts,
            [
                "symbol", "trade_date", "close", "volume", "position",
                "sequence",
            ],
        )
        daily = self._annotate_symbols(daily)
        calendar = pd.DatetimeIndex(
            daily.loc[daily["is_concrete"], "trade_date"].dropna().unique()
        ).sort_values()
        daily = daily.loc[daily["root"].isin(tickers)]
        observed = self._infer_vendor_main(daily, tickers)
        grid = pd.MultiIndex.from_product(
            [tickers, calendar], names=["root", "trade_date"]
        ).to_frame(index=False)
        mapping = grid.merge(
            observed, on=["root", "trade_date"], how="left", validate="one_to_one"
        ).sort_values(["root", "trade_date"])

        concrete = daily.loc[daily["is_concrete"]].copy()
        concrete = self._deduplicate(concrete, ["trade_date", "root", "symbol"])
        executable_close = pd.to_numeric(concrete["close"], errors="coerce")
        executable = concrete.loc[
            concrete["volume"].fillna(0).gt(0)
            & executable_close.gt(0)
            & np.isfinite(executable_close)
        ]

        # A newly observed dominant can become tomorrow's contract only when
        # both the held and proposed contracts are executable at today's close.
        # Partial/suspended data therefore keeps the prior causal decision; it
        # never falls forward to the same-day dominant.
        decisions = pd.Series(index=mapping.index, dtype=object)
        for root, group in mapping.groupby("root", sort=False):
            quoted_by_date = (
                executable.loc[executable["root"].eq(root)]
                .groupby("trade_date")["symbol"]
                .agg(set)
                .to_dict()
            )
            current_contract = None
            for row in group.itertuples():
                candidate = None if pd.isna(row.symbol) else str(row.symbol)
                if candidate is not None:
                    if current_contract is None:
                        current_contract = candidate
                    elif candidate != current_contract:
                        quoted = quoted_by_date.get(row.trade_date, set())
                        if current_contract in quoted and candidate in quoted:
                            current_contract = candidate
                decisions.loc[row.Index] = current_contract
        mapping["decision_contract"] = decisions
        mapping["contract"] = mapping.groupby("root")[
            "decision_contract"
        ].shift(self.dominant_lag_days)
        plans = []
        for root, group in mapping.groupby("root", sort=False):
            adjustment = 1.0
            previous_contract = None
            previous_date = None
            for row in group.itertuples(index=False):
                if pd.isna(row.contract):
                    # No prior-day liquid contract: publish no synthetic price
                    # and restart only after a new causal schedule is available.
                    adjustment = 1.0
                    previous_contract = None
                    previous_date = None
                    continue
                contract = str(row.contract)
                if (
                    previous_contract is not None
                    and contract != previous_contract
                    and previous_date is not None
                ):
                    root_prices = concrete.loc[
                        concrete["root"].eq(root)
                        & concrete["symbol"].isin([previous_contract, contract])
                        # Both legs must be executable at the preceding close.
                        & concrete["trade_date"].eq(previous_date),
                        ["trade_date", "symbol", "close"],
                    ].pivot_table(
                        index="trade_date", columns="symbol", values="close",
                        aggfunc="last",
                    )
                    if previous_contract not in root_prices or contract not in root_prices:
                        common = pd.DataFrame()
                    else:
                        common = root_prices[
                            [previous_contract, contract]
                        ].dropna(how="any")
                    if common.empty:
                        # 换月对无共同交易日: fail-closed (与 continuous_contract.py 一致)
                        # 数据补全后 2015-2026 全历史无触发 (已验证 skip=0);
                        # 若未来数据缺失则显式报错, 不静默生成断层连续序列
                        from data.continuous_contract import RolloverAdjustmentError

                        raise RolloverAdjustmentError(
                            f"no common close on {previous_date.date()} for "
                            f"{previous_contract}->{contract}"
                        )
                    old_close = float(common.iloc[-1][previous_contract])
                    new_close = float(common.iloc[-1][contract])
                    if (
                        not np.isfinite(old_close) or old_close <= 0.0
                        or not np.isfinite(new_close) or new_close <= 0.0
                    ):
                        raise ValueError(
                            f"invalid rollover closes for {previous_contract}->{contract}: "
                            f"old={old_close}, new={new_close}"
                        )
                    adjustment *= old_close / new_close
                plans.append(
                    {
                        "trade_date": row.trade_date,
                        "root": root,
                        "contract": contract,
                        "adjustment": adjustment,
                    }
                )
                previous_contract = contract
                previous_date = row.trade_date

        plan = pd.DataFrame(plans)
        if not plan.empty:
            plan = plan.loc[
                plan["trade_date"].between(start_ts, end_ts)
            ].reset_index(drop=True)
        self._plan_cache[key] = plan
        while len(self._plan_cache) > 4:
            self._plan_cache.popitem(last=False)
        return plan.copy(deep=False)

    @staticmethod
    def _resample_long(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
        if frame.empty:
            return frame
        work = frame.copy()
        work["trade_datetime"] = pd.to_datetime(
            work["trade_datetime"]
        ).dt.floor(rule)
        aggregations = {
            field: method
            for field, method in _BAR_AGGREGATIONS.items()
            if field in work.columns
        }
        return (
            work.groupby(["trade_datetime", "root"], sort=True, as_index=False)
            .agg(aggregations)
        )

    def _selected_long(
        self,
        tickers: tuple[str, ...],
        start,
        end,
        frequency: str,
        fields: tuple[str, ...],
    ) -> pd.DataFrame:
        native_frequency, resample_rule = self._frequency_routes[frequency]
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        source_fingerprint = self._selected_cache_source_fingerprint(
            native_frequency, start_ts, end_ts
        )
        cached = self._read_selected_cache(
            frequency,
            start_ts,
            end_ts,
            tickers,
            fields,
            source_fingerprint,
        )
        if cached is not None:
            return cached
        plan = self._continuous_plan(tickers, start, end)
        if plan.empty:
            return pd.DataFrame()

        raw_fields = [_RAW_FIELD_MAP[field] for field in fields if field in _RAW_FIELD_MAP]
        columns = ["symbol", "trade_datetime", "trade_date", "sequence"] + raw_fields
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
            raw = self._read_partitions(
                native_frequency, read_start, month_end, columns
            )
            if raw.empty:
                continue
            raw = self._annotate_symbols(raw)
            if native_frequency != "daily":
                raw = self._assign_intraday_trade_dates(raw)
            raw = raw.loc[
                raw["root"].isin(tickers)
                & raw["is_concrete"]
            ]
            month_plan = plan.loc[
                plan["trade_date"].between(
                    pd.Timestamp(month_start).normalize(),
                    pd.Timestamp(month_end).normalize(),
                )
            ]
            selected = raw.merge(
                month_plan,
                on=["trade_date", "root"],
                how="inner",
                validate="many_to_one",
            )
            selected = selected.loc[selected["symbol"].eq(selected["contract"])]
            if selected.empty:
                continue
            selected["trade_datetime"] = pd.to_datetime(selected["trade_datetime"])
            selected = self._deduplicate(
                selected, ["trade_datetime", "root"]
            )
            for field in fields:
                raw_field = _RAW_FIELD_MAP.get(field)
                if (
                    raw_field is not None
                    and raw_field in selected
                    and field not in selected
                ):
                    selected[field] = selected[raw_field]
            for field in _PRICE_FIELDS.intersection(fields):
                selected[field] = selected[field].astype(float) * selected["adjustment"]
            keep = ["trade_datetime", "root"] + [
                field for field in fields if field in selected.columns
            ]
            frames.append(selected[keep])

        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        result = self._deduplicate(result, ["trade_datetime", "root"])
        if resample_rule:
            result = self._resample_long(result, resample_rule)
        result = result.sort_values(["trade_datetime", "root"]).reset_index(drop=True)
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

    def _daily_curve_long(
        self,
        tickers: tuple[str, ...],
        start,
        end,
    ) -> pd.DataFrame:
        """Aggregate concrete-contract OI without using synthetic contracts."""
        buffer_start = pd.Timestamp(start).normalize() - pd.Timedelta(days=10)
        raw = self._read_partitions(
            "daily",
            buffer_start,
            pd.Timestamp(end).normalize(),
            [
                "symbol", "trade_datetime", "trade_date", "position",
                "volume", "sequence",
            ],
        )
        raw = self._annotate_symbols(raw)
        if raw.empty:
            return pd.DataFrame()
        concrete = raw.loc[
            raw["root"].isin(tickers)
            & raw["is_concrete"]
        ].copy()
        concrete = self._deduplicate(
            concrete, ["trade_date", "root", "symbol"]
        )
        if concrete.empty:
            return pd.DataFrame()
        concrete["position"] = pd.to_numeric(
            concrete["position"], errors="coerce"
        ).clip(lower=0.0)
        concrete["volume"] = pd.to_numeric(
            concrete["volume"], errors="coerce"
        ).clip(lower=0.0)
        concrete = concrete.sort_values(["root", "symbol", "trade_date"])
        previous = concrete.groupby(["root", "symbol"])["position"].shift(1)
        concrete["oi_increased"] = (
            concrete["position"].gt(previous).where(previous.notna())
        )
        concrete["position_sq"] = concrete["position"].pow(2)

        keys = ["trade_date", "root"]
        aggregate = concrete.groupby(keys, sort=True).agg(
            curve_total_oi=("position", "sum"),
            curve_total_volume=("volume", "sum"),
            curve_contract_count=("position", lambda value: int(value.gt(0).sum())),
            curve_oi_breadth=("oi_increased", "mean"),
            position_sq_sum=("position_sq", "sum"),
        )
        top2 = (
            concrete.sort_values(
                keys + ["position"],
                ascending=[True, True, False],
                na_position="last",
            )
            .groupby(keys, sort=True)
            .head(2)
            .groupby(keys, sort=True)["position"]
            .sum()
            .rename("curve_top2_oi")
        )
        aggregate = aggregate.join(top2, how="left")
        denominator = aggregate["curve_total_oi"].replace(0.0, np.nan)
        aggregate["curve_oi_concentration"] = (
            aggregate["curve_top2_oi"] / denominator
        )
        aggregate["curve_oi_hhi"] = (
            aggregate.pop("position_sq_sum") / denominator.pow(2)
        )
        result = aggregate.reset_index().rename(
            columns={"trade_date": "trade_datetime"}
        )
        result["trade_datetime"] = pd.to_datetime(result["trade_datetime"])
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        return result.loc[
            result["trade_datetime"].between(start_ts, end_ts)
        ].sort_values(["trade_datetime", "root"]).reset_index(drop=True)

    @staticmethod
    def _aggregate_intraday_states(
        raw: pd.DataFrame,
        daily: pd.DataFrame,
    ) -> pd.DataFrame:
        frames = []
        daily_by_root = {
            root: group for root, group in daily.groupby("root", sort=False)
        }
        for (root, trade_date), group in raw.groupby(
            ["root", "trade_date"], sort=True
        ):
            day = pd.Timestamp(trade_date).normalize()
            root_daily = daily_by_root.get(root, pd.DataFrame())
            baseline = pd.Series(dtype=float)
            if not root_daily.empty:
                previous_dates = root_daily.loc[
                    root_daily["trade_date"].lt(day), "trade_date"
                ]
                if not previous_dates.empty:
                    baseline_date = previous_dates.max()
                    baseline_rows = root_daily.loc[
                        root_daily["trade_date"].eq(baseline_date)
                    ]
                    baseline = baseline_rows.set_index("symbol")["position"]

            observed = group.pivot_table(
                index="trade_datetime",
                columns="symbol",
                values="position",
                aggfunc="last",
            ).sort_index()
            if observed.empty:
                continue
            contracts = observed.columns.union(baseline.index)
            observed = observed.reindex(columns=contracts)
            initial = baseline.reindex(contracts).fillna(0.0)
            states = observed.copy()
            states.iloc[0] = states.iloc[0].fillna(initial)
            states = states.ffill().fillna(0.0).clip(lower=0.0)

            state_values = states.to_numpy(dtype=float, copy=False)
            total_oi = state_values.sum(axis=1)
            if state_values.shape[1] <= 2:
                top2_oi = total_oi.copy()
            else:
                top2_oi = np.partition(
                    state_values, state_values.shape[1] - 2, axis=1
                )[:, -2:].sum(axis=1)
            contract_count = (state_values > 0.0).sum(axis=1)
            denominator = np.where(total_oi > 0.0, total_oi, np.nan)
            concentration = top2_oi / denominator
            hhi = np.square(state_values).sum(axis=1) / np.square(denominator)

            previous_states = states.shift(1)
            previous_states.iloc[0] = initial
            comparable = previous_states.notna()
            increased = states.gt(previous_states) & comparable
            breadth_denominator = comparable.sum(axis=1).replace(0, np.nan)
            breadth = increased.sum(axis=1) / breadth_denominator

            volume = group.pivot_table(
                index="trade_datetime",
                columns="symbol",
                values="volume",
                aggfunc="sum",
            ).reindex(index=states.index).fillna(0.0).sum(axis=1)
            frames.append(pd.DataFrame({
                "trade_datetime": states.index,
                "root": root,
                "curve_total_oi": total_oi,
                "curve_top2_oi": top2_oi,
                "curve_total_volume": volume.to_numpy(dtype=float),
                "curve_contract_count": contract_count,
                "curve_oi_breadth": breadth.to_numpy(dtype=float),
                "curve_oi_concentration": concentration,
                "curve_oi_hhi": hhi,
            }))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _resample_curve_long(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
        if frame.empty:
            return frame
        work = frame.copy()
        work["trade_datetime"] = pd.to_datetime(
            work["trade_datetime"]
        ).dt.floor(rule)
        return (
            work.groupby(["trade_datetime", "root"], sort=True, as_index=False)
            .agg({
                "curve_total_oi": "last",
                "curve_top2_oi": "last",
                "curve_total_volume": "sum",
                "curve_contract_count": "last",
                "curve_oi_breadth": "mean",
                "curve_oi_concentration": "last",
                "curve_oi_hhi": "last",
            })
        )

    def _intraday_curve_long(
        self,
        tickers: tuple[str, ...],
        start,
        end,
        frequency: str,
    ) -> pd.DataFrame:
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
            cached = self._read_curve_month_cache(
                frequency, period, tickers, source_fingerprint
            )
            if cached is not None:
                frames.append(cached)
                continue
            raw = self._read_partitions(
                native_frequency,
                month_start.normalize() - pd.Timedelta(days=7),
                month_end,
                [
                    "symbol", "trade_datetime", "trade_date", "position",
                    "volume",
                ],
            )
            raw = self._annotate_symbols(raw)
            raw = self._assign_intraday_trade_dates(raw)
            raw = raw.loc[
                raw["root"].isin(tickers)
                & raw["is_concrete"]
            ].copy()
            raw = raw.loc[
                raw["trade_date"].between(
                    month_start.normalize(), month_end.normalize()
                )
            ]
            raw = self._deduplicate(
                raw, ["trade_datetime", "root", "symbol"]
            )
            if raw.empty:
                continue
            raw["position"] = pd.to_numeric(
                raw["position"], errors="coerce"
            ).clip(lower=0.0)
            raw["volume"] = pd.to_numeric(
                raw["volume"], errors="coerce"
            ).clip(lower=0.0)

            daily = self._read_partitions(
                "daily",
                month_start.normalize() - pd.Timedelta(days=15),
                month_end.normalize(),
                ["symbol", "trade_date", "position", "sequence"],
            )
            daily = self._annotate_symbols(daily)
            daily = daily.loc[
                daily["root"].isin(tickers)
                & daily["is_concrete"]
            ].copy()
            daily = self._deduplicate(
                daily, ["trade_date", "root", "symbol"]
            )
            daily["position"] = pd.to_numeric(
                daily["position"], errors="coerce"
            ).clip(lower=0.0)
            aggregate = self._aggregate_intraday_states(raw, daily)
            if not aggregate.empty:
                if resample_rule:
                    aggregate = self._resample_curve_long(
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
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        result = self._deduplicate(result, ["trade_datetime", "root"])
        dated = self._assign_intraday_trade_dates(
            result[["trade_datetime"]].copy()
        )
        start_date = pd.Timestamp(start).normalize()
        end_date = pd.Timestamp(end).normalize()
        result = result.loc[
            dated["trade_date"].between(start_date, end_date).to_numpy()
        ]
        return result.sort_values(["trade_datetime", "root"]).reset_index(drop=True)

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
            long = self._selected_long(
                tickers, start, end, frequency, available_fields
            )
            for field in available_fields:
                if field not in long:
                    continue
                panel = long.pivot(
                    index="trade_datetime", columns="root", values=field
                )
                panel.index = pd.DatetimeIndex(panel.index)
                panels[field] = panel.sort_index().reindex(columns=tickers)
        if curve_fields:
            if frequency == "daily":
                curve_long = self._daily_curve_long(tickers, start, end)
            else:
                curve_long = self._intraday_curve_long(
                    tickers, start, end, frequency
                )
            for field in curve_fields:
                if field not in curve_long:
                    continue
                panel = curve_long.pivot(
                    index="trade_datetime", columns="root", values=field
                )
                panel.index = pd.DatetimeIndex(panel.index)
                panels[field] = panel.sort_index().reindex(columns=tickers)
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
        plan = self._continuous_plan(roots, start, end)
        if plan.empty:
            return pd.DataFrame(columns=roots, dtype=object)
        schedule = plan.pivot(
            index="trade_date", columns="root", values="contract"
        )
        schedule.index = pd.DatetimeIndex(schedule.index)
        return schedule.sort_index().reindex(columns=roots)

    def fetch_contract_curve_at_frequency(
        self,
        tickers,
        start,
        end,
        fields: List[str],
        frequency: str = "daily",
    ) -> pd.DataFrame:
        """Return exact concrete-contract rows for curve-based factors.

        Unlike :meth:`fetch_price_at_frequency`, this method does not select a
        dominant contract or adjust prices.  It centralizes partition checks,
        alias canonicalization, exact root parsing, economic epochs, exchange
        trading dates and duplicate handling for factors that genuinely need
        more than one listed contract.
        """
        frequency = _normalise_frequency(frequency)
        roots = self._normalise_tickers(tickers)
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if not roots or start_ts > end_ts:
            return pd.DataFrame()

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
        raw = self._read_partitions(
            native_frequency, read_start, end_ts, raw_columns, roots=roots
        )
        raw = self._annotate_symbols(raw)
        if raw.empty:
            return pd.DataFrame()
        if native_frequency != "daily":
            raw = self._assign_intraday_trade_dates(raw)
        raw = raw.loc[
            raw["root"].isin(roots)
            & raw["is_concrete"]
            & raw["trade_date"].between(
                start_ts.normalize(), end_ts.normalize()
            )
        ].copy()
        if raw.empty:
            return pd.DataFrame()

        raw = self._deduplicate(
            raw, ["trade_datetime", "root", "symbol"]
        )
        rename = {
            source_field: requested_field
            for requested_field, source_field in field_map.items()
            if requested_field in requested and source_field != requested_field
        }
        raw = raw.rename(columns=rename)
        keep = [
            "trade_datetime", "trade_date", "root", "symbol",
            *requested,
        ]
        raw = raw.loc[:, list(dict.fromkeys(keep))]

        if resample_rule and native_frequency != "daily":
            raw["_bar_time"] = pd.to_datetime(
                raw["trade_datetime"]
            ).dt.floor(resample_rule)
            aggregations = {
                field: _BAR_AGGREGATIONS[field] for field in requested
            }
            raw = (
                raw.sort_values("_bar_time")
                .groupby(
                    ["_bar_time", "trade_date", "root", "symbol"],
                    sort=True,
                    as_index=False,
                )
                .agg(aggregations)
                .rename(columns={"_bar_time": "trade_datetime"})
            )
        return raw.sort_values(
            ["trade_datetime", "root", "symbol"]
        ).reset_index(drop=True)

    def fetch_contract_pair_prices(
        self, tickers, start, end, field: str = "close"
    ) -> Dict[str, pd.DataFrame]:
        """Return the first two observable maturities on each trading day."""
        roots = self._normalise_tickers(tickers)
        if field not in {"open", "high", "low", "close", "settle"}:
            raise ValueError(f"unsupported contract-pair field: {field!r}")
        curve = self.fetch_contract_curve_at_frequency(
            roots,
            start,
            end,
            [field, "position"],
            frequency="daily",
        )
        calendar = self.fetch_calendar(start, end)
        empty = pd.DataFrame(index=calendar, columns=roots, dtype=float)
        if curve.empty:
            return {"near": empty.copy(), "far": empty.copy()}

        value = pd.to_numeric(curve[field], errors="coerce")
        position = pd.to_numeric(curve["position"], errors="coerce")
        curve = curve.loc[
            value.gt(0.0) & np.isfinite(value) & position.gt(0.0)
        ].copy()
        if curve.empty:
            return {"near": empty.copy(), "far": empty.copy()}
        parts = contract_symbol_parts(curve["symbol"])
        curve["delivery_year"] = parts["delivery_year"].to_numpy()
        curve["delivery_month"] = parts["delivery_month"].to_numpy()
        curve = curve.dropna(subset=["delivery_year", "delivery_month"])
        curve = curve.sort_values([
            "trade_date", "root", "delivery_year", "delivery_month", "symbol"
        ])
        curve["_maturity_rank"] = (
            curve.groupby(["trade_date", "root"]).cumcount() + 1
        )

        result = {}
        for label, rank in (("near", 1), ("far", 2)):
            selected = curve.loc[curve["_maturity_rank"].eq(rank)]
            panel = selected.pivot(
                index="trade_date", columns="root", values=field
            )
            panel.index = pd.DatetimeIndex(panel.index)
            result[label] = panel.reindex(index=calendar, columns=roots)
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
        frame = self._read_partitions(
            "daily", timestamp - pd.Timedelta(days=45), timestamp,
            ["symbol", "trade_date"],
        )
        frame = self._annotate_symbols(frame)
        roots = sorted(frame.loc[frame["is_concrete"], "root"].unique())
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
        frame = self._annotate_symbols(pd.concat([
            pd.read_parquet(path, columns=["symbol", "trade_date"])
            .groupby("symbol", as_index=False)["trade_date"].min()
            for path in files
        ], ignore_index=True))
        listing = (
            frame.loc[frame["is_concrete"] & frame["root"].isin(roots)]
            .groupby("root")["trade_date"].min()
        )
        return pd.to_datetime(listing.reindex(roots)).astype("datetime64[ns]")

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
        latest = max(
            pd.to_datetime(
                pd.read_parquet(path, columns=["trade_date"])["trade_date"],
                errors="coerce",
            ).max()
            for path in files
        )
        if pd.isna(latest):
            raise RuntimeError(
                f"latest daily parquet shards contain no valid trade_date: {directory}"
            )
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
        frame = self._read_partitions(
            "daily", start_ts, end_ts, ["symbol", "trade_date"]
        )
        frame = self._annotate_symbols(frame)
        if frame.empty:
            res = pd.DatetimeIndex([])
        else:
            dates = frame.loc[frame["is_concrete"], "trade_date"].dropna().unique()
            res = pd.DatetimeIndex(sorted(dates))
        # LRU: 最多缓存 32 个日历 (回测窗口固定, 命中率高)
        if len(self._CALENDAR_CACHE) >= 32:
            self._CALENDAR_CACHE.pop(next(iter(self._CALENDAR_CACHE)))
        self._CALENDAR_CACHE[key] = res
        return res

__all__ = ["MissingParquetPartitionError", "ParquetFuturesSource"]
