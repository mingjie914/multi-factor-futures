"""Read-only DuckDB adapter for the existing local futures algorithms."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
from typing import Iterable, Optional

import duckdb
import pandas as pd

from core.registry import register
from data.parquet_source import MissingParquetPartitionError, ParquetFuturesSource


_MARKET_TABLES = {
    "daily": ("1d", "market.bars_1d"),
    "1min": ("1m", "market.bars_1m"),
    "5min": ("5m", "market.bars_5m"),
    "15min": ("15m", "market.bars_15m"),
}
_SCHEMA_VERSION = "futures_data_v1_seat_contract_v2"
_SEAT_TABLES = {
    "raw_seat_position": ("trade_date", "seat.raw_seat_position"),
    "derive_product_seat": ("trade_date", "seat.derive_product_seat"),
    "derive_main_contract_seat": ("trade_date", "seat.derive_main_contract_seat"),
    "derive_product_daily": ("trade_date", "seat.derive_product_daily"),
    "delivery_seat": ("delivery_date", "seat.delivery_seat"),
    "delivery_summary": ("delivery_date", "seat.delivery_summary"),
}


def _fingerprint(payload) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@register("data_source", "duckdb_futures")
class DuckDBFuturesSource(ParquetFuturesSource):
    """Reuse all continuous-contract and frequency semantics over DuckDB rows."""

    def __init__(
        self,
        duckdb_config: Optional[dict] = None,
        parquet_config: Optional[dict] = None,
    ) -> None:
        super().__init__(parquet_config=parquet_config, _validate_storage=False)
        config = dict(duckdb_config or {})
        path_text = str(config.get("path", "")).strip()
        if not path_text:
            raise ValueError("duckdb.path is required")
        self.database_path = Path(path_text).expanduser().resolve()
        if not self.database_path.is_file():
            raise FileNotFoundError(self.database_path)
        self._db_lock = threading.RLock()
        self._db = duckdb.connect(str(self.database_path), read_only=True)
        releases = self._db.execute(
            "SELECT release_id, market_component_id, seat_component_id, status, "
            "schema_version "
            "FROM meta.releases WHERE is_current"
        ).fetchall()
        if (
            len(releases) != 1
            or releases[0][3] != "certified"
            or releases[0][4] != _SCHEMA_VERSION
        ):
            self._db.close()
            raise RuntimeError(
                f"expected one certified current DuckDB release, found {releases}"
            )
        self.release_id = str(releases[0][0])
        self.market_component_id = str(releases[0][1])
        self.seat_component_id = str(releases[0][2])
        required = str(config.get("required_release_id", "")).strip()
        if required and required != self.release_id:
            self._db.close()
            raise RuntimeError(
                f"DuckDB release {self.release_id} does not match required {required}"
            )
        self.cache_namespace = f"DuckDBFuturesSource_v1:{self.release_id}"

    def close(self) -> None:
        with self._db_lock:
            if self._db is not None:
                self._db.close()
                self._db = None

    def _execute_df(self, sql: str, params=None) -> pd.DataFrame:
        with self._db_lock:
            if self._db is None:
                raise RuntimeError("DuckDB source is closed")
            return self._db.execute(sql, params or []).fetchdf()

    def _partition_rows(self, native_frequency: str) -> list[tuple[str, str]]:
        dataset, _ = _MARKET_TABLES[native_frequency]
        with self._db_lock:
            if self._db is None:
                raise RuntimeError("DuckDB source is closed")
            return [
                (str(month), str(source_sha))
                for month, source_sha in self._db.execute(
                    "SELECT year_month, source_files_sha256 FROM meta.partitions "
                    "WHERE component = 'market' AND dataset = ? ORDER BY year_month",
                    [dataset],
                ).fetchall()
            ]

    def _month_files(self, native_frequency: str, start, end) -> list[Path]:
        available = dict(self._partition_rows(native_frequency))
        periods = list(pd.period_range(
            pd.Timestamp(start).to_period("M"),
            pd.Timestamp(end).to_period("M"),
            freq="M",
        ))
        if not available:
            return []
        first = pd.Period(min(available), freq="M")
        last = pd.Period(max(available), freq="M")
        missing = [
            period for period in periods
            if first <= period <= last and str(period) not in available
        ]
        if missing:
            raise MissingParquetPartitionError(
                f"missing {native_frequency} DuckDB month(s): "
                + ", ".join(str(period) for period in missing)
            )
        return [
            Path(native_frequency) / f"year_month={period}" / available[str(period)]
            for period in periods if str(period) in available
        ]

    def _files_fingerprint(self, _root_path: Path, files: Iterable[Path]) -> str:
        return _fingerprint((self.market_component_id, sorted(str(path) for path in files)))

    def _selected_cache_source_fingerprint(
        self, native_frequency: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> str:
        files = self._month_files(
            native_frequency, start.normalize() - pd.Timedelta(days=7), end
        )
        files.extend(self._month_files(
            "daily",
            start.normalize() - pd.Timedelta(days=self.schedule_buffer_days),
            end.normalize() + pd.Timedelta(days=7),
        ))
        return self._files_fingerprint(self.root_path, files)

    def _curve_cache_source_fingerprint(
        self,
        native_frequency: str,
        month_start: pd.Timestamp,
        month_end: pd.Timestamp,
    ) -> str:
        files = self._month_files(
            native_frequency, month_start - pd.Timedelta(days=7), month_end
        )
        files.extend(self._month_files(
            "daily", month_start - pd.Timedelta(days=15),
            month_end + pd.Timedelta(days=7),
        ))
        return self._files_fingerprint(self.root_path, files)

    def _timestamp_unit(
        self,
        native_frequency: str,
        start,
        end,
        described: list[tuple],
    ) -> str:
        dataset, _ = _MARKET_TABLES[native_frequency]
        months = [
            str(period) for period in pd.period_range(
                pd.Timestamp(start).to_period("M"),
                pd.Timestamp(end).to_period("M"), freq="M",
            )
        ]
        placeholders = ", ".join("?" for _ in months)
        with self._db_lock:
            if self._db is None:
                raise RuntimeError("DuckDB source is closed")
            hashes = {
                str(row[0]) for row in self._db.execute(
                    "SELECT schema_sha256 FROM meta.partitions WHERE dataset = ? "
                    f"AND year_month IN ({placeholders})",
                    [dataset, *months],
                ).fetchall()
            }
        variants = {}
        for unit, duck_type in (("us", "TIMESTAMP"), ("ns", "TIMESTAMP_NS")):
            schema = [
                tuple(
                    duck_type if index == 1 and str(row[0]) == "trade_datetime"
                    else value
                    for index, value in enumerate(row)
                )
                for row in described
            ]
            variants[_fingerprint(schema)] = unit
        unknown = hashes - set(variants)
        if unknown:
            raise RuntimeError(
                f"unknown {native_frequency} partition schema hashes: {sorted(unknown)}"
            )
        units = {variants[value] for value in hashes}
        return "ns" if "ns" in units else "us"

    def _read_storage_partitions(
        self,
        native_frequency: str,
        start,
        end,
        columns: Iterable[str],
    ) -> tuple[pd.DataFrame, list[str], set[str]]:
        self._month_files(native_frequency, start, end)
        _, table = _MARKET_TABLES[native_frequency]
        with self._db_lock:
            if self._db is None:
                raise RuntimeError("DuckDB source is closed")
            described = self._db.execute(f"DESCRIBE {table}").fetchall()
            types = {str(row[0]): str(row[1]) for row in described}
            available = set(types)
        requested_columns = list(dict.fromkeys(columns))
        requested = [column for column in requested_columns if column in available]
        if not requested:
            return pd.DataFrame(), [], available
        selected = list(requested)
        if "symbol" in selected:
            for column in ("exchange", "trade_date", "trade_datetime"):
                if column in available and column not in selected:
                    selected.append(column)
        order = [
            column for column in ("trade_datetime", "exchange", "symbol", "sequence")
            if column in available
        ]
        sql = (
            "SELECT " + ", ".join(f'"{column}"' for column in selected)
            + f" FROM {table} WHERE trade_date >= ? AND trade_date <= ?"
        )
        if order:
            sql += " ORDER BY " + ", ".join(f'"{column}"' for column in order)
        frame = self._execute_df(
            sql, [pd.Timestamp(start).date(), pd.Timestamp(end).date()]
        )
        timestamp_unit = self._timestamp_unit(
            native_frequency, start, end, described
        )
        for column in selected:
            if types[column] == "DATE" and column in frame:
                frame[column] = frame[column].astype("datetime64[s]")
            elif types[column].startswith("TIMESTAMP") and column in frame:
                frame[column] = frame[column].astype(f"datetime64[{timestamp_unit}]")
        return frame, requested, available

    def fetch_latest_trade_date(self) -> pd.Timestamp:
        frame = self._execute_df("SELECT MAX(trade_date) AS latest FROM market.bars_1d")
        latest = frame.iloc[0, 0]
        if pd.isna(latest):
            raise RuntimeError("DuckDB daily table has no trade_date")
        return pd.Timestamp(latest).normalize()

    def fetch_seat_table(
        self,
        table: str,
        start,
        end,
        roots: Iterable[str],
        columns: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        if table not in _SEAT_TABLES:
            raise ValueError(f"unsupported seat table {table!r}")
        date_column, qualified = _SEAT_TABLES[table]
        with self._db_lock:
            if self._db is None:
                raise RuntimeError("DuckDB source is closed")
            described = self._db.execute(f"DESCRIBE {qualified}").fetchall()
            types = {str(row[0]): str(row[1]) for row in described}
            available = list(types)
        selected = available if columns is None else list(dict.fromkeys(columns))
        unsupported = sorted(set(selected) - set(available))
        if unsupported:
            raise ValueError(f"unsupported {table} columns: {unsupported}")
        normalised_roots = tuple(dict.fromkeys(str(root).strip().upper() for root in roots))
        if not normalised_roots:
            return pd.DataFrame(columns=selected)
        placeholders = ", ".join("?" for _ in normalised_roots)
        order = [
            column for column in (
                date_column, "exchange", "root", "product_code", "contract_code",
                "seat_name", "is_aggregated",
            ) if column in available
        ]
        sql = (
            "SELECT " + ", ".join(f'"{column}"' for column in selected)
            + f" FROM {qualified} WHERE {date_column} >= ? AND {date_column} <= ? "
            + f"AND root IN ({placeholders}) ORDER BY "
            + ", ".join(f'"{column}"' for column in order)
        )
        frame = self._execute_df(
            sql,
            [pd.Timestamp(start).date(), pd.Timestamp(end).date(), *normalised_roots],
        )
        for column in selected:
            if types[column] == "DATE" and column in frame:
                frame[column] = pd.to_datetime(frame[column]).dt.date
        return frame


__all__ = ["DuckDBFuturesSource"]
