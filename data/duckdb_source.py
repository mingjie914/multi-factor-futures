"""Read-only DuckDB adapter for the existing local futures algorithms."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import threading
from typing import Iterable, Optional

import duckdb
import pandas as pd
import polars as pl

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

    def _execute_polars(self, sql: str, params=None) -> pl.DataFrame:
        """Execute one query and preserve DuckDB's native Polars result."""
        with self._db_lock:
            if self._db is None:
                raise RuntimeError("DuckDB source is closed")
            return self._db.execute(sql, params or []).pl()

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
        return _fingerprint({
            "schema_version": _SCHEMA_VERSION,
            "partitions": sorted(str(path) for path in files),
        })

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

    def checkpoint_source_fingerprint(self, start, end) -> str:
        """Fingerprint relevant partitions without coupling to the latest release."""
        first = str(
            (pd.Timestamp(start).normalize() - pd.Timedelta(
                days=self.schedule_buffer_days
            )).to_period("M")
        )
        last = str(
            (pd.Timestamp(end).normalize() + pd.Timedelta(days=7)).to_period("M")
        )
        with self._db_lock:
            if self._db is None:
                raise RuntimeError("DuckDB source is closed")
            rows = self._db.execute(
                "SELECT component, dataset, year_month, source_files_sha256 "
                "FROM meta.partitions WHERE year_month >= ? AND year_month <= ? "
                "ORDER BY component, dataset, year_month",
                [first, last],
            ).fetchall()
        return _fingerprint({
            "schema_version": _SCHEMA_VERSION,
            "first_month": first,
            "last_month": last,
            "partitions": rows,
        })

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

    def _storage_query(
        self,
        native_frequency: str,
        start,
        end,
        columns: Iterable[str],
        roots: Iterable[str] | None = None,
    ) -> tuple[str | None, list, list[str], list[str], set[str], dict, list[tuple]]:
        """Build the one filtered storage query shared by both frame backends."""
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
            return None, [], [], [], available, types, described
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
        params = [pd.Timestamp(start).date(), pd.Timestamp(end).date()]
        normalised_roots = tuple(dict.fromkeys(
            str(root).strip().upper() for root in (roots or ()) if str(root).strip()
        ))
        if normalised_roots:
            clauses = [
                'regexp_matches(upper(trim("symbol")), ?)'
                for _ in normalised_roots
            ]
            sql += " AND (" + " OR ".join(clauses) + ")"
            params.extend(f"^{re.escape(root)}[0-9]+$" for root in normalised_roots)
        if order:
            sql += " ORDER BY " + ", ".join(f'"{column}"' for column in order)
        return sql, params, requested, selected, available, types, described

    def _read_storage_partitions_polars(
        self,
        native_frequency: str,
        start,
        end,
        columns: Iterable[str],
        roots: Iterable[str] | None = None,
    ) -> tuple[pl.DataFrame, list[str], set[str]]:
        """Read one filtered raw market slice without crossing into pandas."""
        sql, params, requested, selected, available, types, described = (
            self._storage_query(
                native_frequency, start, end, columns, roots
            )
        )
        if sql is None:
            return pl.DataFrame(), requested, available
        frame = self._execute_polars(sql, params)
        timestamp_unit = self._timestamp_unit(
            native_frequency, start, end, described
        )
        casts = []
        for column in selected:
            if column not in frame.columns:
                continue
            if types[column] == "DATE":
                casts.append(pl.col(column).cast(pl.Date))
            elif types[column].startswith("TIMESTAMP"):
                casts.append(pl.col(column).cast(pl.Datetime(timestamp_unit)))
        if casts:
            frame = frame.with_columns(casts)
        return frame, requested, available

    def _read_selected_partitions_polars(
        self,
        native_frequency: str,
        start,
        end,
        columns: Iterable[str],
        roots: tuple[str, ...],
        plan: pl.DataFrame,
    ) -> pl.DataFrame:
        """Join the causal contract plan before minute rows leave DuckDB."""
        if plan.is_empty():
            return pl.DataFrame()
        _, _, requested, selected, _, types, described = self._storage_query(
            native_frequency, start, end, columns, roots
        )
        if not requested:
            return pl.DataFrame()
        _, table = _MARKET_TABLES[native_frequency]
        projection = ", ".join(f'b."{column}"' for column in selected)
        order = [
            column
            for column in ("trade_datetime", "exchange", "symbol", "sequence")
            if column in selected
        ]
        sql = (
            f"SELECT {projection} FROM {table} b JOIN _mf_selected_plan p ON "
            "("
            "upper(trim(b.symbol)) = p.contract OR ("
            "upper(trim(b.exchange)) IN ('CZC', 'CZCE', 'ZCE') AND "
            "upper(trim(b.symbol)) = regexp_extract(p.contract, '^([A-Z]+)', 1) "
            "|| substr(regexp_extract(p.contract, '([0-9]{4})$', 1), 2, 3))) "
            "WHERE b.trade_date >= ? AND b.trade_date <= ?"
        )
        if order:
            sql += " ORDER BY " + ", ".join(f'b."{column}"' for column in order)
        with self._db_lock:
            if self._db is None:
                raise RuntimeError("DuckDB source is closed")
            self._db.register(
                "_mf_selected_plan",
                plan.select("contract").unique().to_arrow(),
            )
            try:
                frame = self._db.execute(
                    sql,
                    [pd.Timestamp(start).date(), pd.Timestamp(end).date()],
                ).pl()
            finally:
                self._db.unregister("_mf_selected_plan")
        timestamp_unit = self._timestamp_unit(
            native_frequency, start, end, described
        )
        casts = []
        for column in selected:
            if column not in frame.columns:
                continue
            if types[column] == "DATE":
                casts.append(pl.col(column).cast(pl.Date))
            elif types[column].startswith("TIMESTAMP"):
                casts.append(pl.col(column).cast(pl.Datetime(timestamp_unit)))
        return frame.with_columns(casts) if casts else frame

    def fetch_term_contracts_at_frequency(
        self, tickers, start, end, frequency: str = "1min"
    ) -> Optional[pd.DataFrame]:
        """Push the exact near/far candidate reduction into DuckDB."""

        frequency = str(frequency).strip().lower()
        if frequency in {"1min", "1m"}:
            native_frequency, table = "1min", "market.bars_1m"
        elif frequency in {"5min", "5m"}:
            native_frequency, table = "5min", "market.bars_5m"
        else:
            return None
        roots = self._normalise_tickers(tickers)
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if not roots or start_ts > end_ts:
            return pd.DataFrame()
        read_start = start_ts.normalize() - pd.Timedelta(days=7)
        self._month_files(native_frequency, read_start, end_ts)
        placeholders = ", ".join("?" for _ in roots)
        sql = f"""
            WITH candidates AS (
                SELECT
                    trade_datetime,
                    trade_date,
                    upper(trim(symbol)) AS symbol,
                    close,
                    position,
                    volume,
                    regexp_extract(upper(trim(symbol)), '^([A-Z]+)[0-9]{{4}}$', 1) AS root,
                    try_cast(regexp_extract(upper(trim(symbol)), '([0-9]{{4}})$', 1) AS INTEGER) AS expiry
                FROM {table}
                WHERE trade_date >= ? AND trade_date <= ?
                  AND regexp_matches(upper(trim(symbol)), '^[A-Z]+[0-9]{{4}}$')
                  AND regexp_extract(upper(trim(symbol)), '^([A-Z]+)', 1)
                      IN ({placeholders})
            ), ranked AS (
                SELECT *, row_number() OVER (
                    PARTITION BY trade_datetime, root ORDER BY expiry, symbol
                ) AS maturity_rank
                FROM candidates
                WHERE close > 0 AND isfinite(close)
                  AND position > 0 AND isfinite(position)
            )
            SELECT trade_datetime, trade_date, root, symbol, close, position, volume,
                   maturity_rank AS _maturity_rank, expiry AS _expiry_code
            FROM ranked
            WHERE maturity_rank <= 2
            ORDER BY trade_datetime, root, expiry, symbol
        """
        frame = self._execute_polars(
            sql,
            [read_start.date(), end_ts.normalize().date(), *roots],
        )
        if frame.is_empty():
            return frame.to_pandas()
        frame = self._annotate_symbols_polars(frame)
        frame = self._assign_intraday_trade_dates_polars(frame)
        return frame.filter(
            pl.col("root").is_in(roots)
            & pl.col("is_concrete")
            & pl.col("trade_date").is_between(
                start_ts.normalize().date(), end_ts.normalize().date(),
                closed="both",
            )
        ).sort(["trade_datetime", "root", "symbol"]).to_pandas()

    def fetch_latest_trade_date(self) -> pd.Timestamp:
        frame = self._execute_polars(
            "SELECT MAX(trade_date) AS latest FROM market.bars_1d"
        )
        latest = frame.item(0, 0)
        if pd.isna(latest):
            raise RuntimeError("DuckDB daily table has no trade_date")
        return pd.Timestamp(latest).normalize()

    def fetch_listing_dates(self, tickers) -> pd.Series:
        roots = self._normalise_tickers(tickers)
        if not roots:
            return pd.Series(dtype="datetime64[ns]")
        frame = self._annotate_symbols_polars(self._execute_polars(
            "SELECT symbol, MIN(trade_date) AS trade_date "
            "FROM market.bars_1d GROUP BY symbol ORDER BY symbol"
        ))
        listing = frame.filter(
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
        frame = self._execute_polars(
            sql,
            [pd.Timestamp(start).date(), pd.Timestamp(end).date(), *normalised_roots],
        )
        casts = [
            pl.col(column).cast(pl.Date)
            for column in selected
            if types[column] == "DATE" and column in frame.columns
        ]
        if casts:
            frame = frame.with_columns(casts)
        return frame.to_pandas()


__all__ = ["DuckDBFuturesSource"]
