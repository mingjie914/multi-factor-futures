"""Read-only health check for the selected market-data source and local cache."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.config import load_config
from data.contract_symbols import (
    MARKET_FIELDS,
    canonicalize_contract_aliases,
    contract_symbol_parts,
)
from data.market_quality import prepare_close_data


def _model_dict(model) -> dict:
    if model is None:
        return {}
    if isinstance(model, dict):
        return dict(model)
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    if hasattr(model, "__dict__"):
        return vars(model)
    raise TypeError(f"cannot convert {type(model).__name__} to config mapping")


def _check_latest_parquet_contract_keys(parquet_config: dict) -> dict:
    """Validate the latest published partition without taxing backtest reads."""
    root_text = str(parquet_config.get("root_path", "")).strip()
    if not root_text:
        return {"status": "not_configured"}
    root = Path(root_text).expanduser().resolve()
    if not root.is_dir():
        return {"status": "unavailable", "path": str(root)}

    datasets = dict(parquet_config.get("datasets") or {})
    datasets = {
        "daily": datasets.get("daily", "futureshistoryprices1d"),
        "1min": datasets.get("1min", "futureshistoryprices1m"),
        "5min": datasets.get("5min", "futureshistoryprices5m"),
        "15min": datasets.get("15min", "futureshistoryprices15m"),
    }
    checks = {}
    invalid = False
    latest_daily_date = None
    for frequency, relative in datasets.items():
        dataset = root / str(relative)
        partitions = sorted(dataset.glob("year_month=*") if dataset.is_dir() else [])
        if not partitions:
            checks[frequency] = {"status": "missing", "path": str(dataset)}
            invalid = True
            continue
        try:
            periods = [
                pd.Period(path.name.split("=", 1)[1], freq="M")
                for path in partitions
            ]
        except (IndexError, ValueError) as exc:
            checks[frequency] = {
                "status": "invalid",
                "path": str(dataset),
                "message": f"invalid partition name: {exc}",
            }
            invalid = True
            continue
        expected = set(pd.period_range(min(periods), max(periods), freq="M"))
        missing_periods = sorted(expected.difference(periods))
        empty_periods = [
            period for period, path in zip(periods, partitions)
            if not any(path.glob("*.parquet"))
        ]
        if missing_periods or empty_periods:
            checks[frequency] = {
                "status": "invalid",
                "path": str(dataset),
                "first_partition": str(min(periods)),
                "last_partition": str(max(periods)),
                "missing_partitions": [str(item) for item in missing_periods],
                "empty_partitions": [str(item) for item in empty_periods],
            }
            invalid = True
            continue
        latest = partitions[-1]
        files = sorted(latest.glob("*.parquet"))
        if not files:
            checks[frequency] = {"status": "missing", "partition": str(latest)}
            invalid = True
            continue
        try:
            import pyarrow.parquet as pq

            available = set(pq.ParquetFile(files[0]).schema_arrow.names)
            columns = [
                column for column in (
                    "symbol", "exchange", "trade_date", "trade_datetime",
                    *MARKET_FIELDS,
                ) if column in available
            ]
            frames = [pd.read_parquet(path, columns=columns) for path in files]
            frame = pd.concat(frames, ignore_index=True)
            latest_trade_date = pd.to_datetime(
                frame["trade_date"], errors="coerce"
            ).max()
            if pd.isna(latest_trade_date):
                raise ValueError("latest partition has no valid trade_date")
            latest_trade_date = latest_trade_date.date().isoformat()
            if frequency == "daily":
                latest_daily_date = latest_trade_date
            exchanges = frame["exchange"].astype(str).str.strip().str.upper()
            symbols = frame["symbol"].astype(str).str.strip().str.upper()
            three_digit = int(
                (exchanges.isin({"CZC", "CZCE", "ZCE"})
                 & symbols.str.fullmatch(r"[A-Z]+\d{3}")).sum()
            )
            canonical = canonicalize_contract_aliases(frame)
            duplicate_rows = len(frame) - len(canonical)
            status = "ok" if three_digit == 0 and duplicate_rows == 0 else "invalid"
            invalid |= status != "ok"
            checks[frequency] = {
                "status": status,
                "partition": latest.name,
                "first_partition": str(min(periods)),
                "last_partition": str(max(periods)),
                "latest_trade_date": latest_trade_date,
                "rows": len(frame),
                "czce_three_digit_rows": three_digit,
                "duplicate_rows": duplicate_rows,
            }
        except Exception as exc:
            invalid = True
            checks[frequency] = {
                "status": "invalid",
                "partition": latest.name,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
    return {
        "status": "invalid" if invalid else "ok",
        "path": str(root),
        "latest_daily_date": latest_daily_date,
        "latest_partitions": checks,
    }


def _check_historical_parquet_contracts(
    parquet_config: dict,
    universe: list[str],
    start,
    end,
    audited_nontrading_closes: dict | None = None,
) -> dict:
    """Run the expensive daily, full-window gate used only by ``--strict``."""
    try:
        from data.parquet_source import ParquetFuturesSource

        source_config = dict(parquet_config)
        source_config.update({
            "eager_fields": False,
            "curve_cache_enabled": False,
            "selected_cache_enabled": False,
            "panel_cache_entries": 0,
        })
        roots = list(dict.fromkeys(str(root).strip().upper() for root in universe))
        if not roots:
            raise ValueError("configured universe is empty")
        source = ParquetFuturesSource(source_config)
        calendar = source.fetch_calendar(start, end)
        if calendar.empty:
            raise ValueError("daily parquet calendar is empty")
        if calendar.has_duplicates or not calendar.is_monotonic_increasing:
            raise ValueError("daily parquet calendar must be unique and sorted")

        schedule = source.fetch_contract_schedule(roots, start, end).reindex(
            index=calendar, columns=roots
        )
        panels = source.fetch_price(roots, start, end, ["close"])
        close = panels.get("close")
        if close is None or close.empty:
            raise ValueError("continuous close panel is empty")
        close = close.reindex(index=calendar, columns=roots)
        empty_roots = [root for root in roots if not close[root].notna().any()]
        if empty_roots:
            raise ValueError(
                "configured roots have no continuous close: " + ", ".join(empty_roots)
            )
        missing_schedule = close.notna() & schedule.isna()
        if bool(missing_schedule.any().any()):
            raise ValueError("observed continuous closes lack concrete contracts")

        contracts = schedule.stack().dropna().astype(str)
        parts = contract_symbol_parts(contracts)
        expected_roots = pd.Series(
            contracts.index.get_level_values(1), index=contracts.index, dtype=object
        )
        invalid_contract = ~parts["is_concrete"].astype(bool) | parts["root"].ne(
            expected_roots
        )
        if bool(invalid_contract.any()):
            sample = contracts.loc[invalid_contract].head(8).tolist()
            raise ValueError(f"invalid concrete contract schedule: {sample}")

        returns, tradable = prepare_close_data(
            close, audited_nontrading_closes
        )
        return {
            "status": "ok",
            "start": pd.Timestamp(calendar[0]).date().isoformat(),
            "end": pd.Timestamp(calendar[-1]).date().isoformat(),
            "trading_days": len(calendar),
            "roots": len(roots),
            "close_observations": int(close.notna().sum().sum()),
            "nontradable_marks": int((close.isna() & returns.notna()).sum().sum()),
            "tradable_observations": int(tradable.sum().sum()),
        }
    except Exception as exc:
        return {
            "status": "invalid",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }


def _check_all_daily_contract_keys(parquet_config: dict) -> dict:
    """Scan every daily partition for aliases, duplicate keys and schema drift."""
    root_text = str(parquet_config.get("root_path", "")).strip()
    if not root_text:
        return {"status": "not_configured"}
    root = Path(root_text).expanduser().resolve()
    datasets = dict(parquet_config.get("datasets") or {})
    dataset = root / str(datasets.get("daily", "futureshistoryprices1d"))
    partitions = sorted(dataset.glob("year_month=*") if dataset.is_dir() else [])
    if not partitions:
        return {"status": "invalid", "message": f"daily dataset missing: {dataset}"}

    required = {
        "symbol", "exchange", "trade_date", "trade_datetime",
        "open", "high", "low", "close", "volume", "position",
    }
    rows = canonical_rows = three_digit_rows = noncanonical_symbol_rows = 0
    first_date = last_date = None
    try:
        import pyarrow.parquet as pq

        for partition in partitions:
            files = sorted(partition.glob("*.parquet"))
            if not files:
                raise ValueError(f"empty daily partition: {partition.name}")
            frames = []
            for path in files:
                available = set(pq.ParquetFile(path).schema_arrow.names)
                missing = sorted(required - available)
                if missing:
                    raise ValueError(
                        f"daily schema missing {missing} in {path}"
                    )
                columns = [
                    column for column in (
                        "symbol", "exchange", "trade_date", "trade_datetime",
                        *MARKET_FIELDS,
                    ) if column in available
                ]
                frames.append(pd.read_parquet(path, columns=columns))
            frame = pd.concat(frames, ignore_index=True)
            symbols = frame["symbol"].astype(str)
            normalized = symbols.str.strip().str.upper()
            exchanges = frame["exchange"].astype(str).str.strip().str.upper()
            three_digit_rows += int(
                (
                    exchanges.isin({"CZC", "CZCE", "ZCE"})
                    & normalized.str.fullmatch(r"[A-Z]+\d{3}")
                ).sum()
            )
            noncanonical_symbol_rows += int(symbols.ne(normalized).sum())
            canonical = canonicalize_contract_aliases(frame)
            rows += len(frame)
            canonical_rows += len(canonical)
            dates = pd.to_datetime(frame["trade_date"], errors="raise")
            part_first, part_last = dates.min(), dates.max()
            first_date = part_first if first_date is None else min(first_date, part_first)
            last_date = part_last if last_date is None else max(last_date, part_last)
    except Exception as exc:
        return {
            "status": "invalid",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }

    duplicate_rows = rows - canonical_rows
    status = (
        "ok"
        if three_digit_rows == 0
        and duplicate_rows == 0
        and noncanonical_symbol_rows == 0
        else "invalid"
    )
    return {
        "status": status,
        "dataset": str(dataset),
        "partitions": len(partitions),
        "rows": rows,
        "first_trade_date": first_date.date().isoformat(),
        "last_trade_date": last_date.date().isoformat(),
        "czce_three_digit_rows": three_digit_rows,
        "duplicate_rows": duplicate_rows,
        "noncanonical_symbol_rows": noncanonical_symbol_rows,
    }


def _check_all_parquet_schemas(parquet_config: dict) -> dict:
    """Validate every published Parquet file using metadata only."""
    root_text = str(parquet_config.get("root_path", "")).strip()
    if not root_text:
        return {"status": "not_configured"}
    root = Path(root_text).expanduser().resolve()
    datasets = dict(parquet_config.get("datasets") or {})
    datasets = {
        "daily": datasets.get("daily", "futureshistoryprices1d"),
        "1min": datasets.get("1min", "futureshistoryprices1m"),
        "5min": datasets.get("5min", "futureshistoryprices5m"),
        "15min": datasets.get("15min", "futureshistoryprices15m"),
    }
    common_required = {
        "exchange", "symbol", "trade_datetime", "trade_date",
        "open", "high", "low", "close", "volume", "amount", "position",
    }
    checks = {}
    invalid = False
    try:
        import pyarrow.parquet as pq

        for frequency, relative in datasets.items():
            dataset = root / str(relative)
            files = sorted(dataset.glob("year_month=*/*.parquet"))
            if not files:
                checks[frequency] = {
                    "status": "invalid", "message": f"dataset has no parquet files: {dataset}"
                }
                invalid = True
                continue
            required = set(common_required)
            if frequency == "daily":
                required.update({"sequence", "settle_price", "pre_settle_price"})
            variants: dict[tuple[str, ...], int] = {}
            missing_examples = []
            for path in files:
                schema = tuple(pq.ParquetFile(path).schema_arrow.names)
                variants[schema] = variants.get(schema, 0) + 1
                missing = sorted(required.difference(schema))
                if missing and len(missing_examples) < 5:
                    missing_examples.append({"file": str(path), "columns": missing})
            status = "ok" if len(variants) == 1 and not missing_examples else "invalid"
            invalid |= status != "ok"
            checks[frequency] = {
                "status": status,
                "dataset": str(dataset),
                "files": len(files),
                "schema_variants": len(variants),
                "missing_required_examples": missing_examples,
            }
    except Exception as exc:
        return {
            "status": "invalid",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "datasets": checks,
        }
    return {"status": "invalid" if invalid else "ok", "datasets": checks}


def _check_all_seat_parquet_keys(parquet_config: dict) -> dict:
    """Validate every seat table consumed by the current factor library."""
    root_text = str(parquet_config.get("root_path", "")).strip()
    if not root_text:
        return {"status": "not_configured"}
    root = Path(root_text).expanduser().resolve()
    relative = str(parquet_config.get("seat_dataset", "futuresseatdata")).strip()
    if not relative:
        return {"status": "invalid", "message": "seat_dataset must not be empty"}
    seat_root = root / relative
    specs = {
        "derive_product_daily": {
            "date": "trade_date",
            "keys": ("trade_date", "exchange", "root"),
            "required": (
                "product_code", "total_long", "total_short", "net_position",
                "long_change", "short_change", "seat_count",
            ),
        },
        "derive_product_seat": {
            "date": "trade_date",
            "keys": ("trade_date", "exchange", "root", "seat_name"),
            "required": (
                "product_code", "long_position", "long_change",
                "short_position", "short_change", "net_position",
                "contract_count",
            ),
        },
        "derive_main_contract_seat": {
            "date": "trade_date",
            "keys": (
                "trade_date", "exchange", "root", "contract_code", "seat_name",
            ),
            "required": (
                "product_code", "symbol", "long_position", "long_change",
                "short_position", "short_change", "net_position", "close",
                "open_interest",
            ),
        },
        "raw_seat_position": {
            "date": "trade_date",
            "keys": (
                "trade_date", "exchange", "root", "contract_code", "seat_name",
                "is_aggregated", "record_grain",
            ),
            "required": (
                "product_code", "symbol", "long_position", "long_change",
                "short_position", "short_change", "net_position",
            ),
        },
        "delivery_summary": {
            "date": "delivery_date",
            "keys": ("delivery_date", "exchange", "root", "contract_code"),
            "required": (
                "product_code", "product_name", "symbol", "receive_quantity",
                "deliver_quantity", "receive_seat_count", "deliver_seat_count",
                "non_futures_net",
            ),
        },
        "delivery_seat": {
            "date": "delivery_date",
            "keys": (
                "delivery_date", "exchange", "root", "contract_code", "seat_name",
            ),
            "required": (
                "product_code", "symbol", "long_position", "long_change",
                "short_position", "short_change", "net_position",
            ),
        },
    }
    checks = {}
    invalid = False
    connection = None
    try:
        import duckdb
        import pyarrow.parquet as pq

        connection = duckdb.connect()
        for name, spec in specs.items():
            dataset = seat_root / name
            partitions = sorted(dataset.glob("year_month=*") if dataset.is_dir() else [])
            files = sorted(dataset.glob("year_month=*/*.parquet"))
            if not partitions or not files:
                checks[name] = {
                    "status": "invalid",
                    "message": f"seat dataset has no parquet files: {dataset}",
                }
                invalid = True
                continue
            periods = [
                pd.Period(path.name.split("=", 1)[1], freq="M") for path in partitions
            ]
            expected = set(pd.period_range(min(periods), max(periods), freq="M"))
            missing_partitions = sorted(expected.difference(periods))
            empty_partitions = [
                period for period, path in zip(periods, partitions)
                if not any(path.glob("*.parquet"))
            ]
            required = set(spec["keys"]).union(spec["required"])
            variants: dict[tuple[str, ...], int] = {}
            missing_examples = []
            for path in files:
                schema = tuple(pq.ParquetFile(path).schema_arrow.names)
                variants[schema] = variants.get(schema, 0) + 1
                missing = sorted(required.difference(schema))
                if missing and len(missing_examples) < 5:
                    missing_examples.append({"file": str(path), "columns": missing})

            if missing_examples:
                checks[name] = {
                    "status": "invalid",
                    "dataset": str(dataset),
                    "files": len(files),
                    "schema_variants": len(variants),
                    "missing_required_examples": missing_examples,
                }
                invalid = True
                continue

            glob_path = (dataset / "year_month=*" / "*.parquet").as_posix().replace(
                "'", "''"
            )
            keys = ", ".join(f'"{column}"' for column in spec["keys"])
            null_key = " OR ".join(
                f'"{column}" IS NULL' for column in spec["keys"]
            )
            date_column = spec["date"]
            source_sql = f"read_parquet('{glob_path}', hive_partitioning=true)"
            rows, null_key_rows, bad_root_rows, first_date, last_date = connection.execute(
                f"""
                SELECT
                    COUNT(*),
                    SUM(CASE WHEN {null_key} THEN 1 ELSE 0 END),
                    SUM(CASE WHEN root IS NULL
                                  OR CAST(root AS VARCHAR) != UPPER(TRIM(CAST(root AS VARCHAR)))
                                  OR NOT regexp_full_match(TRIM(CAST(root AS VARCHAR)), '[A-Z]+')
                             THEN 1 ELSE 0 END),
                    MIN("{date_column}"),
                    MAX("{date_column}")
                FROM {source_sql}
                """
            ).fetchone()
            duplicate_groups = connection.execute(
                f"""
                SELECT COUNT(*) FROM (
                    SELECT {keys}, COUNT(*) AS n
                    FROM {source_sql}
                    GROUP BY {keys}
                    HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
            status = "ok" if (
                len(variants) == 1
                and not missing_partitions
                and not empty_partitions
                and int(null_key_rows or 0) == 0
                and int(bad_root_rows or 0) == 0
                and int(duplicate_groups) == 0
            ) else "invalid"
            invalid |= status != "ok"
            checks[name] = {
                "status": status,
                "dataset": str(dataset),
                "partitions": len(partitions),
                "files": len(files),
                "rows": int(rows),
                "first_date": pd.Timestamp(first_date).date().isoformat(),
                "last_date": pd.Timestamp(last_date).date().isoformat(),
                "schema_variants": len(variants),
                "missing_partitions": [str(item) for item in missing_partitions],
                "empty_partitions": [str(item) for item in empty_partitions],
                "null_key_rows": int(null_key_rows or 0),
                "noncanonical_root_rows": int(bad_root_rows or 0),
                "duplicate_key_groups": int(duplicate_groups),
            }
    except Exception as exc:
        return {
            "status": "invalid",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "datasets": checks,
        }
    finally:
        if connection is not None:
            connection.close()
    return {"status": "invalid" if invalid else "ok", "datasets": checks}


def check_health(config_path: str, *, strict: bool = False) -> dict:
    config = load_config(config_path)
    selected_source = str(config.data.source)
    result = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(config_path).resolve()),
        "selected_source": selected_source,
        "parquet": {"status": "not_selected"},
        "historical_parquet_schemas": {"status": "not_checked"},
        "historical_seat_tables": {"status": "not_checked"},
        "historical_daily_contract_keys": {"status": "not_checked"},
        "historical_daily": {"status": "not_checked"},
        "cache": {},
    }

    if selected_source != "parquet_futures":
        result["parquet"] = {
            "status": "invalid",
            "message": "framework data source must be parquet_futures",
        }
    else:
        parquet_config = _model_dict(config.data.parquet)
        result["parquet"] = _check_latest_parquet_contract_keys(parquet_config)
        if strict and result["parquet"].get("status") == "ok":
            result["historical_parquet_schemas"] = _check_all_parquet_schemas(
                parquet_config
            )
            result["historical_seat_tables"] = _check_all_seat_parquet_keys(
                parquet_config
            )
            result["historical_daily_contract_keys"] = (
                _check_all_daily_contract_keys(parquet_config)
            )
            historical_end = config.date_range.end
            latest_daily_date = result["parquet"].get("latest_daily_date")
            if latest_daily_date and pd.Timestamp(latest_daily_date) > pd.Timestamp(
                historical_end
            ):
                historical_end = latest_daily_date
            result["historical_daily"] = _check_historical_parquet_contracts(
                parquet_config,
                list(config.universe),
                config.date_range.start,
                historical_end,
                _model_dict(config.data).get("audited_nontrading_closes", {}),
            )

    cache_path = Path(config.data.cache.get("path", "./cache"))
    files = list(cache_path.rglob("*.parquet")) if cache_path.is_dir() else []
    result["cache"] = {
        "status": "ok" if files else "empty",
        "path": str(cache_path.resolve()),
        "parquet_files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "latest_modified": (
            datetime.fromtimestamp(
                max(path.stat().st_mtime for path in files), timezone.utc
            ).isoformat()
            if files else None
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument(
        "--strict", action="store_true", help="Exit non-zero if the selected source fails"
    )
    args = parser.parse_args()

    result = check_health(args.config, strict=args.strict)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")

    if args.strict:
        selected_ok = result["parquet"].get("status") == "ok"
        schemas_ok = result["historical_parquet_schemas"].get("status") == "ok"
        seat_ok = result["historical_seat_tables"].get("status") == "ok"
        keys_ok = result["historical_daily_contract_keys"].get("status") == "ok"
        historical_ok = result["historical_daily"].get("status") == "ok"
        if (
            not selected_ok or not schemas_ok or not seat_ok
            or not keys_ok or not historical_ok
        ):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
