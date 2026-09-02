from __future__ import annotations

from datetime import date
import importlib.util
from pathlib import Path
import sys

import duckdb
import pandas as pd
import polars as pl
import pytest

from core.config import load_config
from data.duckdb_source import DuckDBFuturesSource
from data.parquet_source import ParquetFuturesSource


UPDATER_PATH = (
    Path(__file__).resolve().parents[2]
    / "期货行情数据" / "load_pipeline" / "update_duckdb.py"
)


def _load_updater():
    if not UPDATER_PATH.is_file():
        pytest.skip(f"DuckDB publisher is outside this checkout: {UPDATER_PATH}")
    spec = importlib.util.spec_from_file_location("local_update_duckdb", UPDATER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _market_row(symbol, timestamp, close, position):
    timestamp = pd.Timestamp(timestamp)
    return {
        "exchange": "TEST", "symbol": symbol, "trade_datetime": timestamp,
        "open": close - 1.0, "high": close + 2.0, "low": close - 2.0,
        "close": float(close), "settle_price": close - 0.5,
        "pre_settle_price": close - 1.5, "volume": 10, "amount": close * 10.0,
        "position": position, "type": 0, "sequence": 0,
        "trade_date": timestamp.date(),
    }


def _write_month(table_root: Path, name: str, rows) -> Path:
    directory = table_root / name / "year_month=2024-01"
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(directory / "part.parquet", index=False)
    return directory / "part.parquet"


def _fixture(tmp_path: Path, *, build: bool = True):
    updater = _load_updater()
    root = tmp_path / "data"
    table_root = root / "本地表"
    daily = []
    for day, main in (("2024-01-02", "A2401"), ("2024-01-03", "A2405"),
                      ("2024-01-04", "A2405")):
        daily.extend([
            _market_row("A2401", day, 100.0, 1000 if main == "A2401" else 500),
            _market_row("A2405", day, 120.0, 1000 if main == "A2405" else 500),
        ])
    _write_month(table_root, "futureshistoryprices1d", daily)
    minute = [
        _market_row("A2401", pd.Timestamp("2024-01-03 09:00") + pd.Timedelta(minutes=i),
                    100.0 + i, 1000 + i)
        for i in range(6)
    ]
    for name in ("futureshistoryprices1m", "futureshistoryprices5m"):
        _write_month(table_root, name, minute)
    fifteen = [minute[0], _market_row("A2401", "2024-01-03 09:15", 106.0, 1010)]
    _write_month(table_root, "futureshistoryprices15m", fifteen)

    seat_root = table_root / "futuresseatdata"
    common = {
        "trade_date": pd.Timestamp("2024-01-03").date(), "exchange": "TEST",
        "root": "A", "product_code": "A.TEST", "seat_name": "SEAT",
        "long_position": 10, "long_change": 1, "short_position": 4,
        "short_change": 0, "net_position": 6,
    }
    rows = {
        "raw_seat_position": [{**common, "contract_code": "A2401.TEST",
            "symbol": "A2401", "is_aggregated": 0, "record_grain": "contract"}],
        "derive_product_seat": [{**common, "contract_count": 1}],
        "derive_main_contract_seat": [{**common, "contract_code": "A2401.TEST",
            "symbol": "A2401", "close": 100.0, "open_interest": 1000}],
        "derive_product_daily": [{key: value for key, value in common.items()
            if key not in {"seat_name", "long_position", "short_position"}} | {
                "total_long": 10, "total_short": 4, "seat_count": 1}],
        "delivery_seat": [{**common, "delivery_date": common["trade_date"],
            "contract_code": "A2401.TEST", "symbol": "A2401"}],
        "delivery_summary": [{
            "delivery_date": common["trade_date"], "exchange": "TEST", "root": "A",
            "product_code": "A.TEST", "product_name": "A", "contract_code": "A2401.TEST",
            "symbol": "A2401", "receive_quantity": 10, "deliver_quantity": 4,
            "receive_seat_count": 1, "deliver_seat_count": 1, "non_futures_net": 6,
        }],
    }
    for name, frame_rows in rows.items():
        frame = pd.DataFrame(frame_rows).drop(columns="trade_date", errors="ignore") \
            if name.startswith("delivery_") else pd.DataFrame(frame_rows)
        _write_month(seat_root, name, frame.to_dict("records"))
    database = table_root / "test.duckdb"
    if build:
        updater.build_database(
            root, database, tmp_path / "market.json", "market-v1",
            tmp_path / "seat.json", "seat-v2", 1,
        )
    return updater, root, table_root, database


def test_duckdb_preserves_market_frequency_term_structure_and_seat_semantics(
    tmp_path
):
    updater, root, table_root, database = _fixture(tmp_path)
    config = {"root_path": str(table_root), "eager_fields": False}
    parquet = ParquetFuturesSource(config)
    database_source = DuckDBFuturesSource(
        {"path": str(database)}, config
    )
    try:
        for frequency in ("daily", "1min", "5min", "15min", "30min", "hourly"):
            left = parquet.fetch_price_at_frequency(
                ["A"], "2024-01-02", "2024-01-04", ["open", "close", "volume"], frequency
            )
            right = database_source.fetch_price_at_frequency(
                ["A"], "2024-01-02", "2024-01-04", ["open", "close", "volume"], frequency
            )
            assert left.keys() == right.keys()
            for field in left:
                pd.testing.assert_frame_equal(left[field], right[field])
        pd.testing.assert_series_equal(
            parquet.fetch_listing_dates(["A"]),
            database_source.fetch_listing_dates(["A"]),
        )
        pd.testing.assert_frame_equal(
            parquet.fetch_contract_pair_prices(["A"], "2024-01-02", "2024-01-04")["near"],
            database_source.fetch_contract_pair_prices(
                ["A"], "2024-01-02", "2024-01-04"
            )["near"],
        )
        for table, date_column, value_column in (
            ("raw_seat_position", "trade_date", "net_position"),
            ("derive_product_seat", "trade_date", "net_position"),
            ("derive_main_contract_seat", "trade_date", "net_position"),
            ("derive_product_daily", "trade_date", "net_position"),
            ("delivery_seat", "delivery_date", "net_position"),
            ("delivery_summary", "delivery_date", "non_futures_net"),
        ):
            seat = database_source.fetch_seat_table(
                table, "2024-01-03", "2024-01-03", ["A"]
            )
            assert seat.loc[0, ["root", value_column]].tolist() == ["A", 6]
            assert isinstance(seat.loc[0, date_column], date)
    finally:
        database_source.close()
    con = duckdb.connect(str(database), read_only=True)
    try:
        updater.verify_database(con, root, exact=True)
    finally:
        con.close()


def test_duckdb_exposes_a_native_polars_query_boundary(tmp_path):
    _, _, table_root, database = _fixture(tmp_path)
    source = DuckDBFuturesSource(
        {"path": str(database)},
        {"root_path": str(table_root), "eager_fields": False},
    )
    try:
        frame = source._execute_polars(
            "SELECT trade_date, symbol, close FROM market.bars_1d "
            "ORDER BY trade_date, symbol"
        )
        assert isinstance(frame, pl.DataFrame)
        assert frame.columns == ["trade_date", "symbol", "close"]
        assert frame.height == 6

        raw, requested, available = source._read_storage_partitions_polars(
            "daily",
            "2024-01-02",
            "2024-01-04",
            ["trade_date", "symbol", "close"],
            ["A"],
        )
        assert isinstance(raw, pl.DataFrame)
        assert requested == ["trade_date", "symbol", "close"]
        assert set(requested) <= available
        assert raw.schema["trade_date"] == pl.Date
        assert raw["symbol"].to_list() == [
            "A2401", "A2405", "A2401", "A2405", "A2401", "A2405"
        ]
        selected = source._read_selected_partitions_polars(
            "daily",
            "2024-01-02",
            "2024-01-04",
            ["trade_date", "symbol", "close"],
            ("A",),
            pl.DataFrame({
                "trade_date": [date(2024, 1, 3)],
                "root": ["A"],
                "contract": ["A2401"],
            }),
        )
        assert selected["symbol"].to_list() == ["A2401", "A2401", "A2401"]
    finally:
        source.close()


def test_duckdb_price_pipeline_has_no_pandas_query_boundary(tmp_path):
    _, _, table_root, database = _fixture(tmp_path)
    source = DuckDBFuturesSource(
        {"path": str(database)},
        {"root_path": str(table_root), "eager_fields": False},
    )
    assert not hasattr(source, "_execute_df")
    try:
        panel = source.fetch_price_at_frequency(
            ["A"],
            "2024-01-02",
            "2024-01-04",
            ["open", "close", "volume"],
            "5min",
        )
    finally:
        source.close()

    assert panel["close"].loc["2024-01-03 09:05", "A"] == 105.0


def test_default_config_separates_research_cutoff_and_observation_end(monkeypatch):
    monkeypatch.delenv("MF_DATA_SOURCE", raising=False)
    monkeypatch.delenv("MF_DATE_END", raising=False)
    monkeypatch.delenv("MF_RESEARCH_CUTOFF", raising=False)

    config = load_config("config/default.yaml")

    assert config.data.source == "duckdb_futures"
    assert config.date_policy.research_cutoff == "2026-05-15"
    assert config.date_range.end == "latest_available"


def test_duckdb_market_cache_fingerprint_is_partition_scoped(tmp_path):
    _, _, table_root, database = _fixture(tmp_path)
    source = DuckDBFuturesSource(
        {"path": str(database)},
        {"root_path": str(table_root), "eager_fields": False},
    )
    try:
        before = source._files_fingerprint(Path(), [Path("1min/month")])
        source.release_id = "later-daily-release"
        source.market_component_id = "market-revision"
        source.seat_component_id = "seat-revision"
        after = source._files_fingerprint(Path(), [Path("1min/month")])
        assert before == after
    finally:
        source.close()


def test_checkpoint_fingerprint_is_slice_based_not_release_id(tmp_path):
    _, _, table_root, database = _fixture(tmp_path)
    source = DuckDBFuturesSource(
        {"path": str(database)},
        {"root_path": str(table_root), "eager_fields": False},
    )
    try:
        before = source.checkpoint_source_fingerprint(
            "2024-01-02", "2024-01-04"
        )
        source.release_id = "later-daily-release"
        after = source.checkpoint_source_fingerprint(
            "2024-01-02", "2024-01-04"
        )
    finally:
        source.close()

    assert before == after


def test_duckdb_month_update_is_exact_and_rolls_back_on_failure(tmp_path, monkeypatch):
    updater, root, table_root, database = _fixture(tmp_path)
    parquet_file = next((table_root / "futureshistoryprices1m").rglob("*.parquet"))
    frame = pd.read_parquet(parquet_file)
    frame["close"] += 1.0
    frame.to_parquet(parquet_file, index=False)
    declared = {spec.key: set() for spec in updater.DATASETS}
    declared["1m"] = {"2024-01"}
    con = duckdb.connect(str(database))
    try:
        updater.sync_database(
            con, root, declared, tmp_path / "market2.json", "market-v2",
            tmp_path / "seat.json", "seat-v2",
        )
        expected = con.execute("SELECT SUM(close) FROM market.bars_1m").fetchone()[0]
        frame["close"] += 1.0
        frame.to_parquet(parquet_file, index=False)
        monkeypatch.setattr(updater, "exact_difference", lambda *args: (1, 0))
        with pytest.raises(RuntimeError, match="post-insert difference"):
            updater.sync_database(
                con, root, declared, tmp_path / "market3.json", "market-v3",
                tmp_path / "seat.json", "seat-v2",
            )
        assert con.execute("SELECT SUM(close) FROM market.bars_1m").fetchone()[0] == expected
        assert updater.current_release(con)["market_component_id"] == "market-v2"
    finally:
        con.close()


def test_duckdb_initial_build_resumes_only_certified_months(tmp_path, monkeypatch):
    updater, root, _, database = _fixture(tmp_path, build=False)
    original = updater.exact_difference
    calls = 0

    def fail_after_two_months(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("injected interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(updater, "exact_difference", fail_after_two_months)
    with pytest.raises(RuntimeError, match="injected interruption"):
        updater.build_database(
            root, database, tmp_path / "market.json", "market-v1",
            tmp_path / "seat.json", "seat-v2", 1,
        )
    building = database.with_name(
        f"{database.stem}."
        f"{updater.make_release_id('market-v1', 'seat-v2')[:12]}.building.duckdb"
    )
    con = duckdb.connect(str(building))
    try:
        assert con.execute("SELECT COUNT(*) FROM meta.partitions").fetchone()[0] == 2
        # The already-running formal build predates the explicit build marker;
        # completed partition metadata must still be safely resumable.
        con.execute("DELETE FROM meta.releases WHERE status = 'building'")
    finally:
        con.close()

    monkeypatch.setattr(updater, "exact_difference", original)
    updater.build_database(
        root, database, tmp_path / "market.json", "market-v1",
        tmp_path / "seat.json", "seat-v2", 1,
    )
    assert database.is_file()
    assert not building.exists()
    con = duckdb.connect(str(database), read_only=True)
    try:
        updater.verify_database(con, root, exact=True)
    finally:
        con.close()
