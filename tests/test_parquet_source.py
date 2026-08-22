from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.config import load_config
from data.contract_symbols import ContractAliasConflictError
from data.parquet_source import MissingParquetPartitionError, ParquetFuturesSource


DATASETS = {
    "daily": "futureshistoryprices1d",
    "1min": "futureshistoryprices1m",
    "15min": "futureshistoryprices15m",
}


def _row(symbol, timestamp, close, volume, position, *, sequence=0):
    timestamp = pd.Timestamp(timestamp)
    return {
        "exchange": "TEST",
        "symbol": symbol,
        "trade_datetime": timestamp,
        "open": float(close) - 1.0,
        "high": float(close) + 2.0,
        "low": float(close) - 2.0,
        "close": float(close),
        "settle_price": float(close) - 0.5,
        "volume": int(volume),
        "amount": float(close) * float(volume),
        "position": int(position),
        "type": 0,
        "sequence": int(sequence),
        "trade_date": timestamp.date(),
    }


def _write_dataset(root: Path, dataset: str, rows):
    directory = root / dataset / "year_month=2024-01"
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if dataset != DATASETS["daily"]:
        frame = frame.drop(columns="sequence", errors="ignore")
    frame.to_parquet(directory / "part.parquet", index=False)


def _fixture_root(tmp_path: Path) -> Path:
    daily_rows = []
    for day, old_close, new_close, main in (
        ("2024-01-02", 100.0, 110.0, "A2401"),
        ("2024-01-03", 102.0, 120.0, "A2405"),
        ("2024-01-04", 104.0, 126.0, "A2405"),
    ):
        daily_rows.extend(
            [
                _row(
                    " A2401", day, old_close,
                    900 if main == "A2401" else 500,
                    1800 if main == "A2401" else 1000,
                ),
                _row(
                    "A2405", day, new_close,
                    500 if main == "A2401" else 900,
                    1000 if main == "A2401" else 1800,
                ),
                _row(
                    "A2409", day, new_close + 10.0, 50,
                    {"2024-01-02": 200, "2024-01-03": 220, "2024-01-04": 210}[day],
                ),
                _row("A9999", day, old_close if main == "A2401" else new_close,
                     500 if main == "A2401" else 900,
                     1000 if main == "A2401" else 1800),
            ]
        )
    _write_dataset(tmp_path, DATASETS["daily"], daily_rows)

    minute_rows = []
    for minute, close in enumerate((102.0, 103.0, 104.0, 105.0, 106.0, 107.0)):
        minute_rows.append(
            _row(" A2401", pd.Timestamp("2024-01-03 09:00") + pd.Timedelta(minutes=minute),
                 close, 10 + minute, 1000 + minute)
        )
    for minute, close in enumerate((126.0, 127.0, 128.0, 129.0, 130.0, 131.0)):
        minute_rows.append(
            _row("A2405", pd.Timestamp("2024-01-04 09:00") + pd.Timedelta(minutes=minute),
                 close, 20 + minute, 1800 + minute)
        )
    _write_dataset(tmp_path, DATASETS["1min"], minute_rows)

    fifteen_rows = [
        _row(" A2401", "2024-01-03 09:00", 102.0, 100, 1000),
        _row(" A2401", "2024-01-03 09:15", 103.0, 120, 1010),
        _row("A2405", "2024-01-03 09:00", 120.0, 30, 1810),
        _row("A2405", "2024-01-04 09:00", 126.0, 200, 1800),
        _row("A2405", "2024-01-04 09:15", 127.0, 220, 1810),
    ]
    _write_dataset(tmp_path, DATASETS["15min"], fifteen_rows)
    return tmp_path


def test_daily_uses_previous_day_main_and_causal_roll_adjustment(tmp_path):
    root = _fixture_root(tmp_path)
    source = ParquetFuturesSource({"root_path": str(root), "eager_fields": False})

    panel = source.fetch_price_at_frequency(
        ["A"], "2024-01-02", "2024-01-04", ["close", "oi"], "daily"
    )

    # 01-02 没有更早交易日可供选约，严格因果模式不发布伪造价格。
    assert panel["close"].index.equals(pd.date_range("2024-01-03", periods=2))
    assert panel["close"].loc["2024-01-03", "A"] == 102.0
    expected_roll_adjusted = 126.0 * (102.0 / 120.0)
    assert np.isclose(panel["close"].loc["2024-01-04", "A"], expected_roll_adjusted)
    assert panel["oi"].loc["2024-01-04", "A"] == 1800.0


def test_daily_settlement_uses_the_same_causal_contract_and_adjustment(tmp_path):
    root = _fixture_root(tmp_path)
    source = ParquetFuturesSource({"root_path": str(root), "eager_fields": False})

    panel = source.fetch_price_at_frequency(
        ["A"], "2024-01-02", "2024-01-04", ["close", "settle"], "daily"
    )

    assert panel["settle"].loc["2024-01-03", "A"] == pytest.approx(101.5)
    expected = 125.5 * (102.0 / 120.0)
    assert panel["settle"].loc["2024-01-04", "A"] == pytest.approx(expected)


def test_latest_trade_date_comes_from_all_latest_daily_shards(tmp_path):
    root = _fixture_root(tmp_path)
    directory = root / DATASETS["daily"] / "year_month=2024-01"
    pd.DataFrame([
        _row("A2405", "2024-01-05", 127.0, 10, 100),
    ]).to_parquet(directory / "later.parquet", index=False)
    source = ParquetFuturesSource({"root_path": str(root), "eager_fields": False})

    assert source.fetch_latest_trade_date() == pd.Timestamp("2024-01-05")


def test_intraday_route_keeps_real_bar_index_and_resamples(tmp_path):
    root = _fixture_root(tmp_path)
    source = ParquetFuturesSource({"root_path": str(root), "eager_fields": False})

    one_minute = source.fetch_price_at_frequency(
        ["A"], "2024-01-03", "2024-01-03 23:59", ["close"], "1min"
    )["close"]
    five_minute = source.fetch_price_at_frequency(
        ["A"], "2024-01-03", "2024-01-03 23:59", ["open", "high", "low", "close", "volume"], "5min"
    )

    assert len(one_minute) == 6
    assert isinstance(one_minute.index, pd.DatetimeIndex)
    assert list(five_minute["close"].index) == [
        pd.Timestamp("2024-01-03 09:00"),
        pd.Timestamp("2024-01-03 09:05"),
    ]
    assert five_minute["open"].iloc[0, 0] == 101.0
    assert five_minute["high"].iloc[0, 0] == 108.0
    assert five_minute["low"].iloc[0, 0] == 100.0
    assert five_minute["close"].iloc[0, 0] == 106.0
    assert five_minute["volume"].iloc[0, 0] == sum(range(10, 15))

    curve = source.fetch_contract_curve_at_frequency(
        ["A"], "2024-01-03", "2024-01-03 23:59",
        ["open", "high", "low", "close", "volume", "position", "settle"],
        "5min",
    ).iloc[0]
    assert curve[["open", "high", "low", "close"]].tolist() == [
        101.0, 108.0, 100.0, 106.0,
    ]
    assert curve["volume"] == sum(range(10, 15))
    assert curve["position"] == 1004
    assert curve["settle"] == 105.5


def test_contract_curve_uses_exact_roots_without_prefix_collisions(tmp_path):
    root = _fixture_root(tmp_path)
    rows = [
        _row("A2405", "2024-01-03 09:00", 100.0, 10, 100),
        _row("AP2405", "2024-01-03 09:00", 200.0, 20, 200),
        _row("P2405", "2024-01-03 09:00", 300.0, 30, 300),
        _row("PB2405", "2024-01-03 09:00", 400.0, 40, 400),
    ]
    _write_dataset(root, DATASETS["1min"], rows)
    source = ParquetFuturesSource({"root_path": str(root), "eager_fields": False})

    curve = source.fetch_contract_curve_at_frequency(
        ["A", "P"], "2024-01-03", "2024-01-03", ["close", "position"],
        "1min",
    )

    assert curve["root"].tolist() == ["A", "P"]
    assert curve["symbol"].tolist() == ["A2405", "P2405"]


def test_dominant_selection_requires_positive_open_interest(tmp_path):
    source = ParquetFuturesSource({
        "root_path": str(_fixture_root(tmp_path)), "eager_fields": False
    })
    frame = pd.DataFrame([
        _row("A2401", "2024-01-03", 100.0, 100, 0),
        {**_row("A2405", "2024-01-03", 101.0, 10, 1), "position": np.nan},
    ])

    selected = source._infer_vendor_main(frame, ("A",))

    assert selected.empty


def test_contract_pairs_use_the_first_two_observable_maturities(tmp_path):
    root = _fixture_root(tmp_path)
    source = ParquetFuturesSource({"root_path": str(root), "eager_fields": False})

    close_pair = source.fetch_contract_pair_prices(
        ["A"], "2024-01-02", "2024-01-04", field="close"
    )
    settle_pair = source.fetch_contract_pair_prices(
        ["A"], "2024-01-02", "2024-01-04", field="settle"
    )

    assert close_pair["near"].loc["2024-01-03", "A"] == pytest.approx(102.0)
    assert close_pair["far"].loc["2024-01-03", "A"] == pytest.approx(120.0)
    assert settle_pair["near"].loc["2024-01-03", "A"] == pytest.approx(101.5)
    assert settle_pair["far"].loc["2024-01-03", "A"] == pytest.approx(119.5)


def test_czce_three_digit_alias_is_canonicalized_and_identical_rows_deduplicate(
    tmp_path,
):
    root = _fixture_root(tmp_path)
    rows = [
        {**_row("FG609", "2026-08-03 09:00", 100.0, 10, 20), "exchange": "CZCE"},
        {**_row("FG2609", "2026-08-03 09:00", 100.0, 10, 20), "exchange": "CZCE"},
    ]
    directory = root / DATASETS["1min"] / "year_month=2026-08"
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop(columns="sequence").to_parquet(
        directory / "part.parquet", index=False
    )
    source = ParquetFuturesSource({"root_path": str(root), "eager_fields": False})

    actual = source._read_partitions(
        "1min", "2026-08-03", "2026-08-03", ["symbol", "trade_date", "close"]
    )

    assert actual["symbol"].tolist() == ["FG2609"]


def test_czce_alias_market_conflict_fails_even_when_requested_field_matches(tmp_path):
    root = _fixture_root(tmp_path)
    rows = [
        {**_row("FG609", "2026-08-03 09:00", 100.0, 10, 20), "exchange": "CZCE"},
        {**_row("FG2609", "2026-08-03 09:00", 100.0, 11, 20), "exchange": "CZCE"},
    ]
    directory = root / DATASETS["1min"] / "year_month=2026-08"
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop(columns="sequence").to_parquet(
        directory / "part.parquet", index=False
    )
    source = ParquetFuturesSource({"root_path": str(root), "eager_fields": False})

    with np.testing.assert_raises(ContractAliasConflictError):
        source._read_partitions(
            "1min", "2026-08-03", "2026-08-03", ["symbol", "trade_date", "close"]
        )


def test_already_canonical_duplicate_market_conflict_fails_closed(tmp_path):
    root = _fixture_root(tmp_path)
    rows = [
        {**_row("FG2609", "2026-08-03 09:00", 100.0, 10, 20), "exchange": "CZCE"},
        {**_row("FG2609", "2026-08-03 09:00", 100.0, 11, 20), "exchange": "CZCE"},
    ]
    directory = root / DATASETS["1min"] / "year_month=2026-08"
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop(columns="sequence").to_parquet(
        directory / "part.parquet", index=False
    )
    source = ParquetFuturesSource({"root_path": str(root), "eager_fields": False})

    with np.testing.assert_raises(ContractAliasConflictError):
        source._read_partitions(
            "1min", "2026-08-03", "2026-08-03", ["symbol", "trade_date", "close"]
        )


def test_partition_read_failure_does_not_return_partial_market_data(
    tmp_path, monkeypatch
):
    root = _fixture_root(tmp_path)
    directory = root / DATASETS["daily"] / "year_month=2024-01"
    healthy = directory / "part.parquet"
    failing = directory / "zz-failing.parquet"
    failing.write_bytes(healthy.read_bytes())
    source = ParquetFuturesSource({"root_path": str(root), "eager_fields": False})
    original = pd.read_parquet

    def fail_one_partition(path, *args, **kwargs):
        if Path(path).name == failing.name:
            raise OSError("simulated shard failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", fail_one_partition)
    with pytest.raises(RuntimeError, match="failed to read market partition"):
        source._read_partitions(
            "daily", "2024-01-02", "2024-01-04",
            ["symbol", "trade_date", "close"],
        )


def test_partition_schema_drift_fails_instead_of_dropping_requested_field(tmp_path):
    root = _fixture_root(tmp_path)
    directory = root / DATASETS["daily"] / "year_month=2024-01"
    partial = pd.DataFrame([
        _row("A2405", "2024-01-05", 127.0, 10, 20)
    ]).drop(columns="position")
    partial.to_parquet(directory / "00-missing-position.parquet", index=False)
    source = ParquetFuturesSource({"root_path": str(root), "eager_fields": False})

    with pytest.raises(RuntimeError, match="inconsistent daily parquet schema"):
        source._read_partitions(
            "daily", "2024-01-02", "2024-01-05",
            ["symbol", "trade_date", "position"],
        )


def test_daily_curve_fields_use_all_concrete_contracts(tmp_path):
    root = _fixture_root(tmp_path)
    source = ParquetFuturesSource({"root_path": str(root), "eager_fields": False})

    panel = source.fetch_price_at_frequency(
        ["A"], "2024-01-02", "2024-01-04",
        ["curve_total_oi", "curve_top2_oi", "curve_oi_breadth",
         "curve_oi_concentration", "curve_oi_hhi"],
        "daily",
    )

    assert panel["curve_total_oi"].loc["2024-01-03", "A"] == 3020.0
    assert panel["curve_top2_oi"].loc["2024-01-03", "A"] == 2800.0
    assert np.isclose(panel["curve_oi_breadth"].loc["2024-01-03", "A"], 2 / 3)
    assert np.isclose(
        panel["curve_oi_concentration"].loc["2024-01-03", "A"],
        2800.0 / 3020.0,
    )
    expected_hhi = (1000.0**2 + 1800.0**2 + 220.0**2) / 3020.0**2
    assert np.isclose(panel["curve_oi_hhi"].loc["2024-01-03", "A"], expected_hhi)


def test_intraday_curve_carries_sparse_contract_state_from_previous_day(tmp_path):
    root = _fixture_root(tmp_path)
    source = ParquetFuturesSource({"root_path": str(root), "eager_fields": False})

    panel = source.fetch_price_at_frequency(
        ["A"], "2024-01-03", "2024-01-03 23:59",
        ["curve_total_oi", "curve_top2_oi", "curve_oi_breadth"],
        "15min",
    )

    assert panel["curve_total_oi"].loc["2024-01-03 09:00", "A"] == 3010.0
    assert panel["curve_total_oi"].loc["2024-01-03 09:15", "A"] == 3020.0
    assert panel["curve_top2_oi"].loc["2024-01-03 09:15", "A"] == 2820.0
    assert np.isclose(
        panel["curve_oi_breadth"].loc["2024-01-03 09:15", "A"], 1 / 3
    )


def test_intraday_curve_cache_reuses_validated_month_across_instances(
    tmp_path, monkeypatch
):
    root = _fixture_root(tmp_path / "market")
    cache_path = tmp_path / "curve-cache"
    config = {
        "root_path": str(root),
        "eager_fields": False,
        "curve_cache_enabled": True,
        "curve_cache_path": str(cache_path),
    }
    first = ParquetFuturesSource(config)
    expected = first.fetch_price_at_frequency(
        ["A"], "2024-01-03", "2024-01-03 23:59",
        ["curve_total_oi", "curve_oi_breadth"], "15min",
    )

    second = ParquetFuturesSource(config)
    monkeypatch.setattr(
        second,
        "_aggregate_intraday_states",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("valid persistent cache was not reused")
        ),
    )
    actual = second.fetch_price_at_frequency(
        ["A"], "2024-01-03", "2024-01-03 23:59",
        ["curve_total_oi", "curve_oi_breadth"], "15min",
    )

    pd.testing.assert_frame_equal(actual["curve_total_oi"], expected["curve_total_oi"])
    pd.testing.assert_frame_equal(actual["curve_oi_breadth"], expected["curve_oi_breadth"])
    assert len(list(cache_path.glob("curve_v4_15min_2024-01_*.parquet"))) == 1


def test_intraday_curve_cache_isolated_by_frequency_and_invalidated_by_source(
    tmp_path, monkeypatch
):
    root = _fixture_root(tmp_path / "market")
    cache_path = tmp_path / "curve-cache"
    config = {
        "root_path": str(root),
        "eager_fields": False,
        "curve_cache_enabled": True,
        "curve_cache_path": str(cache_path),
    }
    first = ParquetFuturesSource(config)
    first.fetch_price_at_frequency(
        ["A"], "2024-01-03", "2024-01-03 23:59",
        ["curve_total_oi"], "15min",
    )
    first.fetch_price_at_frequency(
        ["A"], "2024-01-03", "2024-01-03 23:59",
        ["curve_total_oi"], "5min",
    )
    assert len(list(cache_path.glob("curve_v4_15min_2024-01_*.parquet"))) == 1
    assert len(list(cache_path.glob("curve_v4_5min_2024-01_*.parquet"))) == 1

    source_file = next(
        (root / DATASETS["15min"] / "year_month=2024-01").glob("*.parquet")
    )
    stat = source_file.stat()
    os.utime(source_file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    second = ParquetFuturesSource(config)
    original = second._aggregate_intraday_states
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(second, "_aggregate_intraday_states", counted)
    second.fetch_price_at_frequency(
        ["A"], "2024-01-03", "2024-01-03 23:59",
        ["curve_total_oi"], "15min",
    )
    assert calls


def test_selected_contract_cache_reuses_validated_request_across_instances(
    tmp_path, monkeypatch
):
    root = _fixture_root(tmp_path / "market")
    cache_path = tmp_path / "selected-cache"
    config = {
        "root_path": str(root),
        "eager_fields": False,
        "selected_cache_enabled": True,
        "selected_cache_path": str(cache_path),
    }
    first = ParquetFuturesSource(config)
    expected = first.fetch_price_at_frequency(
        ["A"], "2024-01-03", "2024-01-03 23:59",
        ["open", "close", "volume"], "5min",
    )

    second = ParquetFuturesSource(config)
    monkeypatch.setattr(
        second,
        "_read_partitions",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("valid selected-contract cache was not reused")
        ),
    )
    actual = second.fetch_price_at_frequency(
        ["A"], "2024-01-03", "2024-01-03 23:59",
        ["open", "close", "volume"], "5min",
    )

    for field in expected:
        pd.testing.assert_frame_equal(actual[field], expected[field])
    assert len(list(cache_path.glob("selected_v5_5min_*.parquet"))) == 1


def test_selected_contract_cache_rejects_duplicate_long_keys(tmp_path):
    root = _fixture_root(tmp_path / "market")
    cache_path = tmp_path / "selected-cache"
    config = {
        "root_path": str(root),
        "eager_fields": False,
        "selected_cache_enabled": True,
        "selected_cache_path": str(cache_path),
    }
    source = ParquetFuturesSource(config)
    source.fetch_price_at_frequency(
        ["A"], "2024-01-03", "2024-01-03 23:59", ["close"], "5min"
    )
    data_path = next(cache_path.glob("selected_v5_5min_*.parquet"))
    metadata_path = next(cache_path.glob("selected_v5_5min_*.json"))
    cached = pd.read_parquet(data_path)
    pd.concat([cached, cached.iloc[[0]]], ignore_index=True).to_parquet(
        data_path, index=False
    )
    metadata = pd.read_json(metadata_path, typ="series")

    actual = source._read_selected_cache(
        "5min",
        pd.Timestamp(metadata["start"]),
        pd.Timestamp(metadata["end"]),
        tuple(metadata["tickers"]),
        tuple(metadata["fields"]),
        str(metadata["source_fingerprint"]),
    )

    assert actual is None


def test_panel_cache_adds_curve_fields_without_rebuilding_selected_panel(
    tmp_path, monkeypatch
):
    source = ParquetFuturesSource({
        "root_path": str(_fixture_root(tmp_path)),
        "eager_fields": True,
        "curve_cache_enabled": False,
        "selected_cache_enabled": False,
    })
    source.fetch_price_at_frequency(
        ["A"], "2024-01-03", "2024-01-03 23:59", ["close"], "15min"
    )
    monkeypatch.setattr(
        source,
        "_selected_long",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("existing OHLCV panel was rebuilt")
        ),
    )

    curve = source.fetch_price_at_frequency(
        ["A"], "2024-01-03", "2024-01-03 23:59",
        ["curve_total_oi"], "15min",
    )
    assert not curve["curve_total_oi"].empty


def test_default_config_uses_env_parquet_root(
    tmp_path, monkeypatch
):
    root = _fixture_root(tmp_path / "market")
    monkeypatch.setenv("MF_DATA_SOURCE", "parquet_futures")
    monkeypatch.setenv("MF_PARQUET_ROOT", str(root))

    config = load_config("config/default.yaml")

    assert config.data.source == "parquet_futures"
    assert config.data.parquet.root_path == str(root)
    assert config.processing
    assert config.universe


def test_machine_local_config_cannot_override_research_semantics(tmp_path):
    project = Path(__file__).resolve().parents[1]
    (tmp_path / "default.yaml").write_text(
        (project / "config" / "default.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "local.yaml").write_text(
        "universe: [RB]\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="machine-local data runtime settings"):
        load_config(str(tmp_path / "default.yaml"))


def test_framework_factory_builds_parquet_only_source(tmp_path, monkeypatch):
    from data.manager import DataManager

    root = _fixture_root(tmp_path / "market")
    monkeypatch.setenv("MF_DATA_SOURCE", "parquet_futures")
    monkeypatch.setenv("MF_PARQUET_ROOT", str(root))
    config = load_config("config/default.yaml")
    config.data.parquet.datasets = dict(DATASETS)
    config.data.parquet.curve_cache_enabled = False
    config.data.parquet.selected_cache_enabled = False

    manager = DataManager.from_config(config)

    assert config.data.source == "parquet_futures"
    assert config.data.cache["enabled"] is False
    assert not hasattr(manager.source, "_macro_source")
    assert manager.source.root_active_from["FU"] == pd.Timestamp("2018-07-16")


def test_intraday_night_bar_uses_next_exchange_trading_day(tmp_path, monkeypatch):
    source = ParquetFuturesSource({
        "root_path": str(_fixture_root(tmp_path)), "eager_fields": False
    })
    monkeypatch.setattr(
        source,
        "fetch_calendar",
        lambda start, end: pd.DatetimeIndex(["2024-01-05", "2024-01-08"]),
    )
    frame = pd.DataFrame({
        "trade_datetime": pd.to_datetime([
            "2024-01-04 21:00", "2024-01-05 09:00", "2024-01-07 21:00"
        ]),
        "trade_date": pd.to_datetime([
            "2024-01-04", "2024-01-05", "2024-01-07"
        ]),
    })

    actual = source._assign_intraday_trade_dates(frame)

    assert actual["trade_date"].tolist() == [
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-01-08"),
    ]


def test_session_index_groups_night_and_day_bars_on_the_exchange_day(
    tmp_path, monkeypatch
):
    source = ParquetFuturesSource({
        "root_path": str(_fixture_root(tmp_path)), "eager_fields": False
    })
    monkeypatch.setattr(
        source,
        "fetch_calendar",
        lambda start, end: pd.DatetimeIndex(["2024-01-05", "2024-01-08"]),
    )
    wall_clock = pd.to_datetime([
        "2024-01-05 21:00", "2024-01-06 01:00", "2024-01-08 09:00"
    ])

    actual = source.trading_session_index(wall_clock)

    assert actual.normalize().unique().tolist() == [pd.Timestamp("2024-01-08")]
    assert actual.is_monotonic_increasing
    assert actual.tolist() == pd.to_datetime([
        "2024-01-08 03:00", "2024-01-08 07:00", "2024-01-08 15:00"
    ]).tolist()


def test_intraday_trade_date_assignment_fails_when_daily_calendar_is_empty(
    tmp_path, monkeypatch
):
    source = ParquetFuturesSource({
        "root_path": str(_fixture_root(tmp_path)), "eager_fields": False
    })
    monkeypatch.setattr(
        source, "fetch_calendar", lambda start, end: pd.DatetimeIndex([])
    )
    frame = pd.DataFrame({
        "trade_datetime": [pd.Timestamp("2024-01-04 21:00")],
        "trade_date": [pd.Timestamp("2024-01-04")],
    })

    with pytest.raises(ValueError, match="daily parquet calendar is empty"):
        source._assign_intraday_trade_dates(frame)


def test_calendar_uses_concrete_contracts_without_vendor_9999_rows(tmp_path):
    rows = [
        _row("A2401", "2024-01-02", 100.0, 10, 20),
        _row("A2401", "2024-01-03", 101.0, 10, 20),
    ]
    _write_dataset(tmp_path, DATASETS["daily"], rows)
    _write_dataset(tmp_path, DATASETS["1min"], [])
    _write_dataset(tmp_path, DATASETS["15min"], [])
    source = ParquetFuturesSource({"root_path": str(tmp_path)})

    actual = source.fetch_calendar("2024-01-02", "2024-01-03")

    assert actual.equals(pd.DatetimeIndex(["2024-01-02", "2024-01-03"]))


def test_missing_internal_month_partition_fails_closed(tmp_path):
    root = _fixture_root(tmp_path)
    march = root / DATASETS["daily"] / "year_month=2024-03"
    march.mkdir(parents=True)
    pd.DataFrame([
        _row("A2405", "2024-03-01", 130.0, 10, 20)
    ]).to_parquet(march / "part.parquet", index=False)
    source = ParquetFuturesSource({"root_path": str(root)})

    with pytest.raises(MissingParquetPartitionError, match="2024-02"):
        source._month_files("daily", "2024-01-01", "2024-03-31")


def test_active_epoch_restarts_schedule_without_bridging_old_contract(tmp_path):
    root = _fixture_root(tmp_path)
    source = ParquetFuturesSource({
        "root_path": str(root),
        "eager_fields": False,
        "root_active_from": {"A": "2024-01-03"},
    })

    schedule = source.fetch_contract_schedule(
        ["A"], "2024-01-02", "2024-01-04"
    )
    close = source.fetch_price(
        ["A"], "2024-01-02", "2024-01-04", ["close"]
    )["close"]

    assert schedule.index.equals(pd.DatetimeIndex(["2024-01-04"]))
    assert schedule.loc["2024-01-04", "A"] == "A2405"
    assert close.loc["2024-01-04", "A"] == 126.0


def test_same_source_instance_invalidates_panel_when_parquet_changes(tmp_path):
    root = _fixture_root(tmp_path)
    source = ParquetFuturesSource({
        "root_path": str(root), "eager_fields": False,
        "panel_cache_entries": 1,
    })
    before = source.fetch_price(
        ["A"], "2024-01-02", "2024-01-04", ["close"]
    )["close"]
    path = root / DATASETS["daily"] / "year_month=2024-01" / "part.parquet"
    frame = pd.read_parquet(path)
    dates = pd.to_datetime(frame["trade_date"])
    mask = frame["symbol"].astype(str).str.strip().eq("A2401") & dates.eq(
        pd.Timestamp("2024-01-03")
    )
    frame.loc[mask, "close"] = 202.0
    frame.to_parquet(path, index=False)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    after = source.fetch_price(
        ["A"], "2024-01-02", "2024-01-04", ["close"]
    )["close"]

    assert before.loc["2024-01-03", "A"] == 102.0
    assert after.loc["2024-01-03", "A"] == 202.0


def _fixture_no_overlap(tmp_path: Path) -> Path:
    """A2401 与 A2405 无共同交易日 (A2401 仅 01-02, A2405 仅 01-04) -> 换月无共同 close."""
    rows = [
        _row(" A2401", "2024-01-02", 100.0, 500, 1000),
        _row("A9999", "2024-01-02", 100.0, 500, 1000),
        # 01-03 无 A2401 (停牌/退市), 01-04 才有 A2405
        _row("A2405", "2024-01-04", 126.0, 900, 1800),
        _row("A9999", "2024-01-04", 126.0, 900, 1800),
        # 01-05 才执行 01-04 观察到的新主力；上一收盘无旧合约报价，必须失败关闭。
        _row("A2405", "2024-01-05", 127.0, 900, 1800),
        _row("A9999", "2024-01-05", 127.0, 900, 1800),
    ]
    _write_dataset(tmp_path, DATASETS["daily"], rows)
    # 初始化要求 1min/15min 目录存在 (空数据集即可)
    _write_dataset(tmp_path, DATASETS["1min"], [])
    _write_dataset(tmp_path, DATASETS["15min"], [])
    return tmp_path


def test_unexecutable_dominant_switch_is_deferred_without_same_day_fallback(tmp_path):
    """两腿不能在同一收盘成交时保留旧计划，不伪造新主力连续价。"""
    root = _fixture_no_overlap(tmp_path)
    source = ParquetFuturesSource({"root_path": str(root), "eager_fields": False})
    schedule = source.fetch_contract_schedule(
        ["A"], "2024-01-02", "2024-01-05"
    )
    assert schedule.loc["2024-01-04", "A"] == "A2401"
    assert schedule.loc["2024-01-05", "A"] == "A2401"
    with pytest.raises(RuntimeError, match="could not build requested fields: close"):
        source.fetch_price_at_frequency(
            ["A"], "2024-01-02", "2024-01-05", ["close"], "daily"
        )
