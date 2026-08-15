from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import load_config
from data.contract_symbols import ContractAliasConflictError
from data.parquet_source import ParquetFuturesSource


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
                _row(" A2401", day, old_close, 500, 1000),
                _row("A2405", day, new_close, 900, 1800),
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

    assert panel["close"].index.equals(pd.date_range("2024-01-02", periods=3))
    assert panel["close"].loc["2024-01-03", "A"] == 102.0
    expected_roll_adjusted = 126.0 * (102.0 / 120.0)
    assert np.isclose(panel["close"].loc["2024-01-04", "A"], expected_roll_adjusted)
    assert panel["oi"].loc["2024-01-04", "A"] == 1800.0


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
    assert np.isclose(panel["curve_oi_breadth"].loc["2024-01-03", "A"], 1 / 3)
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
    assert len(list(cache_path.glob("curve_v2_15min_2024-01_*.parquet"))) == 1


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
    assert len(list(cache_path.glob("curve_v2_15min_2024-01_*.parquet"))) == 1
    assert len(list(cache_path.glob("curve_v2_5min_2024-01_*.parquet"))) == 1

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
    assert len(list(cache_path.glob("selected_v2_5min_*.parquet"))) == 1


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


def test_parquet_research_config_extends_default_and_uses_env_root(
    tmp_path, monkeypatch
):
    root = _fixture_root(tmp_path / "market")
    monkeypatch.setenv("MF_PARQUET_ROOT", str(root))

    config = load_config("config/parquet_research.yaml")

    assert config.data.source == "parquet_futures"
    assert config.data.parquet.root_path == str(root)
    assert config.processing
    assert config.universe


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


def _fixture_no_overlap(tmp_path: Path) -> Path:
    """A2401 与 A2405 无共同交易日 (A2401 仅 01-02, A2405 仅 01-04) -> 换月无共同 close."""
    rows = [
        _row(" A2401", "2024-01-02", 100.0, 500, 1000),
        _row("A9999", "2024-01-02", 100.0, 500, 1000),
        # 01-03 无 A2401 (停牌/退市), 01-04 才有 A2405
        _row("A2405", "2024-01-04", 126.0, 900, 1800),
        _row("A9999", "2024-01-04", 126.0, 900, 1800),
    ]
    _write_dataset(tmp_path, DATASETS["daily"], rows)
    # 初始化要求 1min/15min 目录存在 (空数据集即可)
    _write_dataset(tmp_path, DATASETS["1min"], [])
    _write_dataset(tmp_path, DATASETS["15min"], [])
    return tmp_path


def test_no_common_close_raises_rollover_adjustment_error(tmp_path):
    """换月无共同收盘价必须抛 RolloverAdjustmentError (fail-closed)."""
    root = _fixture_no_overlap(tmp_path)
    source = ParquetFuturesSource({"root_path": str(root), "eager_fields": False})
    from data.continuous_contract import RolloverAdjustmentError
    try:
        source.fetch_price_at_frequency(
            ["A"], "2024-01-02", "2024-01-05", ["close"], "daily"
        )
        raise AssertionError("应抛 RolloverAdjustmentError 但未抛")
    except RolloverAdjustmentError as e:
        assert "no common close" in str(e)
