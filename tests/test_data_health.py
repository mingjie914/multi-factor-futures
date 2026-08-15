from __future__ import annotations

import pandas as pd

from workflows.diagnostics.data_health import _check_latest_parquet_contract_keys


def _write_partition(root, dataset: str, rows: list[dict]) -> None:
    partition = root / dataset / "year_month=2026-08"
    partition.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(partition / "part.parquet", index=False)


def _row(symbol: str) -> dict:
    return {
        "symbol": symbol,
        "exchange": "CZCE",
        "trade_date": pd.Timestamp("2026-08-03").date(),
        "trade_datetime": pd.Timestamp("2026-08-03 09:00"),
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 10.0,
        "amount": 1000.0,
        "position": 20.0,
    }


def test_latest_parquet_contract_health_accepts_canonical_unique_rows(tmp_path):
    datasets = {
        "daily": "daily", "1min": "minute", "5min": "five", "15min": "fifteen"
    }
    for dataset in datasets.values():
        _write_partition(tmp_path, dataset, [_row("FG2609")])

    result = _check_latest_parquet_contract_keys({
        "root_path": str(tmp_path), "datasets": datasets,
    })

    assert result["status"] == "ok"


def test_latest_parquet_contract_health_rejects_three_digit_alias(tmp_path):
    datasets = {
        "daily": "daily", "1min": "minute", "5min": "five", "15min": "fifteen"
    }
    for dataset in datasets.values():
        _write_partition(tmp_path, dataset, [_row("FG609")])

    result = _check_latest_parquet_contract_keys({
        "root_path": str(tmp_path), "datasets": datasets,
    })

    assert result["status"] == "invalid"
    assert result["latest_partitions"]["daily"]["czce_three_digit_rows"] == 1
