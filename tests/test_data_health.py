from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

import workflows.diagnostics.data_health as data_health
from workflows.diagnostics.data_health import (
    _check_all_daily_contract_keys,
    _check_all_parquet_schemas,
    _check_all_seat_parquet_keys,
    _check_historical_parquet_contracts,
    _check_latest_parquet_contract_keys,
    check_health,
)


def _write_partition(root, dataset: str, rows: list[dict]) -> None:
    partition = root / dataset / "year_month=2026-08"
    partition.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(partition / "part.parquet", index=False)


def _seat_rows() -> dict[str, list[dict]]:
    common = {
        "trade_date": pd.Timestamp("2026-08-03").date(),
        "exchange": "SHFE",
        "root": "RB",
        "product_code": "RB",
    }
    positions = {
        "long_position": 10.0,
        "long_change": 1.0,
        "short_position": 8.0,
        "short_change": -1.0,
        "net_position": 2.0,
    }
    return {
        "derive_product_daily": [{
            **common,
            "total_long": 10.0,
            "total_short": 8.0,
            "net_position": 2.0,
            "long_change": 1.0,
            "short_change": -1.0,
            "seat_count": 1,
        }],
        "derive_product_seat": [{
            **common,
            "seat_name": "seat",
            **positions,
            "contract_count": 1,
        }],
        "derive_main_contract_seat": [{
            **common,
            "contract_code": "RB2610",
            "symbol": "RB2610",
            "seat_name": "seat",
            **positions,
            "close": 3000.0,
            "open_interest": 100.0,
        }],
        "raw_seat_position": [{
            **common,
            "contract_code": "RB2610",
            "symbol": "RB2610",
            "seat_name": "seat",
            "is_aggregated": False,
            "record_grain": "contract",
            **positions,
        }],
        "delivery_summary": [{
            "delivery_date": pd.Timestamp("2026-08-03").date(),
            "exchange": "SHFE",
            "root": "RB",
            "product_code": "RB",
            "product_name": "rebar",
            "contract_code": "RB2610",
            "symbol": "RB2610",
            "receive_quantity": 1.0,
            "deliver_quantity": 1.0,
            "receive_seat_count": 1,
            "deliver_seat_count": 1,
            "non_futures_net": 0.0,
        }],
    }


def _write_seat_tables(root, *, duplicate_product_daily: bool = False) -> None:
    for table, rows in _seat_rows().items():
        payload = rows * 2 if table == "derive_product_daily" and duplicate_product_daily else rows
        _write_partition(root, f"futuresseatdata/{table}", payload)


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


def test_parquet_health_checks_only_published_local_source(tmp_path, monkeypatch):
    config = SimpleNamespace(data=SimpleNamespace(
        source="parquet_futures",
        parquet={"root_path": str(tmp_path)},
        cache={"path": str(tmp_path / "cache")},
        audited_nontrading_closes={},
    ))
    monkeypatch.setattr(data_health, "load_config", lambda path: config)
    monkeypatch.setattr(
        data_health,
        "_check_latest_parquet_contract_keys",
        lambda value: {"status": "ok", "path": value["root_path"]},
    )

    result = check_health("parquet.yaml")

    assert result["selected_source"] == "parquet_futures"
    assert result["parquet"]["status"] == "ok"


def _daily_row(symbol: str, day: str, *, exchange: str = "TEST") -> dict:
    row = _row(symbol)
    timestamp = pd.Timestamp(day)
    row.update({
        "exchange": exchange,
        "trade_date": timestamp.date(),
        "trade_datetime": timestamp,
        "sequence": 0,
    })
    return row


def _historical_root(tmp_path, daily_rows: list[dict]) -> dict:
    datasets = {
        "daily": "daily", "1min": "minute", "5min": "five", "15min": "fifteen"
    }
    _write_partition(tmp_path, datasets["daily"], daily_rows)
    for frequency in ("1min", "5min", "15min"):
        _write_partition(
            tmp_path, datasets[frequency],
            [_daily_row("A2609", "2026-08-03")],
        )
    return {"root_path": str(tmp_path), "datasets": datasets}


def test_historical_parquet_gate_accepts_causal_concrete_schedule(tmp_path):
    config = _historical_root(tmp_path, [
        _daily_row("A2609", day)
        for day in ("2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06")
    ])

    result = _check_historical_parquet_contracts(
        config, ["A"], "2026-08-03", "2026-08-06"
    )

    assert result["status"] == "ok"
    assert result["roots"] == 1
    assert result["trading_days"] == 4


def test_all_daily_contract_keys_scans_old_partitions(tmp_path):
    dataset = tmp_path / "daily"
    for month, symbol in (("2026-07", "FG607"), ("2026-08", "FG2609")):
        partition = dataset / f"year_month={month}"
        partition.mkdir(parents=True)
        row = _daily_row(symbol, f"{month}-03", exchange="CZCE")
        pd.DataFrame([row]).to_parquet(partition / "part.parquet", index=False)

    result = _check_all_daily_contract_keys({
        "root_path": str(tmp_path), "datasets": {"daily": "daily"},
    })

    assert result["status"] == "invalid"
    assert result["partitions"] == 2
    assert result["czce_three_digit_rows"] == 1


def test_all_parquet_schemas_scans_old_minute_partitions(tmp_path):
    config = _historical_root(tmp_path, [
        _daily_row("A2609", day)
        for day in ("2026-08-03", "2026-08-04")
    ])
    old = tmp_path / "minute" / "year_month=2026-07"
    old.mkdir(parents=True)
    pd.DataFrame([_daily_row("A2609", "2026-07-31")]).drop(
        columns="position"
    ).to_parquet(old / "part.parquet", index=False)

    result = _check_all_parquet_schemas(config)

    assert result["status"] == "invalid"
    assert result["datasets"]["1min"]["schema_variants"] == 2
    assert result["datasets"]["1min"]["missing_required_examples"]


def test_seat_parquet_gate_accepts_current_consumed_tables(tmp_path):
    _write_seat_tables(tmp_path)

    result = _check_all_seat_parquet_keys({"root_path": str(tmp_path)})

    assert result["status"] == "ok"
    assert result["datasets"]["derive_product_daily"]["rows"] == 1


def test_seat_parquet_gate_rejects_duplicate_business_key(tmp_path):
    _write_seat_tables(tmp_path, duplicate_product_daily=True)

    result = _check_all_seat_parquet_keys({"root_path": str(tmp_path)})

    assert result["status"] == "invalid"
    assert (
        result["datasets"]["derive_product_daily"]["duplicate_key_groups"] == 1
    )


def test_historical_parquet_gate_rejects_unknown_post_listing_gap(tmp_path):
    config = _historical_root(tmp_path, [
        _daily_row("A2609", "2026-08-03"),
        _daily_row("A2609", "2026-08-04"),
        _daily_row("B2609", "2026-08-05"),
        _daily_row("A2609", "2026-08-06"),
    ])

    result = _check_historical_parquet_contracts(
        config, ["A"], "2026-08-03", "2026-08-06"
    )

    assert result["status"] == "invalid"
    assert result["error_type"] == "CloseDataQualityError"


def test_strict_health_runs_full_history_gate(tmp_path, monkeypatch):
    nested_cache = tmp_path / "cache" / "selected_contracts"
    nested_cache.mkdir(parents=True)
    (nested_cache / "request.parquet").write_bytes(b"cache")
    config = SimpleNamespace(
        data=SimpleNamespace(
            source="parquet_futures",
            parquet={"root_path": str(tmp_path)},
            cache={"path": str(tmp_path / "cache")},
            audited_nontrading_closes={},
        ),
        universe=["A"],
        date_range=SimpleNamespace(start="2026-08-03", end="2026-08-06"),
    )
    monkeypatch.setattr(data_health, "load_config", lambda path: config)
    monkeypatch.setattr(
        data_health,
        "_check_latest_parquet_contract_keys",
        lambda value: {"status": "ok"},
    )
    monkeypatch.setattr(
        data_health,
        "_check_all_parquet_schemas",
        lambda value: {"status": "ok"},
    )
    monkeypatch.setattr(
        data_health,
        "_check_all_daily_contract_keys",
        lambda value: {"status": "ok"},
    )
    monkeypatch.setattr(
        data_health,
        "_check_all_seat_parquet_keys",
        lambda value: {"status": "ok"},
    )
    calls = []
    monkeypatch.setattr(
        data_health,
        "_check_historical_parquet_contracts",
        lambda *args, **kwargs: calls.append(1) or {"status": "ok"},
    )

    result = check_health("parquet.yaml", strict=True)

    assert calls == [1]
    assert result["historical_parquet_schemas"]["status"] == "ok"
    assert result["historical_seat_tables"]["status"] == "ok"
    assert result["historical_daily_contract_keys"]["status"] == "ok"
    assert result["historical_daily"]["status"] == "ok"
    assert result["cache"]["parquet_files"] == 1
