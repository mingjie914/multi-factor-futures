from __future__ import annotations

import pandas as pd
import pytest

import research.portfolio_experiment_support as experiment_support


def test_latest_local_date_reads_every_shard_and_uses_trade_date(
    tmp_path,
):
    del tmp_path

    class Source:
        @staticmethod
        def fetch_latest_trade_date():
            return pd.Timestamp("2026-08-14")

    manager = type("Manager", (), {"source": Source()})()

    assert experiment_support.latest_local_date(manager) == pd.Timestamp("2026-08-14")


def test_latest_local_date_requires_formal_source_capability():
    manager = type("Manager", (), {"source": object()})()

    with pytest.raises(NotImplementedError, match="latest trade date"):
        experiment_support.latest_local_date(manager)
