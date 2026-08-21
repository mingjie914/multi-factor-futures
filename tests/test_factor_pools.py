from __future__ import annotations

from pathlib import Path
import runpy

import yaml

import factors.library  # noqa: F401 - populate the discovery registry
from core.config import load_config
from core.registry import list_registered


def test_no_superseded_candidate_pool_is_promoted():
    root = Path(__file__).resolve().parents[1]
    pools = yaml.safe_load(
        (root / "research" / "factor_pools.yaml").read_text(encoding="utf-8")
    )
    assert list_registered("factor")["factor"]
    assert pools["historical_screened_pool"] == []
    assert pools["oos_validated_pool"] == []
    assert pools["portfolio_eligible_pool"] == []
    assert pools["deployment_approved_pool"] == []


def test_validated_factor_watchlist_loads_without_becoming_production_factors():
    config = load_config("config/validated_factors.yaml")
    registered = set(list_registered("factor")["factor"])

    assert len(config.factors) == 10
    assert len(config.validated_candidates) == 13
    assert set(config.validated_candidates) <= registered
    assert set(config.factors) & set(config.validated_candidates) == {
        "intraday_price_peak_count_20d"
    }


def test_retired_6f_snapshot_configs_match_their_frozen_definitions():
    root = Path(__file__).resolve().parents[1]
    for name in ("6f", "6f_icir"):
        snapshot = root / "snapshot" / name
        factors = runpy.run_path(str(snapshot / "combined.py"))["FACTORS"]
        config = yaml.safe_load((snapshot / "config.yaml").read_text(encoding="utf-8"))
        assert config["factors"] == list(factors)
