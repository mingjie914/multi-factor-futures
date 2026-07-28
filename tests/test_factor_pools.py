from __future__ import annotations

from pathlib import Path

import yaml

import factors.library  # noqa: F401 - populate the discovery registry
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
