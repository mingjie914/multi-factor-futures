from __future__ import annotations

from pathlib import Path

import yaml

import factors.library  # noqa: F401 - populate the discovery registry
from core.registry import list_registered


def test_historical_candidates_are_registered_but_not_promoted():
    root = Path(__file__).resolve().parents[1]
    pools = yaml.safe_load(
        (root / "research" / "factor_pools.yaml").read_text(encoding="utf-8")
    )
    registered = set(list_registered("factor")["factor"])
    candidate_sets = pools["historical_screened_pool"]
    candidates = {
        factor["name"]
        for candidate_set in candidate_sets
        for factor in candidate_set["factors"]
    }

    assert candidates
    assert candidates <= registered
    assert all(not item["independent_oos"] for item in candidate_sets)
    assert all(not item["portfolio_eligible"] for item in candidate_sets)
    assert all(not item["deployment_eligible"] for item in candidate_sets)
    assert pools["oos_validated_pool"] == []
    assert pools["portfolio_eligible_pool"] == []
    assert pools["deployment_approved_pool"] == []
