from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from workflows.factor_selection import (
    _compact_representatives,
    _load_library,
    _recommended_nested_candidate,
    _run_nested_portfolio_search,
)


def test_effective_library_selection_accepts_registered_daily_horizon_three(tmp_path):
    library = tmp_path / "library.json"
    library.write_text(json.dumps({
        "schema_version": 3,
        "factors": [{
            "factor": "intraday_probe",
            "family": "intraday",
            "status": "effective",
            "signal_frequency": "daily",
            "input_bar_frequency": "1min",
            "best_period": 3,
            "direction": 1,
        }],
    }), encoding="utf-8")
    config = SimpleNamespace(factor_library=SimpleNamespace(path=str(library)))

    _, rows = _load_library(config, allowed_horizons=None)

    assert rows[0]["best_period"] == 3
    assert rows[0]["family"] == "intraday"


def test_effective_library_selection_rejects_retired_schema(tmp_path):
    library = tmp_path / "library.json"
    library.write_text(
        json.dumps({"schema_version": 1, "factors": []}), encoding="utf-8"
    )
    config = SimpleNamespace(factor_library=SimpleNamespace(path=str(library)))

    with pytest.raises(ValueError, match="unsupported effective factor library schema"):
        _load_library(config, allowed_horizons=None)


def test_compact_selection_keeps_strongest_distinct_cluster_representatives():
    rows = [
        {
            "factor": "strong_a",
            "cluster_id": 1,
            "family": "intraday_advanced",
            "segment_positive_ratio": 1.0,
            "worst_segment_mean_ic": 0.03,
            "mean_ic": 0.04,
            "coverage": 1.0,
            "rank_churn": 0.2,
        },
        {
            "factor": "strong_b",
            "cluster_id": 2,
            "family": "intraday_advanced",
            "segment_positive_ratio": 1.0,
            "worst_segment_mean_ic": 0.02,
            "mean_ic": 0.03,
            "coverage": 1.0,
            "rank_churn": 0.2,
        },
        {
            "factor": "weak_c",
            "cluster_id": 3,
            "family": "intraday_advanced",
            "segment_positive_ratio": 2 / 3,
            "worst_segment_mean_ic": 0.01,
            "mean_ic": 0.02,
            "coverage": 1.0,
            "rank_churn": 0.2,
        },
    ]

    assert _compact_representatives(rows, 2) == ["strong_a", "strong_b"]


def test_recommended_nested_candidate_uses_smallest_near_best_size():
    base = {
        "positive_segment_ratio": 1.0,
        "median_annual_return": 0.12,
        "worst_drawdown": -0.10,
        "annual_turnover": 20.0,
    }
    path = [
        {**base, "factor_count": 4, "worst_sharpe": 0.93,
         "median_sharpe": 1.13, "factors": ["a", "b", "c", "d"]},
        {**base, "factor_count": 5, "worst_sharpe": 1.00,
         "median_sharpe": 1.20, "factors": ["a", "b", "c", "d", "e"]},
        {**base, "factor_count": 6, "worst_sharpe": 1.02,
         "median_sharpe": 1.22, "factors": ["a", "b", "c", "d", "e", "f"]},
    ]

    selected = _recommended_nested_candidate(path, sharpe_tolerance=0.10)

    assert selected["factor_count"] == 4


def test_nested_search_produces_a_true_incremental_path():
    dates = pd.bdate_range("2025-01-01", periods=80)
    ic = pd.DataFrame({
        "a": 0.02,
        "b": 0.018,
        "c": 0.016,
        "d": 0.014,
    }, index=dates)

    class Evaluator:
        def ledger(self, factors, recipe):
            del recipe
            scale = 0.0001 * len(factors)
            values = pd.Series(scale, index=dates, dtype=float)
            values.iloc[0] = 0.0
            return pd.DataFrame({
                "net_return": values,
                "executed_traded_notional": 0.1,
            }, index=dates)

    path, _, stop_reason = _run_nested_portfolio_search(
        evaluator=Evaluator(),
        portfolio_ic=ic,
        representatives=["a", "b", "c", "d"],
        recipe=SimpleNamespace(),
        segments=[(dates[0], dates[39]), (dates[40], dates[-1])],
        max_factors=4,
        exact_width=3,
        patience=3,
    )

    assert [row["factor_count"] for row in path] == [2, 3, 4]
    assert all(
        set(path[index - 1]["factors"]).issubset(path[index]["factors"])
        for index in range(1, len(path))
    )
    assert stop_reason == "candidate_pool_exhausted"
