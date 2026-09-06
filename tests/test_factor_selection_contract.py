from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from workflows.factor_selection import _compact_representatives, _load_library


def test_effective_library_selection_accepts_registered_daily_horizon_three(tmp_path):
    library = tmp_path / "library.json"
    library.write_text(json.dumps({
        "schema_version": 2,
        "factors": [{
            "factor": "intraday_probe",
            "family": "intraday",
            "status": "effective",
            "frequency": "daily",
            "selected_period": 3,
            "approved_periods": [3],
            "direction": 1,
        }],
    }), encoding="utf-8")
    config = SimpleNamespace(factor_library=SimpleNamespace(path=str(library)))

    _, rows = _load_library(config, allowed_horizons=None)

    assert rows[0]["approved_periods"] == [3]
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
