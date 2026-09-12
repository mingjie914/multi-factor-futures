from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from workflows.factor_selection import (
    _compact_representatives,
    _load_library,
)


def test_search_cap_groups_match_production_not_fine_sector_taxonomy():
    from strategies.combined import SECTOR_OF
    from workflows.factor_selection import PORTFOLIO_CAP_GROUPS
    assert PORTFOLIO_CAP_GROUPS == SECTOR_OF
    assert PORTFOLIO_CAP_GROUPS["AU"] == PORTFOLIO_CAP_GROUPS["CU"]
    assert PORTFOLIO_CAP_GROUPS["IF"] == PORTFOLIO_CAP_GROUPS["T"]
    assert len(set(PORTFOLIO_CAP_GROUPS.values())) == 5


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


def test_shortlist_does_not_prefer_smallest_and_keeps_tradeoffs():
    from workflows.factor_selection import _portfolio_shortlist
    base = dict(status="evaluated", segment_count=5, positive_segment_ratio=1.,
                median_annual_return=.12, full_annual_return=.12, worst_drawdown=-.1,
                annual_turnover=20.)
    rows = [
        dict(base, factor_count=2, factors=["a", "b"], worst_sharpe=.9, median_sharpe=1.1),
        dict(base, factor_count=6, factors=list("abcdef"), worst_sharpe=1.1, median_sharpe=1.3),
        dict(base, factor_count=8, factors=list("ghijklmn"), worst_sharpe=1., median_sharpe=1.5),
    ]
    assert {r["factor_count"] for r in _portfolio_shortlist(rows)} == {6, 8}


def test_multipath_search_explores_beyond_three_flat_sizes():
    import numpy as np
    from workflows.factor_selection import _run_portfolio_search
    dates = pd.bdate_range("2020-01-01", periods=160)
    rng = np.random.default_rng(2)
    ic = pd.DataFrame(rng.normal(.02, .1, (160, 8)), index=dates, columns=list("abcdefgh"))
    class Evaluator:
        def ledger(self, factors, recipe):
            values = .0002 + .001 * np.sin(np.arange(160))
            values[0] = 0.0
            return pd.DataFrame({"net_return": values, "executed_traded_notional": .1}, index=dates)
    path, rows, reason = _run_portfolio_search(
        evaluator=Evaluator(), portfolio_ic=ic, representatives=list(ic), recipe=None,
        segments=[(dates[i], dates[i+39]) for i in range(0,160,40)],
        max_factors=8, exact_width=2, beam_width=4)
    assert [r["factor_count"] for r in path] == list(range(2,9))
    assert reason == "candidate_pool_exhausted"
    assert len([r for r in rows if r["factor_count"] == 2]) == 2


def test_exact_winner_has_a_reserved_expansion():
    import numpy as np
    from workflows.factor_selection import _run_portfolio_search
    dates = pd.bdate_range("2020-01-01", periods=160)
    rng = np.random.default_rng(7)
    ic = pd.DataFrame(rng.normal(.02, .1, (160, 6)), index=dates, columns=list("abcdef"))
    calls = []
    class Evaluator:
        def ledger(self, factors, recipe):
            calls.append(tuple(factors))
            # The second exact pair wins despite being behind the first proxy.
            mean = .001 if len(calls) == 2 else .0001
            values = mean + .001 * np.sin(np.arange(160))
            values[0] = 0.0
            return pd.DataFrame({"net_return": values, "executed_traded_notional": .1}, index=dates)
    _run_portfolio_search(
        evaluator=Evaluator(), portfolio_ic=ic, representatives=list(ic), recipe=None,
        segments=[(dates[i], dates[i+39]) for i in range(0,160,40)],
        max_factors=3, exact_width=2, beam_width=4)
    assert len(calls[2]) == 3
    assert set(calls[1]).issubset(calls[2])


def test_budgeted_pool_covers_all_factors_resumes_and_ignores_future(tmp_path):
    import duckdb
    import numpy as np
    from workflows.factor_selection import _run_budgeted_pool_search
    dates = pd.bdate_range("2020-01-01", periods=180)
    rng = np.random.default_rng(42)
    ic = pd.DataFrame(rng.normal(.03, .1, (180, 8)), index=dates, columns=list("abcdefgh"))
    segments = [(dates[i], dates[i+39]) for i in range(0, 160, 40)]
    class Evaluator:
        end = dates[159]
        def __init__(self):
            self.calls = []
        def clear_transient_caches(self):
            pass
        def ledger(self, members, recipe):
            self.calls.append(tuple(members))
            if len(self.calls) == 1:
                raise ValueError("no positive weight")
            r = .0002 + .001*np.sin(np.arange(160))
            r[0] = 0
            return pd.DataFrame({"net_return": r, "nav": 1000*np.cumprod(1+r),
                                 "executed_traded_notional": .1}, index=dates[:160])
    def run(folder, evaluator, frame):
        folder.mkdir(exist_ok=True)
        with duckdb.connect(str(folder / "results.duckdb")) as db:
            return _run_budgeted_pool_search(evaluator=evaluator, portfolio_ic=frame,
                pool=list(frame), recipe=None, segments=segments,
                clusters={n: i % 3 for i, n in enumerate(frame)}, db=db, output=folder,
                budget=40, beam_width=4, round_width=8)
    evaluator = Evaluator()
    _, rows, reason = run(tmp_path / "original", evaluator, ic)
    assert reason == "exact_backtest_budget_exhausted"
    assert len(rows) == len(set(evaluator.calls)) == 40
    assert {n for r in rows if r["phase"] == "seed_coverage" for n in r["factors"]} == set(ic)
    assert rows[0]["status"] == "rejected_runtime"
    assert max(r["factor_count"] for r in rows) >= 4
    resumed = Evaluator()
    _, replay, _ = run(tmp_path / "original", resumed, ic)
    assert resumed.calls == []
    assert replay == rows
    poisoned = ic.copy()
    poisoned.loc[dates[159]:] = rng.normal(-100, 1000, (21, 8))
    _, comparison, _ = run(tmp_path / "future_poisoned", Evaluator(), poisoned)
    assert [(r["factors"], r["status"]) for r in comparison] == [(r["factors"], r["status"]) for r in rows]
