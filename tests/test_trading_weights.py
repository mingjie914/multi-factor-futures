from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from research.portfolio_experiment_support import FactorPanelRunner
from trading.artifacts import read_artifact, write_artifact
from trading.weights import require_closed_date, strategy_contracts


def test_immutable_artifact_repeats_and_detects_corruption(tmp_path):
    rows = [{"strategy_id": "s", "weight": -0.12345678901234567}]
    identity = {"as_of": "2026-09-11", "source": "a"}
    first = write_artifact(tmp_path, identity, {"weights.csv": rows})
    stamp = (first / "weights.csv").stat().st_mtime_ns
    assert write_artifact(tmp_path, identity, {"weights.csv": rows}) == first
    assert (first / "weights.csv").stat().st_mtime_ns == stamp
    assert read_artifact(first)["identity"] == identity
    (first / "weights.csv").write_text("wrong", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        read_artifact(first)


def test_artifact_new_input_creates_separate_revision(tmp_path):
    a = write_artifact(tmp_path, {"source": "a"}, {"weights.csv": [{"weight": 1}]})
    b = write_artifact(tmp_path, {"source": "b"}, {"weights.csv": [{"weight": 1}]})
    assert a != b and a.is_dir() and b.is_dir()


def test_as_of_is_a_closed_published_day_not_system_today():
    now = datetime(2026, 9, 15, 14, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    require_closed_date("2026-09-11", "2026-09-11", now=now)
    with pytest.raises(ValueError, match="published"):
        require_closed_date("2026-09-15", "2026-09-11", now=now)
    with pytest.raises(ValueError, match="close"):
        require_closed_date("2026-09-15", "2026-09-15", now=now)


def test_current_catalog_binds_eighteen_factors_and_not_legacy_ten():
    contracts = strategy_contracts(["multi_source_resilient"])
    assert len(contracts[0]["factors"]) == 18
    assert contracts[0]["strategy_id"] == "multi_source_resilient"
    assert contracts[0]["recipe"]["factor_weight_method"] == "lw_abs"
    assert contracts[0]["recipe"]["gross_exposure"] == 2.0
    assert contracts[0]["formal"] is True
    assert strategy_contracts({"formal_main": "@formal"})[0]["catalog_strategy_id"] == "multi_source_resilient"
    with pytest.raises(ValueError, match="archived"):
        strategy_contracts(["multi_source_balanced"])


def test_tail_chunks_keep_global_boundaries_and_exact_values():
    dates = pd.bdate_range("2020-01-01", periods=1207)
    raw = pd.Series(np.random.default_rng(81).normal(size=len(dates)), index=dates)
    full = pd.Series(np.nan, index=dates)
    tail = full.copy()
    original = list(FactorPanelRunner._iter_factor_chunks(dates))
    selected = list(FactorPanelRunner._iter_factor_chunks(dates, compute_start=dates[-85]))
    assert len(selected) < len(original)
    for target, request in original:
        full.loc[target] = raw.loc[request].rolling(120).sum().shift(1).reindex(target)
    for target, request in selected:
        assert any(target.equals(t) and request.equals(r) for t, r in original)
        tail.loc[target] = raw.loc[request].rolling(120).sum().shift(1).reindex(target)
    pd.testing.assert_series_equal(tail.loc[dates[-85]:], full.loc[dates[-85]:], check_exact=True)


def test_tail_checkpoint_cannot_reuse_full_or_different_tail_contract():
    from types import SimpleNamespace
    runner = FactorPanelRunner.__new__(FactorPanelRunner)
    runner.cal = pd.bdate_range("2026-01-01", periods=10)
    runner.u = ["RB"]
    runner.env = SimpleNamespace(data_manager=SimpleNamespace(source=SimpleNamespace(cache_namespace="a")))
    full = runner._checkpoint_contract(["x"])
    runner.compute_start = pd.Timestamp("2026-01-06")
    tail = runner._checkpoint_contract(["x"])
    assert full != tail


def test_buffered_instances_share_factor_identity_but_not_holding_state():
    from types import SimpleNamespace
    from research.historical_portfolio_search import PortfolioEvaluator, PortfolioRecipe
    instances = strategy_contracts({"account_a": "@formal", "account_b": "@formal"})
    assert [r["strategy_id"] for r in instances] == ["account_a", "account_b"]
    assert instances[0]["factors"] == instances[1]["factors"]
    date = pd.Timestamp("2026-09-11")
    evaluator = PortfolioEvaluator.__new__(PortfolioEvaluator)
    evaluator.runner = SimpleNamespace(u=list("ABCD"), env=SimpleNamespace(sector_of=dict.fromkeys("ABCD", "x")))
    evaluator._score_matrix = lambda *a: pd.DataFrame([[4, 3, 2, 1]], index=[date], columns=list("ABCD"))
    evaluator._risk_eligible = lambda *a: list("ABCD")
    evaluator._asset_weights = lambda day, pool, recipe: pd.Series(1.0, index=pool)
    recipe = PortfolioRecipe(top_n=1, sector_cap=0, asset_max_fraction=1, rank_exit_buffer=1)
    cold = evaluator.target_for_holdings(["a"], recipe, date, {})
    held = evaluator.target_for_holdings(["a"], recipe, date, {"B": 1, "D": -1})
    assert cold["A"] == 1 and cold.get("B", 0) == 0
    assert held.get("A", 0) == 0 and held["B"] == 1 and held["D"] == -1


def test_missing_buffer_state_fails_before_factor_or_market_calculation(monkeypatch):
    from types import SimpleNamespace
    from trading.weights import generate_weights
    source = SimpleNamespace(fetch_latest_trade_date=lambda: pd.Timestamp("2026-09-11"), close=lambda: None)
    monkeypatch.setattr("trading.weights.DataManager.from_config", lambda c: SimpleNamespace(source=source))
    with pytest.raises(ValueError, match="dated, reconciled holdings"):
        generate_weights({"formal_main": "@formal"})


def test_runtime_identity_changes_when_numeric_backend_changes(monkeypatch):
    from trading.weights import runtime_contract
    monkeypatch.setenv("MF_FACTOR_KERNEL_MODE", "reference")
    reference = runtime_contract()
    monkeypatch.setenv("MF_FACTOR_KERNEL_MODE", "native")
    native = runtime_contract()
    assert reference != native
    assert reference["factor_kernel_mode"] == "reference"
    assert "python" in reference and "native_binary_sha256" in native
