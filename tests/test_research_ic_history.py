from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPTS = Path(__file__).parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from exp22_full_pool import prepare_ic_history
from exp_core import ExpEnv, prepare_erc_returns
from optimization.risk_budgeting import RiskBudgetingOptimizer


def test_late_starting_factor_waits_for_minimum_ic_history():
    index = pd.bdate_range("2020-01-01", periods=59)
    history = pd.DataFrame(
        {
            "stable_a": np.linspace(-0.1, 0.1, len(index)),
            "stable_b": np.linspace(0.2, -0.2, len(index)),
            "late": [np.nan] * 58 + [0.05],
        },
        index=index,
    )

    clean = prepare_ic_history(history, minimum_observations=30)

    assert list(clean.columns) == ["stable_a", "stable_b"]
    assert len(clean) == 59


def test_late_starting_factor_enters_after_enough_complete_history():
    index = pd.bdate_range("2020-01-01", periods=59)
    late = pd.Series(np.nan, index=index)
    late.iloc[-30:] = np.linspace(-0.05, 0.05, 30)
    history = pd.DataFrame(
        {
            "stable_a": np.linspace(-0.1, 0.1, len(index)),
            "stable_b": np.linspace(0.2, -0.2, len(index)),
            "late": late,
        },
        index=index,
    )

    clean = prepare_ic_history(history, minimum_observations=30)

    assert list(clean.columns) == ["stable_a", "stable_b", "late"]
    assert len(clean) == 30
    assert clean.notna().all().all()


def test_erc_history_excludes_unlisted_asset_without_emptying_pool():
    index = pd.bdate_range("2022-01-01", periods=20)
    returns = pd.DataFrame(
        {
            "old_a": np.linspace(-0.01, 0.01, len(index)),
            "old_b": np.linspace(0.02, -0.02, len(index)),
            "future": np.nan,
        },
        index=index,
    )

    clean = prepare_erc_returns(returns, ["old_a", "future", "old_b"])

    assert list(clean.columns) == ["old_a", "old_b"]
    assert len(clean) == 20


def test_erc_fallback_maps_weights_to_filtered_assets(monkeypatch):
    index = pd.bdate_range("2022-01-01", periods=20)
    env = ExpEnv.__new__(ExpEnv)
    env.cal = index
    env.daily_ret = pd.DataFrame(
        {
            "old_a": np.linspace(-0.01, 0.01, len(index)),
            "future": np.nan,
            "old_b": np.linspace(0.02, -0.02, len(index)),
        },
        index=index,
    )

    def fail_erc(*args, **kwargs):
        raise ValueError("force inverse-volatility fallback")

    monkeypatch.setattr(RiskBudgetingOptimizer, "_erc_weights", fail_erc)
    weights = env.erc_w(["future", "old_a", "old_b"], index[-1] + pd.Timedelta(days=1))

    assert set(weights) == {"old_a", "old_b"}
    assert np.isclose(sum(weights.values()), 1.0)


def test_capped_pool_filters_unlisted_asset_before_ranking():
    index = pd.bdate_range("2022-01-01", periods=20)
    env = ExpEnv.__new__(ExpEnv)
    env.cal = index
    env.daily_ret = pd.DataFrame(
        {
            "old_a": np.linspace(-0.01, 0.01, len(index)),
            "future": np.nan,
            "old_b": np.linspace(0.02, -0.02, len(index)),
        },
        index=index,
    )
    env.sector_of = {}
    row = pd.Series({"future": -100.0, "old_a": 1.0, "old_b": 2.0})

    picks = env.capped(
        row,
        cap_n=3,
        ascending=True,
        date=index[-1] + pd.Timedelta(days=1),
    )

    assert picks == ["old_a", "old_b"]
    assert len(env._eligible_symbols_cache) == 1
