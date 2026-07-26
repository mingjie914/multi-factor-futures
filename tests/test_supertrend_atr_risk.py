from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.supertrend_atr_risk import SupertrendATRRiskStrategy


def _ohlc(periods: int = 150):
    dates = pd.date_range("2023-01-02", periods=periods, freq="B")
    step = np.arange(periods, dtype=float)
    close = pd.DataFrame(
        {
            "A": 100.0 * np.exp(0.003 * step + 0.025 * np.sin(step / 6.0)),
            "B": 90.0 * np.exp(-0.002 * step + 0.035 * np.sin(step / 5.0)),
            "C": 110.0 * np.exp(0.001 * step + 0.06 * np.sin(step / 4.0)),
            "D": 105.0 * np.exp(-0.001 * step + 0.04 * np.cos(step / 7.0)),
        },
        index=dates,
    )
    spread = pd.DataFrame(
        0.01 + 0.004 * np.abs(np.sin(step[:, None] / 9.0)),
        index=dates,
        columns=["x"],
    ).to_numpy()
    spread = np.repeat(spread, close.shape[1], axis=1)
    high = close * (1.0 + spread)
    low = close * (1.0 - spread)
    return high, low, close


def _strategy(**overrides):
    params = {
        "rebalance_freq": 5,
        "risk_window": 40,
        "target_volatility": 0.12,
        "asset_vol_budget": 0.025,
        "sector_vol_budget": 0.06,
        "gross_cap": 1.5,
        "net_cap": 0.35,
        "turnover_cap": 0.50,
        "sector_map": {"A": "s1", "B": "s1", "C": "s2", "D": "s2"},
    }
    params.update(overrides)
    return SupertrendATRRiskStrategy(**params)


def test_supertrend_sleeve_is_t_plus_one_and_point_in_time():
    high, low, close = _ohlc()
    base = _strategy().run(high, low, close)

    first_decision = base.weights_history.index[0]
    first_effective = close.index[close.index.get_loc(first_decision) + 1]
    assert base.turnover.loc[first_decision] == 0.0
    assert base.turnover.loc[first_effective] > 0.0

    cutoff = close.index[100]
    changed_high, changed_low, changed_close = high.copy(), low.copy(), close.copy()
    changed_close.loc[changed_close.index > cutoff, "A"] *= 3.0
    changed_high.loc[changed_high.index > cutoff, "A"] *= 3.0
    changed_low.loc[changed_low.index > cutoff, "A"] *= 3.0
    changed = _strategy().run(changed_high, changed_low, changed_close)
    pd.testing.assert_frame_equal(
        base.weights_history.loc[:cutoff], changed.weights_history.loc[:cutoff]
    )


def test_supertrend_sleeve_has_long_short_states_and_respects_risk_limits():
    strategy = _strategy()
    result = strategy.run(*_ohlc())
    weights = result.weights_history

    assert np.isfinite(weights.to_numpy()).all()
    assert (weights > 0).any().any()
    assert (weights < 0).any().any()
    assert weights.abs().sum(axis=1).max() <= strategy.gross_cap + 1e-8
    assert weights.sum(axis=1).abs().max() <= strategy.net_cap + 1e-8
    assert result.metrics["max_asset_vol_proxy"] <= strategy.asset_vol_budget + 1e-8
    assert (
        result.metrics["max_sector_standalone_vol"]
        <= strategy.sector_vol_budget + 1e-8
    )
    assert result.metrics["max_turnover"] <= strategy.turnover_cap + 1e-8


def test_supertrend_costs_nav_and_turnover_are_aligned():
    class CostModel:
        def estimate_cost(self, target, current, date):
            return float((target - current).abs().sum()) * 0.001

    result = _strategy().run(*_ohlc(), cost_model=CostModel())
    first_trade = result.turnover[result.turnover > 0].index[0]
    assert result.costs.loc[first_trade] == pytest.approx(
        result.turnover.loc[first_trade] * 0.001
    )
    assert result.metrics["total_transaction_cost"] == pytest.approx(
        result.costs.sum()
    )
    assert result.nav.iloc[0] == 1.0
    assert np.isfinite(result.nav.to_numpy()).all()
    assert result.failure_ledger == []


def test_supertrend_missing_ohlc_fails_closed_for_affected_asset():
    high, low, close = _ohlc()
    high.loc[high.index[80:], "D"] = np.nan
    low.loc[low.index[80:], "D"] = np.nan
    result = _strategy().run(high, low, close)

    decisions = result.weights_history.loc[result.weights_history.index >= close.index[100]]
    assert not decisions.empty
    assert (decisions["D"].abs() <= 1e-12).all()
    assert result.failure_ledger == []


def test_supertrend_rejects_mixed_ohlc_price_scales():
    high, low, close = _ohlc()
    close["A"] *= 1.30
    with pytest.raises(ValueError, match="price scales are inconsistent"):
        _strategy().run(high, low, close)


def test_scheduled_only_mode_has_fewer_decisions():
    high, low, close = _ohlc()
    event_driven = _strategy(rebalance_on_flip=True).run(high, low, close)
    scheduled = _strategy(rebalance_on_flip=False).run(high, low, close)
    assert scheduled.metrics["decision_count"] < event_driven.metrics["decision_count"]
