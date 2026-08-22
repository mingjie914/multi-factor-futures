from __future__ import annotations

import numpy as np
import pandas as pd

from external_strategies.guosen_trend_index.current_core_compare import _report_periods

from external_strategies.guosen_trend_index.production_compare import (
    ANNUAL_FEE,
    ANNUAL_ROLL_COST,
    F6,
    F10,
    F13,
    F14,
    KEPT47_FACTORS,
    NEW21,
    PERIODS_PER_YEAR,
    TRADE_COST_RATE,
    _factor_weights,
    _ledger_from_weights,
    _validate_fixed_factor_sets,
)


def test_current_comparison_keeps_oos_fixed_and_extends_only_simulated_live():
    periods = dict(
        (name, (start, end))
        for name, start, end in _report_periods(pd.Timestamp("2026-09-30"))
    )
    assert periods["oos_20250101_20260514"] == (
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2026-05-14"),
    )
    assert periods["simulated_live_from_20260515"] == (
        pd.Timestamp("2026-05-15"),
        pd.Timestamp("2026-09-30"),
    )


def test_validated_factor_pool_and_named_sets_are_stable():
    _validate_fixed_factor_sets()
    assert (len(F6), len(F10), len(F13), len(F14)) == (6, 10, 13, 14)
    assert set(F13) - set(F10) == {
        "intraday_jump_intensity_20d",
        "intraday_dtws_20d",
        "intraday_seat_long_short_seat_ratio_20d",
    }
    assert {
        "intraday_open_close_volume_ratio_20d",
        "intraday_turnover_velocity_20d",
    }.issubset(F10)
    assert len(dict.fromkeys(F6 + KEPT47_FACTORS + NEW21)) == 74


def test_factor_weights_are_long_only_and_normalized():
    history = pd.DataFrame(
        {
            "a": np.linspace(0.01, 0.03, 40),
            "b": np.linspace(0.02, 0.01, 40),
            "c": np.sin(np.linspace(0.0, 3.0, 40)) * 0.01,
        }
    )

    weights = _factor_weights(history)

    assert weights.ge(0.0).all()
    assert np.isclose(weights.sum(), 1.0)


def test_ledger_recomputes_turnover_cost_after_leverage_scaling():
    dates = pd.bdate_range("2024-01-02", periods=3)
    weights = pd.DataFrame(
        [[0.5, -0.5], [0.25, -0.25], [0.25, -0.25]],
        index=dates,
        columns=["A", "B"],
    )
    returns = pd.DataFrame(0.0, index=dates, columns=weights.columns)

    ledger = _ledger_from_weights(weights, returns, dates[0], dates[-1])

    assert ledger.iloc[0]["net_return"] == 0.0
    daily_holding_cost = (ANNUAL_FEE + ANNUAL_ROLL_COST) / PERIODS_PER_YEAR
    assert np.isclose(ledger.iloc[0]["turnover"], 0.0)
    assert np.isclose(ledger.iloc[0]["decision_turnover"], 1.0)
    assert np.isclose(ledger.iloc[1]["executed_traded_notional"], 1.0)
    expected_turnover = 2.0 * (
        0.5 / (1.0 - TRADE_COST_RATE - daily_holding_cost) - 0.25
    )
    assert np.isclose(ledger.iloc[1]["turnover"], 1.0)
    assert np.isclose(ledger.iloc[1]["decision_turnover"], expected_turnover)
    assert np.isclose(ledger.iloc[2]["turnover"], expected_turnover)
    expected = -(TRADE_COST_RATE + daily_holding_cost)
    assert np.isclose(ledger.iloc[1]["net_return"], expected)
    assert np.isclose(ledger.iloc[1]["gross_exposure"], 1.0)
