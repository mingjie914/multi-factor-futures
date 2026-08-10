from __future__ import annotations

import numpy as np
import pandas as pd

from external_strategies.guosen_trend_index.production_compare import (
    ANNUAL_FEE,
    F6,
    F10,
    F13,
    F14,
    KEPT47,
    NEW21,
    PERIODS_PER_YEAR,
    TRADE_COST_RATE,
    _factor_weights,
    _ledger_from_weights,
)


def test_validated_factor_pool_and_named_sets_are_stable():
    assert (len(F6), len(F10), len(F13), len(F14)) == (6, 10, 13, 14)
    assert len(dict.fromkeys(F6 + KEPT47 + NEW21)) == 74


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
    assert np.isclose(ledger.iloc[1]["turnover"], 0.5)
    expected = -(0.5 * TRADE_COST_RATE + ANNUAL_FEE / PERIODS_PER_YEAR)
    assert np.isclose(ledger.iloc[1]["net_return"], expected)
    assert np.isclose(ledger.iloc[1]["gross_exposure"], 0.5)
