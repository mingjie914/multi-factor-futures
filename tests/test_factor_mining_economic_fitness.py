from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_mining.api import TargetSpec
from factor_mining.validation import (
    PreparedTarget,
    ValidationConfig,
    evaluate_candidate,
    predictive_ic_decay,
    signal_rank_persistence,
)


def _aligned_target(cost_bps: float) -> tuple[np.ndarray, PreparedTarget]:
    rng = np.random.default_rng(20260728)
    values = rng.normal(scale=0.002, size=(80, 8)).astype(np.float32)
    ranks = pd.DataFrame(values).rank(axis=1, pct=True).to_numpy(dtype=np.float32)
    index = pd.date_range("2024-01-02 09:00", periods=len(values), freq="1min")
    symbols = pd.Index([f"F{i}" for i in range(values.shape[1])])
    target = PreparedTarget(
        index=index,
        symbols=symbols,
        values=values,
        rank_values=ranks,
        spec=TargetSpec(name="forward_5p", horizon_bars=5, cost_bps=cost_bps),
    )
    raw_signal = np.full_like(values, np.nan)
    raw_signal[:-1] = values[1:]
    return raw_signal, target


def test_economic_fitness_prorates_fixed_annual_cost_without_turnover_charge():
    signal, free_target = _aligned_target(cost_bps=0.0)
    # Deliberately extreme annual rate keeps the score away from its +1 clip;
    # the arithmetic assertion below is the actual policy regression guard.
    annual_cost_bps = 50_000.0
    _, costly_target = _aligned_target(cost_bps=annual_cost_bps)
    config = ValidationConfig(
        min_cross_section=4,
        min_time_observations=20,
        neutralize_volatility=False,
        rebalance_every_bars=5,
        economic_fitness_weight=1.0,
        complexity_penalty=0.0,
    )

    free = evaluate_candidate(signal, free_target, config, full_diagnostics=False)
    costly = evaluate_candidate(signal, costly_target, config, full_diagnostics=False)

    assert free.metrics["rebalance_every_bars"] == 5
    assert free.metrics["rank_weight_turnover_mean"] > 0.0
    assert costly.metrics["rank_weight_net_mean"] < free.metrics["rank_weight_net_mean"]
    assert (
        free.metrics["rank_weight_net_mean"]
        - costly.metrics["rank_weight_net_mean"]
    ) == pytest.approx(
            annual_cost_bps / 10_000.0 * 5.0 / (240.0 * 252.0)
    )
    assert costly.metrics["cost_uses_turnover"] is False
    assert costly.metrics["mining_cost_definition"] == (
        "fixed_annual_cost_prorated_by_target_holding_bars"
    )
    assert costly.metrics["cost_adjusted_return_score"] < free.metrics[
        "cost_adjusted_return_score"
    ]
    assert costly.fitness < free.fitness
def test_forward_ic_curve_and_signal_half_life_have_bar_semantics():
    ascending = np.arange(8, dtype=np.float32)
    descending = ascending[::-1]
    signal = np.vstack([
        ascending if row % 2 == 0 else descending for row in range(12)
    ])
    ranks = pd.DataFrame(signal).rank(axis=1, pct=True).to_numpy(dtype=np.float32)
    index = pd.date_range("2024-01-02 09:00", periods=len(signal), freq="1min")
    symbols = pd.Index([f"F{i}" for i in range(signal.shape[1])])
    targets = {
        horizon: PreparedTarget(
            index=index,
            symbols=symbols,
            values=signal,
            rank_values=ranks,
            spec=TargetSpec(name=f"forward_{horizon}p", horizon_bars=horizon),
        )
        for horizon in (1, 3)
    }

    decay = predictive_ic_decay(signal, targets, min_cross_section=4)
    persistence = signal_rank_persistence(
        signal, lags=(1, 2, 3), min_cross_section=4
    )

    assert decay["horizon_unit"] == "decision_bars"
    assert decay["curve"]["1"]["mean_rank_ic"] == pytest.approx(1.0)
    assert persistence["curve"]["1"]["mean_rank_autocorrelation"] == pytest.approx(-1.0)
    assert persistence["half_life_bars"] == 1
