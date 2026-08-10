from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from external_strategies.guosen_trend_index.strategy import (
    GuosenTrendIndexBacktester,
    load_snapshot,
)


SNAPSHOT = (
    Path(__file__).parents[1]
    / "external_strategies"
    / "guosen_trend_index"
    / "config.yaml"
)


def test_snapshot_is_isolated_and_selects_five_of_nine():
    spec, factor_sets, raw = load_snapshot(SNAPSHOT)

    assert spec.universe == ("AU", "CU", "IF", "IC", "RB", "SC", "M", "TF", "T")
    assert spec.selection_count == 5
    assert spec.expected_final_holdings == 9
    assert spec.warmup_calendar_days > 0
    assert spec.target_volatility == 0.04
    assert set(factor_sets) == {"6f", "10f", "13f", "14f"}
    assert "目标波动率由用户确认为 4%" in raw["snapshot"]["note"]
    assert "portfolio_gross_cap" not in raw


class _RollingFakeEngine:
    def compute_factors(self, names, dates, universe, parallel=False):
        raw = pd.DataFrame(
            np.arange(len(dates), dtype=float)[:, None],
            index=dates,
            columns=[universe[0]],
        )
        value = raw.rolling(3, min_periods=3).mean().shift(1)
        value = value.reindex(columns=universe).ffill(axis=1)
        return {name: value.copy() for name in names}


def test_chunk_overlap_prevents_boundary_warmup_gaps():
    spec, _, _ = load_snapshot(SNAPSHOT)
    spec = replace(spec, factor_chunk_size=5, factor_chunk_overlap=3)
    dates = pd.bdate_range("2024-01-02", periods=15)
    adapter = GuosenTrendIndexBacktester(None, _RollingFakeEngine(), spec)

    result = adapter.compute_factor_values(["factor"], dates)["factor"]

    assert result.loc[dates[5]].notna().all()
    assert result.loc[dates[10]].notna().all()


def test_backtest_is_long_only_capped_and_keeps_full_calendar():
    spec, _, _ = load_snapshot(SNAPSHOT)
    dates = pd.bdate_range("2024-01-02", periods=45)
    base = np.linspace(100.0, 120.0, len(dates))[:, None]
    offsets = np.arange(len(spec.universe), dtype=float)[None, :]
    close = pd.DataFrame(base + offsets, index=dates, columns=spec.universe)
    values = pd.DataFrame(
        np.tile(np.arange(len(spec.universe), dtype=float), (len(dates), 1)),
        index=dates,
        columns=spec.universe,
    )
    adapter = GuosenTrendIndexBacktester(None, None, spec)

    result = adapter.run_from_values(
        {"factor": values}, {"factor": (("factor", 1),)}, close
    )

    assert result.nav.index.equals(dates)
    assert result.weights.ge(0.0).all().all()
    caps = pd.Series(spec.asset_caps)
    assert result.weights.le(caps + 1e-12, axis="columns").all().all()
    assert result.weights.iloc[0].eq(0.0).all()
    assert result.weights.iloc[-1].gt(0.0).any()


def test_each_factor_selects_its_own_top_five_before_weight_aggregation():
    spec, _, _ = load_snapshot(SNAPSHOT)
    dates = pd.bdate_range("2024-01-02", periods=45)
    base = np.linspace(100.0, 130.0, len(dates))[:, None]
    close = pd.DataFrame(
        base + np.arange(len(spec.universe))[None, :],
        index=dates,
        columns=spec.universe,
    )
    ascending = pd.DataFrame(
        np.tile(np.arange(9, dtype=float), (len(dates), 1)),
        index=dates,
        columns=spec.universe,
    )
    descending = 8.0 - ascending
    adapter = GuosenTrendIndexBacktester(None, None, spec)

    weights, diagnostics = adapter.build_signal_weights(
        {"factor_a": ascending, "factor_b": descending},
        {
            "factor_a": (("factor_a", 1),),
            "factor_b": (("factor_b", 1),),
        },
        close,
    )

    # Each factor selects five independently.  Their top-five sets overlap in
    # one asset, so aggregation contains all nine assets; ranking a composite
    # score first would incorrectly leave only five.
    assert diagnostics.iloc[-1]["active_factors"] == 2
    assert diagnostics.iloc[-1]["selected_assets"] == 9
    assert weights.iloc[-1].gt(0.0).sum() == 9


def test_parameter_weights_are_averaged_inside_factor_before_factor_aggregation():
    spec, _, _ = load_snapshot(SNAPSHOT)
    spec = replace(
        spec,
        target_volatility=0.001,
        asset_caps={symbol: 1.0 for symbol in spec.universe},
    )
    dates = pd.bdate_range("2024-01-02", periods=45)
    common_path = np.linspace(100.0, 130.0, len(dates))[:, None]
    close = pd.DataFrame(
        common_path * np.linspace(1.0, 1.8, len(spec.universe))[None, :],
        index=dates,
        columns=spec.universe,
    )
    ascending = pd.DataFrame(
        np.tile(np.arange(9, dtype=float), (len(dates), 1)),
        index=dates,
        columns=spec.universe,
    )
    descending = 8.0 - ascending
    adapter = GuosenTrendIndexBacktester(None, None, spec)

    weights, diagnostics = adapter.build_signal_weights(
        {"a_fast": ascending, "a_slow": ascending, "b": descending},
        {
            "factor_a": (("a_fast", 1), ("a_slow", 1)),
            "factor_b": (("b", 1),),
        },
        close,
    )

    # The two A parameters form one factor portfolio before A and B are
    # averaged.  Therefore an A-only asset and a B-only asset have equal
    # weight; averaging all three variants directly would give A twice B.
    last = weights.iloc[-1]
    assert diagnostics.iloc[-1]["active_factors"] == 2
    assert np.isclose(last[spec.universe[-1]], last[spec.universe[0]])


def test_gross_projection_preserves_caps_ratios_and_exact_total():
    spec, _, _ = load_snapshot(SNAPSHOT)
    adapter = GuosenTrendIndexBacktester(None, None, spec)
    weights = pd.DataFrame(
        [[0.4, 0.2, 0.1, 0.1, 0.05, 0.05, 0.04, 0.03, 0.03]],
        index=[pd.Timestamp("2024-01-02")],
        columns=spec.universe,
    )

    projected = adapter.project_weights_to_gross(weights, 1.0).iloc[0]

    caps = pd.Series(spec.asset_caps)
    assert np.isclose(projected.sum(), 1.0)
    assert projected.le(caps + 1e-12).all()
    assert np.isclose(projected["RB"], projected["SC"])


def test_gross_projection_fails_closed_when_active_caps_are_insufficient():
    spec, _, _ = load_snapshot(SNAPSHOT)
    adapter = GuosenTrendIndexBacktester(None, None, spec)
    weights = pd.DataFrame(
        [[0.1 if symbol in {"IF", "IC"} else 0.0 for symbol in spec.universe]],
        index=[pd.Timestamp("2024-01-02")],
        columns=spec.universe,
    )

    with np.testing.assert_raises(ValueError):
        adapter.project_weights_to_gross(weights, 1.0)
