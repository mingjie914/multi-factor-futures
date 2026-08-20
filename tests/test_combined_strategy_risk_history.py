from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from strategies.combined import (
    FACTORS,
    CombinedStrategy,
)
from optimization.portfolio_construction import PortfolioConstraints


def test_combined_strategy_uses_promoted_10f_only():
    assert FACTORS == {
        "intraday_price_peak_count_20d": 1,
        "intraday_realised_skewness_20d": 1,
        "intraday_drip_stone_20d": -1,
        "intraday_peak_ridge_ratio_20d": -1,
        "intraday_torrent_down_20d": -1,
        "intraday_lowest_time_20d": 1,
        "intraday_term_slope_20d": 1,
        "intraday_open_close_volume_ratio_20d": -1,
        "intraday_turnover_velocity_20d": 1,
        "intraday_price_delay_20d": -1,
    }


def test_production_scores_fail_closed_when_one_factor_is_unavailable():
    calendar = pd.bdate_range("2026-01-01", periods=70)
    universe = ["A", "B", "C"]

    class DataManager:
        def get_calendar(self, start, end):
            return calendar

        def get(self, field, dates, symbols):
            return pd.DataFrame(100.0, index=dates, columns=symbols)

    class Engine:
        def compute_factors(self, names, dates, symbols, parallel):
            values = {
                name: pd.DataFrame(1.0, index=calendar, columns=symbols)
                for name in names
            }
            values[names[0]].loc[:, :] = np.nan
            return values

    strategy = CombinedStrategy.__new__(CombinedStrategy)
    strategy.data_manager = DataManager()
    strategy.engine = Engine()
    strategy._universe = universe

    with pytest.raises(RuntimeError, match="production factor computation failed"):
        strategy.factor_scores("2026-01-01", "2026-04-30")


def test_production_scores_fail_closed_on_a_post_warmup_factor_gap():
    calendar = pd.bdate_range("2026-01-01", periods=8)
    universe = ["A", "B", "C"]

    class DataManager:
        def get_calendar(self, start, end):
            return calendar

    class Engine:
        def compute_factors(self, names, dates, symbols, parallel):
            values = {
                name: pd.DataFrame(1.0, index=calendar, columns=symbols)
                for name in names
            }
            values[names[0]].loc[calendar[4], :] = np.nan
            return values

    strategy = CombinedStrategy.__new__(CombinedStrategy)
    strategy.data_manager = DataManager()
    strategy.engine = Engine()
    strategy._universe = universe
    strategy.portfolio_cfg = SimpleNamespace(factor_weight_method="equal")

    with pytest.raises(RuntimeError, match="became unavailable"):
        strategy.factor_scores("2026-01-01", "2026-01-31")


def test_signal_reuses_one_return_history_without_changing_pool_weights():
    symbols = ["A", "M", "RB", "HC", "CU", "AL", "AU", "AG"]
    dates = pd.bdate_range("2026-01-01", periods=30)
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(len(dates), len(symbols))),
        index=dates,
        columns=symbols,
    )
    calls = []

    strategy = CombinedStrategy.__new__(CombinedStrategy)
    strategy.top_n = 2
    strategy._universe = symbols
    strategy.constraints = PortfolioConstraints(
        top_n_per_side=2,
        sector_count_cap=0,
        asset_min_fraction=0.0,
        asset_max_fraction=1.0,
    )
    strategy.factor_scores = lambda start, end: pd.DataFrame(
        [np.arange(len(symbols), dtype=float)],
        index=[pd.Timestamp(end)],
        columns=symbols,
    )

    def recent_history(date, requested):
        calls.append(tuple(requested))
        return returns.reindex(columns=requested)

    strategy._recent_returns = recent_history
    signal = strategy.signal("2026-02-20")

    assert not signal.empty
    assert calls == [tuple(symbols)]

    pool = ["A", "M"]
    reused = strategy._pool_weights(pool, pd.Timestamp("2026-02-20"), returns)
    direct = strategy._pool_weights(pool, pd.Timestamp("2026-02-20"))
    pd.testing.assert_series_equal(reused, direct)


def test_signal_rejects_a_stale_previous_trading_day():
    strategy = CombinedStrategy.__new__(CombinedStrategy)
    strategy.factor_scores = lambda start, end: pd.DataFrame(
        {"A": [1.0]}, index=[pd.Timestamp("2026-02-20")]
    )

    with pytest.raises(RuntimeError, match="不是可生成收盘信号的交易日"):
        strategy.signal("2026-02-21")


def test_formal_risk_window_keeps_the_same_first_return_as_full_history():
    dates = pd.bdate_range("2025-01-02", periods=140)
    rng = np.random.default_rng(11)
    returns = pd.DataFrame(
        rng.normal(0.0, 0.01, (len(dates), 2)),
        index=dates,
        columns=["A", "B"],
    )
    close = 100.0 * (1.0 + returns).cumprod()

    class DataManager:
        def get_calendar(self, start, end):
            return dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]

        def get(self, field, requested_dates, symbols):
            assert field == "close"
            return close.reindex(index=requested_dates, columns=symbols)

        def prepare_close_data(self, prices):
            return prices.pct_change(fill_method=None), prices.notna()

    strategy = CombinedStrategy.__new__(CombinedStrategy)
    strategy.data_manager = DataManager()
    strategy.portfolio_cfg = SimpleNamespace(risk_lookback_calendar_days=90)
    decision = dates[-1]

    actual = strategy._recent_returns(decision, ["A", "B"])
    start = decision - pd.Timedelta(days=90)
    expected = close.pct_change(fill_method=None).loc[
        lambda frame: (frame.index >= start) & (frame.index < decision),
        ["A", "B"],
    ]

    pd.testing.assert_frame_equal(actual, expected)
    assert actual.iloc[0].notna().all()
