from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors.numerics import (
    histogram_window_l1_stability,
    rolling_linear_slope,
    rolling_split_sum_difference,
    variance_ratio,
)

from factors.library.intraday import (
    _daily_range_volume_ratio,
    _daily_volume_distribution_stability,
    _post_extreme_amount_persistence,
    _segmented_ohlc_relative_volatility,
    IntradayDfTest20d,
    IntradayKlineShortestPathIlliq20d,
    IntradayOiSurgeFollow20d,
    IntradayOiSurgeReversal20d,
    IntradayPricePeakCount20d,
    IntradayTermDtws20d,
    JumpIntensityRank20d,
    OpenCloseVolRank20d,
    PeakCountZscore20d,
)


@pytest.mark.parametrize("window,positive_only", [(5, False), (5, True), (30, False)])
def test_segmented_ohlc_volatility_matches_established_loop(window, positive_only):
    rng = np.random.default_rng(20260822)
    close = rng.lognormal(mean=4.0, sigma=0.1, size=67)
    high = close + rng.random(67)
    low = close - rng.random(67)
    close[:window] = 100.0
    high[:window] = 100.0
    low[:window] = 100.0
    expected = []
    for start in range(0, len(close) - window + 1, window):
        segment = np.concatenate([
            close[start:start + window],
            high[start:start + window],
            low[start:start + window],
        ])
        mean = segment.mean()
        value = segment.std(ddof=0) / mean if mean > 0 else np.nan
        if mean > 0 and (not positive_only or value > 0):
            expected.append(value)

    actual = _segmented_ohlc_relative_volatility(
        close, high, low, window, positive_only=positive_only
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_post_extreme_amount_persistence_matches_established_loop():
    rng = np.random.default_rng(731)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=(67, 3)), axis=0))
    amount = rng.lognormal(8.0, 0.6, size=(67, 3))
    close[5:8, 1] = np.nan
    amount[30:32, 2] = np.nan
    expected = []
    for column in range(close.shape[1]):
        observed = ~(np.isnan(close[:, column]) | np.isnan(amount[:, column]))
        prices, amounts = close[observed, column], amount[observed, column]
        returns = pd.Series(prices).pct_change(fill_method=None)
        high, low = np.percentile(returns.dropna(), (90, 10))
        extreme = (returns > high) | (returns < low)
        future = [
            amounts[row + 1:row + 6].mean()
            for row, is_extreme in enumerate(extreme)
            if is_extreme and row + 5 < len(amounts)
        ]
        expected.append(np.mean(future) / amounts.mean())

    actual = _post_extreme_amount_persistence(close, amount)

    np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0.0)


def test_daily_range_volume_ratio_uses_only_prior_twenty_days():
    days = pd.bdate_range("2024-01-02", periods=24)
    index = pd.DatetimeIndex([
        day + pd.Timedelta(minutes=minute)
        for day in days for minute in range(10)
    ])
    day_number = np.repeat(np.arange(len(days), dtype=float), 10)
    intraday_step = np.tile(np.arange(10, dtype=float), len(days))
    close = pd.DataFrame({"A": 100.0 + day_number + intraday_step}, index=index)
    panel = {
        "close": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "volume": pd.DataFrame({"A": intraday_step + 1.0}, index=index),
    }

    result = _daily_range_volume_ratio(panel)

    assert result.iloc[:20].isna().all().all()
    normalized = index.normalize()
    for row in range(20, len(days)):
        history = normalized.isin(days[row - 20:row])
        midpoint = (
            panel["high"].loc[history].max().iloc[0]
            + panel["low"].loc[history].min().iloc[0]
        ) / 2.0
        current = normalized == days[row]
        expected = panel["volume"].loc[current, "A"].where(
            close.loc[current, "A"] > midpoint, 0.0
        ).sum() / panel["volume"].loc[current, "A"].sum()
        assert result.iloc[row, 0] == expected


def test_kline_shortest_path_excludes_negative_illiq_ratio(monkeypatch):
    dates = pd.bdate_range("2025-01-02", periods=7)
    index = pd.DatetimeIndex([
        day + pd.Timedelta(minutes=minute)
        for day in dates for minute in range(30)
    ])
    frame = pd.DataFrame({"A": 100.0}, index=index)
    amount = frame.copy()
    amount.iloc[::30, 0] = -100.0
    panel = {
        "high": frame + 2.0,
        "low": frame - 1.0,
        "open": frame,
        "close": frame + 1.0,
        "amount": amount,
    }
    import factors.library.intraday as intraday
    monkeypatch.setattr(intraday, "_get_minute_panel", lambda *args, **kwargs: panel)

    result = IntradayKlineShortestPathIlliq20d().compute(None, dates, ["A"])

    assert result.iloc[-1, 0] == pytest.approx(0.05)


def test_rank_variants_apply_cross_sectional_percentiles():
    base = pd.DataFrame([[3.0, 1.0, 2.0]], columns=["A", "B", "C"])
    expected = pd.DataFrame([[1.0, 1.0 / 3.0, 2.0 / 3.0]], columns=base.columns)

    for factor in (JumpIntensityRank20d(), OpenCloseVolRank20d()):
        pd.testing.assert_frame_equal(factor._transform(base), expected)


def test_zscore_variant_is_cross_sectional_and_handles_constant_rows():
    base = pd.DataFrame(
        [[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]], columns=["A", "B", "C"]
    )

    result = PeakCountZscore20d()._transform(base)

    assert np.isclose(result.iloc[0].mean(), 0.0)
    assert np.isclose(result.iloc[0].std(ddof=0), 1.0)
    assert result.iloc[1].isna().all()


def test_df_manual_fallback_produces_a_finite_statistic(monkeypatch):
    import statsmodels.tsa.stattools as stattools

    monkeypatch.setattr(
        stattools,
        "adfuller",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("fallback")),
    )
    series = np.cumsum(np.sin(np.arange(80, dtype=float)))

    result = IntradayDfTest20d._df_tstat(series)

    assert np.isfinite(result)
    assert result >= 0.0


def test_oi_surge_relations_do_not_use_closes_on_or_after_signal_date(monkeypatch):
    dates = pd.bdate_range("2025-01-02", periods=40)
    index = dates + pd.Timedelta(hours=15)
    position = pd.DataFrame(
        {"A": 1000.0 + np.arange(len(index)) ** 2}, index=index
    )
    baseline_close = pd.DataFrame(
        {"A": 100.0 + np.arange(len(index)) * 0.2}, index=index
    )
    changed_close = baseline_close.copy()
    changed_close.loc[index[25]:, "A"] *= 3.0
    panel = {"position": position, "close": baseline_close}

    import factors.library.intraday as intraday

    monkeypatch.setattr(intraday, "_get_minute_panel", lambda *args, **kwargs: panel)
    for factor in (IntradayOiSurgeReversal20d(), IntradayOiSurgeFollow20d()):
        baseline = factor.compute(None, dates, ["A"])
        panel["close"] = changed_close
        changed = factor.compute(None, dates, ["A"])
        assert changed.loc[dates[25], "A"] == baseline.loc[dates[25], "A"]
        panel["close"] = baseline_close


def test_price_peak_count_keeps_daily_count_and_lag_semantics(monkeypatch):
    dates = pd.bdate_range("2025-01-02", periods=4)
    index = pd.DatetimeIndex([
        date + pd.Timedelta(minutes=minute)
        for date in dates
        for minute in range(12)
    ])
    low = pd.DataFrame({"A": 100.0}, index=index)
    high = pd.DataFrame({"A": 101.0}, index=index)
    for date in dates:
        high.loc[date + pd.Timedelta(minutes=5), "A"] = 103.0
    close = pd.DataFrame({"A": 100.0}, index=index)

    import factors.library.intraday as intraday

    monkeypatch.setattr(
        intraday,
        "_get_minute_panel",
        lambda *args, **kwargs: {"high": high, "low": low, "close": close},
    )
    result = IntradayPricePeakCount20d().compute(None, dates, ["A"])

    assert result.iloc[:3, 0].isna().all()
    assert result.iloc[3, 0] == 3.0


def test_term_dtws_drops_the_leading_difference_nan(monkeypatch):
    dates = pd.bdate_range("2025-01-02", periods=7)
    index = pd.DatetimeIndex([
        day + pd.Timedelta(minutes=minute)
        for day in dates for minute in range(30)
    ])
    near = pd.DataFrame({"A": 100.0}, index=index)
    far = near + np.tile(np.arange(30, dtype=float), len(dates))[:, None]

    import factors.library.intraday as intraday

    monkeypatch.setattr(
        intraday,
        "_get_term_structure_panel",
        lambda *args, **kwargs: {"near_close": near, "far_close": far},
    )
    result = IntradayTermDtws20d().compute(None, dates, ["A"])

    assert np.isfinite(result.iloc[-1, 0])


def test_histogram_stability_kernel_matches_window_rest_reference():
    values = np.random.default_rng(42).normal(size=227)
    expected = []
    series = pd.Series(values)
    for start in range(0, len(series), 5):
        window = series.iloc[start:start + 5]
        rest = pd.concat([series.iloc[:start], series.iloc[start + 5:]])
        if len(window) < 3 or len(rest) < 10:
            continue
        inside, _ = np.histogram(window, bins=10, range=(-4, 4))
        outside, _ = np.histogram(rest, bins=10, range=(-4, 4))
        expected.append(float(np.abs(
            inside / inside.sum() - outside / outside.sum()
        ).sum()))

    assert histogram_window_l1_stability(values) == pytest.approx(
        np.std(expected, ddof=0), abs=1e-15
    )


def test_volume_distribution_stability_reuses_the_minute_panel_lifetime():
    index = pd.date_range("2025-01-02 09:00", periods=70, freq="min")
    panel = {"volume": pd.DataFrame({"A": np.arange(70.0) + 1.0}, index=index)}

    first = _daily_volume_distribution_stability(panel)
    second = _daily_volume_distribution_stability(panel)

    assert second is first


def test_rolling_split_sum_difference_matches_window_reference():
    rng = np.random.default_rng(17)
    returns = rng.normal(size=(47, 3))
    scores = rng.normal(size=(47, 3))
    returns[4:10, 1] = np.nan
    scores[25, 2] = np.nan
    expected = np.full_like(returns, np.nan)
    for row in range(20, len(returns)):
        for column in range(returns.shape[1]):
            observed_returns = returns[row - 20:row, column]
            observed_scores = scores[row - 20:row, column]
            valid = ~np.isnan(observed_returns)
            if valid.sum() < 15:
                continue
            observed_returns = observed_returns[valid]
            observed_scores = observed_scores[valid]
            high = observed_scores > np.median(observed_scores)
            expected[row, column] = (
                observed_returns[high].sum() - observed_returns[~high].sum()
            )

    np.testing.assert_allclose(
        rolling_split_sum_difference(returns, scores), expected, equal_nan=True
    )


def test_rolling_linear_slope_matches_established_polyfit():
    values = np.random.default_rng(20260822).normal(size=(47, 3))
    values[:4, 1] = np.nan
    values[35, 1] = np.inf
    values[25, 2] = np.nan
    frame = pd.DataFrame(values)
    expected = frame.rolling(20, min_periods=8).apply(
        lambda x: np.polyfit(np.arange(len(x)), x.values, 1)[0],
        raw=False,
    )

    actual = rolling_linear_slope(values, window=20, min_periods=8)

    np.testing.assert_allclose(actual, expected.to_numpy(), atol=3e-15, rtol=0.0)

    large_offset = 1e12 + np.arange(47, dtype=float)[:, None] * 0.25
    stable = rolling_linear_slope(large_offset, window=20, min_periods=8)
    assert stable[-1, 0] == pytest.approx(0.25, abs=1e-12)


@pytest.mark.parametrize("horizon", [5, 30])
def test_variance_ratio_matches_established_correlation_loop(horizon):
    values = np.random.default_rng(731).normal(size=240)
    mean = values.mean()
    variance = np.sum((values[1:] - mean) ** 2) / (len(values) - 1)
    expected = 1.0
    for lag in range(1, horizon):
        current, previous = values[lag:], values[:-lag]
        correlation = (
            np.corrcoef(current, previous)[0, 1]
            if current.std() > 1e-12 and previous.std() > 1e-12
            else 0.0
        )
        expected += 2.0 * (1.0 - lag / horizon) * correlation

    assert variance > 1e-12
    assert variance_ratio(values, horizon) == pytest.approx(expected - 1.0, abs=1e-15)
