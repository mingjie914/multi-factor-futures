from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import factors.numerics as numerics

from factors.numerics import (
    daily_breakout_statistics,
    daily_candle_path_statistics,
    daily_lagged_pair_statistics,
    daily_oi_statistics,
    daily_pair_statistics,
    daily_price_volume_statistics,
    daily_price_path_statistics,
    daily_return_statistics,
    daily_return_path_statistics,
    daily_smart_money_statistics,
    daily_tail_statistics,
    daily_unary_statistics,
    daily_volume_shock_statistics,
    histogram_window_l1_stability,
    rolling_linear_slope,
    rolling_split_sum_difference,
    variance_ratio,
)

from factors.library.intraday import (
    _daily_feature_frames,
    _daily_hlc_features,
    _daily_range_volume_ratio,
    _daily_volume_distribution_stability,
    _post_extreme_amount_persistence,
    IntradayDfTest20d,
    IntradayKlineShortestPathIlliq20d,
    IntradayMfdfaWidth20d,
    IntradayOiSurgeFollow20d,
    IntradayOiSurgeReversal20d,
    IntradayPricePeakCount20d,
    IntradayTermDtws20d,
    JumpIntensityRank20d,
    OpenCloseVolRank20d,
    PeakCountZscore20d,
)


def test_factor_kernel_mode_auto_selects_installed_native(monkeypatch):
    monkeypatch.delenv("MF_FACTOR_KERNEL_MODE", raising=False)
    monkeypatch.setattr(numerics.importlib.util, "find_spec", lambda name: object())
    assert numerics.factor_kernel_mode() == "native"
    monkeypatch.setattr(numerics.importlib.util, "find_spec", lambda name: None)
    assert numerics.factor_kernel_mode() == "reference"


def test_explicit_native_fails_closed_when_extension_is_missing(monkeypatch):
    monkeypatch.setenv("MF_FACTOR_KERNEL_MODE", "native")

    def missing_module():
        raise ImportError("missing")

    monkeypatch.setattr(numerics, "_load_native_module", missing_module)
    with pytest.raises(RuntimeError, match="not installed"):
        numerics.daily_unary_statistics(
            np.ones((2, 1), dtype=float), np.array([0, 2], dtype=np.int64)
        )


def test_shared_daily_frames_omit_only_fully_missing_dates():
    dates = pd.bdate_range("2025-01-02", periods=3)
    frames = _daily_feature_frames(
        {"metric": np.array([[1.0, np.nan], [np.nan, np.nan], [2.0, 3.0]])},
        dates,
        ["A", "B"],
    )

    assert frames["metric"].index.equals(dates[[0, 2]])
    assert np.isnan(frames["metric"].loc[dates[0], "B"])


def test_price_path_volatility_extremes_exclude_flat_five_bar_segments(monkeypatch):
    pytest.importorskip("_mf_factor_kernels")
    monkeypatch.setenv("MF_FACTOR_KERNEL_MODE", "shadow")
    close = np.arange(120, dtype=float) % 17 + 100.0
    close[:5] = 100.0
    matrix = close[:, None]
    offsets = np.array([0, len(close)], dtype=np.int64)

    result = daily_price_path_statistics(
        matrix, matrix, matrix, matrix, np.ones_like(matrix), offsets
    )
    segments5 = close.reshape(-1, 5)
    vol5_all = segments5.std(axis=1) / segments5.mean(axis=1)
    vol5_positive = vol5_all[vol5_all > 0]
    segments30 = close.reshape(-1, 30)
    vol30 = segments30.std(axis=1) / segments30.mean(axis=1)
    magnitude_threshold = np.percentile(vol5_positive, 95)
    extreme = vol30[vol30 > magnitude_threshold]
    expected_magnitude = (
        extreme.mean() / magnitude_threshold if len(extreme) else 0.0
    )
    trend_threshold = np.percentile(vol5_all, 95)

    assert result["vol_extreme_magnitude"][0, 0] == pytest.approx(expected_magnitude)
    assert result["vol_ratio_trend"][0, 0] == pytest.approx(
        np.mean(vol30 > trend_threshold)
    )


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


@pytest.mark.parametrize("mode", ["reference", "shadow"])
def test_daily_return_statistics_matches_adjacent_pandas_semantics(monkeypatch, mode):
    if mode == "shadow":
        pytest.importorskip("_mf_factor_kernels")
    monkeypatch.setenv("MF_FACTOR_KERNEL_MODE", mode)
    values = np.array([
        [100.0, 50.0], [101.0, np.nan], [np.nan, 51.0], [102.0, 52.0],
        [103.0, 52.0], [104.0, 51.0], [105.0, 53.0], [106.0, 54.0],
    ])
    offsets = np.array([0, 4, 8], dtype=np.int64)
    index = pd.DatetimeIndex(
        [pd.Timestamp("2026-01-02")] * 4 + [pd.Timestamp("2026-01-05")] * 4
    )
    returns = pd.DataFrame(values, index=index).pct_change(fill_method=None)
    grouped = returns.groupby(returns.index)

    actual = daily_return_statistics(values, offsets)

    for field, expected in {
        "count": grouped.count(),
        "sum": grouped.sum(),
        "mean": grouped.mean(),
        "std": grouped.std(ddof=0),
        "max": grouped.max(),
        "skew": grouped.skew(),
    }.items():
        np.testing.assert_allclose(
            actual[field], expected, atol=1e-14, rtol=1e-12, equal_nan=True
        )


def test_native_daily_unary_and_pair_statistics_match_pandas(monkeypatch):
    pytest.importorskip("_mf_factor_kernels")
    monkeypatch.setenv("MF_FACTOR_KERNEL_MODE", "shadow")
    left = np.array([
        [1.0, 2.0], [2.0, np.nan], [3.0, 4.0], [4.0, 8.0],
        [2.0, 7.0], [3.0, 6.0], [np.nan, 5.0], [8.0, 4.0],
    ])
    right = np.array([
        [8.0, 1.0], [6.0, 2.0], [4.0, np.nan], [2.0, 4.0],
        [1.0, 8.0], [3.0, 6.0], [5.0, 4.0], [7.0, 2.0],
    ])
    offsets = np.array([0, 4, 8], dtype=np.int64)

    unary = daily_unary_statistics(left, offsets)
    pair = daily_pair_statistics(left, right, offsets)
    lagged = daily_lagged_pair_statistics(left, right, offsets)
    tail = daily_tail_statistics(left, offsets, [1, 2])

    for day, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        for column in range(left.shape[1]):
            series = pd.Series(left[start:end, column]).dropna()
            assert unary["mean"][day, column] == pytest.approx(series.mean())
            assert unary["std"][day, column] == pytest.approx(series.std(ddof=0))
            assert unary["skew"][day, column] == pytest.approx(series.skew())
            frame = pd.DataFrame({
                "left": left[start:end, column],
                "right": right[start:end, column],
            }).dropna()
            assert pair["covariance"][day, column] == pytest.approx(
                np.cov(frame["left"], frame["right"], ddof=0)[0, 1]
            )
            assert pair["correlation"][day, column] == pytest.approx(
                frame["left"].corr(frame["right"])
            )
            lagged_left = frame["left"].iloc[:-1].reset_index(drop=True)
            lagged_right = frame["right"].iloc[1:].reset_index(drop=True)
            expected_lagged = (
                lagged_left.corr(lagged_right)
                if len(lagged_left) >= 2
                and lagged_left.std(ddof=0) > 0
                and lagged_right.std(ddof=0) > 0
                else np.nan
            )
            np.testing.assert_allclose(
                lagged["correlation"][day, column], expected_lagged, equal_nan=True
            )
            assert tail["last"][day, column] == series.iloc[-1]
            if len(series) >= 3:
                assert tail["means"][day, column, 1] == pytest.approx(
                    series.iloc[-3:-1].mean()
                )


def test_native_hlc_family_matches_established_daily_formulas(monkeypatch):
    pytest.importorskip("_mf_factor_kernels")
    monkeypatch.setenv("MF_FACTOR_KERNEL_MODE", "shadow")
    rng = np.random.default_rng(20260824)
    dates = pd.bdate_range("2026-01-05", periods=4)
    index = pd.DatetimeIndex([
        day + pd.Timedelta(minutes=minute)
        for day in dates for minute in range(35)
    ])
    close = pd.DataFrame(
        100.0 + rng.normal(size=(len(index), 2)), index=index, columns=["A", "B"]
    )
    high = close + rng.random(close.shape)
    low = close - rng.random(close.shape)
    high.iloc[5, 0] = np.nan
    low.iloc[42, 1] = np.nan
    panel = {"high": high, "low": low, "close": close}

    actual = _daily_hlc_features(panel)
    expected = {name: {} for name in actual}
    day = index.normalize()
    for dt in dates:
        for column in close.columns:
            frame = pd.concat(
                [high.loc[day == dt, column], low.loc[day == dt, column],
                 close.loc[day == dt, column]], axis=1
            ).dropna()
            frame.columns = ["high", "low", "close"]
            ratio = (
                (frame["high"] - frame["close"])
                / (frame["close"] - frame["low"]).replace(0, np.nan)
            ).dropna()
            if len(frame) >= 20 and len(ratio) >= 10:
                expected["range_asymmetry"].setdefault(dt, {})[column] = -ratio.mean()
            if len(frame) >= 30:
                width = frame["high"].max() - frame["low"].min()
                expected["range_position"].setdefault(dt, {})[column] = (
                    0.5 if width < 1e-12
                    else ((frame["close"] - frame["low"].min()) / width).mean()
                )
                expected["anchor_distance"].setdefault(dt, {})[column] = (
                    0.0 if width < 1e-12 else -np.minimum(
                        (frame["high"].max() - frame["close"]) / width,
                        (frame["close"] - frame["low"].min()) / width,
                    ).mean()
                )

    for name, values in expected.items():
        frame = pd.DataFrame(values).T.reindex(index=dates, columns=close.columns)
        np.testing.assert_allclose(actual[name], frame, atol=1e-13, rtol=1e-12)


def test_volume_distribution_stability_reuses_the_minute_panel_lifetime():
    index = pd.date_range("2025-01-02 09:00", periods=70, freq="min")
    panel = {"volume": pd.DataFrame({"A": np.arange(70.0) + 1.0}, index=index)}

    first = _daily_volume_distribution_stability(panel)
    second = _daily_volume_distribution_stability(panel)

    assert second is first


def test_native_daily_path_kernels_match_reference(monkeypatch):
    pytest.importorskip("_mf_factor_kernels")
    monkeypatch.setenv("MF_FACTOR_KERNEL_MODE", "shadow")
    rng = np.random.default_rng(20260823)
    dates = pd.bdate_range("2025-01-02", periods=7)
    index = pd.DatetimeIndex([
        day + pd.Timedelta(minutes=minute)
        for day in dates for minute in range(70)
    ])
    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, len(index)))) ,
        index=index,
        columns=["A"],
    )
    volume = pd.DataFrame(
        rng.lognormal(8.0, 0.7, len(index)), index=index, columns=["A"]
    )
    high, low = close + 0.5, close - 0.5
    open_px = close.shift(1).fillna(close)
    position = pd.DataFrame(
        1e5 + np.cumsum(rng.normal(0.0, 50.0, len(index))),
        index=index,
        columns=["A"],
    )
    panel = {
        "close": close, "high": high, "low": low,
        "volume": volume, "position": position,
    }
    import factors.library.intraday as intraday
    monkeypatch.setattr(intraday, "_get_minute_panel", lambda *args, **kwargs: panel)

    stability = _daily_volume_distribution_stability(panel)
    width = IntradayMfdfaWidth20d().compute(None, dates, ["A"])
    offsets = np.arange(0, len(index) + 1, 70, dtype=np.int64)
    breakout = daily_breakout_statistics(
        high.to_numpy(),
        low.to_numpy(),
        close.to_numpy(),
        offsets,
    )
    smart_money = daily_smart_money_statistics(
        close.to_numpy(), volume.to_numpy(), offsets
    )
    oi_features = daily_oi_statistics(
        high.to_numpy(), low.to_numpy(), close.to_numpy(), volume.to_numpy(),
        position.to_numpy(), offsets,
    )
    return_path = daily_return_path_statistics(close.to_numpy(), offsets)
    volume_shock = daily_volume_shock_statistics(
        close.to_numpy(), volume.to_numpy(),
        (close * volume).to_numpy(), offsets,
    )
    candle_path = daily_candle_path_statistics(
        open_px.to_numpy(), high.to_numpy(), low.to_numpy(), close.to_numpy(), offsets
    )
    price_volume = daily_price_volume_statistics(
        high.to_numpy(), low.to_numpy(), close.to_numpy(), volume.to_numpy(),
        (close * volume).to_numpy(), offsets,
    )
    price_path = daily_price_path_statistics(
        open_px.to_numpy(), high.to_numpy(), low.to_numpy(), close.to_numpy(),
        volume.to_numpy(), offsets,
    )

    assert np.isfinite(stability.to_numpy()).all()
    assert np.isfinite(width.iloc[-1, 0])
    assert np.isfinite(breakout["false_retrace"]).all()
    assert np.isfinite(smart_money["volatility_ratio"]).all()
    assert np.isfinite(oi_features["peak_count"]).all()
    assert np.isfinite(return_path["permutation_entropy"]).all()
    assert np.isfinite(volume_shock["large_order_impact"]).all()
    assert np.isfinite(candle_path["adx"]).all()
    assert np.isfinite(price_volume["obv_slope"]).all()
    assert np.isfinite(price_path["choppiness"]).all()


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
