from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from factors.numerics import histogram_window_l1_stability

from factors.library.intraday import (
    IntradayDfTest20d,
    IntradayOiSurgeFollow20d,
    IntradayOiSurgeReversal20d,
    IntradayPricePeakCount20d,
    IntradayTermDtws20d,
    JumpIntensityRank20d,
    OpenCloseVolRank20d,
    PeakCountZscore20d,
)


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
