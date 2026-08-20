from __future__ import annotations

import numpy as np
import pandas as pd

from factors.library.intraday import (
    IntradayDfTest20d,
    IntradayOiSurgeFollow20d,
    IntradayOiSurgeReversal20d,
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
