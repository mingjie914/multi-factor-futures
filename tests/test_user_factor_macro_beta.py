from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.registry import get
from factors.user.macro_beta import FACTOR_SPECS


class MacroProvider:
    def __init__(self, close, macro):
        self.close = close
        self.macro = macro

    def get(self, field, dates, universe):
        if field != "close":
            return pd.DataFrame(index=dates, columns=universe)
        return self.close.reindex(index=dates, columns=universe)

    def get_macro(self, fields, start=None, end=None):
        result = self.macro.reindex(columns=fields)
        if start is not None:
            result = result.loc[pd.Timestamp(start):]
        if end is not None:
            result = result.loc[:pd.Timestamp(end)]
        return result


@pytest.fixture(scope="module")
def sample_data():
    dates = pd.bdate_range("2014-01-01", "2021-12-31")
    universe = pd.Index(["RB", "HC", "CU", "AL", "A", "M", "IF", "IC"])
    rng = np.random.default_rng(17)
    returns = rng.normal(0.0001, 0.008, (len(dates), len(universe)))
    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(returns, axis=0)),
        index=dates,
        columns=universe,
    )

    months = pd.date_range("2010-01-31", "2021-12-31", freq="ME")
    phase = np.arange(len(months), dtype=float)
    macro_fields = sorted({field for spec in FACTOR_SPECS for field in spec.fields})
    macro = pd.DataFrame(index=months)
    for offset, field in enumerate(macro_fields, 1):
        values = (
            50.0
            + 0.02 * offset * phase
            + np.sin(phase / (2.0 + offset / 5.0))
            + rng.normal(0.0, 0.15, len(months))
        )
        if field.startswith("repo_") or field.startswith("shibor_"):
            values = 2.0 + values / 50.0 + 0.05 * offset
        macro[field] = values
    macro["taiwan_electronics"] = 100.0 * np.exp(
        np.cumsum(rng.normal(0.006, 0.03, len(months)))
    )
    macro["leading_index"] = 100.0 * np.exp(
        np.cumsum(rng.normal(0.001, 0.008, len(months)))
    )
    return dates, universe, close, macro


def test_specs_are_unique_registered_and_declare_metadata():
    assert len(FACTOR_SPECS) == 12
    assert len({spec.slug for spec in FACTOR_SPECS}) == 12
    for spec in FACTOR_SPECS:
        factor = get("factor", spec.slug)()
        assert factor.name == spec.slug
        assert factor.category == "macro_sensitivity"
        assert factor.frequency == "daily"
        assert factor.dependencies() == ["close"]
        assert factor.description


@pytest.mark.parametrize("spec", FACTOR_SPECS, ids=lambda spec: spec.slug)
def test_alignment_warmup_and_finite_values(spec, sample_data):
    dates, universe, close, macro = sample_data
    result = get("factor", spec.slug)().compute(
        MacroProvider(close, macro), dates, universe
    )

    assert result.index.equals(dates)
    assert result.columns.equals(universe)
    assert result.iloc[:300].isna().all().all()
    assert np.isfinite(result.iloc[800:].to_numpy()).any()
    assert not np.isinf(result.to_numpy()).any()


def test_observation_waits_one_complete_month_before_use(sample_data):
    dates, universe, close, macro = sample_data
    factor = get("factor", "macro_beta_pmi_new_orders_36m")()
    baseline = factor.compute(MacroProvider(close, macro), dates, universe)
    changed_macro = macro.copy()
    changed_macro.loc[pd.Timestamp("2018-01-31"), "pmi_new_orders"] += 20.0
    revised = factor.compute(
        MacroProvider(close, changed_macro), dates, universe
    )

    pd.testing.assert_frame_equal(
        baseline.loc[:"2018-02-28"], revised.loc[:"2018-02-28"]
    )
    difference = (baseline.loc["2018-03-01":] - revised.loc["2018-03-01":]).abs()
    assert difference.to_numpy().max() > 0.0


def test_current_close_bar_is_not_consumed(sample_data):
    dates, universe, close, macro = sample_data
    factor = get("factor", "macro_beta_repo_curve_36m")()
    baseline = factor.compute(MacroProvider(close, macro), dates, universe)
    changed_close = close.copy()
    changed_close.iloc[-1] *= 100.0
    revised = factor.compute(
        MacroProvider(changed_close, macro), dates, universe
    )
    pd.testing.assert_series_equal(baseline.iloc[-1], revised.iloc[-1])


def test_missing_macro_input_fails_closed(sample_data):
    dates, universe, close, macro = sample_data
    result = get("factor", FACTOR_SPECS[0].slug)().compute(
        MacroProvider(close, pd.DataFrame()), dates, universe
    )
    assert result.isna().all().all()

