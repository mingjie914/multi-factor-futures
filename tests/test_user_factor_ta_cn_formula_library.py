from __future__ import annotations

import numpy as np
import pandas as pd

from core.registry import get
from factors.user.ta_cn_formula_library import FACTOR_SPECS


class FrameProvider:
    def __init__(self, frames, industry):
        self.frames = frames
        self.industry = industry

    def get(self, field, dates, universe):
        return self.frames.get(field, pd.DataFrame()).reindex(
            index=dates, columns=universe
        )

    def get_industry(self, dates, universe):
        return self.industry.reindex(index=dates, columns=universe)


def _sample():
    dates = pd.bdate_range("2019-01-01", periods=340)
    universe = pd.Index([f"F{i:02d}" for i in range(12)])
    rng = np.random.default_rng(37)
    returns = rng.normal(0.0001, 0.01, (len(dates), len(universe)))
    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(returns, axis=0)),
        index=dates, columns=universe,
    )
    open_price = close * np.exp(rng.normal(0.0, 0.002, close.shape))
    width = 0.005 + np.abs(returns)
    high = pd.DataFrame(
        np.maximum(open_price, close) * (1.0 + width),
        index=dates, columns=universe,
    )
    low = pd.DataFrame(
        np.minimum(open_price, close) * (1.0 - width),
        index=dates, columns=universe,
    )
    volume = pd.DataFrame(
        rng.lognormal(8.0, 0.3, close.shape), index=dates, columns=universe
    )
    typical = (open_price + high + low + close) / 4.0
    amount = typical * volume
    industry = pd.DataFrame(
        np.tile(["ferrous", "energy", "agri"] * 4, (len(dates), 1)),
        index=dates, columns=universe,
    )
    return dates, universe, FrameProvider({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "amount": amount,
    }, industry)


def test_all_292_formulas_are_registered_with_explicit_availability():
    assert len(FACTOR_SPECS) == 292
    assert len({spec.slug for spec in FACTOR_SPECS}) == 292
    assert sum(spec.available for spec in FACTOR_SPECS) == 284
    for spec in FACTOR_SPECS:
        factor = get("factor", spec.slug)()
        assert factor.name == spec.slug
        assert factor.frequency == "daily"
        assert factor.category == f"external_formula_{spec.library}"
        assert factor.dependencies()
        if not spec.available:
            assert factor.unavailable_reason


def test_representative_wq_and_gtja_formulas_compute_finite_values():
    dates, universe, provider = _sample()
    for name in ("wq101_alpha101", "wq101_alpha002", "gtja191_alpha001", "gtja191_alpha069"):
        result = get("factor", name)().compute(provider, dates, universe)
        assert result.index.equals(dates)
        assert result.columns.equals(universe)
        assert np.isfinite(result.to_numpy()).any(), name


def test_formula_inputs_do_not_consume_current_bar():
    dates, universe, provider = _sample()
    factor = get("factor", "wq101_alpha101")()
    baseline = factor.compute(provider, dates, universe)
    changed_frames = {name: frame.copy() for name, frame in provider.frames.items()}
    for frame in changed_frames.values():
        frame.iloc[-1] *= 100.0
    revised_provider = FrameProvider(changed_frames, provider.industry)
    revised = factor.compute(revised_provider, dates, universe)
    pd.testing.assert_series_equal(baseline.iloc[-1], revised.iloc[-1])


def test_stock_specific_unavailable_formula_fails_closed():
    dates, universe, provider = _sample()
    result = get("factor", "wq101_alpha056")().compute(
        provider, dates, universe
    )
    assert result.isna().all().all()
