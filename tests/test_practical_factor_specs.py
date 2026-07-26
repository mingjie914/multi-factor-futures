from __future__ import annotations

import numpy as np
import pandas as pd

from factors.practical_bases import PRACTICAL_BASES, compute_practical_base
from factors.specs.practical import SPECS


def _market_data(rows: int = 360, columns: int = 8):
    rng = np.random.default_rng(20260726)
    dates = pd.date_range("2023-01-02", periods=rows, freq="B")
    tickers = [f"F{i}" for i in range(columns)]
    innovations = rng.normal(0.0003, 0.012, size=(rows, columns))
    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(innovations, axis=0)),
        index=dates,
        columns=tickers,
    )
    overnight = rng.normal(0.0, 0.003, size=(rows, columns))
    open_ = close.shift(1).fillna(close.iloc[0]) * (1.0 + overnight)
    spread = pd.DataFrame(
        rng.uniform(0.003, 0.025, size=(rows, columns)),
        index=dates,
        columns=tickers,
    )
    high = pd.DataFrame(
        np.maximum(open_.to_numpy(), close.to_numpy()),
        index=dates,
        columns=tickers,
    ) * (1.0 + spread)
    low = pd.DataFrame(
        np.minimum(open_.to_numpy(), close.to_numpy()),
        index=dates,
        columns=tickers,
    ) * (1.0 - spread)
    volume = pd.DataFrame(
        rng.lognormal(10.0, 0.4, size=(rows, columns)),
        index=dates,
        columns=tickers,
    )
    oi = pd.DataFrame(
        rng.lognormal(11.0, 0.25, size=(rows, columns)),
        index=dates,
        columns=tickers,
    )
    return {
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "oi": oi,
        "_return_1d": close.pct_change(fill_method=None),
    }


def test_practical_specs_are_unique_and_complete():
    assert len(PRACTICAL_BASES) == 42
    assert len(SPECS) == 42 * 6 * 6
    assert len({spec["slug"] for spec in SPECS}) == len(SPECS)
    assert all(spec["frequency"] == "daily" for spec in SPECS)
    assert all(spec["research_tier"] == "candidate" for spec in SPECS)
    assert all(spec["source"] for spec in SPECS)


def test_every_practical_base_is_finite_and_point_in_time():
    market = _market_data()
    perturbed = {name: frame.copy() for name, frame in market.items()}
    for name, frame in perturbed.items():
        if name != "_return_1d":
            frame.iloc[-5:] = frame.iloc[-5:] * 1.7
    perturbed["_return_1d"] = perturbed["close"].pct_change(fill_method=None)

    for base in sorted(PRACTICAL_BASES):
        actual = compute_practical_base(base, {"window": 20}, market)
        changed = compute_practical_base(base, {"window": 20}, perturbed)
        assert actual.shape == market["close"].shape, base
        assert np.isfinite(actual.iloc[-80:].to_numpy()).any(), base
        pd.testing.assert_frame_equal(
            actual.iloc[:-5], changed.iloc[:-5], check_exact=True
        )


def test_open_interest_specs_declare_open_interest_dependency():
    oi_specs = [spec for spec in SPECS if spec["base"].startswith("oi_")]
    assert oi_specs
    assert all("oi" in spec["dependencies"] for spec in oi_specs)
