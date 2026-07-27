from __future__ import annotations

import pandas as pd

from core.config import UniverseSelectionConfig, load_config
from core.sectors import sector_for
from data.universe_selection import LaggedLiquidityUniverseSelector


class PanelProvider:
    def __init__(self, panels):
        self.panels = panels

    def get(self, field, dates, universe):
        return self.panels[field].reindex(index=dates, columns=universe)


def _config(**overrides):
    values = {
        "enabled": True,
        "lookback": 20,
        "min_listing_days": 20,
        "min_data_coverage": 1.0,
        "target_count": 2,
        "min_count": 2,
        "max_count": 2,
        "exit_buffer": 1,
        "sector_minimums": {},
        "sector_maximums": {"ferrous": 4, "other": 4},
    }
    values.update(overrides)
    return UniverseSelectionConfig(**values)


def _daily_provider(dates, universe, liquidity=None, close=None):
    liquidity = liquidity or {
        name: float(len(universe) - position)
        for position, name in enumerate(universe)
    }
    amount = pd.DataFrame(
        {name: liquidity[name] for name in universe}, index=dates
    )
    oi = amount * 10.0
    if close is None:
        close = pd.DataFrame(100.0, index=dates, columns=universe)
    return PanelProvider({"amount": amount, "oi": oi, "close": close})


def test_same_day_liquidity_cannot_change_same_day_membership():
    dates = pd.bdate_range("2023-01-02", periods=90)
    universe = pd.Index(["RB", "HC", "I", "J"])
    base = _daily_provider(dates, universe)
    selector = LaggedLiquidityUniverseSelector(_config())
    base_schedule = selector.build_schedule(base, dates, universe)
    decision = [day for day, names in base_schedule.items() if len(names) == 2][0]

    shocked_panels = {name: frame.copy() for name, frame in base.panels.items()}
    shocked_panels["amount"].loc[decision, "J"] = 1e12
    shocked_panels["oi"].loc[decision, "J"] = 1e12
    shocked_schedule = selector.build_schedule(
        PanelProvider(shocked_panels), dates, universe
    )

    assert base_schedule[decision].tolist() == shocked_schedule[decision].tolist()


def test_sector_floors_caps_and_target_count_are_enforced():
    dates = pd.bdate_range("2023-01-02", periods=50)
    universe = pd.Index([
        "RB", "HC", "I",
        "CU", "AL", "SN",
        "AU", "AG",
        "T", "TL",
    ])
    cfg = _config(
        lookback=5,
        min_listing_days=5,
        target_count=8,
        min_count=8,
        max_count=8,
        sector_minimums={
            "ferrous": 2,
            "nonferrous": 2,
            "precious": 2,
            "bond": 2,
        },
        sector_maximums={
            "ferrous": 2,
            "nonferrous": 2,
            "precious": 2,
            "bond": 2,
        },
    )
    schedule = LaggedLiquidityUniverseSelector(cfg).build_schedule(
        _daily_provider(dates, universe), dates, universe
    )
    selected = [names for names in schedule.values() if len(names)][-1]
    counts = pd.Series([sector_for(name) for name in selected]).value_counts()

    assert len(selected) == 8
    assert counts.to_dict() == {
        "ferrous": 2,
        "nonferrous": 2,
        "precious": 2,
        "bond": 2,
    }


def test_listing_and_coverage_filters_apply_before_selection():
    dates = pd.bdate_range("2023-01-02", periods=90)
    universe = pd.Index(["RB", "HC", "I"])
    close = pd.DataFrame(100.0, index=dates, columns=universe)
    close.loc[dates[:45], "RB"] = float("nan")
    cfg = _config(target_count=3, min_count=2, max_count=3)
    mask = LaggedLiquidityUniverseSelector(cfg).build_eligibility_mask(
        _daily_provider(dates, universe, close=close), dates, universe
    )

    assert not mask.loc[dates[:45], "RB"].any()
    assert mask.loc[dates[-20:], "RB"].any()
    assert not mask.loc[dates[:20]].any(axis=None)


def test_exit_buffer_retains_an_incumbent_at_the_boundary():
    cfg = _config(
        sector_maximums={"other": 3},
        target_count=2,
        min_count=2,
        max_count=2,
    )
    selector = LaggedLiquidityUniverseSelector(cfg)
    selected = selector._select_one_date(
        pd.Series({"C": 1.0, "A": 0.8, "B": 0.7}),
        pd.Index(["A", "B"]),
    )

    assert selected.tolist() == ["A", "B"]


def test_intraday_bars_share_the_monthly_daily_eligibility():
    days = pd.bdate_range("2023-01-02", periods=50)
    dates = pd.DatetimeIndex([
        timestamp
        for day in days
        for timestamp in (
            day + pd.Timedelta(hours=9),
            day + pd.Timedelta(hours=9, minutes=5),
        )
    ])
    universe = pd.Index(["RB", "HC", "I"])
    provider = _daily_provider(dates, universe)
    mask = LaggedLiquidityUniverseSelector(
        _config(lookback=5, min_listing_days=5, target_count=2)
    ).build_eligibility_mask(provider, dates, universe)

    for _, group in mask.groupby(mask.index.normalize()):
        assert group.nunique(axis=0).max() <= 1


def test_night_session_bars_share_the_following_trading_day():
    dates = pd.DatetimeIndex([
        "2024-03-01 21:00:00",  # Friday night, Monday trading session.
        "2024-03-02 00:30:00",
        "2024-03-04 09:00:00",
        "2024-03-04 14:59:00",
    ])
    universe = pd.Index(["RB", "HC"])
    panel = pd.DataFrame(1.0, index=dates, columns=universe)
    selector = LaggedLiquidityUniverseSelector(
        _config(lookback=2, min_listing_days=2)
    )

    daily = selector._daily_panel(panel, "sum")

    assert daily.index.tolist() == [pd.Timestamp("2024-03-04")]
    assert daily.loc[pd.Timestamp("2024-03-04"), "RB"] == 4.0


def test_default_config_enables_distinct_research_universe_selection():
    config = load_config("config/default.yaml")

    assert config.universe_selection.enabled is True
    assert config.universe_selection.target_count == 32
    assert config.asset_selection.enabled is False
    assert [step.type for step in config.processing] == [
        "winsorize", "neutralize", "standardize"
    ]
