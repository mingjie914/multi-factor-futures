from types import SimpleNamespace

import pandas as pd
import pytest

from core.config import load_config
from core.date_policy import (
    apply_research_end,
    factor_validation_window,
    forward_observation_start,
    require_research_end,
    resolve_observation_end,
)


def test_default_factor_validation_window_is_126_is_plus_42_oos_and_warmup():
    config = load_config("config/default.yaml")
    calendar = pd.bdate_range("2025-01-01", "2026-05-15")
    manager = SimpleNamespace(
        get_calendar=lambda start, end: calendar[
            (calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))
        ]
    )

    window = factor_validation_window(config, manager)

    assert window.is_bars == 126
    assert window.oos_bars == 42
    assert window.warmup_calendar_days == 90
    assert window.oos_end == pd.Timestamp("2026-05-15")
    assert window.oos_start == calendar[-42]
    assert window.is_start == calendar[-168]
    assert window.is_end == calendar[-43]
    assert window.factor_start == window.is_start - pd.Timedelta(days=90)


def test_research_end_defaults_to_one_global_cutoff_and_rejects_later_dates():
    config = load_config("config/default.yaml")

    assert require_research_end(config) == pd.Timestamp("2026-05-15")
    assert forward_observation_start(config) == pd.Timestamp("2026-05-16")
    assert require_research_end(config, "2026-05-14") == pd.Timestamp("2026-05-14")
    with pytest.raises(ValueError, match="exceeds the inclusive framework cutoff"):
        require_research_end(config, "2026-05-18")


def test_research_and_observation_ends_resolve_independently():
    config = load_config("config/default.yaml")
    source = SimpleNamespace(
        fetch_latest_trade_date=lambda: pd.Timestamp("2026-08-20")
    )
    manager = SimpleNamespace(source=source)

    assert apply_research_end(config) == pd.Timestamp("2026-05-15")
    config.date_range.end = "latest_available"
    assert resolve_observation_end(config, manager) == pd.Timestamp("2026-08-20")
    assert config.date_range.end == "2026-08-20"


def test_observation_end_cannot_run_past_source_latest_date():
    config = load_config("config/default.yaml")
    config.date_range.end = "2026-08-21"
    manager = SimpleNamespace(source=SimpleNamespace(
        fetch_latest_trade_date=lambda: pd.Timestamp("2026-08-20")
    ))

    with pytest.raises(ValueError, match="exceeds source latest complete trade date"):
        resolve_observation_end(config, manager)
