"""Central, fail-closed research-cutoff and observation-end policy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


LATEST_AVAILABLE = "latest_available"


@dataclass(frozen=True)
class FactorValidationWindow:
    """One default factor-validation split resolved from the exchange calendar."""

    factor_start: pd.Timestamp
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    is_bars: int
    oos_bars: int
    warmup_calendar_days: int


def _normalised_date(value: Any, *, label: str) -> pd.Timestamp:
    text = str(value or "").strip()
    if not text or text == LATEST_AVAILABLE:
        raise ValueError(f"{label} must be an explicit ISO date")
    try:
        timestamp = pd.Timestamp(text).normalize()
    except Exception as exc:
        raise ValueError(f"{label} must be an explicit ISO date: {text!r}") from exc
    return timestamp


def research_cutoff(config) -> pd.Timestamp:
    """Return the one inclusive cutoff shared by every research workflow."""
    policy = getattr(config, "date_policy", None)
    return _normalised_date(
        getattr(policy, "research_cutoff", None),
        label="date_policy.research_cutoff",
    )


def forward_observation_start(config) -> pd.Timestamp:
    """First calendar date strictly after the inclusive research cutoff."""
    return research_cutoff(config) + pd.Timedelta(days=1)


def require_research_end(config, requested_end: Any = None) -> pd.Timestamp:
    """Resolve a research end and reject any date after the global cutoff."""
    cutoff = research_cutoff(config)
    if requested_end is None or str(requested_end).strip() == LATEST_AVAILABLE:
        return cutoff
    end = _normalised_date(requested_end, label="research end")
    if end > cutoff:
        raise ValueError(
            f"research end {end.date()} exceeds the inclusive framework cutoff "
            f"{cutoff.date()}"
        )
    return end


def apply_research_end(config, requested_end: Any = None) -> pd.Timestamp:
    """Clamp a mutable workflow config to an admitted research end."""
    end = require_research_end(config, requested_end)
    start = _normalised_date(config.date_range.start, label="date_range.start")
    if start > end:
        raise ValueError(
            f"research start {start.date()} is after research end {end.date()}"
        )
    config.date_range.end = end.date().isoformat()
    return end


def factor_validation_window(
    config,
    data_manager,
    *,
    frequency: str = "daily_intraday",
    requested_end: Any = None,
) -> FactorValidationWindow:
    """Resolve the framework's default warmup + IS + OOS factor-test window."""
    end = require_research_end(config, requested_end)
    policy = config.validation_policy
    try:
        is_bars = int(policy.wf_train_bars_by_frequency[frequency])
        oos_bars = int(policy.wf_test_bars_by_frequency[frequency])
        warmup_days = int(policy.warmup_days_by_frequency[frequency])
    except KeyError as exc:
        raise ValueError(
            f"factor validation window is not configured for {frequency!r}"
        ) from exc
    if is_bars < 1 or oos_bars < 1 or warmup_days < 0:
        raise ValueError("factor validation window parameters must be positive")

    start = _normalised_date(config.date_range.start, label="date_range.start")
    calendar = pd.DatetimeIndex(data_manager.get_calendar(start, end)).normalize()
    calendar = calendar[(calendar >= start) & (calendar <= end)]
    required = is_bars + oos_bars
    if len(calendar) < required:
        raise ValueError(
            f"factor validation requires {required} trading days through "
            f"{end.date()}, but the source calendar has {len(calendar)}"
        )
    sample = calendar[-required:]
    is_dates = sample[:is_bars]
    oos_dates = sample[is_bars:]
    return FactorValidationWindow(
        factor_start=is_dates[0] - pd.Timedelta(days=warmup_days),
        is_start=is_dates[0],
        is_end=is_dates[-1],
        oos_start=oos_dates[0],
        oos_end=oos_dates[-1],
        is_bars=is_bars,
        oos_bars=oos_bars,
        warmup_calendar_days=warmup_days,
    )


def resolve_observation_end(config, data_manager) -> pd.Timestamp:
    """Resolve ``latest_available`` through the selected certified source."""
    configured = str(config.date_range.end or "").strip()
    fetcher = getattr(data_manager.source, "fetch_latest_trade_date", None)
    if not callable(fetcher):
        raise NotImplementedError(
            "configured source does not expose its latest complete trade date"
        )
    latest = pd.Timestamp(fetcher()).normalize()
    if configured == LATEST_AVAILABLE:
        end = latest
    else:
        end = _normalised_date(configured, label="date_range.end")
        if end > latest:
            raise ValueError(
                f"observation end {end.date()} exceeds source latest complete "
                f"trade date {latest.date()}"
            )
    start = _normalised_date(config.date_range.start, label="date_range.start")
    if start > end:
        raise ValueError(
            f"observation start {start.date()} is after observation end {end.date()}"
        )
    config.date_range.end = end.date().isoformat()
    return end
