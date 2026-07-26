"""Core type definitions for the multi-factor framework.

This module defines all cross-module type aliases and dataclasses used
throughout the framework, providing a single source of truth for data shapes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # noqa: F401
import pandas as pd

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

DateIndex = pd.DatetimeIndex
"""Index of datetime values, used to represent trading dates."""

TickerIndex = pd.Index
"""Index of ticker symbols."""

Date = pd.Timestamp
"""A single trading date."""

PricePanel = Dict[str, pd.DataFrame]
"""Price data organised as {open/high/low/close/volume/... : dates x tickers}."""

IndustryMapping = pd.Series
"""Industry classification mapping. Index = ticker, value = industry_code."""

Universe = TickerIndex
"""A universe of ticker symbols (alias of TickerIndex)."""

UniverseSchedule = pd.DataFrame
"""Time-varying universe membership. Index = date, columns = ticker (bool)."""

ReturnMatrix = pd.DataFrame
"""Forward returns matrix. Index = dates, columns = tickers."""

FactorMatrix = pd.DataFrame
"""Factor exposure matrix. Index = dates, columns = tickers."""

ProcessedFactorMatrix = pd.DataFrame
"""Factor matrix after processing (outlier handling, neutralisation, etc.)."""

FactorReturns = pd.DataFrame
"""Factor returns time series. Index = dates, columns = factor_names."""

FactorCov = pd.DataFrame
"""Factor covariance matrix. Index = factor_names, columns = factor_names."""

SpecificRisk = pd.Series
"""Idiosyncratic (specific) risk per ticker."""

WeightVector = pd.Series
"""Portfolio weight vector. Index = ticker, value = weight."""

ExpectedReturns = pd.Series
"""Expected returns vector. Index = ticker, value = expected_return."""

SignalFrame = pd.DataFrame
"""Signal data frame. Columns correspond to Signal dataclass fields."""

NAVSeries = pd.Series
"""Net asset value time series."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MarketState:
    """Snapshot of market state for a given date.

    Attributes:
        date: Trading date.
        prices: Current prices (typically close / settlement).
        high: High prices.
        low: Low prices.
        open: Open prices.
        volume: Trading volume.
        atr: Average true range.
        pre_settle: Previous settlement price.
        additional: Extra market data fields.
    """

    date: Date
    prices: pd.Series
    high: pd.Series
    low: pd.Series
    open: pd.Series
    volume: pd.Series
    atr: pd.Series
    pre_settle: pd.Series
    additional: Dict[str, pd.Series] = field(default_factory=dict)


@dataclass
class Position:
    """A single open position.

    Attributes:
        ticker: Instrument identifier.
        side: 'long' or 'short'.
        quantity: Number of contracts / shares held.
        entry_price: Price at entry.
        entry_date: Date the position was opened.
        current_price: Latest available price.
        high_since_entry: Highest price observed since entry.
        pnl: Unrealised profit / loss (currency units).
        pnl_pct: Unrealised profit / loss as a percentage.
    """

    ticker: str
    side: str  # 'long' | 'short'
    quantity: int
    entry_price: float
    entry_date: Date
    current_price: float
    high_since_entry: float
    pnl: float
    pnl_pct: float


@dataclass
class Signal:
    """Trading signal for a single instrument on a single date.

    Attributes:
        date: Signal date.
        ticker: Instrument identifier.
        action: Operation type.
        target_position: Desired position size after execution.
        take_profit: Take-profit price level (optional).
        stop_loss: Stop-loss price level (optional).
        trailing_stop: Trailing stop distance (optional).
        holding_period: Maximum holding periods (optional).
        mode: Strategy mode identifier.
        reason: Human-readable reason for the signal.
        metadata: Arbitrary extra payload.
    """

    date: Date
    ticker: str
    action: str  # 'open_long' | 'open_short' | 'close_long' | 'close_short' |
                 # 'add_long' | 'add_short' | 'reduce_long' | 'reduce_short' | 'hold'
    target_position: int
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    trailing_stop: Optional[float] = None
    holding_period: Optional[int] = None
    mode: str = ""
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveredFactor:
    """A factor discovered through automated mining.

    Attributes:
        name: Human-readable factor name.
        expression: Mathematical / logical expression defining the factor.
        category: Factor category (default 'auto_mined').
        eval_result: Optional evaluation results.
    """

    name: str
    expression: str
    category: str = "auto_mined"
    eval_result: Optional[Dict] = None

    def to_factor_instance(self) -> "Factor":
        """Convert this discovered factor into a concrete Factor instance.

        Note:
            This is a placeholder pending real import to avoid circular
            dependencies at module level.
        """
        raise NotImplementedError("Concrete Factor conversion not yet implemented.")
