"""Core type definitions for the multi-factor framework.

This module defines all cross-module type aliases and dataclasses used
throughout the framework, providing a single source of truth for data shapes.
"""
from __future__ import annotations

from typing import Dict

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

NAVSeries = pd.Series
"""Net asset value time series."""
