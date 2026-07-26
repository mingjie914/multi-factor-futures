from __future__ import annotations

from typing import Dict

import pandas as pd

from core.types import Date, ExpectedReturns, FactorMatrix, Universe


def combine_equal(
    factors: Dict[str, FactorMatrix],
    date: Date,
    universe: Universe,
) -> ExpectedReturns:
    """Combine multiple factor exposures with equal weight.

    Args:
        factors: Dictionary of factor name -> FactorMatrix.
        date: Target date.
        universe: Universe of tickers to produce expected returns for.

    Returns:
        ExpectedReturns series indexed by universe tickers.
    """
    vals = []
    for name, f in factors.items():
        if date in f.index:
            vals.append(f.loc[date].reindex(universe).fillna(0))
    if not vals:
        return pd.Series(0.0, index=universe)
    return pd.concat(vals, axis=1).mean(axis=1)


def combine_ic_weighted(
    factors: Dict[str, FactorMatrix],
    ic_series: Dict[str, float],
    date: Date,
    universe: Universe,
) -> ExpectedReturns:
    """Combine multiple factor exposures weighted by their IC values.

    Args:
        factors: Dictionary of factor name -> FactorMatrix.
        ic_series: Dictionary of factor name -> IC value (weight).
        date: Target date.
        universe: Universe of tickers to produce expected returns for.

    Returns:
        ExpectedReturns series indexed by universe tickers.
    """
    total_w = sum(abs(v) for v in ic_series.values()) or 1.0
    result = pd.Series(0.0, index=universe)
    for name, f in factors.items():
        if date in f.index and name in ic_series:
            w = ic_series[name] / total_w
            result += w * f.loc[date].reindex(universe).fillna(0)
    return result
