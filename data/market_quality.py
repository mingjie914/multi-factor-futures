"""Strict close-price quality gates shared by research and backtests."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd


class CloseDataQualityError(RuntimeError):
    """Raised when a post-listing close gap is not an audited market closure."""


def audited_nontrading_mask(
    close: pd.DataFrame,
    events: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Build an explicit non-trading mask from audited root/date records."""
    prices = pd.DataFrame(close)
    mask = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    for root, dates in dict(events or {}).items():
        root = str(root).upper()
        if root not in mask.columns:
            continue
        for value in dates:
            date = pd.Timestamp(value)
            if date in mask.index:
                mask.loc[date, root] = True
    return mask


def prepare_close_data(
    close: pd.DataFrame,
    audited_nontrading_closes: Mapping[str, Sequence[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return causal marked returns and close-execution availability.

    Leading missing values are treated as pre-listing history.  Any missing
    close after an instrument's first observation fails closed unless its
    root/date is explicitly listed as an audited market closure.  An approved
    closure keeps the last observable mark, is not tradable on that close, and
    records the full intervening move when the next close is observed.
    """
    prices = pd.DataFrame(close, copy=True).astype(float)
    nontrading = audited_nontrading_mask(prices, audited_nontrading_closes)
    after_listing = prices.notna().cummax()
    unexpected = prices.isna() & after_listing & ~nontrading
    if bool(unexpected.any().any()):
        locations = [
            f"{root}@{pd.Timestamp(date).date()}"
            for date, root in unexpected.stack().loc[lambda values: values].index[:8]
        ]
        raise CloseDataQualityError(
            "unapproved close gaps after first observation: " + ", ".join(locations)
        )

    marked = prices.ffill()
    returns = marked.pct_change(fill_method=None).where(marked.notna())
    tradable = prices.notna() & ~nontrading
    return returns, tradable


__all__ = [
    "CloseDataQualityError",
    "audited_nontrading_mask",
    "prepare_close_data",
]
