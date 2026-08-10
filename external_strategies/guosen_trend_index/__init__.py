"""Guosen China Assets Trend Allocation Index-style adapter."""

from .strategy import (
    ExternalBacktestResult,
    GuosenTrendIndexBacktester,
    GuosenTrendIndexSpec,
    load_snapshot,
)

__all__ = [
    "ExternalBacktestResult",
    "GuosenTrendIndexBacktester",
    "GuosenTrendIndexSpec",
    "load_snapshot",
]
