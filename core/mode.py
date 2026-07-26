"""Strategy mode enumeration for the multi-factor framework."""
from __future__ import annotations

from enum import Enum


class StrategyMode(Enum):
    """Supported trading strategy modes.

    Attributes:
        TREND: Trend-following strategy.
        SWING: Swing-trading strategy.
        REVERSAL: Mean-reversion / reversal strategy.
        BREAKOUT: Breakout strategy.
        CUSTOM: Custom / user-defined strategy.
    """

    TREND = "trend"
    SWING = "swing"
    REVERSAL = "reversal"
    BREAKOUT = "breakout"
    CUSTOM = "custom"
