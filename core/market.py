"""Market enumeration (deprecated - use config 'market' string field directly).
    # multi_factor
"""
from __future__ import annotations
from enum import Enum


class Market(Enum):
    ASHARE = "ashare"
    FUTURES = "futures"