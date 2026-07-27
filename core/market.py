"""Legacy market enum kept for import compatibility.

New code should use the validated ``FrameworkConfig.market`` string.
"""
from __future__ import annotations

from enum import Enum


class Market(Enum):
    ASHARE = "ashare"
    FUTURES = "futures"
