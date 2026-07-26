from __future__ import annotations

from typing import Tuple, Dict

import numpy as np  # noqa: F401

from core.types import Position, MarketState
from core.interfaces import SLTPRule
from core.registry import register


@register("sl_tp_rule", "take_profit")
class TakeProfitRule(SLTPRule):
    name = "take_profit"

    def __init__(self, tgt_pct: float = 0.05):
        self.tgt_pct = tgt_pct

    def should_close(
        self, position: Position, market_state: MarketState, entry_context: Dict
    ) -> Tuple[bool, str]:
        if position.pnl_pct >= self.tgt_pct:
            return True, f"take_profit_{self.tgt_pct:.0%}"
        return False, ""


@register("sl_tp_rule", "hard_stop_loss")
class HardStopLossRule(SLTPRule):
    name = "hard_stop_loss"

    def __init__(self, sl_pct: float = 0.05):
        self.sl_pct = sl_pct

    def should_close(
        self, position: Position, market_state: MarketState, entry_context: Dict
    ) -> Tuple[bool, str]:
        if position.pnl_pct <= -self.sl_pct:
            return True, f"stop_loss_{self.sl_pct:.0%}"
        return False, ""


@register("sl_tp_rule", "trailing_stop")
class TrailingStopRule(SLTPRule):
    name = "trailing_stop"

    def __init__(self, tr_pct: float = 0.03):
        self.tr_pct = tr_pct

    def should_close(
        self, position: Position, market_state: MarketState, entry_context: Dict
    ) -> Tuple[bool, str]:
        if position.high_since_entry > position.entry_price:
            drawdown = (
                position.high_since_entry - position.current_price
            ) / position.high_since_entry
            if drawdown >= self.tr_pct:
                return True, f"trailing_stop_{self.tr_pct:.0%}"
        return False, ""
