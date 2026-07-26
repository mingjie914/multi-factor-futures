from __future__ import annotations

from typing import List, Dict, Optional, Tuple  # noqa: F401

import pandas as pd
import numpy as np

from core.types import (
    Date,
    MarketState,
    Signal,
    SignalFrame,
    WeightVector,
)
from core.interfaces import SignalGenerator
from core.registry import register
from signals.signal import signals_to_frame


@register("signal_generator", "trend_following")
class TrendFollowingGenerator(SignalGenerator):
    """趋势跟踪信号生成器.

    开仓: 权重 > 0 且趋势确认 (MA5 > MA20)
    平仓: 权重 = 0 或 反向趋势
    """

    name = "trend_following"
    mode = "trend"

    def __init__(self, fast_ma: int = 5, slow_ma: int = 20):
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma

    def generate(
        self,
        target_weights: WeightVector,
        current_positions: pd.DataFrame,
        factor_snapshot: Dict[str, pd.Series],
        market_state: MarketState,
        mode_params: Dict,
        date: Date,
    ) -> SignalFrame:
        signals: List[Signal] = []
        for ticker in target_weights.index:
            tw = target_weights.get(ticker, 0.0)
            # 判断是否已有持仓
            has_position = False
            if not current_positions.empty:
                pos = current_positions[current_positions["ticker"] == ticker]
                has_position = len(pos) > 0 and pos.iloc[0]["quantity"] > 0

            price = market_state.prices.get(ticker, np.nan)
            if pd.isna(price) or price <= 0:
                continue

            if tw > 0 and not has_position:
                sig = Signal(
                    date=date,
                    ticker=ticker,
                    action="open_long",
                    target_position=int(tw * 10),
                    reason="trend_open",
                )
            elif tw <= 0 and has_position:
                sig = Signal(
                    date=date,
                    ticker=ticker,
                    action="close_long",
                    target_position=0,
                    reason="trend_close",
                )
            else:
                sig = Signal(
                    date=date,
                    ticker=ticker,
                    action="hold",
                    target_position=0,
                    reason="hold",
                )
            signals.append(sig)

        return signals_to_frame(signals)