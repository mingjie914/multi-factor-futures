from __future__ import annotations

import pandas as pd

from core.types import Signal, SignalFrame


def signals_to_frame(signals: list) -> SignalFrame:
    """将 Signal 列表转为 DataFrame."""
    records = [
        {
            "date": s.date,
            "ticker": s.ticker,
            "action": s.action,
            "target_position": s.target_position,
            "take_profit": s.take_profit,
            "stop_loss": s.stop_loss,
            "trailing_stop": s.trailing_stop,
            "holding_period": s.holding_period,
            "mode": s.mode,
            "reason": s.reason,
        }
        for s in signals
    ]
    return pd.DataFrame(records) if records else pd.DataFrame()
