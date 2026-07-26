from __future__ import annotations

import os

import pandas as pd

from core.types import SignalFrame


class SignalOutput:
    def __init__(self, output_dir: str = "./signals_output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def to_csv(self, signal: SignalFrame, path: str = None) -> str:
        if path is None:
            if signal.empty:
                return ""
            path = os.path.join(
                self.output_dir,
                f"signals_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv",
            )
        signal.to_csv(path, index=False)
        return path

    def to_json(self, signal: SignalFrame, path: str = None) -> str:
        if path is None:
            path = os.path.join(
                self.output_dir,
                f"signals_{pd.Timestamp.now():%Y%m%d_%H%M%S}.json",
            )
        signal.to_json(path, orient="records", date_format="iso")
        return path
