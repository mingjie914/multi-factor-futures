from __future__ import annotations

import pandas as pd

from core.interfaces import ProcessingStep
from core.registry import register


@register("processing_step", "fillna")
class FillNAStep(ProcessingStep):
    """缺失值填充."""

    name = "fillna"

    def __init__(self, method: str = "zero"):
        if method not in {"zero", "forward", "mean"}:
            raise ValueError(f"unsupported fillna method: {method!r}")
        self.method = method

    def transform(
        self, factor: pd.DataFrame, context=None
    ) -> pd.DataFrame:
        if self.method == "zero":
            return factor.fillna(0)
        elif self.method == "forward":
            return factor.ffill().fillna(0)
        elif self.method == "mean":
            return factor.fillna(factor.mean(axis=1), axis=0).fillna(0)
        raise AssertionError(f"unreachable fillna method: {self.method!r}")
