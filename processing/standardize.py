from __future__ import annotations

import numpy as np
import pandas as pd

from core.interfaces import ProcessingStep
from core.registry import register


@register("processing_step", "standardize")
class StandardizeStep(ProcessingStep):
    """截面标准化: Z-score 或 Rank."""

    name = "standardize"

    def __init__(self, method: str = "zscore", by_date: bool = True):
        self.method = method
        self.by_date = by_date

    def transform(
        self, factor: pd.DataFrame, context=None
    ) -> pd.DataFrame:
        result = factor.copy()
        if self.method == "zscore":
            if self.by_date:
                mean = result.mean(axis=1)
                std = result.std(axis=1).replace(0, np.nan)
                result = result.subtract(mean, axis=0).div(std, axis=0)
            else:
                result = (
                    (result - result.mean().mean()) / result.std().std()
                )
        elif self.method == "rank":
            if self.by_date:
                result = result.rank(axis=1, pct=True)
            else:
                result = result.stack().rank(pct=True).unstack()
        return result.fillna(0)
