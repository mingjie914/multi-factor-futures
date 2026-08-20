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
        if method not in {"zscore", "rank"}:
            raise ValueError(f"unsupported standardize method: {method!r}")
        self.method = method
        self.by_date = bool(by_date)

    def transform(
        self, factor: pd.DataFrame, context=None
    ) -> pd.DataFrame:
        result = factor.copy()
        observed = result.notna()
        if self.method == "zscore":
            if self.by_date:
                mean = result.mean(axis=1)
                std = result.std(axis=1).replace(0, np.nan)
                result = result.subtract(mean, axis=0).div(std, axis=0)
            else:
                values = result.to_numpy(dtype=float, copy=False)
                mean = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
                std = float(np.nanstd(values, ddof=1)) if np.isfinite(values).sum() > 1 else np.nan
                result = (result - mean) / std
        elif self.method == "rank":
            if self.by_date:
                result = result.rank(axis=1, pct=True)
            else:
                result = result.stack().rank(pct=True).unstack()
        else:
            raise AssertionError(f"unreachable standardize method: {self.method!r}")
        # A zero cross-sectional score is a valid neutral value for an
        # observed constant row.  A missing source observation is not.
        return result.reindex_like(factor).fillna(0).where(observed)
