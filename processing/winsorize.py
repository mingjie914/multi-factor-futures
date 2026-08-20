from __future__ import annotations

import pandas as pd

from core.interfaces import ProcessingStep
from core.registry import register


@register("processing_step", "winsorize")
class WinsorizeStep(ProcessingStep):
    """去极值: MAD 法或分位数法."""

    name = "winsorize"

    def __init__(self, method: str = "mad", n_sigma: float = 3.0):
        if method not in {"mad", "quantile"}:
            raise ValueError(f"unsupported winsorize method: {method!r}")
        if not 0 < float(n_sigma) < (50 if method == "quantile" else float("inf")):
            raise ValueError(
                "n_sigma must be positive (and below 50 for quantile winsorization)"
            )
        self.method = method
        self.n_sigma = float(n_sigma)

    def transform(
        self, factor: pd.DataFrame, context=None
    ) -> pd.DataFrame:
        result = factor.copy()
        if self.method == "mad":
            med = result.median(axis=1)
            mad = (result.subtract(med, axis=0)).abs().median(axis=1)
            upper = med + self.n_sigma * 1.4826 * mad
            lower = med - self.n_sigma * 1.4826 * mad
            result = result.clip(lower=lower, upper=upper, axis=0)
        elif self.method == "quantile":
            lo = result.quantile(self.n_sigma / 100.0, axis=1)
            hi = result.quantile(1 - self.n_sigma / 100.0, axis=1)
            result = result.clip(lower=lo, upper=hi, axis=0)
        else:
            raise AssertionError(f"unreachable winsorize method: {self.method!r}")
        return result
