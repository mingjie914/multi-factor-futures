from __future__ import annotations

import numpy as np
import pandas as pd

from core.interfaces import ProcessingStep
from core.registry import register


@register("processing_step", "neutralize")
class NeutralizeStep(ProcessingStep):
    """中性化: 对行业/市值做回归取残差.

    ⚠️ 行业数据从 ProcessingContext.industry 获取, 禁止 import data/manager.py
    """

    name = "neutralize"

    def __init__(self, by: list | None = None):
        self.by = by or ["industry"]

    def transform(
        self, factor: pd.DataFrame, context=None
    ) -> pd.DataFrame:
        result = factor.copy()
        if (
            context is None
            or not hasattr(context, "industry")
            or context.industry is None
        ):
            return result
        industry = context.industry
        # 简单实现: 减去行业均值
        common_dates = result.index.intersection(industry.index)
        for date in common_dates:
            ind_series = industry.loc[date]
            ind_aligned = ind_series.reindex(result.columns).fillna("_UNKNOWN_")
            result.loc[date] -= result.loc[date].groupby(ind_aligned).transform("mean")
        return result
