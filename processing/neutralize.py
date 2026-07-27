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

    def __init__(
        self,
        by: list | None = None,
        min_group_size: int = 2,
        missing_policy: str = "error",
    ):
        self.by = by or ["industry"]
        self.min_group_size = max(int(min_group_size), 1)
        self.missing_policy = str(missing_policy).lower()
        if self.by != ["industry"]:
            raise ValueError("neutralize currently supports by: [industry] only")
        if self.missing_policy not in {"error", "skip", "nan"}:
            raise ValueError("neutralize missing_policy must be error, skip, or nan")
        self.fail_closed = self.missing_policy == "error"

    def transform(
        self, factor: pd.DataFrame, context=None
    ) -> pd.DataFrame:
        result = factor.copy().astype(float)
        if (
            context is None
            or not hasattr(context, "industry")
            or context.industry is None
        ):
            if self.missing_policy == "error":
                raise RuntimeError("industry labels are required for neutralization")
            if self.missing_policy == "nan":
                return result * np.nan
            return result

        industry = context.industry
        if isinstance(industry, pd.Series):
            first_labels = industry.reindex(result.columns)
            missing_labels = first_labels.isna().any()
            static_labels = True
        else:
            industry = industry.reindex(
                index=result.index, columns=result.columns
            )
            missing_labels = industry.isna().any().any()
            first_labels = industry.iloc[0]
            static_labels = bool(
                industry.eq(first_labels, axis="columns").to_numpy().all()
            )
        if missing_labels:
            if self.missing_policy == "error":
                raise RuntimeError("industry labels contain missing values")
            if self.missing_policy == "nan":
                if isinstance(industry, pd.Series):
                    result.loc[:, first_labels.index[first_labels.isna()]] = np.nan
                else:
                    result = result.where(industry.notna())
            if isinstance(industry, pd.Series):
                first_labels = first_labels.fillna("_UNKNOWN_")
            else:
                industry = industry.fillna("_UNKNOWN_")
                first_labels = industry.iloc[0]

        # The framework taxonomy is static. This branch avoids a per-date
        # Python loop and remains fast for minute-bar panels.
        if static_labels:
            for label in pd.unique(first_labels):
                columns = first_labels.index[first_labels == label]
                values = result.loc[:, columns]
                counts = values.notna().sum(axis=1)
                centered = values.sub(values.mean(axis=1), axis=0)
                centered.loc[counts < self.min_group_size, :] = np.nan
                result.loc[:, columns] = centered
            return result

        # Dynamic classifications are uncommon but supported without changing
        # the public processing contract.
        values = result.stack(dropna=False).rename("value")
        labels = industry.stack(dropna=False).rename("industry")
        long = pd.concat([values, labels], axis=1)
        groups = long.groupby(
            [long.index.get_level_values(0), "industry"], sort=False
        )["value"]
        residual = long["value"] - groups.transform("mean")
        residual = residual.where(groups.transform("count") >= self.min_group_size)
        return residual.unstack().reindex_like(result)
