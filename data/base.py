from __future__ import annotations

from typing import List

from core.interfaces import DataProvider, DataSource
from core.types import *

# Re-export for convenient importing from data.base
__all__ = [
    "DataProvider",
    "DataSource",
    "merge_price_panels",
]


def merge_price_panels(panels: List[PricePanel]) -> PricePanel:
    """合并多个 PricePanel（用于多源数据融合）. 后面来源覆盖前面.

    Args:
        panels: 按优先级排序的 PricePanel 列表, 后面覆盖前面同名 field.

    Returns:
        合并后的 PricePanel.
    """
    result: PricePanel = {}
    for panel in panels:
        for field, df in panel.items():
            if field not in result:
                result[field] = df
            else:
                # 后面覆盖前面: 对重合的 index/columns 以后者为准
                result[field] = df.combine_first(result[field])
                # combine_first 不会覆盖已有值, 所以再 update 一次
                result[field].update(df)
    return result
