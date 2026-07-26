"""因子矩阵工具函数.

提供 OLSModel 和 BarraFuturesModel 共用的 stack+join+dropna 逻辑,
避免重复计算.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from core.types import FactorMatrix, ReturnMatrix


def stack_factors_and_returns(
    factor_matrices: Dict[str, FactorMatrix],
    forward_returns: ReturnMatrix,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """将多个因子矩阵和收益矩阵 stack 为长表, 对齐后 dropna.

    这是 OLSModel.fit() 和 BarraFuturesModel.estimate() 的公共步骤,
    提取出来避免重复计算. 千级因子场景下只做一次 stack+join.

    Args:
        factor_matrices: {factor_name: FactorMatrix} 字典.
        forward_returns: 收益矩阵 (dates × tickers).

    Returns:
        (merged_df, factor_names_sorted, X_vals, y_vals, date_codes):
        - merged_df: 对齐后的 DataFrame, MultiIndex (date, ticker)
        - factor_names_sorted: 排序后的因子名列表
        - X_vals: (N, K) 因子值数组
        - y_vals: (N,) 收益值数组
        - date_codes: (N,) 日期编码数组 (用于分组回归)
    """
    factor_names = sorted(factor_matrices.keys())
    if not factor_names:
        return pd.DataFrame(), factor_names, np.array([]), np.array([]), np.array([])

    # Stack 所有因子为长表
    # The training pipeline normally supplies already aligned dense panels. A
    # single NumPy mask avoids K separate stack operations and a large index
    # join. Keep the pandas path below for irregular or duplicate-labelled data.
    first = factor_matrices[factor_names[0]]
    aligned = (
        isinstance(first, pd.DataFrame)
        and first.index.is_unique
        and first.columns.is_unique
        and forward_returns.index.equals(first.index)
        and forward_returns.columns.equals(first.columns)
        and all(
            isinstance(factor_matrices[name], pd.DataFrame)
            and factor_matrices[name].index.equals(first.index)
            and factor_matrices[name].columns.equals(first.columns)
            for name in factor_names[1:]
        )
    )
    if aligned:
        try:
            columns = [
                factor_matrices[name].to_numpy(dtype=np.float64, copy=False).reshape(-1)
                for name in factor_names
            ]
            columns.append(
                forward_returns.to_numpy(dtype=np.float64, copy=False).reshape(-1)
            )
            values = np.column_stack(columns)
            valid = ~pd.isna(values).any(axis=1)
            flat_index = pd.MultiIndex.from_product(
                [first.index, first.columns],
                names=[first.index.name, first.columns.name],
            )
            merged = pd.DataFrame(
                values[valid],
                index=flat_index[valid],
                columns=factor_names + ["fwd_ret"],
            ).sort_index(level=0)
            if len(merged) < len(factor_names) + 5:
                return merged, factor_names, np.array([]), np.array([]), np.array([])
            X_vals = merged[factor_names].to_numpy(dtype=np.float64, copy=False)
            y_vals = merged["fwd_ret"].to_numpy(dtype=np.float64, copy=False)
            date_level = merged.index.get_level_values(0).values
            _, date_codes = np.unique(date_level, return_inverse=True)
            return merged, factor_names, X_vals, y_vals, date_codes
        except (TypeError, ValueError):
            pass

    panels = []
    for name in factor_names:
        f = factor_matrices[name]
        stacked = f.stack()
        stacked.name = name
        panels.append(stacked)
    X_all = pd.concat(panels, axis=1)

    # Stack 收益
    y_stacked = forward_returns.stack()
    y_stacked.name = "fwd_ret"

    # 对齐并 dropna
    merged = X_all.join(y_stacked, how="inner").dropna()

    if len(merged) < len(factor_names) + 5:
        return merged, factor_names, np.array([]), np.array([]), np.array([])

    # 按日期排序
    merged = merged.sort_index(level=0)
    X_vals = merged[factor_names].values.astype(np.float64, copy=False)
    y_vals = merged["fwd_ret"].values.astype(np.float64, copy=False)

    # 提取日期编码
    date_level = merged.index.get_level_values(0).values
    dates_unique, date_codes = np.unique(date_level, return_inverse=True)

    return merged, factor_names, X_vals, y_vals, date_codes
