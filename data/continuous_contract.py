"""连续合约构建: 比例后复权法 (Ratio Back-Adjustment).

换月时, 旧合约平仓 + 新合约开仓会产生价格跳空.
比例后复权通过乘以累计复权因子消除跳空, 保证收益率连续性.

算法:
    换月日 t (旧合约作为主力的最后一天):
        old_close = 旧合约 t 日收盘价
        new_close = 新合约 t 日收盘价
        ratio = old_close / new_close

    累计因子 = 历次换月 ratio 的乘积
    后复权价格 = 原始价格 × 累计因子

收益连续性验证:
    换月日 t 的收益 (旧合约最后一天):
        r_t = old_close_t / old_close_{t-1} - 1

    换月日 t+1 的收益 (新合约第一天):
        r_{t+1} = (new_close_{t+1} × ratio) / old_close_t - 1
                 = new_close_{t+1} / new_close_t - 1  ← 新合约收益 ✓
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

# 需要复权的价格字段 (乘以累计因子)
PRICE_FIELDS = {"open", "high", "low", "close", "settle", "pre_settle"}

class RolloverAdjustmentError(ValueError):
    """Raised when a continuous contract cannot be priced without a roll gap."""


def build_continuous_series(
    contract_data: Dict[str, pd.DataFrame],
    schedule: pd.DataFrame,
    fields: List[str],
) -> pd.DataFrame:
    """为单个品种构建比例后复权连续合约.

    价格字段: 比例后复权 (乘以累计因子)
    成交量/持仓量: 始终取当日选定合约原值，保持各数据源语义一致

    Args:
        contract_data: {具体合约代码: DataFrame(date, fields)}
        schedule: 主力合约切换时间表, 列 [main_contract, start_date, end_date]
        fields: 需要处理的字段列表

    Returns:
        DataFrame: index=date, columns=fields 的连续合约数据
    """
    schedule = schedule.sort_values("start_date").reset_index(drop=True)

    segments: List[pd.DataFrame] = []
    cumulative_ratio = 1.0

    for i, row in schedule.iterrows():
        contract = row["main_contract"]
        seg_start = row["start_date"]
        seg_end = row["end_date"]

        if contract not in contract_data:
            continue

        df = contract_data[contract].loc[seg_start:seg_end].copy()
        if df.empty:
            continue

        if i > 0:
            prev_contract = schedule.iloc[i - 1]["main_contract"]
            rollover_date = schedule.iloc[i - 1]["end_date"]

            # 价格: 比例后复权
            ratio = _compute_rollover_ratio(
                contract_data, prev_contract, contract, rollover_date
            )
            cumulative_ratio *= ratio

        # 价格字段乘以累计复权因子
        for field in fields:
            if field in PRICE_FIELDS and field in df.columns:
                df[field] = df[field] * cumulative_ratio

        segments.append(df)

    if not segments:
        return pd.DataFrame()

    result = pd.concat(segments)
    result = result[~result.index.duplicated(keep="last")]
    return result.sort_index()


def _compute_rollover_ratio(
    contract_data: Dict[str, pd.DataFrame],
    old_contract: str,
    new_contract: str,
    rollover_date,
) -> float:
    """计算换月日的比例复权因子.

    ratio = old_close / new_close

    在换月日, 旧合约和新合约同时有交易数据.
    用旧合约收盘价除以新合约收盘价, 使得:
    - 旧合约最后一天的收益 = 旧合约自身收益
    - 新合约第一天的收益 = 新合约自身收益 (无跳空)

    The latest common trading date at or before ``rollover_date`` is used so
    an expiry-day missing quote does not manufacture a price jump.  If the two
    contracts never overlap, formal continuous-contract construction fails.
    """
    old_df = contract_data.get(old_contract)
    new_df = contract_data.get(new_contract)
    if (
        old_df is None or new_df is None
        or "close" not in old_df.columns or "close" not in new_df.columns
    ):
        raise RolloverAdjustmentError(
            f"missing close series for rollover {old_contract}->{new_contract}"
        )
    cutoff = pd.Timestamp(rollover_date)
    old_close = pd.to_numeric(old_df.loc[:cutoff, "close"], errors="coerce")
    new_close = pd.to_numeric(new_df.loc[:cutoff, "close"], errors="coerce")
    common = old_close.dropna().index.intersection(new_close.dropna().index)
    if len(common):
        overlap_date = common.max()
        old_value = float(old_close.loc[overlap_date])
        new_value = float(new_close.loc[overlap_date])
        if pd.notna(old_value) and pd.notna(new_value) and new_value > 0.0:
            return old_value / new_value
    raise RolloverAdjustmentError(
        f"no common close at or before {cutoff.date()} for "
        f"{old_contract}->{new_contract}"
    )
