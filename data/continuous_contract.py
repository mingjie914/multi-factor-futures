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

# 换月窗口内需要新旧合约求和的字段 (成交量/持仓量)
# 换月期间旧合约成交量逐渐下降、新合约逐渐上升, 真实市场活跃度 = 旧 + 新
ADDITIVE_FIELDS = {"volume", "oi", "amount", "oi_change"}

# 换月窗口: 新合约成为主力后的前 N 个交易日, 与旧合约求和
ROLLOVER_WINDOW = 5


def build_continuous_series(
    contract_data: Dict[str, pd.DataFrame],
    schedule: pd.DataFrame,
    fields: List[str],
) -> pd.DataFrame:
    """为单个品种构建比例后复权连续合约.

    价格字段: 比例后复权 (乘以累计因子)
    成交量/持仓量: 换月窗口内新旧合约求和, 窗口外取主力合约原值

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

            # 成交量/持仓量: 换月窗口内新旧合约求和
            _sum_additive_fields(df, contract_data, prev_contract, fields)

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


def _sum_additive_fields(
    df: pd.DataFrame,
    contract_data: Dict[str, pd.DataFrame],
    old_contract: str,
    fields: List[str],
) -> None:
    """换月窗口内, 将旧合约的成交量/持仓量加到新合约上.

    换月期间两个合约同时活跃, 真实市场成交量 = 新合约量 + 旧合约量.
    窗口限制为 ROLLOVER_WINDOW 个交易日, 避免过度求和.
    """
    additive = [f for f in fields if f in ADDITIVE_FIELDS]
    if not additive:
        return

    old_df = contract_data.get(old_contract)
    if old_df is None:
        return

    # 取新合约段的前 ROLLOVER_WINDOW 个交易日
    window_dates = df.index[:ROLLOVER_WINDOW]
    for dt in window_dates:
        if dt not in old_df.index:
            continue
        for field in additive:
            if field not in df.columns or field not in old_df.columns:
                continue
            old_val = old_df.loc[dt, field]
            new_val = df.loc[dt, field]
            if pd.notna(old_val) and pd.notna(new_val):
                df.loc[dt, field] = new_val + old_val


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

    Returns:
        复权因子. 若数据缺失则返回 1.0 (不调整).
    """
    old_close = None
    new_close = None

    if old_contract in contract_data:
        old_df = contract_data[old_contract]
        if rollover_date in old_df.index and "close" in old_df.columns:
            old_close = old_df.loc[rollover_date, "close"]

    if new_contract in contract_data:
        new_df = contract_data[new_contract]
        if rollover_date in new_df.index and "close" in new_df.columns:
            new_close = new_df.loc[rollover_date, "close"]

    if (
        old_close is not None
        and new_close is not None
        and pd.notna(old_close)
        and pd.notna(new_close)
        and new_close > 0
    ):
        return float(old_close / new_close)
    return 1.0
