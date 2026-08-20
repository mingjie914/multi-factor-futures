from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict


TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.0


def compute_sharpe(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = RISK_FREE_RATE,
) -> float:
    """Compute annualized Sharpe from periodic excess returns.

    ``risk_free_rate`` is annualized and defaults explicitly to zero, which is
    the framework-wide research and backtest convention.
    """
    r = returns.dropna()
    if len(r) < 5:
        return 0.0
    excess = r - float(risk_free_rate) / periods_per_year
    std = excess.std()
    return float(excess.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0


def compute_max_drawdown(nav: pd.Series) -> float:
    if nav.empty:
        return 0.0
    peak = nav.expanding().max()
    dd = (nav - peak) / peak
    return float(dd.min())


def compute_annual_return(nav: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    if len(nav) < 2:
        return 0.0
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    # N marked NAV observations span N-1 return intervals.
    n_years = (len(nav) - 1) / periods_per_year
    return float((1 + total_ret) ** (1 / n_years) - 1) if n_years > 0 else 0.0


def compute_win_rate(signals: pd.DataFrame) -> float:
    if signals.empty:
        return 0.0
    wins = (
        signals[signals["pnl"] > 0]
        if "pnl" in signals.columns
        else signals[signals["reason"].str.contains("take_profit", na=False)]
    )
    return len(wins) / len(signals) if len(signals) > 0 else 0.0


def compute_all_metrics(
    nav: pd.Series,
    returns: pd.Series = None,
    signals: pd.DataFrame = None,
    periods_per_year: int = 252,
    risk_free_rate: float = RISK_FREE_RATE,
) -> Dict[str, float]:
    ret = returns if returns is not None else nav.pct_change(fill_method=None)
    metrics = {
        "annual_return": compute_annual_return(nav, periods_per_year),
        "sharpe": compute_sharpe(
            ret, periods_per_year, risk_free_rate=risk_free_rate
        ),
        "max_drawdown": compute_max_drawdown(nav),
        "volatility": float(ret.dropna().std() * np.sqrt(periods_per_year))
        if len(ret.dropna()) > 0
        else 0.0,
        "win_rate": compute_win_rate(signals) if signals is not None else 0.0,
        "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1) if len(nav) > 0 else 0.0,
        "risk_free_rate": float(risk_free_rate),
    }
    # Calmar
    metrics["calmar"] = (
        metrics["annual_return"] / abs(metrics["max_drawdown"])
        if metrics["max_drawdown"] != 0
        else 0.0
    )
    return metrics


def compute_split_metrics(
    nav: pd.Series,
    returns: pd.Series = None,
    train_ratio: float = 0.75,
    periods_per_year: int = 252,
    risk_free_rate: float = RISK_FREE_RATE,
    minimum_train_bars: int = 750,
    minimum_test_bars: int = 250,
) -> Dict[str, Dict[str, float]]:
    """历史分段诊断: 按 train_ratio 分割回测期间并分别计算前后段指标.

    该结果只是固定历史样本内的分段诊断，不是真正的未见样本外验证。

    Args:
        nav: 日度净值序列.
        returns: 日度收益序列 (可选, 默认从 nav 推导).
        train_ratio: 前段占比 (默认 0.75, 即前 75% 为前段).
        periods_per_year: 年化频率 (日度=252).

    Returns:
        {"train": {metrics}, "test": {metrics}}
    """
    if nav.empty:
        return {"train": {}, "test": {}}

    ret = returns if returns is not None else nav.pct_change(fill_method=None)
    n = len(nav)
    split_idx = int(n * train_ratio)
    if (
        split_idx < int(minimum_train_bars)
        or n - split_idx < int(minimum_test_bars)
        or split_idx / max(n - split_idx, 1) < 3.0
    ):
        return {"train": {}, "test": {}}

    # 训练期: 从起点到 split_idx (归一化 nav 从 1 开始)
    nav_train = nav.iloc[:split_idx]
    nav_train = nav_train / nav_train.iloc[0]
    ret = ret.reindex(nav.index)
    ret_train = ret.iloc[1:split_idx]

    # 测试期: 从 split_idx 到终点 (归一化 nav 从 1 开始)
    nav_test = nav.iloc[split_idx:]
    nav_test = nav_test / nav_test.iloc[0]
    # The first test NAV is the segment anchor; its return belongs to the
    # preceding train-to-test boundary and must not be counted in either side.
    ret_test = ret.iloc[split_idx + 1:]

    return {
        "train": compute_all_metrics(
            nav_train,
            ret_train,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
        ),
        "test": compute_all_metrics(
            nav_test,
            ret_test,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
        ),
    }
