# -*- coding: utf-8 -*-
"""monitoring.io — 落盘/读取 monitoring 数据(原子写, UTF-8).

数据目录: monitoring_data/
- factor_health.json     因子健康快照(日度更新)
- signals/*.csv          因子信号与收益面板(由 scripts/run_monitoring.py build-signals 生成)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from monitoring.config import MONITOR_DIR, SIGNALS_DIR

_UTF8 = "utf-8"


def ensure_dirs() -> None:
    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)


def write_json(data: Any, path: str | Path) -> None:
    """原子写 JSON(临时文件 + rename), UTF-8 无 BOM."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding=_UTF8) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding=_UTF8) as f:
        return json.load(f)


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, encoding=_UTF8)
    os.replace(tmp, path)


def read_csv(path: str | Path, index_col: int | str | None = 0) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=index_col)
    # monitoring 面板 index 均为交易日, 统一解析为 DatetimeIndex
    try:
        df.index = pd.DatetimeIndex(df.index)
    except Exception:
        pass
    return df


# ---- 因子健康文件 ----
def health_path() -> Path:
    return MONITOR_DIR / "factor_health.json"


def save_health(data: dict) -> None:
    ensure_dirs()
    write_json(data, health_path())


def load_health() -> dict:
    return read_json(health_path(), default={})


# ---- 信号与收益面板 ----
def signals_path() -> Path:
    return SIGNALS_DIR / "factor_signals.csv"


def returns_path() -> Path:
    return SIGNALS_DIR / "daily_returns.csv"


def close_path() -> Path:
    return SIGNALS_DIR / "close.csv"


def save_signals(signals: dict[str, pd.DataFrame], returns: pd.DataFrame, close: pd.DataFrame) -> None:
    ensure_dirs()
    # 信号: {因子名: DataFrame(dates×品种)} → 单 CSV: 列 = MultiIndex? 简化为每因子一个文件
    for name, df in signals.items():
        write_csv(df, SIGNALS_DIR / f"signal_{name}.csv")
    write_csv(returns, returns_path())
    write_csv(close, close_path())
    meta = {"factors": list(signals.keys()), "dates": [str(d.date()) for d in returns.index],
            "universe": list(returns.columns)}
    write_json(meta, SIGNALS_DIR / "signals_meta.json")


def load_signals() -> dict[str, pd.DataFrame]:
    """加载信号面板; 无数据时返回空 dict."""
    meta = read_json(SIGNALS_DIR / "signals_meta.json", default={})
    factors = meta.get("factors", [])
    out: dict[str, pd.DataFrame] = {}
    for name in factors:
        p = SIGNALS_DIR / f"signal_{name}.csv"
        df = read_csv(p)
        if not df.empty:
            out[name] = df
    return out


def load_returns() -> pd.DataFrame:
    return read_csv(returns_path())


def load_close() -> pd.DataFrame:
    return read_csv(close_path())
