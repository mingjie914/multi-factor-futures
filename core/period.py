"""周期抽象层 — 统一的周期语义定义.

本模块定义了因子框架的"周期"概念, 解决以下问题:
1. slug 中的 `5d` 后缀实际语义为"5个周期" (bar数), 而非"5天"
2. 1/5/15/30/60 分钟研究需要统一的周期上下文
3. 持有期 (holding_period) 同样是"周期数"语义

核心概念:
- PeriodUnit: 周期单位 (daily / 15min / 30min / hourly)
- PeriodContext: 周期上下文, 封装周期单位 + 年化因子
- parse_slug_window: 从因子名解析周期数
- holding_periods_for_window: 推荐持有期列表
- parse_holding_periods: 解析 --periods 命令行参数

向后兼容:
- 不传 PeriodContext 时, 所有行为与现有完全一致
- 现有 "5d" slug 仍按"5个日度周期"解释
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional

import numpy as np


class PeriodUnit(str, Enum):
    """周期单位枚举.

    每个枚举值对应一个字符串标识, 用于配置文件和 SPEC 字典中的 `frequency` 字段.
    继承 str 以便直接序列化和与字符串比较.
    """

    DAILY = "daily"          # 日度 (1个交易日 = 1个bar)
    MINUTE_1 = "1min"
    MINUTE_5 = "5min"
    MINUTE_15 = "15min"      # 15分钟 (国内期货日盘 ~4小时 = 16个bar/天)
    MINUTE_30 = "30min"      # 30分钟 (8个bar/天)
    HOURLY = "hourly"        # 小时 (4个bar/天)


# 各周期单位的元数据: 每交易日bar数, 每年bar数
# 年化因子 = 每交易日bar数 × 252 (假设252个交易日/年)
_PERIOD_META = {
    PeriodUnit.DAILY: {"bars_per_day": 1, "bars_per_year": 252},
    PeriodUnit.MINUTE_1: {"bars_per_day": 240, "bars_per_year": 240 * 252},
    PeriodUnit.MINUTE_5: {"bars_per_day": 48, "bars_per_year": 48 * 252},
    PeriodUnit.MINUTE_15: {"bars_per_day": 16, "bars_per_year": 16 * 252},
    PeriodUnit.MINUTE_30: {"bars_per_day": 8, "bars_per_year": 8 * 252},
    PeriodUnit.HOURLY: {"bars_per_day": 4, "bars_per_year": 4 * 252},
}


@dataclass(frozen=True)
class PeriodContext:
    """周期上下文 — 封装周期单位和年化因子.

    用于在因子计算和回测中传递周期信息. 默认为日度.

    Attributes:
        unit: 周期单位
        bars_per_day: 每交易日的bar数
        bars_per_year: 每年的bar数 (用于年化计算, 如波动率年化)
    """

    unit: PeriodUnit = PeriodUnit.DAILY
    bars_per_day: int = 1
    bars_per_year: int = 252

    @classmethod
    def from_unit(cls, unit: PeriodUnit) -> "PeriodContext":
        """从周期单位构造上下文.

        Args:
            unit: 周期单位

        Returns:
            PeriodContext 实例 (使用预定义的 bars_per_day / bars_per_year)
        """
        meta = _PERIOD_META[unit]
        return cls(
            unit=unit,
            bars_per_day=meta["bars_per_day"],
            bars_per_year=meta["bars_per_year"],
        )

    @classmethod
    def from_string(cls, freq: str) -> "PeriodContext":
        """从字符串构造上下文.

        Args:
            freq: 频率字符串 (如 "daily", "15min", "30min", "hourly")

        Returns:
            PeriodContext 实例

        Raises:
            ValueError: 不支持的频率字符串
        """
        aliases = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "60m": "hourly",
            "60min": "hourly",
            "1h": "hourly",
            "daily_intraday": "daily",
        }
        normalised = aliases.get(str(freq).lower(), str(freq).lower())
        try:
            unit = PeriodUnit(normalised)
        except ValueError as e:
            raise ValueError(
                f"不支持的频率: {freq!r}, 支持的值: "
                f"{[u.value for u in PeriodUnit]}"
            ) from e
        return cls.from_unit(unit)

    @property
    def is_daily(self) -> bool:
        """是否为日度周期."""
        return self.unit == PeriodUnit.DAILY


# slug 中窗口数字的正则: 匹配 `_5d`, `_10d`, `_20d` (日度) 或 `_16p`, `_80p` (分钟)
# 注意: 必须是非贪婪且锚定到后缀分隔符或字符串末尾, 避免误匹配
_SLUG_WINDOW_RE = re.compile(r"_(\d+)(d|p)(?:_|$)")


def parse_slug_window(slug: str) -> Optional[str]:
    """从因子 slug 解析窗口标签.

    示例:
        "ts_rank_close_5d_smooth" → "5d"
        "macd_diff_10d_z" → "10d"
        "momentum_20d" → "20d"
        "basis_zscore_20d" → "20d"
        "intraday_momentum_5d" → "5d"
        "oi_change_20d" → "20d"
        "return_16p_z" → "16p"   (15min 频率, 16个bar)
        "realized_vol_80p_rank" → "80p"
        "rsi_14" → None  (无 `d`/`p` 后缀)

    Args:
        slug: 因子名

    Returns:
        窗口标签 (如 "5d", "10d", "20d", "16p", "80p") 或 None (无匹配)
    """
    m = _SLUG_WINDOW_RE.search(slug)
    return f"{m.group(1)}{m.group(2)}" if m else None


# 默认的窗口 → 持有期映射 (日度场景)
# `p` 只表示 bar 数，不隐含分钟频率；持有期也按当前频率的 bar 数解释。
_DEFAULT_HOLDING_PERIODS = {
    # 日度窗口
    "5d": [3, 5, 10],
    "10d": [5, 10, 20],
    "20d": [10, 20, 40],
    # 分钟级窗口 (15min 频率, 16bar=1天)
    "16p": [1, 3, 5],
    "80p": [3, 5, 10],
    "240p": [5, 10, 20],
}
_DEFAULT_FALLBACK_PERIODS = [1, 5, 10, 20]


def holding_periods_for_window(
    window: str,
    period_ctx: Optional[PeriodContext] = None,
) -> List[int]:
    """根据计算窗口推荐持有期列表.

    返回值始终是当前 frequency 下的 bar 数，不自动换算为交易日。

    Args:
        window: 窗口标签 (如 "5d", "10d", "20d", "16p", "80p", "240p", "other")
        period_ctx: 周期上下文 (可选, 默认 None 表示日度)

    Returns:
        持有期列表 (周期数, 已排序去重, **新列表副本**)
    """
    # 返回副本以避免调用方修改模块级常量
    periods = _DEFAULT_HOLDING_PERIODS.get(window)
    if periods is None:
        return list(_DEFAULT_FALLBACK_PERIODS)
    return list(periods)


def parse_holding_periods(arg: Optional[str]) -> Optional[List[int]]:
    """解析 --periods 命令行参数.

    支持以下格式:
        None 或空 → None (表示使用窗口推断的默认持有期)
        "1,5,10,20" → [1, 5, 10, 20]
        "1,5,10,20,40" → [1, 5, 10, 20, 40]
        "5" → [5]

    Args:
        arg: 命令行参数字符串 (逗号分隔)

    Returns:
        持有期列表 (已排序去重), 或 None (使用默认)

    Raises:
        ValueError: 参数格式错误或包含非正整数
    """
    if not arg:
        return None
    try:
        periods = [int(x.strip()) for x in arg.split(",") if x.strip()]
    except ValueError as e:
        raise ValueError(
            f"无效的 --periods 参数: {arg!r}, "
            f"应为逗号分隔的整数列表, 例如 '1,5,10,20'"
        ) from e
    if not periods:
        return None
    if any(p <= 0 for p in periods):
        raise ValueError(
            f"--periods 参数必须为正整数, 收到: {arg!r}"
        )
    return sorted(set(periods))


def parse_period_values(values: object) -> List[int]:
    """Parse pipe/comma separated or iterable positive horizon values."""
    if values is None:
        return []
    if isinstance(values, str):
        raw = values.replace(",", "|").split("|")
    elif isinstance(values, Iterable):
        raw = list(values)
    else:
        raw = [values]
    parsed = []
    for value in raw:
        try:
            period = int(float(value))
        except (TypeError, ValueError):
            continue
        if period > 0:
            parsed.append(period)
    return sorted(set(parsed))


def approved_horizon_ensemble(
    best_period: object,
    valid_periods: object,
    *,
    enabled: bool,
    neighbor_count: int,
    max_log_distance: float,
) -> List[int]:
    """Return the best horizon plus approved neighbours only."""
    best = parse_period_values([best_period])
    if not best:
        return []
    anchor = best[0]
    if not enabled:
        return [anchor]
    candidates = parse_period_values(valid_periods)
    if anchor not in candidates:
        candidates.append(anchor)
    ranked = sorted(
        set(candidates),
        key=lambda period: (
            abs(np.log(float(period) / float(anchor))), period,
        ),
    )
    allowed = [
        period for period in ranked
        if abs(np.log(float(period) / float(anchor))) <= float(max_log_distance)
    ]
    return allowed[:1 + max(int(neighbor_count), 0)] or [anchor]
