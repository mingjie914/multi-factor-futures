from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np  # noqa: F401

from core.types import WeightVector
from core.registry import register
from core.interfaces import Constraint, ConstraintContext


@register("constraint", "long_only")
class LongOnlyConstraint(Constraint):
    name = "long_only"

    def apply(self, problem, variables: Dict, context: "ConstraintContext") -> None:
        w = variables.get("w")
        if w is not None:
            variables.setdefault("_constraints", []).append(w >= 0)


@register("constraint", "weight_sum")
class WeightSumConstraint(Constraint):
    name = "weight_sum"

    def __init__(self, target: float = 1.0):
        self.target = target

    def apply(self, problem, variables: Dict, context: "ConstraintContext") -> None:
        import cvxpy as cp

        w = variables.get("w")
        if w is not None:
            variables.setdefault("_constraints", []).append(cp.sum(w) == self.target)


@register("constraint", "net_exposure")
class NetExposureConstraint(Constraint):
    """净敞口范围约束: lower <= sum(w) <= upper.

    用于多空组合, 允许净多头/净空头/市场中立.
    例: lower=-0.5, upper=0.5 → 净敞口在 [-50%, +50%] 范围内.
    """

    name = "net_exposure"

    def __init__(self, lower: float = -0.5, upper: float = 0.5):
        self.lower = lower
        self.upper = upper

    def apply(self, problem, variables: Dict, context: "ConstraintContext") -> None:
        import cvxpy as cp

        w = variables.get("w")
        if w is not None:
            cs = variables.setdefault("_constraints", [])
            cs.append(cp.sum(w) >= self.lower)
            cs.append(cp.sum(w) <= self.upper)


@register("constraint", "position_limit")
class PositionLimitConstraint(Constraint):
    name = "position_limit"

    def __init__(self, lower: float = 0.0, upper: float = 0.05):
        self.lower = lower
        self.upper = upper

    def apply(self, problem, variables: Dict, context: "ConstraintContext") -> None:
        w = variables.get("w")
        if w is not None:
            variables.setdefault("_constraints", []).extend([w >= self.lower, w <= self.upper])


@register("constraint", "turnover")
class TurnoverConstraint(Constraint):
    name = "turnover"

    def __init__(self, limit: float = 0.30):
        self.limit = limit

    def apply(self, problem, variables: Dict, context: "ConstraintContext") -> None:
        import cvxpy as cp

        w = variables.get("w")
        if w is not None and context is not None:
            w_prev = context.current_weights
            if w_prev is not None and len(w_prev) > 0:
                # reindex 到当前 universe, 动态 universe 下品种上市日期不同
                prev_arr = w_prev.reindex(context.universe).fillna(0.0).values
                variables.setdefault("_constraints", []).append(
                    cp.norm1(w - prev_arr) <= self.limit
                )


@register("constraint", "margin")
class MarginConstraint(Constraint):
    name = "margin"

    def __init__(self, limit: float = 0.5):
        self.limit = limit

    def apply(self, problem, variables: Dict, context: "ConstraintContext") -> None:
        pass  # 实现需要合约乘数和保证金率信息，暂略


@register("constraint", "leverage")
class LeverageConstraint(Constraint):
    """总杠杆约束: sum(|w|) <= limit (gross exposure).

    区别于 weight_sum (净敞口 = sum(w)):
    - weight_sum=1 → 净多头 100%
    - leverage=3 → 多+空绝对值之和 <= 300%
    例: 多 2.0 + 空 -1.0 → sum=1.0 (净多头), sum|w|=3.0 (总杠杆 3 倍)

    支持 vol_target (波动率目标):
    - 设定 vol_target 后, 实际杠杆 = min(limit, target_vol / realized_vol)
    - realized_vol 由 ConstraintContext 传入 (近期组合年化波动率)
    - 低波环境自动加杠杆, 高波环境自动降杠杆
    """

    name = "leverage"

    def __init__(self, limit: float = 3.0, vol_target: float = 0.0):
        self.limit = limit
        self.vol_target = vol_target  # 目标年化波动率, 0 表示不启用

    def apply(self, problem, variables: Dict, context: "ConstraintContext") -> None:
        import cvxpy as cp

        w = variables.get("w")
        if w is None:
            return

        # 动态杠杆: 根据 vol_target 和 realized_vol 调整上限
        effective_limit = self.limit
        if self.vol_target > 0 and context is not None:
            realized_vol = getattr(context, "realized_vol", 0.0)
            if realized_vol > 1e-6:
                # 目标杠杆 = 目标波动率 / 已实现波动率, 不超过硬上限
                dynamic_limit = self.vol_target / realized_vol
                effective_limit = min(self.limit, dynamic_limit)
                # 下限保护: 至少保留 0.5 倍杠杆, 避免极端降杠杆
                effective_limit = max(effective_limit, 0.5)

        variables.setdefault("_constraints", []).append(
            cp.norm1(w) <= effective_limit
        )


@register("constraint", "sector_exposure")
class SectorExposureConstraint(Constraint):
    """板块暴露约束: 每个板块的净敞口 |sum(w_sector)| <= limit.

    用于控制单板块集中度风险。板块映射来自统一 SECTOR_MAP，
    其中股指/国债、有色/贵金属分别计算暴露。
    例: limit=0.15 → 任一板块净敞口不超过 ±15%.
    """

    name = "sector_exposure"

    def __init__(self, limit: float = 0.15):
        self.limit = limit

    def apply(self, problem, variables: Dict, context: "ConstraintContext") -> None:
        import cvxpy as cp

        w = variables.get("w")
        if w is None or context is None:
            return

        # 从 cross_commodity 获取板块映射 (延迟导入避免循环依赖)
        try:
            from core.sectors import SECTOR_MAP
        except ImportError:
            return

        universe = context.universe
        sectors = {}
        for i, sym in enumerate(universe):
            sec = SECTOR_MAP.get(str(sym), "other")
            sectors.setdefault(sec, []).append(i)

        cs = variables.setdefault("_constraints", [])
        for sector, indices in sectors.items():
            if len(indices) == 0:
                continue
            sector_w = w[indices]
            cs.append(cp.sum(sector_w) >= -self.limit)
            cs.append(cp.sum(sector_w) <= self.limit)


@register("constraint", "drawdown_control")
class DrawdownControlConstraint(Constraint):
    """回撤控制约束: 当组合回撤超过阈值时, 降杠杆.

    机制 (CTA基金常用):
    - 回撤 < warning_dd: 正常运行 (leverage_limit 不变)
    - warning_dd <= 回撤 < critical_dd: 降杠杆到 leverage_limit * 0.5
    - 回撤 >= critical_dd: 清仓 (leverage_limit = 0.1, 仅保留最小仓位)

    通过动态调整 leverage 上限实现, 配合 LeverageConstraint 使用.
    在 ConstraintContext.realized_vol 同级传入 current_drawdown.

    参数:
    - warning_dd: 预警回撤阈值 (如 -0.05 = -5%)
    - critical_dd: 临界回撤阈值 (如 -0.10 = -10%)
    - leverage_limit: 正常状态杠杆上限
    """

    name = "drawdown_control"

    def __init__(
        self,
        warning_dd: float = -0.05,
        critical_dd: float = -0.10,
        leverage_limit: float = 2.0,
    ):
        self.warning_dd = warning_dd
        self.critical_dd = critical_dd
        self.leverage_limit = leverage_limit

    def apply(self, problem, variables: Dict, context: "ConstraintContext") -> None:
        import cvxpy as cp

        w = variables.get("w")
        if w is None:
            return

        # 从 context 获取当前回撤 (由回测引擎注入)
        current_dd = getattr(context, "current_drawdown", 0.0) if context else 0.0

        # 根据回撤水平确定有效杠杆上限
        if current_dd <= self.critical_dd:
            # 临界回撤: 清仓, 仅保留最小仓位
            effective_limit = 0.1
        elif current_dd <= self.warning_dd:
            # 预警回撤: 降杠杆50%
            effective_limit = self.leverage_limit * 0.5
        else:
            # 正常状态
            effective_limit = self.leverage_limit

        variables.setdefault("_constraints", []).append(
            cp.norm1(w) <= effective_limit
        )
