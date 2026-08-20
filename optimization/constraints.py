from __future__ import annotations

from typing import Dict

import numpy as np  # noqa: F401
import pandas as pd

from core.registry import register
from core.interfaces import Constraint
from core.sectors import sector_for


def _weight_variable(variables: Dict):
    weight = variables.get("w")
    if weight is None:
        raise KeyError("optimizer did not provide the 'w' weight variable")
    return weight


def turnover_transition(
    current_weights, universe
) -> tuple[pd.Series, float]:
    """Return active prior weights and mandatory exits outside ``universe``."""
    active = pd.Index(universe)
    if active.has_duplicates:
        raise ValueError("turnover universe must contain unique instruments")
    current = (
        pd.Series(dtype=float)
        if current_weights is None
        else pd.Series(current_weights, dtype=float)
    )
    if current.index.has_duplicates:
        raise ValueError("current_weights must contain unique instruments")
    if not np.isfinite(current.to_numpy(dtype=float)).all():
        raise ValueError("current_weights must be finite")
    previous_active = current.reindex(active).fillna(0.0)
    forced_exit = float(current.loc[~current.index.isin(active)].abs().sum())
    return previous_active, forced_exit


@register("constraint", "long_only")
class LongOnlyConstraint(Constraint):
    name = "long_only"

    def apply(self, problem, variables: Dict, context: "ConstraintContext") -> None:
        w = _weight_variable(variables)
        variables.setdefault("_constraints", []).append(w >= 0)


@register("constraint", "weight_sum")
class WeightSumConstraint(Constraint):
    name = "weight_sum"

    def __init__(self, target: float = 1.0):
        self.target = float(target)
        if not np.isfinite(self.target):
            raise ValueError("weight-sum target must be finite")

    def apply(self, problem, variables: Dict, context: "ConstraintContext") -> None:
        import cvxpy as cp

        w = _weight_variable(variables)
        variables.setdefault("_constraints", []).append(cp.sum(w) == self.target)


@register("constraint", "net_exposure")
class NetExposureConstraint(Constraint):
    """净敞口范围约束: lower <= sum(w) <= upper.

    用于多空组合, 允许净多头/净空头/市场中立.
    例: lower=-0.5, upper=0.5 → 净敞口在 [-50%, +50%] 范围内.
    """

    name = "net_exposure"

    def __init__(self, lower: float = -0.5, upper: float = 0.5):
        self.lower = float(lower)
        self.upper = float(upper)
        if not np.isfinite([self.lower, self.upper]).all() or self.lower > self.upper:
            raise ValueError("net-exposure bounds must be finite and ordered")

    def apply(self, problem, variables: Dict, context: "ConstraintContext") -> None:
        import cvxpy as cp

        w = _weight_variable(variables)
        cs = variables.setdefault("_constraints", [])
        cs.append(cp.sum(w) >= self.lower)
        cs.append(cp.sum(w) <= self.upper)


@register("constraint", "position_limit")
class PositionLimitConstraint(Constraint):
    name = "position_limit"

    def __init__(self, lower: float = 0.0, upper: float = 0.05):
        self.lower = float(lower)
        self.upper = float(upper)
        if not np.isfinite([self.lower, self.upper]).all() or self.lower > self.upper:
            raise ValueError("position bounds must be finite and ordered")

    def apply(self, problem, variables: Dict, context: "ConstraintContext") -> None:
        w = _weight_variable(variables)
        variables.setdefault("_constraints", []).extend([w >= self.lower, w <= self.upper])


@register("constraint", "turnover")
class TurnoverConstraint(Constraint):
    name = "turnover"

    def __init__(self, limit: float = 0.30):
        self.limit = float(limit)
        if not np.isfinite(self.limit) or self.limit < 0.0:
            raise ValueError("turnover limit must be finite and non-negative")

    def apply(self, problem, variables: Dict, context: "ConstraintContext") -> None:
        import cvxpy as cp

        w = _weight_variable(variables)
        if context is None:
            raise ValueError("turnover constraint requires optimization context")
        w_prev = context.current_weights
        if w_prev is not None and len(w_prev) > 0:
            previous, forced_exit = turnover_transition(
                w_prev, context.universe
            )
            discretionary_budget = max(self.limit - forced_exit, 0.0)
            variables["_forced_exit_turnover"] = forced_exit
            variables.setdefault("_constraints", []).append(
                cp.norm1(w - previous.to_numpy(dtype=float))
                <= discretionary_budget
            )


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
        self.limit = float(limit)
        self.vol_target = float(vol_target)  # 目标年化波动率, 0 表示不启用
        if not np.isfinite(self.limit) or self.limit <= 0.0:
            raise ValueError("leverage limit must be finite and positive")
        if not np.isfinite(self.vol_target) or self.vol_target < 0.0:
            raise ValueError("vol_target must be finite and non-negative")

    def apply(self, problem, variables: Dict, context: "ConstraintContext") -> None:
        import cvxpy as cp

        w = _weight_variable(variables)

        # 动态杠杆: 根据 vol_target 和 realized_vol 调整上限
        effective_limit = self.limit
        if self.vol_target > 0:
            if context is None:
                raise ValueError("vol-target leverage constraint requires optimization context")
            realized_vol = float(getattr(context, "realized_vol", 0.0))
            if not np.isfinite(realized_vol) or realized_vol < 0.0:
                raise ValueError("realized_vol must be finite and non-negative")
            if realized_vol > 1e-6:
                # 目标杠杆 = 目标波动率 / 已实现波动率, 不超过硬上限
                dynamic_limit = self.vol_target / realized_vol
                # 下限保护: 至少保留 0.5 倍杠杆, 避免极端降杠杆
                # The configured hard limit always dominates that floor.
                effective_limit = min(self.limit, max(dynamic_limit, 0.5))

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
        self.limit = float(limit)
        if not np.isfinite(self.limit) or self.limit < 0.0:
            raise ValueError("sector-exposure limit must be finite and non-negative")

    def apply(self, problem, variables: Dict, context: "ConstraintContext") -> None:
        import cvxpy as cp

        w = _weight_variable(variables)
        if context is None:
            raise ValueError("sector-exposure constraint requires optimization context")

        universe = context.universe
        sectors = {}
        for i, sym in enumerate(universe):
            sec = sector_for(str(sym))
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
        if not np.isfinite([warning_dd, critical_dd, leverage_limit]).all():
            raise ValueError("drawdown-control parameters must be finite")
        if critical_dd >= warning_dd or warning_dd > 0.0:
            raise ValueError(
                "drawdown thresholds require critical_dd < warning_dd <= 0"
            )
        if leverage_limit <= 0.0:
            raise ValueError("drawdown leverage limit must be positive")

    def apply(self, problem, variables: Dict, context: "ConstraintContext") -> None:
        import cvxpy as cp

        w = _weight_variable(variables)

        # 从 context 获取当前回撤 (由回测引擎注入)
        if context is None:
            raise ValueError("drawdown-control constraint requires optimization context")
        current_dd = float(getattr(context, "current_drawdown", 0.0))
        if not np.isfinite(current_dd) or current_dd > 0.0:
            raise ValueError("current_drawdown must be finite and non-positive")

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
