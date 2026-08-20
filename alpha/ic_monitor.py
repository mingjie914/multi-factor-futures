"""滚动 IC 监控 + 自动失效剔除.

在 walk-forward 回测中, 因子的有效性会随时间衰减.
本模块跟踪每个因子的滚动 IC, 当因子近期 IC 持续低于阈值时自动剔除,
避免 "死因子" 稀释 alpha 信号.

集成方式:
    1. 回测引擎在每次重训后调用 ic_monitor.update() 更新 IC 历史
    2. 预测时调用 ic_monitor.get_active_factors() 过滤失效因子

参考: Grinold & Kahn, Active Portfolio Management — IC 衰减监控.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd


class ICMonitor:
    """滚动 IC 监控器.

    跟踪每个因子的滚动 IC, 当近期 IC 持续低于阈值时自动剔除.

    Args:
        window: 滚动 IC 计算窗口 (天数), 推荐 60.
        min_ic: 最小有效 |IC| 阈值, 低于此值视为无效.
        decay_tolerance: 连续低于阈值的容忍天数, 超过后才剔除.
            避免短期噪声导致误剔.
        reactivation_ic: 重新激活的 IC 阈值, 高于此值可恢复.
    """

    def __init__(
        self,
        window: int = 60,
        min_ic: float = 0.02,
        decay_tolerance: int = 20,
        reactivation_ic: float = 0.03,
        min_cross_section: int = 3,
    ):
        if int(window) <= 0 or int(decay_tolerance) <= 0:
            raise ValueError("IC window and decay_tolerance must be positive")
        if not np.isfinite([min_ic, reactivation_ic]).all():
            raise ValueError("IC thresholds must be finite")
        if int(min_cross_section) < 3:
            raise ValueError("min_cross_section must be at least 3")
        self._window = int(window)
        self._min_ic = float(min_ic)
        self._decay_tolerance = int(decay_tolerance)
        self._reactivation_ic = float(reactivation_ic)
        self._min_cross_section = int(min_cross_section)

        # 滚动 IC 历史: {factor_name: pd.Series(index=date, values=ic)}
        self._ic_history: Dict[str, pd.Series] = {}
        # 连续低于阈值的天数计数
        self._decay_counter: Dict[str, int] = {}
        # 当前被剔除的因子集合
        self._inactive_factors: Set[str] = set()
        # 最后更新日期
        self._last_date: Optional[pd.Timestamp] = None

    def update(
        self,
        factor_exposures: Dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
    ) -> None:
        """更新因子 IC 历史.

        计算训练窗口内每个因子的截面 IC (Spearman 秩相关),
        追加到滚动 IC 历史中, 并更新因子活跃状态.

        Args:
            factor_exposures: {factor_name: DataFrame(dates × tickers)}
            forward_returns: DataFrame(dates × tickers) 未来收益
        """
        if not factor_exposures or forward_returns.empty:
            return

        # Refit windows overlap heavily. Only append dates not seen before and
        # reuse the return ranks across factors with the same asset columns.
        common_dates = forward_returns.index
        pending: Dict[tuple, dict] = {}
        for name, fmat in factor_exposures.items():
            if fmat is None or fmat.empty:
                continue
            existing = self._ic_history.get(name)
            new_dates = (
                common_dates
                if existing is None
                else common_dates.difference(existing.index, sort=False)
            )
            if len(new_dates) > 0:
                common_cols = fmat.columns.intersection(forward_returns.columns)
                key = (tuple(pd.DatetimeIndex(new_dates)), tuple(common_cols))
                group = pending.setdefault(
                    key,
                    {"dates": new_dates, "columns": common_cols, "entries": []},
                )
                group["entries"].append((name, fmat, existing))

        for group in pending.values():
            new_dates = group["dates"]
            common_cols = group["columns"]
            entries = group["entries"]
            daily_by_factor = self._compute_daily_ic_batch(
                {name: frame.reindex(index=new_dates, columns=common_cols)
                 for name, frame, _ in entries},
                forward_returns.reindex(index=new_dates, columns=common_cols),
                min_cross_section=self._min_cross_section,
            )
            for name, _, existing in entries:
                daily_ics = daily_by_factor.get(name, pd.Series(dtype=float))
                if existing is None:
                    self._ic_history[name] = daily_ics
                elif not daily_ics.empty:
                    self._ic_history[name] = pd.concat(
                        [existing, daily_ics]
                    ).sort_index()

        for name, fmat in factor_exposures.items():
            if fmat is None or fmat.empty:
                continue
            self._update_decay_counter(name)

        if common_dates is not None and len(common_dates) > 0:
            self._last_date = common_dates[-1]

    def _compute_daily_ic(
        self,
        factor_mat: pd.DataFrame,
        fwd_returns: pd.DataFrame,
    ) -> pd.Series:
        """Compute pairwise-complete daily cross-sectional Spearman IC."""
        common_cols = factor_mat.columns.intersection(fwd_returns.columns)
        dates = factor_mat.index.intersection(fwd_returns.index, sort=False)
        if len(common_cols) < self._min_cross_section or len(dates) == 0:
            return pd.Series(np.nan, index=dates, dtype=float)
        return self._compute_daily_ic_batch(
            {"factor": factor_mat.reindex(index=dates, columns=common_cols)},
            fwd_returns.reindex(index=dates, columns=common_cols),
            min_cross_section=self._min_cross_section,
        )["factor"]

    @staticmethod
    def _compute_daily_ic_batch(
        factor_mats: Dict[str, pd.DataFrame],
        fwd_returns: pd.DataFrame,
        *,
        min_cross_section: int = 3,
    ) -> Dict[str, pd.Series]:
        """Compute pairwise-complete aligned ICs with two rank operations."""
        if int(min_cross_section) < 3:
            raise ValueError("min_cross_section must be at least 3")
        if not factor_mats:
            return {}
        names = list(factor_mats)
        dates = fwd_returns.index
        columns = fwd_returns.columns
        if len(columns) < min_cross_section:
            return {
                name: pd.Series(np.nan, index=dates, dtype=float)
                for name in names
            }
        n_dates, n_assets = fwd_returns.shape
        block = pd.concat(
            [factor_mats[name].reindex(index=dates, columns=columns) for name in names],
            keys=names,
            names=["factor", "date"],
        )
        return_block = pd.concat(
            [fwd_returns for _ in names], keys=names, names=["factor", "date"]
        )
        pairwise = block.notna() & return_block.notna()
        factor_ranked = block.where(pairwise).rank(axis=1).to_numpy(dtype=float).reshape(
            len(names), n_dates, n_assets
        )
        return_ranked = return_block.where(pairwise).rank(axis=1).to_numpy(
            dtype=float
        ).reshape(len(names), n_dates, n_assets)
        count = np.sum(np.isfinite(factor_ranked), axis=2, keepdims=True)
        factor_mean = np.divide(
            np.nansum(factor_ranked, axis=2, keepdims=True), count,
            out=np.zeros(count.shape, dtype=float), where=count > 0,
        )
        return_mean = np.divide(
            np.nansum(return_ranked, axis=2, keepdims=True), count,
            out=np.zeros(count.shape, dtype=float), where=count > 0,
        )
        factor_centered = factor_ranked - factor_mean
        return_centered = return_ranked - return_mean
        numerator = np.nansum(factor_centered * return_centered, axis=2)
        denominator = np.sqrt(
            np.nansum(factor_centered**2, axis=2)
            * np.nansum(return_centered**2, axis=2)
        )
        ic = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan, dtype=float),
            where=(denominator > 0) & (count[:, :, 0] >= min_cross_section),
        )
        return {
            name: pd.Series(ic[index], index=dates)
            for index, name in enumerate(names)
        }

    def _update_decay_counter(self, factor_name: str) -> None:
        """更新因子的衰减计数器, 判断是否应剔除/恢复."""
        history = self._ic_history.get(factor_name)
        if history is None or len(history) == 0:
            return

        valid = history.dropna()
        if valid.empty:
            self._inactive_factors.add(factor_name)
            self._decay_counter[factor_name] = self._decay_tolerance
            return
        rolling_ic = valid.rolling(self._window, min_periods=1).mean()
        recent_ic = float(rolling_ic.iloc[-1])
        below = rolling_ic.abs().lt(self._min_ic)
        trailing_below = 0
        for value in reversed(below.tolist()):
            if not value:
                break
            trailing_below += 1
        self._decay_counter[factor_name] = trailing_below

        if factor_name in self._inactive_factors:
            # 已剔除: 检查是否恢复
            if abs(recent_ic) >= self._reactivation_ic:
                self._inactive_factors.discard(factor_name)
                self._decay_counter[factor_name] = 0
        else:
            if trailing_below >= self._decay_tolerance:
                self._inactive_factors.add(factor_name)

    def get_active_factors(self, all_factors: List[str]) -> List[str]:
        """返回当前活跃的因子列表 (剔除失效因子)."""
        return [f for f in all_factors if f not in self._inactive_factors]

    def get_health_report(self) -> Dict[str, dict]:
        """返回因子健康报告."""
        report = {}
        for name, history in self._ic_history.items():
            recent = history.dropna().tail(self._window)
            recent_ic = float(recent.mean()) if not recent.empty else np.nan
            report[name] = {
                "recent_ic": recent_ic,
                "is_active": name not in self._inactive_factors,
                "decay_count": self._decay_counter.get(name, 0),
                "n_obs": int(history.notna().sum()),
            }
        return report

    @property
    def inactive_factors(self) -> Set[str]:
        """当前被剔除的因子集合."""
        return self._inactive_factors.copy()

    @property
    def last_date(self) -> Optional[pd.Timestamp]:
        return self._last_date
