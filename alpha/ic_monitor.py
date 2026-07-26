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
    ):
        self._window = window
        self._min_ic = min_ic
        self._decay_tolerance = decay_tolerance
        self._reactivation_ic = reactivation_ic

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
        *,
        return_ranked: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """逐日计算截面 Spearman IC.

        向量化实现: 对每行 (截面) 计算 rank 相关.
        """
        # 对齐列
        common_cols = factor_mat.columns.intersection(fwd_returns.columns)
        if len(common_cols) < 3:
            return pd.Series(dtype=float)

        f = factor_mat[common_cols]

        # Rank with pandas to preserve average-tie semantics, then perform all
        # row reductions in NumPy. This avoids thousands of tiny DataFrame
        # arithmetic and reduction objects during walk-forward refits.
        f_ranked = f.rank(axis=1).to_numpy(dtype=float)
        r_ranked = (
            return_ranked.reindex(index=f.index, columns=common_cols).to_numpy(
                dtype=float
            )
            if return_ranked is not None
            else fwd_returns[common_cols].rank(axis=1).to_numpy(dtype=float)
        )
        f_count = np.sum(np.isfinite(f_ranked), axis=1, keepdims=True)
        r_count = np.sum(np.isfinite(r_ranked), axis=1, keepdims=True)
        f_mean = np.divide(
            np.nansum(f_ranked, axis=1, keepdims=True),
            f_count,
            out=np.zeros((len(f_ranked), 1), dtype=float),
            where=f_count > 0,
        )
        r_mean = np.divide(
            np.nansum(r_ranked, axis=1, keepdims=True),
            r_count,
            out=np.zeros((len(r_ranked), 1), dtype=float),
            where=r_count > 0,
        )
        f_c = f_ranked - f_mean
        r_c = r_ranked - r_mean
        numerator = np.nansum(f_c * r_c, axis=1)
        denominator = np.sqrt(
            np.nansum(f_c**2, axis=1) * np.nansum(r_c**2, axis=1)
        )
        ic = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan, dtype=float),
            where=denominator != 0,
        )
        return pd.Series(ic, index=f.index)

    @staticmethod
    def _compute_daily_ic_batch(
        factor_mats: Dict[str, pd.DataFrame],
        fwd_returns: pd.DataFrame,
    ) -> Dict[str, pd.Series]:
        """Compute aligned factor ICs with one pandas rank operation."""
        if not factor_mats or fwd_returns.shape[1] < 3:
            return {name: pd.Series(dtype=float) for name in factor_mats}
        names = list(factor_mats)
        dates = fwd_returns.index
        n_dates, n_assets = fwd_returns.shape
        block = pd.concat(
            [factor_mats[name] for name in names],
            keys=names,
            names=["factor", "date"],
        )
        factor_ranked = block.rank(axis=1).to_numpy(dtype=float).reshape(
            len(names), n_dates, n_assets
        )
        return_ranked = fwd_returns.rank(axis=1).to_numpy(dtype=float)
        factor_count = np.sum(np.isfinite(factor_ranked), axis=2, keepdims=True)
        return_count = np.sum(np.isfinite(return_ranked), axis=1, keepdims=True)
        factor_mean = np.divide(
            np.nansum(factor_ranked, axis=2, keepdims=True),
            factor_count,
            out=np.zeros(factor_count.shape, dtype=float),
            where=factor_count > 0,
        )
        return_mean = np.divide(
            np.nansum(return_ranked, axis=1, keepdims=True),
            return_count,
            out=np.zeros(return_count.shape, dtype=float),
            where=return_count > 0,
        )
        factor_centered = factor_ranked - factor_mean
        return_centered = return_ranked - return_mean
        numerator = np.nansum(
            factor_centered * return_centered[None, :, :], axis=2
        )
        denominator = np.sqrt(
            np.nansum(factor_centered**2, axis=2)
            * np.nansum(return_centered**2, axis=1)[None, :]
        )
        ic = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan, dtype=float),
            where=denominator != 0,
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

        # 取最近 window 天的 IC
        recent = history.tail(self._window)
        if len(recent) == 0:
            return

        # 近期平均 IC
        recent_ic = float(recent.mean())

        if factor_name in self._inactive_factors:
            # 已剔除: 检查是否恢复
            if abs(recent_ic) >= self._reactivation_ic:
                self._inactive_factors.discard(factor_name)
                self._decay_counter[factor_name] = 0
        else:
            # 活跃: 检查是否需要剔除
            if abs(recent_ic) < self._min_ic:
                self._decay_counter[factor_name] = (
                    self._decay_counter.get(factor_name, 0) + 1
                )
                if self._decay_counter[factor_name] >= self._decay_tolerance:
                    self._inactive_factors.add(factor_name)
            else:
                self._decay_counter[factor_name] = 0

    def get_active_factors(self, all_factors: List[str]) -> List[str]:
        """返回当前活跃的因子列表 (剔除失效因子)."""
        return [f for f in all_factors if f not in self._inactive_factors]

    def get_health_report(self) -> Dict[str, dict]:
        """返回因子健康报告."""
        report = {}
        for name, history in self._ic_history.items():
            recent = history.tail(self._window)
            recent_ic = float(recent.mean()) if len(recent) > 0 else 0.0
            report[name] = {
                "recent_ic": recent_ic,
                "is_active": name not in self._inactive_factors,
                "decay_count": self._decay_counter.get(name, 0),
                "n_obs": len(history),
            }
        return report

    @property
    def inactive_factors(self) -> Set[str]:
        """当前被剔除的因子集合."""
        return self._inactive_factors.copy()

    @property
    def last_date(self) -> Optional[pd.Timestamp]:
        return self._last_date
