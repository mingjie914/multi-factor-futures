"""OLS 截面回归收益预测模型.

用 numpy.linalg.lstsq 实现, 不需要 sklearn.

v2 改进: 因子方向自适应 (近期 IC 符号翻转).
- 旧版: 用全样本 IC 判定符号, 对短期因子无效 (不同市场环境 IC 符号会变)
- 新版: 用近期 N 日 IC 判定符号, 适应市场环境切换
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.interfaces import ReturnModel
from core.registry import register
from core.types import (
    Date,
    ExpectedReturns,
    FactorMatrix,
    ReturnMatrix,
    Universe,
    UniverseSchedule,
)
from factors.utils import stack_factors_and_returns


def _fit_linear_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    *,
    fit_intercept: bool,
    ridge_alpha: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Fit OLS/Ridge without penalizing the intercept."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    alpha = float(ridge_alpha)
    if alpha < 0:
        raise ValueError("ridge_alpha must be non-negative")
    if alpha == 0.0:
        design = (
            np.column_stack([np.ones(X.shape[0]), X]) if fit_intercept else X
        )
        fitted, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        if fit_intercept:
            return np.asarray(fitted[1:], dtype=float), float(fitted[0])
        return np.asarray(fitted, dtype=float), 0.0

    if fit_intercept:
        x_mean = X.mean(axis=0)
        y_mean = float(y.mean())
        X_centered = X - x_mean
        y_centered = y - y_mean
    else:
        x_mean = np.zeros(X.shape[1], dtype=float)
        y_mean = 0.0
        X_centered = X
        y_centered = y
    system = X_centered.T @ X_centered + alpha * np.eye(X.shape[1])
    coef = np.linalg.solve(system, X_centered.T @ y_centered)
    intercept = y_mean - float(x_mean @ coef) if fit_intercept else 0.0
    return np.asarray(coef, dtype=float), float(intercept)


def _select_ridge_alpha_time_series(
    X: np.ndarray,
    y: np.ndarray,
    dates,
    alphas: List[float],
    *,
    fit_intercept: bool,
    n_folds: int = 3,
) -> float:
    """Choose Ridge strength with ordered expanding-window validation."""
    candidates = sorted({float(value) for value in alphas if float(value) >= 0.0})
    if not candidates:
        raise ValueError("ridge_alphas must contain a non-negative candidate")
    date_values = pd.DatetimeIndex(dates)
    unique_dates = date_values.unique().sort_values()
    if len(unique_dates) < 20:
        return candidates[len(candidates) // 2]

    n_folds = max(int(n_folds), 1)
    initial = max(int(len(unique_dates) * 0.5), 10)
    remaining = len(unique_dates) - initial
    fold_size = max(remaining // n_folds, 1)
    losses = {alpha: [] for alpha in candidates}
    for fold in range(n_folds):
        train_end = initial + fold * fold_size
        valid_end = len(unique_dates) if fold == n_folds - 1 else min(
            train_end + fold_size, len(unique_dates)
        )
        if train_end >= valid_end:
            continue
        train_cutoff = unique_dates[train_end - 1]
        valid_cutoff = unique_dates[valid_end - 1]
        train_mask = date_values <= train_cutoff
        valid_mask = (date_values > train_cutoff) & (date_values <= valid_cutoff)
        if train_mask.sum() <= X.shape[1] or valid_mask.sum() == 0:
            continue
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        if fit_intercept:
            x_mean = X_train.mean(axis=0)
            y_mean = float(y_train.mean())
            X_centered = X_train - x_mean
            y_centered = y_train - y_mean
        else:
            x_mean = np.zeros(X.shape[1], dtype=float)
            y_mean = 0.0
            X_centered = X_train
            y_centered = y_train
        xtx = X_centered.T @ X_centered
        xty = X_centered.T @ y_centered
        identity = np.eye(X.shape[1])
        for alpha in candidates:
            if alpha == 0.0:
                coef, intercept = _fit_linear_coefficients(
                    X_train,
                    y_train,
                    fit_intercept=fit_intercept,
                    ridge_alpha=0.0,
                )
            else:
                coef = np.linalg.solve(xtx + alpha * identity, xty)
                intercept = (
                    y_mean - float(x_mean @ coef) if fit_intercept else 0.0
                )
            error = y_valid - (X_valid @ coef + intercept)
            losses[alpha].append(float(np.mean(error ** 2)))
    valid_scores = {
        alpha: float(np.mean(values))
        for alpha, values in losses.items()
        if values
    }
    if not valid_scores:
        return candidates[len(candidates) // 2]
    # Prefer the stronger regularizer when validation losses are exactly tied.
    return min(valid_scores, key=lambda alpha: (valid_scores[alpha], -alpha))


@register("return_model", "ols")
class OLSModel(ReturnModel):
    """Pooled OLS 截面回归收益预测模型.

    将所有日期和品种的因子暴露与收益堆叠为长表,
    用 numpy.linalg.lstsq 一次性回归, 无需 sklearn.

    方向校正 v2: 用近期 recent_ic_window 天的 IC 判定因子方向,
    而非全样本 IC. 近期 IC 能更好反映当前市场环境下的因子有效性.

    Args:
        fit_intercept: 是否拟合截距.
        recent_ic_window: 近期 IC 计算窗口 (天数).
            0 表示用全样本 IC (兼容旧版行为).
            推荐 60 (约 3 个月), 短期因子可用 20.
        direction_adaptive: 是否启用方向自适应.
            True: 用近期 IC 符号翻转 (推荐).
            False: 用全样本 IC 符号校正 (旧版行为).
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        recent_ic_window: int = 60,
        direction_adaptive: bool = True,
    ):
        self._coef: np.ndarray | None = None
        self._intercept: float = 0.0
        self._fit_intercept = fit_intercept
        self._fitted = False
        self._recent_ic_window = recent_ic_window
        self._direction_adaptive = direction_adaptive
        # 保存 fit 时的因子方向 (用于 predict 时保持一致)
        self._factor_directions: np.ndarray | None = None

    def fit(
        self,
        factors: Dict[str, FactorMatrix],
        forward_returns: ReturnMatrix,
        universe: UniverseSchedule = None,
    ) -> "OLSModel":
        """一次性池化回归: stack 所有因子和收益 → 对齐 → lstsq."""
        factor_names = sorted(factors.keys())
        if not factor_names:
            return self

        merged, factor_names, X_vals, y_vals, _ = stack_factors_and_returns(
            factors, forward_returns
        )
        if len(merged) < len(factor_names) + 5:
            return self

        if self._fit_intercept:
            X_vals = np.column_stack([np.ones(X_vals.shape[0]), X_vals])

        coef, _, _, _ = np.linalg.lstsq(X_vals, y_vals, rcond=None)

        if self._fit_intercept:
            self._intercept = float(coef[0])
            self._coef = coef[1:]
        else:
            self._intercept = 0.0
            self._coef = coef

        # 方向校正 v2: 近期 IC 符号翻转
        # 旧版用全样本 IC, 新版用近期 N 日 IC, 更适应市场环境变化
        self._factor_directions = np.ones(len(factor_names))

        if self._direction_adaptive and self._recent_ic_window > 0:
            # 近期 IC: 只用最后 recent_ic_window 天的数据
            recent_ic = self._compute_recent_ic(merged, factor_names, self._recent_ic_window)
            self._apply_direction_correction(recent_ic, factor_names)
        else:
            # 旧版: 全样本 IC 符号校正
            full_ic = self._compute_full_ic(merged, factor_names)
            self._apply_direction_correction(full_ic, factor_names)

        self._fitted = True
        return self

    def _compute_recent_ic(
        self, merged: pd.DataFrame, factor_names: list, window: int
    ) -> np.ndarray:
        """计算近期 (最后 window 天) 的因子 IC."""
        # 用日期索引取最后 window 天 (merged 是 MultiIndex(date, ticker))
        unique_dates = merged.index.get_level_values(0).unique()
        if len(unique_dates) > window:
            recent_dates = unique_dates[-window:]
            recent_data = merged.loc[recent_dates]
        else:
            recent_data = merged
        if len(recent_data) < 20:
            # 近期数据不足, 回退到全样本
            return self._compute_full_ic(merged, factor_names)

        X_raw = recent_data[factor_names].values
        y_raw = recent_data["fwd_ret"].values
        return self._compute_ic_vector(X_raw, y_raw, len(factor_names))

    def _compute_full_ic(
        self, merged: pd.DataFrame, factor_names: list
    ) -> np.ndarray:
        """计算全样本因子 IC."""
        X_raw = merged[factor_names].values
        y_raw = merged["fwd_ret"].values
        return self._compute_ic_vector(X_raw, y_raw, len(factor_names))

    @staticmethod
    def _compute_ic_vector(
        X_raw: np.ndarray, y_raw: np.ndarray, n_factors: int
    ) -> np.ndarray:
        """向量化计算因子 IC (Pearson 相关)."""
        if len(y_raw) < 10:
            return np.full(n_factors, np.nan)
        # 中心化
        Xc = X_raw - np.nanmean(X_raw, axis=0)
        yc = y_raw - np.nanmean(y_raw)
        # 处理 NaN: 用 0 替换 (相当于不参与协方差)
        Xc = np.nan_to_num(Xc, nan=0.0)
        yc = np.nan_to_num(yc, nan=0.0)
        # IC[i] = (Xc[:, i] @ yc) / (sqrt(sum(Xc[:, i]^2)) * sqrt(sum(yc^2)))
        numerator = Xc.T @ yc  # (K,)
        x_norm = np.sqrt((Xc ** 2).sum(axis=0))  # (K,)
        y_norm = float(np.sqrt((yc ** 2).sum()))
        denom = x_norm * y_norm  # (K,)
        factor_ics = np.full(n_factors, np.nan)
        valid = denom > 0
        factor_ics[valid] = numerator[valid] / denom[valid]
        return factor_ics

    def _apply_direction_correction(
        self, factor_ics: np.ndarray, factor_names: list
    ) -> None:
        """记录因子方向 (CR-010修复: 不再修改 coef, 仅记录用于日志诊断).

        方向协议: 因子矩阵始终保持原始方向, 模型系数自然携带正负号.
        lstsq 的 coef 已反映因子与收益的真实关系 (含正负号), 无需再次翻转.
        本方法仅记录 IC 符号供日志诊断使用, 不影响 predict 计算.
        """
        if self._coef is None or len(factor_ics) == 0:
            return
        # CR-010: coef 保持 lstsq 原始结果, 不做方向翻转
        # _factor_directions 仅用于日志记录, 不参与 predict 计算
        directions = np.ones(len(factor_names))
        # 记录 IC<0 的因子方向 (仅诊断用)
        neg_ic = ~np.isnan(factor_ics) & (factor_ics < 0)
        directions = np.where(neg_ic, -1, directions)
        self._factor_directions = directions

    def predict(
        self,
        factors: Dict[str, FactorMatrix],
        universe: Universe,
        date: Date,
    ) -> ExpectedReturns:
        if not self._fitted or self._coef is None:
            return pd.Series(0.0, index=universe)

        xs = []
        for name in sorted(factors.keys()):
            f = factors[name]
            if date in f.index:
                xs.append(f.loc[date])
        if not xs:
            return pd.Series(0.0, index=universe)

        X = pd.concat(xs, axis=1).reindex(universe).fillna(0).values
        # CR-010: 因子矩阵保持原始方向, 系数自然携带正负号, 不再乘 _factor_directions
        pred = X @ self._coef + self._intercept
        return pd.Series(pred, index=universe)


@register("return_model", "ridge")
class RidgeModel(OLSModel):
    """Pooled Ridge baseline with ordered inner time-series validation."""

    def __init__(
        self,
        fit_intercept: bool = True,
        recent_ic_window: int = 60,
        direction_adaptive: bool = True,
        ridge_alpha: float = 1.0,
        ridge_alphas: Optional[List[float]] = None,
        ridge_cv_folds: int = 3,
    ):
        super().__init__(fit_intercept, recent_ic_window, direction_adaptive)
        self._ridge_alpha = float(ridge_alpha)
        self._ridge_alphas = list(ridge_alphas or [])
        self._ridge_cv_folds = int(ridge_cv_folds)
        self.selected_alpha_: Optional[float] = None

    def fit(
        self,
        factors: Dict[str, FactorMatrix],
        forward_returns: ReturnMatrix,
        universe: UniverseSchedule = None,
    ) -> "RidgeModel":
        factor_names = sorted(factors.keys())
        if not factor_names:
            return self
        merged, factor_names, X_vals, y_vals, _ = stack_factors_and_returns(
            factors, forward_returns
        )
        if len(merged) < len(factor_names) + 5:
            return self
        alpha = self._ridge_alpha
        if self._ridge_alphas:
            alpha = _select_ridge_alpha_time_series(
                X_vals,
                y_vals,
                merged.index.get_level_values(0),
                self._ridge_alphas,
                fit_intercept=self._fit_intercept,
                n_folds=self._ridge_cv_folds,
            )
        self._coef, self._intercept = _fit_linear_coefficients(
            X_vals,
            y_vals,
            fit_intercept=self._fit_intercept,
            ridge_alpha=alpha,
        )
        self.selected_alpha_ = alpha
        self._factor_directions = np.ones(len(factor_names))
        if self._direction_adaptive and self._recent_ic_window > 0:
            factor_ic = self._compute_recent_ic(
                merged, factor_names, self._recent_ic_window
            )
        else:
            factor_ic = self._compute_full_ic(merged, factor_names)
        self._apply_direction_correction(factor_ic, factor_names)
        self._fitted = True
        return self


@register("return_model", "ic_weighted")
class ICWeightedModel(ReturnModel):
    """IC 加权收益预测模型.

    不做回归, 直接用近期因子 IC 作为权重加权合成因子暴露.
    相比 OLS 的优势:
    - 无多重共线性问题 (不做回归)
    - IC 权重自动反映因子有效性
    - 对因子数量不敏感 (不会过拟合)

    参考: Grinold & Kahn, Active Portfolio Management.

    Args:
        ic_window: IC 计算窗口 (天数), 推荐 60.
        ic_decay: IC 指数衰减半衰期 (天数), 0 表示不衰减.
            衰减权重使近期 IC 权重更大.
    """

    def __init__(
        self,
        ic_window: int = 60,
        ic_decay: int = 0,
    ):
        self._ic_window = ic_window
        self._ic_decay = ic_decay
        self._factor_ics: Dict[str, float] = {}
        # CR-010: 删除 _factor_directions, IC 本身已带符号, 无需单独记录方向
        self._fitted = False

    def fit(
        self,
        factors: Dict[str, FactorMatrix],
        forward_returns: ReturnMatrix,
        universe: UniverseSchedule = None,
    ) -> "ICWeightedModel":
        """计算每个因子的近期 IC 作为合成权重."""
        factor_names = sorted(factors.keys())
        if not factor_names:
            return self

        merged, factor_names, _, _, _ = stack_factors_and_returns(
            factors, forward_returns
        )
        if len(merged) < 20:
            return self

        # 取近期数据
        if self._ic_window > 0:
            # 近期 window 天 (按日期数 × 品种数近似)
            unique_dates = merged.index.get_level_values(0).unique()
            recent_dates = unique_dates[-self._ic_window:]
            recent_data = merged.loc[recent_dates]
        else:
            recent_data = merged

        X_raw = recent_data[factor_names].values
        y_raw = recent_data["fwd_ret"].values

        # 计算每个因子的 IC
        ics = OLSModel._compute_ic_vector(X_raw, y_raw, len(factor_names))

        # 指数衰减加权 (若启用)
        if self._ic_decay > 0 and self._ic_window > 0:
            # 对近 ic_window 天的数据做指数衰减
            # 这里简化: 用半衰期调整 IC 权重
            decay_factor = np.exp(-np.log(2) / self._ic_decay)
            ics = ics * decay_factor

        for i, name in enumerate(factor_names):
            ic = float(ics[i]) if not np.isnan(ics[i]) else 0.0
            # CR-010: IC 本身已带符号 (ic<0 时权重为负), 不再单独记录方向
            self._factor_ics[name] = ic

        self._fitted = True
        return self

    def predict(
        self,
        factors: Dict[str, FactorMatrix],
        universe: Universe,
        date: Date,
    ) -> ExpectedReturns:
        """用 IC 加权合成因子暴露作为预期收益."""
        if not self._fitted:
            return pd.Series(0.0, index=universe)

        # 收集当日因子暴露
        factor_exposures: Dict[str, pd.Series] = {}
        for name in sorted(factors.keys()):
            f = factors[name]
            if date in f.index:
                factor_exposures[name] = f.loc[date].reindex(universe).fillna(0)

        if not factor_exposures:
            return pd.Series(0.0, index=universe)

        # CR-010: IC 已带符号, 权重 w=ic/Σ|ic| 自然含正负号, 不再乘 direction
        # result = Σ (IC_i × exposure_i) / Σ|IC_i|
        total_w = sum(abs(v) for v in self._factor_ics.values()) or 1.0
        result = pd.Series(0.0, index=universe)
        for name, exposure in factor_exposures.items():
            ic = self._factor_ics.get(name, 0.0)
            w = ic / total_w
            result += w * exposure

        return result


@register("return_model", "sector_grouped_ols")
class SectorGroupedOLSModel(ReturnModel):
    """板块分组 OLS 收益预测模型 — 解决品种适用性问题.

    核心改进: 按板块分组拟合 OLS, 不同板块使用不同的因子系数.
    解决"一个因子对黑色系有效但对农产品无效"的问题.

    设计动机:
    - 原 OLSModel 池化回归, 所有品种共用一套系数
    - 实际上因子在不同板块的 alpha 差异巨大 (如动量因子在黑色系有效, 在农产品无效)
    - 按板块分组拟合, 每个板块内的品种有相似的因子-收益关系

    Args:
        fit_intercept: 是否拟合截距.
        min_samples_per_sector: 每个板块最小样本数, 不足则回退到全局池化.
        fallback_to_global: 板块样本不足时是否回退到全局池化模型.
        sector_factor_map: 分板块因子集映射 {sector: [factor_name, ...]}.
            方案B核心: 每个板块只用适配性研究确认的有效因子拟合,
            过滤无效因子以降低噪声. None 或空字典表示所有板块使用全部因子
            (向后兼容). 未在 map 中的板块也使用全部因子.
    """

    # 板块映射 (8板块细分: 股指/国债分开, 有色/贵金属分开)
    # 细分依据: 股指期货与国债期货的因子-收益关系截然不同
    #           有色金属(工业金属)与贵金属的驱动因素不同
    from core.sectors import SECTOR_MAP as _SECTOR_MAP

    def __init__(
        self,
        fit_intercept: bool = True,
        min_samples_per_sector: int = 100,
        fallback_to_global: bool = True,
        sector_factor_map: Optional[Dict[str, List[str]]] = None,
        factor_weight_caps: Optional[Dict[str, float]] = None,
        unmapped_sector_policy: str = "global",
        ridge_alpha: float = 0.0,
        ridge_alphas: Optional[List[float]] = None,
        ridge_cv_folds: int = 3,
    ):
        self._fit_intercept = fit_intercept
        self._min_samples = min_samples_per_sector
        self._fallback_to_global = fallback_to_global
        # 方案B: 分板块因子集 {sector: [factor_name, ...]}
        self._sector_factor_map: Dict[str, List[str]] = sector_factor_map or {}
        self._factor_weight_caps = {
            str(name): float(cap)
            for name, cap in (factor_weight_caps or {}).items()
        }
        if any(not 0.0 < cap <= 1.0 for cap in self._factor_weight_caps.values()):
            raise ValueError("factor_weight_caps values must be in (0, 1]")
        if unmapped_sector_policy not in {"global", "zero"}:
            raise ValueError("unmapped_sector_policy must be 'global' or 'zero'")
        self._unmapped_sector_policy = unmapped_sector_policy
        # 每个板块的系数 {sector: {"coef": ndarray, "intercept": float, "factor_indices": list}}
        self._sector_models: Dict[str, dict] = {}
        # 全局回退模型 (样本不足时使用)
        self._global_coef: np.ndarray | None = None
        self._global_intercept: float = 0.0
        self._global_factor_indices: list = []  # 全局模型使用的因子列索引
        self._factor_names: list = []
        self._fitted = False
        self._ridge_alpha = float(ridge_alpha)
        self._ridge_alphas = list(ridge_alphas or [])
        self._ridge_cv_folds = int(ridge_cv_folds)
        self.selected_alpha_: Dict[str, float] = {}

    def _get_sector(self, ticker: str) -> str:
        """获取品种所属板块."""
        return self._SECTOR_MAP.get(str(ticker), "other")

    def fit(
        self,
        factors: Dict[str, FactorMatrix],
        forward_returns: ReturnMatrix,
        universe: UniverseSchedule = None,
    ) -> "SectorGroupedOLSModel":
        """按板块分组拟合 OLS.

        方案B: 若 sector_factor_map 已配置, 每个板块只用该板块的有效因子
        子集拟合, 过滤无效因子以降低噪声. 未配置的板块使用全部因子.
        """
        factor_names = sorted(factors.keys())
        if not factor_names:
            return self
        self._factor_names = factor_names
        self._factor_cap_vector = np.asarray(
            [self._factor_weight_caps.get(name, 1.0) for name in factor_names],
            dtype=float,
        )

        merged, factor_names, X_vals, y_vals, _ = stack_factors_and_returns(
            factors, forward_returns
        )
        if len(merged) < len(factor_names) + 5:
            return self

        # 全局模型 (回退用) — 使用全部因子
        row_dates = pd.DatetimeIndex(merged.index.get_level_values(0))
        global_alpha = self._choose_ridge_alpha(X_vals, y_vals, row_dates)
        self._global_coef, self._global_intercept = _fit_linear_coefficients(
            X_vals,
            y_vals,
            fit_intercept=self._fit_intercept,
            ridge_alpha=global_alpha,
        )
        self.selected_alpha_["global"] = global_alpha
        # 全局模型使用全部因子列
        self._global_factor_indices = list(range(len(factor_names)))

        # 按板块分组拟合
        tickers = merged.index.get_level_values(1)
        sectors = pd.Series([self._get_sector(t) for t in tickers], index=merged.index)

        for sec in sectors.unique():
            mask = (sectors == sec).values
            n_sec = mask.sum()

            if (
                self._sector_factor_map
                and sec not in self._sector_factor_map
                and self._unmapped_sector_policy == "zero"
            ):
                self._sector_models[sec] = {
                    "coef": np.array([], dtype=float),
                    "intercept": 0.0,
                    "factor_indices": [],
                }
                continue

            # 方案B: 获取该板块的有效因子子集
            sec_factor_names = self._sector_factor_map.get(sec)
            if sec_factor_names:
                # 只保留在 factor_names 中存在的因子
                sec_indices = [
                    factor_names.index(f)
                    for f in sec_factor_names if f in factor_names
                ]
            else:
                # 未配置该板块的因子集, 使用全部因子
                sec_indices = list(range(len(factor_names)))

            # 确保因子子集不为空
            if not sec_indices:
                sec_indices = list(range(len(factor_names)))

            if n_sec < self._min_samples:
                # 样本不足, 使用全局模型 (但只取该板块有效因子列)
                if self._fallback_to_global:
                    self._sector_models[sec] = {
                        "coef": self._global_coef[sec_indices].copy(),
                        "intercept": self._global_intercept,
                        "factor_indices": sec_indices,
                    }
                continue

            # 只取该板块有效因子的列
            X_sec = X_vals[mask][:, sec_indices]
            y_sec = y_vals[mask]
            try:
                sector_alpha = self._choose_ridge_alpha(
                    X_sec, y_sec, row_dates[mask]
                )
                coef_s, intercept_s = _fit_linear_coefficients(
                    X_sec,
                    y_sec,
                    fit_intercept=self._fit_intercept,
                    ridge_alpha=sector_alpha,
                )
                self.selected_alpha_[sec] = sector_alpha
                self._sector_models[sec] = {
                    "coef": coef_s,
                    "intercept": intercept_s,
                    "factor_indices": sec_indices,
                }
            except Exception:
                if self._fallback_to_global:
                    self._sector_models[sec] = {
                        "coef": self._global_coef[sec_indices].copy(),
                        "intercept": self._global_intercept,
                        "factor_indices": sec_indices,
                    }

        self._fitted = True
        return self

    def _choose_ridge_alpha(self, X, y, dates) -> float:
        if not self._ridge_alphas:
            return self._ridge_alpha
        return _select_ridge_alpha_time_series(
            X,
            y,
            dates,
            self._ridge_alphas,
            fit_intercept=self._fit_intercept,
            n_folds=self._ridge_cv_folds,
        )

    def predict(
        self,
        factors: Dict[str, FactorMatrix],
        universe: Universe,
        date: Date,
    ) -> ExpectedReturns:
        """按品种所属板块使用对应系数预测.

        方案B: 每个品种使用其板块的因子子集和对应系数进行预测.
        """
        if not self._fitted or self._global_coef is None:
            return pd.Series(0.0, index=universe)

        # 收集当日因子暴露
        xs = []
        for name in self._factor_names:
            f = factors.get(name)
            if f is not None and date in f.index:
                xs.append(f.loc[date])
            elif f is not None:
                frame = f if f.index.is_monotonic_increasing else f.sort_index()
                pos = int(frame.index.searchsorted(pd.Timestamp(date), side="right")) - 1
                if pos < 0:
                    raise ValueError(
                        f"factor {name!r} has no observation available as of {date}"
                    )
                xs.append(frame.iloc[pos])
            else:
                xs.append(pd.Series(0.0, index=universe, name=name))
        if not xs:
            return pd.Series(0.0, index=universe)

        X = pd.concat(xs, axis=1).reindex(universe).fillna(0).values
        X = X * self._factor_cap_vector[None, :]

        # 按品种板块选择系数 + 因子子集
        result = np.zeros(len(universe))
        for i, tick in enumerate(universe):
            sec = self._get_sector(tick)
            if sec in self._sector_models:
                model = self._sector_models[sec]
                indices = model["factor_indices"]
                result[i] = X[i][indices] @ model["coef"] + model["intercept"]
            else:
                # 回退: 使用全局模型 (全部因子)
                result[i] = X[i] @ self._global_coef + self._global_intercept

        return pd.Series(result, index=universe)


@register("return_model", "sector_grouped_ridge")
class SectorGroupedRidgeModel(SectorGroupedOLSModel):
    """Sector-specific Ridge baseline; all selection remains training-window only."""

    def __init__(
        self,
        fit_intercept: bool = True,
        min_samples_per_sector: int = 100,
        fallback_to_global: bool = True,
        sector_factor_map: Optional[Dict[str, List[str]]] = None,
        factor_weight_caps: Optional[Dict[str, float]] = None,
        unmapped_sector_policy: str = "global",
        ridge_alpha: float = 1.0,
        ridge_alphas: Optional[List[float]] = None,
        ridge_cv_folds: int = 3,
    ):
        super().__init__(
            fit_intercept=fit_intercept,
            min_samples_per_sector=min_samples_per_sector,
            fallback_to_global=fallback_to_global,
            sector_factor_map=sector_factor_map,
            factor_weight_caps=factor_weight_caps,
            unmapped_sector_policy=unmapped_sector_policy,
            ridge_alpha=ridge_alpha,
            ridge_alphas=(ridge_alphas or [0.01, 0.1, 1.0, 10.0]),
            ridge_cv_folds=ridge_cv_folds,
        )
