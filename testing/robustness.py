"""因子稳健性检验.

检验因子在不同时间段和不同子市场的稳定性,
避免筛选出仅在特定时段有效的过拟合因子.

检验内容:
1. 时间分段稳定性: 将样本期分为N段, 检验每段IC符号一致性
2. 子市场稳定性: 按板块分组, 检验因子在各板块的IC一致性
3. 跨段IR一致性: 各段IR的标准差 (越低越稳定)

业界标准:
- IC符号一致率 >= 70% (各段IC同向)
- 跨段IR标准差 < 0.3 (稳定性)
- 至少 80% 的子段 |IC| > 0.01 (经济意义)
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from core.registry import register
from core.types import FactorMatrix, ReturnMatrix, UniverseSchedule
from testing.base import FactorTest, TestResult


class RobustnessResult(TestResult):
    """稳健性检验结果."""

    def __init__(
        self,
        segment_ics: Dict[str, float],
        segment_irs: Dict[str, float],
        ic_sign_consistency: float,
        ir_std: float,
        min_segment_ic_abs: float,
        passes_ic_consistency: bool,
        passes_ir_stability: bool,
        sector_ics: Dict[str, float] = None,
    ):
        self.segment_ics = segment_ics
        self.segment_irs = segment_irs
        self.ic_sign_consistency = ic_sign_consistency
        self.ir_std = ir_std
        self.min_segment_ic_abs = min_segment_ic_abs
        self.passes_ic_consistency = passes_ic_consistency
        self.passes_ir_stability = passes_ir_stability
        # CR-028: 子市场 (板块) IC 检验结果
        self.sector_ics = sector_ics or {}

    def to_dict(self) -> dict:
        return {
            "segment_ics": {k: float(v) for k, v in self.segment_ics.items()},
            "segment_irs": {k: float(v) for k, v in self.segment_irs.items()},
            "ic_sign_consistency": float(self.ic_sign_consistency),
            "ir_std": float(self.ir_std),
            "min_segment_ic_abs": float(self.min_segment_ic_abs),
            "passes_ic_consistency": self.passes_ic_consistency,
            "passes_ir_stability": self.passes_ir_stability,
            "passes_all": self.passes_ic_consistency and self.passes_ir_stability,
            # CR-028: 子市场检验结果
            "sector_ics": {k: float(v) for k, v in self.sector_ics.items()},
        }

    def summary(self) -> str:
        d = self.to_dict()
        ic_status = "✓" if d["passes_ic_consistency"] else "✗"
        ir_status = "✓" if d["passes_ir_stability"] else "✗"
        seg_ic_str = ", ".join(
            f"{k}={v:.4f}" for k, v in d["segment_ics"].items()
        )
        # CR-028: 展示子市场 IC
        sector_str = ", ".join(
            f"{k}={v:.4f}" for k, v in d["sector_ics"].items()
        ) if d["sector_ics"] else "N/A"
        return (
            f"稳健性: IC符号一致率={d['ic_sign_consistency']:.1%} {ic_status}, "
            f"IR_std={d['ir_std']:.3f} {ir_status} | "
            f"分段IC: {seg_ic_str} | 板块IC: {sector_str}"
        )


@register("factor_test", "robustness")
class RobustnessTest(FactorTest):
    """因子稳健性检验 (时间分段 + 子市场).

    时间分段: 将IC检验区间均分为n_segments段, 分别计算IC和IR.
    子市场检验: 可选, 按板块分组计算各板块IC.

    稳定性标准:
    - IC符号一致率: 各段IC同号的比例 >= 70%
    - IR标准差: 各段IR的离散度 < 0.3
    - 最小段IC绝对值: 至少 |IC| > 0.01
    """

    name = "robustness"

    def __init__(
        self,
        n_segments: int = 4,
        min_ic_abs: float = 0.01,
        min_cross_section: int = 10,
        run_sector_test: bool = True,
    ):
        self.n_segments = n_segments
        self.min_ic_abs = min_ic_abs
        # CR-028: 最小截面样本数 (pairwise dropna 后)
        self.min_cross_section = min_cross_section
        # CR-028: 是否执行子市场 (板块) 检验
        self.run_sector_test = run_sector_test

    def run(
        self,
        factor: FactorMatrix,
        forward_returns: ReturnMatrix,
        universe: UniverseSchedule = None,
        **params,
    ) -> RobustnessResult:
        # 对齐日期
        common_dates = factor.index.intersection(forward_returns.index)
        f_aligned = factor.loc[common_dates]
        r_aligned = forward_returns.loc[common_dates]

        # 时间分段: 均分为 n_segments 段
        n = len(common_dates)
        if n < self.n_segments * 20:
            # 样本不足, 返回保守结果
            return RobustnessResult(
                segment_ics={"full": 0.0},
                segment_irs={"full": 0.0},
                ic_sign_consistency=0.0,
                ir_std=0.0,
                min_segment_ic_abs=0.0,
                passes_ic_consistency=False,
                passes_ir_stability=False,
                sector_ics={},
            )

        segment_size = n // self.n_segments
        segment_ics: Dict[str, float] = {}
        segment_irs: Dict[str, float] = {}
        segment_valid: Dict[str, bool] = {}  # 记录每段是否有足够样本

        for seg_idx in range(self.n_segments):
            start = seg_idx * segment_size
            end = (seg_idx + 1) * segment_size if seg_idx < self.n_segments - 1 else n
            seg_dates = common_dates[start:end]

            f_seg = f_aligned.loc[seg_dates]
            r_seg = r_aligned.loc[seg_dates]

            from testing.ic_test import _vectorized_spearman_ic

            daily_ic_series, _ = _vectorized_spearman_ic(
                f_seg, r_seg, min_stocks=self.min_cross_section
            )
            daily_ics = daily_ic_series.to_list()

            seg_key = f"seg{seg_idx+1}"
            if len(daily_ics) < 5:
                segment_ics[seg_key] = 0.0
                segment_irs[seg_key] = 0.0
                segment_valid[seg_key] = False
                continue

            segment_valid[seg_key] = True
            ic_mean = float(np.mean(daily_ics))
            ic_std = float(np.std(daily_ics, ddof=1)) if len(daily_ics) > 1 else 0.01
            ir = ic_mean / ic_std if ic_std > 1e-10 else 0.0

            segment_ics[seg_key] = ic_mean
            segment_irs[seg_key] = ir

        # IC符号一致率: 各段IC同号的比例
        # 用第一段方向作为基准 (避免用后续分段信息确定主方向, 消除数据窥探)
        ic_values = list(segment_ics.values())
        if len(ic_values) == 0:
            ic_sign_consistency = 0.0
        else:
            # 用第一段IC符号作为基准方向 (实时交易中最早可得的方向)
            dominant_sign = np.sign(ic_values[0]) if ic_values[0] != 0 else 0
            if dominant_sign == 0:
                # 第一段IC为零, 用全样本均值方向
                full_ic_mean = np.mean(ic_values)
                dominant_sign = np.sign(full_ic_mean) if full_ic_mean != 0 else 1.0
            same_sign_count = sum(1 for v in ic_values if np.sign(v) == dominant_sign)
            ic_sign_consistency = same_sign_count / len(ic_values)

        # IR标准差
        ir_values = list(segment_irs.values())
        ir_std = float(np.std(ir_values, ddof=1)) if len(ir_values) > 1 else 0.0

        # 最小段IC绝对值 (用于经济意义检查)
        min_segment_ic_abs = float(min(abs(v) for v in ic_values)) if ic_values else 0.0

        # 经济意义检查: 至少 80% 的分段 |IC| >= min_ic_abs
        n_above = sum(1 for v in ic_values if abs(v) >= self.min_ic_abs)
        passes_ic_abs = (n_above / len(ic_values)) >= 0.80 if ic_values else False

        # 稳定性判断
        passes_ic_consistency = (
            ic_sign_consistency >= 0.70
            and passes_ic_abs
        )
        # IR标准差: 只用有效段 (排除因样本不足置0的空段)
        # 使用 segment_valid 标记判断 (避免闭包引用循环变量 daily_ics 的 bug)
        valid_irs = [
            v for k, v in segment_irs.items()
            if segment_valid.get(k, False)
        ]
        ir_std = float(np.std(valid_irs, ddof=1)) if len(valid_irs) > 1 else 0.0
        passes_ir_stability = ir_std < 0.3

        # CR-028: 子市场 (板块) 稳健性检验 — 按商品板块执行真实子样本 IC
        sector_ics: Dict[str, float] = {}
        if self.run_sector_test:
            sector_ics = self._compute_sector_ics(f_aligned, r_aligned)

        return RobustnessResult(
            segment_ics=segment_ics,
            segment_irs=segment_irs,
            ic_sign_consistency=ic_sign_consistency,
            ir_std=ir_std,
            min_segment_ic_abs=min_segment_ic_abs,
            passes_ic_consistency=passes_ic_consistency,
            passes_ir_stability=passes_ir_stability,
            sector_ics=sector_ics,
        )

    def _compute_sector_ics(
        self,
        f_aligned: pd.DataFrame,
        r_aligned: pd.DataFrame,
    ) -> Dict[str, float]:
        """CR-028: 按商品板块执行真实子样本 IC 检验.

        使用 factors/library/cross_commodity.py 的 SECTOR_MAP 将品种分组,
        对每个板块独立计算 Spearman rank IC 均值.

        Args:
            f_aligned: 对齐后的因子矩阵 (日期×品种)
            r_aligned: 对齐后的前向收益矩阵 (日期×品种)

        Returns:
            {板块名: IC均值} 字典 (样本不足的板块被跳过)
        """
        from core.sectors import SECTOR_MAP

        # 按板块分组品种
        tickers = f_aligned.columns
        sector_to_tickers: Dict[str, list] = {}
        for t in tickers:
            sector = SECTOR_MAP.get(str(t), "other")
            sector_to_tickers.setdefault(sector, []).append(t)

        sector_ics: Dict[str, float] = {}
        for sector, sector_tickers in sector_to_tickers.items():
            # 板块内品种数不足, 跳过
            if len(sector_tickers) < self.min_cross_section:
                continue
            f_sector = f_aligned[sector_tickers]
            r_sector = r_aligned[sector_tickers]

            from testing.ic_test import _vectorized_spearman_ic

            daily_ic_series, _ = _vectorized_spearman_ic(
                f_sector, r_sector, min_stocks=self.min_cross_section
            )
            daily_ics = daily_ic_series.to_list()

            if len(daily_ics) >= 5:
                sector_ics[sector] = float(np.mean(daily_ics))

        return sector_ics


class CalendarYearRobustnessResult(TestResult):
    """Natural-year diagnostics and the frozen scorecard inputs."""

    def __init__(self, **values):
        self.__dict__.update(values)

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def summary(self) -> str:
        status = "observation" if self.observation_channel else "eligible"
        return (
            f"calendar-year robustness: years={self.n_valid_years}, "
            f"direction={self.ic_sign_consistency:.1%}, "
            f"effect={self.effect_year_ratio:.1%}, "
            f"score={self.scorecard_score:.3f}, status={status}"
        )


def _moving_block_mean_interval(
    daily_values: pd.Series,
    *,
    samples: int,
    confidence: float = 0.95,
    seed: int = 20260728,
) -> tuple[float, float]:
    values = np.asarray(daily_values.dropna(), dtype=float)
    if len(values) < 20 or samples < 1:
        return float("nan"), float("nan")
    block = max(int(round(np.sqrt(len(values)))), 2)
    starts = np.arange(max(len(values) - block + 1, 1))
    blocks_needed = int(np.ceil(len(values) / block))
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for offset in range(0, samples, 128):
        batch = min(128, samples - offset)
        sampled_starts = rng.choice(starts, size=(batch, blocks_needed))
        indices = (
            sampled_starts[:, :, None] + np.arange(block)[None, None, :]
        ).reshape(batch, -1)[:, :len(values)]
        means[offset:offset + batch] = values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)


class CalendarYearRobustnessTest(FactorTest):
    """Evaluate stability by natural year without treating minute bars as days."""

    name = "calendar_year_robustness"

    def __init__(
        self,
        *,
        min_ic_abs: float = 0.01,
        direction_ratio: float = 0.60,
        effect_ratio: float = 0.65,
        minimum_years: int = 5,
        minimum_days_per_year: int = 20,
        min_cross_section: int = 10,
        bootstrap_samples: int = 399,
        scorecard_weights: Dict[str, float] = None,
        hit_rate_threshold: float = 0.52,
        ir_std_max: float = 0.30,
        scorecard_threshold: float = 0.75,
        scorecard_enforced: bool = False,
    ):
        self.min_ic_abs = float(min_ic_abs)
        self.direction_ratio = float(direction_ratio)
        self.effect_ratio = float(effect_ratio)
        self.minimum_years = int(minimum_years)
        self.minimum_days_per_year = int(minimum_days_per_year)
        self.min_cross_section = int(min_cross_section)
        self.bootstrap_samples = int(bootstrap_samples)
        self.scorecard_weights = scorecard_weights or {
            "annual_direction": 0.25,
            "annual_effect": 0.25,
            "hit_rate": 0.25,
            "ir_stability": 0.25,
        }
        self.hit_rate_threshold = float(hit_rate_threshold)
        self.ir_std_max = float(ir_std_max)
        self.scorecard_threshold = float(scorecard_threshold)
        self.scorecard_enforced = bool(scorecard_enforced)

    def run(
        self,
        factor: FactorMatrix,
        forward_returns: ReturnMatrix,
        universe: UniverseSchedule = None,
        **params,
    ) -> CalendarYearRobustnessResult:
        del universe
        common_dates = factor.index.intersection(forward_returns.index)
        f_aligned = factor.loc[common_dates]
        r_aligned = forward_returns.loc[common_dates]
        from testing.ic_test import _vectorized_spearman_ic

        ic_series, _ = _vectorized_spearman_ic(
            f_aligned, r_aligned, min_stocks=self.min_cross_section
        )
        ic_series = ic_series.replace([np.inf, -np.inf], np.nan).dropna()
        orientation = int(params.get("orientation", 1))
        orientation = 1 if orientation >= 0 else -1
        oriented = ic_series * orientation

        years = {}
        year_irs = {}
        valid_years = []
        for year, values in oriented.groupby(oriented.index.year):
            trading_days = int(pd.DatetimeIndex(values.index).normalize().nunique())
            if trading_days < self.minimum_days_per_year:
                continue
            mean = float(values.mean())
            std = float(values.std(ddof=1))
            years[str(int(year))] = mean
            year_irs[str(int(year))] = mean / std if std > 0.0 else 0.0
            valid_years.append(int(year))

        year_values = np.asarray(list(years.values()), dtype=float)
        direction_consistency = (
            float(np.mean(year_values > 0.0)) if len(year_values) else 0.0
        )
        effect_ratio = (
            float(np.mean(year_values >= self.min_ic_abs))
            if len(year_values) else 0.0
        )
        ir_values = np.asarray(list(year_irs.values()), dtype=float)
        ir_std = float(ir_values.std(ddof=1)) if len(ir_values) > 1 else 0.0

        daily_ic = oriented.groupby(oriented.index.normalize()).mean()
        hit_rate = float((daily_ic > 0.0).mean()) if len(daily_ic) else 0.0
        component_passes = {
            "annual_direction": direction_consistency >= self.direction_ratio,
            "annual_effect": effect_ratio >= self.effect_ratio,
            "hit_rate": hit_rate >= self.hit_rate_threshold,
            "ir_stability": ir_std < self.ir_std_max,
        }
        score = float(sum(
            float(self.scorecard_weights.get(name, 0.0)) * float(passed)
            for name, passed in component_passes.items()
        ))

        failed = [value < self.min_ic_abs for value in year_values]
        longest_failure = current_failure = 0
        for is_failed in failed:
            current_failure = current_failure + 1 if is_failed else 0
            longest_failure = max(longest_failure, current_failure)
        ci_low, ci_high = _moving_block_mean_interval(
            daily_ic, samples=self.bootstrap_samples
        )
        observation = len(valid_years) < self.minimum_years
        scorecard_pass = score >= self.scorecard_threshold
        return CalendarYearRobustnessResult(
            segment_ics=years,
            segment_irs=year_irs,
            ic_sign_consistency=direction_consistency,
            effect_year_ratio=effect_ratio,
            ir_std=ir_std,
            hit_rate=hit_rate,
            n_valid_years=len(valid_years),
            valid_years=valid_years,
            observation_channel=observation,
            observation_reason=(
                f"requires_{self.minimum_years}_calendar_years"
                if observation else ""
            ),
            longest_consecutive_failure_years=int(longest_failure),
            worst_year_ic=(float(year_values.min()) if len(year_values) else 0.0),
            block_bootstrap_ic_ci=[ci_low, ci_high],
            scorecard_components=component_passes,
            scorecard_weights=dict(self.scorecard_weights),
            scorecard_score=score,
            scorecard_threshold=self.scorecard_threshold,
            scorecard_enforced=self.scorecard_enforced,
            passes_scorecard=(scorecard_pass if self.scorecard_enforced else True),
            passes_ic_consistency=bool(
                direction_consistency >= self.direction_ratio
                and effect_ratio >= self.effect_ratio
            ),
            passes_ir_stability=ir_std < self.ir_std_max,
        )
