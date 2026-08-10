"""Causal historical search helpers for the native production structure.

This module is deliberately isolated from the production strategy.  It compares
factor aggregation, cross-sectional selection and within-pool allocation using
the already audited factor/risk panels, then supports a narrow cluster-aware
factor search.  Every rolling estimate excludes the current return observation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


PERIODS_PER_YEAR = 242


@dataclass(frozen=True, order=True)
class PortfolioRecipe:
    """One finite, predeclared production-construction recipe."""

    factor_weight: str = "lw_abs"
    top_n: int = 10
    sector_cap: int = 3
    asset_weight: str = "erc"

    @property
    def name(self) -> str:
        cap = "none" if self.sector_cap <= 0 else str(self.sector_cap)
        return (
            f"{self.factor_weight}__top{self.top_n}_bottom{self.top_n}"
            f"__cap{cap}__{self.asset_weight}"
        )

    def to_dict(self) -> dict:
        return {"name": self.name, **asdict(self)}


class CausalEligibilityEnvironment:
    """Small post-computation replacement for the minute-data experiment env."""

    def __init__(
        self,
        calendar: Sequence[pd.Timestamp],
        daily_returns: pd.DataFrame,
        sector_of: Mapping[str, str],
    ):
        self.cal = pd.DatetimeIndex(calendar)
        self.daily_ret = daily_returns
        self.sector_of = dict(sector_of)
        self._eligible_symbols_cache: dict[tuple[pd.Timestamp, int], list[str]] = {}

    def eligible_symbols(self, date, minimum_observations: int = 10) -> list[str]:
        date = pd.Timestamp(date)
        key = (date, int(minimum_observations))
        if key in self._eligible_symbols_cache:
            return self._eligible_symbols_cache[key]
        calendar = self.cal[
            (self.cal >= date - pd.Timedelta(days=90)) & (self.cal < date)
        ]
        history = self.daily_ret.reindex(calendar).replace([np.inf, -np.inf], np.nan)
        eligible = history.columns[
            history.notna().sum(axis=0).ge(int(minimum_observations))
        ].tolist()
        self._eligible_symbols_cache[key] = eligible
        return eligible


def ledoit_wolf_covariance(ic_matrix: pd.DataFrame) -> np.ndarray:
    """Return the same constant-correlation shrinkage estimate as production."""

    values = ic_matrix.to_numpy(dtype=float)
    rows, columns = values.shape
    if rows < 2 or columns < 2:
        raise ValueError("Ledoit-Wolf covariance requires at least 2x2 observations")
    sample_cov = np.cov(values, rowvar=False, ddof=1)
    sample_corr = np.corrcoef(values, rowvar=False)
    upper = sample_corr[np.triu_indices(columns, k=1)]
    average_corr = float(np.nanmean(upper)) if len(upper) else 0.0
    if not np.isfinite(average_corr):
        average_corr = 0.0
    target_corr = (
        np.eye(columns) * (1.0 - average_corr)
        + np.ones((columns, columns)) * average_corr
    )
    std = np.std(values, axis=0, ddof=1)
    target_cov = np.outer(std, std) * target_corr
    centered = values - np.mean(values, axis=0)
    pi = sum(
        np.sum((np.outer(row, row) - sample_cov) ** 2)
        for row in centered
    ) / rows
    gamma = float(np.sum((target_cov - sample_cov) ** 2))
    shrinkage = float(np.clip(pi / gamma, 0.0, 1.0)) if gamma > 0.0 else 0.5
    covariance = shrinkage * target_cov + (1.0 - shrinkage) * sample_cov
    return np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)


def _normalise_positive(raw: np.ndarray, columns: Sequence[str]) -> pd.Series:
    raw = np.asarray(raw, dtype=float)
    raw = np.where(np.isfinite(raw) & (raw > 0.0), raw, 0.0)
    total = float(raw.sum())
    if total <= 0.0:
        return pd.Series(1.0 / len(columns), index=columns, dtype=float)
    return pd.Series(raw / total, index=columns, dtype=float)


def _cap_simplex(weights: pd.Series, cap: float) -> pd.Series:
    """Project positive normalized weights onto a simple capped simplex."""

    if weights.empty:
        return weights
    effective_cap = max(float(cap), 1.0 / len(weights))
    result = weights.clip(lower=0.0).astype(float)
    if result.sum() <= 0.0:
        result[:] = 1.0 / len(result)
        return result
    result /= result.sum()
    for _ in range(len(result) + 2):
        high = result > effective_cap + 1e-12
        if not high.any():
            break
        result.loc[high] = effective_cap
        low = ~high
        residual = 1.0 - float(result.loc[high].sum())
        if not low.any() or residual <= 0.0:
            break
        base = result.loc[low]
        result.loc[low] = (
            residual / int(low.sum()) if base.sum() <= 0.0
            else base / base.sum() * residual
        )
    return result / result.sum()


def factor_weights(history: pd.DataFrame, method: str) -> pd.Series:
    """Estimate causal non-short factor weights from a complete IC history."""

    clean = history.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if clean.shape[1] < 2 or len(clean) == 0:
        return pd.Series(dtype=float)
    columns = list(clean.columns)
    if method == "equal" or len(clean) < 30:
        return pd.Series(1.0 / len(columns), index=columns, dtype=float)

    mean_ic = clean.mean().to_numpy(dtype=float)
    if method == "diag_icir":
        volatility = clean.std(ddof=1).replace(0.0, np.nan).to_numpy(dtype=float)
        return _normalise_positive(mean_ic / volatility, columns)

    covariance = ledoit_wolf_covariance(clean)
    if method == "lw_abs":
        try:
            raw = np.linalg.solve(covariance, mean_ic)
        except np.linalg.LinAlgError:
            raw = np.abs(mean_ic)
        return _normalise_positive(np.abs(raw), columns)
    if method == "lw_positive":
        ridge = max(float(np.trace(covariance)) / len(columns), 1e-12) * 1e-6
        try:
            raw = np.linalg.solve(
                covariance + ridge * np.eye(len(columns)), mean_ic
            )
        except np.linalg.LinAlgError:
            raw = mean_ic.copy()
        weights = _normalise_positive(np.maximum(raw, 0.0), columns)
        return _cap_simplex(weights, 0.35)
    raise ValueError(f"unknown factor-weight method: {method}")


def prepare_complete_history(
    history: pd.DataFrame,
    minimum_observations: int = 30,
) -> pd.DataFrame:
    """Admit a factor only after enough finite observations, then align rows."""

    clean = history.replace([np.inf, -np.inf], np.nan)
    required = min(int(minimum_observations), len(clean))
    if required <= 0:
        return clean.iloc[0:0, 0:0]
    columns = clean.columns[clean.notna().sum(axis=0).ge(required)]
    return clean.loc[:, columns].dropna(axis=0, how="any")


def select_pool(
    score: pd.Series,
    *,
    eligible: Sequence[str],
    sector_of: Mapping[str, str],
    top_n: int,
    sector_cap: int,
    ascending: bool,
) -> list[str]:
    """Select one side from eligible symbols with an optional sector count cap."""

    ordered = (
        score.reindex(list(eligible)).replace([np.inf, -np.inf], np.nan).dropna()
        .sort_values(ascending=ascending, kind="stable").index.tolist()
    )
    picks: list[str] = []
    counts: dict[str, int] = {}
    for symbol in ordered:
        sector = str(sector_of.get(symbol, "其他"))
        if sector_cap > 0 and counts.get(sector, 0) >= int(sector_cap):
            continue
        picks.append(str(symbol))
        counts[sector] = counts.get(sector, 0) + 1
        if len(picks) >= int(top_n):
            break
    return picks


def performance_metrics(returns: pd.Series) -> dict[str, float | int]:
    values = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 2:
        return {"observations": int(len(values))}
    growth = float((1.0 + values).prod())
    annual_return = growth ** (PERIODS_PER_YEAR / len(values)) - 1.0 if growth > 0 else -1.0
    annual_volatility = float(values.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))
    nav = (1.0 + values).cumprod()
    max_drawdown = float((nav / nav.cummax() - 1.0).min())
    return {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": (
            float(values.mean() * PERIODS_PER_YEAR / annual_volatility)
            if annual_volatility > 0.0 else 0.0
        ),
        "max_drawdown": max_drawdown,
        "calmar": annual_return / abs(max_drawdown) if max_drawdown < 0.0 else 0.0,
        "total_return": growth - 1.0,
        "observations": int(len(values)),
    }


def calendar_segments(start: pd.Timestamp, end: pd.Timestamp, years: int = 2) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Build deterministic, non-overlapping calendar blocks for robust ranking."""

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    segments = []
    cursor = start
    while cursor <= end:
        segment_end = min(cursor + pd.DateOffset(years=years) - pd.Timedelta(days=1), end)
        segments.append((cursor, segment_end))
        cursor = segment_end + pd.Timedelta(days=1)
    return segments


def robust_summary(
    ledger: pd.DataFrame,
    segments: Iterable[tuple[pd.Timestamp, pd.Timestamp]],
) -> dict[str, float | int]:
    """Summarize segment robustness without hiding the worst historical block."""

    rows = []
    for start, end in segments:
        returns = ledger.loc[start:end, "net_return"].copy()
        if len(returns) < 20:
            continue
        rows.append(performance_metrics(returns))
    if not rows:
        return {
            "segment_count": 0,
            "positive_segment_ratio": 0.0,
            "worst_sharpe": -10.0,
            "median_sharpe": -10.0,
            "median_annual_return": -1.0,
            "worst_drawdown": -1.0,
        }
    sharpes = np.asarray([row["sharpe"] for row in rows], dtype=float)
    annual_returns = np.asarray([row["annual_return"] for row in rows], dtype=float)
    drawdowns = np.asarray([row["max_drawdown"] for row in rows], dtype=float)
    return {
        "segment_count": len(rows),
        "positive_segment_ratio": float(np.mean(sharpes > 0.0)),
        "worst_sharpe": float(np.min(sharpes)),
        "median_sharpe": float(np.median(sharpes)),
        "median_annual_return": float(np.median(annual_returns)),
        "worst_drawdown": float(np.min(drawdowns)),
    }


def robustness_key(summary: Mapping[str, float | int]) -> tuple:
    """Transparent lexicographic preference used everywhere in the search."""

    return (
        float(summary.get("positive_segment_ratio", 0.0)),
        float(summary.get("worst_sharpe", -10.0)),
        float(summary.get("median_sharpe", -10.0)),
        float(summary.get("median_annual_return", -1.0)),
        float(summary.get("worst_drawdown", -1.0)),
    )


def aggregate_robust_summaries(rows: Sequence[Mapping[str, float | int]]) -> dict[str, float | int]:
    """Aggregate recipe evidence across seed factor sets conservatively."""

    if not rows:
        return robust_summary(pd.DataFrame(), [])
    return {
        "segment_count": int(sum(int(row.get("segment_count", 0)) for row in rows)),
        "positive_segment_ratio": float(np.mean([
            float(row.get("positive_segment_ratio", 0.0)) for row in rows
        ])),
        "worst_sharpe": float(min(float(row.get("worst_sharpe", -10.0)) for row in rows)),
        "median_sharpe": float(np.median([
            float(row.get("median_sharpe", -10.0)) for row in rows
        ])),
        "median_annual_return": float(np.median([
            float(row.get("median_annual_return", -1.0)) for row in rows
        ])),
        "worst_drawdown": float(min(float(row.get("worst_drawdown", -1.0)) for row in rows)),
    }


def training_factor_diagnostics(
    ic: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    minimum_coverage: float = 0.50,
) -> pd.DataFrame:
    """Compute fixed-direction training-only factor diagnostics."""

    train = ic.loc[start:end].replace([np.inf, -np.inf], np.nan)
    rows = []
    segments = calendar_segments(start, end, years=2)
    for name in train.columns:
        series = train[name]
        coverage = float(series.notna().mean())
        segment_sharpes = []
        segment_means = []
        for segment_start, segment_end in segments:
            values = series.loc[segment_start:segment_end].dropna()
            if len(values) < 20:
                continue
            std = float(values.std(ddof=1))
            segment_sharpes.append(
                float(values.mean() / std * np.sqrt(PERIODS_PER_YEAR)) if std > 0.0 else -10.0
            )
            segment_means.append(float(values.mean()))
        rows.append({
            "factor": str(name),
            "coverage": coverage,
            "mean_ic": float(series.mean()),
            "median_segment_sharpe": float(np.median(segment_sharpes)) if segment_sharpes else -10.0,
            "worst_segment_sharpe": float(np.min(segment_sharpes)) if segment_sharpes else -10.0,
            "positive_segment_ratio": float(np.mean(np.asarray(segment_means) > 0.0)) if segment_means else 0.0,
            "eligible": bool(
                coverage >= minimum_coverage
                and float(series.mean()) > 0.0
                and segment_means
                and float(np.mean(np.asarray(segment_means) > 0.0)) >= 0.50
            ),
        })
    return pd.DataFrame(rows).sort_values(
        ["eligible", "worst_segment_sharpe", "median_segment_sharpe", "factor"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def cluster_factors(
    ic: pd.DataFrame,
    factors: Sequence[str],
    *,
    correlation_threshold: float = 0.65,
) -> dict[str, int]:
    """Cluster factors by absolute training IC correlation using complete linkage."""

    names = [name for name in factors if name in ic.columns]
    if not names:
        return {}
    if len(names) == 1:
        return {names[0]: 1}
    corr = ic[names].corr(min_periods=30).abs().fillna(0.0).clip(0.0, 1.0)
    corr_values = corr.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(corr_values, 1.0)
    distance = np.clip(1.0 - corr_values, 0.0, 1.0)
    condensed = squareform(distance, checks=False)
    tree = linkage(condensed, method="complete")
    labels = fcluster(tree, t=1.0 - float(correlation_threshold), criterion="distance")
    return {name: int(label) for name, label in zip(names, labels)}


def _ic_set_summary(
    ic: pd.DataFrame,
    factors: Sequence[str],
    segments: Sequence[tuple[pd.Timestamp, pd.Timestamp]],
) -> dict[str, float]:
    combined = ic[list(factors)].mean(axis=1, skipna=True)
    sharpes = []
    means = []
    for start, end in segments:
        values = combined.loc[start:end].dropna()
        if len(values) < 20:
            continue
        std = float(values.std(ddof=1))
        sharpes.append(float(values.mean() / std * np.sqrt(PERIODS_PER_YEAR)) if std > 0 else -10.0)
        means.append(float(values.mean()))
    if not sharpes:
        return {"positive_segment_ratio": 0.0, "worst_sharpe": -10.0, "median_sharpe": -10.0}
    return {
        "positive_segment_ratio": float(np.mean(np.asarray(means) > 0.0)),
        "worst_sharpe": float(np.min(sharpes)),
        "median_sharpe": float(np.median(sharpes)),
    }


def beam_factor_sets(
    ic: pd.DataFrame,
    diagnostics: pd.DataFrame,
    clusters: Mapping[str, int],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    minimum_size: int = 4,
    maximum_size: int = 12,
    beam_width: int = 20,
    output_limit: int = 12,
) -> list[dict]:
    """Return stable cluster-constrained factor sets without exponential search."""

    eligible = diagnostics.loc[diagnostics["eligible"], "factor"].astype(str).tolist()
    if len(eligible) < minimum_size:
        return []
    segments = calendar_segments(start, end, years=2)
    rank = {name: index for index, name in enumerate(eligible)}
    seeds = [(name,) for name in eligible[: min(len(eligible), beam_width)]]
    beam = seeds
    completed: list[tuple[tuple, tuple[str, ...], dict]] = []
    for size in range(2, maximum_size + 1):
        candidates: dict[tuple[str, ...], tuple[tuple, dict]] = {}
        for current in beam:
            current_clusters = {clusters.get(name) for name in current}
            last_rank = max(rank[name] for name in current)
            for name in eligible:
                if rank[name] <= last_rank or name in current:
                    continue
                if clusters.get(name) in current_clusters:
                    continue
                proposal = tuple((*current, name))
                summary = _ic_set_summary(ic.loc[start:end], proposal, segments)
                key = (
                    summary["positive_segment_ratio"],
                    summary["worst_sharpe"],
                    summary["median_sharpe"],
                    -len(proposal),
                )
                candidates[proposal] = (key, summary)
        if not candidates:
            break
        ranked = sorted(
            ((key, factors, summary) for factors, (key, summary) in candidates.items()),
            key=lambda item: item[0], reverse=True,
        )
        beam = [factors for _, factors, _ in ranked[:beam_width]]
        if size >= minimum_size:
            completed.extend(ranked[:beam_width])
    best = sorted(completed, key=lambda item: item[0], reverse=True)[:output_limit]
    return [
        {"rank": index + 1, "factors": list(factors), **summary}
        for index, (_, factors, summary) in enumerate(best)
    ]


def factor_set_jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    a, b = set(left), set(right)
    union = a | b
    return float(len(a & b) / len(union)) if union else 1.0


class PortfolioEvaluator:
    """Exact causal evaluator for the existing 38-asset production structure."""

    def __init__(
        self,
        runner,
        *,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        trade_cost_rate: float = 0.0002,
        annual_fee: float = 0.001,
        ic_window: int = 60,
    ):
        self.runner = runner
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.trade_cost_rate = float(trade_cost_rate)
        self.annual_fee = float(annual_fee)
        self.ic_window = int(ic_window)
        self.dates = pd.DatetimeIndex(runner.cal)
        self.dates = self.dates[(self.dates >= self.start) & (self.dates <= self.end)]
        self._score_cache: dict[tuple[tuple[str, ...], str], pd.DataFrame] = {}
        self._asset_weight_cache: dict[tuple[pd.Timestamp, tuple[str, ...], str], pd.Series] = {}
        self._ledger_cache: dict[tuple[tuple[str, ...], PortfolioRecipe], pd.DataFrame] = {}

    def bounded(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> "PortfolioEvaluator":
        """Create an isolated evaluator for one declared train or test interval."""

        return PortfolioEvaluator(
            self.runner,
            start=start,
            end=end,
            trade_cost_rate=self.trade_cost_rate,
            annual_fee=self.annual_fee,
            ic_window=self.ic_window,
        )

    def _score_matrix(self, factors: Sequence[str], method: str) -> pd.DataFrame:
        factor_tuple = tuple(factors)
        key = (factor_tuple, method)
        if key in self._score_cache:
            return self._score_cache[key]
        missing = sorted(set(factor_tuple) - set(self.runner.ranks))
        if missing:
            raise KeyError(f"factor runner did not compute: {missing}")
        score = pd.DataFrame(np.nan, index=self.dates, columns=self.runner.u, dtype=float)
        ic = self.runner.ic[list(factor_tuple)]
        for date in self.dates:
            history = prepare_complete_history(
                ic.loc[:date].iloc[-self.ic_window:-1], minimum_observations=30
            )
            weights = factor_weights(history, method)
            if weights.empty:
                continue
            row = pd.Series(0.0, index=self.runner.u, dtype=float)
            for name, value in weights.items():
                row = row.add(
                    self.runner.ranks[name].loc[date].fillna(0.0) * float(value),
                    fill_value=0.0,
                )
            total = float(row.sum())
            if total > 0.0:
                row /= total
            score.loc[date] = row
        self._score_cache[key] = score
        return score

    def _risk_history(self, date: pd.Timestamp, pool: Sequence[str]) -> pd.DataFrame:
        calendar = pd.DatetimeIndex(self.runner.cal)
        calendar = calendar[
            (calendar >= date - pd.Timedelta(days=90)) & (calendar < date)
        ]
        history = self.runner.daily_ret.reindex(index=calendar, columns=list(pool))
        history = history.replace([np.inf, -np.inf], np.nan)
        eligible = history.columns[history.notna().sum(axis=0).ge(10)]
        if len(eligible) < 2:
            return history.iloc[0:0, 0:0]
        return history.loc[:, eligible].dropna(axis=0, how="any")

    def _asset_weights(
        self,
        date: pd.Timestamp,
        pool: Sequence[str],
        method: str,
    ) -> pd.Series:
        key = (pd.Timestamp(date), tuple(pool), method)
        if key in self._asset_weight_cache:
            return self._asset_weight_cache[key]
        history = self._risk_history(pd.Timestamp(date), pool)
        if history.shape[0] < 10 or history.shape[1] < 2:
            result = pd.Series(dtype=float)
        elif method == "equal":
            result = pd.Series(1.0 / history.shape[1], index=history.columns, dtype=float)
        elif method == "inverse_volatility":
            inverse = 1.0 / history.std(ddof=0).replace(0.0, np.nan)
            inverse = inverse.replace([np.inf, -np.inf], np.nan).dropna()
            result = inverse / inverse.sum() if len(inverse) else pd.Series(dtype=float)
        elif method == "erc":
            from optimization.risk_budgeting import RiskBudgetingOptimizer

            covariance_raw = history.cov().to_numpy(dtype=float)
            covariance = (
                0.70 * covariance_raw
                + 0.30 * np.diag(np.diag(covariance_raw))
            )
            try:
                values = RiskBudgetingOptimizer._erc_weights(
                    covariance, np.ones(history.shape[1])
                )
                result = pd.Series(values, index=history.columns, dtype=float)
            except (RuntimeError, ValueError):
                inverse = 1.0 / history.std(ddof=0).replace(0.0, np.nan)
                inverse = inverse.replace([np.inf, -np.inf], np.nan).dropna()
                result = inverse / inverse.sum() if len(inverse) else pd.Series(dtype=float)
        else:
            raise ValueError(f"unknown asset-weight method: {method}")
        if not result.empty:
            result = result / result.sum()
        self._asset_weight_cache[key] = result
        return result

    def weights(self, factors: Sequence[str], recipe: PortfolioRecipe) -> pd.DataFrame:
        score = self._score_matrix(factors, recipe.factor_weight)
        weights = pd.DataFrame(0.0, index=self.dates, columns=self.runner.u, dtype=float)
        sector_of = self.runner.env.sector_of
        for date in self.dates:
            row = score.loc[date].dropna()
            if len(row) < 2:
                continue
            eligible = self.runner.env.eligible_symbols(date)
            if row.reindex(eligible).notna().sum() < 2 * int(recipe.top_n):
                continue
            long_pool = select_pool(
                row,
                eligible=eligible,
                sector_of=sector_of,
                top_n=recipe.top_n,
                sector_cap=recipe.sector_cap,
                ascending=False,
            )
            short_pool = select_pool(
                row,
                eligible=eligible,
                sector_of=sector_of,
                top_n=recipe.top_n,
                sector_cap=recipe.sector_cap,
                ascending=True,
            )
            long_weights = self._asset_weights(date, long_pool, recipe.asset_weight)
            short_weights = self._asset_weights(date, short_pool, recipe.asset_weight)
            if long_weights.empty or short_weights.empty:
                continue
            weights.loc[date, long_weights.index] = long_weights.to_numpy()
            weights.loc[date, short_weights.index] -= short_weights.to_numpy()
        return weights

    def ledger_from_weights(
        self,
        weights: pd.DataFrame,
        *,
        cost_multiplier: float = 1.0,
    ) -> pd.DataFrame:
        weights = weights.reindex(index=self.dates, columns=self.runner.u).fillna(0.0)
        asset_returns = self.runner.daily_ret.reindex(index=self.dates, columns=self.runner.u)
        gross_return = (weights * asset_returns).sum(axis=1, min_count=1).fillna(0.0)
        turnover = weights.diff().abs().sum(axis=1)
        if len(turnover):
            turnover.iloc[0] = float(weights.iloc[0].abs().sum())
        trade_cost = turnover * self.trade_cost_rate * float(cost_multiplier)
        management_fee = pd.Series(
            self.annual_fee / PERIODS_PER_YEAR, index=self.dates, dtype=float
        )
        net_return = gross_return - trade_cost - management_fee
        ledger = pd.DataFrame({
            "gross_return": gross_return,
            "turnover": turnover,
            "trade_cost": trade_cost,
            "management_fee": management_fee,
            "net_return": net_return,
            "gross_exposure": weights.abs().sum(axis=1),
            "net_exposure": weights.sum(axis=1),
        })
        if len(ledger):
            ledger.iloc[0, ledger.columns.get_loc("gross_return")] = 0.0
            ledger.iloc[0, ledger.columns.get_loc("turnover")] = 0.0
            ledger.iloc[0, ledger.columns.get_loc("trade_cost")] = 0.0
            ledger.iloc[0, ledger.columns.get_loc("management_fee")] = 0.0
            ledger.iloc[0, ledger.columns.get_loc("net_return")] = 0.0
        ledger["nav"] = 1000.0 * (1.0 + ledger["net_return"]).cumprod()
        return ledger

    def ledger(self, factors: Sequence[str], recipe: PortfolioRecipe) -> pd.DataFrame:
        key = (tuple(factors), recipe)
        if key not in self._ledger_cache:
            self._ledger_cache[key] = self.ledger_from_weights(
                self.weights(factors, recipe)
            )
        return self._ledger_cache[key]

    def clear_transient_caches(self) -> None:
        """Release large score/ledger matrices after a completed research stage."""

        self._score_cache.clear()
        self._ledger_cache.clear()
        self._asset_weight_cache.clear()
