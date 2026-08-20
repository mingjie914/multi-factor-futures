"""Causal historical search helpers for production-style portfolio recipes.

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

from backtest.metrics import TRADING_DAYS_PER_YEAR
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from backtest.research_ledger import build_close_marked_ledger
from optimization.factor_weighting import (
    causal_history,
    combine_available_factor_scores,
    factor_weights,
    prepare_complete_history,
)
from optimization.portfolio_construction import (
    PortfolioConstraints,
    allocate_sleeve,
    causal_risk_window,
    combine_sleeves,
    prepare_risk_history,
    select_long_short_pools,
    select_pool as _shared_select_pool,
)
from optimization.costs import SimpleFuturesCost


PERIODS_PER_YEAR = TRADING_DAYS_PER_YEAR


@dataclass(frozen=True, order=True)
class PortfolioRecipe:
    """One finite, predeclared portfolio-construction recipe."""

    factor_weight: str = "lw_abs"
    top_n: int = 10
    sector_cap: int = 3
    asset_weight: str = "erc"
    asset_min_fraction: float = 0.005
    asset_max_fraction: float = 0.20
    gross_exposure: float = 2.0
    asset_max_overrides: tuple[tuple[str, float], ...] = ()
    sector_weight_caps: tuple[tuple[str, float], ...] = ()

    @property
    def name(self) -> str:
        cap = "none" if self.sector_cap <= 0 else str(self.sector_cap)
        name = (
            f"{self.factor_weight}__top{self.top_n}_bottom{self.top_n}"
            f"__cap{cap}__{self.asset_weight}"
        )
        if self.asset_min_fraction == 0.0 and self.asset_max_fraction == 1.0:
            name += "__assetcapnone"
        return name

    @property
    def constraints(self) -> PortfolioConstraints:
        return PortfolioConstraints(
            top_n_per_side=int(self.top_n),
            sector_count_cap=int(self.sector_cap),
            asset_min_fraction=float(self.asset_min_fraction),
            asset_max_fraction=float(self.asset_max_fraction),
            asset_max_overrides=dict(self.asset_max_overrides),
            sector_weight_caps=dict(self.sector_weight_caps),
            gross_exposure=float(self.gross_exposure),
            covariance_shrinkage=0.30,
            minimum_risk_observations=10,
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

    return _shared_select_pool(
        score,
        eligible=eligible,
        sector_of=sector_of,
        top_n=top_n,
        sector_count_cap=sector_cap,
        ascending=ascending,
    )


def performance_metrics(
    returns: pd.Series,
    *,
    periods_per_year: int = PERIODS_PER_YEAR,
    initial_anchor: bool = False,
) -> dict[str, float | int]:
    """Compute metrics from real return intervals, optionally after a NAV anchor."""
    periods = int(periods_per_year)
    if periods <= 0:
        raise ValueError("periods_per_year must be positive")
    numeric = pd.to_numeric(returns, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("performance returns must be finite")
    if initial_anchor:
        if numeric.empty or not np.isclose(float(numeric.iloc[0]), 0.0, atol=1e-12):
            raise ValueError("initial_anchor requires a leading zero return")
        numeric = numeric.iloc[1:]
    values = numeric
    if (values <= -1.0).any():
        raise ValueError("performance return cannot be <= -100%")
    if len(values) < 2:
        return {"observations": int(len(values))}
    growth = float((1.0 + values).prod())
    annual_return = growth ** (periods / len(values)) - 1.0
    annual_volatility = float(values.std(ddof=1) * np.sqrt(periods))
    nav = pd.Series(
        np.concatenate(([1.0], (1.0 + values).cumprod().to_numpy(dtype=float)))
    )
    max_drawdown = float((nav / nav.cummax() - 1.0).min())
    return {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": (
            float(values.mean() * periods / annual_volatility)
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
    *,
    initial_anchor: bool = False,
) -> dict[str, float | int]:
    """Summarize segment robustness without hiding the worst historical block."""

    rows = []
    for start, end in segments:
        returns = ledger.loc[start:end, "net_return"].copy()
        if len(returns) < 20:
            continue
        rows.append(performance_metrics(
            returns,
            initial_anchor=bool(
                initial_anchor
                and len(returns)
                and returns.index[0] == ledger.index[0]
            ),
        ))
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


def exhaustive_factor_set_shortlist(
    ic: pd.DataFrame,
    factors: Sequence[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    minimum_size: int = 2,
    per_size_limit: int = 2,
    global_limit: int = 12,
    batch_size: int = 2048,
) -> tuple[list[dict], dict[str, int]]:
    """Screen every subset by training IC and return a small exact-test shortlist.

    The exhaustive stage is deliberately limited to inexpensive training-only
    IC evidence.  The caller must run the returned shortlist through the exact
    portfolio engine before selecting a set.  Keeping candidates per factor
    count avoids giving middle-sized sets an advantage merely because they have
    many more possible combinations.
    """

    names = list(dict.fromkeys(str(name) for name in factors))
    missing = sorted(set(names) - set(ic.columns))
    if missing:
        raise KeyError(f"IC panel does not contain factors: {missing}")
    if not 2 <= int(minimum_size) <= len(names):
        raise ValueError("minimum_size must be between 2 and factor count")
    if min(int(per_size_limit), int(global_limit), int(batch_size)) <= 0:
        raise ValueError("shortlist limits and batch_size must be positive")

    train = ic.loc[pd.Timestamp(start):pd.Timestamp(end), names]
    train = train.replace([np.inf, -np.inf], np.nan)
    values = train.to_numpy(dtype=float)
    finite = np.isfinite(values)
    finite_values = finite.astype(float)
    filled = np.where(finite, values, 0.0)
    segments = calendar_segments(pd.Timestamp(start), pd.Timestamp(end), years=2)
    segment_positions = [
        np.flatnonzero((train.index >= left) & (train.index <= right))
        for left, right in segments
    ]
    segment_positions = [positions for positions in segment_positions if len(positions)]
    if not segment_positions:
        return [], {"examined_subsets": 0, "shortlisted_subsets": 0}

    best_by_size: dict[int, list[tuple[tuple, int, dict]]] = {}
    best_global: list[tuple[tuple, int, dict]] = []
    examined = 0
    maximum_mask = 1 << len(names)
    for batch_start in range(1, maximum_mask, int(batch_size)):
        masks_as_int = np.arange(
            batch_start, min(batch_start + int(batch_size), maximum_mask), dtype=np.uint64
        )
        masks = ((masks_as_int[:, None] >> np.arange(len(names), dtype=np.uint64)) & 1)
        sizes = masks.sum(axis=1).astype(int)
        admitted = sizes >= int(minimum_size)
        if not admitted.any():
            continue
        masks_as_int = masks_as_int[admitted]
        masks = masks[admitted].astype(float)
        sizes = sizes[admitted]
        examined += len(masks)

        sums = filled @ masks.T
        counts = finite_values @ masks.T
        combined = np.divide(
            sums,
            counts,
            out=np.full_like(sums, np.nan, dtype=float),
            where=counts > 0.0,
        )
        segment_sharpes = []
        segment_means = []
        for positions in segment_positions:
            segment = combined[positions]
            observations = np.isfinite(segment).sum(axis=0)
            means = np.nanmean(segment, axis=0)
            stds = np.nanstd(segment, axis=0, ddof=1)
            sharpes = np.divide(
                means,
                stds,
                out=np.full_like(means, -10.0),
                where=(observations >= 20) & np.isfinite(stds) & (stds > 0.0),
            ) * np.sqrt(PERIODS_PER_YEAR)
            segment_means.append(means)
            segment_sharpes.append(sharpes)
        means_matrix = np.vstack(segment_means)
        sharpes_matrix = np.vstack(segment_sharpes)
        positive_ratio = np.mean(means_matrix > 0.0, axis=0)
        worst_sharpe = np.min(sharpes_matrix, axis=0)
        median_sharpe = np.median(sharpes_matrix, axis=0)

        for column, mask_value in enumerate(masks_as_int.tolist()):
            size = int(sizes[column])
            summary = {
                "positive_segment_ratio": float(positive_ratio[column]),
                "worst_sharpe": float(worst_sharpe[column]),
                "median_sharpe": float(median_sharpe[column]),
            }
            key = (
                summary["positive_segment_ratio"],
                summary["worst_sharpe"],
                summary["median_sharpe"],
                -size,
            )
            row = (key, int(mask_value), summary)
            bucket = best_by_size.setdefault(size, [])
            bucket.append(row)
            bucket.sort(key=lambda item: item[0], reverse=True)
            del bucket[int(per_size_limit):]
            best_global.append(row)
        best_global.sort(key=lambda item: item[0], reverse=True)
        del best_global[int(global_limit):]

    selected: dict[int, tuple[tuple, int, dict]] = {}
    for bucket in best_by_size.values():
        for row in bucket:
            selected[row[1]] = row
    for row in best_global:
        selected[row[1]] = row
    ranked = sorted(selected.values(), key=lambda item: item[0], reverse=True)
    shortlist = []
    for rank, (_, mask_value, summary) in enumerate(ranked, 1):
        chosen = [name for bit, name in enumerate(names) if mask_value & (1 << bit)]
        shortlist.append({
            "rank": rank,
            "factor_count": len(chosen),
            "factors": chosen,
            **summary,
        })
    return shortlist, {
        "examined_subsets": int(examined),
        "shortlisted_subsets": len(shortlist),
    }


def factor_set_jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    a, b = set(left), set(right)
    union = a | b
    return float(len(a & b) / len(union)) if union else 1.0


class PortfolioEvaluator:
    """Causal evaluator sharing production selection and allocation semantics."""

    def __init__(
        self,
        runner,
        *,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        cost_model: SimpleFuturesCost | None = None,
        ic_window: int = 60,
        risk_lookback_calendar_days: int = 90,
    ):
        self.runner = runner
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.cost_model = cost_model or SimpleFuturesCost()
        self.ic_window = int(ic_window)
        self.risk_lookback_calendar_days = int(risk_lookback_calendar_days)
        if self.risk_lookback_calendar_days < 10:
            raise ValueError("risk_lookback_calendar_days must be at least 10")
        self.dates = pd.DatetimeIndex(runner.cal)
        self.dates = self.dates[(self.dates >= self.start) & (self.dates <= self.end)]
        self._score_cache: dict[tuple[tuple[str, ...], str], pd.DataFrame] = {}
        self._asset_weight_cache: dict[
            tuple[pd.Timestamp, tuple[str, ...], PortfolioRecipe], pd.Series
        ] = {}
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
            cost_model=self.cost_model,
            ic_window=self.ic_window,
            risk_lookback_calendar_days=self.risk_lookback_calendar_days,
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
        if method == "equal":
            weights = pd.Series(
                1.0 / len(factor_tuple), index=factor_tuple, dtype=float
            )
            for date in self.dates:
                score.loc[date] = combine_available_factor_scores(
                    {name: self.runner.ranks[name].loc[date] for name in factor_tuple},
                    weights,
                    self.runner.u,
                )
            self._score_cache[key] = score
            return score

        ic = self.runner.ic[list(factor_tuple)]
        for date in self.dates:
            history = prepare_complete_history(
                causal_history(ic, date, self.ic_window),
                minimum_observations=30,
            )
            weights = factor_weights(history, method)
            if weights.empty:
                continue
            score.loc[date] = combine_available_factor_scores(
                {name: self.runner.ranks[name].loc[date] for name in weights.index},
                weights,
                self.runner.u,
            )
        self._score_cache[key] = score
        return score

    def _risk_history(self, date: pd.Timestamp, pool: Sequence[str]) -> pd.DataFrame:
        history = causal_risk_window(
            self.runner.daily_ret,
            date,
            self.risk_lookback_calendar_days,
        )
        return history.reindex(columns=list(pool))

    def _risk_eligible(
        self,
        date: pd.Timestamp,
        candidates: Sequence[str],
        minimum_observations: int,
    ) -> list[str]:
        history = self._risk_history(date, candidates).replace(
            [np.inf, -np.inf], np.nan
        )
        return history.columns[
            history.notna().sum(axis=0).ge(int(minimum_observations))
        ].tolist()

    def _asset_weights(
        self,
        date: pd.Timestamp,
        pool: Sequence[str],
        recipe: PortfolioRecipe,
    ) -> pd.Series:
        key = (pd.Timestamp(date), tuple(pool), recipe)
        if key in self._asset_weight_cache:
            return self._asset_weight_cache[key]
        history = self._risk_history(pd.Timestamp(date), pool)
        constraints = recipe.constraints
        history = prepare_risk_history(
            history,
            pool,
            constraints.minimum_risk_observations,
        )
        result = allocate_sleeve(
            history,
            method=recipe.asset_weight,
            constraints=constraints,
            sector_of=self.runner.env.sector_of,
        )
        self._asset_weight_cache[key] = result
        return result

    def weights(self, factors: Sequence[str], recipe: PortfolioRecipe) -> pd.DataFrame:
        score = self._score_matrix(factors, recipe.factor_weight)
        weights = pd.DataFrame(0.0, index=self.dates, columns=self.runner.u, dtype=float)
        sector_of = self.runner.env.sector_of
        started = False
        for date in self.dates:
            row = score.loc[date].dropna()
            if len(row) < 2:
                if started:
                    raise RuntimeError(
                        f"{date.date()} factor score became unavailable after portfolio start"
                    )
                continue
            eligible = self._risk_eligible(
                date,
                self.runner.u,
                recipe.constraints.minimum_risk_observations,
            )
            if row.reindex(eligible).notna().sum() < 2 * int(recipe.top_n):
                if started:
                    raise RuntimeError(
                        f"{date.date()} eligible universe became insufficient after "
                        "portfolio start"
                    )
                continue
            long_pool, short_pool = select_long_short_pools(
                row,
                eligible=eligible,
                sector_of=sector_of,
                constraints=recipe.constraints,
            )
            long_weights = self._asset_weights(date, long_pool, recipe)
            short_weights = self._asset_weights(date, short_pool, recipe)
            combined = combine_sleeves(
                long_weights,
                short_weights,
                universe=self.runner.u,
                long_pool=long_pool,
                short_pool=short_pool,
                constraints=recipe.constraints,
                sector_of=sector_of,
            )
            weights.loc[date, combined.index] = combined.to_numpy()
            started = True
        if not started:
            raise RuntimeError(
                "portfolio could not be constructed anywhere in the requested interval"
            )
        return weights

    def ledger_from_weights(
        self,
        weights: pd.DataFrame,
        *,
        cost_multiplier: float = 1.0,
    ) -> pd.DataFrame:
        schedule = getattr(self.runner, "contract_schedule", None)
        schedule_getter = getattr(self.runner, "get_contract_schedule", None)
        if callable(schedule_getter):
            schedule = schedule_getter()
        result = build_close_marked_ledger(
            weights.reindex(index=self.dates, columns=self.runner.u),
            self.runner.daily_ret.reindex(
                index=self.dates, columns=self.runner.u
            ),
            **self.cost_model.ledger_parameters(),
            cost_multiplier=cost_multiplier,
            contract_schedule=schedule,
            decision_tradable=getattr(self.runner, "close_tradable", None),
            initial_nav=1000.0,
        )
        ledger = result.daily.copy()
        ledger["nav"] = ledger["nav_after"]
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
