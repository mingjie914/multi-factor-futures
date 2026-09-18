"""Frozen effective-factor subset selection for the intraday daily contract.

This workflow consumes the current admitted effective-factor library (not a
fixed-size candidate pool), computes an independent long-history interval with
required warm-up, records daily-signal clusters, and evaluates multiple forward
search paths under the production construction recipe. It
never mutates the effective library and never uses post-cutoff data for
selection. The input count is always discovered from the configured effective
library path, so a future library version needs no workflow change.
"""
from __future__ import annotations

import csv
import importlib.util
from itertools import combinations
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from core.config import load_config, validate_rank_buffer_route
from core.date_policy import factor_admission_start, research_cutoff
from core.period import iter_overlapping_chunks
from core.registry import list_registered
from core.sectors import FRAMEWORK_UNIVERSE, portfolio_selection_group_for
from optimization.costs import SimpleFuturesCost
from optimization.factor_weighting import (
    rank_information_coefficients,
)
from pipeline.runner import PipelineRunner
from research.artifacts import sha256_file
from research.effective_factor_library import load_library
from research.governance import factor_family
from research.portfolio_experiment_support import FactorPanelRunner, configured_futures_cost_model
from research.historical_portfolio_search import (
    PortfolioEvaluator,
    PortfolioRecipe,
    performance_metrics,
    robust_summary,
    robustness_key,
)


SELECTION_SCHEMA_VERSION = 4
CLUSTER_CORRELATION_THRESHOLD = 0.50
MIN_CROSS_SECTION = 10
N_IS_SEGMENTS = 4
COMPACT_MAX_FACTORS = 12
PORTFOLIO_SEARCH_MAX_FACTORS = 20
PORTFOLIO_SEARCH_EXACT_WIDTH = 3
PORTFOLIO_SEARCH_BEAM_WIDTH = 20
PORTFOLIO_CAP_GROUPS = {
    name: portfolio_selection_group_for(name) for name in FRAMEWORK_UNIVERSE
}


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty selection table: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _rank_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, method="average", pct=True)


def _daily_spearman_ic(
    factor: pd.DataFrame,
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *, minimum_cross_section: int = MIN_CROSS_SECTION,
) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    for date in dates:
        x = pd.to_numeric(factor.loc[date], errors="coerce")
        y = pd.to_numeric(returns.loc[date], errors="coerce")
        mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) < minimum_cross_section:
            continue
        xr = x.loc[mask].rank(method="average")
        yr = y.loc[mask].rank(method="average")
        corr = xr.corr(yr)
        if pd.notna(corr) and np.isfinite(float(corr)):
            values[pd.Timestamp(date)] = float(corr)
    return pd.Series(values, dtype=float).sort_index()


def _segments(dates: pd.DatetimeIndex) -> list[pd.DatetimeIndex]:
    chunks = np.array_split(np.asarray(dates), N_IS_SEGMENTS)
    return [pd.DatetimeIndex(chunk) for chunk in chunks if len(chunk)]


def _metric(values: pd.Series) -> tuple[float, float, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    ir = mean / std if np.isfinite(std) and std > 0.0 else float("nan")
    return mean, float(values.gt(0.0).mean()), ir


def _exposure_correlation(
    ranks: dict[str, pd.DataFrame], names: Iterable[str]
) -> pd.DataFrame:
    series = {}
    for name in names:
        frame = ranks[name]
        # Pandas 3 defaults to the new stack implementation, where the
        # legacy ``dropna`` argument is rejected.  The old implementation is
        # intentional here because the correlation panel must retain the
        # rectangular date×instrument missing-value positions.  Keep the
        # fallback for the project's supported Pandas 1.5+ range.
        try:
            stacked = frame.stack(dropna=False, future_stack=False)
        except TypeError:
            stacked = frame.stack(dropna=False)
        series[name] = stacked
    panel = pd.DataFrame(series)
    corr = panel.corr(min_periods=MIN_CROSS_SECTION * 3).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    return corr.clip(-1.0, 1.0)


def _cluster(corr: pd.DataFrame, names: list[str]) -> dict[str, int]:
    if len(names) == 1:
        return {names[0]: 1}
    values = corr.reindex(index=names, columns=names).to_numpy(dtype=float)
    values = np.nan_to_num(np.abs(values), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(values, 1.0)
    distance = np.clip(1.0 - values, 0.0, 1.0)
    tree = linkage(squareform(distance, checks=False), method="complete")
    labels = fcluster(
        tree,
        t=1.0 - CLUSTER_CORRELATION_THRESHOLD,
        criterion="distance",
    )
    return {name: int(label) for name, label in zip(names, labels)}


def _representative_key(row: dict) -> tuple:
    return (
        float(row.get("segment_positive_ratio", 0.0)),
        float(row.get("worst_segment_mean_ic", -np.inf)),
        float(row.get("mean_ic", -np.inf)),
        float(row.get("coverage", 0.0)),
        -float(row.get("rank_churn", np.inf)),
        str(row["factor"]),
    )


def _compact_representatives(rows: list[dict], max_count: int) -> list[str]:
    """Keep the strongest distinct-cluster representatives up to one limit."""
    ranked = sorted(rows, key=_representative_key, reverse=True)
    return [str(row["factor"]) for row in ranked[:max_count]]




def _portfolio_shortlist(rows: list[dict], limit: int = 5) -> list[dict]:
    """Keep diverse, non-dominated net-performance candidates without size bias."""
    valid = [r for r in rows if r["status"] == "evaluated"
             and r["segment_count"] >= 4 and r["positive_segment_ratio"] >= 0.6
             and r["median_sharpe"] > 0 and r["full_annual_return"] > 0]
    def objectives(row):
        return (*robustness_key(row), -float(row["annual_turnover"]))
    frontier = [r for r in valid if not any(
        all(a >= b for a, b in zip(objectives(other), objectives(r)))
        and any(a > b for a, b in zip(objectives(other), objectives(r)))
        for other in valid if other is not r)]
    chosen = []
    for row in sorted(frontier, key=objectives, reverse=True):
        members = set(row["factors"])
        if all(len(members & set(r["factors"])) / len(members | set(r["factors"])) < 0.85
               for r in chosen):
            chosen.append(row)
        if len(chosen) == limit:
            break
    return chosen


def _production_recipe(config) -> PortfolioRecipe:
    validate_rank_buffer_route(config, actual_holdings_supported=True)
    return PortfolioRecipe.from_config(config.production_portfolio)


def _configured_cost_model(config) -> SimpleFuturesCost:
    return configured_futures_cost_model(config)


def _run_portfolio_search(
    *, evaluator: PortfolioEvaluator, portfolio_ic: pd.DataFrame,
    representatives: list[str], recipe: PortfolioRecipe, segments: list[tuple],
    max_factors: int = PORTFOLIO_SEARCH_MAX_FACTORS,
    exact_width: int = PORTFOLIO_SEARCH_EXACT_WIDTH,
    beam_width: int = PORTFOLIO_SEARCH_BEAM_WIDTH,
    output: Path | None = None,
    cache_db=None,
) -> tuple[list[dict], list[dict], str]:
    """Multi-path forward proposals; exact net portfolios judge every size.

    Historical blocks are development evidence conditional on today's admitted
    pool. They are not independent out-of-sample folds.
    """
    pool = sorted(set(representatives))
    if len(pool) < 2 or min(exact_width, beam_width) < 1 or max_factors < 2:
        raise ValueError("portfolio search requires at least two factors and positive widths")
    cached = {}
    if cache_db is not None:
        cache_db.execute("""CREATE TABLE IF NOT EXISTS forward_results (
            signature VARCHAR PRIMARY KEY, result VARCHAR, net_returns DOUBLE[])""")
        cached = {key: json.loads(value) for key, value in cache_db.execute(
            "SELECT signature,result FROM forward_results").fetchall()}
    panel = portfolio_ic[pool]
    arrays = [panel.loc[left:right].to_numpy(dtype=float) for left, right in segments]
    positions = {name: i for i, name in enumerate(pool)}
    def proxy(members):
        ids = [positions[n] for n in members]
        means, sharpes = [], []
        for values in arrays:
            x = values[:, ids]
            count = np.isfinite(x).sum(axis=1)
            combined = np.divide(np.nansum(x, axis=1), count,
                                 out=np.full(len(x), np.nan), where=count > 0)
            combined = combined[np.isfinite(combined)]
            if len(combined) < 20:
                return (0.0, -10.0, -10.0)
            mean, std = combined.mean(), combined.std(ddof=1)
            means.append(mean)
            sharpes.append(mean / std * np.sqrt(252) if std > 0 else -10.0)
        return (float(np.mean(np.asarray(means) > 0)), float(min(sharpes)), float(np.median(sharpes)))
    beam = []
    preferred_parent = None
    evaluations, path = [], []
    limit = min(max_factors, len(pool))
    for size in range(2, limit + 1):
        proposals = (set(combinations(pool, 2)) if size == 2 else
                     {tuple(sorted((*members, n))) for members in beam for n in pool if n not in members})
        ranked = sorted(((proxy(m), m) for m in proposals), reverse=True)
        # Reserve one expansion of the previous exact winner. Merely sorting
        # its parent earlier would not affect the next global proxy ranking.
        if preferred_parent is not None:
            continuation = next((r for r in ranked if set(preferred_parent).issubset(r[1])), None)
            if continuation is not None:
                ranked.remove(continuation)
                ranked.insert(0, continuation)
        # Preserve multiple starting points; no permanent cluster representatives.
        beam = []
        for key, members in ranked:
            if all(len(set(members) & set(other)) / len(set(members) | set(other)) < 0.95
                   for other in beam):
                beam.append(members)
            if len(beam) >= beam_width:
                break
        if not beam:
            break
        exact_rows = []
        for members in beam:
            if any(len(set(members) & set(other["factors"])) /
                   len(set(members) | set(other["factors"])) >= 0.85 for other in exact_rows):
                continue
            key = json.dumps(members)
            if key in cached:
                row = cached[key]
                evaluations.append(row)
                if row["status"] == "evaluated":
                    exact_rows.append(row)
                if len(exact_rows) >= exact_width:
                    break
                continue
            started = time.perf_counter()
            row = dict(step=size, factor_count=size, added_factor="", factors=list(members),
                       status="rejected_runtime", error="", seconds=0.0,
                       segment_count=0, positive_segment_ratio=0.0, worst_sharpe=-10.0,
                       median_sharpe=-10.0, median_annual_return=-1.0, worst_drawdown=-1.0,
                       annual_turnover=float("nan"), full_annual_return=float("nan"),
                       full_sharpe=float("nan"), full_max_drawdown=float("nan"),
                       full_total_return=float("nan"))
            returns = []
            try:
                ledger = evaluator.ledger(members, recipe)
                if any(len(ledger.loc[left:right]) < 20 for left, right in segments):
                    raise RuntimeError("insufficient portfolio history in a declared block")
                metrics = performance_metrics(ledger["net_return"], initial_anchor=True)
                row.update(status="evaluated", **robust_summary(ledger, segments, initial_anchor=True),
                           annual_turnover=float(ledger["executed_traded_notional"].iloc[1:].mean() * 252),
                           full_annual_return=metrics["annual_return"], full_sharpe=metrics["sharpe"],
                           full_max_drawdown=metrics["max_drawdown"], full_total_return=metrics["total_return"])
                exact_rows.append(row)
                returns = ledger["net_return"].tolist()
            except (RuntimeError, ValueError) as exc:
                row["error"] = str(exc)
            row["seconds"] = time.perf_counter() - started
            if cache_db is not None:
                cache_db.execute("INSERT INTO forward_results VALUES (?,?,?)",
                                 [key, json.dumps(row), returns])
                cached[key] = row
            evaluations.append(row)
            if output is not None:
                _write_json(output / "portfolio_search_progress.json", evaluations)
            print(f"portfolio size={size} status={row['status']} seconds={row['seconds']:.1f}", flush=True)
            if len(exact_rows) >= exact_width:
                break
        if exact_rows:
            winner = max(exact_rows, key=robustness_key)
            path.append(winner)
            preferred_parent = tuple(winner["factors"])
            # Exact portfolio evidence influences the next size, while other
            # proxy paths survive even when three consecutive sizes deteriorate.
            beam = list(dict.fromkeys([tuple(r["factors"]) for r in
                        sorted(exact_rows, key=robustness_key, reverse=True)] + beam))[:beam_width]
    return path, evaluations, ("candidate_pool_exhausted" if limit == len(pool)
                              else f"declared_compute_budget_size_{limit}")


def _run_budgeted_pool_search(*, evaluator, portfolio_ic, pool, recipe, segments,
                              clusters, db, output, budget=512, beam_width=6,
                              round_width=16, cache_namespace=None, search_sizes=(),
                              initial_candidates=(), search_seed=20260915):
    """Explicit research-only multi-start search; never changes default selection.

    Portfolio-first recipes start with exact single-factor eligibility and net
    returns; other recipes use IC seed pairs. Proxies only order proposals;
    exact net backtests choose the moving beam and the final Pareto archive.
    Growth lanes survive temporary deterioration. Size is not a stop rule.
    """
    names = sorted(set(pool))
    custom = bool(getattr(recipe, "investment_universe", ()))
    periods = recipe.periods_per_year if custom else 252
    portfolio_proposals = custom and recipe.combination_order == "portfolio_first"
    if (search_sizes or initial_candidates) and not portfolio_proposals:
        raise ValueError("size-stratified search requires a portfolio-first configured recipe")
    if initial_candidates and not search_sizes:
        raise ValueError("initial_candidates require size-stratified search")
    if any(type(n) is not int or n < 2 for n in search_sizes) or len(set(search_sizes)) != len(search_sizes):
        raise ValueError("search_sizes must contain unique integers of at least two")
    if any(len(m) < 2 or len(set(m)) != len(m) or set(m) - set(names) for m in initial_candidates):
        raise ValueError("initial_candidates must contain unique admitted members")
    if custom and not cache_namespace:
        raise ValueError("configured search requires a frozen data/code cache namespace")
    if custom and recipe.factor_groups:
        raise ValueError("subset search cannot split parameter groups; evaluate complete declared groups explicitly")
    if len(names) < 2 or budget < len(names) or not 1 <= beam_width <= round_width:
        raise ValueError("budget must cover the full pool and widths must be positive")
    # IC at T contains T+1 returns. Drop the last selection-day label even if
    # the caller supplied a panel extending into the observation period.
    panel = portfolio_ic.loc[:evaluator.end, names].iloc[:-1]
    arrays = [panel.loc[left:right].to_numpy(dtype=float) for left, right in segments]
    positions, proxies = {n: i for i, n in enumerate(names)}, {}
    def proxy(members):
        if members not in proxies:
            means, sharpes = [], []
            for values in arrays:
                x = values[:, [positions[n] for n in members]]
                count = np.isfinite(x).sum(axis=1)
                y = np.divide(np.nansum(x, axis=1), count,
                              out=np.full(len(x), np.nan), where=count > 0)
                y = y[np.isfinite(y)]
                if len(y) < 20:
                    proxies[members] = (0.0, -10.0, -10.0)
                    break
                mean, std = y.mean(), y.std(ddof=1)
                means.append(mean)
                sharpes.append(mean / std * np.sqrt(periods) if std > 0 else -10.0)
            else:
                proxies[members] = (float(np.mean(np.asarray(means) > 0)),
                                    float(min(sharpes)), float(np.median(sharpes)))
        return proxies[members]
    db.execute("""CREATE TABLE IF NOT EXISTS pool_results (
        signature VARCHAR PRIMARY KEY, result VARCHAR, dates DATE[],
        nav DOUBLE[], net_returns DOUBLE[])""")
    cached = {key: json.loads(value) for key, value in db.execute(
        "SELECT signature,result FROM pool_results").fetchall()}
    evaluations, visited, path, single_returns = [], set(), [], {}
    def evaluate(members, phase, iteration, size_target=None):
        members = tuple(sorted(members))
        if members in visited or len(evaluations) >= budget:
            return None
        key = json.dumps({"algorithm": "portfolio-rounded-singleton-seeds-v3", "scope": cache_namespace, "recipe": recipe.to_dict(),
                          "start": str(evaluator.start), "end": str(evaluator.end),
                          "cost": evaluator.cost_model.ledger_parameters(),
                          "ic_window": evaluator.ic_window,
                          "risk_lookback_calendar_days": evaluator.risk_lookback_calendar_days,
                          "segments": [(str(a), str(b)) for a, b in segments], "factors": members},
                         sort_keys=True) if custom else json.dumps(members)
        row = cached.get(key)
        if row is None:
            started, cpu = time.perf_counter(), time.process_time()
            row = {"factors": list(members), "factor_count": len(members),
                   "phase": phase, "step": iteration, "status": "rejected_runtime",
                   "error": "", "segment_count": 0, "positive_segment_ratio": 0.0,
                   "worst_sharpe": -10.0, "median_sharpe": -10.0,
                   "median_annual_return": -1.0, "worst_drawdown": -1.0,
                   "annual_turnover": None, "full_annual_return": None}
            if size_target is not None:
                row["search_size_target"] = size_target
            dates, nav, returns = [], [], []
            try:
                if portfolio_proposals and len(members) == 1:
                    values = evaluator.runner.factor_values[members[0]].loc[evaluator.start:evaluator.end]
                    counts = np.isfinite(values.to_numpy(dtype=float)).sum(axis=1)
                    required = recipe.selection_count * (2 if recipe.position_mode == "long_short" else 1)
                    bad = values.index[counts < required]
                    if len(bad):
                        raise RuntimeError(f"fixed-pool coverage below {required} on {len(bad)} days; first={bad[0].date()}")
                    represented = values.round(recipe.score_decimals) if recipe.score_decimals is not None else values
                    if not represented.nunique(axis=1, dropna=True).gt(1).any():
                        raise RuntimeError("no cross-sectional information after configured rounding; tie-selected fixed basket")
                ledger = evaluator.ledger(members, recipe)
                if ledger.index.max() > evaluator.end:
                    raise AssertionError("selection ledger escaped the cutoff")
                metrics = performance_metrics(ledger["net_return"], initial_anchor=True, periods_per_year=periods)
                if any(len(ledger.loc[left:right]) < 20 for left, right in segments):
                    raise RuntimeError("insufficient history in a declared block")
                row.update(status="evaluated", **robust_summary(ledger, segments, initial_anchor=True, periods_per_year=periods),
                           annual_turnover=float(ledger["executed_traded_notional"].iloc[1:].mean()*periods),
                           **{f"full_{k}": v for k, v in metrics.items()})
                dates, nav, returns = list(ledger.index.date), ledger["nav"].tolist(), ledger["net_return"].tolist()
            except (RuntimeError, ValueError) as exc:
                row["error"] = str(exc)
            row.update(seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu)
            db.execute("INSERT INTO pool_results VALUES (?,?,?,?,?)",
                       [key, json.dumps(row), dates, nav, returns])
            cached[key] = row
            evaluator.clear_transient_caches()
        if portfolio_proposals and len(members) == 1 and row["status"] == "evaluated":
            dates, returns = db.execute("SELECT dates,net_returns FROM pool_results WHERE signature=?", [key]).fetchone()
            single_returns[members[0]] = pd.Series(returns, index=pd.DatetimeIndex(dates))
        visited.add(members)
        evaluations.append(row)
        _write_json(output / "pool_progress.json", {
            "attempted": len(evaluations), "budget": budget,
            "cached_total": len(cached), "phase": phase, "round": iteration,
            "search_size_target": size_target,
            "evaluated": sum(r["status"] == "evaluated" for r in evaluations),
            "covered_factors": len({n for r in evaluations for n in r["factors"]}),
            "successful_covered_factors": len({n for r in evaluations if r["status"] == "evaluated"
                                               for n in r["factors"]})})
        print(f"pool {len(evaluations)}/{budget} {phase} {len(members)}f {row['status']}", flush=True)
        return row
    def choose(rows, count, existing=()):
        selected = list(existing)
        for row in sorted((r for r in rows if r and r["status"] == "evaluated"),
                          key=lambda r: (*robustness_key(r), -r["annual_turnover"]), reverse=True):
            members = tuple(row["factors"])
            if all(len(set(members)&set(other))/len(set(members)|set(other)) < .95 for other in selected):
                selected.append(members)
            if len(selected) >= count:
                break
        return selected
    single_rows = []
    if portfolio_proposals:
        single_rows = [evaluate((name,), "single_factor_eligibility", 0) for name in names]
        # Exclude data/construction failures only, never discard a valid factor
        # for low stand-alone return. Averaged net returns are a proposal proxy:
        # final caps, netting and drift costs still require the shared ledger.
        names = sorted(single_returns)
        panel = pd.DataFrame(single_returns).loc[:evaluator.end]
        arrays = [panel.loc[left:right].to_numpy(dtype=float) for left, right in segments]
        positions, proxies = {n: i for i, n in enumerate(panel.columns)}, {}
        _write_json(output / "factor_eligibility.json", {
            "qualified": names, "evaluations": single_rows,
            "performance_filter": False, "observation_used": False})
    if search_sizes:
        # Reserve exact evaluations for disjoint size bands, including large
        # starting portfolios. Singleton qualification never competes with them.
        sizes = sorted({min(n, len(names)) for n in search_sizes if len(names) >= 2})
        if not sizes:
            raise ValueError("size-stratified search requires two qualified factors")
        if any(set(m) - set(names) for m in initial_candidates):
            raise ValueError("initial candidate failed current fixed-pool qualification")
        target = lambda n: min(sizes, key=lambda s: (abs(n-s), s))
        bands = {s: [n for n in range(2, len(names)+1) if target(n) == s] for s in sizes}
        remaining = budget-len(evaluations)
        if remaining < len(sizes):
            raise ValueError("budget must reserve at least one evaluation per size band")
        quotas = {s: remaining//len(sizes)+int(i < remaining % len(sizes)) for i, s in enumerate(sizes)}
        if any(quotas[s] < len(bands[s])+sum(target(len(m)) == s for m in initial_candidates) for s in sizes):
            raise ValueError("size-band budget must cover every cardinality and initial candidate")
        used, rows_by_size, exploration = {s: 0 for s in sizes}, {s: [] for s in sizes}, {}
        rng = np.random.default_rng(search_seed)

        def assess(members, phase, iteration):
            size = target(len(members))
            if used[size] >= quotas[size]:
                return None
            row = evaluate(members, phase, iteration, size)
            if row is not None:
                used[size] += 1
                rows_by_size[size].append(row)
            return row

        ranked = [r["factors"][0] for r in sorted(
            (r for r in single_rows if r and r["status"] == "evaluated"),
            key=lambda r: (*robustness_key(r), -r["annual_turnover"]), reverse=True)]
        cluster_members = {}
        for name in ranked:
            cluster_members.setdefault(clusters[name], []).append(name)
        diversified = [group[i] for i in range(max(map(len, cluster_members.values())))
                       for group in cluster_members.values() if i < len(group)]
        for members in initial_candidates:
            assess(members, "prior_locked_seed", 0)
        for size in sizes:
            seeds = [ranked[:count] for count in bands[size]] + [diversified[:size]]
            seeds += [list(rng.choice(names, size=size, replace=False)) for _ in range(2)]
            for members in seeds:
                assess(members, "size_seed", 0)
            for members in initial_candidates:
                if len(members) <= size:
                    extended = list(members)+[n for n in diversified if n not in members][:size-len(members)]
                    assess(extended, "incumbent_size_seed", 0)
        active, iteration = set(sizes), 0
        def write_size_progress():
            _write_json(output / "size_search.json", {
                "seed": search_seed, "targets": sizes, "bands": bands,
                "quotas": quotas, "attempted_by_size": used, "round": iteration,
                "proposal_sample_per_parent": 64, "exact_batch_per_size": 4,
                "seed_every_cardinality": True,
                "extend_initial_candidates": True,
                "initial_candidates": initial_candidates})
        write_size_progress()
        while active and len(evaluations) < budget:
            iteration += 1
            round_rows = []
            for size in sizes:
                if size not in active:
                    continue
                if used[size] >= quotas[size]:
                    active.remove(size)
                    continue
                parents = choose(rows_by_size[size], 2)
                if size in exploration and exploration[size] not in parents:
                    parents.append(exploration[size])
                proposals = set()
                low, high = min(bands[size]), max(bands[size])
                for parent in parents:
                    outside = sorted(set(names)-set(parent))
                    operations = (["add"] if outside and len(parent) < high else [])
                    operations += (["delete"] if len(parent) > low else [])
                    operations += (["swap"] if outside else [])
                    for _ in range(64):
                        if not operations:
                            break
                        operation = str(rng.choice(operations))
                        members = set(parent)
                        if operation in ("delete", "swap"):
                            members.remove(str(rng.choice(parent)))
                        if operation in ("add", "swap"):
                            members.add(str(rng.choice(outside)))
                        proposals.add(tuple(sorted(members)))
                # Fresh starts let a size band escape a weak local neighborhood.
                for _ in range(4):
                    count = int(rng.integers(low, high+1))
                    proposals.add(tuple(sorted(rng.choice(names, size=count, replace=False))))
                ordered = sorted(proposals-visited, key=lambda m: (proxy(m), m), reverse=True)
                if not ordered:
                    active.remove(size)
                    continue
                batch = ordered[:3]
                if len(ordered) > 3:
                    batch.append(ordered[int(rng.integers(3, len(ordered)))])
                for members in batch:
                    row = assess(members, "size_add_delete_swap", iteration)
                    if row is not None:
                        round_rows.append(row)
                        if row["status"] == "evaluated":
                            exploration[size] = tuple(row["factors"])
            if round_rows:
                valid = [r for r in round_rows if r["status"] == "evaluated"]
                if valid:
                    path.append(max(valid, key=robustness_key))
            write_size_progress()
        reason = "exact_backtest_budget_exhausted" if len(evaluations) >= budget else "sampled_size_neighborhoods_exhausted"
        _write_json(output / "pool_selection.json", {
            "stop_reason": reason, "budget": budget, "attempted": len(evaluations),
            "scope": sorted(set(pool)), "constructible_scope": names,
            "proxy": "mean_single_factor_net_returns", "search_sizes": sizes,
            "shortlist": _portfolio_shortlist(evaluations), "path": path,
            "observation_used_for_selection": False})
        return path, evaluations, reason
    pairs = sorted(combinations(names, 2), key=lambda m: (proxy(m), m), reverse=True)
    seeds = []
    for name in names if pairs else ():
        pair = next((m for m in pairs if name in m and clusters[m[0]] != clusters[m[1]]),
                    next(m for m in pairs if name in m))
        if pair not in seeds:
            seeds.append(pair)
    seed_rows = [evaluate(m, "seed_coverage", 0) for m in seeds]
    if portfolio_proposals and len(names) >= 2:
        seed_rows.append(evaluate(tuple(names), "full_qualified_pool", 0))
    beam = choose(single_rows + seed_rows, beam_width)
    iteration = 0
    while beam and len(evaluations) < budget:
        iteration += 1
        proposals, growth = set(), set()
        for parent in beam:
            added = {tuple(sorted((*parent, n))) for n in names if n not in parent} - visited
            proposals.update(added)
            if added:
                growth.add(max(added, key=lambda m: (proxy(m), m)))
            for old in parent:
                remainder = tuple(n for n in parent if n != old)
                if len(remainder) >= 2:
                    proposals.add(remainder)
                proposals.update(tuple(sorted((*remainder, n))) for n in names if n not in parent)
        proposals -= visited
        if not proposals:
            break
        ordered = sorted(growth, key=lambda m: (proxy(m), m), reverse=True)
        ordered += sorted(proposals-growth, key=lambda m: (proxy(m), m), reverse=True)
        rows = [evaluate(m, "grow" if m in growth else "add_delete_swap", iteration)
                for m in ordered[:round_width] if len(evaluations) < budget]
        grown = choose([r for r in rows if r and r["phase"] == "grow"], max(1, beam_width//2))
        incumbents = [r for r in evaluations if tuple(r["factors"]) in beam]
        beam = choose(rows + incumbents, beam_width, grown)
        if rows and any(r and r["status"] == "evaluated" for r in rows):
            path.append(max((r for r in rows if r and r["status"] == "evaluated"), key=robustness_key))
    reason = "exact_backtest_budget_exhausted" if len(evaluations) >= budget else "reachable_proposals_exhausted"
    _write_json(output / "pool_selection.json", {
        "stop_reason": reason, "budget": budget, "attempted": len(evaluations),
        "scope": sorted(set(pool)), "constructible_scope": names,
        "proxy": "mean_single_factor_net_returns" if portfolio_proposals else "H1_IC",
        "shortlist": _portfolio_shortlist(evaluations), "path": path,
        "observation_used_for_selection": False})
    return path, evaluations, reason


def _load_library(
    config,
    *,
    source_run_dir: str | None = None,
    allowed_horizons: tuple[int, ...] | None = None,
) -> tuple[Path, list[dict]]:
    if source_run_dir is not None:
        root = Path(source_run_dir)
        if not root.is_absolute():
            root = Path(__file__).resolve().parents[1] / root
        root = root.resolve()
        summary_path = root / "validation_summary.json"
        contract_path = root / "run_contract.json"
        passed_path = root / "passed_factors.csv"
        if not summary_path.exists() or not contract_path.exists() or not passed_path.exists():
            raise FileNotFoundError(
                "common-horizon selection requires a finalized validation run: "
                f"{root}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        horizon_mode = str(summary.get("horizon_mode", ""))
        horizon = int(summary.get("common_horizon", 0) or 0)
        if (
            horizon_mode != "common_horizon"
            or horizon < 1
            or (allowed_horizons is not None and horizon not in allowed_horizons)
        ):
            raise ValueError(
                "selection source is not the requested common-horizon run: "
                f"mode={horizon_mode!r}, horizon={horizon}"
            )
        if contract.get("horizon_policy", {}).get("common_horizon") != horizon:
            raise ValueError("validation run and summary horizon contracts disagree")
        frame = pd.read_csv(passed_path, encoding="utf-8-sig")
        if "final_pass" not in frame or "factor" not in frame:
            raise ValueError("passed_factors.csv is missing the final-pass schema")
        frame = frame.loc[frame["final_pass"].astype(bool)].copy()
        if frame.empty:
            raise ValueError("common-horizon validation run has no passed factors")
        rows = []
        for record in frame.to_dict(orient="records"):
            name = str(record["factor"])
            ic = float(record.get("is_ic", 0.0) or 0.0)
            rows.append({
                "factor": name,
                "status": "validation_passed",
                "signal_frequency": "daily",
                "input_bar_frequency": "",
                "family": str(record.get("family", "") or factor_family(name)),
                "registered_horizons": str(record.get("registered_horizons", "")),
                "best_period": horizon,
                "direction": 1 if ic >= 0.0 else -1,
                "source_run": root.name,
                "oos_ic": record.get("oos_ic"),
            })
        names = {str(row["factor"]) for row in rows}
        if len(names) != len(rows):
            raise ValueError("common-horizon passed factors contain duplicates")
        return passed_path.resolve(), rows

    path = Path(config.factor_library.path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    payload = load_library(path)
    factors = [
        row for row in payload.get("factors", [])
        if isinstance(row, dict) and row.get("status") == "effective"
    ]
    if not factors:
        raise ValueError("effective factor library has no effective members")
    names = {str(row.get("factor")) for row in factors}
    if len(names) != len(factors):
        raise ValueError("effective factor library contains duplicate names")
    for row in factors:
        period = int(row.get("best_period", 0) or 0)
        if period < 1 or (
            allowed_horizons is not None and period not in allowed_horizons
        ):
            raise ValueError(
                f"factor {row.get('factor')!r} has invalid best period {period}"
            )
        if str(row.get("signal_frequency")) != "daily":
            raise ValueError(
                f"factor {row.get('factor')!r} is not daily: "
                f"{row.get('signal_frequency')!r}"
            )
    return path.resolve(), factors


def run_effective_factor_selection(
    *,
    run_id: str,
    config_path: str = "config/default.yaml",
    max_compact_factors: int = COMPACT_MAX_FACTORS,
    source_run_dir: str | None = None,
    common_horizon: int | None = None,
    group_by_best_period: bool = False,
    selection_start: str = "2016-03-31",
    candidate_names: list[str] | None = None,
    portfolio_search_budget: int = 512,
    search_end: str | None = None,
    resume: bool = False,
    search_sizes: tuple[int, ...] = (),
    initial_candidates: list[list[str]] | None = None,
    search_seed: int = 20260915,
) -> Path:
    """Run governed effective-library subset selection and write one immutable run."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(run_id)):
        raise ValueError("run_id only allows letters, numbers, dot, underscore and hyphen")
    if int(max_compact_factors) < 1:
        raise ValueError("max_compact_factors must be positive")
    if source_run_dir is not None:
        if common_horizon is None or int(common_horizon) < 1:
            raise ValueError(
                "common-horizon selection requires a positive common_horizon"
            )
        selection_horizons = (int(common_horizon),)
    else:
        selection_horizons = None

    project = Path(__file__).resolve().parents[1]
    output = project / "runs" / "factor_selection" / str(run_id)
    config = load_config(config_path)
    custom = bool(config.production_portfolio.investment_universe)
    if (search_sizes or initial_candidates) and (not custom or config.production_portfolio.combination_order != "portfolio_first"):
        raise ValueError("size-stratified search requires a portfolio-first configured recipe")
    if initial_candidates and not search_sizes:
        raise ValueError("initial_candidates require size-stratified search")
    if any(type(n) is not int or n < 2 for n in search_sizes) or len(set(search_sizes)) != len(search_sizes):
        raise ValueError("search_sizes must contain unique integers of at least two")
    if (search_end is not None or resume) and not custom:
        raise ValueError("bounded/resumable search requires an explicit strategy configuration")
    if custom and (source_run_dir is not None or group_by_best_period):
        raise ValueError("configured portfolio selection requires the mixed daily effective pool")
    if custom and config.production_portfolio.factor_groups:
        raise ValueError("subset search cannot split parameter groups; evaluate complete declared groups explicitly")
    library_path, library_rows = _load_library(
        config,
        source_run_dir=source_run_dir,
        allowed_horizons=selection_horizons,
    )
    if candidate_names is not None:
        if not candidate_names or len(set(candidate_names)) != len(candidate_names) or set(candidate_names) - {str(r['factor']) for r in library_rows}:
            raise ValueError("candidate_names must be a unique non-empty effective-library subset")
        library_rows = [r for r in library_rows if r["factor"] in candidate_names]
    if selection_horizons is None:
        selection_horizons = tuple(sorted({
            int(row["best_period"]) for row in library_rows
        }))
    if (output / "run_contract.json").exists():
        raise FileExistsError("finalized selection runs are immutable")
    output.mkdir(parents=True, exist_ok=resume)
    runner = PipelineRunner(config=config)
    selection_start = (factor_admission_start(config) if source_run_dir is not None
                       else pd.Timestamp(selection_start))
    cutoff = research_cutoff(config)
    selection_end = pd.Timestamp(search_end) if search_end is not None else cutoff
    if not selection_start <= selection_end <= cutoff:
        raise ValueError("search_end must be between selection_start and research_cutoff")
    warmup_days = int(config.validation_policy.warmup_days_by_frequency["daily"])
    factor_start = selection_start - pd.Timedelta(days=warmup_days)
    source_fingerprint = runner.data_manager.source.checkpoint_source_fingerprint(factor_start, cutoff)
    selection_dates = pd.DatetimeIndex(
        runner.data_manager.get_calendar(selection_start, selection_end)
    )
    requested_dates = pd.DatetimeIndex(
        runner.data_manager.get_calendar(factor_start, cutoff)
    )
    if len(selection_dates) < 80:
        raise ValueError("factor-set selection requires at least 80 trading days")
    universe = pd.Index(config.universe)
    names = sorted(str(row["factor"]) for row in library_rows)
    if any(len(m) < 2 or len(set(m)) != len(m) or set(m) - set(names) for m in (initial_candidates or [])):
        raise ValueError("initial_candidates must contain unique admitted members")
    if search_sizes and portfolio_search_budget < len(names)+len(search_sizes):
        raise ValueError("budget must reserve at least one evaluation per size band")
    if custom:
        import duckdb
        import scipy
        from factors.numerics import factor_kernel_contract
        native = importlib.util.find_spec("_mf_factor_kernels")
        runtime = {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                   "scipy": scipy.__version__, "duckdb": duckdb.__version__, **factor_kernel_contract(),
                   "native_binary_sha256": sha256_file(Path(native.origin)) if native and native.origin else None}
        plan = {
            "factors": library_rows, "library_sha256": sha256_file(library_path),
            "reference_universe": list(universe), "recipe": _production_recipe(config).to_dict(),
            "costs": _configured_cost_model(config).ledger_parameters(),
            "selection_start": str(selection_start.date()), "selection_end": str(selection_end.date()),
            "evaluation_end": str(cutoff.date()), "exact_budget": portfolio_search_budget,
            "beam_width": None if search_sizes else 6, "round_width": None if search_sizes else 16,
            "source_fingerprint": source_fingerprint,
            "size_search": {"targets": list(search_sizes), "random_seed": search_seed,
                            "initial_candidates": initial_candidates or [],
                            "proposal_sample_per_parent": 64, "exact_batch_per_size": 4,
                            "seed_every_cardinality": True,
                            "extend_initial_candidates": True,
                            "incumbents_per_size": 2, "exploration_paths_per_size": 1} if search_sizes else None,
            "factor_code": FactorPanelRunner._source_tree_fingerprint(),
            "runtime": runtime,
            "code": {str(p.relative_to(project)): sha256_file(p) for p in (
                Path(__file__), project / "research/historical_portfolio_search.py",
                project / "optimization/portfolio_construction.py", project / "backtest/research_ledger.py",
                project / "optimization/costs.py", project / "optimization/factor_weighting.py")},
            "resolved_config": json.loads(config.model_dump_json() if hasattr(config, "model_dump_json") else config.json()),
            "historical_role": "development_and_conditional_validation_on_current_effective_pool",
            "method_search": False, "direction_search": False, "observation_used_for_selection": False,
        }
        plan = json.loads(json.dumps(plan, default=str))
        plan_path = output / "search_plan.json"
        if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise RuntimeError("frozen selection plan changed")
        _write_json(plan_path, plan)
    factor_compute_started = time.perf_counter()
    from factors.library.intraday import clear_transient_data_caches
    if custom:
        factor_panel = FactorPanelRunner(names, start=factor_start, end=cutoff,
            factor_directions={str(r["factor"]): int(r["direction"]) for r in library_rows},
            checkpoint_dir=output / "factor_panel_checkpoint", config=config,
            universe=list(universe), retain_values=True)
        raw = factor_panel.factor_values
        factor_timings = [row for chunk in factor_panel.factor_chunk_timings for row in chunk["factor_timings"]]
        source_fingerprint = factor_panel._checkpoint_contract(names)["source_fingerprint"]
    else:
        raw = {}
        for target_dates, chunk_dates in iter_overlapping_chunks(requested_dates, 400, 128):
            part = FactorPanelRunner._compute_part(
                runner.factor_engine, names, chunk_dates, universe, raw
            )
            for name, frame in part.items():
                if name not in raw:
                    raw[name] = pd.DataFrame(np.nan, index=requested_dates, columns=universe)
                raw[name].loc[target_dates] = frame.reindex(index=target_dates, columns=universe)
            runner.factor_engine.clear_cache()
            clear_transient_data_caches()
            print(f"factor panel through {target_dates[-1].date()}: {len(raw)}/{len(names)}", flush=True)
        factor_timings = list(runner.factor_engine.computation_timings)
    factor_compute_seconds = time.perf_counter() - factor_compute_started
    if set(raw) != set(names):
        raise ValueError("factor engine did not return all admitted factor names")
    registry = list_registered("factor").get("factor", {})
    missing_registry = sorted(set(names) - set(registry))
    if missing_registry:
        raise ValueError("effective factors are not registered: " + ", ".join(missing_registry))
    for row in library_rows:
        name = str(row["factor"])
        declared_input = str(
            getattr(registry[name], "input_bar_frequency", "1min")
        )
        stored_input = str(row.get("input_bar_frequency", "") or "")
        if stored_input and stored_input != declared_input:
            raise ValueError(
                f"factor {name!r} input bar metadata {stored_input!r} does not "
                f"match registered implementation {declared_input!r}"
            )
        row["input_bar_frequency"] = declared_input
    directions = {str(row["factor"]): int(row["direction"]) for row in library_rows}
    periods = {str(row["factor"]): int(row["best_period"]) for row in library_rows}
    reference_universe = list(universe)
    if custom:
        investment = config.production_portfolio.investment_universe
        if set(investment) - set(universe):
            raise ValueError("investment roots require qualification in the factor reference universe")
        universe = pd.Index(investment)
        raw = {name: frame.reindex(columns=universe) for name, frame in raw.items()}

    returns = {
        horizon: runner.data_manager.get_forward_returns(
            requested_dates, universe, period=horizon
        )
        for horizon in selection_horizons
    }
    segments = _segments(selection_dates)
    all_diagnostics: list[dict] = []
    ranks: dict[str, pd.DataFrame] = {}
    full_ranks: dict[str, pd.DataFrame] = {}
    for row in library_rows:
        name = str(row["factor"])
        horizon = periods[name]
        decimals = config.production_portfolio.score_decimals if custom else None
        raw_rank = _rank_frame(raw[name].round(decimals) if decimals is not None else raw[name])
        direction = directions[name]
        full_ranks[name] = raw_rank if direction == 1 else (1.0 - raw_rank)
        ranks[name] = full_ranks[name].loc[selection_dates]
        ic = _daily_spearman_ic(
            ranks[name],
            returns[horizon].loc[selection_dates],
            selection_dates,
            minimum_cross_section=min(MIN_CROSS_SECTION, len(universe)) if custom else MIN_CROSS_SECTION,
        )
        if custom:
            # Exclude labels whose H-day return ends after the search boundary.
            ic = ic.reindex(selection_dates[:-horizon]).dropna()
        segment_values = []
        for segment in segments:
            mean, positive_ratio, ir = _metric(ic.reindex(segment))
            segment_values.append((mean, positive_ratio, ir))
        means = [item[0] for item in segment_values if np.isfinite(item[0])]
        positive_ratios = [item[1] for item in segment_values if np.isfinite(item[1])]
        prior = ranks[name].diff().abs().stack().dropna()
        row_out = {
            "factor": name,
            "horizon": horizon,
            "signal_frequency": row["signal_frequency"],
            "direction": direction,
            "family": str(row.get("family", "") or factor_family(name)),
            "coverage": float(ic.notna().mean()) if len(selection_dates) else 0.0,
            "mean_ic": float(ic.mean()) if not ic.empty else float("nan"),
            "ic_std": float(ic.std(ddof=1)) if len(ic) > 1 else float("nan"),
            "ic_pos_ratio": float(ic.gt(0.0).mean()) if not ic.empty else float("nan"),
            "segment_positive_ratio": float(np.mean(np.asarray(means) > 0.0)) if means else 0.0,
            "worst_segment_mean_ic": float(np.min(means)) if means else float("nan"),
            "median_segment_mean_ic": float(np.median(means)) if means else float("nan"),
            "rank_churn": float(prior.mean()) if not prior.empty else float("nan"),
        }
        if source_run_dir is not None:
            observed_oos_ic = row.get("oos_ic")
            row_out.update({
                "observed_oos_ic": observed_oos_ic,
                "observed_oos_same_direction": (
                    bool(float(observed_oos_ic) * direction > 0.0)
                    if observed_oos_ic is not None and pd.notna(observed_oos_ic)
                    else None
                ),
                "observed_oos_used_for_selection": False,
            })
        for idx, (mean, positive_ratio, ir) in enumerate(segment_values, 1):
            row_out[f"segment_{idx}_mean_ic"] = mean
            row_out[f"segment_{idx}_positive_ratio"] = positive_ratio
            row_out[f"segment_{idx}_ic_ratio"] = ir
        all_diagnostics.append(row_out)

    cluster_rows: list[dict] = []
    factor_sets: dict[str, dict] = {}
    correlation_files: dict[str, str] = {}
    if group_by_best_period or source_run_dir is not None:
        groups = [
            (f"h{horizon}", sorted(name for name in names if periods[name] == horizon))
            for horizon in selection_horizons
        ]
    else:
        groups = [("mixed", names)]
    for group, group_names in groups:
        corr = _exposure_correlation(ranks, group_names)
        corr_path = output / f"exposure_correlation_{group}.csv"
        corr.to_csv(corr_path, encoding="utf-8-sig")
        correlation_files[group] = corr_path.name
        clusters = _cluster(corr, group_names)
        rows_in_group = [
            row for row in all_diagnostics if str(row["factor"]) in set(group_names)
        ]
        rows_by_name = {str(row["factor"]): row for row in rows_in_group}
        for row in rows_in_group:
            row["cluster_id"] = int(clusters[str(row["factor"])])
        for name in group_names:
            cluster_rows.append({
                "group": group,
                "factor": name,
                "best_period": periods[name],
                "cluster_id": int(clusters[name]),
                "cluster_size": int(sum(value == clusters[name] for value in clusters.values())),
                "is_representative": False,
                "family": rows_by_name[name]["family"],
                "mean_ic": rows_by_name[name]["mean_ic"],
                "segment_positive_ratio": rows_by_name[name]["segment_positive_ratio"],
                "rank_churn": rows_by_name[name]["rank_churn"],
            })
        representatives: list[str] = []
        for cluster_id in sorted(set(clusters.values())):
            members = [
                rows_by_name[name] for name in group_names
                if clusters[name] == cluster_id
            ]
            chosen = max(members, key=_representative_key)
            representatives.append(str(chosen["factor"]))
            for row in cluster_rows:
                if row["group"] == group and row["cluster_id"] == cluster_id and row["factor"] == chosen["factor"]:
                    row["is_representative"] = True
        representatives = sorted(representatives, key=lambda name: _representative_key(rows_by_name[name]), reverse=True)
        factor_sets[f"balanced_core_{group}"] = {
            "group": group,
            "purpose": "all cluster representatives; portfolio-search candidate pool",
            "factors": representatives,
        }
        # Retain the old bounded view only for the explicit common-horizon
        # observation workflow. It is not a current strategy candidate.
        if source_run_dir is not None:
            compact = _compact_representatives(
                [rows_by_name[name] for name in representatives],
                int(max_compact_factors),
            )
            factor_sets[f"compact_core_{group}"] = {
                "group": group,
                "purpose": "legacy bounded common-horizon observation",
                "factors": compact,
            }

    aggregate_names = ["balanced_core"]
    if source_run_dir is not None:
        aggregate_names.append("compact_core")
    for name in aggregate_names:
        by_group = [
            factor_sets[f"{name}_{group}"]["factors"]
            for group, _ in groups
        ]
        factor_sets[name] = {
            "groups": {
                group: values
                for (group, _), values in zip(groups, by_group)
            },
            "factors": sorted(set().union(*map(set, by_group))),
            "purpose": (
                "cluster representatives; best_period is evidence only"
                if name == "balanced_core" else
                "legacy common-horizon observation only"
            ),
        }

    nested_path: list[dict] = []
    candidate_evaluations: list[dict] = []
    nested_stop_reason: str | None = None
    portfolio_search_seconds = 0.0
    extra_files: list[str] = []
    if source_run_dir is None and not group_by_best_period:
        portfolio_started = time.perf_counter()
        close = runner.data_manager.get("close", requested_dates, universe)
        daily_ret, close_tradable = runner.data_manager.prepare_close_data(close)
        full_portfolio_ic = rank_information_coefficients(
            full_ranks,
            daily_ret,
            minimum_cross_section=3,
        )
        portfolio_ic = full_portfolio_ic
        schedule = runner.data_manager.get_contract_schedule(
            requested_dates, universe
        )
        adapter = SimpleNamespace(
            cal=requested_dates,
            u=list(universe),
            ranks=full_ranks,
            factor_values=raw, factor_directions=directions, reference_universe=reference_universe,
            ic=full_portfolio_ic,
            daily_ret=daily_ret,
            close_tradable=close_tradable,
            contract_schedule=schedule,
            env=SimpleNamespace(
                sector_of={name: PORTFOLIO_CAP_GROUPS[name] for name in universe}
            ),
        )
        recipe = _production_recipe(config)
        if (
            recipe.constraints.minimum_risk_observations
            != int(config.production_portfolio.minimum_risk_observations)
            or not np.isclose(
                recipe.constraints.covariance_shrinkage,
                float(config.production_portfolio.covariance_shrinkage),
            )
        ):
            raise ValueError(
                "PortfolioRecipe risk defaults no longer match production_portfolio"
            )
        evaluator = PortfolioEvaluator(
            adapter,
            start=selection_dates[0],
            end=selection_dates[-1],
            cost_model=_configured_cost_model(config),
            ic_window=int(config.production_portfolio.ic_window),
            risk_lookback_calendar_days=int(
                config.production_portfolio.risk_lookback_calendar_days
            ),
        )
        # Clusters describe redundancy, but no member is permanently excluded.
        representatives = names
        history_segments = [
            (max(selection_dates[0], pd.Timestamp(left)), min(selection_end, pd.Timestamp(right)))
            for left, right in (("2016-03-31", "2019-12-31"), ("2020-01-01", "2021-12-31"),
                                ("2022-01-01", "2023-12-31"), ("2024-01-01", "2024-12-31"),
                                ("2025-01-01", str(cutoff.date())))
            if pd.Timestamp(right) >= selection_dates[0] and pd.Timestamp(left) <= selection_end
        ]
        if custom:
            import duckdb
            cache_namespace = json.dumps({"data": source_fingerprint, "library": sha256_file(library_path),
                "runtime": plan["runtime"],
                "factor_code": FactorPanelRunner._source_tree_fingerprint(),
                "directions": directions, "reference_universe": reference_universe,
                "code": {str(p.relative_to(project)): sha256_file(p) for p in (
                    project / "research/historical_portfolio_search.py", project / "optimization/portfolio_construction.py",
                    project / "backtest/research_ledger.py", Path(__file__))}}, sort_keys=True)
            with duckdb.connect(str(output / "search.duckdb")) as db:
                nested_path, candidate_evaluations, nested_stop_reason = _run_budgeted_pool_search(
                    evaluator=evaluator, portfolio_ic=portfolio_ic, pool=names, recipe=recipe,
                    segments=history_segments, clusters=clusters, db=db, output=output,
                    budget=portfolio_search_budget, cache_namespace=cache_namespace,
                    search_sizes=search_sizes, initial_candidates=initial_candidates or (), search_seed=search_seed)
            if search_sizes:
                valid_rows = [r for r in candidate_evaluations if r["status"] == "evaluated"]
                nested_path = [max((r for r in valid_rows if r["factor_count"] == size), key=robustness_key)
                               for size in sorted({r["factor_count"] for r in valid_rows})]
        else:
            nested_path, candidate_evaluations, nested_stop_reason = _run_portfolio_search(
                evaluator=evaluator,
                portfolio_ic=portfolio_ic,
                representatives=representatives,
                recipe=recipe,
                segments=history_segments,
                output=output,
            )
        shortlist = _portfolio_shortlist(candidate_evaluations)
        if custom and selection_end < cutoff:
            # Lock membership and ordering before reading validation performance.
            _write_json(output / "locked_candidates.json", shortlist)
            observation_evaluator = evaluator.bounded(selection_dates[0], cutoff)
            observation_rows, observation_returns = [], {}
            for i, row in enumerate(shortlist, 1):
                result = {"candidate": f"candidate_{i}_{row['factor_count']}f",
                          "factors": row["factors"], "status": "evaluated", "error": ""}
                try:
                    ledger = observation_evaluator.ledger(row["factors"], recipe)
                    values = ledger.loc[ledger.index > selection_end, "net_return"]
                    result.update(performance_metrics(values, periods_per_year=recipe.periods_per_year))
                    observation_returns[result["candidate"]] = ledger["net_return"]
                except (RuntimeError, ValueError) as exc:
                    result.update(status="rejected_runtime", error=str(exc))
                observation_rows.append(result)
            _write_json(output / "conditional_validation.json", {
                "after": str(selection_end.date()), "end": str(cutoff.date()),
                "used_for_selection": False, "independent_oos": False, "candidates": observation_rows})
            pd.DataFrame(observation_returns).to_csv(output / "locked_candidate_full_returns.csv")
            extra_files.extend(["locked_candidates.json", "conditional_validation.json", "locked_candidate_full_returns.csv"])
        factor_sets["portfolio_candidates"] = {
            "purpose": "diverse Pareto candidates; historical development evidence only",
            "candidates": shortlist,
            "qualified_count": len(shortlist),
        }
        candidate_returns = pd.DataFrame({
            f"candidate_{i}_{row['factor_count']}f": evaluator.ledger(row["factors"], recipe)["net_return"]
            for i, row in enumerate(shortlist, 1)
        })
        if not candidate_returns.empty:
            candidate_returns.to_csv(output / "candidate_returns.csv")
            candidate_returns.corr().to_csv(output / "candidate_return_correlation.csv")
            extra_files.extend(["candidate_returns.csv", "candidate_return_correlation.csv"])
            block_metrics = []
            for i, row in enumerate(shortlist, 1):
                ledger = evaluator.ledger(row["factors"], recipe)
                for left, right in history_segments:
                    values = ledger.loc[left:right, "net_return"]
                    block_metrics.append({
                        "candidate": f"candidate_{i}_{row['factor_count']}f",
                        "start": str(left.date()), "end": str(right.date()),
                        **performance_metrics(values, initial_anchor=values.index[0] == ledger.index[0],
                                              periods_per_year=recipe.periods_per_year),
                    })
            _write_csv(output / "candidate_block_metrics.csv", block_metrics)
            extra_files.append("candidate_block_metrics.csv")
            from run_portfolio_workflow import _write_comparison_plot
            labels = {name: f"候选{i}（{row['factor_count']}因子）"
                      for i, (name, row) in enumerate(zip(candidate_returns, shortlist), 1)}
            plot_rows = []
            for name, row in zip(candidate_returns, shortlist):
                metrics = performance_metrics(candidate_returns[name], initial_anchor=True, periods_per_year=recipe.periods_per_year)
                plot_rows.append({"strategy": labels[name], **metrics,
                                  "volatility": metrics["annual_volatility"],
                                  "annualized_turnover": row["annual_turnover"]})
            _write_comparison_plot(
                output, (1 + candidate_returns).cumprod().rename(columns=labels), plot_rows,
                title=f"候选组合净值对比（{selection_dates[0]:%Y-%m-%d}至{selection_end:%Y-%m-%d}）\n历史开发样本；统一组合配方与成本，非独立样本外",
            )
            extra_files.append("nav_comparison.png")
        recommended = shortlist[0] if shortlist else None
        factor_sets["recommended_core"] = {
            "purpose": (
                "first robust candidate for display only; no automatic promotion; "
                "membership still requires independent post-selection validation"
            ),
            "factor_count": int(recommended["factor_count"]) if recommended else 0,
            "factors": list(recommended["factors"]) if recommended else [],
            "selection_metrics": ({
                key: recommended[key]
                for key in (
                    "positive_segment_ratio", "worst_sharpe", "median_sharpe",
                    "median_annual_return", "worst_drawdown", "annual_turnover",
                    "full_annual_return", "full_sharpe", "full_max_drawdown",
                    "full_total_return",
                )
            } if recommended else {}),
        }
        factor_sets["size_candidates"] = {
            "purpose": "best exact candidate at every explored size; paths need not be nested",
            "stop_reason": nested_stop_reason,
            "candidates": [
                {
                    "factor_count": int(row["factor_count"]),
                    "factors": list(row["factors"]),
                    **{
                        key: row[key]
                        for key in (
                            "positive_segment_ratio", "worst_sharpe",
                            "median_sharpe", "median_annual_return",
                            "worst_drawdown", "annual_turnover", "full_sharpe",
                        )
                    },
                }
                for row in nested_path
            ],
        }
        portfolio_search_seconds = time.perf_counter() - portfolio_started
        for filename, rows_to_write in (
            ("size_candidate_path.csv", nested_path),
            ("portfolio_candidate_evaluations.csv", candidate_evaluations),
        ):
            serializable = [
                {**row, "factors": "|".join(row["factors"])}
                for row in rows_to_write
            ]
            if serializable:
                _write_csv(output / filename, serializable)
                extra_files.append(filename)

    _write_csv(output / "factor_diagnostics.csv", all_diagnostics)
    _write_csv(output / "factor_clusters.csv", cluster_rows)
    _write_json(output / "factor_sets.json", factor_sets)
    summary = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "workflow": "effective_factor_subset_selection",
        "library_path": str(library_path),
        "source_run_dir": str(Path(source_run_dir).resolve()) if source_run_dir else None,
        "selection_mode": (
            "common_horizon" if source_run_dir else
            "best_period_sleeves" if group_by_best_period else
            "mixed_daily_multipath_portfolio"
        ),
        "common_horizon": int(common_horizon) if common_horizon is not None else None,
        "library_count": len(names),
        "candidate_scope": names,
        "factor_reference_universe": reference_universe,
        "investment_universe": list(universe),
        "resolved_costs": _configured_cost_model(config).ledger_parameters() if custom else None,
        "resolved_recipe": recipe.to_dict() if custom else None,
        "data_source": config.data.source,
        "data_sha256": source_fingerprint,
        "research_cutoff": cutoff.date().isoformat(),
        "warmup": [
            factor_start.date().isoformat(),
            (selection_start - pd.Timedelta(days=1)).date().isoformat(),
        ],
        "selection_sample": [
            selection_start.date().isoformat(),
            selection_end.date().isoformat(),
            len(selection_dates),
        ],
        "post_cutoff_data_used": False,
        "historical_evidence_role": "development_stability_conditional_on_current_admitted_pool",
        "admission_window": [str(factor_admission_start(config).date()), str(cutoff.date())],
        "library_sha256": sha256_file(library_path),
        "input_bar_frequencies": {
            frequency: sum(
                str(row["input_bar_frequency"]) == frequency
                for row in library_rows
            )
            for frequency in sorted({
                str(row["input_bar_frequency"]) for row in library_rows
            })
        },
        "signal_frequency": "daily",
        "horizon_unit": "daily bars / trading days",
        "horizon_counts": {
            str(horizon): sum(periods[name] == horizon for name in names)
            for horizon in selection_horizons
        },
        "best_period_role": "admission_evidence_only",
        "cluster_correlation": {"metric": "direction-adjusted daily cross-sectional rank exposure", "method": "complete_linkage", "threshold_abs_corr": CLUSTER_CORRELATION_THRESHOLD},
        "oos_used_for_selection": False,
        "portfolio_search": (
            {
                "enabled": True,
                "recipe": recipe.to_dict(),
                "periods_per_year": recipe.periods_per_year,
                "cap_groups": {name: PORTFOLIO_CAP_GROUPS[name] for name in universe},
                "ic_horizon": 1,
                "ic_window": int(config.production_portfolio.ic_window),
                "risk_lookback_calendar_days": int(
                    config.production_portfolio.risk_lookback_calendar_days
                ),
                "minimum_risk_observations": int(
                    config.production_portfolio.minimum_risk_observations
                ),
                "covariance_shrinkage": float(
                    config.production_portfolio.covariance_shrinkage
                ),
                "segments": [[str(a.date()), str(b.date())] for a, b in history_segments],
                "initial_factor_count": 1 if custom and recipe.combination_order == "portfolio_first" else 2,
                "final_factor_count_is_fixed": False,
                "maximum_explored_factor_count": max((r["factor_count"] for r in candidate_evaluations), default=0),
                "exact_budget": portfolio_search_budget if custom else None,
                "exact_shortlist_width": 4 if search_sizes else (16 if custom else PORTFOLIO_SEARCH_EXACT_WIDTH),
                "beam_width": 2 if search_sizes else (6 if custom else PORTFOLIO_SEARCH_BEAM_WIDTH),
                "size_search": plan["size_search"] if custom else None,
                "size_penalty": False,
                "early_stop_on_nonimprovement": False,
                "proxy_role": ("mean single-factor net returns; exact net portfolios determine shortlist"
                               if custom and recipe.combination_order == "portfolio_first" else
                               "H1 IC proposal only; exact net portfolios determine shortlist"),
                "stop_reason": nested_stop_reason,
                "recommended_factor_count": int(
                    factor_sets["recommended_core"]["factor_count"]
                ),
            }
            if candidate_evaluations else {"enabled": False}
        ),
        "performance": {
            "factor_compute_seconds": factor_compute_seconds,
            "portfolio_search_seconds": portfolio_search_seconds,
            "factor_seconds": sum(float(row["seconds"]) for row in factor_timings),
            "shared_overhead_seconds": max(
                0.0,
                factor_compute_seconds
                - sum(float(row["seconds"]) for row in factor_timings),
            ),
            "factor_timings": sorted(
                factor_timings,
                key=lambda row: (-float(row["seconds"]), str(row["factor"])),
            ),
        },
        "factor_sets": {
            key: value for key, value in factor_sets.items()
            if key in {"balanced_core", "compact_core", "recommended_core", "portfolio_candidates"}
        },
        "correlation_files": correlation_files,
    }
    _write_json(output / "selection_summary.json", summary)
    files = [
        "factor_diagnostics.csv", "factor_clusters.csv", "factor_sets.json",
        "selection_summary.json", *correlation_files.values(), *extra_files,
    ]
    if custom:
        files.extend(name for name in ("search_plan.json", "factor_eligibility.json", "pool_selection.json", "size_search.json")
                     if (output / name).exists())
    contract = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "run_id": str(run_id),
        "workflow": "effective-factor-subset-selection",
        "selection_contract": summary,
        "files": {name: {"sha256": sha256_file(output / name)} for name in files},
    }
    _write_json(output / "run_contract.json", contract)
    runner.factor_engine.clear_cache()
    return output
