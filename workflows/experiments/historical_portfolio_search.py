"""Isolated legacy expanding-window portfolio-method experiment.

This module is not the framework's formal factor-admission workflow. Explicit
portfolio experiments reuse the validated library without changing admission.
Legacy expanding-window modes freeze choices within each fold; the full-pool
selection modes instead use declared historical development blocks and a fixed
cutoff. These blocks are not independent walk-forward admission evidence, nor
a reconstruction of when each factor idea was originally invented.

Run from the project root::

    E:\\Python\\Pythonvenv\\Scripts\\python.exe -X utf8 -B \
        -m workflows.experiments.historical_portfolio_search
"""
from __future__ import annotations

import argparse
import gc
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
from typing import Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.metrics import TRADING_DAYS_PER_YEAR  # noqa: E402
from core.config import load_config  # noqa: E402
from core.date_policy import factor_validation_window, research_cutoff  # noqa: E402
from pipeline.runner import PipelineRunner  # noqa: E402
from research.historical_portfolio_search import (  # noqa: E402
    CausalEligibilityEnvironment,
    PortfolioEvaluator,
    PortfolioRecipe,
    aggregate_robust_summaries,
    beam_factor_sets,
    calendar_segments,
    cluster_factors,
    factor_set_jaccard,
    factor_neighborhood,
    portfolio_recipe_grid,
    unique_factor_neighborhoods,
    performance_metrics,
    robust_summary,
    robustness_key,
    training_factor_diagnostics,
)
from research.portfolio_experiment_support import (  # noqa: E402
    FACTORS_6F as F6,
    FACTORS_8F,
    FACTORS_10F as F10,
    FACTORS_13F as F13,
    FACTORS_14F as F14,
    FactorPanelRunner as Runner,
    configured_futures_cost_model,
)
from workflows.factor_selection import _load_library  # noqa: E402
from research.validation import (  # noqa: E402
    HISTORICAL_START,
    OOS_END,
    SIMULATED_LIVE_START,
    expanding_window_folds,
    historical_experiment_period_snapshot,
)


COST_MODEL = configured_futures_cost_model()

OUTER_FOLDS = expanding_window_folds()

BASE_RECIPE = PortfolioRecipe("lw_abs", 10, 3, "erc")
RECIPE_8F_ALT = PortfolioRecipe(
    "equal",
    12,
    0,
    "inverse_volatility",
    asset_min_fraction=0.0,
    asset_max_fraction=1.0,
)
RECIPE_CHALLENGERS_8F = {
    "production_method": BASE_RECIPE,
    "equal_top12_no_cap_inverse_volatility": RECIPE_8F_ALT,
}
SEED_FACTOR_SETS = {"6f": F6, "14f": F14, "8f": FACTORS_8F}
KNOWN_FACTOR_SETS = {
    "6f": F6,
    "10f": F10,
    "13f": F13,
    "14f": F14,
    "8f": FACTORS_8F,
}


def _split_trading_dates(
    dates: pd.DatetimeIndex, count: int
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    chunks = np.array_split(np.asarray(dates), int(count))
    return [
        (pd.Timestamp(chunk[0]), pd.Timestamp(chunk[-1]))
        for chunk in chunks if len(chunk)
    ]


def _rank_fixed_recipe_candidates(
    evaluator: PortfolioEvaluator,
    candidates: Sequence[dict],
    segments: Sequence[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[dict]:
    rows = []
    for candidate in _deduplicate_factor_candidates(candidates):
        factors = list(candidate["factors"])
        ledger = evaluator.ledger(factors, BASE_RECIPE)
        summary = robust_summary(ledger, segments, initial_anchor=True)
        rows.append({
            "candidate": candidate["candidate"],
            "factor_count": len(factors),
            "factors": factors,
            **summary,
            "annual_turnover": float(
                ledger["executed_traded_notional"].iloc[1:].mean()
                * TRADING_DAYS_PER_YEAR
            ),
        })
    return sorted(
        rows,
        key=lambda row: (
            *robustness_key(row),
            -float(row["annual_turnover"]),
            -int(row["factor_count"]),
        ),
        reverse=True,
    )


def run_combination_aware_factor_search(
    *,
    run_id: str,
    config_path: str,
    source_validation_dir: str,
    source_selection_dir: str | None = None,
    common_horizon: int = 5,
) -> Path:
    """Select common-horizon factors by exact fixed-recipe portfolio evidence.

    Candidate generation and ranking use only the locked IS window.  The locked
    OOS is evaluated once after the IS winner is frozen and never changes the
    chosen membership.  A factor checkpoint survives interruption and is
    removed after a successful run.
    """
    output = ROOT / "runs" / "portfolio_factor_search" / str(run_id)
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(config_path)
    policy_runner = PipelineRunner(config=config)
    window = factor_validation_window(
        config, policy_runner.data_manager, frequency="daily_intraday"
    )
    cutoff = research_cutoff(config)
    if window.oos_end != cutoff:
        raise ValueError("portfolio factor search must end at the research cutoff")
    source_path, library_rows = _load_library(
        config,
        source_run_dir=source_validation_dir,
        allowed_horizons=(int(common_horizon),),
    )
    factors = sorted(str(row["factor"]) for row in library_rows)
    directions = {str(row["factor"]): int(row["direction"]) for row in library_rows}
    checkpoint = output / "_factor_checkpoints"
    started = time.perf_counter()
    runner = Runner(
        factors,
        start=window.factor_start,
        end=cutoff,
        factor_directions=directions,
        ic_horizon=int(common_horizon),
        checkpoint_dir=checkpoint,
    )
    factor_seconds = time.perf_counter() - started
    runner.get_contract_schedule()
    is_dates = runner.cal[(runner.cal >= window.is_start) & (runner.cal <= window.is_end)]
    oos_dates = runner.cal[(runner.cal >= window.oos_start) & (runner.cal <= window.oos_end)]
    if len(is_dates) != window.is_bars or len(oos_dates) != window.oos_bars:
        raise ValueError("resolved factor-search calendar differs from the locked window")

    candidates_by_prefix: dict[str, list[dict]] = {}
    prefix_specs = (("is84", is_dates[:84], 2), ("is126", is_dates, 3))
    for label, dates, segment_count in prefix_specs:
        diagnostics = training_factor_diagnostics(
            runner.ic[factors], dates[0], dates[-1]
        )
        eligible = diagnostics.loc[diagnostics["eligible"], "factor"].astype(str).tolist()
        clusters = cluster_factors(
            runner.ic.loc[dates[0]:dates[-1]], eligible,
            correlation_threshold=0.85,
        )
        beams = beam_factor_sets(
            runner.ic,
            diagnostics,
            clusters,
            start=dates[0],
            end=dates[-1],
            minimum_size=4,
            maximum_size=14,
            beam_width=32,
            output_limit=24,
            segments=_split_trading_dates(dates, segment_count),
        )
        candidates_by_prefix[label] = [
            {"candidate": f"{label}_beam_{row['rank']}", "factors": row["factors"]}
            for row in beams
        ]

    if source_selection_dir:
        sets_path = Path(source_selection_dir)
        if not sets_path.is_absolute():
            sets_path = ROOT / sets_path
        sets = json.loads((sets_path / "factor_sets.json").read_text(encoding="utf-8"))
        for name in ("balanced_core", "compact_core"):
            values = list(sets.get(name, {}).get("factors", []))
            if values and set(values).issubset(factors):
                candidates_by_prefix["is126"].append({
                    "candidate": f"existing_{name}", "factors": values
                })

    prefix_evaluator = PortfolioEvaluator(
        runner, start=is_dates[0], end=is_dates[83], cost_model=COST_MODEL
    )
    prefix_ranked = _rank_fixed_recipe_candidates(
        prefix_evaluator,
        candidates_by_prefix["is84"],
        _split_trading_dates(is_dates[:84], 2),
    )
    prefix_evaluator.clear_transient_caches()
    candidates = _deduplicate_factor_candidates(
        candidates_by_prefix["is84"] + candidates_by_prefix["is126"]
    )
    is_evaluator = PortfolioEvaluator(
        runner, start=is_dates[0], end=is_dates[-1], cost_model=COST_MODEL
    )
    exact_started = time.perf_counter()
    initial_is_ranked = _rank_fixed_recipe_candidates(
        is_evaluator, candidates, _split_trading_dates(is_dates, 3)
    )
    full_diagnostics = training_factor_diagnostics(
        runner.ic[factors], is_dates[0], is_dates[-1]
    )
    replacement_pool = full_diagnostics.loc[
        full_diagnostics["eligible"], "factor"
    ].astype(str).head(8).tolist()
    refinements = []
    for seed_index, seed in enumerate(initial_is_ranked[:2], 1):
        seed_factors = list(seed["factors"])
        if len(seed_factors) > 4:
            refinements.extend({
                "candidate": f"refine_{seed_index}_drop_{removed}",
                "factors": [name for name in seed_factors if name != removed],
            } for removed in seed_factors)
        refinements.extend({
            "candidate": f"refine_{seed_index}_add_{added}",
            "factors": [*seed_factors, added],
        } for added in replacement_pool if added not in seed_factors)
        for removed in seed_factors:
            refinements.extend({
                "candidate": f"refine_{seed_index}_swap_{removed}_for_{added}",
                "factors": [
                    *(name for name in seed_factors if name != removed), added
                ],
            } for added in replacement_pool if added not in seed_factors)
    candidates = _deduplicate_factor_candidates([*candidates, *refinements])
    is_ranked = _rank_fixed_recipe_candidates(
        is_evaluator, candidates, _split_trading_dates(is_dates, 3)
    )
    exact_seconds = time.perf_counter() - exact_started
    winner = is_ranked[0]

    leave_one_out = []
    winner_ledger = is_evaluator.ledger(winner["factors"], BASE_RECIPE)
    winner_metrics = performance_metrics(
        winner_ledger["net_return"], initial_anchor=True
    )
    for removed in winner["factors"]:
        factors_without = [name for name in winner["factors"] if name != removed]
        ledger = is_evaluator.ledger(factors_without, BASE_RECIPE)
        metrics = performance_metrics(ledger["net_return"], initial_anchor=True)
        leave_one_out.append({
            "removed_factor": removed,
            "sharpe": metrics["sharpe"],
            "sharpe_delta_vs_winner": metrics["sharpe"] - winner_metrics["sharpe"],
            "annual_return": metrics["annual_return"],
            "max_drawdown": metrics["max_drawdown"],
        })
    is_evaluator.clear_transient_caches()

    oos_evaluator = PortfolioEvaluator(
        runner, start=oos_dates[0], end=oos_dates[-1], cost_model=COST_MODEL
    )
    finalist_names = {row["candidate"] for row in is_ranked[:5]}
    oos_rows = []
    navs = {}
    for row in is_ranked:
        if row["candidate"] not in finalist_names:
            continue
        ledger = oos_evaluator.ledger(row["factors"], BASE_RECIPE)
        metrics = performance_metrics(ledger["net_return"], initial_anchor=True)
        oos_rows.append({
            "candidate": row["candidate"],
            "is_rank": is_ranked.index(row) + 1,
            "factor_count": row["factor_count"],
            "factors": row["factors"],
            **metrics,
        })
        navs[row["candidate"]] = ledger["nav"]
    winner_weights = oos_evaluator.weights(winner["factors"], BASE_RECIPE)
    stressed = oos_evaluator.ledger_from_weights(winner_weights, cost_multiplier=2.0)
    stress_metrics = performance_metrics(stressed["net_return"], initial_anchor=True)
    oos_evaluator.clear_transient_caches()

    pd.DataFrame(is_ranked).assign(
        factors=lambda frame: frame["factors"].map("|".join)
    ).to_csv(output / "is_candidate_ranking.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(oos_rows).assign(
        factors=lambda frame: frame["factors"].map("|".join)
    ).to_csv(output / "locked_oos_diagnostics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(leave_one_out).to_csv(
        output / "leave_one_out_is.csv", index=False, encoding="utf-8-sig"
    )
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(13, 7))
    for name, nav in navs.items():
        normalized = nav / nav.iloc[0] * 1000.0
        metric = next(row for row in oos_rows if row["candidate"] == name)
        ax.plot(
            normalized.index,
            normalized.values,
            linewidth=2.0 if name == winner["candidate"] else 1.2,
            label=(f"{name} | 年化{metric['annual_return']:.1%} "
                   f"夏普{metric['sharpe']:.2f} 回撤{metric['max_drawdown']:.1%}"),
        )
    ax.set_title("共同H5因子组合搜索：冻结样本外净值（总敞口2，扣费后）")
    ax.set_ylabel("净值（起点=1000）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "locked_oos_nav.png", dpi=160)
    plt.close(fig)

    profile = runner.performance_profile()
    result = {
        "workflow": "combination_aware_factor_search",
        "source_validation": str(source_path),
        "candidate_factor_count": len(factors),
        "common_horizon": int(common_horizon),
        "recipe": BASE_RECIPE.to_dict(),
        "research_cutoff": cutoff.date().isoformat(),
        "warmup": [window.factor_start.date().isoformat(), (window.is_start - pd.Timedelta(days=1)).date().isoformat()],
        "is": [window.is_start.date().isoformat(), window.is_end.date().isoformat(), window.is_bars],
        "oos": [window.oos_start.date().isoformat(), window.oos_end.date().isoformat(), window.oos_bars],
        "oos_used_for_selection": False,
        "search": {
            "prefixes": ["first_84_is_bars", "all_126_is_bars"],
            "candidate_generation": "IC beam, max one member per abs-IC-correlation cluster at 0.85",
            "candidate_factor_count_range": [4, 14],
            "exact_ranking": "fixed production recipe; 3 IS segments; robustness then turnover",
            "refinement": "top-2 exact candidates; one bounded add/drop/swap pass using top-8 IS diagnostics",
            "initial_candidate_count": len(initial_is_ranked),
            "refinement_candidate_count": len(candidates) - len(initial_is_ranked),
            "candidate_count": len(candidates),
        },
        "winner": {
            **winner,
            "full_is_metrics": winner_metrics,
            "locked_oos_metrics": next(
                row for row in oos_rows if row["candidate"] == winner["candidate"]
            ),
            "locked_oos_cost_2x_metrics": stress_metrics,
        },
        "membership_stability": {
            "is84_winner": prefix_ranked[0]["factors"],
            "is126_winner": winner["factors"],
            "jaccard": factor_set_jaccard(
                prefix_ranked[0]["factors"], winner["factors"]
            ),
        },
        "performance": {
            "factor_panel_seconds": factor_seconds,
            "exact_candidate_search_seconds": exact_seconds,
            "factor_panel_profile": profile,
        },
    }
    _json_dump(output / "search_summary.json", result)
    shutil.rmtree(checkpoint)
    print(json.dumps({
        "output": str(output),
        "winner": winner["candidate"],
        "factors": winner["factors"],
        "is_metrics": winner_metrics,
        "oos_metrics": result["winner"]["locked_oos_metrics"],
    }, ensure_ascii=False, indent=2), flush=True)
    return output


def _json_dump(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _write_factor_profile(path: Path, runner, **extra) -> None:
    """A cache-only resume must not erase previously measured preparation work."""
    if path.exists() and not runner.computed_factor_count:
        return
    _json_dump(path, {"computed": runner.computed_factor_count,
                     "checkpoint_loaded": runner.checkpoint_loaded_factor_count,
                     **runner.performance_profile(), **extra})


def _load_estimable_factor_manifest(path: Path) -> tuple[list[str], str]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    rows = payload.get("all_results")
    if not isinstance(rows, list):
        raise ValueError("factor manifest must contain an all_results list")
    factors = [
        str(row["name"])
        for row in rows
        if isinstance(row, dict)
        and row.get("name")
        and any(
            bool(result.get("estimable"))
            for result in dict(row.get("all_periods") or {}).values()
            if isinstance(result, dict)
        )
    ]
    factors = list(dict.fromkeys(factors))
    if not factors:
        raise ValueError("factor manifest contains no estimable factors")
    return factors, hashlib.sha256(raw).hexdigest()


def _deduplicate_recipes(recipes: Sequence[PortfolioRecipe]) -> list[PortfolioRecipe]:
    return list(dict.fromkeys(recipes))


def _rank_recipes_for_fold(
    evaluator: PortfolioEvaluator,
    recipes: Sequence[PortfolioRecipe],
    factor_sets: Mapping[str, Sequence[str]],
    fold: Mapping[str, str],
    *,
    stage: str,
) -> list[dict]:
    train_start = pd.Timestamp(fold["train_start"])
    train_end = pd.Timestamp(fold["train_end"])
    segments = calendar_segments(train_start, train_end, years=2)
    rows = []
    for recipe in recipes:
        seed_summaries = []
        for seed_name, factors in factor_sets.items():
            ledger = evaluator.ledger(factors, recipe)
            summary = robust_summary(ledger, segments, initial_anchor=True)
            summary["seed"] = seed_name
            seed_summaries.append(summary)
        aggregate = aggregate_robust_summaries(seed_summaries)
        rows.append({
            "stage": stage,
            "fold": fold["fold"],
            **recipe.to_dict(),
            **aggregate,
        })
    ranked = sorted(rows, key=robustness_key, reverse=True)
    print(
        f"  [{stage} {fold['fold']}] selected {ranked[0]['name']} "
        f"from {len(ranked)} candidates",
        flush=True,
    )
    return ranked


def _stage_method_search(
    evaluator: PortfolioEvaluator,
    output: Path,
) -> tuple[dict[str, list[PortfolioRecipe]], pd.DataFrame]:
    """Coordinate-search methods while keeping the finite grid auditable."""

    all_rows: list[dict] = []

    composition_recipes = [
        PortfolioRecipe(method, 10, 3, "erc")
        for method in ("equal", "diag_icir", "lw_abs", "lw_positive")
    ]
    composition_winners: dict[str, list[PortfolioRecipe]] = {}
    selection_winners: dict[str, list[PortfolioRecipe]] = {}
    final_by_fold: dict[str, list[PortfolioRecipe]] = {}
    for fold in OUTER_FOLDS:
        fold_evaluator = evaluator.bounded(
            fold["train_start"], fold["train_end"]
        )
        ranked = _rank_recipes_for_fold(
            fold_evaluator, composition_recipes, SEED_FACTOR_SETS, fold,
            stage="factor_weight",
        )
        all_rows.extend(ranked)
        composition_winners[fold["fold"]] = [
            _recipe_from_row(row) for row in ranked[:2]
        ]
        fold_evaluator.clear_transient_caches()

        selection_recipes = _deduplicate_recipes([
            PortfolioRecipe(recipe.factor_weight, top_n, sector_cap, "erc")
            for recipe in composition_winners[fold["fold"]]
            for top_n in (8, 10, 12)
            for sector_cap in (0, 3)
        ])
        ranked = _rank_recipes_for_fold(
            fold_evaluator, selection_recipes, SEED_FACTOR_SETS, fold,
            stage="selection",
        )
        all_rows.extend(ranked)
        selection_winners[fold["fold"]] = [
            _recipe_from_row(row) for row in ranked[:2]
        ]
        fold_evaluator.clear_transient_caches()

        allocation_recipes = _deduplicate_recipes([
            PortfolioRecipe(
                base.factor_weight,
                base.top_n,
                base.sector_cap,
                asset_weight,
            )
            for base in selection_winners[fold["fold"]]
            for asset_weight in ("equal", "inverse_volatility", "erc")
        ])
        ranked = _rank_recipes_for_fold(
            fold_evaluator, allocation_recipes, SEED_FACTOR_SETS, fold,
            stage="asset_weight",
        )
        all_rows.extend(ranked)
        final_by_fold[fold["fold"]] = [
            _recipe_from_row(row) for row in ranked[:3]
        ]
        fold_evaluator.clear_transient_caches()

    frame = pd.DataFrame(all_rows)
    frame.to_csv(output / "method_search_results.csv", index=False, encoding="utf-8")
    _json_dump(
        output / "method_shortlist.json",
        {
            "composition_winners": {
                name: [recipe.to_dict() for recipe in recipes]
                for name, recipes in composition_winners.items()
            },
            "selection_winners": {
                name: [recipe.to_dict() for recipe in recipes]
                for name, recipes in selection_winners.items()
            },
            "final_by_fold": {
                name: [recipe.to_dict() for recipe in recipes]
                for name, recipes in final_by_fold.items()
            },
        },
    )
    return final_by_fold, frame


def _recipe_from_row(row: Mapping) -> PortfolioRecipe:
    return PortfolioRecipe(
        factor_weight=str(row["factor_weight"]),
        top_n=int(row["top_n"]),
        sector_cap=int(row["sector_cap"]),
        asset_weight=str(row["asset_weight"]),
    )


def _deduplicate_factor_candidates(candidates: Sequence[dict]) -> list[dict]:
    output = []
    seen = set()
    for candidate in candidates:
        factors = tuple(candidate["factors"])
        key = tuple(sorted(factors))
        if not factors or key in seen:
            continue
        seen.add(key)
        output.append(dict(candidate))
    return output


def _search_factor_recipe_pairs(
    evaluator: PortfolioEvaluator,
    recipes: Sequence[PortfolioRecipe],
    candidates: Sequence[dict],
    fold: Mapping[str, str],
) -> tuple[dict, list[dict]]:
    segments = calendar_segments(
        pd.Timestamp(fold["train_start"]), pd.Timestamp(fold["train_end"]), years=2
    )
    rows = []
    for candidate_index, candidate in enumerate(candidates, 1):
        factors = candidate["factors"]
        for recipe in recipes:
            ledger = evaluator.ledger(factors, recipe)
            summary = robust_summary(ledger, segments, initial_anchor=True)
            train = ledger.loc[fold["train_start"]:fold["train_end"]]
            annual_turnover = (
                float(
                    train["executed_traded_notional"].iloc[1:].mean()
                    * TRADING_DAYS_PER_YEAR
                )
                if len(train) > 1
                else np.nan
            )
            rows.append({
                "fold": fold["fold"],
                "candidate": candidate.get("candidate", f"beam_{candidate_index}"),
                "factor_count": len(factors),
                "factors": list(factors),
                **recipe.to_dict(),
                **summary,
                "annual_turnover": annual_turnover,
            })
    ranked = sorted(
        rows,
        key=lambda row: (
            *robustness_key(row),
            -float(row.get("annual_turnover", np.inf)),
            -int(row["factor_count"]),
        ),
        reverse=True,
    )
    return ranked[0], ranked


def _factor_search(
    runner: Runner,
    evaluator: PortfolioEvaluator,
    recipes_by_fold: Mapping[str, Sequence[PortfolioRecipe]],
    output: Path,
    all_factors: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], pd.DataFrame]:
    diagnostic_rows = []
    cluster_rows = []
    candidate_rows = []
    decisions = []
    selected_weight_parts = []
    all_factors = list(dict.fromkeys(all_factors))

    for fold in OUTER_FOLDS:
        train_start = pd.Timestamp(fold["train_start"])
        train_end = pd.Timestamp(fold["train_end"])
        diagnostics = training_factor_diagnostics(
            runner.ic[all_factors], train_start, train_end
        )
        diagnostics.insert(0, "fold", fold["fold"])
        diagnostic_rows.extend(diagnostics.to_dict("records"))
        eligible = diagnostics.loc[diagnostics["eligible"], "factor"].astype(str).tolist()
        clusters = cluster_factors(
            runner.ic.loc[train_start:train_end], eligible,
            correlation_threshold=0.65,
        )
        for factor, cluster in clusters.items():
            cluster_rows.append({"fold": fold["fold"], "factor": factor, "cluster": cluster})
        beam = beam_factor_sets(
            runner.ic,
            diagnostics,
            clusters,
            start=train_start,
            end=train_end,
            minimum_size=4,
            maximum_size=12,
            beam_width=20,
            output_limit=12,
        )
        candidates = [
            {"candidate": name, "factors": list(factors)}
            for name, factors in KNOWN_FACTOR_SETS.items()
        ]
        candidates.extend({
            "candidate": f"beam_{row['rank']}",
            "factors": row["factors"],
            "ic_positive_segment_ratio": row["positive_segment_ratio"],
            "ic_worst_sharpe": row["worst_sharpe"],
            "ic_median_sharpe": row["median_sharpe"],
        } for row in beam)
        candidates = _deduplicate_factor_candidates(candidates)
        training_evaluator = evaluator.bounded(
            fold["train_start"], fold["train_end"]
        )
        winner, ranked = _search_factor_recipe_pairs(
            training_evaluator,
            recipes_by_fold[fold["fold"]],
            candidates,
            fold,
        )
        candidate_rows.extend(ranked)
        training_evaluator.clear_transient_caches()
        test_evaluator = evaluator.bounded(fold["test_start"], fold["test_end"])
        test_weights = test_evaluator.weights(
            winner["factors"], _recipe_from_row(winner)
        )
        selected_weight_parts.append(
            test_weights.loc[fold["test_start"]:fold["test_end"]]
        )
        test_ledger = test_evaluator.ledger_from_weights(test_weights).loc[
            fold["test_start"]:fold["test_end"]
        ]
        test_metrics = performance_metrics(
            test_ledger["net_return"], initial_anchor=True
        )
        decisions.append({
            **dict(fold),
            "selected_candidate": winner["candidate"],
            "selected_factors": winner["factors"],
            "selected_recipe": _recipe_from_row(winner).to_dict(),
            "training_robustness": {
                key: winner[key]
                for key in (
                    "positive_segment_ratio", "worst_sharpe", "median_sharpe",
                    "median_annual_return", "worst_drawdown", "annual_turnover",
                )
            },
            "test_metrics_diagnostic_only": test_metrics,
        })
        print(
            f"  selected {fold['fold']}: {winner['candidate']} "
            f"{winner['factor_count']}f / {winner['name']}",
            flush=True,
        )
        test_evaluator.clear_transient_caches()

    weights = pd.concat(selected_weight_parts).sort_index()
    weights = weights[~weights.index.duplicated(keep="first")]
    weights = weights.reindex(columns=runner.u).fillna(0.0)
    full_weights = pd.DataFrame(0.0, index=evaluator.dates, columns=runner.u)
    full_weights.loc[weights.index] = weights
    ledger = evaluator.ledger_from_weights(full_weights)
    ledger = ledger.loc[weights.index.min():weights.index.max()]
    diagnostics_frame = pd.DataFrame(diagnostic_rows)
    clusters_frame = pd.DataFrame(cluster_rows)
    candidates_frame = pd.DataFrame(candidate_rows)
    diagnostics_frame.to_csv(output / "factor_training_diagnostics.csv", index=False)
    clusters_frame.to_csv(output / "factor_clusters.csv", index=False)
    candidates_frame.assign(
        factors=candidates_frame["factors"].map(lambda value: "|".join(value))
    ).to_csv(output / "factor_candidate_results.csv", index=False)
    weights.to_csv(output / "adaptive_oos_weights.csv")
    ledger.to_csv(output / "adaptive_oos_ledger.csv")
    _json_dump(output / "fold_decisions.json", decisions)
    return weights, ledger, decisions, candidates_frame


def _evaluate_simulated_live(
    runner: Runner,
    final_decision: Mapping,
    output: Path,
) -> dict | None:
    start = pd.Timestamp(SIMULATED_LIVE_START)
    latest = min(pd.Timestamp(runner.cal.max()), pd.Timestamp(runner.daily_ret.index.max()))
    if latest < start:
        return None
    evaluator = PortfolioEvaluator(
        runner,
        start=HISTORICAL_START,
        end=latest,
        cost_model=COST_MODEL,
    )
    recipe = _recipe_from_row(final_decision["selected_recipe"])
    weights = evaluator.weights(final_decision["selected_factors"], recipe)
    ledger = evaluator.ledger_from_weights(weights)
    live_weights = weights.loc[start:latest]
    live_ledger = ledger.loc[start:latest]
    if live_ledger.empty:
        return None
    metrics = {
        "evidence": "fold_4_training_winner_frozen_before_simulated_live",
        "start": str(start.date()),
        "end": str(latest.date()),
        "selected_candidate": final_decision["selected_candidate"],
        "selected_factors": final_decision["selected_factors"],
        "selected_recipe": final_decision["selected_recipe"],
        **performance_metrics(live_ledger["net_return"], initial_anchor=False),
    }
    live_weights.to_csv(output / "simulated_live_weights.csv")
    live_ledger.to_csv(output / "simulated_live_ledger.csv")
    _json_dump(output / "simulated_live_metrics.json", metrics)
    evaluator.clear_transient_caches()
    return metrics


def _rank_fixed_recipes_for_fold(
    evaluator: PortfolioEvaluator,
    factors: Sequence[str],
    recipes: Mapping[str, PortfolioRecipe],
    fold: Mapping[str, str],
) -> list[dict]:
    """Rank a predeclared recipe set using one fold's training data only."""

    segments = calendar_segments(
        pd.Timestamp(fold["train_start"]), pd.Timestamp(fold["train_end"]), years=2
    )
    rows = []
    for challenger, recipe in recipes.items():
        ledger = evaluator.ledger(factors, recipe)
        summary = robust_summary(ledger, segments, initial_anchor=True)
        train = ledger.loc[fold["train_start"]:fold["train_end"]]
        rows.append({
            "challenger": challenger,
            **recipe.to_dict(),
            **summary,
            "annual_turnover": (
                float(
                    train["executed_traded_notional"].iloc[1:].mean()
                    * TRADING_DAYS_PER_YEAR
                )
                if len(train) > 1
                else np.nan
            ),
        })
    return sorted(
        rows,
        key=lambda row: (
            *robustness_key(row),
            -float(row.get("annual_turnover", np.inf)),
        ),
        reverse=True,
    )


def _recipe_walk_forward_8f(
    evaluator: PortfolioEvaluator,
    output: Path,
) -> tuple[pd.DataFrame, list[dict]]:
    """Freeze the 8f recipe choice per fold after training-only comparison.

    The two recipes are declared before each fold is evaluated.  This makes the
    mechanical selection causal, while the alternative recipe remains a
    retrospectively generated research hypothesis until future shadow evidence
    accumulates.
    """

    decisions = []
    selected_parts = []
    for fold in OUTER_FOLDS:
        training = evaluator.bounded(fold["train_start"], fold["train_end"])
        ranked = _rank_fixed_recipes_for_fold(
            training, FACTORS_8F, RECIPE_CHALLENGERS_8F, fold
        )
        winner = ranked[0]
        recipe = _recipe_from_row(winner)
        training.clear_transient_caches()

        testing = evaluator.bounded(fold["test_start"], fold["test_end"])
        test_weights = testing.weights(FACTORS_8F, recipe).loc[
            fold["test_start"]:fold["test_end"]
        ]
        selected_parts.append(test_weights)
        test_ledger = testing.ledger_from_weights(test_weights)
        decisions.append({
            **dict(fold),
            "selected_challenger": winner["challenger"],
            "selected_recipe": recipe.to_dict(),
            "training_robustness": {
                key: winner[key]
                for key in (
                    "positive_segment_ratio", "worst_sharpe", "median_sharpe",
                    "median_annual_return", "worst_drawdown", "annual_turnover",
                )
            },
            "training_ranking": ranked,
            "test_metrics_diagnostic_only": performance_metrics(
                test_ledger["net_return"], initial_anchor=True
            ),
        })
        testing.clear_transient_caches()

    selected_weights = pd.concat(selected_parts).sort_index()
    selected_weights = selected_weights[~selected_weights.index.duplicated(keep="first")]
    selected_weights = selected_weights.reindex(columns=evaluator.runner.u).fillna(0.0)
    start, end = selected_weights.index.min(), selected_weights.index.max()
    selected_ledger = evaluator.ledger_from_weights(selected_weights).loc[start:end]
    selected_weights.to_csv(output / "8f_recipe_walk_forward_weights.csv")
    selected_ledger.to_csv(output / "8f_recipe_walk_forward_ledger.csv")
    _json_dump(output / "8f_recipe_walk_forward_decisions.json", decisions)

    ledgers = {"8f_recipe_walk_forward": selected_ledger}
    rows = [{
        "strategy": "8f_recipe_walk_forward",
        "evidence": "training_selected_test_frozen",
        **performance_metrics(selected_ledger["net_return"], initial_anchor=True),
    }]
    for challenger, recipe in RECIPE_CHALLENGERS_8F.items():
        weights = evaluator.weights(FACTORS_8F, recipe).loc[start:end]
        ledger = evaluator.ledger_from_weights(weights).loc[start:end]
        strategy = f"8f_{challenger}"
        ledgers[strategy] = ledger
        rows.append({
            "strategy": strategy,
            "evidence": (
                "fixed_production_baseline" if recipe == BASE_RECIPE
                else "fixed_retrospective_hypothesis"
            ),
            **performance_metrics(ledger["net_return"], initial_anchor=True),
        })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "8f_recipe_walk_forward_metrics.csv", index=False)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(14, 8))
    for name, ledger in ledgers.items():
        nav = ledger["nav"] / ledger["nav"].iloc[0] * 1000.0
        metric = metrics.loc[metrics["strategy"] == name].iloc[0]
        ax.plot(
            nav.index,
            nav.values,
            linewidth=2.0 if name == "8f_recipe_walk_forward" else 1.3,
            label=(
                f"{name} | 年化{metric['annual_return']:.1%} "
                f"夏普{metric['sharpe']:.2f} 回撤{metric['max_drawdown']:.1%}"
            ),
        )
    ax.set_title("8f组合方法扩展窗口冻结验证（总敞口2，扣费后）")
    ax.set_ylabel("净值（起点=1000）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "nav_8f_recipe_walk_forward.png", dpi=160)
    plt.close(fig)
    return metrics, decisions


def _comparison(
    evaluator: PortfolioEvaluator,
    adaptive_weights: pd.DataFrame,
    adaptive_ledger: pd.DataFrame,
    output: Path,
) -> pd.DataFrame:
    start = adaptive_weights.index.min()
    end = adaptive_weights.index.max()
    rows = []
    navs = {"adaptive_search": adaptive_ledger["nav"]}
    adaptive_metrics = performance_metrics(
        adaptive_ledger["net_return"], initial_anchor=True
    )
    rows.append({"strategy": "adaptive_search", **adaptive_metrics})
    for name, factors in KNOWN_FACTOR_SETS.items():
        weights = evaluator.weights(factors, BASE_RECIPE)
        deployed = pd.DataFrame(0.0, index=evaluator.dates, columns=evaluator.runner.u)
        deployed.loc[start:end] = weights.loc[start:end]
        ledger = evaluator.ledger_from_weights(deployed).loc[start:end]
        rows.append({
            "strategy": name,
            **performance_metrics(ledger["net_return"], initial_anchor=True),
        })
        navs[name] = ledger["nav"]

    full_adaptive = pd.DataFrame(0.0, index=evaluator.dates, columns=evaluator.runner.u)
    full_adaptive.loc[start:end] = adaptive_weights
    stress = evaluator.ledger_from_weights(full_adaptive, cost_multiplier=2.0).loc[start:end]
    rows.append({
        "strategy": "adaptive_search_cost_2x",
        **performance_metrics(stress["net_return"], initial_anchor=True),
    })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "comparison_metrics.csv", index=False)

    pd.DataFrame({
        name: nav.loc[start:end] / nav.loc[start:end].iloc[0] * 1000.0
        for name, nav in navs.items()
    }).to_csv(output / "nav_comparison.csv")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(14, 8))
    for name, nav in navs.items():
        nav = nav.loc[start:end].dropna()
        normalized = nav / nav.iloc[0] * 1000.0
        metric = metrics.loc[metrics["strategy"] == name].iloc[0]
        ax.plot(
            normalized.index,
            normalized.values,
            label=(
                f"{name} | 年化{metric['annual_return']:.1%} "
                f"夏普{metric['sharpe']:.2f} 回撤{metric['max_drawdown']:.1%}"
            ),
            linewidth=2.0 if name == "adaptive_search" else 1.2,
        )
    ax.set_title("扩展窗口历史搜索：滚动样本外净值（总敞口2，扣费后）")
    ax.set_ylabel("净值（起点=1000）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "nav_adaptive_oos_comparison.png", dpi=160)
    plt.close(fig)
    return metrics


def _write_review(
    output: Path,
    decisions: Sequence[dict],
    metrics: pd.DataFrame,
) -> None:
    adaptive = metrics.loc[metrics["strategy"] == "adaptive_search"].iloc[0]
    stress = metrics.loc[metrics["strategy"] == "adaptive_search_cost_2x"].iloc[0]
    lines = [
        "# 冻结候选清单：扩展窗口历史方法与因子集搜索",
        "",
        "> 本研究固定使用本次预声明候选清单，不重建历史因子研发时间。所有统计、聚类、",
        "> 方法选择和因子选择只使用各折训练期；下一测试期完全冻结。结果属于严格时序的",
        "> 反事实历史研究，不等同于真实历史实盘。",
        "",
        "## 滚动样本外结果",
        "",
        f"- 年化收益：{adaptive['annual_return']:.2%}",
        f"- 年化波动：{adaptive['annual_volatility']:.2%}",
        f"- 夏普：{adaptive['sharpe']:.2f}",
        f"- 最大回撤：{adaptive['max_drawdown']:.2%}",
        f"- 双倍交易成本夏普：{stress['sharpe']:.2f}",
        "",
        "## 固定方案对照",
        "",
        "| 方案 | 年化收益 | 夏普 | 最大回撤 |",
        "|---|---:|---:|---:|",
    ]
    for _, row in metrics.loc[
        metrics["strategy"] != "adaptive_search_cost_2x"
    ].iterrows():
        lines.append(
            f"| {row['strategy']} | {row['annual_return']:.2%} | "
            f"{row['sharpe']:.2f} | {row['max_drawdown']:.2%} |"
        )
    lines.extend([
        "",
        "## 各折训练期选择",
        "",
        "| 折 | 训练期 | 测试期 | 候选 | 因子数 | 方法 | 测试期夏普（诊断） |",
        "|---|---|---|---|---:|---|---:|",
    ])
    for decision in decisions:
        recipe = decision["selected_recipe"]
        lines.append(
            f"| {decision['fold']} | {decision['train_start']}~{decision['train_end']} | "
            f"{decision['test_start']}~{decision['test_end']} | "
            f"{decision['selected_candidate']} | {len(decision['selected_factors'])} | "
            f"{recipe['name']} | "
            f"{decision['test_metrics_diagnostic_only']['sharpe']:.2f} |"
        )
    jaccards = [
        factor_set_jaccard(left["selected_factors"], right["selected_factors"])
        for left, right in zip(decisions, decisions[1:])
    ]
    lines.extend([
        "",
        "## 稳定性与使用边界",
        "",
        f"相邻折因子集 Jaccard 均值：{np.mean(jaccards):.2f}" if jaccards else "相邻折不足，无法计算稳定性。",
        "",
        "最终评价以拼接后的 `adaptive_oos_ledger.csv` 为准；单折测试指标只用于诊断，",
        "没有参与该折选择。生产策略和正式配置未被修改。自适应方案即使优于固定基线，",
        "也因各折方法和因子集会变化而只构成候选证据，不能自动替代生产。",
        "",
    ])
    (output / "REVIEW.md").write_text("\n".join(lines), encoding="utf-8")


def _search_source_snapshot(start: str, end: str | None = None) -> dict:
    from data.manager import DataManager
    from research.portfolio_experiment_support import latest_local_date
    manager = DataManager.from_config(load_config("config/default.yaml"))
    try:
        end = end or str(pd.Timestamp(latest_local_date(manager)).date())
        return {"end": end, "source_fingerprint":
                manager.source.checkpoint_source_fingerprint(start, end)}
    finally:
        manager.source.close()


def prepare_neighborhood_search(output: Path, candidate_manifest: Path,
                                validation_results: Path) -> dict:
    """Freeze an explicit native-factor study without changing either library."""
    from core.config import load_strategy_library
    from core.registry import get as registry_get
    from run_portfolio_workflow import _load_factor_definition, _legacy_recipe
    import factors.library  # built-ins only; never load mining catalogs

    config = load_config("config/default.yaml")
    catalog_path = ROOT / "config/strategy_library.yaml"
    catalog = load_strategy_library(str(catalog_path))
    sets = {row.id: row for row in catalog.factor_sets}
    library_path = ROOT / catalog.effective_factor_library
    library = {row["factor"]: row for row in json.loads(
        library_path.read_text(encoding="utf-8"))["factors"]}
    proofs = pd.read_csv(validation_results).set_index("factor")
    migration = json.loads(candidate_manifest.read_text(encoding="utf-8"))
    candidates = {}
    for row in migration:
        name = row["name"]
        cls = registry_get("factor", name)
        if cls.__module__ != "factors.library.intraday":
            raise ValueError(f"candidate is not a native intraday factor: {name}")
        proof = proofs.loc[name]
        if proof["final_pass"] != True or int(cls.expected_direction) != 1:
            raise ValueError(f"candidate admission/orientation mismatch: {name}")
        if cls.input_bar_frequency != row["frequency"]:
            raise ValueError(f"input frequency mismatch: {name}")
        # Native migration already incorporates the old training orientation.
        candidates[name] = {"direction": 1, "input_bar_frequency": row["frequency"],
                            "best_period": int(proof["best_period"])}
    spec = library["volume_price_corr_20d"]
    candidates[spec["factor"]] = {key: spec[key] for key in
                                 ("direction", "input_bar_frequency", "best_period")}
    inputs = [ROOT / "config/default.yaml", ROOT / "config/local.yaml",
              catalog_path, library_path, candidate_manifest, validation_results,
              ROOT / spec["evidence_file"]]
    seeds = []
    signatures = set()
    total = 0
    for strategy in catalog.strategies:
        if strategy.status == "archived":
            continue
        if strategy.factor_definition_path:
            path = ROOT / strategy.factor_definition_path
            definition = _load_factor_definition(path)
            directions = definition["directions"]
            inputs.append(path)
        else:
            names = sets[strategy.factor_set_id].factors
            directions = {name: int(library[name]["direction"]) for name in names}
        merged = {name: row["direction"] for name, row in candidates.items()}
        merged.update(directions)
        count = 0
        for members, _removed, _added in factor_neighborhood(directions, candidates):
            # Direction is part of identity: same names with different frozen
            # signs must never be collapsed into one backtest.
            signature = tuple((name, merged[name]) for name in members)
            signatures.add(signature)
            count += 1
        total += count
        seeds.append({"id": strategy.id, "name": strategy.name,
                      "status": strategy.status, "directions": directions,
                      "neighborhood_count": count})
    base = _legacy_recipe(config)
    recipes = portfolio_recipe_grid(base)
    panel_start = str((pd.Timestamp(HISTORICAL_START) - pd.Timedelta(days=365)).date())
    source_snapshot = _search_source_snapshot(panel_start)
    contract = {
        "schema_version": 1, "mode": "native_neighborhood_cartesian",
        "performance_start": HISTORICAL_START,
        "panel_start": panel_start,
        **source_snapshot,
        "selection_end": "2026-05-15", "observation_start": "2026-05-16",
        "historical_folds": OUTER_FOLDS,
        "fold_role": "historical_stability_not_independent_factor_discovery_oos",
        "search_rule": {"max_remove": 2, "max_add": 2, "minimum_size": 2,
                        "performance_pruning": False, "global_subset_exhaustive": False},
        "candidates": candidates, "seeds": seeds,
        "baseline_recipe": base.to_dict(), "recipes": [r.to_dict() for r in recipes],
        "counts": {"seeds": len(seeds), "candidates": len(candidates),
                   "recipes": len(recipes), "baseline_jobs": len(seeds) * len(recipes),
                   "memberships_before_dedup": total,
                   "memberships_after_direction_aware_dedup": len(signatures),
                   "jobs_before_dedup": total * len(recipes),
                   "jobs_after_dedup": len(signatures) * len(recipes)},
        "frozen_runtime": {"ic_horizon": 1,
                           "ic_window": config.production_portfolio.ic_window,
                           "risk_lookback_calendar_days": config.production_portfolio.risk_lookback_calendar_days,
                           "costs": configured_futures_cost_model().ledger_parameters()},
        "input_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in inputs if path.is_file()},
        "factor_source_sha256": Runner._source_tree_fingerprint(),
    }
    output.mkdir(parents=True, exist_ok=False)
    _json_dump(output / "search_contract.json", contract)
    return contract


def benchmark_neighborhood_search(output: Path) -> dict:
    """Measure small/large frozen baselines before authorizing millions of jobs.

    This pilot excludes new-factor computation and cannot rank the search pool.
    Its ledgers use the same PortfolioEvaluator as the production backtest.
    """
    import cProfile
    import pstats
    from run_portfolio_workflow import _peak_working_set_mib

    contract = json.loads((output / "search_contract.json").read_text(encoding="utf-8"))
    for name, digest in contract["input_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"frozen input changed: {name}")
    if Runner._source_tree_fingerprint() != contract["factor_source_sha256"]:
        raise ValueError("factor implementation changed after search freeze")
    source_snapshot = _search_source_snapshot(contract["panel_start"], contract["end"])
    if source_snapshot["source_fingerprint"] != contract["source_fingerprint"]:
        raise ValueError("market data changed after search freeze")
    if (output / "pilot_profile.json").exists():
        raise FileExistsError("pilot already completed; preserve its measured evidence")
    ordered = sorted(contract["seeds"], key=lambda row: (len(row["directions"]), row["id"]))
    seeds = [ordered[0], ordered[-1]]
    names = sorted({name for seed in seeds for name in seed["directions"]})
    started, cpu_started = time.perf_counter(), time.process_time()
    print(f"pilot factor panel: {len(names)} native factors, {contract['panel_start']} to {contract['end']}", flush=True)
    try:
        runner = Runner(names, start=contract["panel_start"], end=contract["end"],
                        factor_directions={name: 1 for name in names}, ic_horizon=1,
                        checkpoint_dir=output / "factor_panel_checkpoint")
    except RuntimeError as exc:
        _json_dump(output / "pilot_profile.json", {
            "status": "blocked_during_panel_preparation", "error": str(exc),
            "wall_seconds": time.perf_counter() - started,
            "cpu_seconds": time.process_time() - cpu_started,
            "peak_process_mib": _peak_working_set_mib(), "jobs": [],
        })
        raise
    runner.get_contract_schedule()
    factor_seconds = time.perf_counter() - started
    _json_dump(output / "pilot_factor_profile.json", runner.performance_profile())
    base_values = dict(contract["baseline_recipe"])
    base_values.pop("name")
    for key in ("asset_max_overrides", "sector_weight_caps"):
        base_values[key] = tuple(tuple(pair) for pair in base_values[key])
    base = PortfolioRecipe(**base_values)
    # Fixed before observing results: cover all four factor-weight methods,
    # all three allocation methods, and the newly requested Top5.
    choices = [("equal", "equal"), ("diag_icir", "inverse_volatility"),
               ("lw_abs", "erc"), ("lw_positive", "erc")]
    recipes = [base] + [r for r in portfolio_recipe_grid(base)
                        if r.top_n == 5 and r.sector_cap == 3
                        and (r.factor_weight, r.asset_weight) in choices]
    rows = []
    for seed in seeds:
        factors = sorted(seed["directions"])
        view = runner.for_factors(factors, factor_directions=seed["directions"])
        runtime = contract["frozen_runtime"]
        evaluator = PortfolioEvaluator(view, start=contract["performance_start"],
                                       end=contract["end"], cost_model=COST_MODEL,
                                       ic_window=runtime["ic_window"],
                                       risk_lookback_calendar_days=runtime["risk_lookback_calendar_days"])
        for recipe in recipes:
            evaluator.clear_transient_caches()
            profiler = cProfile.Profile()
            wall, cpu = time.perf_counter(), time.process_time()
            row = {"strategy": seed["id"], "factor_count": len(factors),
                   "recipe": recipe.to_dict(), "status": "failed"}
            print(f"pilot backtest: {seed['id']} / {recipe.name}", flush=True)
            try:
                profiler.enable()
                ledger = evaluator.ledger(factors, recipe)
                profiler.disable()
                row.update(status="completed", observations=len(ledger),
                           first_date=str(ledger.index.min().date()),
                           last_date=str(ledger.index.max().date()))
                ledger.to_parquet(output / f"pilot_{seed['id']}__{recipe.name}.parquet")
            except (RuntimeError, ValueError) as exc:
                row["error"] = str(exc)
            finally:
                profiler.disable()
                row.update(wall_seconds=time.perf_counter() - wall,
                           cpu_seconds=time.process_time() - cpu,
                           peak_process_mib=_peak_working_set_mib())
                stats = pstats.Stats(profiler).stats
                row["hotspots"] = [
                    {"file": key[0], "line": key[1], "function": key[2],
                     "calls": value[1], "self_seconds": value[2], "cumulative_seconds": value[3]}
                    for key, value in sorted(stats.items(), key=lambda item: item[1][3], reverse=True)[:15]
                ]
                rows.append(row)
                _json_dump(output / "pilot_progress.json", rows)
    profile = {"role": "cost_measurement_not_selection", "profile_overhead_included": True,
               "new_factor_compute_included": False, "factor_panel_seconds": factor_seconds,
               "wall_seconds": time.perf_counter() - started,
               "cpu_seconds": time.process_time() - cpu_started,
               "peak_process_mib": _peak_working_set_mib(), "jobs": rows}
    _json_dump(output / "pilot_profile.json", profile)
    return profile


def verify_neighborhood_ledgers(output: Path) -> dict:
    """Check the optimized scorer against all ten saved real pilot ledgers."""
    contract = json.loads((output / "search_contract.json").read_text(encoding="utf-8"))
    pilot = json.loads((output / "pilot_profile.json").read_text(encoding="utf-8"))
    checkpoint = output / "factor_panel_checkpoint"
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    names = manifest["contract"]["factors"]
    runner = Runner(names, start=contract["panel_start"], end=contract["end"],
                    factor_directions={name: 1 for name in names}, ic_horizon=1,
                    checkpoint_dir=checkpoint)
    if runner.computed_factor_count:
        raise RuntimeError("parity check must reuse a complete pilot checkpoint")
    runner.get_contract_schedule()
    seeds = {seed["id"]: seed for seed in contract["seeds"]}
    runtime = contract["frozen_runtime"]
    results = []
    for row in pilot["jobs"]:
        if row["status"] != "completed":
            raise ValueError("pilot has failed jobs; inspect before ledger verification")
        seed = seeds[row["strategy"]]
        factors = sorted(seed["directions"])
        view = runner.for_factors(factors, factor_directions=seed["directions"])
        evaluator = PortfolioEvaluator(view, start=contract["performance_start"], end=contract["end"],
                                       cost_model=COST_MODEL, ic_window=runtime["ic_window"],
                                       risk_lookback_calendar_days=runtime["risk_lookback_calendar_days"])
        values = dict(row["recipe"])
        values.pop("name")
        for key in ("asset_max_overrides", "sector_weight_caps"):
            values[key] = tuple(tuple(pair) for pair in values[key])
        recipe = PortfolioRecipe(**values)
        started = time.perf_counter()
        actual = evaluator.ledger(factors, recipe)
        seconds = time.perf_counter() - started
        reference = pd.read_parquet(output / f"pilot_{seed['id']}__{recipe.name}.parquet")
        pd.testing.assert_frame_equal(actual, reference, check_exact=True)
        results.append({"strategy": seed["id"], "recipe": recipe.name,
                        "bitwise_equal": True, "unprofiled_seconds": seconds})
        print(f"ledger parity: {seed['id']} / {recipe.name} / {seconds:.2f}s", flush=True)
        _json_dump(output / "ledger_parity_progress.json", results)
    report = {"status": "passed", "jobs": results}
    _json_dump(output / "ledger_parity.json", report)
    return report


def run_neighborhood_batch(output: Path, *, baseline_only: bool = False,
                           max_jobs: int | None = None) -> dict:
    """Resume exact ledgers in one locked result database, with no rank pruning.

    The completed pilot is required before starting. Its raw factor checkpoint
    is verified using the Runner's own full contract check and reused; only
    missing native factors are computed. No mining or library mutation occurs.
    """
    import duckdb

    if max_jobs is not None and max_jobs < 1:
        raise ValueError("max_jobs must be positive")
    contract_path = output / "search_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    pilot = json.loads((output / "pilot_profile.json").read_text(encoding="utf-8"))
    if pilot.get("role") != "cost_measurement_not_selection" or len(pilot.get("jobs", [])) != 10:
        raise ValueError("complete the cost pilot before starting the joint search")
    parity = json.loads((output / "ledger_parity.json").read_text(encoding="utf-8"))
    if parity.get("status") != "passed":
        raise ValueError("complete real ledger parity before starting the joint search")
    for name, digest in contract["input_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"frozen input changed: {name}")
    if Runner._source_tree_fingerprint() != contract["factor_source_sha256"]:
        raise ValueError("factor code changed after search freeze")
    if _search_source_snapshot(contract["panel_start"], contract["end"])["source_fingerprint"] != contract["source_fingerprint"]:
        raise ValueError("market data changed after search freeze")
    fingerprint = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    execution_hash = hashlib.sha256()
    import inspect
    execution_hash.update(inspect.getsource(run_neighborhood_batch).encode("utf-8"))
    execution_files = [ROOT / "research/historical_portfolio_search.py"]
    for directory in ("backtest", "optimization"):
        execution_files.extend(sorted((ROOT / directory).rglob("*.py")))
    for path in sorted(execution_files):
        execution_hash.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        execution_hash.update(path.read_bytes())
    execution_fingerprint = execution_hash.hexdigest()
    # DuckDB's process-exclusive writer lock prevents duplicate workers before
    # any expensive factor preparation begins. Results live only in this run.
    with duckdb.connect(str(output / "search_results.duckdb")) as db:
        db.execute("CREATE TABLE IF NOT EXISTS study (fingerprint VARCHAR PRIMARY KEY, execution_fingerprint VARCHAR)")
        existing = db.execute("SELECT fingerprint, execution_fingerprint FROM study").fetchall()
        if existing and existing != [(fingerprint, execution_fingerprint)]:
            raise ValueError("result database belongs to a different frozen contract or execution code")
        if not existing:
            db.execute("INSERT INTO study VALUES (?, ?)", [fingerprint, execution_fingerprint])
        db.execute("""CREATE TABLE IF NOT EXISTS memberships (
            id INTEGER PRIMARY KEY, seed VARCHAR, directions VARCHAR,
            removed VARCHAR, added VARCHAR, original BOOLEAN)""")
        db.execute("""CREATE TABLE IF NOT EXISTS results (
            id BIGINT PRIMARY KEY, membership_id INTEGER, recipe_id INTEGER,
            status VARCHAR, error VARCHAR, wall_seconds DOUBLE, cpu_seconds DOUBLE,
            selection_metrics VARCHAR, observation_metrics VARCHAR,
            dates DATE[], nav DOUBLE[], net_returns DOUBLE[])""")
        directions = {name: int(row["direction"]) for name, row in contract["candidates"].items()}
        for seed in contract["seeds"]:
            for name, direction in seed["directions"].items():
                if name in directions and directions[name] != direction:
                    raise ValueError("shared panel requires consistent frozen directions")
                directions[name] = int(direction)
        names = sorted(directions)
        checkpoint = output / "factor_panel_checkpoint"
        old_manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
        old_names = old_manifest["contract"]["factors"]
        missing = sorted(set(names) - set(old_names))
        print(f"search panel: {len(names)} factors; reuse {len(old_names)}, compute {len(missing)}", flush=True)
        runner = Runner(missing or old_names, start=contract["panel_start"], end=contract["end"],
                        factor_directions=directions, ic_horizon=contract["frozen_runtime"]["ic_horizon"],
                        checkpoint_dir=(output / "additional_factor_panel_checkpoint") if missing else checkpoint)
        if not set(old_names).issubset(names):
            raise ValueError("pilot contains factors outside the declared search")
        raw, _ = runner._load_factor_checkpoint(checkpoint, old_names)
        if set(raw) != set(old_names):
            raise ValueError("pilot factor checkpoint is incomplete")
        runner.raw_ranks.update({name: frame.rank(axis=1, pct=True) for name, frame in raw.items()})
        runner = runner.for_factors(names, factor_directions=directions)
        runner.get_contract_schedule()
        _json_dump(output / "additional_factor_profile.json", runner.performance_profile())
        runner.env = CausalEligibilityEnvironment(
            runner.cal, runner.daily_ret, runner.env.sector_of)
        gc.collect()
        runtime = contract["frozen_runtime"]
        evaluator = PortfolioEvaluator(runner, start=contract["performance_start"], end=contract["end"],
                                       cost_model=COST_MODEL, ic_window=runtime["ic_window"],
                                       risk_lookback_calendar_days=runtime["risk_lookback_calendar_days"])
        recipes = []
        for record in contract["recipes"]:
            values = dict(record)
            values.pop("name")
            for key in ("asset_max_overrides", "sector_weight_caps"):
                values[key] = tuple(tuple(pair) for pair in values[key])
            recipes.append(PortfolioRecipe(**values))
        performed = 0
        stop = False
        baseline_recipe_id = next(i for i, record in enumerate(contract["recipes"])
                                  if record == contract["baseline_recipe"])
        segments = calendar_segments(pd.Timestamp(contract["performance_start"]),
                                     pd.Timestamp(contract["selection_end"]), years=2)
        def scheduled_members():
            # Establish all eight current baselines before any alternatives.
            # This changes execution order only, never job IDs or coverage.
            for baseline_pass in (True, False):
                for index, member in enumerate(unique_factor_neighborhoods(
                        contract["seeds"], contract["candidates"])):
                    if (baseline_pass or baseline_only) and not member["original"]:
                        break
                    yield index, member, ([baseline_recipe_id] if baseline_pass
                                          else range(len(recipes)))
        for index, member, recipe_ids in scheduled_members():
            db.execute("INSERT INTO memberships VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
                       [index, member["seed"], json.dumps(member["directions"], sort_keys=True),
                        json.dumps(member["removed"]), json.dumps(member["added"]), member["original"]])
            done = {row[0] for row in db.execute("SELECT recipe_id FROM results WHERE membership_id=?", [index]).fetchall()}
            factors = list(member["directions"])
            for recipe_id in recipe_ids:
                if recipe_id in done:
                    continue
                recipe = recipes[recipe_id]
                wall, cpu = time.perf_counter(), time.process_time()
                status, error, selection, observation = "completed", "", {}, {}
                dates, nav, returns = [], [], []
                try:
                    ledger = evaluator.ledger(factors, recipe)
                    selected = ledger.loc[:contract["selection_end"], "net_return"]
                    observed = ledger.loc[contract["observation_start"]:, "net_return"]
                    selection = performance_metrics(selected, initial_anchor=True)
                    selection.update(robust_summary(ledger, segments, initial_anchor=True))
                    selection["segments"] = [
                        {"start": str(left.date()), "end": str(right.date()),
                         **performance_metrics(
                             ledger.loc[left:right, "net_return"],
                             initial_anchor=bool(left == pd.Timestamp(contract["performance_start"])))}
                        for left, right in segments]
                    observation = performance_metrics(observed)
                    dates = list(ledger.index.date)
                    nav = ledger["nav"].tolist()
                    returns = ledger["net_return"].tolist()
                except (RuntimeError, ValueError) as exc:
                    status, error = "failed", str(exc)
                db.execute("INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           [index * len(recipes) + recipe_id, index, recipe_id, status, error,
                            time.perf_counter() - wall, time.process_time() - cpu,
                            json.dumps(selection), json.dumps(observation), dates, nav, returns])
                # Keep the four reusable score matrices, not 96 full ledgers.
                evaluator._ledger_cache.clear()
                performed += 1
                print(f"search {index}:{recipe_id} {status}", flush=True)
                counts = dict(db.execute("SELECT status, count(*) FROM results GROUP BY status").fetchall())
                _json_dump(output / "search_progress.json", {
                    "performed_this_batch": performed, "counts": counts,
                    "planned_full_jobs": int(contract["counts"]["jobs_after_dedup"]),
                    "baseline_only": baseline_only, "full_search_complete": False,
                    "last_job": index * len(recipes) + recipe_id})
                if max_jobs is not None and performed >= max_jobs:
                    stop = True
                    break
            evaluator.clear_transient_caches()
            if stop:
                break
        counts = dict(db.execute("SELECT status, count(*) FROM results GROUP BY status").fetchall())
        planned = int(contract["counts"]["jobs_after_dedup"])
        progress = {"performed_this_batch": performed, "counts": counts,
                    "planned_full_jobs": planned, "baseline_only": baseline_only,
                    "full_search_complete": sum(counts.values()) == planned}
        _json_dump(output / "search_progress.json", progress)
        return progress


def report_neighborhood_search(output: Path) -> dict:
    """Report exact completed jobs without using observation data to rank them.

    Run after the single writer has closed its batch. All completed NAVs are
    streamed into the framework plot; only the labelled foreground is limited.
    A partial database always produces an explicitly partial report.
    """
    import duckdb
    from run_portfolio_workflow import _write_comparison_plot

    contract_path = output / "search_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    baseline_id = contract["recipes"].index(contract["baseline_recipe"])
    from research.governance import factor_family
    from workflows.factor_selection import _exposure_correlation, _cluster, CLUSTER_CORRELATION_THRESHOLD
    import factors.library  # register native metadata; mining remains opt-in
    directions = {name: int(row["direction"]) for name, row in contract["candidates"].items()}
    for seed in contract["seeds"]:
        directions.update(seed["directions"])
    ranks = {}
    for directory in ("factor_panel_checkpoint", "additional_factor_panel_checkpoint"):
        folder = output / directory
        if not folder.exists():
            continue
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        for name, filename in manifest["completed"].items():
            frame = pd.read_parquet(folder / filename).loc[contract["performance_start"]:contract["selection_end"]]
            ranks[name] = frame.rank(axis=1, pct=True) * directions[name]
    if set(ranks) != set(directions):
        raise ValueError("factor structure report requires the complete frozen panel")
    names = sorted(ranks)
    corr = _exposure_correlation(ranks, names)
    clusters = _cluster(corr, names)
    factor_details = {name: {**contract["candidates"].get(name, {}),
                             "family": factor_family(name), "correlation_cluster": clusters[name],
                             "direction": directions[name], "candidate": name in contract["candidates"],
                             "coverage": float(ranks[name].notna().to_numpy().mean()),
                             "strongest_peers": corr[name].drop(name).abs().nlargest(3).to_dict()}
                      for name in names}
    _json_dump(output / "factor_structure.json", {
        "start": contract["performance_start"], "end": contract["selection_end"],
        "method": "absolute_daily_cross_section_rank_correlation_complete_linkage",
        "threshold": CLUSTER_CORRELATION_THRESHOLD,
        "role": "diagnostic_only_no_candidate_pruning", "factors": factor_details})
    del ranks
    with duckdb.connect(str(output / "search_results.duckdb"), read_only=True) as db:
        expected = hashlib.sha256(contract_path.read_bytes()).hexdigest()
        if db.execute("SELECT fingerprint FROM study").fetchall() != [(expected,)]:
            raise ValueError("report contract does not match result database")
        counts = dict(db.execute("SELECT status, count(*) FROM results GROUP BY status").fetchall())
        planned = int(contract["counts"]["jobs_after_dedup"])
        attempted = sum(counts.values())
        complete = attempted == planned
        recipe_table = pd.DataFrame([{**r, "recipe_id": i} for i, r in enumerate(contract["recipes"])])
        db.register("recipe_table", recipe_table)
        db.execute("""CREATE TEMP VIEW performance AS
            SELECT r.* EXCLUDE(dates,nav,net_returns), m.seed, m.original,
                   m.directions, m.removed, m.added, p.name AS recipe,
                   p.factor_weight, p.top_n, p.sector_cap, p.asset_weight,
                   json_extract(r.selection_metrics,'$.annual_return')::DOUBLE AS annual_return,
                   json_extract(r.selection_metrics,'$.sharpe')::DOUBLE AS sharpe,
                   json_extract(r.selection_metrics,'$.max_drawdown')::DOUBLE AS max_drawdown,
                   json_extract(r.selection_metrics,'$.positive_segment_ratio')::DOUBLE AS positive_segment_ratio,
                   json_extract(r.selection_metrics,'$.worst_sharpe')::DOUBLE AS worst_sharpe
            FROM results r JOIN memberships m ON m.id=r.membership_id
            JOIN recipe_table p ON p.recipe_id=r.recipe_id""")
        # Full machine-readable table, including failures and provenance.
        destination = str((output / "all_performance.parquet").resolve()).replace("'", "''")
        db.execute(f"COPY (SELECT * FROM performance ORDER BY id) TO '{destination}' (FORMAT PARQUET)")
        baselines = db.execute("SELECT id FROM performance WHERE original AND recipe_id=? ORDER BY membership_id",
                               [baseline_id]).fetchall()
        highlight_ids = [int(row[0]) for row in baselines]
        highlighted_paths = [db.execute("SELECT nav,net_returns FROM results WHERE id=?", [job_id]).fetchone()
                             for job_id in highlight_ids]
        # Several complementary objectives, all confined to the selection era.
        # Report leaders, not a claim of statistical significance after search.
        leaders = []
        for objective in ("worst_sharpe", "sharpe", "annual_return", "max_drawdown"):
            leaders.append(db.execute(f"""SELECT id FROM performance
                WHERE status='completed' AND NOT (original AND recipe_id=?)
                ORDER BY positive_segment_ratio DESC NULLS LAST,
                         {objective} DESC NULLS LAST, id LIMIT 20""", [baseline_id]).fetchall())
        for row in (row for rank in zip(*leaders) for row in rank):
            if int(row[0]) not in highlight_ids:
                path = db.execute("SELECT nav,net_returns FROM results WHERE id=?", [int(row[0])]).fetchone()
                if path in highlighted_paths:
                    continue
                highlight_ids.append(int(row[0]))
                highlighted_paths.append(path)
            if len(highlight_ids) >= len(baselines) + 5:
                break
        seeds = {seed["id"]: seed for seed in contract["seeds"]}
        navs, plot_rows, details = {}, [], []
        for job_id in highlight_ids:
            record = db.execute("SELECT * FROM performance WHERE id=?", [job_id]).fetchdf().iloc[0].to_dict()
            seed = seeds[record["seed"]]
            label = str(seed.get("name", seed["id"]))
            is_baseline = bool(record["original"]) and int(record["recipe_id"]) == baseline_id
            if not is_baseline:
                label = f"候选{job_id}·{label}"
            metrics, folds = {}, []
            if record["status"] == "completed":
                dates, nav, returns = db.execute("SELECT dates,nav,net_returns FROM results WHERE id=?", [job_id]).fetchone()
                index = pd.DatetimeIndex(dates)
                navs[label] = pd.Series(nav, index=index) / float(nav[0])
                series = pd.Series(returns, index=index, dtype=float)
                metrics = performance_metrics(series, initial_anchor=True)
                metrics["volatility"] = metrics.get("annual_volatility")
                for fold in contract["historical_folds"]:
                    folds.append({"fold": fold["fold"], "role": contract["fold_role"],
                                  "start": fold["test_start"], "end": fold["test_end"],
                                  **performance_metrics(series.loc[fold["test_start"]:fold["test_end"]])})
            plot_rows.append({"strategy": label, "status": "ok" if metrics else "failed_backtest", **metrics})
            details.append({"job_id": job_id, "label": label, "baseline": is_baseline,
                            "recipe": contract["recipes"][int(record["recipe_id"])],
                            "directions": json.loads(record["directions"]),
                            "removed": json.loads(record["removed"]), "added": json.loads(record["added"]),
                            "status": record["status"], "error": record["error"],
                            "selection": json.loads(record["selection_metrics"]),
                            "observation": json.loads(record["observation_metrics"]),
                            "full": metrics, "historical_diagnostics": folds})
            members = details[-1]["directions"]
            details[-1]["structure"] = {
                "factor_count": len(members),
                "families": sorted({factor_details[n]["family"] for n in members}),
                "correlation_cluster_count": len({clusters[n] for n in members}),
                "new_candidate_count": len(set(members) & set(contract["candidates"]))}
        _json_dump(output / "highlighted_comparisons.json", details)
        bounds = db.execute("""SELECT min(list_min(nav)/nav[1]), max(list_max(nav)/nav[1])
                               FROM results WHERE status='completed'""").fetchone()
        def background_batches():
            cursor = db.execute("SELECT dates,nav,id FROM results WHERE status='completed' ORDER BY id")
            while batch := cursor.fetchmany(128):
                yield pd.DataFrame({str(job_id): pd.Series(values, index=pd.DatetimeIndex(dates)) / float(values[0])
                                    for dates, values, job_id in batch})
        nav_table = pd.DataFrame(navs)
        if not nav_table.empty:
            _write_comparison_plot(output, nav_table, plot_rows,
                title=f"{'全量' if complete else '部分完成'}联合搜索：{counts.get('completed', 0):,}条净值（灰线），彩色为基准与候选",
                cutoff=pd.Timestamp(contract["selection_end"]),
                background_batches=background_batches(), background_limits=bounds)
        timing = db.execute("SELECT avg(wall_seconds), sum(wall_seconds), sum(cpu_seconds) FROM results").fetchone()
        failures = db.execute("SELECT error,count(*) FROM results WHERE status='failed' GROUP BY error ORDER BY count(*) DESC").fetchall()
        coverage = {"counts": counts, "planned": planned, "pending": planned - attempted,
                    "complete": complete, "mean_job_seconds": timing[0],
                    "accumulated_job_wall_seconds": timing[1], "accumulated_job_cpu_seconds": timing[2]}
        _json_dump(output / "report_coverage.json", coverage)
        lines = [f"# {'全量' if complete else '部分完成'}联合搜索结论报告", "",
                 f"已尝试 {attempted:,}/{planned:,} 项：成功 {counts.get('completed', 0):,}，失败 {counts.get('failed', 0):,}，尚未完成 {planned-attempted:,}。",
                 "未完成时不存在全量最优结论；失败项不伪造净值，也不丢弃其记录。", "",
                 f"回测：{contract['performance_start']} 至 {contract['end']}；预热从 {contract['panel_start']} 起。",
                 f"仅用截至 {contract['selection_end']} 的收益及分段稳定性选取展示候选；此后仅作模拟实盘观察。",
                 "当前8策略及所有试验共用修正后的行情和执行口径，旧留存净值不作为本次基准。",
                 "搜索边界：各种子增0–2、删0–2，成员方向去重后乘96种模式；不是任意数量增删的全球穷举。", "",
                 "## 当前基准与多目标候选", "",
                 "| 策略 | 选择期年化 | 选择期夏普 | 选择期回撤 | 分段盈利比例 | 后续观察年化 |",
                 "|---|---:|---:|---:|---:|---:|"]
        for row in details:
            s, o = row["selection"], row["observation"]
            if row["status"] != "completed":
                lines.append(f"| {row['label']} | 未形成：{row['error']} | — | — | — | — |")
            else:
                lines.append(f"| {row['label']} | {s.get('annual_return', np.nan):.2%} | {s.get('sharpe', np.nan):.2f} | "
                             f"{s.get('max_drawdown', np.nan):.2%} | {s.get('positive_segment_ratio', np.nan):.0%} | {o.get('annual_return', np.nan):.2%} |")
        lines.extend(["", "## 解释及产物", "",
                      "- all_performance.parquet：全部已尝试配置的指标、成员方向、增删与失败原因。",
                      "- highlighted_comparisons.json：图中基准/候选的完整配置、分期指标及既定历史折诊断。",
                      "- factor_structure.json：全50因子的注册经济分类、方向、覆盖率、暴露相关簇及最相似因子；仅诊断，不裁剪搜索。",
                      "- search_results.duckdb：逐项完整净值、收益与时间；nav_comparison.png覆盖所有成功净值，无抽样。",
                      "- 候选是选择期多目标表现领导者，不代表已校正大规模择优偏差或已获生产批准。",
                      "- 彩色候选剔除净值及逐日收益完全相同的重复模式；全部模式仍保留在结果表及灰色净值中。",
                      "- 历史折只是稳定性诊断，不声称因子发现独立样本外；观察期不用于排名。",
                      "- 排序使用选择期2年分段稳定性，包含末段不足2年的区间；既定历史折另列作诊断。",
                      "- 默认配置、有效库、生产/备选成员保持不变。",
                      f"- 已尝试任务平均耗时 {timing[0] or 0:.2f} 秒（不含公共因子准备）；资源评估必须计入缓存、失败率和准备成本。", ""])
        lines.extend(["## 失败原因与资源边界", ""])
        lines.extend(f"- {count}项：{error}" for error, count in failures)
        if not complete and timing[0]:
            years = (planned-attempted)*float(timing[0])/86400/365.25
            lines.extend(["", f"按本批实际平均吞吐粗算，剩余任务约需{years:.2f}个串行年；成员邻域的实际失败率和缓存收益可能不同，这不是精确工期。",
                          "现有加速及缓存仍不足以现实地完成联合穷举。不能把768项模式搜索当作包含14候选增删的完整搜索，也不能在未确认资源/工程方案前开启多年级任务。"])
        (output / "conclusion_report.md").write_text("\n".join(lines), encoding="utf-8")
        return coverage


def run_effective_pool_search(output: Path, reference: Path) -> dict:
    """Opt-in full-pool research, with frozen selection/observation boundaries."""
    import duckdb
    from research.effective_factor_library import load_library
    from workflows.factor_selection import (
        _run_budgeted_pool_search, _production_recipe, _configured_cost_model,
        _exposure_correlation, _cluster,
    )
    from run_portfolio_workflow import _write_comparison_plot
    old_path = reference / "search_contract.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    config = load_config("config/default.yaml")
    library_path = ROOT / config.factor_library.path
    effective = [r for r in load_library(library_path)["factors"] if r["status"] == "effective"]
    directions = {r["factor"]: int(r["direction"]) for r in effective}
    extras = {n: r for n, r in old["candidates"].items() if n not in directions}
    directions.update({n: int(r["direction"]) for n, r in extras.items()})
    import factors.library
    from core.registry import get as registry_get
    for name, metadata in [(r["factor"], r) for r in effective] + list(extras.items()):
        cls = registry_get("factor", name)
        if cls.__module__ != "factors.library.intraday" or str(getattr(cls, "input_bar_frequency", "1min")) != metadata["input_bar_frequency"]:
            raise ValueError(f"native factor/bar contract changed: {name}")
    for row in effective:
        if hashlib.sha256((ROOT / row["evidence_file"]).read_bytes()).hexdigest() != row["evidence_sha256"]:
            raise ValueError(f"admission evidence changed: {row['factor']}")
    for name, digest in old["input_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"reference input changed: {name}")
    recipe = _production_recipe(config)
    if json.loads(json.dumps(recipe.to_dict())) != old["baseline_recipe"]:
        raise ValueError("default recipe changed from latest-eight baseline")
    contract_path = output / "pool_contract.json"
    frozen_end = (json.loads(contract_path.read_text(encoding="utf-8"))["end"]
                  if contract_path.exists() else None)
    snapshot = _search_source_snapshot(old["panel_start"], frozen_end)
    inputs = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in
              (library_path, old_path, ROOT / "config/default.yaml", ROOT / "config/local.yaml",
               ROOT / "config/strategy_library.yaml", ROOT / "workflows/factor_selection.py",
               Path(__file__), ROOT / "research/historical_portfolio_search.py")}
    for directory in ("backtest", "optimization"):
        inputs.update({str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (ROOT / directory).rglob("*.py")})
    expected = {"schema_version": 1, "mode": "full_pool_budgeted_multistart",
        "reference": str(reference.resolve()), "input_sha256": inputs,
        "factor_source_sha256": Runner._source_tree_fingerprint(),
        "panel_start": old["panel_start"], "performance_start": old["performance_start"],
        "selection_end": "2026-05-15", "end": snapshot["end"],
        "source_fingerprint": snapshot["source_fingerprint"], "directions": directions,
        "effective_library_count": len(effective), "research_only_extras": extras,
        "admission_evidence_status": "frozen_prior_admission_not_recertified_after_continuous_plan_fix",
        "baseline_only_nonmembers": sorted({n for s in old["seeds"] for n in s["directions"]}-set(directions)),
        "recipe": recipe.to_dict(), "budget": 512, "beam_width": 6, "round_width": 16,
        "proxy_role": "proposal_order_only_not_portfolio_performance",
        "cutoff_ic_label": "exclude_last_selection_day_T_plus_1_label",
        "observation_role": "strictly_after_cutoff_no_selection",
        "minimum_risk_observations": int(config.production_portfolio.minimum_risk_observations),
        "covariance_shrinkage": float(config.production_portfolio.covariance_shrinkage)}
    expected = json.loads(json.dumps(expected))
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != expected:
            raise ValueError("pool contract/code/data changed; do not mix results")
    else:
        output.mkdir(parents=True, exist_ok=False)
        _json_dump(contract_path, expected)
    if recipe.constraints.minimum_risk_observations != expected["minimum_risk_observations"] or not np.isclose(
            recipe.constraints.covariance_shrinkage, expected["covariance_shrinkage"]):
        raise ValueError("research risk constraints differ from production")
    # Exclusive writer is acquired before expensive preparation.
    with duckdb.connect(str(output / "pool_results.duckdb")) as db:
        raw_paths = []
        if (snapshot["source_fingerprint"] == old["source_fingerprint"]
                and snapshot["end"] == old["end"] and expected["factor_source_sha256"] == old["factor_source_sha256"]):
            raw_paths = [reference / "factor_panel_checkpoint", reference / "additional_factor_panel_checkpoint"]
        reused = {n for path in raw_paths for n in json.loads(
            (path / "manifest.json").read_text(encoding="utf-8"))["contract"]["factors"]}
        all_names = sorted(set(directions) | {n for s in old["seeds"] for n in s["directions"]})
        missing = sorted(set(all_names)-reused)
        print(f"pool panel: {len(directions)} candidates; {len(all_names)} including baselines; reuse {len(reused)}; missing {len(missing)}", flush=True)
        panel_started = time.perf_counter()
        runner = Runner(missing or all_names, start=expected["panel_start"], end=expected["end"],
                        factor_directions=directions, ic_horizon=1, checkpoint_dir=output / "factor_panel_checkpoint")
        _write_factor_profile(output / "factor_profile.json", runner,
                              preparation_seconds=time.perf_counter()-panel_started)
        for path in raw_paths:
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            raw, _ = runner._load_factor_checkpoint(path, manifest["contract"]["factors"])
            runner.raw_ranks.update({n: f.rank(axis=1, pct=True) for n, f in raw.items()})
        runner.get_contract_schedule()
        runner.env = CausalEligibilityEnvironment(runner.cal, runner.daily_ret, runner.env.sector_of)
        gc.collect()
        pool_runner = runner.for_factors(sorted(directions), factor_directions=directions)
        cutoff, start = pd.Timestamp(expected["selection_end"]), pd.Timestamp(expected["performance_start"])
        segments = [(start, pd.Timestamp("2019-12-31")),
                    (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")),
                    (pd.Timestamp("2022-01-01"), pd.Timestamp("2023-12-31")),
                    (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
                    (pd.Timestamp("2025-01-01"), cutoff)]
        corr = _exposure_correlation({n: f.loc[start:cutoff] for n, f in pool_runner.ranks.items()}, sorted(directions))
        clusters = _cluster(corr, sorted(directions))
        _json_dump(output / "pool_clusters.json", {"selection_end": str(cutoff.date()),
                                                   "clusters": clusters, "permanent_exclusions": []})
        evaluator = PortfolioEvaluator(pool_runner, start=start, end=cutoff,
            cost_model=_configured_cost_model(config), ic_window=int(config.production_portfolio.ic_window),
            risk_lookback_calendar_days=int(config.production_portfolio.risk_lookback_calendar_days))
        _run_budgeted_pool_search(evaluator=evaluator, portfolio_ic=pool_runner.ic,
            pool=sorted(directions), recipe=recipe, segments=segments, clusters=clusters, db=db, output=output,
            budget=expected["budget"], beam_width=expected["beam_width"], round_width=expected["round_width"])
        selection = json.loads((output / "pool_selection.json").read_text(encoding="utf-8"))
        # Memberships are now frozen on disk before looking at observation NAVs.
        navs, plot_rows, comparisons = {}, [], []
        candidates = [(f"研究候选{i}（{r['factor_count']}因子）", {n: directions[n] for n in r["factors"]}, r)
                      for i, r in enumerate(selection["shortlist"], 1)]
        peers = [(s["name"], s["directions"], None) for s in old["seeds"]] + candidates
        for label, members, selected_row in peers:
            members = dict(sorted(members.items()))
            view = runner.for_factors(list(members), factor_directions=members)
            full = PortfolioEvaluator(view, start=start, end=expected["end"],
                cost_model=_configured_cost_model(config), ic_window=int(config.production_portfolio.ic_window),
                risk_lookback_calendar_days=int(config.production_portfolio.risk_lookback_calendar_days))
            ledger = full.ledger(list(members), recipe)
            if selected_row is not None:
                saved = db.execute("SELECT net_returns FROM pool_results WHERE signature=?",
                                   [json.dumps(tuple(sorted(members)))]).fetchone()[0]
                np.testing.assert_array_equal(ledger.loc[:cutoff, "net_return"].to_numpy(), saved)
            navs[label] = ledger["nav"] / ledger["nav"].iloc[0]
            metrics = performance_metrics(ledger["net_return"], initial_anchor=True)
            plot_rows.append({"strategy": label, **metrics, "volatility": metrics["annual_volatility"],
                              "annualized_turnover": float(ledger["executed_traded_notional"].iloc[1:].mean()*252)})
            comparisons.append({"strategy": label, "directions": members,
                "selection": performance_metrics(ledger.loc[:cutoff, "net_return"], initial_anchor=True),
                "simulated_live": performance_metrics(ledger.loc[ledger.index > cutoff, "net_return"]),
                "full": metrics})
            full.clear_transient_caches()
        pd.DataFrame(navs).to_parquet(output / "comparison_nav.parquet")
        _json_dump(output / "comparison_metrics.json", comparisons)
        _write_comparison_plot(output, pd.DataFrame(navs), plot_rows, cutoff=cutoff,
            title="全候选池预算搜索：冻结候选与现有8组合对比\n2026-05-15之后仅模拟实盘观察；未修改生产组合")
        lines = ["# 全候选池独立搜索报告", "",
            f"候选{len(directions)}个（正式库{len(effective)}，研究补充{len(extras)}）；实际尝试{selection['attempted']}次，预算{selection['budget']}次。",
            f"停止原因：{selection['stop_reason']}；不是全部子集穷举或全局最优证明。",
            "选择截止2026-05-15（含），当日T+1 IC标签不参与提案；随后至冻结最新日只作模拟实盘观察。",
            "本次条件于已有准入快照，未重新认证修补后的因子准入，也未修改有效库、默认配方或8策略。",
            "IC仅用于提案排序；真实回测及5段历史稳定性选择候选，历史段不是独立因子发现样本外。", "",
            "| 组合 | 选择期年化 | 选择期夏普 | 选择期回撤 | 模拟实盘年化 |", "|---|---:|---:|---:|---:|"]
        for row in comparisons:
            a, b = row["selection"], row["simulated_live"]
            lines.append(f"| {row['strategy']} | {a.get('annual_return',np.nan):.2%} | {a.get('sharpe',np.nan):.2f} | {a.get('max_drawdown',np.nan):.2%} | {b.get('annual_return',np.nan):.2%} |")
        (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
        return {"attempted": selection["attempted"], "shortlist": len(selection["shortlist"]),
                "stop_reason": selection["stop_reason"], "completed": True}


def incremental_trade_jobs(peers, candidates, clusters):
    """One-factor additions/same-cluster replacements and isolated B0/B1/B2."""
    jobs = []
    for buffer in (0, 1, 2):
        for peer in peers:
            jobs.append({"id": f"{peer['id']}__B{buffer}", "kind": "baseline" if not buffer else "buffer",
                         "seed": peer["id"], "name": peer["name"], "buffer": buffer,
                         "directions": peer["directions"], "added": "", "removed": ""})
    for peer in peers:
        for name, record in sorted(candidates.items()):
            if name in peer["directions"]:
                continue
            removals = [""] + [old for old in sorted(peer["directions"])
                               if clusters[old] == clusters[name]]
            for removed in removals:
                members = {n: d for n, d in peer["directions"].items() if n != removed}
                members[name] = int(record["direction"])
                label = hashlib.sha256(f"{name}|{removed}".encode()).hexdigest()[:16]
                jobs.append({"id": f"{peer['id']}__{label}", "kind": "replace" if removed else "add",
                             "seed": peer["id"], "name": peer["name"], "buffer": 0,
                             "directions": members, "added": name, "removed": removed})
    return jobs


def _trade_turnover_components(result):
    """Partition actual full traded notional; roll is the residual, not extra fees."""
    effective = result.effective_weights.to_numpy(dtype=float)
    returns = result.asset_returns.to_numpy(dtype=float)
    net = result.daily["net_return"].to_numpy(dtype=float)
    previous = np.zeros_like(effective)
    previous[1:] = effective[:-1] * (1 + returns[:-1]) / (1 + net[:-1, None])
    active_old, active_new = np.abs(previous) > 1e-12, np.abs(effective) > 1e-12
    delta = np.abs(effective - previous)
    entry_exit = (delta * (active_old != active_new)).sum(axis=1)
    reverse = (delta * (active_old & active_new & (previous * effective < 0))).sum(axis=1)
    resize = delta.sum(axis=1) - entry_exit - reverse
    roll = result.daily["executed_traded_notional"].to_numpy() - delta.sum(axis=1)
    if np.min(roll) < -1e-9:
        raise AssertionError("root-level turnover exceeds concrete-contract trading")
    frame = pd.DataFrame({"entry_exit": entry_exit, "reversal": reverse,
                          "resize": resize, "roll_extra": np.maximum(roll, 0)}, index=result.daily.index)
    np.testing.assert_allclose(frame.sum(axis=1), result.daily["executed_traded_notional"], atol=1e-9, rtol=0)
    return frame


def _incremental_trade_result(evaluator, job, recipe):
    from dataclasses import replace
    # A study job is an explicit run override of the default-off base recipe.
    # A positive already-resolved policy must agree, never silently lose state.
    if recipe.rank_exit_buffer and recipe.rank_exit_buffer != job["buffer"]:
        raise ValueError("conflicting rank_exit_buffer in resolved recipe and study job")
    result = evaluator.run(list(job["directions"]), replace(recipe, rank_exit_buffer=job["buffer"]))
    return result, result.metadata["rank_buffer_diagnostics"]


def _incremental_trade_metrics(result, cutoff, start, *, segments=None):
    from optimization.costs import evaluate_cost_sensitivity
    daily = result.daily
    components = _trade_turnover_components(result)
    intervals = {"selection": daily.loc[:cutoff].iloc[1:], "observation": daily.loc[daily.index > cutoff],
                 "full": daily.iloc[1:]}
    metrics = {}
    for label, frame in intervals.items():
        metrics[label] = {**performance_metrics(frame["net_return"]),
            "annualized_turnover": float(frame["executed_traded_notional"].mean() * 252),
            "gross_annual_return": performance_metrics(frame["gross_return"])["annual_return"],
            "annual_trade_fee": float(frame["trade_cost"].mean() * 252),
            "turnover_components": (components.loc[frame.index].mean() * 252).to_dict(),
            "cost_sensitivity": evaluate_cost_sensitivity(frame)}
    segments = (calendar_segments(pd.Timestamp(start), pd.Timestamp(cutoff)) if segments is None
                else [tuple(map(pd.Timestamp, pair)) for pair in segments])
    metrics["robustness"] = robust_summary(daily, segments, initial_anchor=True)
    metrics["segments"] = [{"start": str(left.date()), "end": str(right.date()),
        **performance_metrics(daily.loc[left:right, "net_return"], initial_anchor=left == pd.Timestamp(start))}
        for left, right in segments]
    return metrics


def run_incremental_trade_study(output: Path, reference: Path, admission_runs: Sequence[str],
                                *, max_jobs: int | None = None) -> dict:
    """Explicit, checkpointed factor-increment/buffer experiment; no default writes."""
    import duckdb
    import cProfile
    import pstats
    from core.config import load_strategy_library
    from core.registry import get as registry_get
    from run_portfolio_workflow import _peak_working_set_mib
    from research.artifacts import sha256_file
    from workflows.factor_selection import _cluster, _exposure_correlation, _production_recipe, _configured_cost_model

    output.mkdir(parents=True, exist_ok=True)
    reference = reference.resolve()
    previous = json.loads((reference / "contract.json").read_text(encoding="utf-8"))
    source = previous["source"]
    config = load_config("config/default.yaml")
    recipe = _production_recipe(config)
    if json.loads(json.dumps(recipe.to_dict())) != source["recipe"]:
        raise ValueError("reference construction recipe differs from current default")
    snapshot = _search_source_snapshot(source["panel_start"])
    if snapshot != {"end": source["end"], "source_fingerprint": source["source_fingerprint"]}:
        raise ValueError("refresh the fixed baseline first: current market snapshot differs")
    catalog = load_strategy_library("config/strategy_library.yaml")
    current = {row.id for row in catalog.strategies if row.status in {"preferred", "observing"}}
    peers = [p for p in previous["peers"] if p["id"] in current]
    if {p["id"] for p in peers} != current:
        raise ValueError("reference does not contain every current strategy")
    records = json.loads(Path(catalog.effective_factor_library).read_text(encoding="utf-8"))["factors"]
    library = {r["factor"]: r for r in records if r["status"] == "effective"}
    candidates = {n: r for n, r in library.items() if r["source_run"] in admission_runs}
    if not candidates or set(admission_runs) != {r["source_run"] for r in candidates.values()}:
        raise ValueError("each requested admitted batch must contain effective factors")
    directions = {n: int(r["direction"]) for n, r in candidates.items()}
    for peer in peers:
        for name, direction in peer["directions"].items():
            if name not in library or int(library[name]["direction"]) != direction:
                raise ValueError(f"baseline/library identity differs: {name}")
            directions[name] = direction
    names = sorted(directions)
    protected = [Path(p) for p in previous["input_sha256"]]
    for path in protected:
        if str(path).replace("\\", "/").startswith("config/") and sha256_file(path) != previous["input_sha256"][path.as_posix()]:
            raise ValueError(f"reference configuration changed: {path}")
    cache = reference / "factor_panel_checkpoint"
    old_manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    old_names = old_manifest["contract"]["factors"]
    if set(old_manifest["completed"]) != set(old_names) or set(old_names) | set(candidates) != set(names):
        raise ValueError("reference factor checkpoint membership differs")
    if old_manifest["contract"]["source_tree_sha256"] != previous["factor_source_sha256"]:
        raise ValueError("reference factor source fingerprint differs from its saved contract")
    inputs = protected + [reference / n for n in ("contract.json", "results.duckdb")]
    inputs += [cache / "manifest.json"] + [cache / f for f in old_manifest["completed"].values()]
    code = [ROOT / "research/portfolio_experiment_support.py", ROOT / "research/historical_portfolio_search.py", Path(__file__)]
    for folder in ("backtest", "optimization", "core", "data", "factors"):
        code += sorted((ROOT / folder).rglob("*.py"))
    history_days = {n: int(getattr(registry_get("factor", n), "warmup_trading_days", 0) or 0)
                    for n in candidates}
    history_days = {n: days for n, days in history_days.items() if days}
    history_start = str((pd.Timestamp(source["panel_start"]) - pd.Timedelta(days=2 * max(history_days.values(), default=0))).date())
    contract = {"schema_version": 1, "role": "factor_increment_and_trade_buffer_research_only",
        "reference": str(reference), "source": source, "peers": peers, "candidates": candidates,
        "admission_runs": list(admission_runs), "effective_library_count": len(library),
        "input_sha256": {str(p.resolve()): sha256_file(p) for p in inputs},
        "code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(set(code))},
        "runtime": {"ic_window": config.production_portfolio.ic_window,
                    "risk_lookback_calendar_days": config.production_portfolio.risk_lookback_calendar_days,
                    "ic_horizon": 1, "costs": _configured_cost_model(config).ledger_parameters()},
        "clustering": "complete linkage, absolute pooled daily rank correlation >= .50; pre-cutoff only",
        "jobs": "each baseline B0/B1/B2; all one-factor additions and all same-cluster one-for-one replacements",
        "buffers": [0, 1, 2], "selection_end_inclusive": source["selection_end"],
        "long_factor_history": {"trading_days": history_days, "calendar_read_start": history_start,
            **_search_source_snapshot(history_start, source["end"])},
        "shortlist_rule": "pre-cutoff net annual strictly improves, Sharpe and worst segment Sharpe do not deteriorate; rank by robustness_key then selection Sharpe; at most one per baseline, up to five",
        "observation": "strictly after cutoff; previously viewed, not blind OOS; never rank on it",
        "panel_reuse": "reference raw exposures are explicit hashed control inputs; old manifest never relabeled",
        "next_stages": "W/R and multi-factor/full-pool search are conditional follow-ups, not silently launched",
        "promotion": "none; historical segments are diagnostics, not completed independent portfolio admission"}
    path = output / "contract.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != contract:
        raise ValueError("incremental/trade study contract changed; do not resume with changed inputs")
    _json_dump(path, contract)
    if max_jobs == 0:
        return {"status": "frozen", "candidates": len(candidates), "baselines": len(peers)}
    wall, cpu = time.perf_counter(), time.process_time()
    print(f"factor preparation: reuse {len(old_names)} frozen panels; compute {len(candidates)} new factors", flush=True)
    runner = Runner(sorted(candidates), start=source["panel_start"], end=source["end"], factor_directions=directions,
                    ic_horizon=1, checkpoint_dir=output / "factor_panel_checkpoint")
    for name, filename in old_manifest["completed"].items():
        frame = pd.read_parquet(cache / filename)
        if not frame.index.equals(runner.cal) or list(frame.columns) != runner.u:
            raise ValueError(f"frozen control panel axes differ: {name}")
        runner.raw_ranks[name] = frame.rank(axis=1, pct=True)
    runner = runner.for_factors(names, factor_directions=directions)
    runner.get_contract_schedule()
    _write_factor_profile(output / "factor_profile.json", runner, first_finite_date={
        n: str(runner.ranks[n].dropna(how="all").index.min().date()) for n in names})
    runner.env = CausalEligibilityEnvironment(runner.cal, runner.daily_ret, runner.env.sector_of)
    gc.collect()
    cutoff = pd.Timestamp(source["selection_end"])
    corr = _exposure_correlation({n: runner.ranks[n].loc[source["performance_start"]:cutoff] for n in names}, names)
    clusters = _cluster(corr, names)
    _json_dump(output / "clusters.json", {"clusters": clusters, "names": names, "correlation": corr.to_numpy().tolist(),
        "input_bar_frequency": {n: getattr(registry_get("factor", n), "input_bar_frequency", "1min") for n in names}})
    jobs = incremental_trade_jobs(peers, candidates, clusters)
    _json_dump(output / "jobs.json", jobs)
    evaluator = PortfolioEvaluator(runner, start=source["performance_start"], end=source["end"],
        cost_model=_configured_cost_model(config), ic_window=contract["runtime"]["ic_window"],
        risk_lookback_calendar_days=contract["runtime"]["risk_lookback_calendar_days"])
    return _run_incremental_jobs(output, reference, evaluator, recipe, jobs, source,
                                 max_jobs=max_jobs, started=(wall, cpu))


def _run_incremental_jobs(output, reference, evaluator, recipe, jobs, source, *, max_jobs=None, started=None):
    """Shared native ledger execution and exact B0 controls for explicit studies."""
    import duckdb
    import cProfile
    import pstats
    from run_portfolio_workflow import _peak_working_set_mib
    runner = evaluator.runner
    cutoff = pd.Timestamp(source["selection_end"])
    wall, cpu = started or (time.perf_counter(), time.process_time())
    executed = 0
    with duckdb.connect(str(output / "results.duckdb")) as db, duckdb.connect(str(reference / "results.duckdb"), read_only=True) as old_db:
        db.execute("CREATE TABLE IF NOT EXISTS results (id VARCHAR PRIMARY KEY, payload VARCHAR)")
        done = {row[0] for row in db.execute("SELECT id FROM results").fetchall()}
        for job in jobs:
            if job["id"] in done:
                continue
            began, began_cpu = time.perf_counter(), time.process_time()
            profiler = cProfile.Profile() if executed == 0 else None
            print(f"backtest {len(done)+1}/{len(jobs)}: {job['id']} {job['kind']}", flush=True)
            payload = {**job, "status": "failed", "metrics": {}, "error": ""}
            try:
                if profiler: profiler.enable()
                result, diagnostics = _incremental_trade_result(evaluator, job, recipe)
                if profiler: profiler.disable()
                if job["kind"] == "baseline":
                    key, value = ("job", job["reference_job"]) if "reference_job" in job else ("strategy", job["name"])
                    expected = old_db.execute(f"SELECT * FROM daily WHERE {key}=? ORDER BY date", [value]).df().set_index("date")
                    np.testing.assert_array_equal(result.daily.index, expected.index)
                    for column in result.daily:
                        np.testing.assert_array_equal(result.daily[column].to_numpy(), expected[column].to_numpy())
                    tables = {row[0] for row in old_db.execute("SHOW TABLES").fetchall()}
                    assets = (old_db.execute(f"SELECT * FROM asset_daily WHERE {key}=?", [value]).df()
                              if "asset_daily" in tables else pd.DataFrame())
                    if not assets.empty:
                        for field, actual in (("effective_weight", result.effective_weights), ("asset_return", result.asset_returns), ("contribution", result.contributions)):
                            frozen = assets.pivot(index="date", columns="instrument", values=field).reindex(index=actual.index, columns=actual.columns)
                            np.testing.assert_array_equal(actual, frozen)
                        payload["asset_exact"] = True
                    payload["baseline_exact"] = True
                payload.update(status="completed", metrics=_incremental_trade_metrics(result, cutoff, source["performance_start"],
                    segments=source.get("selection_segments")), diagnostics=diagnostics,
                    risk_count_backend=result.metadata.get("risk_count_backend", "not_recorded"),
                    first_holding_date=str(result.daily.loc[result.daily.gross_exposure > 1e-12].index.min().date()))
                daily = result.daily.reset_index(names="date").assign(job=job["id"])
                db.register("completed_daily", daily)
                db.execute("CREATE TABLE IF NOT EXISTS daily AS SELECT * FROM completed_daily WHERE false")
                db.execute("BEGIN TRANSACTION")
                db.execute("INSERT INTO daily SELECT * FROM completed_daily")
                if job["kind"] in {"baseline", "buffer", "candidate"}:
                    detail = pd.DataFrame({"date": np.repeat(result.daily.index, len(runner.u)), "job": job["id"],
                        "instrument": np.tile(runner.u, len(result.daily)), "effective_weight": result.effective_weights.to_numpy().ravel(),
                        "asset_return": result.asset_returns.to_numpy().ravel(), "contribution": result.contributions.to_numpy().ravel(),
                        "target_weight": result.target_weights.to_numpy().ravel()})
                    db.register("completed_assets", detail)
                    db.execute("CREATE TABLE IF NOT EXISTS asset_daily AS SELECT * FROM completed_assets WHERE false")
                    db.execute("INSERT INTO asset_daily SELECT * FROM completed_assets")
            except (RuntimeError, ValueError) as exc:
                if payload["status"] == "completed":
                    raise  # Artifact/database failures must not masquerade as rejected portfolios.
                payload["error"] = str(exc)
                if job["kind"] == "baseline":
                    raise  # Never proceed after a baseline regression.
                db.execute("BEGIN TRANSACTION")
            finally:
                if profiler:
                    profiler.disable()
                    payload["hotspots"] = [{"file": key[0], "line": key[1], "function": key[2], "calls": val[1], "seconds": val[3]}
                        for key, val in sorted(pstats.Stats(profiler).stats.items(), key=lambda item: -item[1][3])[:15]]
            payload.update(wall_seconds=time.perf_counter()-began, cpu_seconds=time.process_time()-began_cpu,
                           peak_working_set_mib=_peak_working_set_mib())
            db.execute("INSERT INTO results VALUES (?, ?)", [job["id"], json.dumps(payload, ensure_ascii=False)])
            db.execute("COMMIT")
            done.add(job["id"]); executed += 1
            evaluator._score_cache.clear(); evaluator._ledger_cache.clear()
            if len(evaluator._asset_weight_cache) > 50000:
                evaluator._asset_weight_cache.clear()
            _json_dump(output / "progress.json", {"completed": len(done), "planned": len(jobs), "last_job": job["id"],
                "status": payload["status"], "wall_seconds": time.perf_counter()-wall, "cpu_seconds": time.process_time()-cpu,
                "peak_working_set_mib": _peak_working_set_mib(), "defaults_unchanged": True})
            if max_jobs is not None and executed >= max_jobs:
                break
    return {"completed": len(done), "planned": len(jobs), "finished": len(done) == len(jobs)}


def replay_completed_trade_study(output: Path, parent: Path, *, baselines_only=False, max_jobs=None) -> dict:
    """Engineering parity on reviewed native jobs, with separately frozen new code.

    Raw exposures are hashed control inputs, not relabeled factor checkpoints.
    This does not select members, refresh data or overwrite the parent study.
    """
    import duckdb
    from research.artifacts import sha256_file
    from factors.numerics import factor_kernel_contract
    from workflows.factor_selection import _production_recipe, _configured_cost_model
    parent, output = parent.resolve(), output.resolve()
    if output == parent or parent in output.parents:
        raise ValueError("engineering replay must use a separate sibling output")
    previous = json.loads((parent / "contract.json").read_text(encoding="utf-8"))
    accepted = json.loads((parent / "acceptance.json").read_text(encoding="utf-8"))
    if (not accepted["review_complete"]
            or sha256_file(parent / "contract.json") != accepted["computation_contract_sha256"]
            or sha256_file(parent / "results.duckdb") != accepted["results_sha256"]):
        raise ValueError("parent evidence is not sealed or has changed")
    with duckdb.connect(str(parent / "results.duckdb"), read_only=True) as db:
        rows = [json.loads(row[0]) for row in db.execute("SELECT payload FROM results ORDER BY rowid").fetchall()]
    rows = [row for row in rows if not baselines_only or row["kind"] == "baseline"]
    if not rows or any(row["status"] != "completed" for row in rows):
        raise ValueError("replay requires a nonempty completed control set")
    jobs = [{**{key: row[key] for key in ("id", "seed", "name", "buffer", "directions", "added", "removed")},
             "kind": "baseline", "reference_job": row["id"]} for row in rows]
    source = dict(previous["source"])
    if "selection_segments" in previous:
        source["selection_segments"] = previous["selection_segments"]
    if _search_source_snapshot(source["panel_start"], source["end"]) != {
            "end": source["end"], "source_fingerprint": source["source_fingerprint"]}:
        raise ValueError("replay cannot mix market releases")
    config = load_config("config/default.yaml")
    recipe = _production_recipe(config)
    old_recipe = dict(source["recipe"])
    old_recipe.setdefault("rank_exit_buffer", 0)
    if (json.loads(json.dumps(recipe.to_dict())) != old_recipe
            or _configured_cost_model(config).ledger_parameters() != previous["runtime"]["costs"]
            or config.production_portfolio.ic_window != previous["runtime"]["ic_window"]
            or config.production_portfolio.risk_lookback_calendar_days != previous["runtime"]["risk_lookback_calendar_days"]):
        raise ValueError("replay recipe/runtime differs from the frozen controls")
    directions = {}
    for job in jobs:
        for name, sign in job["directions"].items():
            if name in directions and directions[name] != sign:
                raise ValueError("control factor directions conflict")
            directions[name] = sign
    panels = {}
    inputs = previous["input_sha256"]
    for filename, digest in inputs.items():
        path = Path(filename)
        if path.name != "manifest.json" or path.parent.name != "factor_panel_checkpoint":
            continue
        if sha256_file(path) != digest:
            raise ValueError(f"control panel manifest changed: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for name, relative in manifest["completed"].items():
            if name in directions:
                panel = path.parent / relative
                if sha256_file(panel) != inputs.get(str(panel)):
                    raise ValueError(f"control panel is missing its frozen hash or changed: {panel}")
                panels[name] = panel
    if set(panels) != set(directions):
        raise ValueError("control input panels do not cover replay members")
    protected = [parent / name for name in ("contract.json", "acceptance.json", "results.duckdb")]
    protected += list(panels.values()) + [ROOT / name for name in (
        "config/default.yaml", "config/local.yaml", "config/strategy_library.yaml", "factor_library/library.json")]
    code = [ROOT / name for name in previous["code_sha256"]]
    code += [ROOT / "native/mf_factor_kernels/src/lib.rs", ROOT / "strategies/combined.py", ROOT / "run_portfolio_workflow.py"]
    contract = {"role": "native_execution_engineering_parity", "source": source,
                "parent": str(parent), "jobs": jobs, "runtime": previous["runtime"],
                "backend": factor_kernel_contract(), "initial_holdings": "flat",
                "input_sha256": {str(path): sha256_file(path) for path in protected},
                "code_sha256": {str(path): sha256_file(path) for path in sorted(set(code))},
                "final_target": "now recorded, not traded; prior daily/asset accounting must be exact",
                "promotion": "none"}
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / "contract.json"
    if frozen.exists() and json.loads(frozen.read_text(encoding="utf-8")) != contract:
        raise ValueError("engineering replay contract changed; cannot resume")
    _json_dump(frozen, contract)
    if max_jobs == 0:
        return {"planned": len(jobs), "status": "frozen"}
    began = time.perf_counter(), time.process_time()
    runner = Runner([], start=source["panel_start"], end=source["end"], ic_horizon=previous["runtime"]["ic_horizon"])
    for name, path in panels.items():
        frame = pd.read_parquet(path)
        if not frame.index.equals(runner.cal) or list(frame.columns) != runner.u:
            raise ValueError(f"control panel axes differ: {name}")
        runner.raw_ranks[name] = frame.rank(axis=1, pct=True)
    runner = runner.for_factors(sorted(directions), factor_directions=directions)
    runner.get_contract_schedule()
    evaluator = PortfolioEvaluator(runner, start=source["performance_start"], end=source["end"],
        cost_model=_configured_cost_model(config), ic_window=previous["runtime"]["ic_window"],
        risk_lookback_calendar_days=previous["runtime"]["risk_lookback_calendar_days"])
    result = _run_incremental_jobs(output, parent, evaluator, recipe, jobs, source,
                                   max_jobs=max_jobs, started=began)
    result.update(wall_seconds=time.perf_counter()-began[0], cpu_seconds=time.process_time()-began[1])
    _json_dump(output / "performance.json", result)
    return result


def run_trade_interactions(output: Path, parent: Path, *, max_jobs=None) -> dict:
    """Freeze the existing shortlist, replay its B0 ledgers, then isolate B1/B2."""
    from research.artifacts import sha256_file
    from workflows.factor_selection import _production_recipe, _configured_cost_model
    parent = parent.resolve()
    previous = json.loads((parent / "contract.json").read_text(encoding="utf-8"))
    accepted = json.loads((parent / "acceptance.json").read_text(encoding="utf-8"))
    selected = json.loads((parent / "shortlist.json").read_text(encoding="utf-8"))
    if not accepted["review_complete"] or not selected["finished"] or not selected["shortlist"]:
        raise ValueError("interaction requires a reviewed, completed, nonempty frozen shortlist")
    if (sha256_file(parent / "results.duckdb") != accepted["results_sha256"]
            or sha256_file(parent / "shortlist.json") != accepted["artifact_sha256"]["shortlist.json"]
            or sha256_file(parent / "contract.json") != accepted["computation_contract_sha256"]):
        raise ValueError("parent evidence changed after acceptance")
    # Only this explicit workflow's reporting/execution plumbing has evolved.
    for name, digest in {**previous["input_sha256"], **previous["code_sha256"]}.items():
        if Path(name).resolve() != Path(__file__).resolve() and sha256_file(Path(name)) != digest:
            raise ValueError(f"parent frozen input or compute code changed: {name}")
    source = previous["source"]
    if _search_source_snapshot(source["panel_start"]) != {
            "end": source["end"], "source_fingerprint": source["source_fingerprint"]}:
        raise ValueError("market snapshot changed; interaction cannot mix releases")
    history = previous["long_factor_history"]
    if _search_source_snapshot(history["calendar_read_start"], source["end"]) != {
            "end": history["end"], "source_fingerprint": history["source_fingerprint"]}:
        raise ValueError("extra warmup market snapshot changed")
    config = load_config("config/default.yaml")
    recipe = _production_recipe(config)
    if json.loads(json.dumps(recipe.to_dict())) != source["recipe"]:
        raise ValueError("interaction recipe differs from frozen parent")
    peers = [{"id": row["seed"], "name": row["name"].rsplit("(", 1)[0] + f"({len(row['directions'])})",
              "directions": row["directions"], "reference_job": row["id"]} for row in selected["shortlist"]]
    names = sorted({name for peer in peers for name in peer["directions"]})
    directions = {name: direction for peer in peers for name, direction in peer["directions"].items()}
    used_additions = {row["added"] for row in selected["shortlist"]}
    candidates = {n: r for n, r in previous["candidates"].items() if n in used_additions}
    caches = [Path(previous["reference"]) / "factor_panel_checkpoint", parent / "factor_panel_checkpoint"]
    panels, inputs = {}, [Path(p) for p in previous["input_sha256"]]
    inputs += [parent / f for f in ("contract.json", "results.duckdb", "shortlist.json", "acceptance.json", "clusters.json")]
    for cache in caches:
        manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
        inputs.append(cache / "manifest.json")
        for name, filename in manifest["completed"].items():
            if name in names:
                panels[name] = cache / filename
                inputs.append(cache / filename)
    if set(panels) != set(names):
        raise ValueError("reviewed parent panels do not cover frozen candidate members")
    contract = {**previous, "role": "fixed_shortlist_trade_interaction_research_only",
        "reference": str(parent), "interaction_parent": str(parent), "peers": peers, "candidates": candidates,
        "input_sha256": {str(p.resolve()): sha256_file(p) for p in inputs},
        "code_sha256": {name: sha256_file(Path(name)) for name in previous["code_sha256"]},
        "jobs": "five frozen shortlist B0 exact ledger controls, then B1/B2 only; no new membership search",
        "shortlist_rule": "frozen parent shortlist; interaction is diagnostic and never auto-promoted",
        "panel_reuse": "read-only hashed parent panels; no factor recomputation or checkpoint relabeling"}
    jobs = [{"id": f"{peer['id']}__B{buffer}", "kind": "baseline" if not buffer else "buffer",
             "seed": peer["id"], "name": peer["name"], "buffer": buffer, "directions": peer["directions"],
             "added": "", "removed": "", "reference_job": peer["reference_job"]}
            for buffer in (0, 1, 2) for peer in peers]
    output.mkdir(parents=True, exist_ok=True)
    path = output / "contract.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != contract:
        raise ValueError("interaction contract changed; do not mix results")
    _json_dump(path, contract); _json_dump(output / "jobs.json", jobs)
    _json_dump(output / "clusters.json", json.loads((parent / "clusters.json").read_text(encoding="utf-8")))
    if max_jobs == 0:
        return {"status": "frozen", "planned": len(jobs), "factor_count": len(names)}
    began = (time.perf_counter(), time.process_time())
    runner = Runner([], start=source["panel_start"], end=source["end"], ic_horizon=1)
    for name, path in panels.items():
        frame = pd.read_parquet(path)
        if not frame.index.equals(runner.cal) or list(frame.columns) != runner.u:
            raise ValueError(f"interaction frozen panel axes differ: {name}")
        runner.raw_ranks[name] = frame.rank(axis=1, pct=True)
    runner = runner.for_factors(names, factor_directions=directions)
    runner.checkpoint_loaded_factor_count = len(panels)
    _write_factor_profile(output / "factor_profile.json", runner, read_only_parent_panels=True)
    runner.get_contract_schedule()
    runner.env = CausalEligibilityEnvironment(runner.cal, runner.daily_ret, runner.env.sector_of)
    evaluator = PortfolioEvaluator(runner, start=source["performance_start"], end=source["end"],
        cost_model=_configured_cost_model(config), ic_window=contract["runtime"]["ic_window"],
        risk_lookback_calendar_days=contract["runtime"]["risk_lookback_calendar_days"])
    return _run_incremental_jobs(output, parent, evaluator, recipe, jobs, source,
                                 max_jobs=max_jobs, started=began)


def _plot_incremental_trade_diagnostics(output, contract, successful, nav):
    """Saved-ledger comparisons only; no recomputation or candidate selection."""
    from monitoring.weekly_report import _mk_png
    from monitoring.plot_style import signed_heatmap_cmap
    from core.registry import get as registry_get
    plt = _mk_png()
    peers = contract["peers"]
    lookup = {(r["seed"], r["buffer"]): r for r in successful if r["kind"] in {"baseline", "buffer"}}
    changes = (
        ("selection", "annual_return", 100., "净年化变化（百分点）"),
        ("selection", "sharpe", 1., "夏普变化"),
        ("selection", "max_drawdown", 100., "回撤改善（百分点）"),
        ("selection", "annualized_turnover", -1., "年换手节省（倍）"),
        ("robustness", "worst_sharpe", 1., "最差分段夏普变化"),
        ("observation", "total_return", 100., "截止后累计收益变化（百分点，仅观察）"),
    )
    def heatmap(ax, values, xlabels, ylabels, title, fig):
        finite = np.abs(values[np.isfinite(values)])
        bound = max(float(finite.max()) if finite.size else 0., 1e-9)
        im = ax.imshow(values, cmap=signed_heatmap_cmap(), vmin=-bound, vmax=bound, aspect="auto")
        ax.set_xticks(range(len(xlabels)), xlabels)
        ax.set_yticks(range(len(ylabels)), ylabels, fontsize=9)
        ax.set_title(title, fontsize=11); ax.grid(False)
        for i, j in np.ndindex(values.shape):
            ax.text(j, i, f"{values[i,j]:+.2f}" if np.isfinite(values[i,j]) else "—",
                    ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=.035, pad=.03)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, (period, metric, scale, title) in zip(axes.ravel(), changes):
        values = np.full((len(peers), 2), np.nan)
        for i, peer in enumerate(peers):
            for j, buffer in enumerate((1, 2)):
                if (peer["id"], buffer) in lookup and (peer["id"], 0) in lookup:
                    values[i, j] = scale * (lookup[peer["id"], buffer]["metrics"][period][metric]
                                           - lookup[peer["id"], 0]["metrics"][period][metric])
        heatmap(ax, values, ["B1", "B2"], [p["name"] for p in peers], title, fig)
    fig.suptitle(f"{len(peers)}组进出缓冲：相对各自B0的变化\n红色为正、绿色为负，零为白；各图独立色标，截止后不参与选择", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(output / "buffer_effects.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots((len(peers)+1)//2, 2, figsize=(16, 2.7*((len(peers)+1)//2)), squeeze=False)
    cutoff = pd.Timestamp(contract["source"]["selection_end"])
    for ax, peer in zip(axes.ravel(), peers):
        for buffer, color in enumerate(("#737D87", "#6575A2", "#B37758")):
            label = peer["name"] + f"·B{buffer}"
            if label in nav:
                ax.plot(nav[label].index, nav[label], label=f"B{buffer}", color=color, linewidth=1.4)
        ax.axvline(cutoff, color="#555555", linestyle=":", linewidth=1)
        ax.axvspan(cutoff, pd.Timestamp(contract["source"]["end"]), color="#e8f1fc", alpha=.5)
        ax.set_title(peer["name"], fontsize=11); ax.set_ylabel("归一化净值")
        if any(peer["name"] + f"·B{buffer}" in nav for buffer in (0, 1, 2)):
            ax.legend(ncol=3, fontsize=8)
    for ax in axes.ravel()[len(peers):]: ax.set_visible(False)
    fig.suptitle(f"{len(peers)}组固定成员：B0 / B1 / B2逐组净值\n虚线为2026-05-15；其后仅模拟实盘观察", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(output / "buffer_nav.png", dpi=160); plt.close(fig)

    if contract.get("interaction_parent"):
        return  # No membership additions are searched again in an interaction study.
    names = sorted(contract["candidates"])
    added = {(r["seed"], r["added"]): r for r in successful if r["kind"] == "add"}
    values = np.full((len(names), len(peers)), np.nan)
    for i, name in enumerate(names):
        for j, peer in enumerate(peers):
            if (peer["id"], name) in added and (peer["id"], 0) in lookup:
                values[i, j] = 100 * (added[peer["id"], name]["metrics"]["selection"]["annual_return"]
                                     - lookup[peer["id"], 0]["metrics"]["selection"]["annual_return"])
    labels = [getattr(registry_get("factor", n), "description", n) for n in names]
    fig, ax = plt.subplots(figsize=(17, max(7, .38*len(names))))
    heatmap(ax, values, [p["name"] for p in peers], labels,
            f"{len(names)}因子逐一加入{len(peers)}组：截止前净年化变化（百分点）\n仅净年化，不代表同时满足稳定性规则；替换结果另见完整表", fig)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout(); fig.savefig(output / "factor_increment_effects.png", dpi=160); plt.close(fig)


def _trade_interaction_report_lines(output, contract, successful):
    """Four-corner comparison against immutable parent controls; descriptive only."""
    import duckdb
    from monitoring.weekly_report import _mk_png
    parent = Path(contract["interaction_parent"])
    with duckdb.connect(str(parent / "results.duckdb"), read_only=True) as db:
        prior = [json.loads(row[0]) for row in db.execute("SELECT payload FROM results").fetchall()]
    controls = {(r["seed"], r["buffer"]): r for r in prior if r["kind"] in {"baseline", "buffer"}}
    lookup = {(r["seed"], r["buffer"]): r for r in successful}
    lines = ["", "## 如何读本次交互实验", "",
        "本轮只固定上一轮5项候选，不重新选因子。每项先精确复现因子增量后的B0逐日账本，再只改变B1/B2；原成员的B0/B1/B2直接读取上一轮已验收结果。原候选增加实验未保存资产矩阵，因此本轮5个B0只声称逐日账本精确一致，不声称与不存在的旧资产矩阵对账。", "",
        "| 状态 | 成员 | 执行规则 | 回答的问题 |", "|---|---|---|---|",
        "| 原组合B0 | 原成员 | 原排名选池 | 固定起点 |",
        "| 因子增量B0 | 冻结增量成员 | 原排名选池 | 单独加因子的效果 |",
        "| 原组合B1/B2 | 原成员 | 缓冲选池 | 单独缓冲的效果 |",
        "| 因子增量B1/B2 | 同一增量成员 | 同一缓冲选池 | 两者一起是否仍有效 |", "",
        "### 规则与时序：缓冲到底改变了什么", "",
        "日频暴露 → 原合成排名 → B0/B1/B2选池 → 原ERC目标权重 → 原账本的可交易约束/换月/费用 → 次日生效权重与净收益。实际旧持仓来自当日收盘估值后的漂移权重，不是昨日理想目标；反馈只在显式缓冲研究中启用。信号频率、因子方向、best_period、合成、风险约束和费用均不变。", "",
        "B0直接取cap筛选后的每侧前10名。B1/B2分别先生成cap筛选后的11/12名保留集合，优先保留其中同方向实际旧持仓，再按原排名补足10名。不是把多空数量扩大到11/12，也不是机械延长持有1/2天，更不是每天强行交易零次。多空不重叠、cap与可交易冻结仍优先。", "",
        "例子仅作规则示意：假设某一侧cap筛选后依次为A1…A12，旧持仓为A1…A9和A11。B0卖A11换A10；B1/B2可继续持有A11，避免一次退出和进入。若A11跌出相应保留集合、不可保留或方向变化，仍按规则处理。真实缓冲也会改变ERC权重及换月路径，因此结果必须重新过账本，不能仅调整换手统计。", "",
        "### 四个状态的实际收益对比（截止前）", "",
        "下表‘交互差’=两者合用增益−单独加因子增益−单独缓冲增益，均相对原组合B0。它是几何年化的描述性二阶差，不是可加的因子收益归因或显著性检验。", "",
        "| 冻结增量组合 | 缓冲 | 单独因子增益(pp) | 单独缓冲增益(pp) | 合用增益(pp) | 交互差(pp) | 合用净年化 | 合用夏普 | 合用回撤 | 合用年换手 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for peer in contract["peers"]:
        seed = peer["id"]
        if (seed, 0) not in lookup: continue
        origin = controls[seed, 0]["metrics"]["selection"]
        increment = lookup[seed, 0]["metrics"]["selection"]
        for buffer in (1, 2):
            if (seed, buffer) not in lookup: continue
            m = lookup[seed, buffer]["metrics"]["selection"]
            factor_delta = increment["annual_return"]-origin["annual_return"]
            buffer_delta = controls[seed, buffer]["metrics"]["selection"]["annual_return"]-origin["annual_return"]
            both_delta = m["annual_return"]-origin["annual_return"]
            lines.append(f"| {peer['name']} | B{buffer} | {100*factor_delta:+.3f} | {100*buffer_delta:+.3f} | {100*both_delta:+.3f} | {100*(both_delta-factor_delta-buffer_delta):+.3f} | {m['annual_return']:.2%} | {m['sharpe']:.3f} | {m['max_drawdown']:.2%} | {m['annualized_turnover']:.2f} |")
    lines += ["", "## 原10组：换手究竟省在哪里", "",
        "下表和两张配图只使用上一轮原10组的控制账本，未重跑、未覆盖原结果。各年换手是完整买卖开平成交量，除以收益基准，按252交易日年化；不除以2，不再次乘杠杆。2bp完整成交费率×年换手为年化算术成交费用。毛收益损失可能抵消省费，因此同时看净收益、回撤与阶段稳定性。", "",
        "| 原策略 | 缓冲 | 进出 | 反向 | 同向调权 | 换月额外 | 合计 | 2bp年化成交费 | 毛年化 | 净年化 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    original_peers = json.loads((parent / "contract.json").read_text(encoding="utf-8"))["peers"]
    for peer in original_peers:
        for buffer in (0, 1, 2):
            m = controls[peer["id"], buffer]["metrics"]["selection"]
            parts = m["turnover_components"]
            lines.append(f"| {peer['name']} | B{buffer} | {parts['entry_exit']:.2f} | {parts['reversal']:.2f} | {parts['resize']:.2f} | {parts['roll_extra']:.2f} | {m['annualized_turnover']:.2f} | {m['annual_trade_fee']:.2%} | {m['gross_annual_return']:.2%} | {m['annual_return']:.2%} |")
    plt = _mk_png()
    colors = ("#B97870", "#A795B0", "#718FA0", "#A99A6A")
    figures = [("original_turnover_sources.png", "原10组：截止前年换手来源（完整成交量）"),
               ("original_cost_curves.png", "原10组：截止前固定路径交易费率压力（持有费用不变）")]
    for filename, title in figures:
        fig, axes = plt.subplots((len(original_peers)+1)//2, 2, figsize=(15, 2.6*((len(original_peers)+1)//2)), squeeze=False)
        for ax, peer in zip(axes.ravel(), original_peers):
            metrics = [controls[peer["id"], b]["metrics"]["selection"] for b in (0, 1, 2)]
            if "sources" in filename:
                bottom = np.zeros(3)
                for key, label, color in zip(("entry_exit", "reversal", "resize", "roll_extra"), ("进出", "反向", "同向调权", "换月额外"), colors):
                    values = np.array([m["turnover_components"][key] for m in metrics])
                    ax.bar(["B0", "B1", "B2"], values, bottom=bottom, color=color, label=label); bottom += values
                for i, value in enumerate(bottom): ax.text(i, value+2, f"{value:.1f}", ha="center", fontsize=8)
                ax.set_ylim(0, max(bottom)*1.18); ax.set_ylabel("年换手（倍）")
            else:
                for b, m in enumerate(metrics):
                    scenarios = m["cost_sensitivity"]["scenarios"]
                    ax.plot([s["trade_cost_rate"]*10000 for s in scenarios], [s["metrics"]["annual_return"]*100 for s in scenarios],
                            label=f"B{b}", color=("#737D87", "#6575A2", "#B37758")[b], marker=".")
                ax.axhline(0, color="#777777", linewidth=.7); ax.axvline(2, color="#777777", linestyle=":", linewidth=.7)
                ax.set_xlabel("完整成交费率（bp）"); ax.set_ylabel("净年化（%）")
            ax.set_title(peer["name"], fontsize=10); ax.legend(ncol=4, fontsize=7)
        for ax in axes.ravel()[len(original_peers):]: ax.set_visible(False)
        fig.suptitle(title, fontsize=14); fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(output / filename, dpi=160); plt.close(fig)
    lines += ["", "![原10组换手来源](original_turnover_sources.png)", "", "![原10组成本压力](original_cost_curves.png)", "",
        "### 如何验收，而不是只看一条收益数字", "",
        "分别比较：①净收益与毛收益是否兼顾；②换手下降来自进出还是同向调权；③回撤、最差历史段、中位历史段是否恶化；④4/6/8bp下能否承受；⑤收益是否过度集中于少数年份或品种。最后30日短段和截止后的84日观察须单独标示；后者已经查看，不参与选参、不冒充新盲样本外。成本曲线固定持仓路径，不包含真实订单规模、盘口冲击或容量保证。", "",
        "新增因子仍依原准入标准，不把换手加入因子门槛。所有交互结果属于研究诊断，不自动登记或更换生产组合。", ""]
    return lines


def report_incremental_trade_study(output: Path) -> dict:
    """Report all measured jobs; never choose using the post-cutoff observations."""
    import duckdb
    from run_portfolio_workflow import _write_comparison_plot
    contract = json.loads((output / "contract.json").read_text(encoding="utf-8"))
    interaction = bool(contract.get("interaction_parent"))
    planned = len(json.loads((output / "jobs.json").read_text(encoding="utf-8")))
    with duckdb.connect(str(output / "results.duckdb"), read_only=True) as db:
        results = [json.loads(row[0]) for row in db.execute("SELECT payload FROM results ORDER BY id").fetchall()]
        baselines = {r["seed"]: r for r in results if r["kind"] == "baseline" and r["status"] == "completed"}
        successful = [r for r in results if r["status"] == "completed"]
        additions = [r for r in successful if r["kind"] in {"add", "replace"}]
        finished = len(results) == planned
        # Transparent per-baseline shortlist; no forced winner or admission claim.
        improved = []
        for row in additions if finished else []:
            base = baselines[row["seed"]]["metrics"]
            m = row["metrics"]
            if (m["selection"]["annual_return"] > base["selection"]["annual_return"]
                    and m["selection"]["sharpe"] >= base["selection"]["sharpe"]
                    and m["robustness"]["worst_sharpe"] >= base["robustness"]["worst_sharpe"]):
                improved.append(row)
        improved.sort(key=lambda r: (robustness_key(r["metrics"]["robustness"]), r["metrics"]["selection"]["sharpe"]), reverse=True)
        shortlist, seen = [], set()
        for row in improved:
            if row["seed"] not in seen:
                shortlist.append(row); seen.add(row["seed"])
            if len(shortlist) == 5: break
        plotted = [r for r in successful if r["kind"] in {"baseline", "buffer"}] + shortlist
        nav, plot_rows, export = {}, [], []
        for row in successful:
            m, base = row["metrics"], baselines[row["seed"]]["metrics"]
            export.append({"job": row["id"], "strategy": row["name"], "kind": row["kind"], "buffer": row["buffer"],
                "added": row["added"], "removed": row["removed"], "factor_count": len(row["directions"]),
                **{f"{period}_{k}": m[period][k] for period in ("selection", "observation", "full")
                   for k in ("annual_return", "sharpe", "max_drawdown", "annualized_turnover", "gross_annual_return", "total_return")},
                "delta_net_annual": m["selection"]["annual_return"]-base["selection"]["annual_return"],
                "delta_turnover": m["selection"]["annualized_turnover"]-base["selection"]["annualized_turnover"],
                "delta_worst_sharpe": m["robustness"]["worst_sharpe"]-base["robustness"]["worst_sharpe"],
                "first_holding_date": row["first_holding_date"],
                "observation_return": m["observation"]["total_return"], "wall_seconds": row["wall_seconds"]})
        for row in plotted:
            label = (row["name"] + f"·B{row['buffer']}" if row["kind"] in {"baseline", "buffer"} else
                     row["name"].rsplit("(", 1)[0] + f"({len(row['directions'])})·增量{shortlist.index(row)+1}")
            frame = db.execute("SELECT date,nav_after FROM daily WHERE job=? ORDER BY date", [row["id"]]).df().set_index("date")
            nav[label] = frame.nav_after / frame.nav_after.iloc[0]
            m = row["metrics"]["full"]
            plot_rows.append({"strategy": label, **m, "volatility": m["annual_volatility"]})
    pd.DataFrame(export).to_csv(output / "comparison.csv", index=False, encoding="utf-8-sig")
    if not interaction:
        _json_dump(output / "shortlist.json", {"rule": contract["shortlist_rule"], "finished": finished,
            "qualifying_count": len(improved), "shortlist": shortlist, "promoted": False})
    _write_comparison_plot(output, pd.DataFrame(nav), plot_rows, cutoff=pd.Timestamp(contract["source"]["selection_end"]),
        title=("冻结5项因子增量候选：B0/B1/B2交互对照" if interaction else "因子增量与进出缓冲：冻结原10组与单变量对照")
              + "\n截止后仅模拟实盘观察；未变更默认策略")
    _plot_incremental_trade_diagnostics(output, contract, successful, nav)
    lines = ["# 冻结候选交易缓冲交互验证" if interaction else "# 新增有效因子与交易缓冲联合验证", "", "正式因子准入和10组策略均未更改。B1/B2仅放宽已有实际持仓的排名保留边界1/2名，目标仍每侧10个、cap3、ERC；真实冻结仓位不虚构平仓。", "",
        f"区间{contract['source']['performance_start']}至{contract['source']['end']}，额外预热从{contract['source']['panel_start']}开始；统计选择止于含端点2026-05-15，之后只观察，不能作为新盲OOS。", "",
        f"计划{planned}项，已完成{len(results)}项，其中成功{len(successful)}项、失败{len(results)-len(successful)}项。状态：{'全部完成' if finished else '部分完成，暂不生成最终候选结论'}。完整结果见comparison.csv、results.duckdb；成员和诊断簇见jobs.json、clusters.json。", "",
        "## 交易层单变量对照（截止前）", "", "| 策略 | 缓冲 | 净年化 | 夏普 | 最大回撤 | 年换手 | 净年化变化 | 换手变化 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in export:
        if row["kind"] not in {"baseline", "buffer"}: continue
        lines.append(f"| {row['strategy']} | {row['buffer']} | {row['selection_annual_return']:.2%} | {row['selection_sharpe']:.3f} | {row['selection_max_drawdown']:.2%} | {row['selection_annualized_turnover']:.2f} | {row['delta_net_annual']:+.2%} | {row['delta_turnover']:+.2f} |")
    lines += ["", "### 本轮各组均值与成本抗压", "", "以下是逐策略指标的算术平均，不是同时交易各组的合并账户。4/8bp为固定交易路径压力测试，持有费用保持不变，不代表真实冲击成本或容量验证。", "",
        "| 缓冲 | 平均净年化 | 平均夏普 | 平均年换手 | 净年化提高组数 | 8bp仍盈利组数 |", "|---|---:|---:|---:|---:|---:|"]
    for buffer in (0, 1, 2):
        group = [r for r in successful if r["kind"] in {"baseline", "buffer"} and r["buffer"] == buffer]
        if not group: continue
        metrics = [r["metrics"]["selection"] for r in group]
        better = sum(r["metrics"]["selection"]["annual_return"] > baselines[r["seed"]]["metrics"]["selection"]["annual_return"] for r in group)
        positive = sum(s["metrics"]["annual_return"] > 0 for m in metrics for s in m["cost_sensitivity"]["scenarios"] if np.isclose(s["trade_cost_rate"], .0008))
        lines.append(f"| B{buffer} | {np.mean([m['annual_return'] for m in metrics]):.2%} | {np.mean([m['sharpe'] for m in metrics]):.3f} | {np.mean([m['annualized_turnover'] for m in metrics]):.2f} | {better}/{len(group)} | {positive}/{len(group)} |")
    lines += ["", "| 策略/缓冲 | 4bp净年化 | 8bp净年化 | 静态盈亏平衡费率(bp) | 截止后累计净收益 | 全期净年化 |", "|---|---:|---:|---:|---:|---:|"]
    for row in plotted:
        sensitivity = row["metrics"]["selection"]["cost_sensitivity"]
        costs = {round(s["trade_cost_rate"]*10000): s["metrics"]["annual_return"] for s in sensitivity["scenarios"]}
        label = row["name"] + f"·B{row['buffer']}" if row["kind"] in {"baseline", "buffer"} else f"增量{shortlist.index(row)+1}"
        lines.append(f"| {label} | {costs[4]:.2%} | {costs[8]:.2%} | {sensitivity['breakeven_trade_cost_rate']*10000:.3f} | {row['metrics']['observation']['total_return']:.2%} | {row['metrics']['full']['annual_return']:.2%} |")
    if interaction:
        lines.extend(_trade_interaction_report_lines(output, contract, successful))
    else:
        lines += ["", "## 因子增量候选", "", f"按事前比较规则有{len(improved)}项净年化提高，夏普及最差分段夏普不降；按不同基线保留至多5项。排序依次比较正收益分段比例、最差夏普、中位夏普、中位净年化、最差回撤，最后比较整体夏普；并非只取收益第一。没有合格者时不强行给出优胜者。此规则是组合诊断，不是新因子准入门槛。", ""]
        for row in shortlist:
            lines.append(f"- {row['name']}：加入 `{row['added']}`，移除 `{row['removed'] or '无'}`；共{len(row['directions'])}因子；研究期净年化{row['metrics']['selection']['annual_return']:.2%}。")
        lines += ["", "### 全部符合诊断规则的增量（含未进入短名单者）", "", "| 基线 | 加入 | 移除 | 净年化变化(pp) | 夏普变化 | 最差分段夏普变化 | 年换手变化 |", "|---|---|---|---:|---:|---:|---:|"]
        for row in improved:
            m, b = row["metrics"], baselines[row["seed"]]["metrics"]
            lines.append(f"| {row['name']} | `{row['added']}` | `{row['removed'] or '无'}` | {100*(m['selection']['annual_return']-b['selection']['annual_return']):+.3f} | {m['selection']['sharpe']-b['selection']['sharpe']:+.3f} | {m['robustness']['worst_sharpe']-b['robustness']['worst_sharpe']:+.3f} | {m['selection']['annualized_turnover']-b['selection']['annualized_turnover']:+.2f} |")
        from core.registry import get as registry_get
        clusters = json.loads((output / "clusters.json").read_text(encoding="utf-8"))["clusters"]
        lines += ["", "### 本轮21因子身份与诊断簇", "", "输入bar强校验；均输出日频暴露。best_period仅为准入预测期证据标签，不控制持仓期。簇是本轮49因子截至截止日的绝对日度排名相关性完全链接聚类（阈值0.50），不是经济学分类、永久编号或新增准入门槛。诊断通过次数统计增加及替换，零次不代表单因子检验无效。", "",
            "| 因子 | 说明 | 输入bar | best_period | 方向 | 诊断簇 | 增量诊断通过次数 |", "|---|---|---|---:|---:|---:|---:|"]
        for name, item in sorted(contract["candidates"].items()):
            description = getattr(registry_get("factor", name), "description", name)
            lines.append(f"| `{name}` | {description} | {item['input_bar_frequency']} | H{item['best_period']} | {item['direction']:+d} | C{clusters[name]} | {sum(r['added']==name for r in improved)} |")
    lines += ["", "## 性能记录", "", "逐项耗时包含计算和诊断，累计耗时不等于从首次预热开始的总墙钟时间。带cProfile的试跑有额外开销；峰值为各进程历史最高工作集，不是每个任务单独占用。", "",
        "| 任务类型 | 完成项数 | 累计秒 | 中位秒 | P95秒 |", "|---|---:|---:|---:|---:|"]
    for kind in ("baseline", "buffer", "add", "replace"):
        seconds = [r["wall_seconds"] for r in successful if r["kind"] == kind]
        if seconds:
            lines.append(f"| {kind} | {len(seconds)} | {sum(seconds):.1f} | {np.median(seconds):.2f} | {np.quantile(seconds,.95):.2f} |")
    lines += ["", "带剖析的任务热点（累计时间含子调用，不能相加）：", "", "| 任务 | 函数 | 秒 |", "|---|---|---:|"]
    for row in successful:
        for hotspot in row.get("hotspots", [])[:6]:
            lines.append(f"| {row['id']} | `{hotspot['function']}` | {hotspot['seconds']:.2f} |")
    lines += ["", "## 解释边界与后续", "", "净收益已扣现行费用，不重复扣费。年换手为全部开平买卖成交名义量，不除以2、不再乘2倍杠杆；换月额外量等于具体合约成交量减品种净调仓量。分解、2/4/6/8bp等成本压力与分段结果保存于每项metrics。", "",
        "已有真实持仓来自账本当日估值后的漂移仓位；新目标下一日生效。" +
        ("本轮B0精确复现父实验的逐日账本；父实验增量候选未保存旧资产矩阵，不作该项对账声明。" if interaction else
         "B0逐日账本、实际权重、收益及归因精确复现冻结原报告。") +
        "平滑净值或重复目标权重均不视为减少实际交易。", "",
        "本轮只执行冻结合同声明的任务，不声称穷尽所有组合；免交易区间、单因子独立组合、多因子搜索及真正组合级准入尚未由本轮完成。分段稳定性不是独立walk-forward准入。采纳前须继续独立组合验证及真实执行成本检查。", "", "![净值对照](nav_comparison.png)", "", "![缓冲逐组净值](buffer_nav.png)", "", "![缓冲效果](buffer_effects.png)", ""]
    if not interaction:
        lines += ["![新增因子逐一增加](factor_increment_effects.png)", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"completed": len(results), "planned": planned, "finished": finished,
            "successful": len(successful), **({"frozen_candidates": len(contract["peers"])} if interaction else
                {"qualifying": len(improved), "shortlist": len(shortlist)})}


def run_repeated_library_search(output: Path, parent: Path, *, prepare_only=False) -> dict:
    """Repeat both historical selection methods on the current admitted pool."""
    import duckdb
    from core.registry import get as registry_get
    from research.artifacts import sha256_file
    from research.effective_factor_library import load_library
    from workflows.factor_selection import (
        _run_portfolio_search, _run_budgeted_pool_search, _portfolio_shortlist,
        _production_recipe, _configured_cost_model, _exposure_correlation, _cluster,
    )
    parent = parent.resolve()
    previous = json.loads((parent / "contract.json").read_text(encoding="utf-8"))
    accepted = json.loads((parent / "acceptance.json").read_text(encoding="utf-8"))
    reference, source = Path(previous["reference"]), previous["source"]
    if (not accepted["review_complete"]
            or sha256_file(parent / "results.duckdb") != accepted["results_sha256"]
            or sha256_file(parent / "contract.json") != accepted["computation_contract_sha256"]):
        raise ValueError("repeat selection requires intact reviewed parent evidence")
    for name, digest in {**previous["input_sha256"], **previous["code_sha256"]}.items():
        if Path(name).resolve() != Path(__file__).resolve() and sha256_file(Path(name)) != digest:
            raise ValueError(f"frozen input or factor implementation changed: {name}")
    config = load_config("config/default.yaml")
    library_path = ROOT / config.factor_library.path
    effective = [r for r in load_library(library_path)["factors"] if r["status"] == "effective"]
    directions = {r["factor"]: int(r["direction"]) for r in effective}
    names = sorted(directions)
    for row in effective:
        cls = registry_get("factor", row["factor"])
        if (cls.__module__ != "factors.library.intraday"
                or getattr(cls, "input_bar_frequency", "1min") != row["input_bar_frequency"]
                or sha256_file(ROOT / row["evidence_file"]) != row["evidence_sha256"]):
            raise ValueError(f"native identity, input bar or admission evidence changed: {row['factor']}")
    recipe = _production_recipe(config)
    if json.loads(json.dumps(recipe.to_dict())) != source["recipe"]:
        raise ValueError("repeat selection must preserve the baseline recipe")
    if _search_source_snapshot(source["panel_start"]) != {
            "end": source["end"], "source_fingerprint": source["source_fingerprint"]}:
        raise ValueError("market snapshot changed; refresh controls in a separate study")
    history = previous["long_factor_history"]
    if _search_source_snapshot(history["calendar_read_start"], source["end"]) != {
            "end": history["end"], "source_fingerprint": history["source_fingerprint"]}:
        raise ValueError("extra warmup source changed")
    provenance_a = ROOT / "runs/factor_selection/20260908_datarepair_multipath/selection_summary.json"
    provenance_b = ROOT / "runs/portfolio_factor_search/20260911_effective_pool_multistart/pool_contract.json"
    old_a = json.loads(provenance_a.read_text(encoding="utf-8"))["portfolio_search"]
    old_b = json.loads(provenance_b.read_text(encoding="utf-8"))
    methods = {
        "A": {"max_factors": old_a["maximum_explored_factor_count"],
              "exact_width": old_a["exact_shortlist_width"], "beam_width": old_a["beam_width"]},
        "B": {key: old_b[key] for key in ("budget", "beam_width", "round_width")},
    }
    segments = [tuple(map(pd.Timestamp, pair)) for pair in old_a["segments"]]
    if segments[0][0] != pd.Timestamp(source["performance_start"]) or segments[-1][1] != pd.Timestamp(source["selection_end"]):
        raise ValueError("historical selection dates differ from frozen controls")
    peers = previous["peers"]
    if any(directions.get(n) != d for peer in peers for n, d in peer["directions"].items()):
        raise ValueError("current library must contain all benchmark members with identical direction")
    panels, inputs = {}, [Path(p) for p in previous["input_sha256"]]
    inputs += [parent / f for f in ("contract.json", "results.duckdb", "acceptance.json")]
    inputs += [provenance_a, provenance_b, ROOT / "workflows/factor_selection.py"]
    inputs += [ROOT / row["evidence_file"] for row in effective]
    for cache in (reference / "factor_panel_checkpoint", parent / "factor_panel_checkpoint"):
        manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
        inputs.append(cache / "manifest.json")
        for name, filename in manifest["completed"].items():
            if name in directions:
                panels[name] = cache / filename
                inputs.append(cache / filename)
    contract = {**previous, "role": "repeat_historical_A_B_on_current_effective_library",
        "reference": str(reference), "panel_parent": str(parent), "peers": peers, "candidates": {r["factor"]: r for r in effective},
        "effective_library_count": len(names), "directions": directions, "research_only_extras": {},
        "input_sha256": {str(p.resolve()): sha256_file(p) for p in sorted(set(inputs))},
        "code_sha256": {name: sha256_file(Path(name)) for name in previous["code_sha256"]},
        "methods": methods, "selection_segments": [[str(a.date()), str(b.date())] for a, b in segments],
        "jobs": "A historical forward method and B historical budgeted method; benchmark controls never seed search",
        "shortlist_rule": "unchanged portfolio Pareto shortlist independently per method; deduplicate identical sets only for display",
        "clustering": "all members remain eligible; complete linkage abs rank correlation .50 for diagnosis",
        "cutoff_ic_label": "A caller and B core each remove exactly the final T+1 IC label",
        "panel_reuse": f"read-only hashed {len(panels)} parent panels; compute only remaining admitted factors",
        "trade_rule": "B0 only; buffers are not part of factor selection",
        "promotion": "none"}
    path = output / "contract.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != contract:
        raise ValueError("repeat selection contract changed; no mixed resume")
    output.mkdir(parents=True, exist_ok=True)
    _json_dump(path, contract)
    if prepare_only:
        return {"status": "frozen", "pool": len(names), "reused": len(panels),
                "missing": len(set(names)-set(panels)), "methods": methods}
    began, cpu = time.perf_counter(), time.process_time()
    with duckdb.connect(str(output / "search_results.duckdb")) as db:
        missing = sorted(set(names)-set(panels))
        print(f"repeat selection panel: pool={len(names)} reuse={len(panels)} compute={len(missing)}", flush=True)
        runner = Runner(missing, start=source["panel_start"], end=source["end"], factor_directions=directions,
                        ic_horizon=1, checkpoint_dir=output / "factor_panel_checkpoint")
        _write_factor_profile(output / "factor_profile.json", runner,
                              preparation_seconds=time.perf_counter()-began)
        for name, file in panels.items():
            frame = pd.read_parquet(file)
            if not frame.index.equals(runner.cal) or list(frame.columns) != runner.u:
                raise ValueError(f"frozen panel axes differ: {name}")
            runner.raw_ranks[name] = frame.rank(axis=1, pct=True)
        runner = runner.for_factors(names, factor_directions=directions)
        runner.get_contract_schedule()
        runner.env = CausalEligibilityEnvironment(runner.cal, runner.daily_ret, runner.env.sector_of)
        gc.collect()
        start, cutoff = pd.Timestamp(source["performance_start"]), pd.Timestamp(source["selection_end"])
        jobs = [{"id": p["id"]+"__B0", "seed": p["id"], "name": p["name"], "buffer": 0,
                 "kind": "baseline", "directions": p["directions"], "added": "", "removed": ""} for p in peers]
        full = PortfolioEvaluator(runner, start=start, end=source["end"], cost_model=_configured_cost_model(config),
            ic_window=config.production_portfolio.ic_window,
            risk_lookback_calendar_days=config.production_portfolio.risk_lookback_calendar_days)
        _run_incremental_jobs(output, reference, full, recipe, jobs,
                              {**source, "selection_segments": contract["selection_segments"]})
        full.clear_transient_caches()
        print("repeat selection: all original B0 controls exact; starting historical A/B", flush=True)
        corr = _exposure_correlation({n: runner.ranks[n].loc[start:cutoff] for n in names}, names)
        clusters = _cluster(corr, names)
        _json_dump(output / "clusters.json", {"clusters": clusters, "names": names,
                    "correlation": corr.to_numpy().tolist(), "permanent_exclusions": []})
        evaluator = PortfolioEvaluator(runner, start=start, end=cutoff, cost_model=_configured_cost_model(config),
            ic_window=config.production_portfolio.ic_window,
            risk_lookback_calendar_days=config.production_portfolio.risk_lookback_calendar_days)
        path_a, rows_a, reason_a = _run_portfolio_search(
            evaluator=evaluator, portfolio_ic=runner.ic.loc[:cutoff].iloc[:-1],
            representatives=names, recipe=recipe, segments=segments, output=output, cache_db=db, **methods["A"])
        selected_a = _portfolio_shortlist(rows_a)
        _json_dump(output / "forward_selection.json", {"path": path_a, "attempted": len(rows_a),
            "stop_reason": reason_a, "shortlist": selected_a, "observation_used_for_selection": False})
        evaluator.clear_transient_caches()
        _, rows_b, _ = _run_budgeted_pool_search(
            evaluator=evaluator, portfolio_ic=runner.ic, pool=names, recipe=recipe, segments=segments,
            clusters=clusters, db=db, output=output, **methods["B"])
        selected_b = json.loads((output / "pool_selection.json").read_text(encoding="utf-8"))["shortlist"]
        selected, seen = [], {}
        for method, rows in (("A", selected_a), ("B", selected_b)):
            for position, row in enumerate(rows, 1):
                members = tuple(sorted(row["factors"]))
                if members not in seen:
                    selected.append({"id": f"{method}{position}", "factors": list(members),
                        "methods": [method], "selection_evidence": row})
                    seen[members] = selected[-1]
                else:
                    seen[members]["methods"].append(method)
        _json_dump(output / "selection_manifest.json", {"methods": methods, "selected": selected,
            "observation_used_for_selection": False, "production_approved": False,
            "coverage": {method: {"attempts": len(rows), "successful": sum(r["status"]=="evaluated" for r in rows),
                "attempted_factors": len({n for r in rows for n in r["factors"]}),
                "successful_factors": len({n for r in rows if r["status"]=="evaluated" for n in r["factors"]})}
                for method, rows in (("A", rows_a), ("B", rows_b))}})
        # Both method shortlists are frozen above before any observation-period comparisons.
        jobs += [{"id": row["id"], "seed": row["id"], "name": f"旧法{row['id']}候选({len(row['factors'])})",
                  "buffer": 0, "kind": "candidate", "directions": {n: directions[n] for n in row["factors"]},
                  "added": "", "removed": "", "methods": row["methods"]} for row in selected]
        _json_dump(output / "jobs.json", jobs)
        _run_incremental_jobs(output, reference, full, recipe, jobs,
                              {**source, "selection_segments": contract["selection_segments"]})
        with duckdb.connect(str(output / "results.duckdb"), read_only=True) as results_db:
            for row in selected:
                measured = results_db.execute("SELECT net_return FROM daily WHERE job=? AND date<=? ORDER BY date",
                                              [row["id"], cutoff]).fetchnumpy()["net_return"]
                for method in row["methods"]:
                    table = "forward_results" if method == "A" else "pool_results"
                    saved = db.execute(f"SELECT net_returns FROM {table} WHERE signature=?",
                                       [json.dumps(tuple(row["factors"]))]).fetchone()[0]
                    np.testing.assert_array_equal(measured, saved)
        _json_dump(output / "performance.json", {"wall_seconds": time.perf_counter()-began,
            "cpu_seconds": time.process_time()-cpu, "attempted_A": len(rows_a), "attempted_B": len(rows_b),
            "measured_A_seconds": sum(r["seconds"] for r in rows_a), "measured_B_seconds": sum(r["seconds"] for r in rows_b),
            "cached_resume_note": "process wall time excludes work completed by earlier processes",
            "comparison_cutoff_returns_exact": True})
    return report_repeated_library_search(output)


def report_repeated_library_search(output: Path) -> dict:
    """Read completed exact ledgers; never reselect candidates or edit defaults."""
    import duckdb
    from collections import Counter
    from core.registry import get as registry_get
    from monitoring.plot_style import signed_heatmap_cmap
    from run_portfolio_workflow import _write_comparison_plot
    from monitoring.weekly_report import _mk_png
    plt = _mk_png()
    contract = json.loads((output / "contract.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "selection_manifest.json").read_text(encoding="utf-8"))
    jobs = json.loads((output / "jobs.json").read_text(encoding="utf-8"))
    clusters = json.loads((output / "clusters.json").read_text(encoding="utf-8"))["clusters"]
    performance = json.loads((output / "performance.json").read_text(encoding="utf-8"))
    nav, plot_rows, export = {}, [], []
    with duckdb.connect(str(output / "results.duckdb"), read_only=True) as db:
        saved = {key: json.loads(value) for key, value in db.execute("SELECT id,payload FROM results").fetchall()}
        if set(saved) != {j["id"] for j in jobs} or any(r["status"] != "completed" for r in saved.values()):
            raise ValueError("repeat selection report requires every native comparison completed")
        rows = [saved[j["id"]] for j in jobs]
        for row in rows:
            label, metrics = row["name"], row["metrics"]
            frame = db.execute("SELECT date,nav_after FROM daily WHERE job=? ORDER BY date",
                               [row["id"]]).df().set_index("date")
            nav[label] = frame.nav_after / frame.nav_after.iloc[0]
            plot_rows.append({"strategy": label, **metrics["full"], "volatility": metrics["full"]["annual_volatility"]})
            export.append({"id": row["id"], "strategy": label, "kind": row["kind"], "factor_count": len(row["directions"]),
                **{f"{period}_{key}": metrics[period][key] for period in ("selection", "observation", "full")
                   for key in ("annual_return", "sharpe", "max_drawdown", "total_return", "annualized_turnover")},
                **metrics["robustness"]})
    pd.DataFrame(export).to_csv(output / "comparison.csv", index=False, encoding="utf-8-sig")
    _write_comparison_plot(output, pd.DataFrame(nav), plot_rows, cutoff=pd.Timestamp(contract["source"]["selection_end"]),
        title="复用两条历史方法：全有效库候选与现有10组对照\n统一B0执行；截止后仅观察，未改变生产配置")
    values = np.array([[s["annual_return"]*100 for s in row["metrics"]["segments"]] for row in rows])
    fig, ax = plt.subplots(figsize=(12, max(6, .45*len(rows))))
    bound = max(float(np.max(np.abs(values))), .01)
    im = ax.imshow(values, cmap=signed_heatmap_cmap(), vmin=-bound, vmax=bound, aspect="auto")
    ax.set_yticks(range(len(rows)), [r["name"] for r in rows])
    ax.set_xticks(range(len(contract["selection_segments"])),
                  [f"{a}\n—{b}" for a,b in contract["selection_segments"]])
    for i in range(len(rows)):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i,j]:+.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="净年化（%）")
    ax.set_title("5个原历史分段：净年化稳定性\n红正绿负、零白；发展样本分段，不是独立Walk-forward")
    fig.tight_layout()
    fig.savefig(output / "selection_segments.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    lines = ["# 当前有效库：复用两条历史选集方法", "",
        f"候选池{contract['effective_library_count']}个，全部来自正式有效库，无GP/SPEC外接表达式；当前10组仅作基线、不作搜索种子。未修改成员登记、默认参数或因子准入。",
        f"回测{contract['source']['performance_start']}至{contract['source']['end']}，标准面板从{contract['source']['panel_start']}预热；长预热按因子原声明补足。选集止于含端点{contract['source']['selection_end']}，之后已查看区间只作观察。", "",
        "## 方法及实际覆盖", "",
        "A复用20260908多路径正向：2至20因子、beam20、每个规模3项成功精确回测；失败时可在该规模beam内继续尝试，57不是硬尝试上限。B复用20260911多起点：512次预算、beam6、每轮16提案，含增加/删除/替换和增长路径。不是全子集穷举或全局最优证明。",
        "两者均使用原5段、原净回测稳健性/Pareto与Jaccard规则，各自最多保留5组。只合并完全相同成员用于展示，不按观察段重新择优。IC只排提案顺序；跨截止的最后H1标签已排除。成功覆盖和尝试覆盖不同，运行拒绝不是因子准入失效。", "",
        "| 方法 | 尝试 | 成功 | 尝试覆盖因子 | 成功覆盖因子 |", "|---|---:|---:|---:|---:|"]
    for method, coverage in manifest["coverage"].items():
        lines.append(f"| {method} | {coverage['attempts']} | {coverage['successful']} | {coverage['attempted_factors']} | {coverage['successful_factors']} |")
    lines += ["", "## 统一执行要素的实际回测", "",
        "lw_abs、Top10/Bottom10、cap3、ERC、总敞口2倍、B0、原费用。年换手为完整成交量，已含双侧及杠杆，不能再翻倍。短观察段报累计收益而非夸大年化。", "",
        "| 组合 | 截止前年化 | 夏普 | 回撤 | 年换手 | 最差段夏普 | 观察段累计 | 全期年化 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        m = row["metrics"]; a = m["selection"]
        lines.append(f"| {row['name']} | {a['annual_return']:.2%} | {a['sharpe']:.3f} | {a['max_drawdown']:.2%} | {a['annualized_turnover']:.2f} | {m['robustness']['worst_sharpe']:.3f} | {m['observation']['total_return']:.2%} | {m['full']['annual_return']:.2%} |")
    lines += ["", "## 候选成员、簇与原10组交集", "",
        "簇来自本轮188因子截止前绝对日度排名相关性完全链接（0.50）；只作诊断/种子多样性提示，不永久剔除簇内成员。best_period仅准入证据标签，不是持有天数。"]
    for row in rows:
        if row["kind"] != "candidate": continue
        members = set(row["directions"])
        lines += ["", f"### {row['name']}（方法{'/'.join(row['methods'])}）", "",
            "簇构成：" + "、".join(f"C{k}×{v}" for k,v in sorted(Counter(clusters[n] for n in members).items())) + "。",
            "与原组的共同因子数：" + "；".join(f"{p['name']}：{len(members & set(p['directions']))}" for p in contract["peers"]) + "。", "",
            "| 因子 | 注册类别 | 说明 | 簇 | 原始bar | 证据期 | 方向 |", "|---|---|---|---:|---|---:|---:|"]
        for n in sorted(members):
            record = contract["candidates"][n]
            lines.append(f"| `{n}` | {record.get('family','')} | {getattr(registry_get('factor',n),'description',n)} | C{clusters[n]} | {record['input_bar_frequency']} | H{record['best_period']} | {row['directions'][n]:+d} |")
    lines += ["", "## 成本与性能", "", "压力测试固定实际交易路径，持有费用不变，不代表容量认证。", "",
        "| 组合 | 4bp净年化 | 8bp净年化 | 静态盈亏平衡bp |", "|---|---:|---:|---:|"]
    for row in rows:
        sensitivity = row["metrics"]["selection"]["cost_sensitivity"]
        costs = {round(s["trade_cost_rate"]*10000): s["metrics"]["annual_return"] for s in sensitivity["scenarios"]}
        lines.append(f"| {row['name']} | {costs[4]:.2%} | {costs[8]:.2%} | {sensitivity['breakeven_trade_cost_rate']*10000:.3f} |")
    lines += ["", f"本次进程墙钟{performance['wall_seconds']:.1f}秒、CPU {performance['cpu_seconds']:.1f}秒。各精确搜索、因子计算、比较账本的耗时见search_results.duckdb、factor_profile.json及results.duckdb；缓存续跑的本次墙钟不含此前进程。",
        "原10组B0逐日/资产矩阵精确对账，候选延长观察后的截止前收益数组必须与其搜索账本完全一致。历史库由近期准入而定，因此向前发展样本不是当时真实可投资的独立样本外。入选短名单不等于组合级正式准入，不能因观察表现直接切换生产。",
        "", "![统一净值](nav_comparison.png)", "", "![5段稳定性](selection_segments.png)", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"completed": True, "baselines": sum(r["kind"]=="baseline" for r in rows),
            "candidates": len(manifest["selected"]), "promoted": False}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expanding-window native production method and factor search"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--incremental-trade-study", action="store_true",
                        help="explicit admitted-factor additions/replacements and isolated B0/B1/B2")
    parser.add_argument("--incremental-trade-report", action="store_true")
    parser.add_argument("--repeat-selection-parent", type=Path,
                        help="repeat both historical selection methods on the current effective library")
    parser.add_argument("--repeat-selection-report", action="store_true")
    parser.add_argument("--trade-interaction-parent", type=Path,
                        help="explicit reviewed shortlist for isolated B0/B1/B2 interaction; no new factor search")
    parser.add_argument("--admission-runs", help="comma-separated effective-library source_run identities")
    parser.add_argument("--effective-pool-search", action="store_true",
                        help="explicit independent full-pool budgeted search; no default/library mutation")
    parser.add_argument("--reference-run", type=Path,
                        help="frozen latest-eight native search run for baselines and verified panel reuse")
    parser.add_argument("--native-neighborhood-plan", action="store_true",
                        help="freeze the full 96-recipe native-factor neighborhood; no backtest")
    parser.add_argument("--native-neighborhood-pilot", action="store_true",
                        help="benchmark two frozen baselines against an existing native search contract")
    parser.add_argument("--native-neighborhood-run", action="store_true",
                        help="resume the exact joint grid after the completed pilot")
    parser.add_argument("--native-ledger-parity", action="store_true",
                        help="verify optimized real ledgers against the completed pilot")
    parser.add_argument("--native-neighborhood-report", action="store_true",
                        help="report completed exact jobs; requires the batch writer to have stopped")
    parser.add_argument("--baselines-only", action="store_true",
                        help="run all 96 recipes for the eight frozen original memberships only")
    parser.add_argument("--max-jobs", type=int, help="bounded resumable batch; omitted means entire declared scope")
    parser.add_argument("--candidate-manifest", type=Path,
                        help="explicit native migration manifest; never starts GP/SPEC mining")
    parser.add_argument("--validation-results", type=Path,
                        help="formal passed_factors.csv proving native candidates passed")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse an interrupted run's temporary factor-panel cache",
    )
    parser.add_argument(
        "--8f-recipe-only",
        dest="recipe_only_8f",
        action="store_true",
        help="only run the two-recipe expanding-window freeze test for fixed 8f",
    )
    parser.add_argument(
        "--factor-manifest",
        help="immutable discovery ic_by_window_period.json; required for factor search",
    )
    args = parser.parse_args()
    output = Path(args.output)
    if args.repeat_selection_parent or args.repeat_selection_report:
        if (bool(args.repeat_selection_parent) == args.repeat_selection_report or any((
                args.incremental_trade_study, args.incremental_trade_report, args.trade_interaction_parent,
                args.effective_pool_search, args.native_neighborhood_plan, args.native_neighborhood_run,
                args.native_neighborhood_pilot, args.native_ledger_parity, args.native_neighborhood_report,
                args.resume, args.recipe_only_8f, args.reference_run, args.admission_runs,
                args.baselines_only, args.candidate_manifest, args.validation_results, args.factor_manifest))
                or args.max_jobs not in (None, 0) or (args.repeat_selection_report and args.max_jobs is not None)):
            parser.error("repeat selection is an exclusive frozen mode; max-jobs=0 only prepares its contract")
        result = (report_repeated_library_search(output) if args.repeat_selection_report else
                  run_repeated_library_search(output, args.repeat_selection_parent, prepare_only=args.max_jobs == 0))
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return
    if args.incremental_trade_study or args.incremental_trade_report or args.trade_interaction_parent:
        if any((args.effective_pool_search, args.native_neighborhood_plan, args.native_neighborhood_run,
                args.native_neighborhood_pilot, args.native_ledger_parity, args.native_neighborhood_report,
                args.resume, args.recipe_only_8f)) or sum(bool(x) for x in (
                    args.incremental_trade_study, args.incremental_trade_report, args.trade_interaction_parent)) > 1:
            parser.error("incremental trade study/report cannot be mixed with another execution mode")
        if args.trade_interaction_parent:
            if args.reference_run or args.admission_runs or (args.max_jobs is not None and args.max_jobs < 0):
                parser.error("interaction uses its frozen parent; no reference/admission override or negative max-jobs")
            print(json.dumps(run_trade_interactions(output, args.trade_interaction_parent,
                max_jobs=args.max_jobs), ensure_ascii=False), flush=True)
        elif args.incremental_trade_report:
            print(json.dumps(report_incremental_trade_study(output), ensure_ascii=False), flush=True)
        else:
            if not args.reference_run or not args.admission_runs or (args.max_jobs is not None and args.max_jobs < 0):
                parser.error("incremental study requires --reference-run, --admission-runs and nonnegative --max-jobs")
            print(json.dumps(run_incremental_trade_study(output, args.reference_run,
                args.admission_runs.split(","), max_jobs=args.max_jobs), ensure_ascii=False), flush=True)
        return
    if args.effective_pool_search:
        if not args.reference_run or any((args.native_neighborhood_plan, args.native_neighborhood_run,
                args.native_neighborhood_pilot, args.native_ledger_parity, args.native_neighborhood_report, args.resume)):
            parser.error("pool search requires --reference-run and no other execution mode")
        print(json.dumps(run_effective_pool_search(output, args.reference_run), indent=2), flush=True)
        return
    if args.native_neighborhood_report:
        if args.native_neighborhood_plan or args.native_neighborhood_pilot or args.native_neighborhood_run or args.native_ledger_parity or args.resume:
            parser.error("report is separate from all execution modes")
        print(json.dumps(report_neighborhood_search(output), indent=2), flush=True)
        return
    if args.native_ledger_parity:
        if args.native_neighborhood_plan or args.native_neighborhood_pilot or args.native_neighborhood_run or args.resume:
            parser.error("ledger verification is separate from other execution modes")
        print(json.dumps(verify_neighborhood_ledgers(output)["status"]), flush=True)
        return
    if args.native_neighborhood_run:
        if args.native_neighborhood_plan or args.native_neighborhood_pilot or args.resume:
            parser.error("native run is separate from plan, pilot, and legacy resume")
        print(json.dumps(run_neighborhood_batch(output, baseline_only=args.baselines_only,
                                                max_jobs=args.max_jobs), indent=2), flush=True)
        return
    if args.native_neighborhood_pilot:
        if args.native_neighborhood_plan or args.resume:
            parser.error("pilot is separate from plan generation and legacy resume")
        profile = benchmark_neighborhood_search(output)
        print(json.dumps({"wall_seconds": profile["wall_seconds"], "jobs": len(profile["jobs"])}, indent=2), flush=True)
        return
    if args.native_neighborhood_plan:
        if not args.candidate_manifest or not args.validation_results or args.resume:
            parser.error("native plan requires --candidate-manifest and --validation-results; no --resume")
        contract = prepare_neighborhood_search(output, args.candidate_manifest,
                                               args.validation_results)
        print(json.dumps(contract["counts"], ensure_ascii=False, indent=2), flush=True)
        return
    if args.resume:
        if not output.is_dir():
            raise FileNotFoundError(f"resume output does not exist: {output}")
    else:
        output.mkdir(parents=True, exist_ok=False)

    if args.recipe_only_8f:
        valid_factors = list(FACTORS_8F)
        manifest_sha256 = None
    else:
        if not args.factor_manifest:
            parser.error("--factor-manifest is required unless --8f-recipe-only is used")
        manifest_path = Path(args.factor_manifest)
        if not manifest_path.is_file():
            parser.error(f"factor manifest does not exist: {manifest_path}")
        valid_factors, manifest_sha256 = _load_estimable_factor_manifest(manifest_path)
    resolved = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "8f_recipe_only" if args.recipe_only_8f else "full_search",
        "factor_library_policy": "estimable_factors_from_frozen_p0_manifest",
        "factor_manifest_sha256": manifest_sha256,
        "valid_factor_count": len(valid_factors),
        "valid_factors": valid_factors,
        "outer_folds": OUTER_FOLDS,
        "period_protocol": historical_experiment_period_snapshot(),
        "method_grid": (
            {
                "predeclared_recipes": {
                    name: recipe.to_dict()
                    for name, recipe in RECIPE_CHALLENGERS_8F.items()
                },
                "search": "training-rank two recipes, freeze winner to next test fold",
            }
            if args.recipe_only_8f else {
                "factor_weight": ["equal", "diag_icir", "lw_abs", "lw_positive"],
                "top_n": [8, 10, 12],
                "sector_cap": [0, 3],
                "asset_weight": ["equal", "inverse_volatility", "erc"],
                "search": "three-stage coordinate search, top2/top2/top3",
            }
        ),
        "factor_search": (
            {"enabled": False, "fixed_factor_set": "8f"}
            if args.recipe_only_8f else {
                "training_only_cluster_correlation": 0.65,
                "factor_count": [4, 12],
                "beam_width": 20,
                "exact_portfolio_candidates_per_fold": 12,
            }
        ),
        "costs": COST_MODEL.ledger_parameters(),
        "gross_exposure": 2.0,
    }
    resolved_path = output / "resolved_search_config.json"
    if args.resume and resolved_path.exists():
        previous = json.loads(resolved_path.read_text(encoding="utf-8"))
        for key in ("factor_manifest_sha256", "period_protocol"):
            if previous.get(key) != resolved.get(key):
                raise ValueError(f"resume protocol mismatch: {key}")
    else:
        _json_dump(resolved_path, resolved)

    print(f"compute current validated factor pool: {len(valid_factors)}", flush=True)
    panel_cache = output / "_factor_panel_cache.pkl"
    if args.resume and panel_cache.is_file():
        print(f"reuse interrupted factor-panel cache: {panel_cache}", flush=True)
        payload = pd.read_pickle(panel_cache)
        if (
            payload.get("schema_version") != 4
            or payload.get("factor_names") != valid_factors
            or "close_tradable" not in payload
            or "contract_schedule" not in payload
        ):
            raise ValueError(
                "stale factor-panel cache: rerun without --resume to rebuild schema 4"
            )
        environment = CausalEligibilityEnvironment(
            payload["cal"], payload["daily_ret"], payload["sector_of"]
        )
        runner = SimpleNamespace(
            env=environment,
            cal=payload["cal"],
            u=payload["u"],
            daily_ret=payload["daily_ret"],
            close_tradable=payload["close_tradable"],
            contract_schedule=payload["contract_schedule"],
            ranks=payload["ranks"],
            ic=payload["ic"],
        )
    else:
        runner = Runner(valid_factors)
        sector_of = dict(runner.env.sector_of)
        runner.get_contract_schedule()
        runner.env = CausalEligibilityEnvironment(
            runner.cal, runner.daily_ret, sector_of
        )
        gc.collect()
        pd.to_pickle({
            "schema_version": 4,
            "factor_names": valid_factors,
            "cal": runner.cal,
            "u": runner.u,
            "daily_ret": runner.daily_ret,
            "close_tradable": runner.close_tradable,
            "contract_schedule": runner.get_contract_schedule(),
            "ranks": runner.ranks,
            "ic": runner.ic,
            "sector_of": sector_of,
        }, panel_cache)
    evaluator = PortfolioEvaluator(
        runner,
        start=HISTORICAL_START,
        end=OOS_END,
        cost_model=COST_MODEL,
    )
    if args.recipe_only_8f:
        metrics, decisions = _recipe_walk_forward_8f(evaluator, output)
        if panel_cache.is_file():
            panel_cache.unlink()
        print(json.dumps({
            "output": str(output),
            "metrics": metrics.to_dict("records"),
            "selected_by_fold": [
                row["selected_challenger"] for row in decisions
            ],
        }, ensure_ascii=False, indent=2), flush=True)
        return
    shortlist_path = output / "method_shortlist.json"
    if args.resume and shortlist_path.is_file():
        print("reuse completed method search", flush=True)
        shortlist = json.loads(shortlist_path.read_text(encoding="utf-8"))
        recipes_by_fold = {
            fold: [
                PortfolioRecipe(
                    factor_weight=row["factor_weight"],
                    top_n=int(row["top_n"]),
                    sector_cap=int(row["sector_cap"]),
                    asset_weight=row["asset_weight"],
                )
                for row in rows[:3]
            ]
            for fold, rows in shortlist["final_by_fold"].items()
        }
    else:
        print("stage 1-3: method search", flush=True)
        recipes_by_fold, _ = _stage_method_search(evaluator, output)
    evaluator.clear_transient_caches()
    print("stage 4: cluster-aware factor search", flush=True)
    adaptive_weights, adaptive_ledger, decisions, _ = _factor_search(
        runner, evaluator, recipes_by_fold, output, valid_factors
    )
    live_metrics = _evaluate_simulated_live(runner, decisions[-1], output)
    print("final comparison and cost stress", flush=True)
    metrics = _comparison(evaluator, adaptive_weights, adaptive_ledger, output)
    _recipe_walk_forward_8f(evaluator, output)
    _write_review(output, decisions, metrics)
    if panel_cache.is_file():
        panel_cache.unlink()
    print(json.dumps({
        "output": str(output),
        "adaptive_metrics": metrics.loc[
            metrics["strategy"] == "adaptive_search"
        ].iloc[0].to_dict(),
        "simulated_live_metrics": live_metrics,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
