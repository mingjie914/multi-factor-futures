"""Isolated legacy expanding-window portfolio-method experiment.

This module is not the framework's formal factor-admission workflow.  Formal
admission uses ``main.py walkforward`` and the rolling policy in the single
framework config.  Here, the validated factor library is treated as a fixed
research universe for historical comparison only.
At each historical fold, every choice is made using the training interval and
then frozen for the following test interval.  This is a pragmatic historical
study, not a reconstruction of when each factor idea was originally invented.

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expanding-window native production method and factor search"
    )
    parser.add_argument("--output", required=True)
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
