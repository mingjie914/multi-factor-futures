"""Expanding-window search for native factor aggregation and portfolio methods.

The current validated factor library is treated as a fixed research universe.
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
import json
from pathlib import Path
import sys
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
    NEW_VALIDATED_21 as NEW21,
    VALIDATED_47 as KEPT47,
    configured_futures_cost_model,
)


COST_MODEL = configured_futures_cost_model()

OUTER_FOLDS = [
    {
        "fold": "fold_1",
        "train_start": "2016-03-31",
        "train_end": "2019-12-31",
        "test_start": "2020-01-01",
        "test_end": "2021-12-31",
    },
    {
        "fold": "fold_2",
        "train_start": "2016-03-31",
        "train_end": "2021-12-31",
        "test_start": "2022-01-01",
        "test_end": "2023-12-31",
    },
    {
        "fold": "fold_3",
        "train_start": "2016-03-31",
        "train_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2024-12-31",
    },
    {
        "fold": "fold_4",
        "train_start": "2016-03-31",
        "train_end": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2026-08-06",
    },
]

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


def _json_dump(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


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
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], pd.DataFrame]:
    diagnostic_rows = []
    cluster_rows = []
    candidate_rows = []
    decisions = []
    selected_weight_parts = []
    all_factors = list(dict.fromkeys(F6 + list(KEPT47) + list(NEW21)))

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
    args = parser.parse_args()
    output = Path(args.output)
    if args.resume:
        if not output.is_dir():
            raise FileNotFoundError(f"resume output does not exist: {output}")
    else:
        output.mkdir(parents=True, exist_ok=False)

    valid_factors = (
        list(FACTORS_8F) if args.recipe_only_8f
        else list(dict.fromkeys(F6 + list(KEPT47) + list(NEW21)))
    )
    resolved = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "8f_recipe_only" if args.recipe_only_8f else "full_search",
        "factor_library_policy": "current_validated_library_fixed_for_all_historical_folds",
        "valid_factor_count": len(valid_factors),
        "valid_factors": valid_factors,
        "outer_folds": OUTER_FOLDS,
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
    if not args.resume or not resolved_path.exists():
        _json_dump(resolved_path, resolved)

    print(f"compute current validated factor pool: {len(valid_factors)}", flush=True)
    panel_cache = output / "_factor_panel_cache.pkl"
    if args.resume and panel_cache.is_file():
        print(f"reuse interrupted factor-panel cache: {panel_cache}", flush=True)
        payload = pd.read_pickle(panel_cache)
        if (
            payload.get("schema_version") != 3
            or "close_tradable" not in payload
            or "contract_schedule" not in payload
        ):
            raise ValueError(
                "stale factor-panel cache: rerun without --resume to rebuild schema 3"
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
            "schema_version": 3,
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
        start="2016-03-31",
        end="2026-08-06",
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
        runner, evaluator, recipes_by_fold, output
    )
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
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
