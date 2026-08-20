from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.portfolio_experiment_support import (  # noqa: E402
    BASELINE_6F as PROD6,
    FACTORS_6F as F6,
    FACTORS_10F as F10,
    FACTORS_13F as F13,
    FACTORS_14F as F14,
    FACTOR_DIRECTIONS as DIRS,
    FactorPanelRunner as Runner,
    NEW_FACTOR_DIRECTIONS as NEW21_DIR,
    NEW_VALIDATED_21 as NEW21,
    VALIDATED_47 as KEPT47,
    configured_futures_cost_model,
)
from backtest.metrics import TRADING_DAYS_PER_YEAR  # noqa: E402
from backtest.research_ledger import build_close_marked_ledger  # noqa: E402
from optimization.factor_weighting import factor_weights  # noqa: E402
from research.historical_portfolio_search import (  # noqa: E402
    PortfolioEvaluator,
    PortfolioRecipe,
    performance_metrics,
)
from external_strategies.guosen_trend_index.strategy import load_snapshot  # noqa: E402
from strategies.combined import FACTORS as PRODUCTION_10F  # noqa: E402


PERIODS_PER_YEAR = TRADING_DAYS_PER_YEAR
COST_MODEL = configured_futures_cost_model()
TRADE_COST_RATE = COST_MODEL.turnover_cost_rate
ANNUAL_FEE = COST_MODEL.annual_fee
ANNUAL_ROLL_COST = COST_MODEL.annual_roll_cost
SNAPSHOT_PATH = Path(__file__).with_name("config.yaml")

GUOSEN_BALANCED_6R = [
    "intraday_jump_intensity_20d",
    "intraday_dtws_20d",
    "intraday_drip_stone_20d",
    "intraday_lowest_time_20d",
    "intraday_term_slope_20d",
    "intraday_price_delay_20d",
]


def _factor_weights(history: pd.DataFrame) -> pd.Series:
    """Compatibility wrapper for historical callers and tests."""
    return factor_weights(history, "lw_abs")


def _factor_direction(name: str) -> int:
    return int(DIRS.get(name, NEW21_DIR.get(name, PROD6.get(name, 1))))


def _validate_fixed_factor_sets(snapshot_path: Path = SNAPSHOT_PATH) -> None:
    """Fail if research labels drift from production or their frozen snapshot."""
    _, configured_sets, _ = load_snapshot(snapshot_path)
    expected_sets = {"6f": F6, "10f": F10, "13f": F13, "14f": F14}
    for label, names in expected_sets.items():
        configured = configured_sets.get(label)
        if configured is None:
            raise ValueError(f"snapshot is missing fixed factor set {label!r}")
        flattened = {}
        for group_name, variants in configured.items():
            if len(variants) != 1 or variants[0][0] != group_name:
                raise ValueError(f"snapshot factor set {label!r} is not flat")
            flattened[group_name] = int(variants[0][1])
        expected = {name: _factor_direction(name) for name in names}
        if flattened != expected:
            raise ValueError(f"snapshot factor set {label!r} drifted from code")
    if dict(PRODUCTION_10F) != {
        name: _factor_direction(name) for name in F10
    }:
        raise ValueError("10f research label drifted from production FACTORS")


def _run_production_weights(
    runner: Runner,
    factor_names: list[str],
    end: pd.Timestamp,
) -> pd.DataFrame:
    dates = runner.cal[runner.cal <= end]
    evaluator = PortfolioEvaluator(
        runner,
        start=dates.min(),
        end=dates.max(),
        cost_model=COST_MODEL,
    )
    return evaluator.weights(
        factor_names,
        PortfolioRecipe("lw_abs", 10, 3, "erc"),
    )


def _ledger_from_weights(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    contract_schedule: pd.DataFrame | None = None,
    decision_tradable: pd.DataFrame | None = None,
) -> pd.DataFrame:
    evaluation_weights = weights.loc[start:end]
    result = build_close_marked_ledger(
        evaluation_weights,
        asset_returns.reindex(
            index=evaluation_weights.index, columns=evaluation_weights.columns
        ),
        **COST_MODEL.ledger_parameters(),
        contract_schedule=contract_schedule,
        decision_tradable=decision_tradable,
        initial_nav=1000.0,
    )
    ledger = result.daily.copy()
    ledger["nav"] = ledger["nav_after"]
    return ledger


def _metrics(returns: pd.Series) -> dict[str, float | int]:
    return performance_metrics(
        returns,
        periods_per_year=PERIODS_PER_YEAR,
        initial_anchor=True,
    )


def _segment_ic_score(
    ic: pd.DataFrame,
    factors: list[str],
    masks: list[pd.Series | np.ndarray],
) -> tuple[float, list[float]]:
    combined = ic[factors].mean(axis=1, skipna=True)
    sharpes = []
    for mask in masks:
        values = combined.loc[mask].dropna()
        volatility = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        sharpes.append(
            float(values.mean() / volatility * np.sqrt(PERIODS_PER_YEAR))
            if volatility > 0.0 else -10.0
        )
    merged_mask = np.logical_or.reduce(masks)
    merged = combined.loc[merged_mask].dropna()
    merged_volatility = float(merged.std(ddof=1)) if len(merged) > 1 else 0.0
    merged_sharpe = (
        float(merged.mean() / merged_volatility * np.sqrt(PERIODS_PER_YEAR))
        if merged_volatility > 0.0 else -10.0
    )
    score = (
        0.30 * merged_sharpe
        + 0.35 * float(np.median(sharpes))
        + 0.35 * float(np.min(sharpes))
        - 0.01 * len(factors)
    )
    return score, sharpes


def _eligible_factors(
    ic: pd.DataFrame,
    candidates: list[str],
    masks: list[np.ndarray],
    minimum_coverage: float = 0.50,
) -> list[str]:
    eligible = []
    for name in candidates:
        series = ic[name]
        segment_values = [series.loc[mask] for mask in masks]
        coverage = [float(values.notna().mean()) for values in segment_values]
        means = [float(values.mean()) for values in segment_values]
        if min(coverage) < minimum_coverage:
            continue
        if float(pd.concat(segment_values).mean()) <= 0.0:
            continue
        if sum(value > 0.0 for value in means) < max(1, len(means) - 1):
            continue
        eligible.append(name)
    return eligible


def _forward_select(
    ic: pd.DataFrame,
    candidates: list[str],
    masks: list[np.ndarray],
    start: list[str] | None = None,
    maximum_size: int = 10,
    correlation_cap: float = 0.55,
) -> tuple[list[str], list[dict]]:
    current = list(start or [])
    current_score = (
        _segment_ic_score(ic, current, masks)[0] if current else -np.inf
    )
    history = []
    development_mask = np.logical_or.reduce(masks)
    while len(current) < maximum_size:
        best_name = None
        best_score = -np.inf
        best_segments: list[float] = []
        for candidate in candidates:
            if candidate in current:
                continue
            if current:
                correlations = ic.loc[development_mask, current].corrwith(
                    ic.loc[development_mask, candidate]
                ).abs()
                if correlations.max(skipna=True) >= correlation_cap:
                    continue
            score, segment_scores = _segment_ic_score(
                ic, current + [candidate], masks
            )
            if score > best_score:
                best_name = candidate
                best_score = score
                best_segments = segment_scores
        if best_name is None:
            break
        if len(current) >= 3 and best_score <= current_score + 1e-8:
            break
        current.append(best_name)
        current_score = best_score
        history.append({
            "step": len(current),
            "added": best_name,
            "score": best_score,
            "segment_ic_sharpes": best_segments,
            "factors": list(current),
        })
    return current, history


def _reference_nav_path(snapshot_path: Path = SNAPSHOT_PATH) -> Path:
    _, _, raw = load_snapshot(snapshot_path)
    value = raw.get("snapshot", {}).get("reference_nav")
    if not value:
        raise ValueError("snapshot.reference_nav is required")
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else snapshot_path.parent / path


def _load_reference(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    path = _reference_nav_path()
    if not path.is_file():
        raise FileNotFoundError(f"Guosen reference NAV not found: {path}")
    frame = pd.read_csv(path, parse_dates=["date"])
    if list(frame.columns) != ["date", "nav"]:
        raise ValueError(f"unexpected Guosen columns: {frame.columns.tolist()}")
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise ValueError("Guosen dates must be unique and sorted ascending")
    if frame["nav"].isna().any() or not np.isfinite(frame["nav"]).all():
        raise ValueError("Guosen NAV must be finite")
    if (frame["nav"] <= 0.0).any():
        raise ValueError("Guosen NAV must be positive")
    nav = frame.set_index("date")["nav"].sort_index().loc[start:end]
    if nav.empty:
        raise ValueError(f"Guosen reference NAV has no data in {start.date()}..{end.date()}")
    if pd.Timestamp(nav.index[0]) != start:
        raise ValueError(f"Guosen reference NAV does not cover requested start {start.date()}")
    return nav / nav.iloc[0] * 1000.0


def _plot(
    navs: Mapping[str, pd.Series],
    start: pd.Timestamp,
    title: str,
    output: Path,
) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(14, 8))
    for name, full_nav in navs.items():
        nav = full_nav.loc[start:].dropna()
        if nav.empty:
            continue
        nav = nav / nav.iloc[0] * 1000.0
        metrics = _metrics(nav.pct_change(fill_method=None).fillna(0.0))
        style = (
            {"color": "black", "linestyle": "--", "linewidth": 2.0}
            if name == "trend" else {}
        )
        ax.plot(
            nav.index,
            nav.values,
            label=(
                f"{name} | 年化{metrics['annual_return']:.1%} "
                f"夏普{metrics['sharpe']:.2f} 回撤{metrics['max_drawdown']:.1%}"
            ),
            **style,
        )
    ax.set_title(title)
    ax.set_ylabel("净值（区间起点=1000）")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Research factor-set search; use current_core_compare for fixed current comparisons"
    )
    parser.add_argument("--start", default="2016-03-31")
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    _validate_fixed_factor_sets()

    valid_factors = list(dict.fromkeys(F6 + KEPT47 + NEW21))
    directions = {
        name: _factor_direction(name)
        for name in valid_factors
    }
    print(f"compute valid factor pool: {len(valid_factors)} factors", flush=True)
    runner = Runner(valid_factors)
    ic = runner.ic.reindex(columns=valid_factors)
    dates = pd.DatetimeIndex(ic.index)
    development_segments = [
        (dates >= pd.Timestamp("2016-03-31")) & (dates <= pd.Timestamp("2019-12-31")),
        (dates >= pd.Timestamp("2020-01-01")) & (dates <= pd.Timestamp("2022-12-31")),
        (dates >= pd.Timestamp("2023-01-01")) & (dates <= pd.Timestamp("2024-12-31")),
    ]
    post_2020_segments = development_segments[1:]
    full_eligible = _eligible_factors(
        ic, valid_factors, development_segments
    )
    post_2020_eligible = _eligible_factors(
        ic, valid_factors, post_2020_segments
    )
    robust_full, robust_full_path = _forward_select(
        ic, full_eligible, development_segments, maximum_size=10
    )
    robust_post_2020, robust_post_path = _forward_select(
        ic, post_2020_eligible, post_2020_segments, maximum_size=10
    )
    augmented_6f, augmented_path = _forward_select(
        ic,
        full_eligible,
        development_segments,
        start=F6,
        maximum_size=12,
    )

    horizon_paths = []
    for segment_count in (1, 2, 3):
        horizon_masks = development_segments[:segment_count]
        horizon_eligible = _eligible_factors(
            ic, valid_factors, horizon_masks
        )
        selected, path = _forward_select(
            ic, horizon_eligible, horizon_masks, maximum_size=8
        )
        horizon_paths.append({
            "through": ["2019", "2022", "2024"][segment_count - 1],
            "selected": selected,
            "path": path,
        })
    frequency = Counter(
        factor
        for horizon in horizon_paths
        for factor in horizon["selected"]
    )
    average_positions = {}
    for factor in frequency:
        positions = [
            horizon["selected"].index(factor)
            for horizon in horizon_paths
            if factor in horizon["selected"]
        ]
        average_positions[factor] = float(np.mean(positions))
    consensus = sorted(
        (factor for factor, count in frequency.items() if count >= 2),
        key=lambda factor: (-frequency[factor], average_positions[factor], factor),
    )[:8]

    factor_sets: dict[str, list[str]] = {
        "6f": F6,
        "10f": F10,
        "13f": F13,
        "14f": F14,
        "guosen_balanced_6r": GUOSEN_BALANCED_6R,
        "search_robust_full": robust_full,
        "8f": robust_post_2020,
        "search_augmented_6f": augmented_6f,
        "search_consensus": consensus,
    }
    factor_sets = {
        name: factors
        for name, factors in factor_sets.items()
        if factors
    }
    deduplicated: dict[tuple[str, ...], str] = {}
    unique_sets: dict[str, list[str]] = {}
    aliases = {}
    for name, factors in factor_sets.items():
        key = tuple(sorted(factors))
        if key in deduplicated:
            aliases[name] = deduplicated[key]
            continue
        deduplicated[key] = name
        unique_sets[name] = factors

    snapshot = {
        "as_of": str(end.date()),
        "valid_factor_count": len(valid_factors),
        "valid_factors": directions,
        "excluded_unvalidated_recent_factors": [
            "intraday_amt_ratio_entropy_60m_20d",
            "intraday_amt_ratio_entropy_trend_20d",
            "intraday_amt_ratio_entropy_volatility_20d",
        ],
        "development_segments": ["2016-2019", "2020-2022", "2023-2024"],
        "holdout_diagnostic": "2025-01-01 through comparison end",
        "full_eligible_count": len(full_eligible),
        "post_2020_eligible_count": len(post_2020_eligible),
        "factor_sets": unique_sets,
        "aliases": aliases,
        "selection_paths": {
            "robust_full": robust_full_path,
            "robust_post_2020": robust_post_path,
            "augmented_6f": augmented_path,
            "expanding_horizons": horizon_paths,
            "consensus_frequency": dict(frequency),
        },
    }
    (output / "resolved_factor_search_snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    diagnostics = []
    for name in valid_factors:
        row = {"factor": name, "direction": directions[name]}
        for label, mask in zip(
            ["2016_2019", "2020_2022", "2023_2024"],
            development_segments,
        ):
            values = ic.loc[mask, name]
            row[f"coverage_{label}"] = float(values.notna().mean())
            row[f"mean_ic_{label}"] = float(values.mean())
        diagnostics.append(row)
    pd.DataFrame(diagnostics).to_csv(
        output / "factor_development_diagnostics.csv", index=False
    )

    reference_nav = _load_reference(start, end)
    asset_returns = runner.daily_ret.reindex(columns=runner.u)
    contract_schedule = runner.get_contract_schedule()
    native_ledgers = {}
    equal_ledgers = {}
    metrics_rows = []
    period_starts = {
        "from_2016_03_31": start,
        "from_2020": pd.Timestamp("2020-01-01"),
    }
    for name, factors in unique_sets.items():
        print(f"exact production backtest: {name} ({len(factors)} factors)", flush=True)
        native_weights = _run_production_weights(runner, factors, end)
        native_ledger = _ledger_from_weights(
            native_weights,
            asset_returns,
            start,
            end,
            contract_schedule,
            runner.close_tradable,
        )
        gross = native_weights.abs().sum(axis=1)
        equal_weights = native_weights.div(gross.where(gross > 0.0), axis=0).fillna(0.0)
        equal_ledger = _ledger_from_weights(
            equal_weights,
            asset_returns,
            start,
            end,
            contract_schedule,
            runner.close_tradable,
        )
        native_ledgers[name] = native_ledger
        equal_ledgers[name] = equal_ledger
        native_dir = output / "production_gross_2"
        equal_dir = output / "equal_gross_1"
        native_dir.mkdir(exist_ok=True)
        equal_dir.mkdir(exist_ok=True)
        native_weights.loc[start:end].to_csv(
            native_dir / f"{name}_target_weights.csv"
        )
        native_ledger.to_csv(native_dir / f"{name}_ledger.csv")
        equal_weights.loc[start:end].to_csv(
            equal_dir / f"{name}_target_weights.csv"
        )
        equal_ledger.to_csv(equal_dir / f"{name}_ledger.csv")

    scenarios = {
        "production_gross_2": native_ledgers,
        "equal_gross_1": equal_ledgers,
    }
    for scenario, ledgers in scenarios.items():
        scenario_dir = output / scenario
        for period_name, period_start in period_starts.items():
            reference_period = reference_nav.loc[period_start:]
            reference_returns = reference_period.pct_change(
                fill_method=None
            ).fillna(0.0)
            metrics_rows.append({
                "scenario": scenario,
                "period": period_name,
                "strategy": "trend",
                **_metrics(reference_returns),
            })
            navs = {"trend": reference_nav}
            for name, ledger in ledgers.items():
                returns = ledger.loc[period_start:, "net_return"].copy()
                if len(returns):
                    returns.iloc[0] = 0.0
                row = {
                    "scenario": scenario,
                    "period": period_name,
                    "strategy": name,
                    **_metrics(returns),
                    "average_gross_exposure": float(
                        ledger.loc[period_start:, "gross_exposure"].replace(0.0, np.nan).mean()
                    ),
                    "annual_turnover": float(
                        ledger.loc[
                            period_start:, "executed_traded_notional"
                        ].iloc[1:].mean() * PERIODS_PER_YEAR
                    ),
                }
                metrics_rows.append(row)
                navs[name] = ledger["nav"]
            _plot(
                navs,
                period_start,
                f"生产 IC_IR/ERC/Top10-Bottom10：{scenario}，{period_name} 起",
                scenario_dir / f"nav_comparison_{period_name}.png",
            )
            focused_groups = {
                "baselines": ["trend", "6f", "10f", "13f", "14f"],
                "candidates": [
                    "trend",
                    "14f",
                    "search_robust_full",
                    "8f",
                    "search_augmented_6f",
                    "search_consensus",
                ],
            }
            for group_name, group_members in focused_groups.items():
                focused = {
                    name: navs[name] for name in group_members if name in navs
                }
                _plot(
                    focused,
                    period_start,
                    (
                        "生产 IC_IR/ERC/Top10-Bottom10："
                        f"{scenario}，{period_name}，{group_name}"
                    ),
                    scenario_dir
                    / f"nav_{group_name}_{period_name}.png",
                )
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(output / "comparison_metrics.csv", index=False)
    summary = {
        "output": str(output),
        "factor_sets": unique_sets,
        "aliases": aliases,
        "metrics": metrics.to_dict(orient="records"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
