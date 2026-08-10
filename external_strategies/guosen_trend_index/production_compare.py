from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
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
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from exp_core import PROD6  # noqa: E402
from exp18_light_forward import DIRS, KEPT47  # noqa: E402
from exp22_full_pool import (  # noqa: E402
    NEW21,
    NEW21_DIR,
    Runner,
    prepare_ic_history,
)


PERIODS_PER_YEAR = 242
TRADE_COST_RATE = 0.0002
ANNUAL_FEE = 0.001
REFERENCE_NAV = Path(r"E:\程明杰公司内容\GSIXTREND.WI_nav.csv")

F6 = list(PROD6)
F14 = F6 + [
    "intraday_ma_count_bullish_20d",
    "intraday_torrent_down_20d",
    "intraday_lowest_time_20d",
    "intraday_term_slope_20d",
    "intraday_open_close_volume_ratio_20d",
    "intraday_seat_long_short_seat_ratio_20d",
    "intraday_turnover_velocity_20d",
    "intraday_price_delay_20d",
]
F13 = [name for name in F14 if name != "intraday_ma_count_bullish_20d"]
F10 = [
    name for name in F13
    if name not in {
        "intraday_turnover_velocity_20d",
        "intraday_open_close_volume_ratio_20d",
        "intraday_seat_long_short_seat_ratio_20d",
    }
]
GUOSEN_BALANCED_6R = [
    "intraday_jump_intensity_20d",
    "intraday_dtws_20d",
    "intraday_drip_stone_20d",
    "intraday_lowest_time_20d",
    "intraday_term_slope_20d",
    "intraday_price_delay_20d",
]


def _lw_cov(ic_matrix: pd.DataFrame) -> np.ndarray:
    rows, columns = ic_matrix.shape
    sample_cov = np.cov(ic_matrix, rowvar=False, ddof=1)
    sample_corr = np.corrcoef(ic_matrix, rowvar=False)
    average_corr = (
        np.mean(sample_corr[np.triu_indices(columns, k=1)])
        if columns > 1 else 0.0
    )
    target_corr = (
        np.eye(columns) * (1.0 - average_corr)
        + np.ones((columns, columns)) * average_corr
    )
    std = np.std(ic_matrix, axis=0, ddof=1)
    target_cov = np.outer(std, std) * target_corr
    centered = ic_matrix - ic_matrix.mean(axis=0)
    pi = sum(
        np.sum(
            (
                centered.iloc[index].to_numpy()[:, None]
                @ centered.iloc[index].to_numpy()[None, :]
                - sample_cov
            ) ** 2
        )
        for index in range(rows)
    ) / rows
    gamma = np.sum((target_cov - sample_cov) ** 2)
    shrinkage = max(0.0, min(1.0, pi / gamma)) if gamma > 0.0 else 0.5
    return shrinkage * target_cov + (1.0 - shrinkage) * sample_cov


def _factor_weights(history: pd.DataFrame) -> pd.Series:
    if history.shape[1] < 2:
        return pd.Series(dtype=float)
    if len(history) < 30:
        return pd.Series(1.0 / history.shape[1], index=history.columns)
    mean_ic = history.mean()
    covariance = _lw_cov(history)
    try:
        raw = np.linalg.solve(covariance, mean_ic.to_numpy())
    except np.linalg.LinAlgError:
        raw = mean_ic.abs().to_numpy()
    raw = np.abs(raw)
    total = float(raw.sum())
    if not np.isfinite(total) or total <= 0.0:
        return pd.Series(1.0 / history.shape[1], index=history.columns)
    return pd.Series(raw / total, index=history.columns)


def _run_production_weights(
    runner: Runner,
    factor_names: list[str],
    end: pd.Timestamp,
) -> pd.DataFrame:
    missing = sorted(set(factor_names) - set(runner.ranks))
    if missing:
        raise KeyError(f"factor runner did not compute: {missing}")
    dates = runner.cal[runner.cal <= end]
    weights = pd.DataFrame(0.0, index=dates, columns=runner.u)
    ic = runner.ic[factor_names]
    for index, date in enumerate(dates):
        history = prepare_ic_history(ic.loc[:date].iloc[-60:-1])
        factor_weights = _factor_weights(history)
        if factor_weights.empty:
            continue
        score = pd.Series(0.0, index=runner.u)
        for name, factor_weight in factor_weights.items():
            score = score.add(
                runner.ranks[name].loc[date].fillna(0.0) * factor_weight,
                fill_value=0.0,
            )
        total = float(score.sum())
        if total > 0.0:
            score /= total
        score = score.dropna()
        if len(score) < 20:
            continue
        long_pool = runner.env.capped(score, ascending=False, date=date)
        short_pool = runner.env.capped(score, ascending=True, date=date)
        long_weights = runner.env.erc_w(long_pool, date) or {}
        short_weights = runner.env.erc_w(short_pool, date) or {}
        if not long_weights or not short_weights:
            continue
        for symbol, value in long_weights.items():
            weights.loc[date, symbol] += value
        for symbol, value in short_weights.items():
            weights.loc[date, symbol] -= value
        if (index + 1) % 500 == 0:
            print(
                f"  {','.join(factor_names[:2])}... weights {index + 1}/{len(dates)}",
                flush=True,
            )
    return weights


def _ledger_from_weights(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    weights = weights.loc[:end].fillna(0.0)
    gross_return = (
        weights * asset_returns.reindex(index=weights.index, columns=weights.columns)
    ).sum(axis=1, min_count=1).fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1)
    if len(turnover):
        turnover.iloc[0] = float(weights.iloc[0].abs().sum())
    trade_cost = turnover * TRADE_COST_RATE
    fee = pd.Series(ANNUAL_FEE / PERIODS_PER_YEAR, index=weights.index)
    net_return = gross_return - trade_cost - fee
    ledger = pd.DataFrame({
        "gross_return": gross_return,
        "turnover": turnover,
        "trade_cost": trade_cost,
        "management_fee": fee,
        "net_return": net_return,
        "gross_exposure": weights.abs().sum(axis=1),
        "net_exposure": weights.sum(axis=1),
    }).loc[start:end]
    if len(ledger):
        ledger.iloc[0, ledger.columns.get_loc("gross_return")] = 0.0
        ledger.iloc[0, ledger.columns.get_loc("turnover")] = 0.0
        ledger.iloc[0, ledger.columns.get_loc("trade_cost")] = 0.0
        ledger.iloc[0, ledger.columns.get_loc("management_fee")] = 0.0
        ledger.iloc[0, ledger.columns.get_loc("net_return")] = 0.0
    ledger["nav"] = 1000.0 * (1.0 + ledger["net_return"]).cumprod()
    return ledger


def _metrics(returns: pd.Series) -> dict[str, float | int]:
    values = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 2:
        return {"observations": int(len(values))}
    growth = float((1.0 + values).prod())
    annual_return = (
        growth ** (PERIODS_PER_YEAR / len(values)) - 1.0
        if growth > 0.0 else -1.0
    )
    annual_volatility = float(values.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))
    nav = (1.0 + values).cumprod()
    return {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": (
            float(values.mean() * PERIODS_PER_YEAR / annual_volatility)
            if annual_volatility > 0.0 else 0.0
        ),
        "max_drawdown": float((nav / nav.cummax() - 1.0).min()),
        "total_return": growth - 1.0,
        "observations": int(len(values)),
    }


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


def _load_reference(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    frame = pd.read_csv(REFERENCE_NAV, parse_dates=["date"])
    nav = frame.set_index("date")["nav"].sort_index().loc[start:end]
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
        description="Compare production IC_IR/ERC long-short portfolios with GSIXTREND.WI"
    )
    parser.add_argument("--start", default="2016-03-31")
    parser.add_argument("--end", default="2026-08-06")
    parser.add_argument("--output")
    args = parser.parse_args()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    output = Path(args.output) if args.output else (
        ROOT / "runs" / "external_guosen_trend_index"
        / f"{datetime.now():%Y%m%d_%H%M%S}_production_compare"
    )
    output.mkdir(parents=True, exist_ok=False)

    valid_factors = list(dict.fromkeys(F6 + KEPT47 + NEW21))
    directions = {
        name: int(DIRS.get(name, NEW21_DIR.get(name, PROD6.get(name, 1))))
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
        "search_robust_post2020": robust_post_2020,
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
        "as_of": "2026-08-09",
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
            native_weights, asset_returns, start, end
        )
        gross = native_weights.abs().sum(axis=1)
        equal_weights = native_weights.div(gross.where(gross > 0.0), axis=0).fillna(0.0)
        equal_ledger = _ledger_from_weights(
            equal_weights, asset_returns, start, end
        )
        native_ledgers[name] = native_ledger
        equal_ledgers[name] = equal_ledger
        native_dir = output / "production_gross_2"
        equal_dir = output / "equal_gross_1"
        native_dir.mkdir(exist_ok=True)
        equal_dir.mkdir(exist_ok=True)
        native_weights.loc[start:end].to_csv(native_dir / f"{name}_weights.csv")
        native_ledger.to_csv(native_dir / f"{name}_ledger.csv")
        equal_weights.loc[start:end].to_csv(equal_dir / f"{name}_weights.csv")
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
                        ledger.loc[period_start:, "turnover"].mean() * PERIODS_PER_YEAR
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
                    "search_robust_post2020",
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
