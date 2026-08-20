from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.config import load_config
from backtest.metrics import TRADING_DAYS_PER_YEAR
from factors.engine import FactorEngine
from pipeline.runner import PipelineRunner
from research.historical_portfolio_search import performance_metrics

from .robustness import exhaustive_subset_search
from .strategy import ExternalBacktestResult, GuosenTrendIndexBacktester, load_snapshot


HERE = Path(__file__).resolve().parent
METRIC_PERIODS_PER_YEAR = TRADING_DAYS_PER_YEAR


def _metrics(
    returns: pd.Series,
    periods_per_year: int,
    weights: pd.DataFrame | None = None,
    turnover: pd.Series | None = None,
) -> dict[str, float | int]:
    result = performance_metrics(
        returns,
        periods_per_year=periods_per_year,
        initial_anchor=True,
    )
    interval_index = returns.index[1:]
    if weights is not None:
        gross = weights.reindex(interval_index).abs().sum(axis=1)
        active_gross = gross[gross.gt(0.0)]
        result.update({
            "average_gross_exposure": float(active_gross.mean()) if len(active_gross) else 0.0,
            "median_gross_exposure": float(active_gross.median()) if len(active_gross) else 0.0,
            "maximum_gross_exposure": float(active_gross.max()) if len(active_gross) else 0.0,
        })
    if turnover is not None:
        traded = turnover.reindex(interval_index)
        if traded.isna().any() or not np.isfinite(traded.to_numpy(dtype=float)).all():
            raise ValueError("turnover must be finite and cover all return intervals")
        result["annual_turnover"] = float(
            traded.mean() * periods_per_year
        )
    return result


def _trim_to_base_date(
    result: ExternalBacktestResult,
    start: pd.Timestamp,
    base_value: float = 1000.0,
) -> ExternalBacktestResult:
    returns = result.returns.loc[start:].copy()
    gross_returns = result.gross_returns.reindex(returns.index).copy()
    turnover = result.turnover.reindex(returns.index).copy()
    costs = result.costs.reindex(returns.index).copy()
    if len(returns):
        returns.iloc[0] = 0.0
        gross_returns.iloc[0] = 0.0
        turnover.iloc[0] = 0.0
        costs.iloc[0] = 0.0
    nav = base_value * (1.0 + returns).cumprod()
    nav.name = "index_level"
    return ExternalBacktestResult(
        nav=nav,
        returns=returns,
        gross_returns=gross_returns,
        turnover=turnover,
        costs=costs,
        weights=result.weights.reindex(returns.index),
        diagnostics=result.diagnostics.reindex(returns.index),
    )


def _load_reference(path: str | Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=["date"])
    if set(frame.columns) != {"date", "nav"}:
        raise ValueError("reference NAV must contain exactly date and nav columns")
    if (
        frame["date"].duplicated().any()
        or frame["nav"].isna().any()
        or not np.isfinite(frame["nav"]).all()
    ):
        raise ValueError("reference NAV contains duplicate dates or non-finite values")
    nav = frame.set_index("date")["nav"].sort_index().loc[start:end]
    if nav.empty or (nav <= 0.0).any():
        raise ValueError("reference NAV has no positive observations in the requested range")
    if pd.Timestamp(nav.index[0]) != start:
        raise ValueError(f"reference NAV does not cover requested start {start.date()}")
    return nav / nav.iloc[0] * 1000.0


def _period_returns_from_nav(nav: pd.Series, start: pd.Timestamp) -> pd.Series:
    rebased = nav.loc[start:].copy()
    returns = rebased.pct_change(fill_method=None).fillna(0.0)
    if len(returns):
        returns.iloc[0] = 0.0
    return returns


def _plot_comparison(
    navs: dict[str, pd.Series],
    start: pd.Timestamp,
    periods_per_year: int,
    title: str,
    output: Path,
) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(14, 8))
    for label, full_nav in navs.items():
        nav = full_nav.loc[start:].dropna()
        if nav.empty:
            continue
        nav = nav / nav.iloc[0] * 1000.0
        metrics = _metrics(
            nav.pct_change(fill_method=None).fillna(0.0), periods_per_year
        )
        style = {"color": "black", "linestyle": "--", "linewidth": 2.0} if label == "trend" else {}
        ax.plot(
            nav.index,
            nav.values,
            label=(
                f"{label} | 年化{metrics['annual_return']:.1%} "
                f"夏普{metrics['sharpe']:.2f} 回撤{metrics['max_drawdown']:.1%}"
            ),
            **style,
        )
    ax.set_title(title)
    ax.set_ylabel("净值（区间起点=1000）")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _write_result(result: ExternalBacktestResult, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    result.nav.to_csv(output / f"{name}_nav.csv")
    result.weights.to_csv(output / f"{name}_effective_weights.csv")
    pd.concat(
        [
            result.returns,
            result.gross_returns,
            result.turnover,
            result.costs,
            result.weights.abs().sum(axis=1).rename("gross_exposure"),
        ],
        axis=1,
    ).join(result.diagnostics).to_csv(output / f"{name}_ledger.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the isolated Guosen index-style adapter")
    parser.add_argument("--start", default="2016-03-31")
    parser.add_argument("--end", required=True)
    parser.add_argument("--factor-set", action="append", dest="factor_sets")
    parser.add_argument("--snapshot", default=str(HERE / "config.yaml"))
    parser.add_argument("--framework-config", default="config/intraday_backtest.yaml")
    parser.add_argument("--equal-gross", type=float, default=1.0)
    parser.add_argument("--search-subsets", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    spec, available_sets, raw_snapshot = load_snapshot(args.snapshot)
    selected_sets = args.factor_sets or list(available_sets)
    unknown = sorted(set(selected_sets) - set(available_sets))
    if unknown:
        raise ValueError(f"unknown factor sets: {unknown}; available={sorted(available_sets)}")
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if end < start:
        raise ValueError("end must be on or after start")

    all_groups = {}
    for set_name in selected_sets:
        for group_name, variants in available_sets[set_name].items():
            existing = all_groups.get(group_name)
            if existing is not None and existing != variants:
                raise ValueError(f"factor group {group_name!r} has conflicting definitions")
            all_groups[group_name] = variants
    framework_config = load_config(args.framework_config)
    runner = PipelineRunner(config=framework_config)
    manager = runner.data_manager
    engine = FactorEngine(manager)
    warmup_start = start - pd.Timedelta(days=spec.warmup_calendar_days)
    dates = pd.DatetimeIndex(manager.get_calendar(warmup_start, end))
    close = manager.get("close", dates, list(spec.universe))
    if close is None or close.empty:
        raise RuntimeError("no close data for external strategy universe")
    schedule_getter = getattr(manager.source, "fetch_contract_schedule", None)
    contract_schedule = (
        schedule_getter(list(spec.universe), dates.min(), dates.max())
        if callable(schedule_getter)
        else None
    )

    adapter = GuosenTrendIndexBacktester(
        manager,
        engine,
        spec,
        framework_config.data.audited_nontrading_closes,
    )
    factor_names = list(dict.fromkeys(
        factor for variants in all_groups.values() for factor, _ in variants
    ))
    factor_values = adapter.compute_factor_values(factor_names, dates)
    portfolios = adapter.build_factor_portfolios(factor_values, all_groups, close)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    reference_path = raw_snapshot.get("snapshot", {}).get("reference_nav")
    if not reference_path:
        raise ValueError("snapshot.reference_nav is required for comparison output")
    reference_nav = _load_reference(reference_path, start, end)

    scenario_results: dict[str, dict[str, ExternalBacktestResult]] = {
        "native_target_vol_4pct": {},
        f"equal_gross_{args.equal_gross:g}": {},
    }
    for set_name in selected_sets:
        subset = {name: portfolios[name] for name in available_sets[set_name]}
        native_weights, diagnostics = adapter.combine_factor_portfolios(subset)
        native = _trim_to_base_date(
            adapter.run_from_weights(
                native_weights,
                close,
                diagnostics,
                contract_schedule=contract_schedule,
            ),
            start,
        )
        equal_weights = adapter.project_weights_to_gross(
            native_weights, args.equal_gross
        )
        equal = _trim_to_base_date(
            adapter.run_from_weights(
                equal_weights,
                close,
                diagnostics,
                contract_schedule=contract_schedule,
            ),
            start,
        )
        scenario_results["native_target_vol_4pct"][set_name] = native
        scenario_results[f"equal_gross_{args.equal_gross:g}"][set_name] = equal

    period_starts = {
        "from_2016_03_31": start,
        "from_2020": max(start, pd.Timestamp("2020-01-01")),
    }
    summary: dict[str, dict] = {}
    metric_rows = []
    for scenario, results in scenario_results.items():
        scenario_dir = output / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)
        navs = {"trend": reference_nav}
        for set_name, result in results.items():
            _write_result(result, scenario_dir, set_name)
            navs[set_name] = result.nav
        summary[scenario] = {}
        for period_name, period_start in period_starts.items():
            summary[scenario][period_name] = {}
            reference_returns = _period_returns_from_nav(reference_nav, period_start)
            reference_metrics = _metrics(
                reference_returns, METRIC_PERIODS_PER_YEAR
            )
            reference_metrics["gross_exposure_note"] = "actual index; user-estimated near 1x"
            summary[scenario][period_name]["trend"] = reference_metrics
            metric_rows.append({
                "scenario": scenario,
                "period": period_name,
                "strategy": "trend",
                **reference_metrics,
            })
            for set_name, result in results.items():
                period_returns = result.returns.loc[period_start:].copy()
                if len(period_returns):
                    period_returns.iloc[0] = 0.0
                metrics = _metrics(
                    period_returns,
                    METRIC_PERIODS_PER_YEAR,
                    result.weights,
                    result.turnover,
                )
                summary[scenario][period_name][set_name] = metrics
                metric_rows.append({
                    "scenario": scenario,
                    "period": period_name,
                    "strategy": set_name,
                    **metrics,
                })
            _plot_comparison(
                navs,
                period_start,
                METRIC_PERIODS_PER_YEAR,
                (
                    f"国信趋势指数形式：{scenario}，{period_name} 起"
                ),
                scenario_dir / f"nav_comparison_{period_name}.png",
            )

    pd.DataFrame(metric_rows).to_csv(output / "comparison_metrics.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "resolved_snapshot.json").write_text(
        json.dumps(raw_snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.search_subsets:
        baseline_sets = {
            name: set(available_sets[name]) for name in selected_sets
        }
        search = exhaustive_subset_search(
            portfolios,
            close,
            spec,
            start,
            end,
            baseline_sets,
            output / "factor_subset_search",
            target_gross=args.equal_gross,
            contract_schedule=contract_schedule,
            audited_nontrading_closes=(
                framework_config.data.audited_nontrading_closes
            ),
        )
        summary["factor_subset_search"] = {
            "ranking_rule": "2016-2024 development segments only; 2025+ is holdout",
            "top10": search.head(10).to_dict(orient="records"),
            "baselines": search.loc[search["baseline_label"].ne("")].to_dict(orient="records"),
        }
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps({"output": str(output), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
