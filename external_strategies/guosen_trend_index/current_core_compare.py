"""Compare 8f/10f/13f under production and fixed alternative recipes.

This entry point is research-only.  It computes one shared factor panel, keeps
the production configuration untouched, and writes only the evidence needed
for the requested current-date comparison.
"""
from __future__ import annotations

import argparse
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.metrics import TRADING_DAYS_PER_YEAR  # noqa: E402
from external_strategies.guosen_trend_index.production_compare import (  # noqa: E402
    _load_reference as _load_guosen_reference,
)
from research.historical_portfolio_search import (  # noqa: E402
    CausalEligibilityEnvironment,
    PortfolioEvaluator,
    PortfolioRecipe,
    performance_metrics,
)
from research.portfolio_experiment_support import (  # noqa: E402
    FACTORS_8F,
    FACTORS_10F as F10,
    FACTORS_13F as F13,
    FactorPanelRunner as Runner,
    configured_futures_cost_model,
    latest_local_date,
)


START = pd.Timestamp("2016-03-31")
RECENT_START = pd.Timestamp("2025-01-01")
COST_MODEL = configured_futures_cost_model()
PRODUCTION_RECIPE = PortfolioRecipe("lw_abs", 10, 3, "erc")
ALTERNATIVE_RECIPE = PortfolioRecipe(
    "equal",
    12,
    0,
    "inverse_volatility",
    asset_min_fraction=0.0,
    asset_max_fraction=1.0,
)
SENSITIVITY_TOP_N = (10, 12, 14)
SENSITIVITY_RISK_DAYS = (60, 90, 120)

STRATEGIES: Mapping[str, tuple[list[str], PortfolioRecipe]] = {
    "8f_ICIR_T10B10_cap3_ERC": (FACTORS_8F, PRODUCTION_RECIPE),
    "10f_ICIR_T10B10_cap3_ERC": (F10, PRODUCTION_RECIPE),
    "13f_ICIR_T10B10_cap3_ERC": (F13, PRODUCTION_RECIPE),
    "8f_equal_T12B12_no_cap_inverse_vol": (FACTORS_8F, ALTERNATIVE_RECIPE),
    "10f_equal_T12B12_no_cap_inverse_vol": (F10, ALTERNATIVE_RECIPE),
    "13f_equal_T12B12_no_cap_inverse_vol": (F13, ALTERNATIVE_RECIPE),
}

DISPLAY_NAMES = {
    "8f_ICIR_T10B10_cap3_ERC": "8f｜ICIR + T10/B10 + cap3 + ERC",
    "10f_ICIR_T10B10_cap3_ERC": "10f｜ICIR + T10/B10 + cap3 + ERC",
    "13f_ICIR_T10B10_cap3_ERC": "13f｜ICIR + T10/B10 + cap3 + ERC",
    "8f_equal_T12B12_no_cap_inverse_vol": "8f｜等权 + T12/B12 + 无cap + 逆波动",
    "10f_equal_T12B12_no_cap_inverse_vol": "10f｜等权 + T12/B12 + 无cap + 逆波动",
    "13f_equal_T12B12_no_cap_inverse_vol": "13f｜等权 + T12/B12 + 无cap + 逆波动",
    "guosen_trend": "国信趋势指数",
}


def _plot(
    navs: Mapping[str, pd.Series],
    *,
    title: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    reference_end: pd.Timestamp,
    output: Path,
) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(15, 8.5))
    colors = {
        "8f_ICIR_T10B10_cap3_ERC": "#54A24B",
        "10f_ICIR_T10B10_cap3_ERC": "#B279A2",
        "13f_ICIR_T10B10_cap3_ERC": "#F58518",
        "8f_equal_T12B12_no_cap_inverse_vol": "#E45756",
        "10f_equal_T12B12_no_cap_inverse_vol": "#72B7B2",
        "13f_equal_T12B12_no_cap_inverse_vol": "#FF9DA6",
        "guosen_trend": "#222222",
    }
    for name, full_nav in navs.items():
        nav = full_nav.loc[start:end].dropna()
        if nav.empty:
            continue
        nav = nav / nav.iloc[0] * 1000.0
        metrics = performance_metrics(
            nav.pct_change(fill_method=None).fillna(0.0),
            initial_anchor=True,
        )
        reference = name == "guosen_trend"
        suffix = f"，截至{reference_end:%Y-%m-%d}" if reference else ""
        ax.plot(
            nav.index,
            nav.values,
            color=colors[name],
            linestyle="--" if reference else "-",
            linewidth=2.2 if reference else 1.7,
            label=(
                f"{DISPLAY_NAMES[name]}{suffix}｜年化{metrics['annual_return']:.1%} "
                f"夏普{metrics['sharpe']:.2f} 回撤{metrics['max_drawdown']:.1%}"
            ),
        )

    ax.set_title(
        f"{title}（内部策略总敞口2；国信为发布净值原值，"
        f"{start:%Y-%m-%d} 至 {end:%Y-%m-%d}）"
    )
    ax.set_ylabel("净值（区间起点=1000）")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    ax.text(
        0.01,
        0.01,
        "内部策略：完整成交名义×0.02%＋总暴露×年化0.105%；"
        "国信：发布净值原值，未另行缩放或扣费。",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def validate_alternative_method(runner, output: Path, end: pd.Timestamp) -> None:
    """Check the 8f alternative around its fixed selection and risk windows."""

    rows = []
    canonical_weights = None
    for risk_days in SENSITIVITY_RISK_DAYS:
        evaluator = PortfolioEvaluator(
            runner,
            start=START,
            end=end,
            cost_model=COST_MODEL,
            risk_lookback_calendar_days=risk_days,
        )
        for top_n in SENSITIVITY_TOP_N:
            recipe = PortfolioRecipe(
                "equal",
                top_n,
                0,
                "inverse_volatility",
                asset_min_fraction=0.0,
                asset_max_fraction=1.0,
            )
            weights = evaluator.weights(FACTORS_8F, recipe)
            ledger = evaluator.ledger_from_weights(weights)
            if risk_days == 90 and top_n == 12:
                canonical_weights = weights
            for period, start in (
                ("from_2016_03_31", START),
                ("from_2025_01_01", RECENT_START),
            ):
                returns = ledger.loc[start:end, "net_return"].copy()
                if len(returns):
                    returns.iloc[0] = 0.0
                rows.append({
                    "period": period,
                    "top_n_per_side": top_n,
                    "risk_lookback_calendar_days": risk_days,
                    **performance_metrics(returns, initial_anchor=True),
                    "annual_turnover": float(
                        ledger.loc[
                            start:end, "executed_traded_notional"
                        ].iloc[1:].mean()
                        * TRADING_DAYS_PER_YEAR
                    ),
                })
    pd.DataFrame(rows).to_csv(output / "method_sensitivity.csv", index=False)

    if canonical_weights is None:
        raise RuntimeError("canonical alternative weights were not produced")
    active = canonical_weights.loc[canonical_weights.abs().sum(axis=1) > 0.0]
    abs_weights = active.abs()
    side_effective = []
    side_max_sector = []
    for sign in (1, -1):
        side = active.clip(lower=0.0) if sign > 0 else (-active.clip(upper=0.0))
        side_total = side.sum(axis=1).replace(0.0, np.nan)
        normalized = side.div(side_total, axis=0)
        side_effective.append(1.0 / normalized.pow(2).sum(axis=1))
        sectors = sorted(set(runner.env.sector_of.values()))
        sector_shares = pd.DataFrame({
            sector: normalized.reindex(
                columns=[
                    symbol
                    for symbol, value in runner.env.sector_of.items()
                    if value == sector
                ],
                fill_value=0.0,
            ).sum(axis=1)
            for sector in sectors
        })
        side_max_sector.append(sector_shares.max(axis=1))

    trailing_vol = (
        runner.daily_ret.reindex(index=active.index, columns=active.columns)
        .shift(1)
        .rolling(60, min_periods=20)
        .std(ddof=0)
    )
    low_vol = trailing_vol.rank(axis=1, pct=True).le(0.25)
    low_vol_share = abs_weights.where(low_vol, 0.0).sum(axis=1).div(
        abs_weights.sum(axis=1).replace(0.0, np.nan)
    )
    concentration = pd.DataFrame({
        "metric": [
            "mean_max_single_asset_abs_weight",
            "p95_max_single_asset_abs_weight",
            "mean_effective_assets_per_side",
            "p05_effective_assets_per_side",
            "mean_max_sector_share_per_side",
            "p95_max_sector_share_per_side",
            "mean_lowest_vol_quartile_gross_share",
            "p95_lowest_vol_quartile_gross_share",
        ],
        "value": [
            abs_weights.max(axis=1).mean(),
            abs_weights.max(axis=1).quantile(0.95),
            pd.concat(side_effective).mean(),
            pd.concat(side_effective).quantile(0.05),
            pd.concat(side_max_sector).mean(),
            pd.concat(side_max_sector).quantile(0.95),
            low_vol_share.mean(),
            low_vol_share.quantile(0.95),
        ],
    })
    concentration.to_csv(output / "8f_alternative_concentration.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default=None, help="Defaults to latest local daily date")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--validate-method",
        action="store_true",
        help="also write the focused 8f alternative sensitivity and concentration audit",
    )
    args = parser.parse_args()

    latest = pd.Timestamp(latest_local_date())
    end = pd.Timestamp(args.end) if args.end else latest
    if end < START or end > latest:
        raise ValueError(f"end must be within {START.date()}..{latest.date()}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    factor_names = list(dict.fromkeys(F10 + F13 + FACTORS_8F))
    print(f"compute shared panel: {len(factor_names)} factors, end={end:%Y-%m-%d}", flush=True)
    runner = Runner(factor_names)
    runner.get_contract_schedule()
    runner.env = CausalEligibilityEnvironment(
        runner.cal, runner.daily_ret, runner.env.sector_of
    )
    evaluator = PortfolioEvaluator(
        runner,
        start=START,
        end=end,
        cost_model=COST_MODEL,
    )

    navs: dict[str, pd.Series] = {}
    returns: dict[str, pd.Series] = {}
    metrics_rows = []
    for name, (factors, recipe) in STRATEGIES.items():
        print(f"evaluate {name}", flush=True)
        ledger = evaluator.ledger(factors, recipe)
        if name == "10f_ICIR_T10B10_cap3_ERC":
            evaluator.weights(factors, recipe).to_csv(
                output / "10f_target_weights.csv"
            )
            ledger.to_csv(output / "10f_ledger.csv")
        navs[name] = ledger["nav"]
        returns[name] = ledger["net_return"]
        for period_name, period_start in (
            ("from_2016_03_31", START),
            ("from_2025_01_01", RECENT_START),
        ):
            period = ledger.loc[period_start:end]
            period_returns = period["net_return"].copy()
            if len(period_returns):
                # Match interval-normalized charts and the established
                # production comparator: the first visible point is the base.
                period_returns.iloc[0] = 0.0
            metrics_rows.append({
                "period": period_name,
                "strategy": name,
                **performance_metrics(period_returns, initial_anchor=True),
                "average_gross_exposure": float(
                    period["gross_exposure"].replace(0.0, np.nan).mean()
                ),
                "annual_turnover": float(
                    period["executed_traded_notional"].iloc[1:].mean()
                    * TRADING_DAYS_PER_YEAR
                ),
                "data_end": str(period.index.max().date()),
            })

    reference = _load_guosen_reference(START, end)
    if reference.empty:
        raise ValueError("Guosen reference has no observations in comparison interval")
    reference_end = pd.Timestamp(reference.index.max())
    navs["guosen_trend"] = reference
    for period_name, period_start in (
        ("from_2016_03_31", START),
        ("from_2025_01_01", RECENT_START),
    ):
        reference_period = reference.loc[period_start:].dropna()
        period_returns = reference_period.pct_change(
            fill_method=None
        ).fillna(0.0)
        metrics_rows.append({
            "period": period_name,
            "strategy": "guosen_trend",
            **performance_metrics(period_returns, initial_anchor=True),
            "average_gross_exposure": np.nan,
            "annual_turnover": np.nan,
            "data_end": str(reference_end.date()),
        })

    evidence = pd.DataFrame(index=evaluator.dates)
    for name in STRATEGIES:
        evidence[f"{name}__net_return"] = returns[name]
        evidence[f"{name}__nav"] = navs[name]
    evidence["guosen_trend__nav"] = reference.reindex(evidence.index)
    evidence.to_csv(output / "strategy_nav_and_returns.csv")
    pd.DataFrame(metrics_rows).to_csv(output / "comparison_metrics.csv", index=False)
    if args.validate_method:
        print("validate alternative method neighbourhood and concentration", flush=True)
        validate_alternative_method(runner, output, end)

    production_navs = {
        name: navs[name] for name in STRATEGIES if name.endswith("_ICIR_T10B10_cap3_ERC")
    }
    production_navs["guosen_trend"] = navs["guosen_trend"]
    strategy_navs = {name: navs[name] for name in STRATEGIES}
    strategy_navs["guosen_trend"] = navs["guosen_trend"]
    _plot(
        production_navs,
        title="8f/10f/13f生产结构＋国信对比",
        start=START,
        end=end,
        reference_end=reference_end,
        output=output / "nav_production_20160331_to_latest.png",
    )
    _plot(
        production_navs,
        title="8f/10f/13f生产结构＋国信对比",
        start=RECENT_START,
        end=end,
        reference_end=reference_end,
        output=output / "nav_production_20250101_to_latest.png",
    )
    _plot(
        strategy_navs,
        title="8f/10f/13f两种方法＋国信对比",
        start=START,
        end=end,
        reference_end=reference_end,
        output=output / "nav_six_methods_20160331_to_latest.png",
    )
    _plot(
        strategy_navs,
        title="8f/10f/13f两种方法＋国信对比",
        start=RECENT_START,
        end=end,
        reference_end=reference_end,
        output=output / "nav_six_methods_20250101_to_latest.png",
    )

    resolved = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "strategy_data_end": str(end.date()),
        "guosen_data_end": str(reference_end.date()),
        "internal_gross_exposure": 2.0,
        **COST_MODEL.ledger_parameters(),
        "guosen_reference_policy": "published_nav_unscaled_no_extra_cost",
        "strategies": {
            name: {"factors": factors, "recipe": recipe.to_dict()}
            for name, (factors, recipe) in STRATEGIES.items()
        },
        "production_configuration_modified": False,
        "method_validation_included": bool(args.validate_method),
    }
    (output / "resolved_config.json").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "REVIEW.md").write_text(
        "# 当前核心方案比较\n\n"
        f"- 策略数据截止：{end:%Y-%m-%d}\n"
        f"- 国信净值截止：{reference_end:%Y-%m-%d}（未向后填充）\n"
        "- 内部策略口径：38品种、日调仓、总敞口2、完整成交名义×0.02%，并按总暴露摊销年化0.105%。\n"
        "- 国信曲线为发布净值原值，未做杠杆缩放或额外扣费；不宣称与内部策略同杠杆、同成本。\n"
        "- 正确性口径：完整字母根精确匹配、决策日前60条IC、最终资产上限投影、收益日按W[T-1]×R[T]记账（等价于收盘目标W[T]从T+1生效）、漂移换手、显式移仓及停牌冻结。\n"
        "- 8f、10f及13f生产结构均为60日ICIR(LW)、Top10/Bottom10、单侧cap3、两侧ERC。\n"
        "- 三个因子集的替代结构均为因子等权、Top12/Bottom12、无cap、两侧逆波动率。\n"
        "- 输出两张生产结构及两张双方法对比图，均同时包含国信趋势指数，并覆盖全期和2025年以来区间。\n"
        "- 修复前标注的OOS/实盘日期不再视为独立前瞻证据，因此图中不作此类着色。\n"
        "- 本运行是已知固定因子集重评估，不是修复后重新完成的因子搜索；不会自动修改配置、快照或交易批准门。\n",
        encoding="utf-8",
    )
    print(json.dumps(resolved, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
