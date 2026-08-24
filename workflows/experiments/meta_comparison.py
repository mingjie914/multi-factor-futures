"""Run sleeves once and compare robust meta-allocation methods."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from core.config import load_config
from pipeline.runner import PipelineRunner


DEFAULT_METHODS = [
    "inverse_volatility",
    "shrinkage_min_variance",
    "risk_parity",
    "hrp",
    "max_sharpe",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--alpha-type",
        choices=["sector_grouped_ols", "sector_grouped_ridge"],
        default=None,
    )
    parser.add_argument(
        "--methods", default=",".join(DEFAULT_METHODS), help="Comma-separated methods"
    )
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    from core.date_policy import apply_research_end

    if args.start:
        config.date_range.start = args.start
    apply_research_end(config, args.end)
    if args.alpha_type:
        config.alpha.type = args.alpha_type
        params = dict(config.alpha.params)
        if args.alpha_type == "sector_grouped_ridge":
            params.update({
                "ridge_alphas": [0.01, 0.1, 1.0, 10.0],
                "ridge_cv_folds": 3,
            })
        else:
            params.pop("ridge_alphas", None)
            params.pop("ridge_cv_folds", None)
            params["ridge_alpha"] = 0.0
        config.alpha.params = params
    config.backtest.report_dir = str(output)

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    config.meta_optimizer.method = "shrinkage_min_variance"
    runner = PipelineRunner(config=config)
    base = runner.run_multi_portfolio()

    configured_weights = [item.capital_weight for item in config.sub_portfolios]
    results = {
        "fixed_configured": runner.recombine_multi_portfolio(
            base, fixed_weights=configured_weights
        )
    }
    for method in methods:
        results[method] = (
            base
            if method == "shrinkage_min_variance"
            else runner.recombine_multi_portfolio(base, method=method)
        )

    rows = []
    navs = {}
    for name, result in results.items():
        metadata = {
            "experiment": "meta_allocation_comparison",
            "variant": name,
            "start": config.date_range.start,
            "end": config.date_range.end,
            "alpha_type": config.alpha.type,
            "evidence_level": "historical_cache_segment_diagnostic",
        }
        result.save(output / name, metadata=metadata)
        metrics = result.combined_result.metrics
        split = result.combined_result.split_metrics.get("test", {})
        row = {
            "variant": name,
            "annual_return": metrics.get("annual_return", 0.0),
            "sharpe": metrics.get("sharpe", 0.0),
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "volatility": metrics.get("volatility", 0.0),
            "calmar": metrics.get("calmar", 0.0),
            "avg_turnover": metrics.get("avg_turnover", 0.0),
            "total_transaction_cost": metrics.get("total_transaction_cost", 0.0),
            "segment_test_sharpe": split.get("sharpe", 0.0),
            "failure_count": len(result.combined_result.failure_ledger),
        }
        for item in result.sub_configs:
            row[f"avg_weight_{item['name']}"] = item["capital_weight"]
        rows.append(row)
        navs[name] = result.combined_result.nav

    comparison = pd.DataFrame(rows).set_index("variant")
    comparison.to_csv(output / "comparison.csv", encoding="utf-8-sig")
    (output / "comparison.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(navs).to_csv(output / "nav_comparison.csv")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        nav_frame = pd.DataFrame(navs).dropna(how="all")
        normalised = nav_frame.div(nav_frame.iloc[0])
        ax = normalised.plot(figsize=(13, 7), linewidth=1.4)
        ax.set_title("Meta-allocation comparison (historical cache diagnostic)")
        ax.set_ylabel("Normalised NAV")
        ax.grid(alpha=0.25)
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(output / "nav_comparison.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass

    print(comparison.to_string(float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
