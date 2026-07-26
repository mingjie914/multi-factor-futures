"""Compare alpha and sector-selection variants on one consistent sample."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd

from core.config import load_config
from pipeline.runner import PipelineRunner


VARIANTS = {
    "sector_grouped_ols": {
        "alpha_type": "sector_grouped_ols",
        "asset_selection": False,
    },
    "sector_grouped_ridge": {
        "alpha_type": "sector_grouped_ridge",
        "asset_selection": False,
    },
    "ridge_hysteresis_top_n": {
        "alpha_type": "sector_grouped_ridge",
        "asset_selection": True,
    },
}


def _configure(base, variant: str):
    config = copy.deepcopy(base)
    settings = VARIANTS[variant]
    config.alpha.type = settings["alpha_type"]
    params = dict(config.alpha.params)
    if settings["alpha_type"] == "sector_grouped_ridge":
        params.update({
            "ridge_alphas": [0.01, 0.1, 1.0, 10.0],
            "ridge_cv_folds": 3,
        })
    else:
        params.pop("ridge_alphas", None)
        params.pop("ridge_cv_folds", None)
        params["ridge_alpha"] = 0.0
    config.alpha.params = params
    config.asset_selection.enabled = settings["asset_selection"]
    config.asset_selection.mode = "hysteresis_top_n"
    config.asset_selection.top_n_per_side = 2
    config.asset_selection.exit_buffer = 1
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument(
        "--variants", default=",".join(VARIANTS), help="Comma-separated variants"
    )
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base = load_config(args.config)
    if args.start:
        base.date_range.start = args.start
    if args.end:
        base.date_range.end = args.end
    if args.cache_only:
        base.data.cache["only"] = True

    requested = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")

    rows = []
    navs = {}
    for variant in requested:
        config = _configure(base, variant)
        variant_dir = output / variant
        config.backtest.report_dir = str(variant_dir)
        result = PipelineRunner(config=config).run_multi_portfolio()
        result.save(
            variant_dir,
            metadata={
                "experiment": "alpha_and_sector_selection_comparison",
                "variant": variant,
                "start": config.date_range.start,
                "end": config.date_range.end,
                "cache_only": bool(args.cache_only),
                "evidence_level": "historical_cache_segment_diagnostic",
            },
        )
        metrics = result.combined_result.metrics
        split = result.combined_result.split_metrics.get("test", {})
        rows.append({
            "variant": variant,
            "annual_return": metrics.get("annual_return", 0.0),
            "sharpe": metrics.get("sharpe", 0.0),
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "volatility": metrics.get("volatility", 0.0),
            "calmar": metrics.get("calmar", 0.0),
            "avg_turnover": metrics.get("avg_turnover", 0.0),
            "total_transaction_cost": metrics.get("total_transaction_cost", 0.0),
            "segment_test_sharpe": split.get("sharpe", 0.0),
            "failure_count": len(result.combined_result.failure_ledger),
        })
        navs[variant] = result.combined_result.nav

    comparison = pd.DataFrame(rows).set_index("variant")
    comparison.to_csv(output / "comparison.csv", encoding="utf-8-sig")
    (output / "comparison.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(navs).to_csv(output / "nav_comparison.csv")
    print(comparison.to_string(float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
