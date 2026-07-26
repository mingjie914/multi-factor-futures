"""Compare standalone defensive-sleeve risk-allocation methods."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from core.config import load_config
from pipeline.runner import PipelineRunner


DEFAULT_ALLOCATIONS = [
    "inverse_volatility",
    "correlation_adjusted_inverse_volatility",
    "risk_parity",
    "hrp",
    "shrinkage_min_variance",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--allocations", default=",".join(DEFAULT_ALLOCATIONS))
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    if args.start:
        config.date_range.start = args.start
    if args.end:
        config.date_range.end = args.end
    if args.cache_only:
        config.data.cache["only"] = True
    config.defensive_sleeve.enabled = True
    config.defensive_sleeve.integration_mode = "standalone"
    config.backtest.report_dir = str(output)
    runner = PipelineRunner(config=config)

    rows = []
    navs = {}
    allocations = [item.strip() for item in args.allocations.split(",") if item.strip()]
    for allocation in allocations:
        runner.config.defensive_sleeve.allocation = allocation
        result = runner.run_defensive_sleeve()
        result.save(
            output / allocation,
            metadata={
                "experiment": "standalone_defensive_allocation_comparison",
                "allocation": allocation,
                "start": config.date_range.start,
                "end": config.date_range.end,
                "cache_only": bool(args.cache_only),
                "integration_mode": "standalone",
                "evidence_level": "historical_cache_segment_diagnostic",
            },
        )
        metrics = result.metrics
        split = result.split_metrics.get("test", {})
        rows.append({
            "allocation": allocation,
            "annual_return": metrics.get("annual_return", 0.0),
            "sharpe": metrics.get("sharpe", 0.0),
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "volatility": metrics.get("volatility", 0.0),
            "calmar": metrics.get("calmar", 0.0),
            "avg_turnover": metrics.get("avg_turnover", 0.0),
            "total_transaction_cost": metrics.get("total_transaction_cost", 0.0),
            "segment_test_sharpe": split.get("sharpe", 0.0),
            "failure_count": len(result.failure_ledger),
        })
        navs[allocation] = result.nav

    comparison = pd.DataFrame(rows).set_index("allocation")
    comparison.to_csv(output / "comparison.csv", encoding="utf-8-sig")
    (output / "comparison.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(navs).to_csv(output / "nav_comparison.csv")
    print(comparison.to_string(float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
