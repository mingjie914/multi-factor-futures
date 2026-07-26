"""Run the optional defensive trend/risk-allocation sleeve standalone."""
from __future__ import annotations

import argparse

from core.config import load_config
from pipeline.runner import PipelineRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--allocation", default=None)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.start:
        config.date_range.start = args.start
    if args.end:
        config.date_range.end = args.end
    if args.cache_only:
        config.data.cache["only"] = True
    if args.output_dir:
        config.backtest.report_dir = args.output_dir
    if args.allocation:
        config.defensive_sleeve.allocation = args.allocation
    config.defensive_sleeve.enabled = True
    config.defensive_sleeve.integration_mode = "standalone"
    runner = PipelineRunner(config=config)
    result = runner.run_defensive_sleeve()
    print(result.summary())
    result.save(
        runner.config.backtest.report_dir,
        metadata={
            "experiment": "standalone_defensive_sleeve",
            "allocation": runner.config.defensive_sleeve.allocation,
            "start": runner.config.date_range.start,
            "end": runner.config.date_range.end,
            "cache_only": bool(args.cache_only),
            "integration_mode": "standalone",
        },
    )
    if args.plot:
        result.plot(
            save_dir=runner.config.backtest.report_dir,
            version="defensive_trend_risk_parity",
        )


if __name__ == "__main__":
    main()
