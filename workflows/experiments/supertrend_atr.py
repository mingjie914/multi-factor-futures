"""Run the standalone Supertrend ATR-risk sleeve."""
from __future__ import annotations

import argparse

import pandas as pd

from core.config import load_config
from pipeline.runner import PipelineRunner
from strategies.supertrend_atr_risk import SupertrendATRRiskStrategy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2025-06-30")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--rebalance-on-flip",
        action="store_true",
        help="Opt into event trades; scheduled five-day rebalancing is the default",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config.date_range.start = args.start
    config.date_range.end = args.end
    config.backtest.report_dir = args.output_dir
    runner = PipelineRunner(config=config)
    dates = runner.data_manager.get_calendar(args.start, args.end)
    if len(dates) == 0:
        dates = pd.date_range(args.start, args.end, freq="B")
    universe = pd.Index(config.universe)
    # Continuous-contract adjustment is path dependent. Fetch all price fields
    # in one request so they share one roll-adjustment scale.
    panel = runner.data_manager.source.fetch_price(
        universe,
        pd.Timestamp(args.start),
        pd.Timestamp(args.end),
        ["high", "low", "close"],
    )
    missing = [field for field in ("high", "low", "close") if field not in panel]
    if missing:
        raise RuntimeError(f"coherent OHLC request missing fields: {missing}")
    high = panel["high"].reindex(index=dates, columns=universe)
    low = panel["low"].reindex(index=dates, columns=universe)
    close = panel["close"].reindex(index=dates, columns=universe)

    sleeve = config.supertrend_sleeve
    if not sleeve.enabled or sleeve.integration_mode != "shadow":
        raise RuntimeError("supertrend_sleeve must be enabled in shadow mode")
    strategy = SupertrendATRRiskStrategy(
        rebalance_freq=sleeve.rebalance_freq,
        target_volatility=sleeve.target_volatility,
        asset_vol_budget=sleeve.asset_vol_budget,
        sector_vol_budget=sleeve.sector_vol_budget,
        hard_asset_cap=sleeve.hard_asset_cap,
        gross_cap=sleeve.gross_cap,
        net_cap=sleeve.net_cap,
        turnover_cap=sleeve.turnover_cap,
        rebalance_on_flip=(sleeve.rebalance_on_flip or args.rebalance_on_flip),
    )
    result = strategy.run(high, low, close, cost_model=runner.cost_model)
    print(result.summary())
    result.save(
        args.output_dir,
        metadata={
            "experiment": strategy.name,
            "start": args.start,
            "end": args.end,
            "atr_window": 20,
            "atr_multiplier": 2.0,
            "target_volatility": strategy.target_volatility,
            "asset_vol_budget": strategy.asset_vol_budget,
            "sector_vol_budget": strategy.sector_vol_budget,
            "hard_asset_cap": strategy.hard_asset_cap,
            "gross_cap": strategy.gross_cap,
            "net_cap": strategy.net_cap,
            "turnover_cap": strategy.turnover_cap,
            "rebalance_on_flip": strategy.rebalance_on_flip,
            "ohlc_source": "single_batch_continuous_contract_request",
        },
    )
    if args.plot:
        result.plot(save_dir=args.output_dir, version="Supertrend ATR Risk")


if __name__ == "__main__":
    main()
