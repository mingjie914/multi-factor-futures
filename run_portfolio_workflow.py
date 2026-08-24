"""IDE entrypoint for durable single/multi-strategy backtests and comparison.

The readable strategy library names parallel factor subsets and strategies;
each strategy still points to one complete framework YAML. Press Run without
command-line workflow routing.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from pathlib import Path

import pandas as pd

from core.config import load_config, load_strategy_library
from pipeline.runner import PipelineRunner
from research.effective_factor_library import (
    effective_factor_names,
    validate_effective_factor_periods,
)


class PortfolioWorkflow(Enum):
    VALIDATE_CONFIGURATIONS = "validate_configurations"
    RUN_AND_COMPARE = "run_and_compare"


# ============================== IDE SETTINGS ==============================
WORKFLOW = PortfolioWorkflow.VALIDATE_CONFIGURATIONS

CATALOG_PATH = "config/strategy_library.yaml"
RUN_ID: str | None = None
# ========================================================================


PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return (candidate if candidate.is_absolute() else PROJECT_ROOT / candidate).resolve()


def _assignments(config, mode: str) -> dict[int, list[str]]:
    if mode == "single":
        return {int(config.backtest.holding_period): list(config.factors)}
    assignments: dict[int, list[str]] = {}
    if not config.sub_portfolios:
        raise ValueError("multi strategy requires non-empty sub_portfolios")
    for sleeve in config.sub_portfolios:
        if not sleeve.factors:
            raise ValueError(f"sub-portfolio {sleeve.name!r} has no factors")
        assignments.setdefault(int(sleeve.holding_period), []).extend(sleeve.factors)
    return assignments


def _validated_specs():
    catalog_path = _resolve(CATALOG_PATH)
    catalog = load_strategy_library(catalog_path)
    factor_sets = {entry.id: entry for entry in catalog.factor_sets}
    library_path = _resolve(catalog.effective_factor_library)
    effective_names = set(effective_factor_names(library_path))
    for factor_set in catalog.factor_sets:
        if factor_set.status != "active":
            continue
        unknown = sorted(set(factor_set.factors) - effective_names)
        if unknown:
            raise ValueError(
                f"factor set {factor_set.id!r} contains non-effective factors: {unknown}"
            )
    validated = []
    for strategy in catalog.strategies:
        if strategy.status == "archived":
            continue
        path = _resolve(strategy.config_path)
        config = load_config(path)
        if config.data.source != "duckdb_futures":
            raise ValueError(
                f"strategy {strategy.id!r} must use the framework's certified "
                "duckdb_futures runtime source"
            )
        assignments = _assignments(config, strategy.mode)
        configured_factors = set().union(*map(set, assignments.values()))
        if strategy.source == "effective_library":
            if not config.factor_library.enforce_portfolio_periods:
                raise ValueError(
                    f"effective-library strategy {strategy.id!r} must enable "
                    "factor_library.enforce_portfolio_periods"
                )
            if _resolve(config.factor_library.path) != library_path:
                raise ValueError(
                    f"strategy {strategy.id!r} does not use the catalog's "
                    "effective factor library"
                )
            expected_factors = set(factor_sets[strategy.factor_set_id].factors)
            if configured_factors != expected_factors:
                raise ValueError(
                    f"strategy {strategy.id!r} factors do not match factor set "
                    f"{strategy.factor_set_id!r}"
                )
            validate_effective_factor_periods(
                library_path, assignments
            )
        validated.append((strategy, path, config))
    if not validated:
        raise ValueError("strategy library has no preferred or observing strategies")
    return catalog_path, catalog, validated


def _config_dict(config) -> dict:
    return config.model_dump() if hasattr(config, "model_dump") else config.dict()


def run_and_compare() -> Path:
    catalog_path, catalog, specs = _validated_specs()
    run_id = RUN_ID or datetime.now().strftime("%Y%m%d_%H%M%S")
    output = _resolve(catalog.output_root) / run_id
    output.mkdir(parents=True, exist_ok=False)
    rows, navs, configs = [], {}, {}

    for strategy, config_path, config in specs:
        name, mode = strategy.id, strategy.mode
        strategy_dir = output / name
        config.backtest.report_dir = str(strategy_dir)
        runner = PipelineRunner(config=config)  # resolves latest available date
        result = (
            runner.run_full_pipeline()
            if mode == "single"
            else runner.run_multi_portfolio()
        )
        result.save(strategy_dir, metadata={
            "strategy": name,
            "status": strategy.status,
            "factor_set_id": strategy.factor_set_id,
            "mode": mode,
            "config_path": str(config_path),
            "observation_end": runner.config.date_range.end,
        })
        if catalog.plot:
            result.plot(save_dir=str(strategy_dir))
        combined = result if mode == "single" else result.combined_result
        navs[name] = combined.nav / float(combined.nav.iloc[0])
        rows.append({
            "strategy": name,
            "status": strategy.status,
            "factor_set_id": strategy.factor_set_id,
            "start": runner.config.date_range.start,
            "end": runner.config.date_range.end,
            **combined.metrics,
        })
        configs[name] = {
            "path": str(config_path),
            "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "resolved": _config_dict(runner.config),
        }

    comparison = pd.DataFrame(rows).set_index("strategy")
    comparison.to_csv(output / "comparison.csv", encoding="utf-8-sig")
    nav_table = pd.DataFrame(navs)
    nav_table.to_csv(output / "nav_comparison.csv")
    (output / "run_contract.json").write_text(json.dumps({
        "schema_version": 1,
        "run_id": run_id,
        "strategy_library": {
            "path": str(catalog_path),
            "sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
            "snapshot": _config_dict(catalog),
        },
        "strategies": configs,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if catalog.plot:
        axes = nav_table.plot(figsize=(12, 6), title="Strategy NAV comparison")
        axes.set_ylabel("Normalized NAV")
        axes.figure.tight_layout()
        axes.figure.savefig(output / "nav_comparison.png", dpi=150)
        import matplotlib.pyplot as plt
        plt.close(axes.figure)
    return output


def main() -> None:
    catalog_path, _, specs = _validated_specs()
    if WORKFLOW is PortfolioWorkflow.VALIDATE_CONFIGURATIONS:
        print(f"strategy_library={catalog_path}")
        for strategy, path, config in specs:
            gate = config.factor_library.enforce_portfolio_periods
            print(
                f"{strategy.id}: status={strategy.status}, mode={strategy.mode}, "
                f"factor_set={strategy.factor_set_id or '-'}, config={path}, "
                f"data_source={config.data.source}, "
                f"dates={config.date_range.start}~{config.date_range.end}, "
                f"effective_period_gate={gate}"
            )
        return
    if WORKFLOW is PortfolioWorkflow.RUN_AND_COMPARE:
        print(f"组合回测结果: {run_and_compare()}")
        return
    raise ValueError(f"unsupported portfolio workflow: {WORKFLOW!r}")


if __name__ == "__main__":
    main()
