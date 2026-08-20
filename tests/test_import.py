r"""验证正式模块可使用当前项目解释器正确导入."""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_MISSING = []


def _record_failure(level: str, label: str, exc: BaseException) -> None:
    _MISSING.append((level, label, f"{type(exc).__name__}: {exc}"))
    print(f"  [FAIL] {level} ({label}): {type(exc).__name__}: {exc}")

def _check(mod: str, label: str, level: str):
    try:
        __import__(mod)
        print(f"  [PASS] {level} ({label})")
    except Exception as exc:
        _record_failure(level, label, exc)


def test_imports():
    _MISSING.clear()

    # L0
    try:
        from core.types import FactorMatrix, ReturnMatrix, WeightVector, NAVSeries
        from core.interfaces import (Factor, DataProvider, DataSource, RiskModel, Optimizer)
        from core.registry import register, get, create, list_registered, register_factor
        from core.market import Market
        from core.logger import setup_logger
        print("  [PASS] Core (L0)")
    except Exception as e:
        _record_failure("L0", "Core", e)

    # L1 — data
    try:
        from data.cache import Cache
        from data.manager import DataManager
        from data.parquet_source import ParquetFuturesSource
        print("  [PASS] Data (L1)")
    except Exception as e:
        _record_failure("L1", "Data", e)

    # L2
    try:
        from factors.engine import FactorEngine
        from factors.processor import FactorProcessor
        print("  [PASS] Factors + Processing (L2)")
    except Exception as e:
        _record_failure("L2", "Factors + Processing", e)

    # L3
    try:
        from testing.ic_test import ICTest
        from testing.layered import LayeredBacktest
        from testing.regression import RegressionTest
        from alpha.ols import OLSModel
        from risk.barra_futures import BarraFuturesModel
        print("  [PASS] Testing + Alpha + Risk (L3)")
    except Exception as e:
        _record_failure("L3", "Testing + Alpha + Risk", e)

    # L4
    try:
        from optimization.mean_variance import MeanVarianceOptimizer
        from optimization.constraints import LongOnlyConstraint, TurnoverConstraint
        from optimization.costs import SimpleFuturesCost
        from optimization.asset_selection import SectorForecastSelector
        print("  [PASS] Optimization (L4)")
    except Exception as e:
        _record_failure("L4", "Optimization", e)

    # L5
    try:
        from backtest.engine import Backtester, BacktestResult
        from backtest.metrics import compute_sharpe, compute_max_drawdown
        print("  [PASS] Backtest (L5)")
    except Exception as e:
        _record_failure("L5", "Backtest", e)

    # L6
    try:
        from pipeline.runner import PipelineRunner
        print("  [PASS] Pipeline (L6)")
    except Exception as e:
        _record_failure("L6", "Pipeline", e)

    print(f"\n{'='*30}")
    if not _MISSING:
        print("  [PASS] 全部模块导入成功!")
    else:
        print(f"  [FAIL] {len(_MISSING)} 个模块导入失败:")
        for level, label, detail in _MISSING:
            print(f"    {level} ({label}): {detail}")
    print(f"{'='*30}")
    assert not _MISSING, f"{len(_MISSING)} import groups failed"


if __name__ == "__main__":
    try:
        test_imports()
    except AssertionError:
        sys.exit(1)
