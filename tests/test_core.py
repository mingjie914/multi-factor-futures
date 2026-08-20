r"""核心组件单元测试: 因子计算、IC 检验、启发式优化器.

用法: python tests/test_core.py
"""
from __future__ import annotations
import sys
import os
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

_passed = 0
_failed = 0


def _check(cond: bool, msg: str):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS: {msg}")
    else:
        _failed += 1
        print(f"  FAIL: {msg}")
        raise AssertionError(msg)


def _run(msg: str):
    print(f"\n--- {msg} ---")


# ============================================================
# Test: FactorEngine cache key
# ============================================================
def test_cache_key_collision():
    _run("FactorEngine cache key - 验证不同日期范围不会碰撞")
    from factors.engine import FactorEngine
    dates1 = pd.date_range("2020-01-01", "2020-12-31", freq="B")
    dates2 = pd.date_range("2021-01-01", "2021-12-31", freq="B")
    universe = pd.Index(["RB", "CU", "AU"])

    factor = SimpleNamespace(name="test")
    k1 = FactorEngine._cache_key(factor, dates1, universe)
    k2 = FactorEngine._cache_key(factor, dates2, universe)
    _check(k1 != k2, f"cache key 不同: {k1[:20]}... != {k2[:20]}...")


# ============================================================
# Test: IC perfect positive
# ============================================================
def test_ic_pearson_perfect_positive():
    _run("IC test - 完全正相关因子 → IC ≈ 1.0")
    from testing.ic_test import ICTest
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    tickers = [f"T{i}" for i in range(20)]
    factor = pd.DataFrame(
        np.random.randn(100, 20), index=dates, columns=tickers
    )
    fwd_ret = factor.copy()

    test = ICTest(methods=["pearson"], decay_periods=[1], forward_period=1)
    result = test.run(factor, fwd_ret, {})
    mean_ic = result.ic_mean
    _check(mean_ic > 0.8, f"mean IC = {mean_ic:.4f} > 0.8")


# ============================================================
# Test: IC empty input
# ============================================================
def test_ic_empty():
    _run("IC test - 空输入不崩溃")
    from testing.ic_test import ICTest
    factor = pd.DataFrame()
    fwd_ret = pd.DataFrame()
    test = ICTest(methods=["pearson", "spearman"])
    result = test.run(factor, fwd_ret, {})
    _check(result is not None, "返回有效结果")


# ============================================================
# Test: Layered empty input
# ============================================================
def test_layered_empty():
    _run("Layered backtest - 空输入不崩溃")
    from testing.layered import LayeredBacktest
    factor = pd.DataFrame()
    fwd_ret = pd.DataFrame()
    test = LayeredBacktest(n_groups=5)
    result = test.run(factor, fwd_ret, {})
    _check(result is not None, "返回有效结果")


# ============================================================
# Test: Stack factors and returns
# ============================================================
def test_stack_factors():
    _run("stack_factors_and_returns - 维度验证")
    from factors.utils import stack_factors_and_returns
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    tickers = ["A", "B", "C"]
    f1 = pd.DataFrame(np.random.randn(10, 3), index=dates, columns=tickers)
    f2 = pd.DataFrame(np.random.randn(10, 3), index=dates, columns=tickers)
    fwd = pd.DataFrame(np.random.randn(10, 3), index=dates, columns=tickers)

    merged, names, X, y, codes = stack_factors_and_returns({"f1": f1, "f2": f2}, fwd)
    _check(len(names) == 2, f"因子数 = {len(names)}")
    _check(X.shape[1] == 2, f"X列数 = {X.shape[1]}")
    _check(X.shape[0] == y.shape[0], f"X行数 = {X.shape[0]} == y行数 = {y.shape[0]}")
    _check(X.shape[0] == len(codes), f"行数 = codes数 = {len(codes)}")


# ============================================================
# Test: Data cache (mock)
# ============================================================
def test_cache_key():
    _run("Data cache - 不同日期范围不同 key")
    from data.cache import Cache
    with tempfile.TemporaryDirectory() as cache_dir:
        cache = Cache(cache_dir=cache_dir)
        tickers = ["RB", "CU"]
        k1 = str(cache._key("futures", "test", "close", tickers, "2020-01-01", "2020-12-31"))
        k2 = str(cache._key("futures", "test", "close", tickers, "2021-01-01", "2021-12-31"))
        _check(k1 != k2, f"key 不同 ({k1[-40:]} != {k2[-40:]})")


def test_cache_rejects_unimplemented_backend():
    from data.cache import Cache
    with tempfile.TemporaryDirectory() as cache_dir:
        try:
            Cache(cache_dir=cache_dir, backend="feather")
        except ValueError as exc:
            _check("parquet" in str(exc), "拒绝未实现的缓存后端")
        else:
            _check(False, "未实现的缓存后端必须失败关闭")


def test_dynamic_universe_schedule_reads_calendar_once():
    from pipeline.runner import PipelineRunner

    class Data:
        calendar_calls = 0

        def get_listing_dates(self, universe):
            return pd.Series(
                pd.to_datetime(["2024-01-01", "2024-01-08"]),
                index=universe,
            )

        def get_calendar(self, start, end):
            self.calendar_calls += 1
            return pd.bdate_range(start, end)

    runner = PipelineRunner.__new__(PipelineRunner)
    runner.data_manager = Data()
    schedule = runner._build_dynamic_universe_schedule(
        pd.Index(["A", "B"]),
        pd.to_datetime(["2024-01-05", "2024-01-07", "2024-01-08"]),
    )

    assert runner.data_manager.calendar_calls == 1
    assert schedule[pd.Timestamp("2024-01-05")].tolist() == ["A"]
    assert schedule[pd.Timestamp("2024-01-08")].tolist() == ["A", "B"]


# ============================================================
# Test: Registry constraint auto-discovery
# ============================================================
def test_registry_constraint():
    _run("Registry - 约束自动发现")
    from core.registry import create
    from optimization import constraints  # noqa: F401
    try:
        c = create("constraint", "net_exposure", lower=-0.5, upper=0.5)
        _check(c is not None, "net_exposure 约束创建成功")
        _check(hasattr(c, "apply"), "约束有 apply 方法")
    except Exception as e:
        _check(False, f"约束创建失败: {e}")


def test_registry_rejects_silent_replacement():
    from core.registry import _REGISTRIES, register

    kind = "test_component"
    name = "duplicate_guard"

    class First:
        pass

    class Second:
        pass

    try:
        assert register(kind, name)(First) is First
        assert register(kind, name)(First) is First
        try:
            register(kind, name)(Second)
        except ValueError as exc:
            assert "duplicate registration" in str(exc)
        else:
            raise AssertionError("duplicate component registration must fail")
        assert _REGISTRIES[kind][name] is First
    finally:
        _REGISTRIES.pop(kind, None)


# ============================================================
# Test: Run scripts import check
# ============================================================
def test_scripts_import():
    _run("Workflows - research / backtest 可导入")
    try:
        import importlib
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        import workflows.research  # noqa: F401
        import workflows.backtest  # noqa: F401
        _check(True, "workflows 模块导入成功")
    except Exception as e:
        _check(False, f"scripts 导入失败: {e}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    test_cache_key_collision()
    test_ic_pearson_perfect_positive()
    test_ic_empty()
    test_layered_empty()
    test_stack_factors()
    test_cache_key()
    test_registry_constraint()
    test_scripts_import()

    total = _passed + _failed
    print(f"\n{'='*50}")
    print(f"  结果: {_passed}/{total} 通过, {_failed} 失败")
    print(f"{'='*50}")
    if _failed > 0:
        sys.exit(1)
