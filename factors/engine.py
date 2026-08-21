from __future__ import annotations

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import numpy as np
import pandas as pd

from core.interfaces import Factor
from core.factor_contract import normalise_frequency, validate_factor_contract
from core.registry import get as registry_get
from core.types import DateIndex, FactorMatrix, Universe
from data.manager import DataManager

log = logging.getLogger(__name__)


class FactorComputationError(RuntimeError):
    """A configured factor could not be computed without changing semantics."""


class FactorEngine:
    """因子计算引擎: 批量计算因子矩阵 + 依赖解析 + 并行计算.

    千级因子扩展设计:
    - 使用 ThreadPoolExecutor 并行 (IO 密集型, 不需要进程隔离)
    - 支持分块计算 (chunk_by_factor): 当因子数 > 100 时, 分批计算避免内存峰值
    - 缓存 key 使用日期哈希, 避免不同日期范围碰撞
    - parallel=False 默认串行, 100 因子以下串行最稳定
    """

    def __init__(
        self, data_manager: DataManager, *, tolerant: bool = False,
        log_failures: bool = True,
    ):
        self._data = data_manager
        self._tolerant = bool(tolerant)
        self._log_failures = bool(log_failures)
        self._frequency = normalise_frequency(
            getattr(data_manager, "frequency", "daily")
        )
        self._factor_cache: Dict[str, FactorMatrix] = {}
        self._failure_ledger: list[dict[str, str]] = []

    @property
    def failures(self) -> tuple[dict[str, str], ...]:
        """Failures observed in explicit discovery-only tolerant mode."""
        return tuple(dict(item) for item in self._failure_ledger)

    @staticmethod
    def _nan_matrix(dates: DateIndex, universe: Universe) -> FactorMatrix:
        return pd.DataFrame(np.nan, index=dates, columns=universe, dtype=float)

    def _handle_failure(
        self,
        stage: str,
        factor_name: str,
        exc: Exception,
        dates: DateIndex,
        universe: Universe,
    ) -> FactorMatrix:
        message = f"factor {factor_name!r} failed during {stage}: {exc}"
        if not self._tolerant:
            raise FactorComputationError(message) from exc
        self._failure_ledger.append({
            "factor": str(factor_name),
            "stage": str(stage),
            "error": f"{type(exc).__name__}: {exc}",
        })
        if self._log_failures:
            log.warning(message, exc_info=True)
        return self._nan_matrix(dates, universe)

    @staticmethod
    def _validate_result(
        factor_name: str,
        result: object,
        dates: DateIndex,
        universe: Universe,
    ) -> FactorMatrix:
        if not isinstance(result, pd.DataFrame):
            raise TypeError("factor compute() must return a DataFrame")
        if result.index.has_duplicates or result.columns.has_duplicates:
            raise ValueError("factor output axes must be unique")
        if result.empty:
            raise ValueError("factor returned an empty matrix")
        aligned = result.reindex(index=dates, columns=universe)
        try:
            aligned = aligned.apply(pd.to_numeric, errors="raise")
        except (TypeError, ValueError) as exc:
            raise TypeError("factor output must be numeric") from exc
        values = aligned.to_numpy(dtype=float, copy=False)
        if np.isinf(values).any():
            raise ValueError("factor output contains infinite values")
        if not np.isfinite(values).any():
            raise ValueError("factor output contains no finite values")
        return aligned.astype(float)

    @staticmethod
    def _cache_key(factor: Factor, dates: DateIndex, universe: Universe) -> str:
        """生成缓存 key, 包含品种集合哈希、日期范围、因子规格哈希避免碰撞.

        CR-017: 旧实现只用 universe 长度, 两个长度相同但成分不同的 universe
        可能命中同一缓存. 现纳入有序品种列表哈希 + 因子规格哈希.
        """
        d0 = dates[0]
        d1 = dates[-1]
        # 有序品种列表的哈希 (避免长度相同但成分不同的 universe 碰撞)
        universe_hash = hashlib.md5(
            ",".join(map(str, list(universe))).encode("utf-8")
        ).hexdigest()[:16]
        # 因子规格哈希: 因子名 + SpecFactor 的 spec 字典 (普通 Factor 退化为 name)
        spec = getattr(factor, "spec", None)
        if spec is not None:
            spec_repr = f"{factor.name}:{sorted(spec.items(), key=lambda x: str(x[0]))}"
        else:
            spec_repr = factor.name
        factor_hash = hashlib.md5(spec_repr.encode("utf-8")).hexdigest()[:16]
        return f"{factor_hash}_{d0}_{d1}_{universe_hash}"

    def compute_factor(
        self, factor: Factor, dates: DateIndex, universe: Universe
    ) -> FactorMatrix:
        """计算单个因子矩阵."""
        if len(dates) == 0 or len(universe) == 0:
            raise ValueError("factor request dates and universe must be non-empty")
        try:
            validate_factor_contract(
                factor, provider_frequency=self._frequency
            )
        except ValueError as exc:
            if not self._tolerant:
                raise
            return self._handle_failure(
                "contract_validation", factor.name, exc, dates, universe
            )
        try:
            cache_key = self._cache_key(factor, dates, universe)
            cached = self._factor_cache.get(cache_key)
            if cached is not None and cached.index.equals(
                pd.Index(dates)
            ) and cached.columns.equals(pd.Index(universe)):
                return cached
            result = self._validate_result(
                factor.name,
                factor.compute(self._data, dates, universe),
                dates,
                universe,
            )
            self._factor_cache[cache_key] = result
            return result
        except Exception as exc:
            return self._handle_failure(
                "compute", factor.name, exc, dates, universe
            )

    def compute_factors(
        self,
        factor_names: List[str],
        dates: DateIndex,
        universe: Universe,
        parallel: bool = False,
        max_workers: int = 2,
        chunk_size: int = 100,
    ) -> Dict[str, FactorMatrix]:
        """批量计算多个因子.

        Args:
            factor_names: 因子名称列表.
            dates: 日期索引.
            universe: 品种池.
            parallel: 是否启用多线程并行 (默认 False, 串行最稳定).
            max_workers: 并行时最大线程数.
            chunk_size: 因子数超过此值时启用分块计算, 避免内存峰值.

        Returns:
            {factor_name: FactorMatrix} 字典.
        """
        names = [str(name) for name in factor_names]
        if len(names) != len(set(names)):
            raise ValueError("factor_names must be unique")
        if not names:
            return {}
        if len(dates) == 0 or len(universe) == 0:
            raise ValueError("factor request dates and universe must be non-empty")
        if pd.Index(dates).has_duplicates or pd.Index(universe).has_duplicates:
            raise ValueError("factor request dates and universe must be unique")

        result: Dict[str, FactorMatrix] = {}
        factors_by_name: Dict[str, Factor] = {}
        for name in names:
            try:
                factor = registry_get("factor", name)()
                validate_factor_contract(
                    factor, provider_frequency=self._frequency
                )
                factors_by_name[name] = factor
            except Exception as exc:
                result[name] = self._handle_failure(
                    "registration", name, exc, dates, universe
                )

        active_names = [name for name in names if name in factors_by_name]
        if not active_names:
            return result

        # Scope raw-data reads to this exact batch. Each declared dependency is
        # loaded once and reused by both SPEC and ordinary factors.
        try:
            self._data.prefetch(
                [factors_by_name[name] for name in active_names], dates, universe
            )
        except Exception as exc:
            if not self._tolerant:
                raise FactorComputationError(
                    f"factor dependency prefetch failed: {exc}"
                ) from exc
            self._failure_ledger.append({
                "factor": "*batch*",
                "stage": "prefetch",
                "error": f"{type(exc).__name__}: {exc}",
            })
            log.warning("factor dependency prefetch failed", exc_info=True)

        spec_result, non_spec_names = self._compute_spec_factors_optimized(
            active_names, dates, universe
        )
        result.update(spec_result)

        factors_obj = [factors_by_name[name] for name in non_spec_names]
        n_factors = len(factors_obj)
        if n_factors > chunk_size:
            log.info(
                f"非 SPEC 因子数 {n_factors} > {chunk_size}, 启用分块计算, "
                f"每块 {chunk_size} 个因子"
            )
            non_spec_result = self._compute_chunked(
                factors_obj, dates, universe, parallel, max_workers, chunk_size
            )
        elif factors_obj:
            non_spec_result = self._compute_batch(
                factors_obj, dates, universe, parallel, max_workers
            )
        else:
            non_spec_result = {}

        result.update(non_spec_result)
        for name in names:
            if name not in result:
                result[name] = self._handle_failure(
                    "result_validation",
                    name,
                    RuntimeError("factor result is missing from the batch"),
                    dates,
                    universe,
                )
        return {name: result[name] for name in names}

    def _compute_spec_factors_optimized(
        self,
        factor_names: List[str],
        dates: DateIndex,
        universe: Universe,
    ) -> tuple:
        """检测并批量计算 SPEC 因子, 返回 (spec 结果, 非 SPEC 因子名列表).

        性能优化路由:
        - 通过因子名后缀识别 SPEC 因子 (transform ∈ {z,delta,smooth,rank,...})
        - 查找对应的 SPEC 字典, 走 compute_spec_factors_batch 批量路径
        - 未匹配的因子走原有逐因子路径

        设计考虑:
        - 向后兼容: 未识别为 SPEC 的因子不影响原有路径
        - 扩展性: 未来新增 base/transform 无需修改本函数
        - 健壮性: 单个 SPEC 查找失败不影响其他因子
        """
        spec_result: Dict[str, FactorMatrix] = {}
        non_spec_names: List[str] = []
        spec_specs: list = []

        try:
            from factors.specs import SPEC_BY_SLUG
        except ImportError:
            # SPEC 模块未加载, 全部走原路径
            return spec_result, list(factor_names)

        for name in factor_names:
            if name in SPEC_BY_SLUG:
                spec_specs.append(SPEC_BY_SLUG[name])
            else:
                non_spec_names.append(name)

        if not spec_specs:
            return spec_result, non_spec_names

        # 批量计算 SPEC 因子
        try:
            from factors.spec_factor import compute_spec_factors_batch
            spec_result = compute_spec_factors_batch(
                spec_specs,
                self._data,
                dates,
                universe,
                tolerant=self._tolerant,
            )
        except Exception as exc:
            if not self._tolerant:
                raise FactorComputationError(
                    f"SPEC factor batch failed: {exc}"
                ) from exc
            self._failure_ledger.append({
                "factor": "*spec_batch*",
                "stage": "compute",
                "error": f"{type(exc).__name__}: {exc}",
            })
            log.warning(
                "SPEC batch failed; tolerant discovery will retry individually",
                exc_info=True,
            )
            non_spec_names = list(factor_names)
            return {}, non_spec_names

        validated: Dict[str, FactorMatrix] = {}
        for spec in spec_specs:
            name = str(spec["slug"])
            try:
                if name not in spec_result:
                    raise RuntimeError("SPEC batch omitted the requested factor")
                validated[name] = self._validate_result(
                    name, spec_result[name], dates, universe
                )
            except Exception as exc:
                validated[name] = self._handle_failure(
                    "result_validation", name, exc, dates, universe
                )
        return validated, non_spec_names

    def _compute_batch(
        self,
        factors_obj: List[Factor],
        dates: DateIndex,
        universe: Universe,
        parallel: bool,
        max_workers: int,
    ) -> Dict[str, FactorMatrix]:
        """单批次计算 (因子数 ≤ chunk_size)."""
        result: Dict[str, FactorMatrix] = {}
        if parallel and len(factors_obj) > 1:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(factors_obj))) as pool:
                futures = {
                    pool.submit(
                        self.compute_factor, f, dates, universe
                    ): f.name
                    for f in factors_obj
                }
                for future in as_completed(futures):
                    name = futures[future]
                    result[name] = future.result()
        else:
            for f in factors_obj:
                result[f.name] = self.compute_factor(f, dates, universe)
        return result

    def _compute_chunked(
        self,
        factors_obj: List[Factor],
        dates: DateIndex,
        universe: Universe,
        parallel: bool,
        max_workers: int,
        chunk_size: int,
    ) -> Dict[str, FactorMatrix]:
        """分块计算: 将因子列表切分为多个块, 逐块计算并释放中间结果."""
        result: Dict[str, FactorMatrix] = {}
        t0 = time.time()
        for i in range(0, len(factors_obj), chunk_size):
            chunk = factors_obj[i : i + chunk_size]
            chunk_result = self._compute_batch(
                chunk, dates, universe, parallel, max_workers
            )
            result.update(chunk_result)
            # 分块间清理缓存, 避免内存累积
            self.clear_cache()
            elapsed = time.time() - t0
            done = min(i + chunk_size, len(factors_obj))
            log.info(
                f"分块进度: {done}/{len(factors_obj)} ({done / len(factors_obj):.0%}), "
                f"耗时 {elapsed:.1f}s"
            )
        return result

    def clear_cache(self):
        self._factor_cache.clear()
