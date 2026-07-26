from __future__ import annotations

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.interfaces import Factor
from core.registry import get as registry_get
from core.types import DateIndex, FactorMatrix, Universe
from data.manager import DataManager

log = logging.getLogger(__name__)

# 千级因子场景下的内存安全阈值: 单因子矩阵过大时启用分块计算
_MAX_MATRIX_ELEMENTS = 5_000_000  # 5M 个元素 ≈ 1000 因子 × 1200 天 × 40 品种 → 每个因子 ~48K 元素


class FactorEngine:
    """因子计算引擎: 批量计算因子矩阵 + 依赖解析 + 并行计算.

    千级因子扩展设计:
    - 使用 ThreadPoolExecutor 并行 (IO 密集型, 不需要进程隔离)
    - 支持分块计算 (chunk_by_factor): 当因子数 > 100 时, 分批计算避免内存峰值
    - 缓存 key 使用日期哈希, 避免不同日期范围碰撞
    - parallel=False 默认串行, 100 因子以下串行最稳定
    """

    def __init__(self, data_manager: DataManager):
        self._data = data_manager
        self._factor_cache: Dict[str, FactorMatrix] = {}

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
        cache_key = self._cache_key(factor, dates, universe)
        if cache_key in self._factor_cache:
            cached = self._factor_cache[cache_key]
            # CR-017: 读取缓存后校验索引和列集合是否匹配 (防止哈希碰撞误命中)
            if cached.index.equals(pd.Index(dates)) and cached.columns.equals(
                pd.Index(universe)
            ):
                return cached
            # 索引或列集合不匹配, 视为缓存未命中, 重新计算
        try:
            result = factor.compute(self._data, dates, universe)
            if result.empty:
                result = pd.DataFrame(np.nan, index=dates, columns=universe)
            else:
                result = result.reindex(index=dates, columns=universe)
            self._factor_cache[cache_key] = result
            return result
        except Exception:
            log.warning(
                f"因子 '{factor.name}' 计算失败", exc_info=True
            )
            return pd.DataFrame(np.nan, index=dates, columns=universe)

    def compute_factors(
        self,
        factor_names: List[str],
        dates: DateIndex,
        universe: Universe,
        parallel: bool = False,
        max_workers: int = 8,
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
        # Scope raw-data reads to this exact batch. This makes every declared
        # dependency, including unavailable fields, a single source/cache read.
        prefetch_factors = []
        for name in factor_names:
            try:
                prefetch_factors.append(registry_get("factor", name)())
            except (KeyError, TypeError):
                continue
        try:
            self._data.prefetch(prefetch_factors, dates, universe)
        except Exception:
            log.debug("factor dependency prefetch failed", exc_info=True)

        # 性能优化: SPEC 因子走批量路径 (按 base 分组, 避免重复计算)
        # 检测 SPEC 因子并路由到 compute_spec_factors_batch
        spec_result, non_spec_names = self._compute_spec_factors_optimized(
            factor_names, dates, universe
        )

        if not non_spec_names:
            return spec_result

        # 非 SPEC 因子走原有路径
        factors_obj = []
        for name in non_spec_names:
            try:
                cls = registry_get("factor", name)
                factors_obj.append(cls())
            except KeyError:
                log.warning(f"因子 '{name}' 未注册, 跳过")
                continue

        if not factors_obj:
            return spec_result

        # 预取所有依赖字段 (批量拉取减少 IO)
        try:
            self._data.prefetch(factors_obj)
        except Exception:
            log.debug("预取失败", exc_info=True)

        # 千级因子: 分块计算, 避免一次性加载所有因子到内存
        n_factors = len(factors_obj)
        if n_factors > chunk_size:
            log.info(
                f"非 SPEC 因子数 {n_factors} > {chunk_size}, 启用分块计算, "
                f"每块 {chunk_size} 个因子"
            )
            non_spec_result = self._compute_chunked(
                factors_obj, dates, universe, parallel, max_workers, chunk_size
            )
        else:
            non_spec_result = self._compute_batch(
                factors_obj, dates, universe, parallel, max_workers
            )

        result = dict(spec_result)
        result.update(non_spec_result)
        return result

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
            from factors.spec_factor import is_spec_factor
            from factors.specs import ALL_SPECS
        except ImportError:
            # SPEC 模块未加载, 全部走原路径
            return spec_result, list(factor_names)

        # 构建 slug -> spec 索引 (一次构建, 复用)
        spec_index = {s["slug"]: s for s in ALL_SPECS}

        for name in factor_names:
            if name in spec_index and is_spec_factor(name):
                spec_specs.append(spec_index[name])
            else:
                non_spec_names.append(name)

        if not spec_specs:
            return spec_result, non_spec_names

        # 批量计算 SPEC 因子
        try:
            from factors.spec_factor import compute_spec_factors_batch
            spec_result = compute_spec_factors_batch(
                spec_specs, self._data, dates, universe
            )
        except Exception:
            log.warning(
                "SPEC 批量计算失败, 回退到逐因子路径", exc_info=True
            )
            # 回退: 将 SPEC 因子加入非 SPEC 列表, 走原路径
            non_spec_names = list(factor_names)
            return {}, non_spec_names

        return spec_result, non_spec_names

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
                    pool.submit(f.compute, self._data, dates, universe): f.name
                    for f in factors_obj
                }
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        result[name] = future.result()
                    except Exception:
                        log.warning(f"因子 '{name}' 并行计算失败", exc_info=True)
                        result[name] = pd.DataFrame(
                            np.nan, index=dates, columns=universe
                        )
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
