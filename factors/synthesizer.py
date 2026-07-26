"""因子合成器: 聚类内等权合成, 替代代表因子选取.

将相关性高的因子簇等权合成为单个因子, 保留全部显著因子的信息,
同时降低因子间冗余, 减少 alpha 模型的多重共线性问题.

设计考虑:
- 普适性: 等权合成不依赖样本期表现, 避免数据窥探
- 健壮性: 单因子异常值通过平均被稀释
- 可解释: 合成因子方向由簇内因子共识决定
- 扩展性: 支持 z-score 标准化后等权, 避免量纲差异

参考: AQR "Factor Zoo" 中的因子簇合成实践
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("multi_factor")


class FactorSynthesizer:
    """因子合成器: 按聚类映射将因子等权合成.

    合成步骤 (每个簇):
    1. 对簇内每个因子矩阵做截面 z-score 标准化 (消除量纲差异)
    2. 对标准化后的矩阵等权求平均
    3. 对合成结果再次 z-score 标准化 (保持分布一致)

    簇内若只有 1 个因子, 直接返回该因子 (标准化后).
    簇内若所有因子在某日全为 NaN, 合成结果为 NaN.

    日内确认 (方案A): 合成后对指定因子应用日内代理特征增强.
    confirmed = z(factor) × (1 + weight × z(confirm_factor))
    - weight > 0: 同向确认 (动量+强势收盘 → 放大)
    - weight < 0: 反向确认 (动量+弱势收盘 → 衰减)
    """

    def __init__(
        self,
        cluster_map: Dict[str, List[str]],
        flip_signs: Dict[str, int] = None,
        confirm_map: Dict[str, Tuple[str, float]] = None,
        ic_weights: Dict[str, float] = None,
    ):
        """初始化因子合成器.

        Args:
            cluster_map: {合成因子名: [原始因子名列表]} 映射.
                         例如 {"synth_1": ["rsi_5d_z", "rsi_5d_delta", ...]}
            flip_signs: {因子名: +1 或 -1} 方向修正映射.
                        来自 correlation_analysis.correct_factor_direction.
                        合成时对 flip=-1 的因子取反, 使同簇因子方向一致.
                        None 表示不做方向修正 (兼容旧版 JSON).
            confirm_map: {被确认因子名: (确认因子名, 确认强度)} 日内确认映射.
                         合成后对被确认因子应用: z(f) × (1 + w × z(confirm))
                         None 表示不做日内确认.
            ic_weights: {因子名: IC权重} IC加权合成映射.
                        若提供, 簇内合成改用IC加权而非等权.
                        权重来自IC检验结果, |IC|高的因子权重大.
                        None 表示等权合成 (默认, 兼容旧版).
        """
        self.cluster_map = cluster_map
        self.flip_signs = flip_signs or {}
        self.confirm_map = confirm_map or {}
        self.ic_weights = ic_weights or {}
        # 反向映射: 原始因子名 -> 合成因子名
        self._factor_to_cluster: Dict[str, str] = {}
        for synth_name, orig_names in cluster_map.items():
            for n in orig_names:
                self._factor_to_cluster[n] = synth_name
        logger.info(
            f"FactorSynthesizer 初始化: {len(cluster_map)} 个合成因子, "
            f"覆盖 {len(self._factor_to_cluster)} 个原始因子, "
            f"{sum(1 for v in self.flip_signs.values() if v == -1)} 个因子方向翻转, "
            f"{len(self.confirm_map)} 个日内确认"
        )

    def for_factors(self, factor_names: List[str]) -> "FactorSynthesizer":
        """返回只包含指定因子的子集合成器.

        用于多子组合场景: 每个子组合只合成自己包含的因子,
        避免计算未使用的因子.

        若指定因子集与任何簇都无交集, 返回 None (无需合成).
        日内确认映射也会过滤: 只保留确认因子在 factor_names 中的项.
        """
        filtered_map: Dict[str, List[str]] = {}
        factor_set = set(factor_names)
        for synth_name, orig_names in self.cluster_map.items():
            # 簇内与指定因子集的交集
            intersection = [n for n in orig_names if n in factor_set]
            if len(intersection) >= 2:
                # 交集 >= 2 才合成, 否则保留单因子
                filtered_map[synth_name] = intersection
        if not filtered_map:
            return None
        # 过滤日内确认: 只保留确认因子在子组合因子列表中的项
        filtered_confirm = {}
        for target, (confirm, weight) in self.confirm_map.items():
            if confirm in factor_set:
                filtered_confirm[target] = (confirm, weight)
        return FactorSynthesizer(
            filtered_map, flip_signs=self.flip_signs, confirm_map=filtered_confirm,
            ic_weights=self.ic_weights,
        )

    def synthesize(
        self,
        factor_matrices: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        """对因子矩阵应用聚类合成.

        Args:
            factor_matrices: {原始因子名: FactorMatrix} 字典.

        Returns:
            {合成因子名: 合成 FactorMatrix} 字典.
            未在 cluster_map 中的因子保留原样.
        """
        result: Dict[str, pd.DataFrame] = {}

        # 1. 保留不在任何簇中的因子
        for name, mat in factor_matrices.items():
            if name not in self._factor_to_cluster:
                result[name] = mat

        # 2. 按簇合成
        for synth_name, orig_names in self.cluster_map.items():
            # 收集簇内可用的因子矩阵 (应用方向修正)
            mats: List[pd.DataFrame] = []
            for n in orig_names:
                if n in factor_matrices:
                    mat = factor_matrices[n]
                    # 方向修正: flip=-1 的因子取反
                    flip = self.flip_signs.get(n, 1)
                    if flip == -1:
                        mat = -mat
                    mats.append(mat)

            if not mats:
                logger.warning(f"合成因子 '{synth_name}': 簇内无可用因子")
                continue

            if len(mats) == 1:
                # 单因子簇: 应用方向修正后标准化返回
                result[synth_name] = self._cross_section_zscore(mats[0])
                continue

            # 多因子簇: 对齐 → 标准化 → 加权平均 → 再标准化
            try:
                if self.ic_weights:
                    # IC加权合成: 用|IC|作为权重, 方向由flip_signs控制
                    orig_names_in_cluster = [
                        n for n in orig_names if n in factor_matrices
                    ]
                    weights = []
                    for n in orig_names_in_cluster:
                        # 取绝对值IC作为权重 (方向已由flip_signs处理)
                        w_val = abs(self.ic_weights.get(n, 0.0))
                        # 权重下限: 避免完全忽略某因子 (保留至少10%等权)
                        w_val = max(w_val, 0.01)
                        weights.append(w_val)
                    # 归一化
                    w_sum = sum(weights)
                    if w_sum > 0:
                        weights = [w / w_sum for w in weights]
                    else:
                        weights = [1.0 / len(mats)] * len(mats)
                    synth_mat = self._synthesize_weighted(mats, weights)
                else:
                    # 等权合成 (默认)
                    synth_mat = self._synthesize_equal_weight(mats)
                result[synth_name] = synth_mat
            except Exception:
                logger.warning(
                    f"合成因子 '{synth_name}' 失败, 使用首个因子",
                    exc_info=True,
                )
                result[synth_name] = self._cross_section_zscore(mats[0])

        # 3. 日内确认: 对指定因子应用日内代理特征增强
        if self.confirm_map:
            result = self._apply_intraday_confirm(result, factor_matrices)

        return result

    def _apply_intraday_confirm(
        self,
        synthesized: Dict[str, pd.DataFrame],
        raw_factors: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        """日内确认: 对合成后的因子应用日内代理特征增强.

        公式: confirmed = z(factor) × (1 + weight × z(confirm_factor))

        - 确认因子从 raw_factors 中获取 (它们是独立因子, 不参与聚类合成)
        - 确认强度 weight 控制增强幅度 (典型 0.2-0.5)
        - 确认因子缺失时跳过 (不修改原因子)

        Args:
            synthesized: 合成后的因子字典 (会被修改)
            raw_factors: 原始因子字典 (用于获取确认因子)

        Returns:
            修改后的合成因子字典.
        """
        for target_name, (confirm_name, weight) in self.confirm_map.items():
            # 被确认因子必须在合成结果中
            if target_name not in synthesized:
                continue
            # 确认因子必须在原始因子中
            if confirm_name not in raw_factors:
                logger.debug(
                    f"日内确认跳过: 确认因子 '{confirm_name}' 不在因子列表中"
                )
                continue

            target_mat = synthesized[target_name]
            confirm_mat = raw_factors[confirm_name]

            try:
                # 对齐索引和列
                confirm_aligned = confirm_mat.reindex(
                    index=target_mat.index, columns=target_mat.columns
                )
                # 截面 z-score 标准化确认因子
                confirm_z = self._cross_section_zscore(confirm_aligned)
                target_z = self._cross_section_zscore(target_mat)
                # 确认增强: z(f) × (1 + w × z(confirm))
                confirmed = target_z * (1 + weight * confirm_z)
                synthesized[target_name] = confirmed
                logger.debug(
                    f"日内确认: {target_name} × {confirm_name} (w={weight})"
                )
            except Exception:
                logger.warning(
                    f"日内确认失败: {target_name} ← {confirm_name}",
                    exc_info=True,
                )

        return synthesized

    def _synthesize_equal_weight(
        self, mats: List[pd.DataFrame]
    ) -> pd.DataFrame:
        """等权合成多个因子矩阵.

        步骤:
        1. 对齐索引和列
        2. 每个矩阵截面 z-score 标准化
        3. 等权求平均 (忽略 NaN)
        4. 结果再次 z-score 标准化

        NaN 处理: 某日某品种若部分因子有值, 用有值的因子平均;
                  若全部 NaN, 结果为 NaN.
        """
        # 对齐所有矩阵到公共索引和列
        common_idx = mats[0].index
        common_cols = mats[0].columns
        aligned: List[pd.DataFrame] = []
        for m in mats:
            m_aligned = m.reindex(index=common_idx, columns=common_cols)
            aligned.append(self._cross_section_zscore(m_aligned))

        # 等权平均: 忽略 NaN
        # 用 numpy nanmean 实现, 但需注意全 NaN 行
        stacked = np.array([m.values for m in aligned])  # (n_factors, n_dates, n_symbols)
        with np.errstate(invalid="ignore"):
            synth = np.nanmean(stacked, axis=0)  # (n_dates, n_symbols)

        # 全 NaN 的位置保持 NaN
        all_nan = np.all(np.isnan(stacked), axis=0)
        synth = np.where(all_nan, np.nan, synth)

        synth_df = pd.DataFrame(synth, index=common_idx, columns=common_cols)

        # 再次 z-score 标准化 (保持合成因子分布一致)
        return self._cross_section_zscore(synth_df)

    def _synthesize_weighted(
        self, mats: List[pd.DataFrame], weights: List[float]
    ) -> pd.DataFrame:
        """IC加权合成多个因子矩阵.

        步骤:
        1. 对齐索引和列
        2. 每个矩阵截面 z-score 标准化
        3. 按 weights 加权求平均 (忽略 NaN, 用有效因子的权重重新归一化)
        4. 结果再次 z-score 标准化

        NaN 处理: 某日某品种若部分因子有值, 用有值的因子按其权重加权平均;
                  若全部 NaN, 结果为 NaN.
        """
        # 对齐所有矩阵到公共索引和列
        common_idx = mats[0].index
        common_cols = mats[0].columns
        aligned: List[pd.DataFrame] = []
        for m in mats:
            m_aligned = m.reindex(index=common_idx, columns=common_cols)
            aligned.append(self._cross_section_zscore(m_aligned))

        n_factors = len(aligned)
        w_arr = np.array(weights[:n_factors], dtype=float)  # (n_factors,)
        stacked = np.array([m.values for m in aligned])  # (n_factors, n_dates, n_symbols)

        # NaN 掩码: True = 有效值
        valid_mask = ~np.isnan(stacked)  # (n_factors, n_dates, n_symbols)

        # 权重扩展到与 stacked 相同形状
        w_expanded = np.broadcast_to(
            w_arr[:, None, None], stacked.shape
        )  # (n_factors, n_dates, n_symbols)

        # NaN 处为 0 权重
        w_masked = w_expanded * valid_mask

        # 加权求和
        with np.errstate(invalid="ignore"):
            weighted_sum = np.nansum(stacked * w_expanded * valid_mask, axis=0)
            weight_sum = w_masked.sum(axis=0)

        # 避免除零
        synth = np.where(
            weight_sum > 1e-12,
            weighted_sum / weight_sum,
            np.nan,
        )

        synth_df = pd.DataFrame(synth, index=common_idx, columns=common_cols)

        # 再次 z-score 标准化
        return self._cross_section_zscore(synth_df)

    @staticmethod
    def _cross_section_zscore(df: pd.DataFrame) -> pd.DataFrame:
        """截面 z-score 标准化 (每行: 减均值除标准差).

        对每行 (单个日期跨所有品种) 做 z-score,
        消除不同因子量纲差异, 使等权合成有意义.

        NaN 处理: 单品种 NaN 保持 NaN;
                  若某行有效值 < 2, 返回全 0 (避免 std=0).
        """
        mean = df.mean(axis=1)
        std = df.std(axis=1, ddof=0)
        # 避免除零: std=0 时用 1 替代, 结果为 0
        std_safe = std.where(std > 1e-10, 1.0)
        z = df.sub(mean, axis=0).div(std_safe, axis=0)
        # std=0 的行返回 0 (所有品种等价)
        z = z.where(std > 1e-10, 0.0)
        return z


def build_cluster_map_from_json(
    corr_json_path: str,
    min_cluster_size: int = 2,
) -> Tuple[Dict[str, List[str]], List[str], Dict[str, int]]:
    """从相关性分析 JSON 构建聚类映射.

    重要: 只合成同一 best_period 的因子, 避免跨持有期信号混淆.
    聚类内若包含不同 best_period 的因子, 按 best_period 拆分为子簇.

    支持新版 JSON (含 flip_signs 和 factors[].flip 字段) 和旧版 JSON (无方向修正).

    Args:
        corr_json_path: factor_correlation.json 路径.
        min_cluster_size: 最小簇大小, 小于此值的簇不合成 (保留原因子).

    Returns:
        (cluster_map, standalone_factors, flip_signs)
        - cluster_map: {合成因子名: [原始因子名]}
        - standalone_factors: 不需合成的独立因子列表
        - flip_signs: {因子名: +1 或 -1} 方向修正映射
    """
    import json
    import os

    with open(corr_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 读取方向修正映射 (新版 JSON)
    flip_signs: Dict[str, int] = data.get("flip_signs", {})

    cluster_map: Dict[str, List[str]] = {}
    standalone: List[str] = []
    synth_counter = 0

    for cluster in data.get("clusters", []):
        # 按 best_period 分组 (避免跨持有期合成)
        period_groups: Dict[int, List[Tuple[str, int]]] = {}
        for f in cluster["factors"]:
            p = f.get("period", 0)
            name = f["name"]
            # 优先从 factors[].flip 读取, 其次从顶层 flip_signs 读取
            flip = f.get("flip", flip_signs.get(name, 1))
            period_groups.setdefault(p, []).append((name, flip))

        for period, factors_with_flip in period_groups.items():
            factors = [name for name, _ in factors_with_flip]
            if len(factors) >= min_cluster_size:
                synth_counter += 1
                synth_name = f"synth_c{cluster['cluster_id']}_p{period}"
                cluster_map[synth_name] = factors
                # 更新 flip_signs (确保合成时方向修正生效)
                for name, flip in factors_with_flip:
                    flip_signs[name] = flip
            else:
                standalone.extend(factors)
                # 独立因子也记录 flip (虽然不合成, 但若上层需要方向修正可使用)
                for name, flip in factors_with_flip:
                    if name not in flip_signs:
                        flip_signs[name] = flip

    return cluster_map, standalone, flip_signs


def build_factor_to_synthesis_name_map(
    cluster_map: Dict[str, List[str]],
    standalone_factors: List[str],
) -> Dict[str, str]:
    """构建 {原始因子名: 合成后名称} 映射.

    用于在回测中追踪因子来源.

    CR-019: 修复循环中引用未定义变量 standalone 的 NameError,
    改为使用函数参数 standalone_factors.
    """
    mapping: Dict[str, str] = {}
    for synth_name, orig_names in cluster_map.items():
        for n in orig_names:
            mapping[n] = synth_name
    for n in standalone_factors:
        mapping[n] = n
    return mapping
