"""因子相关性与聚类分析模块.

对显著因子做相关性分析 + 贪心聚类 (或层次聚类), 输出:
- reports/factor_correlation.json: 高相关对 + 聚类结果 (供 FactorSynthesizer 使用)
- reports/factor_correlation.png:  相关性热力图 + 聚类树状图

设计考虑 (修复旧版缺陷):
1. 方向修正: 负相关因子先翻转再聚类, 避免反向信号互相抵消
2. 滚动相关性 (可选): 避免全样本数据窥探, 用最近 window 天的相关性
3. 阈值自动选择 (可选): 层次聚类 + 轮廓系数, 自动确定聚类粒度
4. 可重复运行: 独立模块, 支持命令行 + 函数调用

Usage:
    # 命令行: 对 ic_by_window_period.json 中的显著因子做相关性分析
    python -m factors.correlation_analysis

    # 指定相关性阈值 (默认 0.6)
    python -m factors.correlation_analysis --threshold 0.5

    # 用滚动相关性 (最近 252 天)
    python -m factors.correlation_analysis --rolling 252

    # 用层次聚类 + 自动选阈值
    python -m factors.correlation_analysis --method hierarchical
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("multi_factor")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_significant_factors(ic_json_path: str) -> List[dict]:
    """从 ic_by_window_period.json 加载显著因子列表."""
    with open(ic_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("significant_factors", [])


def _flatten_factor_matrix(mat: pd.DataFrame) -> pd.Series:
    """将因子矩阵 (日期×品种) 展平为一维向量, 去除 NaN."""
    return mat.stack().dropna()


def compute_factor_correlation(
    factor_matrices: Dict[str, pd.DataFrame],
    factor_names: List[str],
    rolling_window: Optional[int] = None,
) -> pd.DataFrame:
    """计算因子间相关性矩阵.

    Args:
        factor_matrices: {因子名: 矩阵} 字典.
        factor_names: 参与计算的因子名列表.
        rolling_window: 若指定, 用最近 N 天的数据计算相关性 (避免全样本窥探).
                       None 表示用全样本.

    Returns:
        n × n 的相关性矩阵 (pd.DataFrame).
    """
    n = len(factor_names)
    if n == 0:
        return pd.DataFrame()

    # 展平每个因子矩阵为一维向量, 对齐索引
    series_dict: Dict[str, pd.Series] = {}
    for name in factor_names:
        if name not in factor_matrices:
            logger.warning(f"因子 '{name}' 不在 factor_matrices 中, 跳过")
            continue
        mat = factor_matrices[name]
        if rolling_window is not None and rolling_window > 0:
            # 只用最近 rolling_window 天
            mat = mat.tail(rolling_window)
        series_dict[name] = _flatten_factor_matrix(mat)

    if len(series_dict) < 2:
        return pd.DataFrame()

    # 对齐所有因子到公共索引 (品种×日期)
    df = pd.DataFrame(series_dict)
    # 计算相关性, 用 pairwise 完整观测
    corr = df.corr(method="pearson", min_periods=30)
    return corr


def correct_factor_direction(
    corr: pd.DataFrame,
    factor_t_stats: Dict[str, float],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """方向修正: 让所有因子的 IC 符号与 t 值符号一致.

    问题: 两个因子可能 |corr|=0.9 但一正一负, 直接合成会互相抵消.
    解决: 若 corr(i, j) < 0 且 IC 符号相反, 翻转其中一个因子的方向,
          使方向修正后的相关性矩阵中, 同簇因子均为正相关.

    Args:
        corr: 原始相关性矩阵.
        factor_t_stats: {因子名: t 值} 字典, 用于确定因子方向.

    Returns:
        (corrected_corr, flip_signs)
        - corrected_corr: 方向修正后的相关性矩阵
        - flip_signs: {因子名: +1 或 -1}, 记录哪些因子被翻转
    """
    names = list(corr.index)
    n = len(names)
    if n == 0:
        return corr, {}

    # 用 t 值符号作为因子方向的基准
    # 若 t > 0, 因子方向为正 (IC > 0); 若 t < 0, 方向为负
    flip_signs: Dict[str, int] = {name: 1 for name in names}
    for name in names:
        t = factor_t_stats.get(name, 0)
        if t < 0:
            flip_signs[name] = -1

    # 构造方向修正后的相关性矩阵:
    # corrected_corr(i, j) = flip(i) * flip(j) * corr(i, j)
    # 这样同方向因子 (flip 同号) 的 corr 不变, 反方向因子的 corr 符号翻转
    corrected = corr.copy()
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            corrected.iloc[i, j] = flip_signs[ni] * flip_signs[nj] * corr.iloc[i, j]

    return corrected, flip_signs


def greedy_cluster(
    corr: pd.DataFrame,
    threshold: float = 0.6,
) -> List[List[str]]:
    """贪心聚类: 基于 |corr| > threshold 的图连通分量近似.

    算法:
    1. 把每个因子看作图的节点
    2. 对每对 (i, j), 若 |corr(i,j)| > threshold, 连一条边
    3. 贪心策略: 按节点的度数降序处理, 优先加入已有簇 (与簇内任一成员 |corr| > threshold),
       否则开新簇
    4. 单节点簇 (度数=0) 作为独立因子

    Args:
        corr: 相关性矩阵 (建议先做方向修正).
        threshold: 聚类阈值, |corr| 超过此值认为同簇.

    Returns:
        聚类列表, 每个聚类是因子名列表.
    """
    names = list(corr.index)
    n = len(names)
    if n == 0:
        return []

    # 计算每个节点的度数 (与多少个其他因子 |corr| > threshold)
    abs_corr = corr.abs()
    degrees = (abs_corr > threshold).sum(axis=1) - 1  # 减去自身
    degree_order = degrees.sort_values(ascending=False).index.tolist()

    clusters: List[List[str]] = []
    assigned: set = set()

    for node in degree_order:
        if node in assigned:
            continue

        # 尝试加入已有簇
        best_cluster_idx = -1
        best_corr_in_cluster = 0
        for idx, cluster in enumerate(clusters):
            # 与簇内任一成员的最大 |corr|
            max_corr = max(abs_corr.loc[node, member] for member in cluster)
            if max_corr > threshold and max_corr > best_corr_in_cluster:
                best_corr_in_cluster = max_corr
                best_cluster_idx = idx

        if best_cluster_idx >= 0:
            clusters[best_cluster_idx].append(node)
        else:
            clusters.append([node])
        assigned.add(node)

    return clusters


def hierarchical_cluster(
    corr: pd.DataFrame,
    max_clusters: Optional[int] = None,
    distance_threshold: Optional[float] = None,
) -> Tuple[List[List[str]], dict]:
    """层次聚类: 基于 1 - |corr| 的距离矩阵做凝聚聚类.

    优点: 可视化树状图, 自动确定聚类数.
    缺点: 比 greedy 慢, 但因子数 <100 时可忽略.

    Args:
        corr: 相关性矩阵.
        max_clusters: 最大聚类数 (若指定, 自动截断).
        distance_threshold: 距离阈值 (1 - |corr|), 超过此距离不开新簇.

    Returns:
        (clusters, info) 元组.
        info 包含 silhouette_score (轮廓系数) 和 n_clusters.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    names = list(corr.index)
    n = len(names)
    if n < 2:
        return [[name] for name in names], {"n_clusters": n, "silhouette": 0}

    # 距离矩阵 = 1 - |corr|
    distance_matrix = 1 - corr.abs().values
    # 确保对称且对角线为 0
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)

    # 凝聚聚类 (平均链接)
    condensed = squareform(distance_matrix, checks=False)
    Z = linkage(condensed, method="average")

    if distance_threshold is not None:
        labels = fcluster(Z, t=distance_threshold, criterion="distance")
    elif max_clusters is not None:
        labels = fcluster(Z, t=max_clusters, criterion="maxclust")
    else:
        # 默认: 用平均距离的 1.5 倍作为阈值 (启发式)
        mean_dist = np.mean(condensed)
        labels = fcluster(Z, t=mean_dist * 1.0, criterion="distance")

    # 组织为聚类列表
    clusters: List[List[str]] = []
    label_to_cluster: Dict[int, int] = {}
    for idx, label in enumerate(labels):
        if label not in label_to_cluster:
            label_to_cluster[label] = len(clusters)
            clusters.append([])
        clusters[label_to_cluster[label]].append(names[idx])

    # 计算轮廓系数
    try:
        from sklearn.metrics import silhouette_score
        if len(set(labels)) > 1 and len(labels) > 2:
            sil = silhouette_score(distance_matrix, labels, metric="precomputed")
        else:
            sil = 0.0
    except Exception:
        sil = 0.0

    info = {
        "n_clusters": len(clusters),
        "silhouette": float(sil),
        "method": "hierarchical_average",
    }
    return clusters, info


def auto_select_threshold(
    corr: pd.DataFrame,
    thresholds: Optional[List[float]] = None,
) -> Tuple[float, List[List[str]], dict]:
    """自动选择最优聚类阈值: 用轮廓系数评估.

    遍历多个阈值, 选轮廓系数最高的.

    Args:
        corr: 相关性矩阵.
        thresholds: 候选阈值列表, 默认 [0.5, 0.6, 0.7, 0.8].

    Returns:
        (best_threshold, best_clusters, info)
    """
    if thresholds is None:
        thresholds = [0.5, 0.6, 0.7, 0.8]

    best_threshold = 0.6
    best_clusters = greedy_cluster(corr, best_threshold)
    best_sil = -1

    results = []
    for t in thresholds:
        # 用对应的距离阈值做层次聚类
        distance_threshold = 1 - t
        clusters, info = hierarchical_cluster(
            corr, distance_threshold=distance_threshold
        )
        sil = info["silhouette"]
        results.append({"threshold": t, "n_clusters": len(clusters), "silhouette": sil})
        logger.info(f"  阈值 {t:.2f}: {len(clusters)} 簇, 轮廓系数={sil:.3f}")

        if sil > best_sil:
            best_sil = sil
            best_threshold = t
            best_clusters = clusters

    return best_threshold, best_clusters, {
        "candidates": results,
        "best_threshold": best_threshold,
        "best_silhouette": best_sil,
    }


def analyze_and_save(
    factor_matrices: Dict[str, pd.DataFrame],
    significant_factors: List[dict],
    output_dir: str,
    threshold: float = 0.6,
    method: str = "greedy",
    rolling_window: Optional[int] = None,
    auto_threshold: bool = False,
    high_corr_threshold: float = 0.7,
) -> dict:
    """完整的因子相关性分析 + 聚类 + 保存.

    Args:
        factor_matrices: 因子矩阵字典.
        significant_factors: 显著因子列表 (来自 ic_by_window_period.json).
        output_dir: 输出目录 (reports/).
        threshold: 聚类阈值.
        method: 聚类方法 "greedy" 或 "hierarchical".
        rolling_window: 滚动相关性窗口, None 表示全样本.
        auto_threshold: 是否自动选阈值 (用轮廓系数).
        high_corr_threshold: 高相关对报告阈值 (通常 > 聚类阈值).

    Returns:
        分析结果字典 (同时保存为 JSON).
    """
    print("=" * 60)
    print("因子相关性分析 + 聚类")
    print("=" * 60)
    print(f"  显著因子数: {len(significant_factors)}")
    print(f"  聚类方法: {method}")
    print(f"  聚类阈值: {threshold}")
    print(f"  滚动窗口: {rolling_window or '全样本'}")
    print(f"  自动选阈值: {auto_threshold}")

    factor_names = [f["name"] for f in significant_factors]
    factor_t_stats = {f["name"]: f.get("best_t", 0) for f in significant_factors}

    # 1. 计算相关性矩阵
    print("\n[1/5] 计算因子相关性矩阵...")
    corr = compute_factor_correlation(
        factor_matrices, factor_names, rolling_window=rolling_window
    )
    if corr.empty:
        print("  失败: 无可用因子矩阵")
        return {}
    print(f"  完成: {corr.shape[0]} × {corr.shape[1]} 矩阵")

    # 2. 方向修正
    print("\n[2/5] 方向修正 (基于 t 值符号)...")
    corrected_corr, flip_signs = correct_factor_direction(corr, factor_t_stats)
    n_flipped = sum(1 for v in flip_signs.values() if v == -1)
    print(f"  完成: {n_flipped} 个因子被翻转 (t<0 的因子)")

    # 3. 提取高相关对 (用修正后的相关性)
    print(f"\n[3/5] 提取高相关对 (|corr| > {high_corr_threshold})...")
    high_corr_pairs = []
    names = list(corrected_corr.index)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c = corrected_corr.iloc[i, j]
            if abs(c) > high_corr_threshold:
                high_corr_pairs.append({
                    "f1": names[i],
                    "f2": names[j],
                    "corr": float(c),
                })
    high_corr_pairs.sort(key=lambda x: abs(x["corr"]), reverse=True)
    print(f"  完成: {len(high_corr_pairs)} 对高相关因子")

    # 4. 聚类
    print(f"\n[4/5] 聚类 (方法={method})...")
    if auto_threshold:
        threshold, clusters, auto_info = auto_select_threshold(corrected_corr)
        print(f"  自动选择阈值: {threshold:.2f} (轮廓系数={auto_info['best_silhouette']:.3f})")
        cluster_info = auto_info
    elif method == "hierarchical":
        clusters, cluster_info = hierarchical_cluster(
            corrected_corr, distance_threshold=1 - threshold
        )
        print(f"  完成: {len(clusters)} 簇 (轮廓系数={cluster_info['silhouette']:.3f})")
    else:
        clusters = greedy_cluster(corrected_corr, threshold)
        cluster_info = {"method": "greedy", "threshold": threshold}
        print(f"  完成: {len(clusters)} 簇")

    # 统计簇大小分布
    sizes = [len(c) for c in clusters]
    print(f"  簇大小分布: min={min(sizes)}, max={max(sizes)}, "
          f"mean={np.mean(sizes):.1f}, 单因子簇={sum(1 for s in sizes if s == 1)}")

    # 5. 组织输出 + 保存
    print("\n[5/5] 保存结果...")
    factor_info = {f["name"]: f for f in significant_factors}
    clusters_output = []
    for idx, cluster in enumerate(clusters, 1):
        factors_in_cluster = []
        for name in cluster:
            info = factor_info.get(name, {})
            factors_in_cluster.append({
                "name": name,
                "t": float(info.get("best_t", 0)),
                "ic": float(info.get("best_ic", 0)),
                "period": int(info.get("best_period", 0)),
                "flip": flip_signs.get(name, 1),
            })
        clusters_output.append({
            "cluster_id": idx,
            "size": len(cluster),
            "factors": factors_in_cluster,
        })

    output = {
        "n_significant": len(significant_factors),
        "high_corr_threshold": high_corr_threshold,
        "high_corr_pairs": high_corr_pairs,
        "cluster_method": method,
        "cluster_threshold": threshold,
        "rolling_window": rolling_window,
        "n_clusters": len(clusters),
        "flip_signs": flip_signs,
        "clusters": clusters_output,
        "cluster_info": cluster_info,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "factor_correlation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  结果已保存: {out_path}")

    # 绘图
    try:
        _plot_correlation(corrected_corr, clusters, output_dir)
    except Exception as e:
        print(f"  绘图失败: {e}")

    # 打印摘要
    print("\n" + "=" * 60)
    print("聚类摘要")
    print("=" * 60)
    for c in clusters_output:
        if c["size"] == 1:
            continue  # 跳过单因子簇
        factor_names_in_cluster = [f["name"] for f in c["factors"]]
        print(f"  簇{c['cluster_id']} ({c['size']}因子): {', '.join(factor_names_in_cluster)}")

    return output


def _plot_correlation(
    corr: pd.DataFrame,
    clusters: List[List[str]],
    output_dir: str,
):
    """绘制相关性热力图 + 聚类树状图."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 按聚类顺序重排因子
    ordered_names: List[str] = []
    for cluster in clusters:
        ordered_names.extend(cluster)

    corr_ordered = corr.loc[ordered_names, ordered_names]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 1. 相关性热力图
    ax = axes[0]
    im = ax.imshow(corr_ordered.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ordered_names)))
    ax.set_yticks(range(len(ordered_names)))
    ax.set_xticklabels(ordered_names, rotation=90, fontsize=6)
    ax.set_yticklabels(ordered_names, fontsize=6)
    ax.set_title("Factor Correlation Matrix (Direction-Corrected)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 2. 聚类树状图 (若 scipy 可用)
    ax = axes[1]
    try:
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform

        distance_matrix = 1 - corr_ordered.abs().values
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)
        condensed = squareform(distance_matrix, checks=False)
        Z = linkage(condensed, method="average")
        dendrogram(
            Z, labels=ordered_names, ax=ax,
            leaf_rotation=90, leaf_font_size=6,
            color_threshold=0.4,
        )
        ax.set_title("Hierarchical Clustering Dendrogram")
    except Exception as e:
        ax.text(0.5, 0.5, f"Dendrogram unavailable: {e}",
                ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "factor_correlation.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  相关性图已保存: {out_path}")


def main():
    """命令行入口: 对显著因子做相关性分析."""
    parser = argparse.ArgumentParser(description="因子相关性分析 + 聚类")
    parser.add_argument(
        "--ic-json", default=None,
        help="IC 检验结果 JSON 路径 (默认: reports/ic_by_window_period.json)")
    parser.add_argument(
        "--threshold", type=float, default=0.6,
        help="聚类阈值 |corr| > threshold (默认 0.6)")
    parser.add_argument(
        "--method", choices=["greedy", "hierarchical"], default="greedy",
        help="聚类方法 (默认 greedy)")
    parser.add_argument(
        "--rolling", type=int, default=None,
        help="滚动相关性窗口 (天数, 默认全样本)")
    parser.add_argument(
        "--auto-threshold", action="store_true",
        help="自动选最优阈值 (用轮廓系数)")
    parser.add_argument(
        "--high-corr-threshold", type=float, default=0.7,
        help="高相关对报告阈值 (默认 0.7)")
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="配置文件路径 (用于初始化 PipelineRunner)")
    args = parser.parse_args()

    # 加载显著因子
    ic_json_path = args.ic_json or os.path.join(
        _PROJECT_ROOT, "reports", "ic_by_window_period.json"
    )
    if not os.path.exists(ic_json_path):
        print(f"IC 检验结果不存在: {ic_json_path}")
        print("请先运行: python main.py research --all --multi-period --t-threshold 1.96")
        return

    significant_factors = _load_significant_factors(ic_json_path)
    if not significant_factors:
        print("无显著因子, 退出")
        return

    # 初始化 PipelineRunner, 计算因子矩阵
    import sys
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

    from core.logger import setup_logger
    from core.config import load_config
    from data.manager import DataManager
    from factors.engine import FactorEngine

    setup_logger("multi_factor")

    config_path = os.path.join(_PROJECT_ROOT, args.config) if not os.path.isabs(args.config) else args.config
    config = load_config(config_path)

    factor_names = [f["name"] for f in significant_factors]
    ic_start = pd.Timestamp(config.date_range.start)
    ic_end = pd.Timestamp(config.date_range.end)
    factor_start = ic_start - pd.Timedelta(days=365)

    print(f"计算 {len(factor_names)} 个因子矩阵 (含预热)...")
    # CR-030: 旧代码 DataManager(config) 把 FrameworkConfig 当 source 传入, 类型不匹配.
    # 改用 DataManager.from_config() 复用 Runner 的数据源工厂逻辑.
    data_mgr = DataManager.from_config(config)
    calendar = data_mgr.get_calendar(factor_start, ic_end)
    if hasattr(calendar, "tz") and calendar.tz is not None:
        calendar = calendar.tz_localize(None)
    calendar = pd.DatetimeIndex(sorted(set(calendar)))
    universe = pd.Index(config.universe) if config.universe else pd.Index([])

    engine = FactorEngine(data_mgr)
    factor_matrices = engine.compute_factors(
        factor_names, calendar, universe, parallel=False, chunk_size=100
    )
    print(f"完成: {len(factor_matrices)} 个因子矩阵")

    # 运行分析
    output_dir = os.path.join(_PROJECT_ROOT, "reports")
    analyze_and_save(
        factor_matrices=factor_matrices,
        significant_factors=significant_factors,
        output_dir=output_dir,
        threshold=args.threshold,
        method=args.method,
        rolling_window=args.rolling,
        auto_threshold=args.auto_threshold,
        high_corr_threshold=args.high_corr_threshold,
    )


if __name__ == "__main__":
    main()
