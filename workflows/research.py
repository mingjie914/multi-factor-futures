"""因子研究脚本 — 独立运行, 无需配置 PyCharm Run 参数.

Usage:
    # 研究配置文件中的因子 (默认)
    python main.py research

    # 研究指定因子
    python main.py research --factors momentum_20d,skewness_20d

    # 对指定因子执行完整多持有期筛选
    python main.py research --factors momentum_20d,skewness_20d --multi-period

    # 筛选所有已注册因子, 按 t 值排序 (单持有期, 用配置的 holding_period)
    python main.py research --all

    # 多持有期窗口匹配筛选 (推荐): 5d因子测3/5/10周期, 10d因子测5/10/20周期, 20d因子测10/20/40周期
    python main.py research --all --multi-period

    # 显式指定持有期列表: 所有因子在相同的持有期集合上测试 (周期数语义, 非天数)
    # 适用场景: "因子应在所有持有期上测试, 找到最优持有期"
    python main.py research --all --multi-period --periods 1,5,10,20,40

    # 指定周期单位 (默认 daily; 未来支持 15min/30min/hourly)
    python main.py research --all --multi-period --frequency daily

    # 单持有期探索性筛选可放宽阈值；正式多周期筛选固定使用全局 Bonferroni
    python main.py research --all --multi-period --t-threshold 1.74

    # 自定义日期范围 (IC检验区间; 因子计算自动提前1年预热)
    python main.py research --all --multi-period --start 2021-01-01 --end 2025-06-30

    # 因子相关性分析 + 聚类 (需先运行 --all --multi-period)
    python main.py research --correlation
    python main.py research --correlation --corr-method hierarchical
    python main.py research --correlation --corr-rolling 252
    python main.py research --correlation --corr-auto-threshold

周期数语义说明:
    - 因子 slug 中的 "5d" 后缀表示 "5个周期" (bar数), 不是 "5个日历天"
    - 持有期 (holding_period) 同样是 "周期数" 语义
    - 当 --frequency=daily (默认) 时, 1个周期 = 1个交易日
    - 当 --frequency=15min 时, 1个周期 = 1个15分钟bar (未来扩展)
"""
from __future__ import annotations
import sys
import os
import re
import json
import math
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from factors.library import *  # noqa: F401,F403 注册所有因子
from core.period import (
    PeriodContext,
    parse_slug_window,
    parse_holding_periods,
    holding_periods_for_window,
)


def _infer_window(factor_name: str) -> str:
    """从因子名推断窗口标签 (5d/10d/20d/other).

    委托给 core.period.parse_slug_window, 保持向后兼容.
    """
    return parse_slug_window(factor_name) or "other"


def _safe_int(val, default: int = 0) -> int:
    """安全转 int, 处理 NaN/None/空字符串."""
    try:
        if val is None:
            return default
        # NaN 检测: NaN != NaN (无需依赖 pandas)
        if isinstance(val, float) and val != val:
            return default
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_str(val, default: str = "") -> str:
    """安全转 str, 处理 NaN/None."""
    try:
        if val is None:
            return default
        if isinstance(val, float) and val != val:
            return default
        return str(val)
    except (ValueError, TypeError):
        return default


def _parse_requested_factors(raw: str | None) -> list[str]:
    """Parse a stable, de-duplicated comma-separated factor list."""
    if raw is None:
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in raw.split(","):
        name = item.strip()
        if name and name not in seen:
            result.append(name)
            seen.add(name)
    if not result:
        raise ValueError("--factors must contain at least one factor name")
    return result


def _validate_requested_factors(
    requested: list[str], available: set[str]
) -> list[str]:
    """Reject unknown factor names before starting a research run."""
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"unregistered factors: {', '.join(missing)}")
    return requested


def _apply_global_bonferroni(
    results: list[dict], family_alpha: float = 0.05
) -> tuple[int, float]:
    """Apply the global gate and select a horizon only from approved tests.

    Each factor-period pair is one hypothesis. Rejected factors deliberately
    retain ``best_period == 0`` so diagnostics cannot be mistaken for an
    approved holding-period choice.
    """
    total_hypotheses = sum(
        len(factor_result.get("all_periods", {})) for factor_result in results
    )
    cutoff = family_alpha / max(total_hypotheses, 1)

    for factor_result in results:
        factor_result.update({
            "best_period": 0,
            "best_t": 0.0,
            "best_ic_t": 0.0,
            "best_p_value": 1.0,
            "best_ic": 0.0,
            "best_ir": 0.0,
            "best_ic_pos_ratio": 0.0,
        })
        approved_rows = []
        for label, values in factor_result.get("all_periods", {}).items():
            period = int(label.replace("period_", ""))
            raw_p = float(values.get("ols_p_value", 1.0))
            valid_p = bool(
                math.isfinite(raw_p) and int(values.get("ols_n", 0)) >= 2
            )
            is_approved = bool(valid_p and raw_p <= cutoff)
            values["bonferroni_adjusted_p"] = float(
                min(max(raw_p, 0.0) * total_hypotheses, 1.0)
                if valid_p else 1.0
            )
            values["bonferroni_significant"] = is_approved
            if is_approved:
                approved_rows.append((period, values))

        if approved_rows:
            best_period, best_values = min(
                approved_rows,
                key=lambda item: (
                    float(item[1].get("ols_p_value", 1.0)),
                    -abs(float(item[1].get("ols_hac_t", 0.0))),
                    item[0],
                ),
            )
            factor_result.update({
                "best_period": int(best_period),
                "best_t": float(best_values.get("ols_hac_t", 0.0)),
                "best_ic_t": float(best_values.get("ic_hac_t", 0.0)),
                "best_p_value": float(best_values.get("ols_p_value", 1.0)),
                "best_ic": float(best_values.get("ic", 0.0)),
                "best_ir": float(best_values.get("ir_nw", 0.0)),
                "best_ic_pos_ratio": float(
                    best_values.get("ic_pos_ratio", 0.0)
                ),
            })
        factor_result["bonferroni_significant"] = bool(approved_rows)

    return total_hypotheses, cutoff


def _load_adaptivity_data(csv_path: str | None = None) -> dict:
    """加载因子适配性研究结果 (方案A: 最小集成).

    仅从显式指定的 CSV 加载, 返回 {factor_name: row_dict}.
    未指定文件时返回空字典，避免历史报告静默污染本次研究.

    适配性研究提供:
    - best_sector: 最佳板块
    - best_period: 最佳持有期 (适配性研究推荐)
    - valid_sectors: 有效板块列表 (|分隔)
    - n_valid_sectors: 有效板块数 (0 表示该因子在所有板块都无效)
    - decay_type: 衰减类型 (increasing/stable/decaying)
    """
    import pandas as pd
    if not csv_path or not os.path.exists(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception:
        return {}
    if df.empty or "factor" not in df.columns:
        return {}
    # 转为 {factor_name: row_dict} 索引
    result = {}
    for _, row in df.iterrows():
        name = row.get("factor", "")
        if not name or not isinstance(name, str):
            continue
        result[name] = row.to_dict()
    return result


def _periods_for_window(window: str) -> list:
    """窗口对应的持有期列表 (窗口匹配方案).

    委托给 core.period.holding_periods_for_window, 保持向后兼容.
    持有期为"周期数"语义 (非天数).
    """
    return holding_periods_for_window(window)


def _run_single_research(runner, factor_names, config_path):
    """运行一批因子的研究, 逐个打印结果."""
    print("=" * 60)
    print("因子研究模式")
    print("=" * 60)
    print(f"  配置文件: {config_path}")
    print(f"  因子数量: {len(factor_names)}")
    print(f"  日期范围: {runner.config.date_range.start} ~ {runner.config.date_range.end}")

    runner.config.factors = factor_names
    results = runner.run_factor_research()

    for factor_name, tests in results.items():
        print(f"\n[{factor_name}]")
        for test_name, result in tests.items():
            print(f"  {test_name}: {result.summary()}")

    print("\n研究完成.")


def _run_screening(runner, all_factors, config_path, t_threshold, output_dir):
    """批量筛选所有已注册因子 (单持有期), 按 |t| 排序输出."""
    import pandas as pd

    print("=" * 60)
    print(f"因子批量筛选 ({len(all_factors)} 个, 单持有期)")
    print("=" * 60)
    print(f"  配置文件: {config_path}")
    print(f"  t 值阈值: |t| ≥ {t_threshold}")
    print(f"  日期范围: {runner.config.date_range.start} ~ {runner.config.date_range.end}\n")

    results = []
    for i, factor_name in enumerate(all_factors, 1):
        print(f"[{i}/{len(all_factors)}] {factor_name} ... ", end="", flush=True)

        runner.config.factors = [factor_name]
        try:
            research = runner.run_factor_research()
        except Exception as e:
            print(f"失败: {e}")
            results.append({"factor": factor_name, "t_stat": 0, "status": "FAIL"})
            continue

        if factor_name not in research:
            print("无结果")
            results.append({"factor": factor_name, "t_stat": 0, "status": "NO_RESULT"})
            continue

        tests = research[factor_name]

        ic = tests.get("ic")
        ic_dict = ic.to_dict() if ic else {}
        ic_mean = ic_dict.get("ic_mean", 0)
        ir_val = ic_dict.get("ir", 0)
        t_stat = ic_dict.get("t_stat", 0)
        ic_pos = ic_dict.get("ic_pos_ratio", 0)

        layered = tests.get("layered")
        lay_dict = layered.to_dict() if layered else {}
        ls_return = lay_dict.get("group_metrics", {}).get("long_short", {}).get("annual_return", 0)
        mono = lay_dict.get("monotonicity", 0)

        reg = tests.get("regression")
        reg_dict = reg.to_dict() if reg else {}
        fm_t = list(reg_dict.get("t_stats", {}).values())
        r2 = reg_dict.get("avg_r2", 0)

        print(f"IC={ic_mean:.4f} t={t_stat:.2f} L/S={ls_return:.2%}")

        results.append({
            "factor": factor_name,
            "ic_mean": ic_mean,
            "ir": ir_val,
            "t_stat": t_stat,
            "ic_pos": ic_pos,
            "ls_return": ls_return,
            "mono": mono,
            "fm_t": fm_t[0] if fm_t else 0,
            "r2": r2,
            "status": "OK",
        })

    df = pd.DataFrame(results)
    df["abs_t"] = df["t_stat"].abs()
    df = df.sort_values("abs_t", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 80)
    print("因子筛选结果 (按 |t| 降序)")
    print("=" * 80)
    header = f"{'':>2}{'因子名':<30}{'IC_mean':>8}{'IR':>7}{'t_stat':>7}{'L/S':>8}{'Mono':>7}{'FM_t':>7}{'R²':>7}"
    print(header)
    print("-" * 80)

    significant = []
    for _, row in df.iterrows():
        if row["status"] != "OK":
            continue
        mark = "✓" if abs(row["t_stat"]) >= t_threshold else " "
        if abs(row["t_stat"]) >= t_threshold:
            significant.append(row["factor"])
        print(
            f"{mark}{row.name+1:<1} "
            f"{row['factor']:<30}"
            f"{row['ic_mean']:>8.4f}"
            f"{row['ir']:>7.3f}"
            f"{row['t_stat']:>7.2f}"
            f"{row['ls_return']:>8.2%}"
            f"{row['mono']:>7.3f}"
            f"{row['fm_t']:>7.2f}"
            f"{row['r2']:>7.3f}"
        )

    print("-" * 80)
    print(f"\n显著因子 (|t| ≥ {t_threshold}): {len(significant)} 个")
    if significant:
        print("  " + ", ".join(significant))
        print("\n复制到 config/default.yaml 的 factors 列表:")
        print("  factors:")
        for f in significant:
            print(f"    - {f}")

    csv_path = os.path.join(output_dir, "factor_screening.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存: {csv_path}")
    print("筛选完成.")


def _run_multi_period_screening(runner, all_factors, config_path, t_threshold,
                                 factor_start, ic_start, ic_end,
                                 periods_override=None, frequency="daily",
                                 output_dir=None, adaptivity_file=None):
    """多持有期窗口匹配筛选 (推荐模式).

    持有期为"周期数"语义 (非天数); 当 frequency=daily 时, 1个周期=1个交易日.

    两种持有期选取模式:
      1. 窗口匹配模式 (默认, periods_override=None):
         - 5d 因子 → 测 3/5/10 周期持有期
         - 10d 因子 → 测 5/10/20 周期持有期
         - 20d 因子 → 测 10/20/40 周期持有期
         - 其他因子 → 测 1/5/10/20 周期持有期
      2. 显式持有期模式 (periods_override=[1,5,10,20,40]):
         - 所有因子在相同的持有期集合上测试
         - 适用场景: "因子应在所有持有期上测试, 找到最优持有期"

    因子计算从 factor_start 开始 (含1年预热), IC 检验从 ic_start 开始.
    使用 Newey-West HAC 调整 t 统计量.

    Args:
        periods_override: 显式持有期列表 (周期数). None 表示用窗口匹配模式.
        frequency: 周期单位 ("daily"/"15min"/"30min"/"hourly"). 默认 "daily".
                   当前仅用于日志显示和 PeriodContext 构造, 不影响计算逻辑
                   (因为非日度周期需要 DataProvider 频率感知支持).
    """
    import numpy as np
    import pandas as pd
    from data.manager import DataManager
    from factors.engine import FactorEngine
    from core.interfaces import ProcessingContext
    from testing.ic_test import ICTest
    from testing.regression import (
        _newey_west_t_stat,
        _vectorized_univariate_ols,
    )
    from scipy import stats as _scipy_stats

    # 构造 PeriodContext (当前仅用于日志, 未来扩展分钟周期时传入计算)
    try:
        period_ctx = PeriodContext.from_string(frequency)
    except ValueError:
        print(f"  警告: 不支持的 frequency={frequency!r}, 回退到 daily")
        period_ctx = PeriodContext.from_string("daily")

    print("=" * 60)
    print(f"因子多持有期窗口匹配筛选 ({len(all_factors)} 个)")
    print("=" * 60)
    print(f"  配置文件: {config_path}")
    print(f"  因子计算区间: {factor_start.date()} ~ {ic_end.date()} (含预热)")
    print(f"  IC检验区间: {ic_start.date()} ~ {ic_end.date()}")
    print("  正式统计门槛: 未收缩单因子 OLS/HAC p 值 → 全局 Bonferroni")
    print(f"  周期单位: {period_ctx.unit.value} (1个周期 = "
          f"{period_ctx.bars_per_day}个bar/交易日)")

    # 持有期选取模式
    if periods_override is not None:
        print(f"  持有期模式: 显式列表 {periods_override} (所有因子测试相同持有期)")
    else:
        print(f"  持有期模式: 窗口匹配 (5d→[3,5,10], 10d→[5,10,20], 20d→[10,20,40])")

    # 方案A: 加载适配性研究结果, 过滤无效因子
    adaptivity_data = _load_adaptivity_data(adaptivity_file)
    n_filtered = 0  # 预初始化, 避免作用域问题
    if adaptivity_data:
        n_before = len(all_factors)
        # 过滤 n_valid_sectors=0 的因子 (适配性研究确认在所有板块都无效)
        # 同时保留不在适配性研究中的因子 (新因子未测试, 不强制过滤)
        all_factors = [
            f for f in all_factors
            if f not in adaptivity_data
            or _safe_int(adaptivity_data[f].get("n_valid_sectors", 0)) > 0
        ]
        n_filtered = n_before - len(all_factors)
        print(f"  适配性研究: 已加载 {len(adaptivity_data)} 个因子记录")
        if n_filtered > 0:
            print(f"  适配性过滤: 移除 {n_filtered} 个无效因子 (n_valid_sectors=0)")
        print()
    else:
        print("  适配性研究: 未指定有效输入文件，使用纯窗口匹配")
        print()

    # 按窗口分组 (仍需要窗口信息用于结果展示, 即使持有期被覆盖)
    window_groups: dict[str, list[str]] = {}
    for f in all_factors:
        w = _infer_window(f)
        window_groups.setdefault(w, []).append(f)
    print("  分组: " + ", ".join(f"{k}={len(v)}" for k, v in window_groups.items()))
    print()

    data_mgr = runner.data_manager
    universe = pd.Index(runner.config.universe) if runner.config.universe else pd.Index([])

    # 优先从缓存获取交易日历 (绕过 MySQL 连接问题)
    calendar = data_mgr.get_calendar(factor_start, ic_end)
    if hasattr(calendar, "tz") and calendar.tz is not None:
        calendar = calendar.tz_localize(None)
    calendar = pd.DatetimeIndex(sorted(set(calendar)))

    # 若 MySQL 连接失败导致日历为空, 从缓存的 close parquet 提取交易日历
    if len(calendar) == 0:
        import os as _os
        _cache_dir = _os.path.join(_PROJECT_ROOT, "cache")
        _close_files = [f for f in _os.listdir(_cache_dir)
                        if f.startswith("futures_MySQLSource_close_") and f.endswith(".parquet")]
        _best_dates = pd.DatetimeIndex([])
        _best_score = -1
        for _f in _close_files:
            try:
                _df = pd.read_parquet(_os.path.join(_cache_dir, _f))
                _col_overlap = len(set(_df.columns) & set(universe))
                _date_overlap = ((_df.index >= pd.Timestamp(factor_start)) &
                                 (_df.index <= pd.Timestamp(ic_end))).sum()
                _score = _col_overlap * 1000 + _date_overlap
                if _score > _best_score:
                    _best_score = _score
                    _best_dates = pd.DatetimeIndex(_df.index)
            except Exception:
                continue
        if len(_best_dates) > 0:
            calendar = _best_dates
            # 同时 monkey-patch DataManager.get 使其从缓存返回数据
            _ohlcv_cache = {"close": pd.read_parquet(
                _os.path.join(_cache_dir, _close_files[0])
            ).reindex(index=calendar, columns=universe)}
            for _field in ["open", "high", "low", "volume", "oi", "settle", "amount"]:
                for _f in _os.listdir(_cache_dir):
                    if _f.startswith(f"futures_MySQLSource_{_field}_") and _f.endswith(".parquet"):
                        try:
                            _df = pd.read_parquet(_os.path.join(_cache_dir, _f))
                            _df = _df.reindex(index=calendar, columns=universe)
                            _ohlcv_cache[_field] = _df
                        except Exception:
                            pass
                        break
            def _patched_get(field, dates, uni):
                if field in _ohlcv_cache:
                    return _ohlcv_cache[field].reindex(index=dates, columns=uni)
                return pd.DataFrame(index=dates, columns=uni)
            data_mgr.get = _patched_get
            print(f"  (从缓存加载数据: {len(_ohlcv_cache)} 个字段)")

    calendar = pd.DatetimeIndex(sorted(set(calendar)))
    print(f"交易日历: {len(calendar)} 天, universe: {len(universe)} 品种")

    # 计算所有因子矩阵 (一次性, 串行最稳定)
    print(f"\n计算 {len(all_factors)} 个因子矩阵 (含预热期)...")
    t0 = time.time()
    engine = FactorEngine(data_mgr)
    factor_matrices = engine.compute_factors(
        all_factors, calendar, universe, parallel=False, chunk_size=100
    )
    processing_context = ProcessingContext(
        data=data_mgr, dates=calendar, universe=universe
    )
    factor_matrices = runner.processor.process_batch(
        factor_matrices, processing_context
    )
    print(f"  完成: {len(factor_matrices)} 个因子, 耗时 {time.time()-t0:.1f}s")

    # 预计算各持有期的 forward returns
    print("\n预计算 forward returns...")
    fwd_returns_by_period: dict[int, pd.DataFrame] = {}
    # 持有期集合: 显式模式 → periods_override; 窗口匹配模式 → 各窗口并集
    if periods_override is not None:
        all_periods = sorted(set(periods_override))
    else:
        all_periods = sorted({p for w in window_groups.keys() for p in _periods_for_window(w)})
    for p in all_periods:
        fwd_returns_by_period[p] = data_mgr.get_forward_returns(
            calendar, universe, period=p
        )
    print(f"  持有期 (周期数): {all_periods}")

    # 批量 IC 检验 (向量化)
    print("\n=== 开始 IC 检验 ===")
    results = []
    total = len(all_factors)
    t0 = time.time()

    for idx, fname in enumerate(all_factors, 1):
        if fname not in factor_matrices:
            continue
        window = _infer_window(fname)
        # 持有期: 显式模式 → periods_override; 窗口匹配模式 → 按窗口推断
        if periods_override is not None:
            periods = list(periods_override)
        else:
            periods = _periods_for_window(window)

        f_mat = factor_matrices[fname]
        f_ic = f_mat.loc[ic_start:ic_end]

        all_period_results = {}
        for p in periods:
            fwd = fwd_returns_by_period[p].loc[ic_start:ic_end]
            # Census selection consumes Pearson IC only. Avoid computing and
            # discarding Spearman ranks and decay diagnostics for every test.
            test = ICTest(
                methods=["pearson"], decay_periods=[], forward_period=p
            )
            try:
                res = test.run(f_ic, fwd)
                t_stat = res.t_stat
                ic_mean = res.ic_mean
                ir_nw = res.ir_newey_west
                ic_pos_ratio = (res.ic_series > 0).mean() if len(res.ic_series) > 0 else 0.0
                ols_slopes = _vectorized_univariate_ols(
                    f_ic, fwd, min_stocks=10
                )
                ols_t = _newey_west_t_stat(ols_slopes, forward_period=p)
                ols_p = float(
                    2.0 * _scipy_stats.t.sf(
                        abs(ols_t), df=max(len(ols_slopes) - 1, 1)
                    )
                )
                ols_beta = float(ols_slopes.mean()) if len(ols_slopes) else 0.0
            except Exception:
                t_stat = 0.0
                ic_mean = 0.0
                ir_nw = 0.0
                ic_pos_ratio = 0.0
                ols_t = 0.0
                ols_p = 1.0
                ols_beta = 0.0
                ols_slopes = pd.Series(dtype=float)
                res = None

            all_period_results[f"period_{p}"] = {
                "ic": float(ic_mean),
                "ic_hac_t": float(t_stat),
                "t": float(ols_t),
                "ols_beta": float(ols_beta),
                "ols_hac_t": float(ols_t),
                "ols_p_value": float(ols_p),
                "ols_n": int(len(ols_slopes)),
                "inference_model": "unpenalized_univariate_fama_macbeth_ols_hac",
                "ir_nw": float(ir_nw),
                "ic_pos_ratio": float(ic_pos_ratio),
                "n": int(res.n_obs) if res else 0,
            }

        # CR-002: Bonferroni 校正改为全局 m 假设数校正 (不再用 t×√n_periods)
        # 全局 m = n_factors × n_periods, 在筛选阶段统一计算
        n_periods = len(periods)

        results.append({
            "name": fname,
            "window": window,
            "best_period": 0,
            "best_t": 0.0,
            "best_ic_t": 0.0,
            "best_p_value": 1.0,
            "best_ic": 0.0,
            "best_ir": 0.0,
            "best_ic_pos_ratio": 0.0,
            "all_periods": all_period_results,
            "n_periods_tested": n_periods,
            # 方案A: 适配性研究信息 (供输出层展示, 不影响筛选逻辑)
            "adaptivity_best_sector": _safe_str(adaptivity_data.get(fname, {}).get("best_sector", "")),
            "adaptivity_valid_sectors": _safe_str(adaptivity_data.get(fname, {}).get("valid_sectors", "")),
            "adaptivity_n_valid_sectors": _safe_int(adaptivity_data.get(fname, {}).get("n_valid_sectors", 0)),
            "adaptivity_recommended_period": _safe_int(adaptivity_data.get(fname, {}).get("recommended_period", 0)),
            "adaptivity_decay_type": _safe_str(adaptivity_data.get(fname, {}).get("decay_type", "")),
        })

        if idx % 50 == 0 or idx == total:
            elapsed = time.time() - t0
            print(f"  [{idx}/{total}] 耗时 {elapsed:.1f}s")

    # 筛选显著因子 (业界标准筛选流程)
    # 门槛1: t值显著性 (Bonferroni校正后)
    # 门槛2: IC+IR双门槛 (|IC|>0.02 且 IR>0.5)
    # 门槛3: IC命中率 > 52%
    IC_THRESHOLD = 0.02
    IR_THRESHOLD = 0.50
    IC_POS_THRESHOLD = 0.52

    def _passes_ic_ir(r):
        """IC+IR+命中率三重门槛 (命中率按主方向统计)."""
        # 命中率按IC主方向统计: 正向因子看IC>0比例, 反向因子看IC<0比例
        if r["best_ic"] >= 0:
            pos_ok = r["best_ic_pos_ratio"] >= IC_POS_THRESHOLD
        else:
            pos_ok = (1.0 - r["best_ic_pos_ratio"]) >= IC_POS_THRESHOLD
        return (
            abs(r["best_ic"]) >= IC_THRESHOLD
            and abs(r["best_ir"]) >= IR_THRESHOLD
            and pos_ok
        )

    # 第一阶段: 对每个预声明因子×持有期的原始单因子 OLS/HAC p 值
    # 执行全局 Bonferroni。Ridge 尚未参与，也不产生筛选 p 值。
    total_hypotheses, bonferroni_alpha = _apply_global_bonferroni(results)
    bonferroni_t_threshold = float(_scipy_stats.norm.ppf(1 - bonferroni_alpha / 2))

    print(f"\nBonferroni 校正: m={total_hypotheses} 假设, "
          f"alpha_adj={bonferroni_alpha:.6f}, "
          f"参考正态阈值={bonferroni_t_threshold:.3f}")

    t_significant = [r for r in results if r["bonferroni_significant"]]
    # 第二阶段: IC+IR双门槛筛选
    significant = [r for r in t_significant if _passes_ic_ir(r)]
    rejected_by_ir = [r for r in t_significant if not _passes_ic_ir(r)]
    significant.sort(key=lambda x: abs(x["best_t"]), reverse=True)
    governance_cfg = getattr(runner.config, "factor_governance", None)
    governance_enabled = bool(
        governance_cfg is not None and governance_cfg.enabled
    )
    family_governance_audit = {
        "enabled": governance_enabled,
        "applied": False,
        "deferred_until": "after_correlation_deduplication",
    }

    by_period: dict[int, list[str]] = {}
    for r in significant:
        by_period.setdefault(r["best_period"], []).append(r["name"])

    # 统计校正信息 (CR-002: 使用全局 Bonferroni 校正)
    corr_info = f"m={total_hypotheses}, raw OLS/HAC p≤{bonferroni_alpha:.3g}"

    print("\n" + "=" * 80)
    print(f"OLS/IC 检验完成: Bonferroni 显著 {len(t_significant)} 个 → IC+IR双门槛后 {len(significant)} 个")
    print(f"  门槛: OLS/HAC p≤{bonferroni_alpha:.3g}, |IC|≥{IC_THRESHOLD}, |IR|≥{IR_THRESHOLD}, IC命中率≥{IC_POS_THRESHOLD:.0%}")
    print(f"  Bonferroni校正: {corr_info}")
    if rejected_by_ir:
        print(f"  IR门槛淘汰: {len(rejected_by_ir)} 个 (t显著但IC/IR/命中率不达标)")
    print("=" * 80)
    print(f"\n按最优持有期分组:")
    for p in sorted(by_period.keys()):
        print(f"  {p}日: {len(by_period[p])} 个")

    print(f"\n显著因子清单 (按 |t| 降序, 含适配性信息):")
    print(f"{'#':>3} {'因子名':<35} {'窗口':>5} {'持有期':>5} {'t值':>7} {'阈值':>7} {'IC':>8} {'IR':>6} {'命中率':>6} {'最佳板块':>8} {'有效板块数':>8}")
    print("-" * 115)
    for i, r in enumerate(significant, 1):
        best_sec = r.get("adaptivity_best_sector", "") or "-"
        n_valid = r.get("adaptivity_n_valid_sectors", 0)
        print(f"{i:>3} {r['name']:<35} {r['window']:>5} {r['best_period']:>5}d "
              f"{r['best_t']:>7.2f} {bonferroni_t_threshold:>7.2f} {r['best_ic']:>8.4f} "
              f"{r['best_ir']:>6.3f} {r['best_ic_pos_ratio']:>6.1%} {best_sec:>8} {n_valid:>8}")

    # ===================================================================
    # 后置检验: 分层单调性 + 换手率 + 稳健性 (仅对IC+IR显著因子)
    # ===================================================================
    final_factors: list = []  # 预初始化, 避免后续引用未定义变量
    candidate_return_series: dict[str, "pd.Series"] = {}
    selection_diagnostics = {
        "deflated_sharpe": {
            "n_trials": int(total_hypotheses),
            "return_sampling": "non_overlapping_at_selected_holding_period",
        },
        "pbo_cscv": {"pbo": None, "n_splits": 0, "n_candidates": 0},
    }
    if significant:
        from testing.layered import LayeredBacktest
        from testing.turnover import TurnoverTest
        from testing.robustness import RobustnessTest
        from research.statistics import (
            deflated_sharpe_ratio,
            probability_backtest_overfitting,
        )

        print("\n=== 后置检验: 分层单调性 / 换手率 / 稳健性 ===")
        layered_test = LayeredBacktest(n_groups=5)
        turnover_test = TurnoverTest()
        robustness_test = RobustnessTest(n_segments=4)

        for r in significant:
            fname = r["name"]
            best_p = r["best_period"]
            if fname not in factor_matrices:
                continue
            f_mat = factor_matrices[fname].loc[ic_start:ic_end]
            fwd = fwd_returns_by_period[best_p].loc[ic_start:ic_end]

            # 分层单调性
            failures = r.setdefault("post_test_failures", [])
            try:
                layered_res = layered_test.run(
                    f_mat, fwd, holding_period=best_p
                )
                r["layered_monotonicity"] = float(layered_res.monotonicity)
                r["layered_ls_return"] = float(
                    layered_res.group_metrics.get("long_short", {}).get("annual_return", 0.0)
                )
                direction = 1.0 if r["best_ic"] >= 0 else -1.0
                oriented_returns = (
                    layered_res.group_returns["long_short"].dropna() * direction
                )
                candidate_return_series[fname] = oriented_returns
                non_overlapping = oriented_returns.iloc[::max(int(best_p), 1)]
                dsr = deflated_sharpe_ratio(
                    non_overlapping,
                    n_trials=total_hypotheses,
                    periods_per_year=max(
                        int(period_ctx.bars_per_year / max(int(best_p), 1)), 1
                    ),
                    risk_free_rate=0.0,
                )
                r["deflated_sharpe"] = dsr
                r["passes_dsr_95"] = bool(dsr["probability"] >= 0.95)
            except Exception as exc:
                r["layered_monotonicity"] = 0.0
                r["layered_ls_return"] = 0.0
                r["deflated_sharpe"] = {
                    "sharpe": 0.0,
                    "expected_max_sharpe": 0.0,
                    "probability": 0.0,
                    "n_obs": 0,
                    "n_trials": int(total_hypotheses),
                    "risk_free_rate": 0.0,
                }
                r["passes_dsr_95"] = False
                failures.append({
                    "stage": "layered_and_dsr",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                })

            # 换手率
            try:
                turnover_res = turnover_test.run(f_mat)
                r["monthly_turnover"] = float(turnover_res.monthly_turnover)
                r["passes_turnover"] = bool(turnover_res.monthly_turnover < 0.50)
            except Exception as exc:
                r["monthly_turnover"] = 0.0
                r["passes_turnover"] = False
                failures.append({
                    "stage": "turnover",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                })

            # 稳健性 (时间分段)
            try:
                robust_res = robustness_test.run(f_mat, fwd)
                r["ic_sign_consistency"] = float(robust_res.ic_sign_consistency)
                r["ir_std"] = float(robust_res.ir_std)
                r["passes_robustness"] = bool(robust_res.passes_ic_consistency and robust_res.passes_ir_stability)
            except Exception as exc:
                r["ic_sign_consistency"] = 0.0
                r["ir_std"] = 0.0
                r["passes_robustness"] = False
                failures.append({
                    "stage": "robustness",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                })

        if len(candidate_return_series) >= 2:
            candidate_frame = pd.concat(
                candidate_return_series, axis=1, join="inner"
            ).dropna(how="any")
            pbo = probability_backtest_overfitting(
                candidate_frame, n_partitions=8
            )
            pbo_value = pbo.get("pbo")
            if pbo_value is not None and not np.isfinite(pbo_value):
                pbo_value = None
            selection_diagnostics["pbo_cscv"] = {
                "pbo": pbo_value,
                "n_splits": int(pbo.get("n_splits", 0)),
                "n_candidates": int(candidate_frame.shape[1]),
                "n_observations": int(candidate_frame.shape[0]),
                "time_partitioning": "8_contiguous_blocks",
                "logits": pbo.get("logits", []),
            }

        # 打印后置检验摘要
        print(f"\n{'#':>3} {'因子名':<35} {'单调性':>6} {'多空':>7} {'月换手':>7} {'IC一致':>7} {'稳健':>4}")
        print("-" * 85)
        for i, r in enumerate(significant, 1):
            robust_str = "✓" if r.get("passes_robustness") else "✗"
            print(f"{i:>3} {r['name']:<35} {r.get('layered_monotonicity',0):>6.3f} "
                  f"{r.get('layered_ls_return',0):>7.2%} {r.get('monthly_turnover',0):>7.2%} "
                  f"{r.get('ic_sign_consistency',0):>7.1%} {robust_str:>4}")

        # 最终筛选: 通过全部检验的因子
        # 门槛: 单调性 >= 0.5 (Q1-Q5年化收益与组号相关系数, 业界标准)
        MONO_THRESHOLD = 0.5
        final_factors = [
            r for r in significant
            if r.get("passes_turnover", False)
            and r.get("passes_robustness", False)
            and abs(r.get("layered_monotonicity", 0.0)) >= MONO_THRESHOLD
        ]
        rejected_final = [r for r in significant if r not in final_factors]
        rejected_by_mono = [
            r for r in significant
            if r.get("passes_turnover", False)
            and r.get("passes_robustness", False)
            and abs(r.get("layered_monotonicity", 0.0)) < MONO_THRESHOLD
        ]
        print(f"\n最终通过全部检验: {len(final_factors)} 个 "
              f"(换手率+稳健性+单调性淘汰 {len(rejected_final)} 个, "
              f"其中单调性淘汰 {len(rejected_by_mono)} 个)")

    # 保存 JSON 结果
    final_factor_names = (
        [r["name"] for r in final_factors] if significant else []
    )
    out = {
        "config": {
            "factor_start": str(factor_start.date()),
            "ic_start": str(ic_start.date()),
            "ic_end": str(ic_end.date()),
            "t_threshold": bonferroni_t_threshold,
            "bonferroni_correction": True,
            "correction_formula": "raw_unpenalized_ols_hac_p <= 0.05 / m",
            "inference_model": "unpenalized_univariate_fama_macbeth_ols_hac",
            "factor_preprocessing": [
                "mad_winsorize_by_date",
                "zscore_standardize_by_date",
            ],
            "selection_order": [
                "predeclared_exposure_preprocessing",
                "raw_ols_hac_p_values",
                "global_bonferroni",
                "economic_and_robustness_gates",
                "layered_turnover_stability_diagnostics",
            ],
            "downstream_required_order": [
                "correlation_deduplication",
                "family_governance",
                "training_only_ridge",
            ],
            "total_hypotheses": total_hypotheses,
            "bonferroni_alpha": bonferroni_alpha,
            "ic_threshold": IC_THRESHOLD,
            # 周期架构信息 (新增字段, 向后兼容)
            "frequency": period_ctx.unit.value,
            "periods_mode": "explicit" if periods_override is not None else "window_match",
            "periods_override": periods_override,
            "periods_semantics": "周期数 (非天数); daily频率下1周期=1交易日",
            "bars_per_year": period_ctx.bars_per_year,
            "ir_threshold": IR_THRESHOLD,
            "ic_pos_threshold": IC_POS_THRESHOLD,
            "mono_threshold": 0.5,
            "n_factors_total": len(all_factors),
            "n_factors_t_significant": len(t_significant),
            "n_factors_significant": len(significant),
            # 方案A: 适配性研究集成信息
            "adaptivity_enabled": bool(adaptivity_data),
            "adaptivity_source": os.path.abspath(adaptivity_file) if adaptivity_data else None,
        },
        "significant_factors": significant,
        "all_results": results,
        "final_factors": final_factor_names,
        "selection_diagnostics": selection_diagnostics,
        "family_governance": family_governance_audit,
        "summary": {
            "by_window": {k: len(v) for k, v in window_groups.items()},
            "by_best_period": {str(k): len(v) for k, v in by_period.items()},
            "total_significant": len(significant),
            "total_passed_all_tests": len(final_factors) if significant else 0,
            # 方案A: 适配性摘要
            "adaptivity_filtered_invalid": n_filtered if adaptivity_data else 0,
        },
    }
    out_path = os.path.join(output_dir, "ic_by_window_period.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")

    # 输出 YAML 配置片段 (优先用 final_factors, 若后置检验未运行则回退 significant)
    yaml_source = final_factors if significant and final_factors else significant
    print("\n=== config/default.yaml 子组合配置片段 ===")
    print(f"# (基于 {'final_factors' if final_factors else 'significant_factors'})")
    print(f"# 方案A: 含适配性研究标注 (best_sector / valid_sectors)")
    short_factors = [r for r in yaml_source if r["best_period"] in (3, 5)]
    mid_factors = [r for r in yaml_source if r["best_period"] in (10,)]
    long_factors = [r for r in yaml_source if r["best_period"] in (20, 40)]

    def _format_factor_line(r: dict) -> str:
        """格式化因子行, 含适配性标注 (如有)."""
        name = r["name"]
        best_sec = r.get("adaptivity_best_sector", "")
        valid_secs = r.get("adaptivity_valid_sectors", "")
        if best_sec and valid_secs:
            return f"  - {name}  # best_sector={best_sec}, valid={valid_secs}"
        return f"  - {name}"

    print(f"\n# 短期子组合 ({len(short_factors)}个, 3-5日持有期):")
    for r in short_factors:
        print(_format_factor_line(r))
    print(f"\n# 中期子组合 ({len(mid_factors)}个, 10日持有期):")
    for r in mid_factors:
        print(_format_factor_line(r))
    print(f"\n# 长期子组合 ({len(long_factors)}个, 20-40日持有期):")
    for r in long_factors:
        print(_format_factor_line(r))

    print("\n筛选完成.")


def _run_correlation_analysis(
    runner, config_path, factor_start, ic_start, ic_end,
    method="greedy", threshold=0.6, rolling_window=None, auto_threshold=False,
    output_dir=None, screening_file=None,
):
    """对显著因子做相关性分析 + 聚类.

    流程:
    1. 从显式研究目录或筛选文件加载显著因子
    2. 计算因子矩阵 (含预热期)
    3. 调用 factors.correlation_analysis.analyze_and_save 做分析
    4. 输出 factor_correlation.json + .png 到本次研究目录

    Usage:
        python main.py research --correlation
        python main.py research --correlation --corr-method hierarchical
        python main.py research --correlation --corr-rolling 252
        python main.py research --correlation --corr-auto-threshold
    """
    import pandas as pd
    from factors.engine import FactorEngine
    from factors.correlation_analysis import analyze_and_save, _load_significant_factors

    print("=" * 60)
    print("因子相关性分析 + 聚类")
    print("=" * 60)
    print(f"  配置文件: {config_path}")
    print(f"  因子计算区间: {factor_start.date()} ~ {ic_end.date()} (含预热)")
    print(f"  相关性计算区间: {ic_start.date()} ~ {ic_end.date()}")
    print(f"  聚类方法: {method}, 阈值: {threshold}")
    print(f"  滚动窗口: {rolling_window or '全样本'}")
    print(f"  自动选阈值: {auto_threshold}")

    # 加载显著因子
    ic_json_path = screening_file or os.path.join(output_dir, "ic_by_window_period.json")
    if not os.path.exists(ic_json_path):
        print(f"\nIC 检验结果不存在: {ic_json_path}")
        print("请先运行: python main.py research --all --multi-period --t-threshold 1.96")
        return

    significant_factors = _load_significant_factors(ic_json_path)
    if not significant_factors:
        print("无显著因子, 退出")
        return

    factor_names = [f["name"] for f in significant_factors]
    print(f"\n显著因子数: {len(factor_names)}")

    # 计算因子矩阵
    data_mgr = runner.data_manager
    calendar = data_mgr.get_calendar(factor_start, ic_end)
    if hasattr(calendar, "tz") and calendar.tz is not None:
        calendar = calendar.tz_localize(None)
    calendar = pd.DatetimeIndex(sorted(set(calendar)))
    universe = pd.Index(runner.config.universe) if runner.config.universe else pd.Index([])

    print(f"\n计算 {len(factor_names)} 个因子矩阵...")
    t0 = time.time()
    engine = FactorEngine(data_mgr)
    factor_matrices = engine.compute_factors(
        factor_names, calendar, universe, parallel=False, chunk_size=100
    )
    print(f"  完成: {len(factor_matrices)} 个因子, 耗时 {time.time()-t0:.1f}s")

    # 只保留 IC 检验区间的数据用于相关性计算
    for name in factor_matrices:
        factor_matrices[name] = factor_matrices[name].loc[ic_start:ic_end]

    # 运行分析
    analyze_and_save(
        factor_matrices=factor_matrices,
        significant_factors=significant_factors,
        output_dir=output_dir,
        threshold=threshold,
        method=method,
        rolling_window=rolling_window,
        auto_threshold=auto_threshold,
        high_corr_threshold=0.7,
    )


def main():
    import argparse
    import pandas as pd

    try:
        from core.logger import setup_logger
        from pipeline.runner import PipelineRunner
        from core.registry import list_registered
    except ImportError as e:
        print(f"框架模块导入失败: {e}")
        print(f"   请安装依赖: python -m pip install -r requirements-minimal.txt")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="多因子研究 — 因子 IC/分层/回归检验")
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="配置文件路径 (默认: config/default.yaml)")
    parser.add_argument(
        "--factors", default=None,
        help="指定因子, 逗号分隔 (如: momentum_20d,skewness_20d)")
    parser.add_argument(
        "--all", action="store_true",
        help="批量筛选所有已注册因子, 按 |t| 排序")
    parser.add_argument(
        "--multi-period", action="store_true",
        help="多持有期窗口匹配筛选 (推荐): 5d因子测3/5/10周期, 10d因子测5/10/20周期, "
             "20d因子测10/20/40周期 (持有期为周期数, 非天数)")
    parser.add_argument(
        "--periods", default=None,
        help="显式指定持有期列表 (逗号分隔, 周期数语义), 如 '1,5,10,20,40'. "
             "指定后所有因子在相同持有期集合上测试, 覆盖窗口匹配模式. "
             "适用场景: 因子应在所有持有期上测试, 找到最优持有期. "
             "需配合 --multi-period 使用")
    parser.add_argument(
        "--frequency", default="daily",
        choices=["daily", "15min", "30min", "hourly"],
        help="周期单位 (默认 daily). 非日度周期需要 DataProvider 频率感知支持, "
             "当前仅 daily 完整可用")
    parser.add_argument(
        "--t-threshold", type=float, default=1.96,
        help="筛选时 t 值绝对值阈值 (默认 1.96, 即 95%% 置信度)")
    parser.add_argument(
        "--start", default=None,
        help="IC检验起始日期 (默认: 配置文件的 date_range.start)")
    parser.add_argument(
        "--end", default=None,
        help="IC检验结束日期 (默认: 配置文件的 date_range.end)")
    parser.add_argument(
        "--factor-start", default=None,
        help="因子计算起始日 (含预热期, 默认: IC检验起始日往前1年)")
    parser.add_argument(
        "--correlation", action="store_true",
        help="对显著因子做相关性分析 + 聚类, 生成 factor_correlation.json "
             "(需先运行 --all --multi-period 生成 ic_by_window_period.json)")
    parser.add_argument(
        "--corr-method", choices=["greedy", "hierarchical"], default="greedy",
        help="聚类方法 (默认 greedy, 配合 --correlation 使用)")
    parser.add_argument(
        "--corr-threshold", type=float, default=0.6,
        help="聚类阈值 |corr| > threshold (默认 0.6, 配合 --correlation)")
    parser.add_argument(
        "--corr-rolling", type=int, default=None,
        help="滚动相关性窗口天数 (默认全样本, 配合 --correlation)")
    parser.add_argument(
        "--corr-auto-threshold", action="store_true",
        help="自动选最优聚类阈值 (用轮廓系数, 配合 --correlation)")
    parser.add_argument(
        "--run-id", default=None,
        help="研究运行标识；默认按当前时间生成，输出到 runs/<run_id>")
    parser.add_argument(
        "--output-dir", default=None,
        help="显式输出目录；指定后覆盖 runs/<run_id>")
    parser.add_argument(
        "--refuse-existing-output", action="store_true",
        help="若输出目录已存在则拒绝运行；用于不可覆盖的规范化研究")
    parser.add_argument(
        "--cache-only", action="store_true",
        help="严格只使用本地缓存；缓存未命中时禁止访问数据库")
    parser.add_argument(
        "--adaptivity-file", default=None,
        help="显式适配性 CSV。默认不加载历史 reports 产物")
    parser.add_argument(
        "--screening-file", default=None,
        help="相关性分析使用的 ic_by_window_period.json；默认读取本次输出目录")
    args = parser.parse_args()

    if args.all and args.factors:
        parser.error("--all 与 --factors 不能同时使用")
    if args.periods and not args.multi_period:
        parser.error("--periods 必须配合 --multi-period 使用")

    if args.run_id and not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_id):
        parser.error("--run-id 仅允许字母、数字、点、下划线和连字符")
    run_id = args.run_id or time.strftime("research_%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(_PROJECT_ROOT, "runs", run_id)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(_PROJECT_ROOT, output_dir)
    output_dir = os.path.normpath(output_dir)
    if args.refuse_existing_output and os.path.exists(output_dir):
        parser.error(f"输出目录已存在，拒绝覆盖: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    adaptivity_file = args.adaptivity_file
    if adaptivity_file and not os.path.isabs(adaptivity_file):
        adaptivity_file = os.path.join(_PROJECT_ROOT, adaptivity_file)
    if adaptivity_file and not os.path.isfile(adaptivity_file):
        parser.error(f"--adaptivity-file 不存在: {adaptivity_file}")

    screening_file = args.screening_file
    if screening_file and not os.path.isabs(screening_file):
        screening_file = os.path.join(_PROJECT_ROOT, screening_file)
    if screening_file and not os.path.isfile(screening_file):
        parser.error(f"--screening-file 不存在: {screening_file}")

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(_PROJECT_ROOT, config_path)
    config_path = os.path.normpath(config_path)

    setup_logger("multi_factor")

    try:
        from core.config import load_config

        config = load_config(config_path)
        if args.cache_only:
            config.data.cache["only"] = True
        runner = PipelineRunner(config=config)
    except Exception as e:
        print(f"框架初始化失败: {e}")
        print(f"   请检查配置文件: {config_path}")
        sys.exit(1)
    runner.config.backtest.report_dir = output_dir
    print(f"  研究输出目录: {output_dir}")

    # 日期范围
    ic_start = pd.Timestamp(args.start) if args.start else pd.Timestamp(runner.config.date_range.start)
    ic_end = pd.Timestamp(args.end) if args.end else pd.Timestamp(runner.config.date_range.end)
    factor_start = pd.Timestamp(args.factor_start) if args.factor_start else (ic_start - pd.Timedelta(days=365))
    if ic_start > ic_end:
        parser.error("--start 必须早于或等于 --end")
    runner.config.date_range.start = str(ic_start.date())
    runner.config.date_range.end = str(ic_end.date())

    if args.correlation:
        # 因子相关性分析 + 聚类 (需先运行 --all --multi-period)
        _run_correlation_analysis(
            runner, config_path, factor_start, ic_start, ic_end,
            method=args.corr_method,
            threshold=args.corr_threshold,
            rolling_window=args.corr_rolling,
            auto_threshold=args.corr_auto_threshold,
            output_dir=output_dir,
            screening_file=screening_file,
        )
    elif args.all or args.multi_period:
        available_factors = set(
            list_registered("factor").get("factor", {}).keys()
        )
        if args.all:
            selected_factors = sorted(available_factors)
        else:
            try:
                requested = _parse_requested_factors(args.factors)
                selected_factors = _validate_requested_factors(
                    requested, available_factors
                )
            except ValueError as exc:
                parser.error(str(exc))
        if args.multi_period:
            # 解析 --periods (显式持有期列表, 可选)
            periods_override = parse_holding_periods(args.periods)
            _run_multi_period_screening(
                runner, selected_factors, config_path, args.t_threshold,
                factor_start, ic_start, ic_end,
                periods_override=periods_override,
                frequency=args.frequency,
                output_dir=output_dir,
                adaptivity_file=adaptivity_file,
            )
        else:
            # 单持有期模式: 同步配置日期
            runner.config.date_range.start = str(ic_start.date())
            runner.config.date_range.end = str(ic_end.date())
            _run_screening(
                runner, selected_factors, config_path, args.t_threshold, output_dir
            )
    else:
        if args.factors:
            available_factors = set(
                list_registered("factor").get("factor", {}).keys()
            )
            try:
                factor_names = _validate_requested_factors(
                    _parse_requested_factors(args.factors), available_factors
                )
            except ValueError as exc:
                parser.error(str(exc))
        else:
            factor_names = runner.config.factors
        _run_single_research(runner, factor_names, config_path)


if __name__ == "__main__":
    main()
