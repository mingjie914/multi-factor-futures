# -*- coding: utf-8 -*-
"""对比测试: v1(旧持仓top2, 含合成) vs v2(修复: 排除合成+按到期月) 期限结构面板.
1) 面板级: near/far 选择差异、合成合约占比、覆盖度
2) 因子级: 抽样 term 因子在 v1/v2 面板下的日度序列对比 (相关性/覆盖度/数值分布)
用法: python scripts/compare_term_v1_v2.py
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

ROOT = r"E:\程明杰公司内容\multi_factor"
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

# 直接复用 intraday.py 的读取函数
sys.path.insert(0, os.path.join(ROOT, "factors", "library"))
import intraday as ID


def v1_panel(dates, universe, freq="1min"):
    """旧版逻辑(快照): 含合成合约 + 持仓 top2."""
    all_data = ID._read_local_raw(dates, universe, freq=freq)
    if all_data is None or "position" not in all_data.columns:
        return {}
    per_symbol = (all_data.sort_values("_ts")
                  .groupby(["_ts", "root", "symbol"], as_index=False)
                  .agg({"close": "last", "position": "last", "volume": "sum"}))
    per_symbol["_rank"] = per_symbol.groupby(["_ts", "root"])["position"].rank(ascending=False, method="first")
    near = per_symbol[per_symbol["_rank"] == 1].set_index(["_ts", "root"])
    far = per_symbol[per_symbol["_rank"] == 2].set_index(["_ts", "root"])
    panel = {}
    for field, src in [("close", near), ("position", near), ("volume", near)]:
        s = src[field].unstack(level="root")
        s.index = pd.DatetimeIndex(s.index)
        panel[f"near_{field}"] = s
    for field, src in [("close", far), ("position", far), ("volume", far)]:
        s = src[field].unstack(level="root")
        s.index = pd.DatetimeIndex(s.index)
        panel[f"far_{field}"] = s
    return panel


def daily_slope(panel):
    """按 #224 口径算日度斜率 (near-far)/far."""
    near, far = panel["near_close"], panel["far_close"]
    days = sorted(set(near.index.normalize()) | set(far.index.normalize()))
    slopes = {}
    for dt in days:
        grp_n = near.loc[near.index.normalize() == dt]
        grp_f = far.loc[far.index.normalize() == dt]
        vals = {}
        for col in grp_n.columns:
            n = grp_n[col].dropna()
            f = grp_f[col].dropna()
            common = n.index.intersection(f.index)
            if len(common) < 20:
                continue
            sl = ((n.loc[common] - f.loc[common]) / f.loc[common].replace(0, np.nan)).dropna()
            if len(sl) >= 10:
                vals[col] = float(sl.mean())
        if vals:
            slopes[dt] = pd.Series(vals)
    if not slopes:
        return pd.DataFrame()
    df = pd.DataFrame(slopes).T
    df.index = pd.DatetimeIndex(df.index)
    return df


if __name__ == "__main__":
    dates = pd.date_range("2026-04-01", "2026-07-31", freq="5min")
    universe = ["RB", "M", "IF", "AU", "JM", "CU", "IC"]
    print("读取 5min 数据并构建 v1/v2 面板...")
    p1 = v1_panel(dates, universe, freq="5min")
    p2 = ID._get_term_structure_panel(None, dates, universe, freq="5min")

    # ---- 1) 合成合约占比: v1 near/far 命中合成合约的比例 ----
    all_data = ID._read_local_raw(dates, universe, freq="5min")
    conc = all_data[all_data["symbol"].astype(str).str.endswith(("8888", "9998", "9999"))]
    total_sym = all_data["symbol"].nunique()
    print(f"5min 全区间 symbol 总数: {total_sym}, 其中合成合约: {conc['symbol'].nunique()}")
    print(f"  -> v1 未排除合成: near/far 可能命中 {conc['symbol'].nunique()} 个合成合约")

    # ---- 2) 覆盖度: v1 vs v2 far 非 NaN 比例 ----
    def cov(panel, key):
        s = panel[key]
        return float(s.notna().mean().mean())
    print("\n覆盖度 (非NaN比例, 7品种平均):")
    for key in ["near_close", "far_close"]:
        c1, c2 = cov(p1, key), cov(p2, key)
        print(f"  {key}: v1={c1:.4f}  v2={c2:.4f}")

    # ---- 3) 日度斜率对比 (#224 口径) ----
    s1 = daily_slope(p1)
    s2 = daily_slope(p2)
    common_cols = [c for c in s1.columns if c in s2.columns]
    print("\n日度斜率 (#224 口径) 对比:")
    for col in common_cols:
        a, b = s1[col], s2[col]
        common = a.index.intersection(b.index)
        if len(common) < 5:
            print(f"  {col}: 共同样本不足 ({len(common)})")
            continue
        corr = a.loc[common].corr(b.loc[common])
        print(f"  {col}: corr={corr:+.3f}  v1覆盖={a.notna().mean():.2f} v2覆盖={b.notna().mean():.2f} "
              f"v1均值={a.mean():+.4f} v2均值={b.mean():+.4f}")

    # ---- 4) v2 新增字段抽样 ----
    print("\nv2 新增字段样例 (RB, 2026-06):")
    rb_exp = p2.get("near_expiry", pd.DataFrame())
    rb_rem = p2.get("remaining_days", pd.DataFrame())
    rb_flag = p2.get("rollover_flag", pd.DataFrame())
    if "RB" in rb_exp.columns:
        sub = rb_exp["RB"].dropna()
        if len(sub):
            idx = sub.index[len(sub) // 2]
            print(f"  {idx.date()}: near_expiry={int(rb_exp['RB'].loc[idx])} "
                  f"remaining_days={rb_rem['RB'].loc[idx]:.0f} "
                  f"rollover_flag={rb_flag['RB'].loc[idx]:.0f}")
    # 换月标记次数
    if "RB" in rb_flag.columns:
        n_flag = int(rb_flag["RB"].sum())
        print(f"  RB 换月标记(±1日)总次数: {n_flag}")
