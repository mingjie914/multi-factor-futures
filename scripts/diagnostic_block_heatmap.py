"""板块×因子热力图诊断 (只读脚本, 不修改框架).

计算 11 个有效因子在 8 大板块内的多空表现, 输出:
  1. block_performance.csv: 行=因子, 列=板块, 值=板块内多空年化夏普
  2. block_heatmap.png: 热力图 (含品种数标注)
  3. 自动识别"全板块通用因子"和"板块特异因子"

用法:
    python scripts/diagnostic_block_heatmap.py [--start 2025-01-01] [--end 2026-05-31]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_config
from core.sectors import SECTOR_MAP
from factors.engine import FactorEngine

# 11 个有效因子 + 方向 (+1=高暴露看多, -1=高暴露看空)
FACTORS = {
    "intraday_jump_intensity_20d": -1,
    "intraday_price_peak_count_20d": 1,
    "intraday_realised_skewness_20d": 1,
    "intraday_dtws_20d": 1,
    "intraday_drip_stone_20d": -1,
    "intraday_peak_ridge_ratio_20d": -1,
    "intraday_price_volume_elasticity_20d": -1,
    "intraday_roll_spread_20d": -1,
    "intraday_parkinson_vol_ratio_20d": -1,
    "intraday_kyle_lambda_20d": -1,
    "intraday_open_close_volume_ratio_20d": -1,
}


def load_factor_matrices(config_path: str, start: str, end: str):
    """计算因子日度暴露矩阵."""
    cfg = load_config(config_path)
    from pipeline.runner import PipelineRunner
    runner = PipelineRunner(config=cfg)
    engine = FactorEngine(runner.data_manager)
    calendar = pd.DatetimeIndex(runner.data_manager.get_calendar(
        pd.Timestamp(start), pd.Timestamp(end)))
    matrices = {}
    names = list(FACTORS)
    computed = engine.compute_factors(names, calendar.tolist(), cfg.universe,
                                      parallel=False)
    for name in names:
        matrices[name] = computed[name]
    close = runner.data_manager.get("close", calendar, cfg.universe)
    fwd5 = close.pct_change(5).shift(-5)  # 5日前瞻收益
    return matrices, fwd5, cfg.universe, calendar

def block_long_short_sharpe(
    factor_mat: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    tickers: list[str],
    direction: int,
) -> dict:
    """单因子在全体品种上的多空夏普 (用于对照)."""
    aligned_f = factor_mat.reindex(columns=tickers).copy()
    aligned_r = fwd_ret.reindex(index=aligned_f.index, columns=tickers).copy()
    ls_series = []
    for t in aligned_f.index:
        row_f = aligned_f.loc[t]
        row_r = aligned_r.loc[t]
        valid_mask = row_f.notna() & row_r.notna()
        f_vals = row_f[valid_mask]
        r_vals = row_r[valid_mask]
        if len(f_vals) < 4:
            continue
        # 分3组: 高/中/低
        ranked = f_vals.rank(pct=True)
        top = r_vals[ranked >= 0.667].mean()
        bot = r_vals[ranked <= 0.333].mean()
        ls = (top - bot) * direction
        if not np.isnan(ls):
            ls_series.append(ls)
    if len(ls_series) < 10:
        return {"sharpe": np.nan, "n_days": len(ls_series)}
    s = pd.Series(ls_series)
    sharpe = s.mean() / s.std(ddof=0) * np.sqrt(52) if s.std(ddof=0) > 0 else np.nan
    return {"sharpe": sharpe, "n_days": len(s)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/intraday_backtest.yaml")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-31")
    parser.add_argument("--output", default="runs/block_heatmap")
    args = parser.parse_args()

    matrices, fwd5, universe, calendar = load_factor_matrices(
        args.config, args.start, args.end)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # 品种→板块
    ticker_sector = {t: SECTOR_MAP.get(t, "other") for t in universe}
    sectors = sorted(set(ticker_sector.values()))

    rows = []
    heat = {}
    for name, direction in FACTORS.items():
        mat = matrices[name]
        sector_sharpes = {}
        for sec in sectors:
            sec_tickers = [t for t in universe if ticker_sector[t] == sec]
            if len(sec_tickers) < 3:
                sector_sharpes[sec] = np.nan
                continue
            res = block_long_short_sharpe(mat, fwd5, sec_tickers, direction)
            sector_sharpes[sec] = res["sharpe"]
        # 全品种对照
        all_res = block_long_short_sharpe(mat, fwd5, universe, direction)
        sector_sharpes["ALL"] = all_res["sharpe"]
        heat[name] = sector_sharpes
        rows.append({"factor": name, **sector_sharpes})

    df = pd.DataFrame(rows).set_index("factor")
    df.to_csv(out / "block_performance.csv", encoding="utf-8-sig")
    print(f"已保存: {out / 'block_performance.csv'}")

    # 热力图
    fig, ax = plt.subplots(figsize=(14, 8))
    plot_df = df[[c for c in df.columns]]
    im = ax.imshow(plot_df.values, cmap="RdYlGn", vmin=-3, vmax=3, aspect="auto")
    ax.set_xticks(range(len(plot_df.columns)))
    ax.set_xticklabels(plot_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index)
    for i in range(len(plot_df.index)):
        for j in range(len(plot_df.columns)):
            v = plot_df.iloc[i, j]
            txt = f"{v:.2f}" if not np.isnan(v) else "-"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    ax.set_title("因子×板块 多空夏普热力图")
    plt.colorbar(im, label="多空年化夏普")
    plt.tight_layout()
    plt.savefig(out / "block_heatmap.png", dpi=150)
    plt.close()
    print(f"已保存: {out / 'block_heatmap.png'}")

    # 识别结论
    print("\n=== 因子板块适用性 ===")
    for name in df.index:
        row = df.loc[name].dropna()
        valid_secs = [c for c in row.index if c != "ALL"]
        if not valid_secs:
            continue
        n_strong = sum(1 for c in valid_secs if abs(row[c]) > 0.8)
        if n_strong >= 4:
            tag = "全板块通用"
        elif sum(1 for c in valid_secs if abs(row[c]) > 1.5) == 1:
            tag = "板块特异"
        else:
            tag = "局部有效"
        print(f"  {name}: {tag} (有效板块 {n_strong}/7)")


if __name__ == "__main__":
    main()
