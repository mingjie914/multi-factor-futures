"""回测报告生成器 — 从 IC 数据 + 回测输出生成全套图表到统一目录。

用法:
    python scripts/generate_report.py --ic runs/xxx/ic_by_window_period.json \
        --backtest-dir runs/backtest_xxx --output runs/report_xxx \
        --title "6因子日内策略" --oos

输出:
    01_净值曲线.png
    02_因子IC排序.png
    03_分层回测指标.png
    04_相关性聚类.png  (if --correlation passed)
    05_OOS因子IC.png   (if --oos)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe_read_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _short_name(name: str) -> str:
    return name.replace("intraday_", "").replace("_20d", "")


def chart_nav(backtest_dir: str, output_dir: str) -> None:
    """Copy backtest NAV chart if exists, otherwise skip."""
    src = os.path.join(backtest_dir, "backtest_nav.png")
    if os.path.exists(src):
        dst = os.path.join(output_dir, "01_净值曲线.png")
        import shutil
        shutil.copy2(src, dst)


def chart_ic_ranking(ic_path: str, output_dir: str, factor_filter: set | None = None) -> None:
    """IC / |t| / IR 横向对比."""
    data = _safe_read_json(ic_path)
    results = data.get("all_results", [])
    if factor_filter:
        results = [r for r in results if r.get("name", "") in factor_filter]
    results.sort(key=lambda r: abs(r.get("best_t", 0)), reverse=True)

    names = [_short_name(r["name"]) for r in results]
    ics = [r.get("best_ic", 0) for r in results]
    ts = [abs(r.get("best_t", 0)) for r in results]
    irs = [abs(r.get("best_ir", 0)) for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors_ic = ["#2ecc71" if ic > 0 else "#e74c3c" for ic in ics]
    axes[0].barh(names, ics, color=colors_ic)
    axes[0].axvline(0, color="gray", linewidth=0.5)
    axes[0].set_xlabel("IC (Pearson)")
    axes[0].set_title("Factor IC Ranking")

    axes[1].barh(names, ts, color="#3498db")
    axes[1].axvline(2.0, color="red", linestyle="--", label="|t|=2.0")
    axes[1].legend()
    axes[1].set_xlabel("|HAC t-statistic|")
    axes[1].set_title("Factor t-Statistics")

    axes[2].barh(names, irs, color="#9b59b6")
    axes[2].set_xlabel("|IR (NW)|")
    axes[2].set_title("Factor Information Ratio")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_因子IC排序.png"), dpi=150, bbox_inches="tight")
    plt.close()


def chart_layered(ic_path: str, output_dir: str, factor_filter: set | None = None) -> None:
    """分层回测指标."""
    data = _safe_read_json(ic_path)
    results = data.get("all_results", [])
    if factor_filter:
        results = [r for r in results if r.get("name", "") in factor_filter]

    names = [_short_name(r["name"]) for r in results]
    ls_ret = [r.get("layered_ls_return", 0) * 100 for r in results]
    mono = [abs(r.get("layered_monotonicity", 0)) for r in results]
    turnover = [r.get("annual_half_turnover", 0) for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = range(len(names))
    w = 0.25
    axes[0].bar([i - w for i in x], ls_ret, w, color="#2ecc71", label="Long-Short Ann. %")
    axes[0].bar(x, mono, w, color="#e67e22", label="|Monotonicity|")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha="right")
    axes[0].legend()
    axes[0].set_title("Layered Backtest Metrics")

    axes[1].bar(names, turnover, color="#3498db")
    axes[1].set_ylabel("Monthly Turnover")
    axes[1].set_title("Turnover")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_分层回测指标.png"), dpi=150, bbox_inches="tight")
    plt.close()


def chart_oos(ic_path: str, output_dir: str, factor_filter: set | None = None, title: str = "OOS") -> None:
    """OOS 期间因子 IC 对比."""
    data = _safe_read_json(ic_path)
    results = data.get("all_results", [])
    if factor_filter:
        results = [r for r in results if r.get("name", "") in factor_filter]

    names = [_short_name(r["name"]) for r in results]
    ics = [r.get("best_ic", 0) for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ecc71" if ic > 0 else "#e74c3c" for ic in ics]
    bars = ax.barh(names, ics, color=colors)
    for bar, ic in zip(bars, ics):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{ic:.3f}", va="center", fontsize=9)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("IC (Pearson)")
    ax.set_title(f"Factor IC — {title}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_OOS因子IC.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="回测报告生成器")
    parser.add_argument("--ic", required=True, help="ic_by_window_period.json 路径")
    parser.add_argument("--backtest-dir", help="回测输出目录 (含 backtest_nav.png)")
    parser.add_argument("--output", default="runs/report", help="报告输出目录")
    parser.add_argument("--factors", help="因子过滤器, 逗号分隔")
    parser.add_argument("--title", default="", help="报告标题前缀")
    parser.add_argument("--oos", action="store_true", help="生成 OOS 因子 IC 对比图")
    parser.add_argument("--correlation", help="factor_correlation.json 路径")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    factor_filter = (
        set(f.strip() for f in args.factors.split(",") if f.strip())
        if args.factors else None
    )

    if args.backtest_dir:
        chart_nav(args.backtest_dir, str(output_dir))
    chart_ic_ranking(args.ic, str(output_dir), factor_filter)
    chart_layered(args.ic, str(output_dir), factor_filter)
    if args.correlation and os.path.exists(args.correlation):
        import shutil
        shutil.copy2(args.correlation, output_dir / "04_因子相关性聚类.png")
    if args.oos:
        title = args.title or "OOS"
        chart_oos(args.ic, str(output_dir), factor_filter, title=title)

    print(f"报告已生成: {output_dir}")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
