# -*- coding: utf-8 -*-
"""monitoring.weekly_report — 周报三件套(JSON + PNG + Markdown).

输出到 weeklyreport/周报_YYYYMMDD/（子目录按报告日期，内部文件不带日期）:
- 周报.md            三口径表格 + 状态迁移摘要 + 结论
- 因子IC热力图.png       因子 × 时间滚动 IC 热力图
- 因子回撤路径.png       多空累计收益 + 回撤区间
- 归因贡献.png           因子/板块贡献柱状图(多空分行)
- snapshot.json          当周快照(数据版)
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

import monitoring.config as C
from monitoring import io
from monitoring.attribution import AttributionReport
from monitoring.factor_health import FactorHealthMonitor

_PNG_FONT = {"font.sans-serif": ["SimHei", "Microsoft YaHei", "DejaVu Sans"],
             "axes.unicode_minus": False}


def _mk_png() -> "plt":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(_PNG_FONT)
    return plt


def generate(health: FactorHealthMonitor, attribution: AttributionReport | None,
             report_date: str | None = None) -> Path:
    """生成周报, 返回输出目录."""
    date_str = report_date or f"{datetime.now():%Y%m%d}"
    out_dir = C.WEEKLY_REPORT_DIR / f"周报_{date_str}"
    out_dir.mkdir(parents=True, exist_ok=True)

    snap = health.health_snapshot()
    io.write_json(snap, out_dir / "snapshot.json")

    _chart_ic_heatmap(snap, out_dir / "因子IC热力图.png")
    _chart_drawdown(snap, out_dir / "因子回撤路径.png")
    if attribution is not None:
        _chart_attribution(attribution, out_dir / "归因贡献.png")

    md = _render_markdown(snap, attribution, date_str)
    (out_dir / "周报.md").write_text(md, encoding="utf-8")
    return out_dir


def _chart_ic_heatmap(snap: dict, path) -> None:
    plt = _mk_png()
    factors = list(snap["factors"].keys())
    # 滚动 20 日 IC 矩阵(标签跟随 mat, 避免 daily_ic 全 NaN 因子导致列数不匹配)
    mat, labels = [], []
    for name in factors:
        recs = snap["factors"][name].get("daily_ic", [])
        if not recs:
            continue
        s = pd.Series({r["date"]: r["value"] for r in recs}, dtype=float)
        rolling = s.rolling(20, min_periods=5).mean()
        mat.append(rolling)
        labels.append(name)
    if not mat or len(mat) == 0:
        return
    df = pd.concat(mat, axis=1)
    df.columns = [f.split("intraday_")[1][:12] if f.startswith("intraday_") else f[:12]
                  for f in labels]
    fig, ax = plt.subplots(figsize=(12, max(3, 0.5 * len(df.columns) + 1)))
    im = ax.imshow(df.T.values, aspect="auto", cmap="RdYlGn", vmin=-0.06, vmax=0.06)
    ax.set_yticks(range(len(df.columns))); ax.set_yticklabels(df.columns)
    ax.set_xlabel("日期"); ax.set_title("因子滚动 20 日 IC 热力图")
    ax.grid(False)
    plt.colorbar(im, ax=ax, label="IC")
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def _chart_drawdown(snap: dict, path) -> None:
    plt = _mk_png()
    factors = list(snap["factors"].keys())
    n = len(factors)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.4 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, factors):
        recs = snap["factors"][name].get("drawdown_cycle", {}).get("cum_ret", [])
        if not recs:
            ax.set_title(name); continue
        s = pd.Series({r["date"]: r["value"] for r in recs}, dtype=float)
        ax.plot(s.index, s.values, lw=1.2)
        peak = s.rolling(20, min_periods=1).max()
        ax.fill_between(s.index, s.values, peak.values, where=(s < peak), color="red", alpha=0.3)
        ax.set_title(f"{name}  回撤深度={snap['factors'][name]['drawdown_cycle']['depth']:.1%}  "
                     f"状态={snap['factors'][name]['state']}")
        ax.grid(alpha=0.3)
    fig.suptitle("因子多空累计收益与回撤区间")
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def _chart_attribution(attr: AttributionReport, path) -> None:
    plt = _mk_png()
    fc = attr.factor_contribution()
    if fc.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    # 因子层(最近一周)
    week = fc[["week"]].dropna().sort_values("week")
    if len(week):
        axes[0].barh(week.index, week["week"].values, color=["#c0392b" if v < 0 else "#27ae60" for v in week["week"]])
    axes[0].set_title("因子层贡献(上周)"); axes[0].axvline(0, color="k", lw=0.6); axes[0].grid(alpha=0.3)
    # 板块层
    sc = attr.sector_contribution()
    if not sc.empty:
        w = sc[["week"]].dropna().sort_values("week")
        if len(w):
            labels = [f"{i[0]}({i[1]})" for i in w.index]
            axes[1].barh(labels, w["week"].values,
                         color=["#c0392b" if v < 0 else "#27ae60" for v in w["week"]])
    axes[1].set_title("板块层贡献(上周, 多/空分行)"); axes[1].axvline(0, color="k", lw=0.6)
    axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _render_markdown(snap: dict, attribution: AttributionReport | None,
                     report_date: str | None) -> str:
    as_of = snap.get("as_of", "-")
    date_str = report_date or f"{datetime.now():%Y-%m-%d}"
    lines = [
        f"# 因子监控与归因周报({date_str})",
        "",
        f"> 数据截至 {as_of} | 状态机阈值: IC={C.IC_RETIRED}(连续{C.IC_DAYS_CONSECUTIVE}日) / 回撤={C.DD_THRESHOLD:.0%}(历史峰值) / 反弹=创{C.REBOUND_WINDOW}日新高",
        "",
        "## 一、因子健康状态",
        "",
        "| 因子 | 方向 | 状态 | 全样本IC | 60日IC | 20日IC | 上周IC | 上周多空 | 回撤深度 | 20日动量 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, info in snap["factors"].items():
        m = info["metrics"]
        short = name.replace("intraday_", "").replace("_20d", "")
        lines.append(
            f"| {short} | {info['direction']:+d} | {info['state']} | "
            f"{_fmt(m.get('all', {}).get('ic'))} | {_fmt(m.get('60d', {}).get('ic'))} | "
            f"{_fmt(m.get('20d', {}).get('ic'))} | {_fmt(m.get('week', {}).get('ic'))} | "
            f"{_fmt(m.get('week', {}).get('port_ret'))} | "
            f"{_fmt(info['drawdown_cycle']['depth'])} | {_fmt(m.get('momentum_20d'))} |"
        )
    lines += ["", "## 二、组合收益归因", ""]
    if attribution is not None:
        fc = attribution.factor_contribution()
        if not fc.empty:
            lines += ["### 因子层贡献(周/月/YTD)", "",
                      "| 因子 | 上周 | 本月 | YTD |", "|---|---|---|---|"]
            for name, row in fc.iterrows():
                short = str(name).replace("intraday_", "").replace("_20d", "")
                lines.append(f"| {short} | {_fmt(row.get('week'))} | {_fmt(row.get('month'))} | {_fmt(row.get('ytd'))} |")
        sc = attribution.sector_contribution()
        if not sc.empty:
            lines += ["", "### 板块层贡献(上周, 多/空分行)", "",
                      "| 板块 | 方向 | 上周 | 本月 | YTD |", "|---|---|---|---|---|"]
            for (sec, side), row in sc.iterrows():
                lines.append(f"| {sec} | {side} | {_fmt(row.get('week'))} | {_fmt(row.get('month'))} | {_fmt(row.get('ytd'))} |")
        ac = attribution.asset_contribution()
        if not ac.empty:
            lines += ["", "### 品种层贡献(近一周, 上多下空)", "",
                      "| 品种 | 方向 | 板块 | 平均权重 | 贡献 |", "|---|---|---|---|---|"]
            for name, row in ac.head(24).iterrows():
                lines.append(f"| {name} | {row['direction']} | {row['sector']} | {row['avg_weight']:.3f} | {row['contribution']:.4f} |")
    lines += ["", "## 三、结论", "",
              "（由人工结合图表与表格研判: 状态迁移 / 回撤因子 / 需剔除或再激活项）", ""]
    return "\n".join(lines)
