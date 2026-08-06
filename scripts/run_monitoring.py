# -*- coding: utf-8 -*-
"""run_monitoring — 因子监控与归因仪表盘入口(手动触发).

用法:
  python scripts/run_monitoring.py build-signals   # 重算生产因子信号 + 日度收益, 落盘 monitoring_data/signals/
  python scripts/run_monitoring.py update          # 日度增量: 更新因子健康状态机, 落盘 monitoring_data/factor_health.json
  python scripts/run_monitoring.py report [YYYYMMDD]  # 生成周报三件套到 weeklyreport/周报_YYYYMMDD/

设计文档: docs/因子监控与归因仪表盘_设计文档.md
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

import monitoring.config as C
from monitoring import io
from monitoring.attribution import AttributionReport
from monitoring.factor_health import FactorHealthMonitor
from monitoring.weekly_report import generate as generate_weekly_report


def cmd_build_signals() -> None:
    """重算生产因子信号与日度收益(复用 diag_production_nav 的信号构建模式)."""
    from core.config import load_config
    from pipeline.runner import PipelineRunner
    from factors.engine import FactorEngine

    cfg = load_config(C.BACKTEST_CONFIG)
    runner = PipelineRunner(config=cfg)
    univ = list(C.UNIVERSE38)
    start = pd.Timestamp(C.DATA_START)
    # 用远期 end 求数据源日历, 再截断到实际最后交易日, 避免面板固定到硬编码日期
    cal_all = pd.DatetimeIndex(runner.data_manager.get_calendar(start, pd.Timestamp("2100-01-01")))
    # 防御: 数据源缓存占位/异常回退会返回至 2100 年的 ~26000 个工作日(manager.py _cache_only 路径)
    if len(cal_all) == 0:
        print("[build-signals] 数据源无可用日历")
        sys.exit(1)
    if len(cal_all) > 5000 or cal_all[-1].year >= 2100:
        print(f"[build-signals] 日历异常({len(cal_all)} 日, 末交易日 {cal_all[-1].date()}), "
              "疑似命中缓存占位; 请检查数据源后重试")
        sys.exit(1)
    cal = cal_all[(cal_all >= start) & (cal_all <= cal_all[-1])]
    engine = FactorEngine(runner.data_manager)
    comp = engine.compute_factors(list(C.PRODUCTION_FACTORS), cal, univ, parallel=True)
    signals = {n: comp[n] for n in C.PRODUCTION_FACTORS if n in comp}
    close = runner.data_manager.get("close", cal, univ)
    returns = close.pct_change()
    io.save_signals(signals, returns, close)
    print(f"[build-signals] 完成: {len(signals)} 因子, {len(cal)} 交易日, {len(univ)} 品种")
    print(f"  信号: {io.SIGNALS_DIR}")


def cmd_update() -> None:
    """日度增量: 用落盘信号更新状态机."""
    signals = io.load_signals()
    returns = io.load_returns()
    if not signals or returns.empty:
        print("[update] 无信号数据, 请先运行 build-signals")
        sys.exit(1)
    monitor = FactorHealthMonitor()
    monitor.update(signals, returns)
    snap = monitor.health_snapshot()
    io.save_health(snap)
    print(f"[update] 完成, as_of={snap['as_of']}")
    for name, info in snap["factors"].items():
        print(f"  {name}: {info['state']}  回撤深度={info['drawdown_cycle']['depth']:.1%}")


def cmd_report(report_date: str | None = None) -> None:
    """生成周报三件套.

    周报日期优先级: 显式参数 > 数据截至日(snapshot.as_of) > 系统日期.
    """
    snap = io.load_health()
    if not snap or not snap.get("factors"):
        print("[report] 无健康数据, 请先运行 update")
        sys.exit(1)
    if report_date is None and snap.get("as_of"):
        # as_of 形如 2026-07-31 → 周报日期 20260731(与数据实际最后交易日一致)
        report_date = str(snap["as_of"]).replace("-", "")
    signals = io.load_signals()
    returns = io.load_returns()
    monitor = FactorHealthMonitor()
    monitor.update(signals, returns)
    # 归因: ledger 目录按最新折/子组合找
    ledger_dir = _find_latest_ledger_dir()
    attr = AttributionReport(signals=signals, returns=returns, ledger_dir=ledger_dir)
    out = generate_weekly_report(monitor, attr, report_date)
    print(f"[report] 周报已生成: {out}")


def _find_latest_ledger_dir() -> Path | None:
    """从 runs/ 下找最近一次的 research_*.csv 目录(按修改时间)."""
    runs = Path(__file__).resolve().parents[1] / "runs"
    if not runs.exists():
        return None
    cands = sorted(runs.rglob("research_asset_returns.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0].parent if cands else None


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "build-signals":
        cmd_build_signals()
    elif cmd == "update":
        cmd_update()
    elif cmd == "report":
        cmd_report(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        print(f"未知命令: {cmd}\n{__doc__}")
        sys.exit(1)


if __name__ == "__main__":
    main()
