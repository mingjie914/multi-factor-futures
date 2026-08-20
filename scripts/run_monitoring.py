# -*- coding: utf-8 -*-
"""run_monitoring — 因子监控与归因仪表盘入口(手动触发).

用法:
  python -m scripts.run_monitoring build-signals   # 重算固定观察因子信号 + 日度收益, 落盘 monitoring_data/signals/
  python -m scripts.run_monitoring update          # 日度增量: 更新因子健康状态机, 落盘 monitoring_data/factor_health.json
  python -m scripts.run_monitoring report [YYYYMMDD] [LEDGER_DIR]
      # 生成周报；显式提供 ledger 目录时增加板块/品种归因

设计文档: docs/因子监控与归因仪表盘_设计文档.md
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import monitoring.config as C
from monitoring import io
from monitoring.attribution import AttributionReport
from monitoring.factor_health import FactorHealthMonitor
from monitoring.weekly_report import generate as generate_weekly_report


def cmd_build_signals() -> None:
    """按当前固定观察因子定义重算信号与日度收益。"""
    from core.config import load_config
    from data.manager import DataManager
    from factors.engine import FactorEngine
    from factors import library as _factor_library  # noqa: F401

    cfg = load_config(C.BACKTEST_CONFIG)
    manager = DataManager.from_config(cfg)
    univ = list(C.UNIVERSE38)
    start = pd.Timestamp(C.DATA_START)
    latest = getattr(manager.source, "fetch_latest_trade_date", None)
    if not callable(latest):
        raise NotImplementedError("configured source does not expose its latest trade date")
    end = pd.Timestamp(latest()).normalize()
    cal_all = pd.DatetimeIndex(manager.get_calendar(start, end))
    if len(cal_all) == 0:
        raise RuntimeError("monitoring source returned an empty trading calendar")
    cal = cal_all[(cal_all >= start) & (cal_all <= end)]
    engine = FactorEngine(manager)
    comp = engine.compute_factors(list(C.PRODUCTION_FACTORS), cal, univ, parallel=True)
    signals = {n: comp[n] for n in C.PRODUCTION_FACTORS if n in comp}
    close = manager.get("close", cal, univ)
    returns, _ = manager.prepare_close_data(close)
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


def cmd_report(
    report_date: str | None = None,
    ledger_dir: str | None = None,
) -> None:
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
    attr = AttributionReport(signals=signals, returns=returns, ledger_dir=ledger_dir)
    out = generate_weekly_report(monitor, attr, report_date)
    print(f"[report] 周报已生成: {out}")


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] in {"-h", "--help"}:
        print("用法: python -m scripts.run_monitoring {build-signals|update|report} [参数]")
        return
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "build-signals":
        cmd_build_signals()
    elif cmd == "update":
        cmd_update()
    elif cmd == "report":
        cmd_report(
            sys.argv[2] if len(sys.argv) > 2 else None,
            sys.argv[3] if len(sys.argv) > 3 else None,
        )
    else:
        print(f"未知命令: {cmd}\n{__doc__}")
        sys.exit(1)


if __name__ == "__main__":
    main()
