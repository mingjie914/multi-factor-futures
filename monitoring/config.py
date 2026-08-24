# -*- coding: utf-8 -*-
"""monitoring.config — 因子监控与归因仪表盘参数集中.

设计文档: docs/因子监控与归因仪表盘_设计文档.md
原则: 监控阈值与检验流程一致(不比检验更严); 观察模块不修改生产配置.
"""
from __future__ import annotations

from pathlib import Path

from strategies.combined import FACTORS, SECTOR_MAP, UNIVERSE38 as STRATEGY_UNIVERSE

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---- 目录 ----
# 数据目录用 monitoring_data 避免与 monitoring/ 包目录同名混淆
MONITOR_DIR = PROJECT_ROOT / "monitoring_data"
SIGNALS_DIR = MONITOR_DIR / "signals"
WEEKLY_REPORT_DIR = PROJECT_ROOT / "weeklyreport"

# ---- 监控参数(与检验流程一致, 不比检验更严) ----
IC_RETIRED = 0.02            # 失效线 = testing/ic_test.py 准入门槛
IC_DAYS_CONSECUTIVE = 20     # 连续 N 日滚动 IC < 阈值 → RETIRED(沿用 alpha/ic_monitor.py 逻辑)
DD_THRESHOLD = 0.30          # 多空累计收益回撤深度阈值(相对历史最高净值)
REBOUND_WINDOW = 20          # 反弹确认 = 创 N 日滚动最高
REBOUND_CONFIRM_DAYS = 2     # 创新高后维持天数(去抖)
IC_WINDOWS = (60, 20)        # 滚动 IC 窗口(日), 60 日为主判定, 20 日为快照
IC_REACTIVATE_DAYS = 5       # RETIRED 再激活: 最近 N 日滚动 IC 已回到阈值之上
REPORT_PERIODS = ("ytd", "month", "week")  # 周报三口径

# ---- 状态机 ----
STATES = ("ACTIVE", "WATCH", "RETIRED")
STATE_ACTIVE = "ACTIVE"
STATE_WATCH = "WATCH"
STATE_RETIRED = "RETIRED"

# ---- 固定观察因子、板块与品种池 ----
# 从当前策略常量构造只读本地容器，避免再次维护一套会过期的生产定义。
OBSERVATION_FACTORS: dict[str, int] = dict(FACTORS)
SECTORS: dict[str, list[str]] = {
    sector: list(members) for sector, members in SECTOR_MAP.items()
}
UNIVERSE38: list[str] = list(STRATEGY_UNIVERSE)
SIGNAL_TOP_N = 10            # 多空组合: 因子值前/后 N 名品种(信号组合收益与归因用)

# ---- 回测配置(信号构建时引用) ----
OBSERVATION_CONFIG = "config/default.yaml"
DATA_START = "2025-01-01"
# 信号构建窗口上限: 运行时自动截断到数据源可用最后交易日(见 run_monitoring.cmd_build_signals)
