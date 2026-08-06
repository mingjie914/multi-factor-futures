# -*- coding: utf-8 -*-
"""monitoring 单测: 状态机迁移 / 回撤-反弹 / 归因对账 / 周报三件套."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import monitoring.config as C
import monitoring.factor_health as FH
import monitoring.attribution as AT
import monitoring.weekly_report as WR
from monitoring import io


# ---------- 工具 ----------
def make_ic_series(n=120, seed=7, signal_strength=0.0, drop_after=0):
    """构造 IC 序列: 前段(可选强相关) + 后段噪声."""
    rng = np.random.default_rng(seed)
    ic = rng.normal(0.0, 0.02, n)
    if drop_after > 0:
        ic[:drop_after] = rng.normal(0.08, 0.02, drop_after)  # 前段高 IC
        ic[drop_after:] = rng.normal(0.0, 0.01, n - drop_after)  # 后段 ≈ 0
    return pd.Series(ic, index=pd.bdate_range("2026-01-01", periods=n))


def make_port_ret(n=120, seed=3, drift=0.0):
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, 0.01, n)
    return pd.Series(r, index=pd.bdate_range("2026-01-01", periods=n))


# ---------- A: 状态机 ----------
class TestStateMachine:
    def test_retired_on_persistent_low_ic(self):
        """滚动 60 日 IC 连续 20 日 < 0.02 → RETIRED."""
        ic = make_ic_series(drop_after=40)  # 后 80 天 IC≈0
        port = make_port_ret()
        state = FH.FactorHealthMonitor()._decide_state(ic, port, C.STATE_ACTIVE)
        assert state == C.STATE_RETIRED

    def test_watch_on_drawdown(self):
        """20 日窗口内多空累计收益回撤 > 30% 且 IC 正常 → WATCH."""
        ic = make_ic_series(drop_after=0)
        ic.iloc[:] = 0.05  # IC 稳定高于阈值
        # 前 100 日平稳微涨, 后 20 日每天 -2% → 20 日窗口回撤约 -33%
        n = 120
        idx = pd.bdate_range("2026-01-01", periods=n)
        port = pd.Series(0.0005, index=idx)
        port.iloc[100:] = -0.02
        state = FH.FactorHealthMonitor()._decide_state(ic, port, C.STATE_ACTIVE)
        assert state == C.STATE_WATCH

    def test_rebound_reactivates_to_active(self):
        """回撤后创 20 日新高且维持 2 日 → ACTIVE."""
        ic = make_ic_series(drop_after=0)
        ic.iloc[:] = 0.05
        # 累计收益先跌 40% 再涨回并超过初始峰值(+30%)
        n = 120
        idx = pd.bdate_range("2026-01-01", periods=n)
        cum = pd.Series(np.concatenate([np.linspace(0.0, -0.40, 60),
                                        np.linspace(-0.40, 0.30, n - 60)]), index=idx)
        port = cum.diff().fillna(0.0)  # 日收益(由累计序列反推)
        state = FH.FactorHealthMonitor()._decide_state(ic, port, C.STATE_WATCH)
        assert state == C.STATE_ACTIVE

    def test_active_when_healthy(self):
        """IC 高且无回撤 → 维持 ACTIVE."""
        ic = make_ic_series(drop_after=0)
        ic.iloc[:] = 0.06
        # 低波动正漂移, 保证 20 日窗口回撤 < 30%
        port = make_port_ret(drift=0.002)
        port = port * 0.2  # 缩小波动, 避免随机大回撤
        state = FH.FactorHealthMonitor()._decide_state(ic, port, C.STATE_ACTIVE)
        assert state == C.STATE_ACTIVE


# ---------- A: 指标与回撤周期 ----------
class TestMetrics:
    def test_drawdown_cycle_depth(self):
        m = FH.FactorHealthMonitor()
        n = 100
        idx = pd.bdate_range("2026-01-01", periods=n)
        port = pd.Series(np.linspace(0.0, -0.50, n), index=idx)  # 单调跌 50%
        cyc = m._drawdown_cycle(port)
        assert cyc["depth"] < -0.30
        assert cyc["in_drawdown"] is True

    def test_snapshot_structure(self):
        m = FH.FactorHealthMonitor({"f1": 1, "f2": -1})
        n = 80
        idx = pd.bdate_range("2026-01-01", periods=n)
        rng = np.random.default_rng(1)
        ret = pd.DataFrame(rng.normal(0, 0.01, (n, 5)),
                           index=idx, columns=list("ABCDE"))
        sig = pd.DataFrame(rng.normal(0, 1, (n, 5)),
                           index=idx, columns=list("ABCDE"))
        m.update({"f1": sig, "f2": -sig}, ret)
        snap = m.health_snapshot()
        assert "f1" in snap["factors"] and "f2" in snap["factors"]
        assert snap["factors"]["f1"]["state"] in C.STATES
        assert "metrics" in snap["factors"]["f1"]
        assert "drawdown_cycle" in snap["factors"]["f1"]


# ---------- B: 归因对账 ----------
class TestAttribution:
    def _ledger(self):
        n = 20
        idx = pd.bdate_range("2026-06-01", periods=n)
        cols = ["RB", "HC", "I", "CU", "AL", "M", "A", "IC"]
        rng = np.random.default_rng(5)
        returns = pd.DataFrame(rng.normal(0, 0.01, (n, len(cols))), index=idx, columns=cols)
        weights = pd.DataFrame(rng.normal(0, 0.05, (n, len(cols))), index=idx, columns=cols)
        contrib = returns * weights
        return {"returns": returns, "weights": weights, "contributions": contrib}

    def test_attribution_identity(self):
        """三层归因与 contributions 对账: 板块求和 == 总贡献; 品种明细求和一致."""
        frames = self._ledger()
        attr = AT.AttributionReport(ledger_dir=None)
        attr.ledger = frames
        sc = attr.sector_contribution()
        total = frames["contributions"].sum().sum()
        sec_total = sc[["ytd"]].sum().sum() if "ytd" in sc.columns else 0.0
        assert abs(sec_total - total) < 1e-6, f"板块归因 {sec_total} != 总贡献 {total}"
        ac = attr.asset_contribution()
        assert abs(ac["contribution"].sum() - total) < 1e-6

    def test_asset_direction_split(self):
        frames = self._ledger()
        attr = AT.AttributionReport()
        attr.ledger = frames
        ac = attr.asset_contribution()
        assert set(ac["direction"].unique()) <= {"多", "空"}
        longs = ac[ac["direction"] == "多"]
        shorts = ac[ac["direction"] == "空"]
        # 上多下空: 多头的平均权重应为正, 空头为负
        assert (longs["avg_weight"] > 0).all() if len(longs) else True
        assert (shorts["avg_weight"] < 0).all() if len(shorts) else True


# ---------- 周报 ----------
class TestWeeklyReport:
    def test_generate_files(self, tmp_path, monkeypatch):
        """mock 数据生成三件套, 断言文件存在."""
        monkeypatch.setattr(C, "WEEKLY_REPORT_DIR", tmp_path / "weeklyreport")
        m = FH.FactorHealthMonitor({"f1": 1})
        n = 60
        idx = pd.bdate_range("2026-05-01", periods=n)
        rng = np.random.default_rng(2)
        ret = pd.DataFrame(rng.normal(0, 0.01, (n, 4)), index=idx, columns=list("ABCD"))
        sig = pd.DataFrame(rng.normal(0, 1, (n, 4)), index=idx, columns=list("ABCD"))
        m.update({"f1": sig}, ret)
        attr = AT.AttributionReport(signals={"f1": sig}, returns=ret)
        out = WR.generate(m, attr, report_date="20260807")
        assert out.exists()
        assert (out / "周报.md").exists()
        assert (out / "snapshot.json").exists()
        assert (out / "因子IC热力图.png").exists() or (out / "因子回撤路径.png").exists()
