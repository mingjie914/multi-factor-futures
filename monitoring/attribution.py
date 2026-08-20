# -*- coding: utf-8 -*-
"""monitoring.attribution — B: 组合收益归因(因子/板块/品种三层, 三口径).

参考研报 008(CNES1 因子贡献量化: 单因子/前N大贡献占比)与 021/022(CTA 策略×板块盈亏归因).

数据源:
- 因子层: 信号组合收益(前 N 多头 - 后 N 空头, 方向化), 复用 factor_health.signal_portfolio_returns;
- 板块/品种层: 最近可用的 ResearchReturnLedger 产物
  research_*.csv(asset_returns / effective_weights / contributions).

纯增量模块; 遵守持仓展示规则(品种逐行、上多下空、按权重绝对值降序).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import monitoring.config as C
from monitoring import io
from monitoring.factor_health import directed_score, signal_portfolio_returns


def _load_ledger_csv(ledger_dir: str | Path) -> dict[str, pd.DataFrame]:
    """读取完整 research ledger 三件套，显式路径缺失时失败关闭."""
    d = Path(ledger_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"research ledger directory is missing: {d}")
    frames: dict[str, pd.DataFrame] = {}
    for key, fname in (("returns", "research_asset_returns.csv"),
                       ("weights", "research_effective_weights.csv"),
                       ("contributions", "research_return_contributions.csv")):
        p = d / fname
        if not p.is_file():
            raise FileNotFoundError(f"research ledger file is missing: {p}")
        frames[key] = io.read_csv(p)
    return frames


def _period_slice(contrib: pd.DataFrame, start: pd.Timestamp | None,
                  end: pd.Timestamp | None) -> pd.DataFrame:
    idx = pd.DatetimeIndex(contrib.index)
    mask = pd.Series(True, index=idx)
    if start is not None:
        mask &= idx >= start
    if end is not None:
        mask &= idx <= end
    return contrib.loc[mask]


class AttributionReport:
    """三层归因. ledger 可传 dict(returns/weights/contributions)或目录路径."""

    def __init__(self, signals: dict[str, pd.DataFrame] | None = None,
                 returns: pd.DataFrame | None = None,
                 ledger_dir: str | Path | None = None,
                 factors: dict[str, int] | None = None,
                 sector_map: dict[str, list[str]] | None = None):
        self.signals = signals or {}
        self.returns = returns if returns is not None else pd.DataFrame()
        self.factors = dict(factors) if factors is not None else dict(C.PRODUCTION_FACTORS)
        self.sector_map = sector_map or C.SECTORS
        frames = _load_ledger_csv(ledger_dir) if ledger_dir else {}
        self.ledger = frames
        # 品种→板块反查
        self._sector_of: dict[str, str] = {}
        for sec, members in self.sector_map.items():
            for m in members:
                self._sector_of[m] = sec

    # ---- 因子层(信号组合收益) ----
    def factor_contribution(self, periods: dict[str, tuple] | None = None,
                            multi_only: bool = False) -> pd.DataFrame:
        """每因子三口径贡献(周/月/YTD). Returns DataFrame(因子 × 口径)."""
        if periods is None:
            periods = _default_periods(self.returns)
        rows: dict[str, dict] = {}
        for name, direction in self.factors.items():
            sig = self.signals.get(name)
            if sig is None or sig.empty or self.returns.empty:
                continue
            score = directed_score(sig, direction)
            pr = signal_portfolio_returns(score, self.returns, top_n=C.SIGNAL_TOP_N,
                                          multi_only=multi_only)
            row: dict = {"direction": direction}
            for label, (start, end) in periods.items():
                sel = _period_slice(pr.to_frame("r"), start, end)["r"]
                row[label] = float(sel.sum()) if len(sel) else None
            rows[name] = row
        return pd.DataFrame(rows).T if rows else pd.DataFrame()

    # ---- 板块层 / 品种层(基于 ledger contributions) ----
    def sector_contribution(self, periods: dict[str, tuple] | None = None) -> pd.DataFrame:
        """板块贡献(多头/空头分行), 对齐 021/022 CTA 周报的板块盈亏归因."""
        if self.ledger.get("contributions") is None or self.ledger["contributions"].empty:
            return pd.DataFrame()
        if periods is None:
            periods = _default_periods(self.ledger["contributions"])
        contrib = self.ledger["contributions"]
        weights = self.ledger["weights"]
        out: dict[tuple, dict] = {}
        # 板块集合 = 映射板块 + "其他"(未映射品种), 保证归因对账完整
        mapped = set(m for members in self.sector_map.values() for m in members)
        others = [c for c in contrib.columns if c not in mapped]
        sectors = dict(self.sector_map)
        if others:
            sectors["其他"] = others
        for sec in sectors:
            members = [m for m in sectors[sec] if m in contrib.columns]
            if not members:
                continue
            for label, (start, end) in periods.items():
                cc = _period_slice(contrib, start, end)
                ww = _period_slice(weights, start, end)
                cm = cc[members]
                wm = ww[members]
                long_mask = wm > 1e-12
                short_mask = wm < -1e-12
                # 多头/空头品种各自的贡献和(空头贡献本身为负, 保持符号)
                long_c = cm.where(long_mask).sum(axis=1)
                short_c = cm.where(short_mask).sum(axis=1)
                out.setdefault((sec, "多"), {})[label] = (
                    float(long_c.sum()) if not long_c.dropna().empty else None)
                out.setdefault((sec, "空"), {})[label] = (
                    float(short_c.sum()) if not short_c.dropna().empty else None)
        df = pd.DataFrame(out).T
        df.index.names = ["板块", "方向"]
        return df

    def asset_contribution(self, period: tuple | None = None) -> pd.DataFrame:
        """品种明细贡献(遵守展示规则: 上多下空, 按权重绝对值降序)."""
        if self.ledger.get("contributions") is None or self.ledger["contributions"].empty:
            return pd.DataFrame()
        start, end = period or (None, None)
        cc = _period_slice(self.ledger["contributions"], start, end)
        ww = _period_slice(self.ledger["weights"], start, end)
        contrib = cc.sum(axis=0)
        avg_w = ww.mean(axis=0)
        out = pd.DataFrame({
            "direction": np.where(avg_w > 1e-12, "多", np.where(avg_w < -1e-12, "空", "平")),
            "avg_weight": avg_w,
            "contribution": contrib,
            "sector": [self._sector_of.get(c, "其他") for c in contrib.index],
        })
        out = out[out["direction"] != "平"].sort_values("avg_weight", key=abs, ascending=False)
        return out


def _default_periods(df: pd.DataFrame) -> dict[str, tuple]:
    """按数据最后日期推三口径: 上周(最后5交易日)/本月/年初至今."""
    idx = pd.DatetimeIndex(df.index).dropna()
    if len(idx) == 0:
        return {"week": (None, None), "month": (None, None), "ytd": (None, None)}
    end = idx[-1]
    week_start = idx[-5] if len(idx) >= 5 else idx[0]
    month_start = pd.Timestamp(end.year, end.month, 1)
    ytd_start = pd.Timestamp(end.year, 1, 1)
    return {"week": (week_start, end), "month": (month_start, end), "ytd": (ytd_start, end)}
