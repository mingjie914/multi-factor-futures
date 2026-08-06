"""B 阶段(换月事件工程)单测: _read_local_daily 修复与换月日历/新因子.

覆盖审阅要求的 5 项:
1. trade_date 对齐: 1d 的 _ts 为"前一自然日 16:00", 面板行标签应落在 trade_date;
2. 合成合约排除: 8888/9999 持仓最大也不应出现在输出;
3. 换月日 NaN: 主力切换日 settle 置 NaN, 前后日正常;
4. _get_rollover_calendar 与 _read_local_daily 的换月日一致;
5. #436-438 compute 无未来数据泄漏(shift(1) 后首行为 NaN).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import factors.library.intraday as ID


def make_daily_raw(rows):
    """rows: list of (trade_date, symbol, position, settle, close) → raw DataFrame."""
    df = pd.DataFrame(rows, columns=["trade_date", "symbol", "position", "settle_price", "close"])
    df["root"] = df["symbol"].str.extract(r"^([A-Za-z]+)")[0]
    # 模拟真实 1d: _ts = 前一自然日 16:00
    df["_ts"] = pd.to_datetime(df["trade_date"]) - pd.Timedelta(days=1)
    df["_ts"] = df["_ts"] + pd.Timedelta(hours=16)
    return df


class EmptyData:
    def get(self, *args, **kwargs):
        return None


@pytest.fixture(autouse=True)
def _patch_raw(monkeypatch):
    rows = [
        # 交易日, 合约, 持仓, settle, close
        ("2026-05-11", "RB2605", 100000, 3200.0, 3205.0),
        ("2026-05-11", "RB2606", 60000, 3180.0, 3185.0),
        ("2026-05-11", "RB8888", 999999, 9999.0, 9999.0),   # 合成合约, 持仓最大
        ("2026-05-12", "RB2605", 80000, 3210.0, 3212.0),
        ("2026-05-12", "RB2606", 90000, 3190.0, 3195.0),   # 换月: 2606 反超
        ("2026-05-12", "RB8888", 999999, 9999.0, 9999.0),
        ("2026-05-13", "RB2605", 40000, 3200.0, 3198.0),
        ("2026-05-13", "RB2606", 110000, 3200.0, 3202.0),
        ("2026-05-13", "RB8888", 999999, 9999.0, 9999.0),
    ]
    raw = make_daily_raw(rows)
    monkeypatch.setattr(ID, "_read_local_raw", lambda dates, universe, freq="daily": raw)
    yield


DATES = pd.date_range("2026-05-11", "2026-05-13", freq="D")
UNIVERSE = ["RB"]


def test_trade_date_alignment_and_synthetic_exclusion():
    """面板行标签落在 trade_date 而非前一自然日; 合成合约值不出现."""
    settle = ID._read_local_daily(EmptyData(), DATES, UNIVERSE, "settle")
    assert settle.loc["2026-05-11", "RB"] == pytest.approx(3200.0)   # RB2605 主力(排除8888)
    assert np.isnan(settle.loc["2026-05-12", "RB"])                  # 换月日
    assert settle.loc["2026-05-13", "RB"] == pytest.approx(3200.0)   # RB2606 现为主力
    # 无 9999.0(合成)混入
    assert (settle["RB"].dropna() != 9999.0).all()


def test_rollover_day_is_nan():
    """05-12 主力 2605→2606 切换日 settle 为 NaN, 前后日正常."""
    settle = ID._read_local_daily(EmptyData(), DATES, UNIVERSE, "settle")
    assert np.isnan(settle.loc["2026-05-12", "RB"])
    assert settle.loc["2026-05-11", "RB"] == pytest.approx(3200.0)
    assert settle.loc["2026-05-13", "RB"] == pytest.approx(3200.0)   # 2606 现为主力


def test_rollover_calendar_matches_daily_nan():
    """换月日历与 _read_local_daily 的换月日集合一致."""
    cal = ID._get_rollover_calendar(DATES, UNIVERSE)
    assert list(cal["RB"]) == [pd.Timestamp("2026-05-12")]
    settle = ID._read_local_daily(EmptyData(), DATES, UNIVERSE, "settle")
    for d in cal["RB"]:
        assert np.isnan(settle.loc[d, "RB"])


def test_oi_field_no_duplicate_column_bug():
    """oi 字段(col=position)不应因重复列名导致面板全 NaN."""
    oi = ID._read_local_daily(EmptyData(), DATES, UNIVERSE, "oi")
    assert not oi["RB"].isna().all()
    # 05-11 主力 2605(pos=100000), 05-13 主力 2606(pos=110000); 05-12 换月 NaN
    assert oi.loc["2026-05-11", "RB"] == pytest.approx(100000.0)
    assert np.isnan(oi.loc["2026-05-12", "RB"])
    assert oi.loc["2026-05-13", "RB"] == pytest.approx(110000.0)


def test_rollover_nan_scoped_to_own_root(monkeypatch):
    """换月日置 NaN 必须按 (日期, 品种) 单元格, 不得误伤同日未换月的其他品种."""
    rows = [
        ("2026-05-11", "RB2605", 100000, 3200.0, 3205.0),
        ("2026-05-11", "RB2606", 60000, 3180.0, 3185.0),
        ("2026-05-12", "RB2605", 80000, 3210.0, 3212.0),
        ("2026-05-12", "RB2606", 90000, 3190.0, 3195.0),   # RB 换月
        ("2026-05-13", "RB2605", 40000, 3200.0, 3198.0),
        ("2026-05-13", "RB2606", 110000, 3200.0, 3202.0),
        # M: 全程 2607 主力, 不换月, 05-12 应保留值
        ("2026-05-11", "M2607", 50000, 2800.0, 2802.0),
        ("2026-05-12", "M2607", 51000, 2810.0, 2812.0),
        ("2026-05-13", "M2607", 52000, 2820.0, 2822.0),
    ]
    monkeypatch.setattr(ID, "_read_local_raw",
                        lambda dates, universe, freq="daily": make_daily_raw(rows))
    dates2 = pd.date_range("2026-05-11", "2026-05-13", freq="D")
    settle = ID._read_local_daily(EmptyData(), dates2, ["RB", "M"], "settle")
    # RB 换月日 NaN
    assert np.isnan(settle.loc["2026-05-12", "RB"])
    # M 同日(05-12)不换月, 值必须保留
    assert settle.loc["2026-05-12", "M"] == pytest.approx(2810.0)
    assert settle.loc["2026-05-11", "M"] == pytest.approx(2800.0)
    assert settle.loc["2026-05-13", "M"] == pytest.approx(2820.0)


def test_new_factors_no_lookahead():
    """#436-438 compute 输出 shift(1) 后首行应为 NaN(无未来数据泄漏)."""
    dates = pd.date_range("2026-03-01", "2026-08-01", freq="B")
    universe = ["RB", "M", "IF", "AU", "JM", "CU", "IC"]
    for cls_name in ["IntradayDaysToRollover20d", "IntradayRolloverSettleGap20d",
                     "IntradayRolloverBasisGap20d"]:
        cls = getattr(ID, cls_name)
        out = cls().compute(EmptyData(), dates, universe)
        assert isinstance(out, pd.DataFrame)
        # compute 内部已 shift(1), 首行应无值(无泄漏)
        assert out.iloc[0].isna().all()
        assert out.index[0] == dates[0]
