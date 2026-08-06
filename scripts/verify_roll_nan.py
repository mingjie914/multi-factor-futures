# -*- coding: utf-8 -*-
"""验证: 每品种主力切换日(换月日)在 settle 面板上均为 NaN"""
import sys, warnings, pandas as pd, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, r"E:\程明杰公司内容\multi_factor")
import factors.library.intraday as ID

class FakeData:
    def get(self, col, dates, universe):
        return None

dates = pd.date_range("2026-04-01", "2026-08-01", freq="D")
universe = ["RB", "M", "IF", "AU", "JM", "CU", "IC"]
settle = ID._read_local_daily(FakeData(), dates, universe, "settle")

for root in universe:
    raw = ID._read_local_raw(dates, [root], freq="daily")
    df = raw[raw["root"] == root].copy()
    df = df[df["symbol"].map(ID._expiry_ym).notna()]
    df["td"] = pd.to_datetime(df["trade_date"]).dt.normalize()
    # 每交易日主力 symbol
    top = df.sort_values(["td", "position"], ascending=[True, False]).groupby("td").head(1).sort_values("td")
    syms = top["symbol"].astype(str).to_numpy()
    tds = pd.to_datetime(top["td"].to_numpy())
    roll_days = []
    for i in range(1, len(syms)):
        if syms[i] and syms[i - 1] and syms[i] != syms[i - 1]:
            roll_days.append(pd.Timestamp(tds[i]).normalize())
    # 面板在换月日是否 NaN
    ok = all(pd.isna(settle.loc[d, root]) if d in settle.index else True for d in roll_days)
    days_str = [str(d.date()) for d in roll_days]
    # 面板中额外 NaN 交易日(排除周末) = 无数据日
    trade_days = set(pd.to_datetime(df["td"]))
    extra_nan = [str(d.date()) for d in settle.index if d in trade_days and pd.isna(settle.loc[d, root]) and pd.Timestamp(d) not in roll_days]
    print(f"{root}: 换月日 {len(roll_days)} 次 {days_str[:6]} | 全部置NaN: {ok} | 额外交易日NaN: {extra_nan[:6]}")
