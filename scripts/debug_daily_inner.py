# -*- coding: utf-8 -*-
"""逐行复现 _read_local_daily 本地分支, 定位全 NaN 原因"""
import sys, warnings, pandas as pd, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, r"E:\程明杰公司内容\multi_factor")
import factors.library.intraday as ID

dates = pd.date_range("2026-07-25", "2026-08-05", freq="D")
universe = ["JM"]
raw = ID._read_local_raw(dates, universe, freq="daily")
print("raw:", None if raw is None else raw.shape, "| trade_date in cols:", "trade_date" in raw.columns if raw is not None else "-")

col = "settle_price"
df = raw[["_ts", "root", "position", "symbol", col]].copy()
df["_ts"] = pd.to_datetime(df["trade_date"]).dt.normalize()
df = df.dropna(subset=["position"])
print("df after dropna:", df.shape)
df = df[df["symbol"].map(ID._expiry_ym).notna()]
print("df after expiry filter:", df.shape)
if df.empty:
    sys.exit(0)
idx = df.groupby(["_ts", "root"])["position"].idxmax()
print("idxmax 数量:", len(idx), "| 唯一:", idx.is_unique)
main = df.loc[idx]
print("main 行数:", len(main))
print(main[["_ts", "symbol", "position", col]].sort_values("_ts").to_string())
