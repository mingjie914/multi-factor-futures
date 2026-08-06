# -*- coding: utf-8 -*-
"""精确复现 _read_local_daily 内部: 看 _ts.normalize 口径下的主力序列与换月判定"""
import sys, warnings, pandas as pd, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, r"E:\程明杰公司内容\multi_factor")
import factors.library.intraday as ID

dates = pd.date_range("2026-07-25", "2026-08-05", freq="D")
raw = ID._read_local_raw(dates, ["JM"], freq="daily")
print("raw 中 _ts 与 trade_date 关系 (JM 07-29~08-01 附近):")
sub = raw[(raw["root"] == "JM")].copy()
sub = sub[sub["symbol"].map(ID._expiry_ym).notna()]
sub = sub[["_ts", "trade_date", "symbol", "position", "settle_price"]].copy()
sub["_ts"] = pd.to_datetime(sub["_ts"])
mask = (sub["_ts"].dt.normalize() >= "2026-07-28") & (sub["_ts"].dt.normalize() <= "2026-08-04")
sub = sub[mask].sort_values("_ts")
for _, r in sub.iterrows():
    print(f"  _ts={str(r['_ts'])[:16]}  trade_date={str(r['trade_date'])[:10]}  {r['symbol']}  pos={r['position']}  settle={r['settle_price']}")
