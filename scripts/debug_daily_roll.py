# -*- coding: utf-8 -*-
"""debug: _read_local_daily 换月 NaN 是否生效 (JM 2026-07-30 应为 NaN)"""
import sys, warnings, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, r"E:\程明杰公司内容\multi_factor")
import factors.library.intraday as ID

class FakeData:
    def get(self, col, dates, universe):
        return None

dates = pd.date_range("2026-07-01", "2026-08-05", freq="D")
universe = ["JM"]
settle = ID._read_local_daily(FakeData(), dates, universe, "settle")
print("JM 2026-07-25 ~ 2026-08-03:")
for d in pd.date_range("2026-07-25", "2026-08-03", freq="D"):
    v = settle.loc[d, "JM"] if d in settle.index else None
    print(f"  {d.date()}: {v}")

# 直接检查 raw 中 JM 主力序列与切换 (排除合成)
raw = ID._read_local_raw(dates, ["JM"], freq="daily")
jm = raw[raw["root"] == "JM"].copy()
jm = jm[jm["symbol"].map(ID._expiry_ym).notna()]  # 排除合成
jm["_ts"] = pd.to_datetime(jm["_ts"]).dt.normalize()
top = jm.sort_values(["trade_date", "position"], ascending=[True, False]).groupby("trade_date").head(1)
print("\nJM 具体合约主力序列(07-25~08-03):")
for _, r in top.iterrows():
    if str(r["trade_date"])[5:10] >= "07-25":
        print(f"  {str(r['trade_date'])[:10]}  {r['symbol']}  pos={r['position']}  settle={r['settle_price']}")
