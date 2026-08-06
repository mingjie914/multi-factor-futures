# -*- coding: utf-8 -*-
"""只读勘察 v3: 合成合约是否进入 top2 + 1d 主力切换日 settle 跳变"""
import pandas as pd
import os, warnings
warnings.filterwarnings("ignore")

BASE = r"E:\程明杰公司内容\期货行情数据\本地表"

# 1) _read_local_raw 等价: 1min 分区读全部 symbol(含合成?)
p = os.path.join(BASE, "futureshistoryprices1m", "year_month=2026-06", "data_0.parquet")
df = pd.read_parquet(p, columns=["symbol", "trade_datetime", "position", "close"])
rb = df[df["symbol"].str.startswith("RB")]
day = "2026-06-18"
sub = rb[rb["trade_datetime"].dt.date.astype(str) == day]
top = sub.groupby("symbol")["position"].last().sort_values(ascending=False).head(5)
print("RB 2026-06-18 持仓 top5(含合成合约?):")
print(top.to_string())
print()

# 2) 1d 主力切换日 settle 跳变 (JM 焦煤, 052 研报同品种)
p1 = os.path.join(BASE, "futureshistoryprices1d", "year_month=2026-07", "data_0.parquet")
d1 = pd.read_parquet(p1, columns=["symbol", "trade_date", "close", "position", "settle_price"])
jm = d1[d1["symbol"].str.startswith("JM") & ~d1["symbol"].str.endswith(("8888", "9998", "9999"))]
jm = jm.sort_values("trade_date")
main_days = []
for dt, g in jm.groupby("trade_date"):
    g2 = g.dropna(subset=["position"])
    if not g2.empty:
        top1 = g2.sort_values("position", ascending=False).iloc[0]
        main_days.append((dt, top1["symbol"], top1["settle_price"], top1["close"]))
prev = None
jumps = []
for dt, sym, settle, close in main_days:
    if prev is not None and sym != prev[1]:
        jumps.append((str(prev[0])[:10], prev[1], sym, prev[2], settle, prev[3], close))
    prev = (dt, sym, settle, close)
print("JM 2026-05~07 主力切换日 settle/close 跳变(旧主力→新主力):")
for j in jumps[:8]:
    print(f"  {j[0]} {j[1]}->{j[2]}: settle {j[3]} -> {j[4]} (Δ={j[4]-j[3]:.2f}) | close {j[5]} -> {j[6]} (Δ={j[6]-j[5]:.2f})")
if not jumps:
    print("  (该区间无主力切换)")
print()

# 3) 近月-次月 是否始终存在流动性 (M 豆粕: 用到期月最近的两个合约)
p2 = os.path.join(BASE, "futureshistoryprices1d", "year_month=2026-06", "data_0.parquet")
d2 = pd.read_parquet(p2, columns=["symbol", "trade_date", "position", "close"])
d2 = d2[d2["symbol"].str.match(r"^M\d{4}$")]
d2 = d2.sort_values("trade_date")
sample = d2[d2["trade_date"].astype(str) == "2026-06-18"]
if not sample.empty:
    print("M 2026-06-18 各到期月 position/close:")
    s = sample.groupby("symbol")[["position", "close"]].last().sort_index()
    print(s.to_string())
