# -*- coding: utf-8 -*-
"""只读勘察 v2: 排除合成合约后 near/far 稳定性、到期月解析、年化可行性"""
import pandas as pd
import numpy as np
import os, re, warnings
warnings.filterwarnings("ignore")

BASE = r"E:\程明杰公司内容\期货行情数据\本地表"
MONTHS = ["2026-05", "2026-06", "2026-07"]
ROOTS = ["RB", "M", "IF", "AU", "JM", "CU", "IC"]
SYN = {"8888", "9998", "9999"}


def expiry(sym):
    m = re.match(r"^[A-Za-z]+(\d{4})$", sym)
    if not m:
        return None
    yy, mm = int(m.group(1)[:2]), int(m.group(1)[2:])
    return 2000 + yy, mm


def load(root, months):
    fs = []
    for ym in months:
        p = os.path.join(BASE, "futureshistoryprices1m", f"year_month={ym}", "data_0.parquet")
        if os.path.exists(p):
            df = pd.read_parquet(p, columns=["symbol", "trade_datetime", "close", "volume", "position", "trade_date"])
            df = df[df["symbol"].str.match(rf"^{root}[A-Z0-9]{{4}}$")]
            fs.append(df)
    return pd.concat(fs) if fs else pd.DataFrame()


for root in ROOTS:
    df = load(root, MONTHS)
    if df.empty:
        print(f"{root}: 无数据")
        continue
    concrete = df[~df["symbol"].str.endswith(("8888", "9998", "9999"))].copy()
    top2 = (concrete.sort_values("trade_date")
            .groupby(["trade_date", "symbol"])["position"].last().reset_index()
            .sort_values(["trade_date", "position"], ascending=[True, False]))
    top2["rank"] = top2.groupby("trade_date")["position"].rank(ascending=False, method="first")
    near = top2[top2["rank"] == 1].set_index("trade_date")
    far = top2[top2["rank"] == 2].set_index("trade_date")
    days = sorted(near.index.unique())
    switches = 0
    prev = None
    for d in days:
        m = far["symbol"].get(d)
        if m is not None:
            mm = expiry(m)
            if prev is not None and mm != prev:
                switches += 1
            if mm is not None:
                prev = mm
    both = 0
    total = 0
    far_is_3plus = 0
    far_stable_month = 0  # far 与近月相差恰好1个月
    for d in days:
        n, f = near["symbol"].get(d), far["symbol"].get(d)
        if n is None or f is None:
            continue
        nm, fm = expiry(n), expiry(f)
        if nm and fm:
            total += 1
            y, mo = nm
            mo2 = mo + 1
            if mo2 > 12:
                y, mo2 = y + 1, 1
            if (y, mo2) == fm:
                both += 1
            if fm > (y, mo2):
                far_is_3plus += 1
    nd = len(days)
    d0 = days[len(days) // 2] if nd else None
    n0 = near["symbol"].get(d0) if d0 else None
    f0 = far["symbol"].get(d0) if d0 else None
    p0n = near["position"].get(d0) if d0 else None
    p0f = far["position"].get(d0) if d0 else None
    print(f"{root}: 样本日{nd} far切换{switches} | far=近月+1月: {both}/{total} ({100*both/max(total,1):.0f}%) | far为远季(>近月+1): {far_is_3plus}/{total} ({100*far_is_3plus/max(total,1):.0f}%)")
    if d0 is not None:
        print(f"   样例 {d0}: near={n0} pos={p0n:.0f}  far={f0} pos={p0f:.0f}")
