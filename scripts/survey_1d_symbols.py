# -*- coding: utf-8 -*-
"""勘察 1d 数据 symbol 形态与合成合约命名, 供 _read_local_daily 排除规则设计."""
import pandas as pd, os, re, sys

BASE = r"E:\程明杰公司内容\期货行情数据\本地表"
ROOTS = ["RB", "M", "IF", "AU", "JM", "CU", "IC"]
MONTHS = ["2026-05", "2026-06", "2026-07"]

for root in ROOTS:
    syms = set()
    for ym in MONTHS:
        p = os.path.join(BASE, "futureshistoryprices1d", f"year_month={ym}", "data_0.parquet")
        if os.path.exists(p):
            df = pd.read_parquet(p, columns=["symbol"])
            syms |= set(df[df["symbol"].str.startswith(root)]["symbol"].unique())
    concrete = sorted(s for s in syms if re.match(rf"^{root}\d{{4}}$", s))
    synthetic = sorted(s for s in syms if not re.match(rf"^{root}\d{{4}}$", s))
    print(f"{root}: 具体合约 {len(concrete)} 个: {concrete[:8]}{'...' if len(concrete)>8 else ''}")
    print(f"   非标准(合成?) {len(synthetic)} 个: {synthetic[:10]}")
