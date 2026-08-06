# -*- coding: utf-8 -*-
"""查 settle 因子的编号注释与注册名."""
import re

src = open(r"E:\程明杰公司内容\multi_factor\factors\library\intraday.py", encoding="utf-8").read()
lines = src.split("\n")
classes = ["IntradaySettleDrift20d", "IntradaySettlePosition20d", "IntradaySettleOiChange20d",
           "IntradaySettleGap20d", "IntradaySettleVolRatio20d", "IntradaySettleCloseBasis20d",
           "IntradaySettleBasisMomentum20d", "IntradaySettleOiSignal20d", "IntradaySettleBasisRank20d",
           "IntradaySettleBasisZ20d", "IntradaySettleDiffRank20d", "IntradaySettleSurgeZ20d"]
for c in classes:
    for i, l in enumerate(lines):
        if l.startswith("class " + c):
            j = i
            while j > 0 and not re.match(r"#\s*\d+\.", lines[j]):
                j -= 1
            m = re.search(r"#\s*(\d+)\.", lines[j]) if j >= 0 else None
            k = i
            while k > 0 and "@register_factor" not in lines[k]:
                k -= 1
            rm = re.search(r'@register_factor\("([a-z0-9_]+)"', lines[k]) if k >= 0 else None
            print(f"#{m.group(1) if m else '?'} {rm.group(1) if rm else '?'}  <- {c}")
            break
