# -*- coding: utf-8 -*-
"""集成验证: 真实 runs 产物的归因跑通 + 对账."""
import sys, warnings
sys.path.insert(0, r"E:\程明杰公司内容\multi_factor")
warnings.filterwarnings("ignore")
import pandas as pd
import monitoring.attribution as AT

ledger_dir = r"E:\程明杰公司内容\multi_factor\runs\wf_intraday_6\折1\portfolio\sub_portfolios\mid_term"
attr = AT.AttributionReport(ledger_dir=ledger_dir)
sc = attr.sector_contribution()
print("板块归因(周/月/YTD):")
print(sc.to_string() if not sc.empty else "(空)")
ac = attr.asset_contribution()
print("\n品种层(前 12 行, 上多下空):")
print(ac.head(12).to_string() if not ac.empty else "(空)")
total = pd.read_csv(ledger_dir + "/research_return_contributions.csv", index_col=0).sum().sum()
sc_ytd = sc["ytd"].sum() if "ytd" in sc.columns else 0.0
print(f"\n对账: 板块归因 YTD 合计 = {sc_ytd:.6f}  vs 总贡献 = {total:.6f}  -> {'OK' if abs(sc_ytd - total) < 1e-6 else 'FAIL'}")
