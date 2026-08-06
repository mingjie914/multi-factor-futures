# -*- coding: utf-8 -*-
"""验证 _read_local_daily 修复: 主力=具体合约, 换月日 NaN, 无跨合约跳变."""
import sys, warnings, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, r"E:\程明杰公司内容\multi_factor")
import factors.library.intraday as ID

class FakeData:
    def get(self, col, dates, universe):
        return None

dates = pd.date_range("2026-04-01", "2026-08-01", freq="D")
universe = ["RB", "M", "IF", "AU", "JM", "CU", "IC"]
settle = ID._read_local_daily(FakeData(), dates, universe, "settle")
print("settle 面板形状:", settle.shape)
for root in universe:
    s = settle[root].dropna()
    if s.empty:
        print(f"{root}: 无数据"); continue
    nan_days = settle.index[settle[root].isna()].strftime("%Y-%m-%d")
    jumps = s.diff().dropna()
    big = jumps[abs(jumps) > 20]
    print(f"{root}: 有效{len(s)}日 | 换月NaN日 {list(nan_days)[:6]} | |Δsettle|>20跳变 {len(big)}")
