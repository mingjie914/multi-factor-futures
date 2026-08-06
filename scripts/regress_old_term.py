# -*- coding: utf-8 -*-
"""回归验证: 旧 term 因子在修复后面板下 compute 正常 (不崩溃、覆盖不降级)"""
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"E:\程明杰公司内容\multi_factor")
sys.path.insert(0, r"E:\程明杰公司内容\multi_factor\factors\library")
import pandas as pd
import intraday as ID

dates = pd.date_range("2026-03-01", "2026-07-31", freq="D")
universe = ["RB", "M", "IF", "AU", "JM", "CU", "IC"]
# 抽样旧 term 因子(依赖 _get_term_structure_panel 的 6min/5min 面板)
classes = [
    "IntradayTermSlope20d",        # #224
    "IntradayTermSlopeChange20d",  # #225
    "IntradayTermSpreadVol20d",    # #226
    "IntradayTermOiRatio20d",      # #227
    "IntradayTermSlopeMaCross20d", # #228
    "IntradayTermVolSpread20d",    # #229
    "IntradayAnnualizedBasisZ20d", # #419
    "IntradayBasisReversionConviction20d",  # #420
    "IntradayRollYieldDualscore20d",        # #421
]
for cls_name in classes:
    cls = getattr(ID, cls_name, None)
    if cls is None:
        print(f"{cls_name}: 类不存在")
        continue
    try:
        out = cls().compute(None, dates, universe)
        if isinstance(out, pd.DataFrame) and not out.empty:
            cov = float(out.notna().mean().mean())
            n_finite = float(out.stack().dropna().shape[0])
            print(f"{cls_name}: OK  覆盖={cov:.3f}  有限值={int(n_finite)}")
        else:
            print(f"{cls_name}: 空返回 ({type(out).__name__})")
    except Exception as e:
        print(f"{cls_name}: FAIL  {type(e).__name__}: {e}")
