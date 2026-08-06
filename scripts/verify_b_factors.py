# -*- coding: utf-8 -*-
"""B 阶段新因子 #436-438 验证: compute 成功 + 覆盖度 + 数值合理性."""
import sys, warnings, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, r"E:\程明杰公司内容\multi_factor")
sys.path.insert(0, r"E:\程明杰公司内容\multi_factor\factors\library")
import intraday as ID

dates = pd.date_range("2026-03-01", "2026-08-01", freq="B")
universe = ["RB", "M", "IF", "AU", "JM", "CU", "IC"]

for cls_name in ["IntradayDaysToRollover20d",
                 "IntradayRolloverSettleGap20d",
                 "IntradayRolloverBasisGap20d"]:
    cls = getattr(ID, cls_name, None)
    if cls is None:
        print(f"{cls_name}: 类不存在"); continue
    f = cls()
    try:
        out = f.compute(None, dates, universe)
        if isinstance(out, pd.DataFrame) and not out.empty:
            cov = float(out.notna().mean().mean())
            nvals = int(out.notna().sum().sum())
            print(f"{cls_name}: OK  覆盖={cov:.3f}  有限值={nvals}")
            for c in universe:
                s = out[c].dropna()
                if len(s):
                    print(f"    {c}: 覆盖={s.size/out.shape[0]:.2f} 均值={s.mean():+.4f} 非零={float((s!=0).mean()):.2f}")
        else:
            print(f"{cls_name}: 返回空 {type(out)}")
    except Exception as e:
        print(f"{cls_name}: FAIL {type(e).__name__}: {e}")
