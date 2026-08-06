# -*- coding: utf-8 -*-
"""验证新增因子 #433-435: 注册、compute 运行、覆盖度、换月标记有效性"""
import sys
import os
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"E:\程明杰公司内容\multi_factor")
sys.path.insert(0, r"E:\程明杰公司内容\multi_factor\factors\library")
import pandas as pd
import intraday as ID

# 1) 注册确认
names = [n for n in ID.REGISTERED_FACTORS] if hasattr(ID, "REGISTERED_FACTORS") else []
new_names = [n for n in dir(ID) if n.startswith("IntradayAnnualizedBasis") or
             n.startswith("IntradayBasisMomentum") or n.startswith("IntradayRolloverFrequency")]
print("新因子类:", new_names)
# 通过 register 表查
reg = getattr(ID, "REGISTERED_FACTORS", None)
if reg is None:
    # 尝试从模块级注册表
    for attr in dir(ID):
        if "register" in attr.lower() and not attr.startswith("__"):
            print("  注册表属性:", attr)
else:
    for n in ["intraday_annualized_basis_20d", "intraday_basis_momentum_20d", "intraday_rollover_frequency_20d"]:
        print(f"  {n}: 已注册={n in reg}" if isinstance(reg, dict) else f"  {n}: in list={n in reg}")

# 2) compute 运行 (因子 frequency=daily, 框架按日度 dates 调用)
dates = pd.date_range("2026-03-01", "2026-07-31", freq="D")
universe = ["RB", "M", "IF", "AU", "JM", "CU", "IC"]
for cls_name in ["IntradayAnnualizedBasis20d", "IntradayBasisMomentum20d", "IntradayRolloverFrequency20d"]:
    cls = getattr(ID, cls_name)
    f = cls()
    try:
        out = f.compute(None, dates, universe)
        if isinstance(out, pd.DataFrame) and not out.empty:
            cov = out.notna().mean().mean()
            # 按列看覆盖
            per_col = out.notna().mean()
            print(f"{cls_name}: OK 行={len(out)} 总覆盖={cov:.3f}")
            for c in universe:
                print(f"    {c}: 覆盖={per_col.get(c, float('nan')):.2f} 均值={out[c].mean():+.4f} 非零={ (out[c]!=0).mean():.2f}")
        else:
            print(f"{cls_name}: 返回空/异常形状 {type(out)}")
    except Exception as e:
        import traceback
        print(f"{cls_name}: FAIL {e}")
        traceback.print_exc()
