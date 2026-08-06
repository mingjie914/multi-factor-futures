# -*- coding: utf-8 -*-
"""settle 修复回归(带 close): 12 个旧 settle 因子 compute 成功 + 覆盖度."""
import sys, warnings, pandas as pd, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, r"E:\程明杰公司内容\multi_factor")
sys.path.insert(0, r"E:\程明杰公司内容\multi_factor\factors\library")
import intraday as ID


class FakeData:
    """从本地 1d 提供 close/volume 面板(与 _read_local_daily 同口径: 主力具体合约)."""
    def get(self, field, dates, universe):
        if field not in ("close", "volume", "open", "high", "low", "position"):
            return None
        col = field
        idx = pd.DatetimeIndex(dates)
        try:
            raw = ID._read_local_raw(idx, universe, freq="daily")
            if raw is None or col not in raw.columns:
                return None
            df = raw[["trade_date", "root", "position", "symbol", col]].copy()
            df["td"] = pd.to_datetime(df["trade_date"]).dt.normalize()
            df = df.dropna(subset=["position"])
            df = df[df["symbol"].map(ID._expiry_ym).notna()]
            if df.empty:
                return None
            idm = df.groupby(["td", "root"])["position"].idxmax()
            main = df.loc[idm]
            pivot = main.pivot(index="td", columns="root", values=col)
            pivot.index = pd.DatetimeIndex(pivot.index)
            return pivot.reindex(index=idx, columns=universe)
        except Exception:
            return None


dates = pd.date_range("2026-03-01", "2026-08-01", freq="B")
universe = ["RB", "M", "IF", "AU", "JM", "CU", "IC"]
classes = ["IntradaySettleDrift20d", "IntradaySettlePosition20d", "IntradaySettleOiChange20d",
           "IntradaySettleGap20d", "IntradaySettleVolRatio20d", "IntradaySettleCloseBasis20d",
           "IntradaySettleBasisMomentum20d", "IntradaySettleOiSignal20d", "IntradaySettleBasisRank20d",
           "IntradaySettleBasisZ20d", "IntradaySettleDiffRank20d", "IntradaySettleSurgeZ20d"]
data = FakeData()
for cls_name in classes:
    cls = getattr(ID, cls_name, None)
    if cls is None:
        print(f"{cls_name}: 类不存在"); continue
    try:
        out = cls().compute(data, dates, universe)
        if isinstance(out, pd.DataFrame) and not out.empty:
            cov = float(out.notna().mean().mean())
            nvals = int(out.notna().sum().sum())
            fin = float(np.abs(out.values[out.notna().values]).max()) if nvals else 0.0
            flag = " !!!INF/异常" if not np.isfinite(fin) else ""
            print(f"{cls_name}: OK  覆盖={cov:.3f}  有限值={nvals}  maxAbs={fin:.4g}{flag}")
        else:
            print(f"{cls_name}: 返回空 {type(out)}")
    except Exception as e:
        print(f"{cls_name}: FAIL {type(e).__name__}: {e}")
