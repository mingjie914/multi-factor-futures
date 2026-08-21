# 30min 时段因子：用法与公式说明

> 维护: 2026-08-07 | 范围: intraday_advanced 中全部 30min 时段因子（#517/#528/#531/#532）
> 背景: 正式数据源会把15min分区显式重采样为30min；以下4个因子为固定交易时段边界，
> 仍主动读取真1min并在因子内聚合。本文件为它们的权威用法/公式说明。

---

## 0. 30min 标准写法（所有 30min 因子必须遵守）

```python
panel = _get_minute_panel(data, dates, universe, freq="1min", force_1min=True)
close = panel["close"].resample("30min").last()   # close 取窗口末分钟收盘价
vol   = panel["volume"].resample("30min").sum()   # volume 取窗口内之和
```

要点：
- `force_1min=True`：兼容入口，明确强制1min原始数据；不存在全局频率回退
- `resample("30min")`：按整点/半点对齐（09:00/09:30/10:00...），夜盘独立成窗、不跨日合并
- 输出统一为**日频**（`frequency="daily"`，每交易日 1 个值），滚动平滑 + `shift(1)` 防未来
- 这四个因子不要改为直接30min读取，否则会改变其冻结的时段聚合语义；通用30min路由本身是显式重采样

---

## 1. #517 intraday_golden_ratio_reversal_20d — 日内黄金分割反转

**公式**
```
anchor_t    = 当日第 1 个 30min bar 收盘价 (开盘 30 分钟后的价格水平)
factor_t    = ln(当日最后 30min bar 收盘价 / anchor_t)
cum21       = Σ_{过去21日} factor_t
输出        = -cum21 (负向因子)
```

**含义 / 方向**
- 收盘相对开盘时段价格偏离的 21 日累积 → 高值 = 早盘后持续走高 → 反转 → **负向**
- 与 #65 `intraday_return`（日内整体收益）区分：本因子是**21 日累积对数偏离**（反转视角）

**用法**：1min 输入 → 30min 聚合（close=last）→ 日频；锚 = 当日首 30min bar；`rolling(21)` 累计；`shift(1)`。

**验证**：2026-04-01 ~ 08-04 实测 nn=480、80 交易日有值；聚合后时间戳间隔 00:30:00 ✓

---

## 2. #528 intraday_reversal_consistency_20d — 反转一致性（#517 改进）

**公式**
```
factor_t    = ln(当日最后 30min bar 收盘价 / anchor_t)   # 同 #517 的日内偏离
trend20     = polyfit(1..20, factor_{t-19..t}) 的斜率     # 20 日趋势
输出        = -trend20 (负向因子)
```

**含义 / 方向**
- #517 度量 21 日累积偏离（水平），本因子度量其 **20 日趋势（斜率）** → 累积反转持续加深 = 反转动能 → **负向**
- 与 #517（水平）互补：本因子是反转的**动量**

**用法**：与 #517 同源同聚合；`rolling(20)` 趋势；`shift(1)`。

**验证**：实测 nn=462、77 交易日有值 ✓

---

## 3. #531 intraday_amt_ratio_entropy_30m_20d — 30分钟时段价量熵

**公式**
```
price_ratio_b  = close_30m,b / Σ_b close_30m,b          # 时段价格占比
volume_ratio_b = volume_30m,b / Σ_b volume_30m,b        # 时段成交量占比
p_b            = price_ratio_b × volume_ratio_b         # 联合权重 (再归一化为概率)
entropy_t      = -Σ_b p_b × ln(p_b)                     # 全天信息熵
输出           = rolling(20) 均值 (正向因子)
```

**含义 / 方向**
- 高熵 = 价量成交在日内各时段**分散均衡** → 资金参与持续、结构稳定 → **正向**
- 低熵 = 成交集中在少数时段 → 突发冲击/短时资金拥挤 → **负向**
- 与 #37 `volume_price_entropy`（分钟级 2D 直方图熵）、#478 `vol_bucket_entropy`（分桶熵）区分：
  本因子是 **30min 时段**的价占比×量占比联合权重熵，粒度更粗、含价格维度

**用法**：1min 输入 → 30min 聚合（close=last, volume=sum）→ 日频；`rolling(20)` 均值；`shift(1)`。

**验证**：实测 nn=480、80 交易日有值 ✓

---

## 4. #532 intraday_price_vol_entropy_diff_20d — 价格-成交量熵差（原创改进）

**公式**
```
price_ent = -Σ_b price_ratio_b × ln(price_ratio_b)     # 价格分布熵
vol_ent   = -Σ_b volume_ratio_b × ln(volume_ratio_b)   # 成交量分布熵
diff_t    = price_ent − vol_ent
输出      = -rolling(20) 均值 (负向因子)
```

**含义 / 方向**
- 价格时段熵高但成交量熵低 = 价格在多个时段变动、量却集中在少数时段 → 无量上涨(拉抬)/冲击集中 → 结构不健康 → **负向**
- 价低量高 = 量能分散但价格集中 → 持续建仓 → 正向
- 与 #531（联合熵**水平**）互补：本因子是两类分布的**结构差**

**用法**：与 #531 同源同聚合；`rolling(20)` 均值；`shift(1)`。

**验证**：实测 nn=480、80 交易日有值 ✓

---

## 附：同类风险排查记录

| 检查项 | 结果 |
|---|---|
| `freq="30min"` 实际调用残留 | 无（4 处残留均为注释警告） |
| #57 `variance_ratio_30m` | 不受影响——其 "30m" 是 **q=30 个 bar** 的 Lo-MacKinlay 方差比，用 1min/5min 源，非 30min 时段 |
| 管道层 | `_FREQ_DIR_MAP` 已加 ⚠ 注释，禁止未来直接请求 `freq="30min"` |
