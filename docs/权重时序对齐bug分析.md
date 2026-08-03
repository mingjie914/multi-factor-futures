# 权重时序对齐 bug 分析 (2026-08-04)

> 状态: **已修复 (2026-08-04)**. 见文末修复记录.

## 一、你的质疑与结论

你问: "T 日收益 = 权重 × T 日价格变动, 还是 T-1 日权重 × T 日价格变动?"

**正确答案: T-1 权重 × T 日价格变动.** (你的直觉正确)

## 二、因子内建 shift(1) 的语义

以 `intraday_jump_intensity_20d` 为例, compute 结尾:

```python
return _roll_mean(daily, 20, 5).reindex(dates).shift(1).reindex(columns=universe)
```

- `daily` = T 日日内聚合值 (用了 T 日全天 1min 数据)
- `_roll_mean(daily, 20, 5)` = T 日 20 日均值 (含 T 日)
- `.shift(1)` = **T 日值改用 ≤T-1 日的 20 日均值** → **T-1 收盘后即可计算, 无未来信息**

所以: **因子 T 日值 = 基于 ≤T-1 数据 → T-1 收盘可生成权重 → T 日持有 → 赚 T 日收益**
即: **权重[T] × 收益[T]** (同日), 等价于 **T-1 信号 × T 日价格变动**.

## 三、production 脚本的 bug

`scripts/diag_production_nav.py`:

```python
daily_ret = close.pct_change()
fwd = daily_ret.shift(-1)        # fwd[T] = T+1 日收益
...
for t in rebal.index:            # t = 权重日 (因子已 shift1 = T-1 信号)
    ...
    nxt = cal[(cal > t) & (cal <= t + 1天)]   # nxt = t+1
    for d in nxt:
        r = fwd.loc[d]           # fwd[t+1] = daily_ret[t+2]  ← 两日偏移!
```

**实际时序 = 权重T × 收益T+2 (两日滞后)**. 正确应为 权重T × 收益T (同日).

## 四、实证复算 (用导出权重 + 真实收益)

| 对齐方式 | 年化 | 夏普 | 回撤 |
|---------|------|------|------|
| 权重T × 收益T (同日, 正确) | 16.8% | 2.09 | -4.3% |
| 权重T × 收益T+1 | 18.1% | 1.95 | -5.0% |
| 权重T × 收益T+2 (production 脚本实际) | 19.0% | 2.19 | -4.4% |
| **production 报告声称 (2.27)** | 19.9% | 2.27 | -5.1% |

**关键**: production 脚本(2.19) 与 权重文件复算同日(2.09) 不一致, 说明脚本本身有额外偏差
(可能: ERC 权重计算差异 / rebalance 日期过滤差异). 需修正后统一.

## 五、需要修正的内容清单

1. **`scripts/diag_production_nav.py`**: 改 `fwd` 逻辑为 同日收益
   (`r = daily_ret.loc[t]`, 去掉 `shift(-1)` 与 `nxt` 两日偏移)
2. **`scripts/export_weights.py`**: 确认权重文件日期语义 = "T-1 收盘信号, T 日持有"
   (当前已正确: 因子 shift1 后取 score.loc[t], 即 T-1 信号) — 但需与修正后回测对齐验证
3. **所有基于 production 脚本的报告数字**: 2.27/19.9%/-5.1% 等需用修正后口径重算
   (`docs/有效因子库.md`, `docs/策略基准记录.md`, `docs/生产方案运行报告_20260804.md`,
   `框架工作流程与使用方法.md`)
4. **验证脚本**: `scripts/verify_alignment2.py` (手工 ERC, 避免 scipy) 待环境就绪后跑,
   确认修正后 同日口径 与 权重文件复算 完全一致

## 六、为什么之前没发现

- 因子内建 shift(1) 使"看起来像次日生效", 掩盖了脚本额外的一日偏移
- 2.27 数字来自脚本自身, 与权重文件从未交叉验证
- 你的外部程序复算需求恰好暴露了时序不一致

## 七、待环境恢复后立即执行

1. 恢复 `E:\Python\Pythonenv` (含 pandas/scipy)
2. 跑 `verify_alignment2.py` 确认正确口径数值
3. 修 `diag_production_nav.py` + `export_weights.py` 对齐
4. 重算全部报告数字并更新文档


---

## 修复记录 (2026-08-04)

**已执行**:
1. scripts/diag_production_nav.py: 移除 wd = daily_ret.shift(-1) 与 
xt 两日偏移,
   改为 
 = daily_ret.loc[t] (同日收益 = T-1 信号 × T 日价格变动)
2. scripts/export_weights.py: 确认同日对齐, 重新导出 weights/daily_weights.csv
3. scripts/verify_final.py: 方案A(production修正后) vs 方案B(权重文件复算)
   **逐日收益完全一致 (372 天, 最大差异 0.000000)** — 权重文件与框架口径严格吻合

**修正后正确口径** (2025-01-01~2026-07-31, 7因子+38池+cap3+ERC+日度):
- 全段: 年化 15.8% / 夏普 1.73 / 回撤 -6.1%
- OOS (3-1~5-15): 夏普 -0.50 / 回撤 -3.9%
- 实盘 (5-16~7-31): 夏普 0.37 / 回撤 -2.6%

**此前报告的 2.27/19.9%/-5.1% 为时序错误口径, 已废弃** (两日滞后虚高).
3 份文档 (有效因子库/策略基准/生产报告) 中的数字需按新口径更新.
