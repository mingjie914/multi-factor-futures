# 因子库三层结构 (Validated Factor Library)

> 维护: 2026-08-04 | 自动生成: `scripts/export_factor_library.py`
> **三层结构**:
> - **总因子池**: 全部 396 个 intraday_advanced 因子 (含 K/V 系列), 定义于 `factors/library/intraday.py`
> - **有效因子库**: 通过 FDR + 相关性去重 + 候选间去重的独立候选 (不入组合, 备后续复评)
> - **组合因子集**: 进入等权打分组合的生产因子 (最终组合)

---

## 第一层: 组合因子集 (生产, 6 个)

**当前 6 个** | 来源: `strategies/combined.py::FACTORS` | 品种池: 38 | 调仓: 日度 | 权重: 池内 ERC | 板块配额: cap=3

| # | 注册名 | 方向 | intraday编号 | 简要说明 | 簇 |
|---|--------|------|-------------|----------|-----|
| 1 | `intraday_jump_intensity_20d` | 负向 | 8 | 跳跃强度 (价格跳跃剧烈→博彩特征) | 跳跃 |
| 2 | `intraday_price_peak_count_20d` | 正向 | 13 | 价峰计数 (无缺口跳跃→价格发现) | 跳跃 |
| 3 | `intraday_realised_skewness_20d` | 正向 | 4 | 已实现偏度 (正偏度→正收益) | 分布 |
| 4 | `intraday_dtws_20d` | 正向 | 9 | 跌幅时间重心 (尾盘下跌→反转) | 分布 |
| 5 | `intraday_drip_stone_20d` | 负向 | 15 | 滴水穿石 (高频量节奏周期) | 频谱 |
| 6 | `intraday_peak_ridge_ratio_20d` | 负向 | 10 | 峰脊比 (脉冲 vs 持续放量) | 频谱 |

**性能** (2026-08-04 时序修正后, 38品种+cap=3, T-1信号×T日收益): 全段夏普 **2.04** / 年化 18.9% / 回撤 -5.9% (2025-01~2026-07) / OOS 1.33 / 实盘 -0.33

**说明**: 2026-08-04 时序 bug 修正后 6 因子 (2.04) > 7 因子 (1.73), 席位因子 #326 回退。生产维持 6 因子。

---

## 第二层: 有效因子库 (通过全量检验, 27 个独立候选)

> 2026-08-04 全量检验 (396 因子) → 132 通过 FDR → 与生产 6 因子去重 (corr<0.5) → 64 个独立 → 候选间贪心去重 → **27 个**。
> 注: 前向选择显示 25 因子组合夏普 2.80 但**实盘段证伪** (-0.61 vs 6因子 -0.33), 故**不入组合**, 保留备复评。

| # | 注册名 | 方向 | |t| | IC | 周期 |
|---|--------|------|------|-----|------|
| 1 | `intraday_zero_ret_freq_20d` | 负向 | 11.95 | -0.317 | 20 |
| 2 | `intraday_open_close_drift_20d` | 正向 | 6.80 | +0.194 | 20 |
| 3 | `intraday_volatility_clustering_20d` | 负向 | 6.54 | -0.193 | 20 |
| 4 | `intraday_oi_vol_corr_daily_20d` | 正向 | 6.30 | +0.159 | 20 |
| 5 | `intraday_oi_time_centroid_20d` | 负向 | 5.85 | -0.220 | 20 |
| 6 | `intraday_wash_trade_20d` | 正向 | 5.54 | +0.213 | 20 |
| 7 | `intraday_settle_position_20d` | 负向 | 5.05 | -0.126 | 20 |
| 8 | `intraday_cross_vol_20d` | 正向 | 4.13 | +0.155 | 20 |
| 9 | `intraday_amihud_vol_ratio_20d` | 正向 | 4.12 | +0.096 | 20 |
| 10 | `intraday_oi_skew_stability_20d` | 负向 | 3.66 | -0.148 | 20 |
| 11 | `intraday_depth_trend_20d` | 正向 | 3.57 | +0.106 | 20 |
| 12 | `intraday_open_close_volume_ratio_20d` | 负向 | 3.39 | -0.088 | 20 |
| 13 | `intraday_oi_quantile_range_20d` | 负向 | 3.30 | -0.122 | 20 |
| 14 | `intraday_settle_gap_20d` | 正向 | 3.26 | +0.096 | 10 |
| 15 | `intraday_amihud_trend_20d` | 正向 | 3.14 | +0.056 | 5 |
| 16 | `intraday_price_delay_20d` | 负向 | 2.96 | -0.100 | 20 |
| 17 | `intraday_overnight_absorption_20d` | 正向 | 2.87 | +0.121 | 20 |
| 18 | `intraday_session_symmetry_20d` | 负向 | 2.81 | -0.069 | 10 |
| 19 | `intraday_lowest_time_20d` | 正向 | 2.79 | +0.119 | 20 |
| 20 | `intraday_oi_peak_ridge_ratio_20d` | 负向 | 2.75 | -0.068 | 5 |
| 21 | `intraday_seat_long_short_seat_ratio_20d` | 正向 | 2.70 | +0.066 | 10 |
| 22 | `intraday_volume_rank_ratio_20d` | 正向 | 2.70 | +0.064 | 20 |
| 23 | `intraday_herding_20d` | 正向 | 2.69 | +0.159 | 20 |
| 24 | `intraday_volume_time_shape_20d` | 正向 | 2.67 | +0.079 | 20 |
| 25 | `intraday_extreme_freq_balance_20d` | 负向 | 2.51 | -0.045 | 10 |
| 26 | `intraday_term_vol_ratio_20d` | 正向 | 2.31 | +0.103 | 10 |
| 27 | `intraday_turnover_velocity_20d` | 正向 | 2.28 | +0.163 | 20 |

**处置结论**: 全部 27 个通过全量检验但**组合验证显示不入组合更优** (实盘段衰减),
保留为有效候选池, 供未来数据延长或市场环境变化时复评。

---

## 第三层: 总因子池 (全部 396 个)

定义于 `factors/library/intraday.py` (含 #1-#380 编号因子 + K1-K10 + V1-V10 系列)。
无需重复全量检验, 已在 2026-08-04 完成一次全量筛选 (结果见第二层)。

---

## 因子库更新流程

1. 新因子开发 → 加入 `factors/library/intraday.py`
2. 全量检验: `main.py research --factors <全部> --run-id <批次>`
3. 与生产 6 因子相关性去重 (corr<0.5) + 候选间去重
4. 组合增益验证 (前向选择) → 通过实盘段验证才入组合因子集
5. 更新本文档三层结构
