# prod_snapshot_12f_icir — 12因子 IC_IR 产品分支（只读快照）

> 创建: 2026-08-05 | 状态: **候选产品分支，暂不替换生产**
> 生产组合仍为 `prod_snapshot_6f`（6因子+等权+cap3）。本分支独立演进，作为对比基线。

## 组合配置

| 项目 | 值 |
|------|-----|
| 因子集 | **12 个**（6生产 + 6前向搜索增量） |
| 加权 | **池内 IC_IR**（滚动60日, Ledoit-Wolf收缩协方差） |
| 选池 | cap=3 板块配额（全市场排名, 每板块最多3个多/空） |
| 品种池 | UNIVERSE38 |
| 调仓 | 日度 |
| 杠杆 | 2.0（多头+1.0 / 空头-1.0） |

## 12 因子（方向）

### 生产 6
| 因子 | 方向 |
|------|------|
| intraday_jump_intensity_20d | -1 |
| intraday_price_peak_count_20d | +1 |
| intraday_realised_skewness_20d | +1 |
| intraday_dtws_20d | +1 |
| intraday_drip_stone_20d | -1 |
| intraday_peak_ridge_ratio_20d | -1 |

### 前向搜索增量 6
| 因子 | 方向 | 边际贡献 |
|------|------|---------|
| intraday_oi_time_centroid_20d | -1 | Δ-1.10 |
| intraday_wash_trade_20d | -1 | Δ-0.16 |
| intraday_price_delay_20d | -1 | Δ-0.56 |
| intraday_amihud_trend_20d | +1 | Δ-0.33 |
| intraday_amihud_vol_ratio_20d | +1 | Δ-0.27 |
| intraday_volume_time_shape_20d | +1 | Δ+0.01 |

## 性能（2026-08-05, 时序修正后口径）

| 指标 | 值 |
|------|-----|
| 全段(2025-01~2026-07) 夏普 | **3.80** |
| 年化 | 31.9% |
| 最大回撤 | -3.8% |
| 2025 夏普 | 4.11（收益 24.7%） |
| 2026 夏普 | 3.44（收益 17.5%） |
| OOS(3/1-5/15) | 2.22 |
| **实盘(5/16+)** | **+4.19** |
| 负月 | 1/18（最差月 -1.02%） |

**对比**：6因子 IC_IR 全段 2.56 / 实盘 +2.24；本分支全段 3.80 / 实盘 +4.19。

## 文件
- `combined.py` — 12因子+IC_IR+cap3 配置（只读）
- `export_weights.py` — 权重导出（IC_IR 滚动加权）
- `weights_daily.csv` — 363 交易日权重（date|symbol|weight）

## 备注
- 前向搜索在 27 候选内挑 6 增量，存在样本内选择偏差；实盘段（5/16+）+4.19 待更长验证
- 若后续确认稳健，可升级为生产；否则保持候选分支
