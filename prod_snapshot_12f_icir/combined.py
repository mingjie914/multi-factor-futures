"""12因子 IC_IR 产品分支 — 只读快照配置 (2026-08-05).

与 prod_snapshot_6f 的区别:
  - 因子集: 12 (6生产 + 6个前向搜索增量)
  - 加权: 池内 IC_IR (滚动60日, Ledoit-Wolf收缩协方差) — 非等权
  - 选池: cap=3 板块配额 (同生产)
  - 品种池: UNIVERSE38
  - 调仓: 日度

注意: 此分支是"候选产品", 暂不替换生产组合. 生产仍为 prod_snapshot_6f.
"""
# ============ 因子集 (12) ============
FACTORS = {
    # 生产 6
    'intraday_jump_intensity_20d': -1,
    'intraday_price_peak_count_20d': +1,
    'intraday_realised_skewness_20d': +1,
    'intraday_dtws_20d': +1,
    'intraday_drip_stone_20d': -1,
    'intraday_peak_ridge_ratio_20d': -1,
    # 前向搜索增量 6
    'intraday_oi_time_centroid_20d': -1,
    'intraday_wash_trade_20d': -1,
    'intraday_price_delay_20d': -1,
    'intraday_amihud_trend_20d': +1,
    'intraday_amihud_vol_ratio_20d': +1,
    'intraday_volume_time_shape_20d': +1,
}

# ============ 品种池 38 ============
UNIVERSE = [
    "A", "AG", "AL", "AU", "CU", "FU", "HC", "I", "IC", "IF", "IH", "J", "JM",
    "M", "MA", "NI", "P", "RB", "RM", "RU", "SA", "SN", "SR", "T", "TA", "TL",
    "TS", "Y", "ZN",
    "IM", "TF", "CF", "OI", "LH", "JD", "SC", "V", "UR",
]

# ============ 板块映射 (cap=3 配额) ============
SECTOR_MAP = {
    "有色": ["CU", "AL", "ZN", "NI", "SN", "AG", "AU"],
    "黑色": ["RB", "HC", "I", "J", "JM"],
    "能化": ["FU", "MA", "RU", "SA", "TA", "SC", "V", "UR"],
    "农产品": ["A", "M", "P", "RM", "Y", "SR", "CF", "OI", "LH", "JD"],
    "金融": ["IC", "IF", "IH", "T", "TL", "TS", "IM", "TF"],
}
SECTOR_CAP = 3

# ============ 组合参数 ============
WEIGHT_SCHEME = 'ic_ir'       # 池内 IC_IR (滚动60日 Ledoit-Wolf)
IC_IR_WINDOW = 60
COV_SHRINKAGE = 0.30
REBALANCE = 'D'               # 日度
LEVERAGE = 2.0                # 多头 +1.0 / 空头 -1.0

# ============ 性能 (2026-08-05, 时序修正后) ============
# 全段(2025-01~2026-07): 夏普 3.80, 年化 31.9%, 回撤 -3.8%
# 2025: 夏普 4.11 (收益 24.7%)
# 2026: 夏普 3.44 (收益 17.5%)
# OOS(3/1-5/15): 2.22 / 实盘(5/16+): +4.19
# 负月: 1/18, 最差月 -1.02%
# 边际: 每个因子都有正贡献 (最弱 volume_time_shape Δ+0.01)
