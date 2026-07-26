"""技术指标类因子 SPEC 定义 (来自 DolphinDB alpha_db 推导).

Technicals 类: MACD、KDJ、BOLL、OBV、BIAS、DMI、VR、WR、TRIX、DPO、MAVOL、ARBR、ASI、BBI、NEWDMA
16 个 base × 8 种 transform × 3 种窗口 = 384 个因子

算法来源: DolphinDB alpha_db/daily_alpha_table 中的 19 类标准技术分析指标.
其中 RSI 已在 oscillator.py 中实现, MTM 与 momentum 类似, 此处覆盖其余 16 类.
全部仅依赖 OHLCV 数据, 保持与 SPEC 框架一致.
"""
from __future__ import annotations

# 通用参数模板 (与现有 spec 文件一致, fast/slow 用于 MACD/NEWDMA 等双均线指标)
_PARAMS_5D = {"window": 5, "norm": 20, "lag": 1, "smooth": 3, "skip": 5, "fast": 3, "slow": 5}
_PARAMS_10D = {"window": 10, "norm": 20, "lag": 1, "smooth": 3, "skip": 5, "fast": 3, "slow": 5}
_PARAMS_20D = {"window": 20, "norm": 60, "lag": 5, "smooth": 5, "skip": 5, "fast": 5, "slow": 20}

# 8 种变换 (与现有 spec 文件一致)
_TRANSFORMS = ["z", "delta", "smooth", "rank", "vol_scaled", "stability", "confirm_volume", "compress"]

# 16 个技术指标 base 与中文说明
_BASES = {
    "macd_diff":      ("MACD柱",      "MACD 柱状图 (DIF-DEA), 衡量动量变化"),
    "kdj_j":          ("KDJ_J值",     "KDJ 指标的 J 值, 超买超卖信号"),
    "boll_position":  ("布林位置",    "收盘价在布林带中的位置 [-1,+1]"),
    "boll_width":     ("布林宽度",    "布林带宽度, 衡量波动率压缩"),
    "obv_slope":      ("OBV斜率",     "能量潮 OBV 的滚动斜率"),
    "bias":           ("乖离率",      "收盘价相对均线偏离率"),
    "dmi_adx":        ("DMI_ADX",     "DMI 趋向指标 ADX, 衡量趋势强度"),
    "vr_ratio":       ("VR量比",      "上涨/下跌成交量比率"),
    "wr_ratio":       ("WR威廉",      "威廉指标, 超买超卖"),
    "trix":           ("TRIX",        "三重平滑收益率, 过滤噪声"),
    "dpo":            ("DPO去趋势",   "去趋势价格, 突出周期性"),
    "mavol_ratio":    ("MAVOL量比",   "当前量/均量, 衡量活跃度"),
    "arbr_sentiment": ("ARBR情绪",    "ARBR 情绪指标 AR 信号"),
    "asi_slope":      ("ASI斜率",     "累计摆动指标斜率"),
    "bbi_position":   ("BBI位置",     "多空指标位置, close 相对 BBI 偏离"),
    "newdma_trend":   ("NEWDMA趋势",  "动态均线趋势 (快慢均线差)"),
}


def _make_specs() -> list:
    """生成技术指标类全部 SPEC."""
    specs = []
    for base, (name_cn, desc) in _BASES.items():
        for window_label, params in [("5d", _PARAMS_5D), ("10d", _PARAMS_10D), ("20d", _PARAMS_20D)]:
            for transform in _TRANSFORMS:
                slug = f"{base}_{window_label}_{transform}"
                specs.append({
                    "slug": slug,
                    "name_cn": f"{window_label}{name_cn}_{transform}",
                    "base": base,
                    "transform": transform,
                    "params": params.copy(),
                    "category": "technicals",
                    "description": desc,
                })
    return specs


SPECS = _make_specs()
