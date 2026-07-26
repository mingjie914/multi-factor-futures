"""方向类因子 SPEC 定义 (借鉴 QuantSkills directional-alpha 库).

Directional 类: 趋势/动量/突破/通道
12 个 base × 8 种 transform × 3 种窗口 = 288 个因子

base 分组:
- 趋势类: sma_gap, ema_gap, dual_ema_gap, sma_slope, trend_strength, efficiency
- 动量类: return, skip_return, reversal
- 突破类: breakout, breakdown
- 通道类: range_position
"""
from __future__ import annotations

# 通用参数模板 (与现有 spec 文件一致)
_PARAMS_5D = {"window": 5, "norm": 20, "lag": 1, "smooth": 3, "skip": 5, "fast": 3, "slow": 5}
_PARAMS_10D = {"window": 10, "norm": 20, "lag": 1, "smooth": 3, "skip": 5, "fast": 3, "slow": 5}
_PARAMS_20D = {"window": 20, "norm": 60, "lag": 5, "smooth": 5, "skip": 5, "fast": 5, "slow": 20}

# 8 种变换 (与现有 spec 文件一致)
_TRANSFORMS = ["z", "delta", "smooth", "rank", "vol_scaled", "stability", "confirm_volume", "compress"]

# 12 个方向类 base 与中文说明
_BASES = {
    # 趋势类
    "sma_gap":        ("SMA偏离",     "收盘价相对简单均线偏离率"),
    "ema_gap":        ("EMA偏离",     "收盘价相对指数均线偏离率"),
    "dual_ema_gap":   ("双EMA差",     "快慢EMA差值占收盘价比率"),
    "sma_slope":      ("SMA斜率",     "简单均线线性回归斜率, 衡量趋势方向"),
    "trend_strength": ("趋势强度",    "夏普式趋势强度: 收益/波动"),
    "efficiency":     ("趋势效率",    "Kaufman效率比: 净位移/路径总和"),
    # 动量类
    "return":         ("收益率",      "过去 w 日收益率"),
    "skip_return":    ("跳期动量",    "跳过 w//2 期的收益率, 规避短期反转"),
    "reversal":       ("反转",        "负收益率, 用于均值回归"),
    # 突破类
    "breakout":       ("上轨突破",    "收盘价相对前期高点的突破幅度"),
    "breakdown":      ("下轨跌破",    "收盘价相对前期低点的跌破幅度 (负值)"),
    # 通道类
    "range_position": ("通道位置",    "收盘价在高低通道中的位置, 居中到0"),
}


def _make_specs() -> list:
    """生成方向类全部 SPEC."""
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
                    "category": "directional",
                    "description": desc,
                })
    return specs


SPECS = _make_specs()
