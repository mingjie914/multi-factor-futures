"""震荡类因子 SPEC 定义.

Oscillator 类: RSI 强度、RSI 反转、随机区间位置
3 个 base × 8 种 transform × 3 种窗口 = 72 个因子
"""
from __future__ import annotations

# 通用参数模板 (借鉴 QuantSkills)
_PARAMS_5D = {"window": 5, "norm": 20, "lag": 1, "smooth": 3, "skip": 5, "fast": 3, "slow": 5}
_PARAMS_10D = {"window": 10, "norm": 20, "lag": 1, "smooth": 3, "skip": 5, "fast": 3, "slow": 5}
_PARAMS_20D = {"window": 20, "norm": 60, "lag": 5, "smooth": 5, "skip": 5, "fast": 5, "slow": 20}

# 8 种变换
_TRANSFORMS = ["z", "delta", "smooth", "rank", "vol_scaled", "stability", "confirm_volume", "compress"]

# base 因子与中文说明
_BASES = {
    "rsi": ("RSI强度", "RSI 相对强弱指标, 衡量价格动量"),
    "rsi_reversal": ("RSI反转", "RSI 反转信号, 用于均值回归"),
    "stoch": ("随机区间位置", "收盘价在近期高低区间的百分位"),
}


def _make_specs() -> list:
    """生成震荡类全部 SPEC."""
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
                    "category": "oscillator",
                    "description": desc,
                })
    return specs


SPECS = _make_specs()
