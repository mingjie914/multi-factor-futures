"""形态类因子 SPEC 定义.

Pattern 类: 缺口累积、日内收益、上影线压力、下影线支撑
4 个 base × 8 种 transform × 3 种窗口 = 96 个因子
"""
from __future__ import annotations

_PARAMS_5D = {"window": 5, "norm": 20, "lag": 1, "smooth": 3, "skip": 5, "fast": 3, "slow": 5}
_PARAMS_10D = {"window": 10, "norm": 20, "lag": 1, "smooth": 3, "skip": 5, "fast": 3, "slow": 5}
_PARAMS_20D = {"window": 20, "norm": 60, "lag": 5, "smooth": 5, "skip": 5, "fast": 5, "slow": 20}

_TRANSFORMS = ["z", "delta", "smooth", "rank", "vol_scaled", "stability", "confirm_volume", "compress"]

_BASES = {
    "gap_sum": ("缺口累积", "开盘缺口累积和, 衡量隔夜跳空"),
    "intraday": ("日内收益", "收盘减开盘收益的滚动均值"),
    "upper_wick": ("上影线压力", "上影线占比的滚动均值, 衡量上方压力"),
    "lower_wick": ("下影线支撑", "下影线占比的滚动均值, 衡量下方支撑"),
}


def _make_specs() -> list:
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
                    "category": "pattern",
                    "description": desc,
                })
    return specs


SPECS = _make_specs()
