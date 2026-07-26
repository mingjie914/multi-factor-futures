"""波动率类因子 SPEC 定义.

Volatility 类: ATR 占比、振幅占比、实现波动率、下行波动率
4 个 base × 8 种 transform × 3 种窗口 = 96 个因子
"""
from __future__ import annotations

_PARAMS_5D = {"window": 5, "norm": 20, "lag": 1, "smooth": 3, "skip": 5, "fast": 3, "slow": 5}
_PARAMS_10D = {"window": 10, "norm": 20, "lag": 1, "smooth": 3, "skip": 5, "fast": 3, "slow": 5}
_PARAMS_20D = {"window": 20, "norm": 60, "lag": 5, "smooth": 5, "skip": 5, "fast": 5, "slow": 20}

_TRANSFORMS = ["z", "delta", "smooth", "rank", "vol_scaled", "stability", "confirm_volume", "compress"]

_BASES = {
    "atr_ratio": ("ATR占比", "真实波幅占收盘价比例, 衡量波动幅度"),
    "range_ratio": ("振幅占比", "高低差占收盘价比例的均值"),
    "realized_vol": ("实现波动率", "收益率滚动标准差"),
    "downside_vol": ("下行波动率", "仅负收益的滚动标准差"),
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
                    "category": "volatility",
                    "description": desc,
                })
    return specs


SPECS = _make_specs()
