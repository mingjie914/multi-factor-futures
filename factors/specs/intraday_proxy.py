"""日内代理特征因子 SPEC 定义.

从日度 OHLC 提取日内行为信息, 作为日度因子的确认信号.
由于商品期货无分钟数据, 用日度 OHLC 推导日内代理特征.

6 个 base × 8 种 transform × 3 种窗口 = 144 个因子
全部因子均遵循 SPEC 规范 (base + transform + window).
"""
from __future__ import annotations

_PARAMS_5D = {"window": 5, "norm": 20, "lag": 1, "smooth": 3, "skip": 5, "fast": 3, "slow": 5}
_PARAMS_10D = {"window": 10, "norm": 20, "lag": 1, "smooth": 3, "skip": 5, "fast": 3, "slow": 5}
_PARAMS_20D = {"window": 20, "norm": 60, "lag": 5, "smooth": 5, "skip": 5, "fast": 5, "slow": 20}

_TRANSFORMS = ["z", "delta", "smooth", "rank", "vol_scaled", "stability", "confirm_volume", "compress"]

_BASES = {
    "intraday_strength": (
        "日内强度",
        "收盘相对开盘的强度占当日振幅比例, >0.5强势收盘",
    ),
    "close_position": (
        "收盘位置",
        "收盘在当日high-low区间中的相对位置, 接近1=强势",
    ),
    "body_ratio": (
        "实体占比",
        "K线实体占当日振幅的比例, 高=趋势日",
    ),
    "upper_wick_ratio": (
        "上影线占比",
        "上影线占当日振幅的比例, 高=上方抛压",
    ),
    "lower_wick_ratio": (
        "下影线占比",
        "下影线占当日振幅的比例, 高=下方承接",
    ),
    "overnight_intraday_split": (
        "隔夜日内分解",
        "日内收益占总收益比例, 高=信息日内释放(趋势可持续)",
    ),
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
                    "category": "intraday_proxy",
                    "description": desc,
                })
    return specs


SPECS = _make_specs()
