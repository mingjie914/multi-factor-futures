"""Positioning and participation factor SPEC definitions.

The low-churn trend signal combines price momentum with the inverse square
root of volume/open-interest turnover. Lower churn indicates that a price
move is carried by positions that remain open instead of rapid recycling.
"""
from __future__ import annotations


_PARAMS = {
    "5d": {"window": 5, "norm": 20, "lag": 1, "smooth": 3},
    "10d": {"window": 10, "norm": 20, "lag": 1, "smooth": 3},
    "20d": {"window": 20, "norm": 60, "lag": 5, "smooth": 5},
}

_TRANSFORMS = [
    "z",
    "delta",
    "smooth",
    "rank",
    "vol_scaled",
    "stability",
    "confirm_volume",
    "compress",
]


def _make_specs() -> list[dict]:
    specs = []
    for window_label, params in _PARAMS.items():
        for transform in _TRANSFORMS:
            specs.append(
                {
                    "slug": f"low_churn_trend_{window_label}_{transform}",
                    "name_cn": f"{window_label}低换手趋势_{transform}",
                    "base": "low_churn_trend",
                    "transform": transform,
                    "params": params.copy(),
                    "category": "positioning_participation",
                    "dependencies": [
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "oi",
                    ],
                    "description": (
                        "价格趋势除以成交量/持仓量换手率的平方根；低换手下的趋势"
                        "预期更具有持续性，所有输入滞后一根日线"
                    ),
                }
            )
    return specs


SPECS = _make_specs()
