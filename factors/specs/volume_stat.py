"""量价统计类因子 SPEC 定义 (借鉴 QuantSkills volume-stat-alpha 库).

Volume Stat 类: 量能/流动性/量价/排序/分布
8 个 base × 8 种 transform × 3 种窗口 = 192 个因子

注意: obv_slope 已在 technicals.py 中实现, 此处不重复定义.

base 分组:
- 量能类:   volume_ratio, volume_z
- 流动性类: dollar_volume
- 量价类:   price_volume_corr
- 排序类:   ts_rank_close, ts_rank_volume
- 分布类:   ret_skew, ret_kurt
"""
from __future__ import annotations

# 通用参数模板 (与现有 spec 文件一致)
_PARAMS_5D = {"window": 5, "norm": 20, "lag": 1, "smooth": 3, "skip": 5, "fast": 3, "slow": 5}
_PARAMS_10D = {"window": 10, "norm": 20, "lag": 1, "smooth": 3, "skip": 5, "fast": 3, "slow": 5}
_PARAMS_20D = {"window": 20, "norm": 60, "lag": 5, "smooth": 5, "skip": 5, "fast": 5, "slow": 20}

# 8 种变换 (与现有 spec 文件一致)
_TRANSFORMS = ["z", "delta", "smooth", "rank", "vol_scaled", "stability", "confirm_volume", "compress"]

# 8 个量价统计类 base 与中文说明 (obv_slope 已在 technicals.py 中定义, 不重复)
_BASES = {
    # 量能类
    "volume_ratio":      ("量比",       "当前成交量相对滚动均量偏离"),
    "volume_z":          ("量能Z值",    "成交量滚动 z-score, 衡量量能异常"),
    # 流动性类
    "dollar_volume":     ("成交额比",   "成交额相对滚动均值偏离, 衡量流动性"),
    # 量价类
    "price_volume_corr": ("量价相关",   "收益率与成交量变化率的滚动相关性"),
    # 排序类
    "ts_rank_close":     ("收盘时序排名", "收盘价时序百分位排名, 居中到0"),
    "ts_rank_volume":    ("量时序排名", "成交量时序百分位排名, 居中到0"),
    # 分布类
    "ret_skew":          ("收益偏度",   "收益率滚动偏度, 负偏度有 crash risk premium"),
    "ret_kurt":          ("收益峰度",   "收益率滚动峰度, 高峰度有尾部风险溢价"),
}


def _make_specs() -> list:
    """生成量价统计类全部 SPEC."""
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
                    "category": "volume_stat",
                    "description": desc,
                })
    return specs


SPECS = _make_specs()
