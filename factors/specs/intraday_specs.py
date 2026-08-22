"""日内分钟级因子 SPEC 定义 (15分钟频率).

本模块定义 frequency="15min" 的 SPEC 因子, 使用主框架已配置本地分钟数据源计算.
与日度 SPEC (factors/specs/*.py) 的区别:
- frequency 字段为 "15min"
- slug 后缀用 "p" (period) 而非 "d" (day), 避免与日度因子混淆
- window 参数语义为"15分钟bar数" (1天=16个bar)

窗口选择 (基于国内期货日盘 ~4小时 = 16个15min bar):
- 16p:  1个交易日 (16个bar)
- 80p:  5个交易日 (80个bar)
- 240p: 15个交易日 (~3周, 240个bar)

base 因子 (复用现有日度 base, 在分钟级数据上计算):
- return:          bar级动量 (分钟收益率)
- realized_vol:    bar级波动率 (分钟收益率std)
- breakout:        微观突破 (分钟级上轨突破)
- efficiency:      微观Kaufman趋势效率
- rsi:             微观RSI相对强弱
- volume_ratio:    微观量比 (分钟成交量/滚动均量)
- ts_rank_close:   微观时序排名 (分钟收盘价百分位)
- sma_gap:         微观SMA偏离 (分钟级均线偏离)

8 个 base × 8 种 transform × 3 种窗口 = 192 个因子

数据依赖:
- 需要已发布的本地15分钟Parquet
- 发现扫描中缺失数据返回 NaN；正式选中因子缺失依赖时失败关闭
- 计算后按日重采样 (取每日最后一个bar的值), 输出日度DataFrame

集成方式:
- SpecFactor.compute 根据 frequency 字段路由: "15min" 走分钟数据路径
- 分钟数据通过 DataManager/FrequencyDataProvider 获取
- 计算结果按日聚合后, 与日度因子无缝共存
"""
from __future__ import annotations

# 15min 频率的参数模板 (window 为 15分钟 bar 数)
# norm/lag/smooth 等参数也按 bar 数设置
_PARAMS_16P = {
    "window": 16, "norm": 80, "lag": 4, "smooth": 8,
    "skip": 16, "fast": 4, "slow": 16,
}
_PARAMS_80P = {
    "window": 80, "norm": 240, "lag": 16, "smooth": 24,
    "skip": 80, "fast": 16, "slow": 80,
}
_PARAMS_240P = {
    "window": 240, "norm": 480, "lag": 48, "smooth": 72,
    "skip": 240, "fast": 48, "slow": 240,
}

# 8 种变换 (与现有 spec 文件一致)
_TRANSFORMS = [
    "z", "delta", "smooth", "rank",
    "vol_scaled", "stability", "confirm_volume", "compress",
]

# 8 个 base 与中文说明 (复用现有日度 base, 在分钟级数据上计算)
_BASES = {
    "return":         ("bar级动量",      "分钟收益率滚动均值, 微观动量信号"),
    "realized_vol":   ("bar级波动率",    "分钟收益率滚动std, 微观波动率"),
    "breakout":       ("微观突破",       "分钟收盘相对前期高点的突破幅度"),
    "efficiency":     ("微观趋势效率",   "Kaufman效率比 (分钟级), 净位移/路径总和"),
    "rsi":            ("微观RSI",        "分钟级RSI相对强弱指标"),
    "volume_ratio":   ("微观量比",       "分钟成交量/滚动均量, 微观量能"),
    "ts_rank_close":  ("微观时序排名",   "分钟收盘价时序百分位排名"),
    "sma_gap":        ("微观SMA偏离",    "分钟收盘价相对简单均线偏离率"),
}

# 频率标识: 所有因子均为 15min 频率
_FREQUENCY = "15min"


def _make_specs() -> list:
    """生成日内分钟级全部 SPEC."""
    specs = []
    for base, (name_cn, desc) in _BASES.items():
        for window_label, params in [
            ("16p", _PARAMS_16P),
            ("80p", _PARAMS_80P),
            ("240p", _PARAMS_240P),
        ]:
            for transform in _TRANSFORMS:
                slug = f"{base}_{window_label}_{transform}"
                specs.append({
                    "slug": slug,
                    "name_cn": f"{window_label}{name_cn}_{transform}",
                    "base": base,
                    "transform": transform,
                    "params": params.copy(),
                    "category": "intraday_specs",
                    "frequency": _FREQUENCY,
                    "description": desc,
                })
    return specs


SPECS = _make_specs()
