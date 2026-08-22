from __future__ import annotations

# === Hand-authored factor library ===
# 组织结构:
#   intraday.py                   自创日内高频因子（运行时注册表是唯一数量口径）
#   technical_factors.py          日度技术面 (动量/反转/偏度/趋势/波动率)
#   term_structure_factors.py     期限结构与基差
#   volume_oi_factors.py          量价与持仓
#   cross_commodity.py            跨品种 (含 SECTOR_MAP, 被其他模块引用)
#   cross_frequency.py            跨频率
from . import (
    intraday,
    technical_factors,
    term_structure_factors,
    volume_oi_factors,
    cross_commodity,
    cross_frequency,
)

# === SPEC-driven factors; the runtime log is the authoritative count ===
from factors.specs import register_all_spec_factors

_registered = register_all_spec_factors()

# Load user-authored factors after built-ins so duplicate names fail closed.
from factors import user as _user_factors  # noqa: E402,F401
