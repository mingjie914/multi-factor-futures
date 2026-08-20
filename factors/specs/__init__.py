"""Aggregate built-in, intraday and practical candidate SPEC definitions.

Counts are validated from the registry at runtime. Practical candidates are
kept in a separate module and carry research-tier/source metadata so a large
enumeration cannot be mistaken for production approval.
"""
from __future__ import annotations

from collections import Counter

from factors.specs.oscillator import SPECS as _OSC_SPECS
from factors.specs.volatility import SPECS as _VOL_SPECS
from factors.specs.pattern import SPECS as _PAT_SPECS
from factors.specs.drawdown import SPECS as _DD_SPECS
from factors.specs.technicals import SPECS as _TECH_SPECS
from factors.specs.intraday_proxy import SPECS as _IPX_SPECS
from factors.specs.directional import SPECS as _DIR_SPECS
from factors.specs.volume_stat import SPECS as _VSTAT_SPECS
from factors.specs.intraday_specs import SPECS as _ISPEC_SPECS
from factors.specs.positioning_participation import SPECS as _POSITIONING_SPECS
from factors.specs.practical import SPECS as _PRACTICAL_SPECS

# 聚合全部 SPEC (日度 + 分钟级)
ALL_SPECS = (
    _OSC_SPECS + _VOL_SPECS + _PAT_SPECS + _DD_SPECS
    + _TECH_SPECS + _IPX_SPECS + _DIR_SPECS + _VSTAT_SPECS
    + _ISPEC_SPECS + _POSITIONING_SPECS
    + _PRACTICAL_SPECS
)

# 按类别分组
SPECS_BY_CATEGORY = {
    "oscillator": _OSC_SPECS,
    "volatility": _VOL_SPECS,
    "pattern": _PAT_SPECS,
    "drawdown": _DD_SPECS,
    "technicals": _TECH_SPECS,
    "intraday_proxy": _IPX_SPECS,
    "directional": _DIR_SPECS,
    "volume_stat": _VSTAT_SPECS,
    "intraday_specs": _ISPEC_SPECS,
    "positioning_participation": _POSITIONING_SPECS,
    "practical": _PRACTICAL_SPECS,
}

# 全部因子 slug 列表
ALL_SLUGS = [s["slug"] for s in ALL_SPECS]
_DUPLICATE_SLUGS = sorted(
    slug for slug, count in Counter(ALL_SLUGS).items() if count > 1
)
if _DUPLICATE_SLUGS:
    raise ValueError(f"duplicate SPEC factor slugs: {_DUPLICATE_SLUGS[:10]}")
SPEC_BY_SLUG = {spec["slug"]: spec for spec in ALL_SPECS}


def register_all_spec_factors() -> list:
    """注册全部 SPEC 因子到全局 registry.

    Returns:
        已注册的因子 slug 列表.
    """
    from factors.spec_factor import register_spec_factor

    registered = []
    for spec in ALL_SPECS:
        register_spec_factor(spec)
        registered.append(spec["slug"])
    return registered


def get_specs_by_category(category: str) -> list:
    """获取指定类别的 SPEC 列表."""
    return SPECS_BY_CATEGORY.get(category, [])


def get_specs_by_base(base: str) -> list:
    """获取指定 base 的 SPEC 列表."""
    return [s for s in ALL_SPECS if s["base"] == base]
