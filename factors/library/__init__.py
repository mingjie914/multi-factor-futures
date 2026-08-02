from __future__ import annotations

# === Hand-authored factor library ===
# 组织结构:
#   intraday.py                   自创日内高频因子 (224个, intraday_advanced)
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

try:
    _registered = register_all_spec_factors()
    import logging
    logging.getLogger("multi_factor").info(
        f"已注册 {len(_registered)} 个 SPEC 因子 "
        f"(日度: oscillator/volatility/pattern/drawdown/technicals/intraday_proxy/directional/volume_stat; "
        f"分钟级: intraday_specs)"
    )
except Exception as e:
    import logging
    logging.getLogger("multi_factor").warning(f"SPEC 因子注册失败: {e}")

# Load user-authored factors after built-ins so duplicate names fail closed.
from factors import user as _user_factors  # noqa: E402,F401
