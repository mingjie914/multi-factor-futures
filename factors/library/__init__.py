from __future__ import annotations

# === 原有实验性因子 (保留, 不动) ===
from . import (
    liquidity,
    momentum,
    reversal,
    term_structure,
    term_structure_slope,
    volatility,
    volume_oi,
    volume_oi_ratio,
    skewness,
    volume_price,
    oi_momentum,
    intraday_range,
    overnight_return,
    settle_close_basis,
    vstd_normalized,
    trend_strength,
    skewness_long,
    short_period,
    cross_commodity,
    basis_momentum,
    intraday,
    cross_frequency,
)

# === QuantSkills SPEC 驱动因子 (1488 个: 1296 日度 + 192 分钟级) ===
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
