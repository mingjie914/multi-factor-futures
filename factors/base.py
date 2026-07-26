from __future__ import annotations
from core.interfaces import Factor
from core.registry import register_factor as _register

# Re-export for user convenience
Factor = Factor
register_factor = _register


def factor_info(f: Factor) -> dict:
    """返回因子元信息."""
    return {
        "name": f.name,
        "category": f.category,
        "frequency": f.frequency,
        "description": f.description,
    }
