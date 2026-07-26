"""Auto-discovery and guarded registration for user-authored factors."""
from __future__ import annotations

import importlib
import pkgutil

from core.registry import list_registered, register_factor


def register_user_factor(name: str, category: str = "custom"):
    """Register a user factor without allowing it to replace an existing factor."""
    existing = list_registered("factor").get("factor", {})
    if name in existing:
        raise ValueError(f"user factor name already registered: {name}")
    return register_factor(name, category=category)


def load_user_factors() -> tuple[str, ...]:
    """Import public modules in this package in deterministic name order."""
    module_names = sorted(
        module.name
        for module in pkgutil.iter_modules(__path__)
        if not module.name.startswith("_")
    )
    for module_name in module_names:
        importlib.import_module(f"{__name__}.{module_name}")
    return tuple(module_names)


LOADED_USER_FACTOR_MODULES = load_user_factors()


__all__ = [
    "LOADED_USER_FACTOR_MODULES",
    "load_user_factors",
    "register_user_factor",
]
