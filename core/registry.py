"""Global component registry for the multi-factor framework.

Provides a simple key-value store for registering and retrieving pluggable
components (factors, processing steps, risk models, etc.) by kind and name.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Type

_REGISTRIES: Dict[str, Dict[str, type]] = {}
"""Internal storage: {kind -> {name -> class}}."""


def register(kind: str, name: str) -> Any:
    """Decorator that registers a class under *kind* and *name*.

    Args:
        kind: Component category (e.g. 'factor', 'risk_model').
        name: Human-readable identifier for the class.

    Returns:
        The decorator function.
    """

    def decorator(cls: type) -> type:
        _REGISTRIES.setdefault(kind, {})[name] = cls
        return cls

    return decorator


def get(kind: str, name: str) -> type:
    """Retrieve a registered class by *kind* and *name*.

    Raises:
        KeyError: If *kind* or *name* has not been registered.
    """
    if kind not in _REGISTRIES or name not in _REGISTRIES[kind]:
        raise KeyError(f"'{name}' not registered in '{kind}'")
    return _REGISTRIES[kind][name]


def create(kind: str, name: str, **kwargs: Any) -> Any:
    """Instantiate a registered class with the given keyword arguments.

    Args:
        kind: Component category.
        name: Human-readable identifier.
        **kwargs: Arguments forwarded to the class constructor.

    Returns:
        An instance of the registered class.
    """
    return get(kind, name)(**kwargs)


def list_registered(
    kind: Optional[str] = None,
) -> Dict[str, Dict[str, type]]:
    """List all registered components, optionally filtered by *kind*.

    Args:
        kind: If provided, only return entries for this category.

    Returns:
        A dictionary mapping kind -> {name -> class}.
    """
    if kind is not None:
        return {kind: _REGISTRIES.get(kind, {})}
    return dict(_REGISTRIES)


def register_factor(name: str, category: str = "custom") -> Any:
    """Convenience decorator for registering a factor.

    Equivalent to ``register('factor', name)``.

    Args:
        name: Factor name.
        category: Factor category (default 'custom').

    Returns:
        The decorator function.
    """

    def decorator(cls: type) -> type:
        from core.factor_contract import bind_factor_contract

        bind_factor_contract(cls, name)
        return register("factor", name)(cls)

    return decorator
