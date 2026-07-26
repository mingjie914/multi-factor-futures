"""Logging utilities for the multi-factor framework."""
from __future__ import annotations

import logging
import sys
from typing import Optional

_LOGGERS: dict = {}
"""Global cache of logger instances."""

_DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "multi_factor",
    level: int = logging.INFO,
    fmt: Optional[str] = None,
    datefmt: Optional[str] = None,
) -> logging.Logger:
    """Configure and return a logger with the given *name* and *level*.

    If the logger already exists it is returned as-is (singleton pattern).

    Args:
        name: Logger name (default 'multi_factor').
        level: Logging level (default ``logging.INFO``).
        fmt: Optional log message format string.
        datefmt: Optional date format string.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt or _DEFAULT_FORMAT,
            datefmt=datefmt or _DEFAULT_DATE_FORMAT,
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    _LOGGERS[name] = logger
    return logger


def get_logger(name: str) -> logging.Logger:
    """Retrieve an existing logger by *name*.

    If the logger has not been initialised via :func:`setup_logger`, a default
    one will be created.

    Args:
        name: Logger name.

    Returns:
        A :class:`logging.Logger` instance.
    """
    if name not in _LOGGERS:
        return setup_logger(name)
    return _LOGGERS[name]
