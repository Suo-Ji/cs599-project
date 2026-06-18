"""Centralized logging configuration.

Configures a single named logger used across the project. Optional rich-based
console formatting is enabled when configured. All log messages use objective,
deterministic wording — no figurative or anthropomorphic language.
"""

from __future__ import annotations

import logging
from typing import Optional

from .config import AppConfig, get_config

_LOGGER_NAME = "rag"


def setup_logging(config: Optional[AppConfig] = None, level: Optional[str] = None) -> logging.Logger:
    """Initialize and return the project-wide logger.

    Idempotent: clears existing handlers so repeated calls do not duplicate output.
    """
    cfg = config or get_config()
    effective_level = (level or cfg.logging.level).upper()

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(effective_level)
    logger.handlers.clear()
    logger.propagate = False

    if cfg.logging.rich_console:
        try:
            from rich.logging import RichHandler

            handler: logging.Handler = RichHandler(
                show_time=True, show_path=False, rich_tracebacks=True, markup=False
            )
            fmt = "%(message)s"
        except ImportError:  # pragma: no cover - rich is a soft dependency at runtime
            handler = logging.StreamHandler()
            fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    else:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """Return a child logger under the project namespace."""
    return logging.getLogger(name) if name == _LOGGER_NAME else logging.getLogger(f"{_LOGGER_NAME}.{name}")
