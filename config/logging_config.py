"""Centralized Logging Configuration for Drishti Edge Wearable System.

Provides structured, level-aware logging across HAL, perception, cognitive,
and safety modules with standardized formatting, ISO timestamps, and configurable
log levels for field debugging on Linux and Raspberry Pi.
"""

import os
import sys
import logging
from typing import Optional

_LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_INITIALIZED = False


def setup_logging(level: Optional[int] = None) -> None:
    """Initialize root logging configuration for Drishti."""
    global _INITIALIZED
    if _INITIALIZED:
        return

    if level is None:
        env_level = os.environ.get("DRISHTI_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger("drishti")
    root_logger.setLevel(level)
    if not root_logger.handlers:
        root_logger.addHandler(handler)

    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """Retrieve a child logger under the 'drishti' namespace.
    
    Args:
        name: Module or subsystem name (e.g. 'hal.camera', 'perception.detector').
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    if not _INITIALIZED:
        setup_logging()

    if not name.startswith("drishti."):
        logger_name = f"drishti.{name}"
    else:
        logger_name = name

    return logging.getLogger(logger_name)
