"""Logging integration interfaces.

Thin wrappers around :mod:`logging` to give every sub-module a consistently
named logger and a one-call setup function for console / file output.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

__all__ = [
    "get_logger",
    "configure_logging",
]

_ROOT_LOGGER_NAME = "hamiltonian_modal"


def get_logger(name: str = _ROOT_LOGGER_NAME) -> logging.Logger:
    """Return a :class:`logging.Logger` scoped under ``hamiltonian_modal``.

    Parameters
    ----------
    name:
        Dotted logger name.  If it does not already start with
        ``"hamiltonian_modal"`` it is automatically prefixed so all
        package loggers form a single hierarchy.

    Returns
    -------
    logging.Logger
    """
    if not name.startswith(_ROOT_LOGGER_NAME):
        name = f"{_ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(name)


def configure_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    fmt: str = "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
) -> None:
    """Configure the root ``hamiltonian_modal`` logger.

    Sets up a :class:`logging.StreamHandler` to *stderr* and, optionally, a
    :class:`logging.FileHandler` when *log_file* is provided.  Safe to call
    multiple times — existing handlers are removed first.

    Parameters
    ----------
    level:
        Logging level (e.g. ``logging.DEBUG``, ``logging.INFO``).
    log_file:
        Optional path to a log file.  The parent directory must exist.
    fmt:
        Format string for :class:`logging.Formatter`.
    """
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.handlers.clear()
    logger.setLevel(level)

    formatter = logging.Formatter(fmt)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(str(log_file))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
