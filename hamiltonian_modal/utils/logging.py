"""Logging and experiment-tracking utilities.

Provides:

* :func:`get_logger` — thin wrapper around ``logging.getLogger`` that
  sets a sensible default format when no handlers are yet configured.
* :func:`init_wandb` — initialises Weights & Biases if the ``wandb``
  package is installed; gracefully falls back to a no-op ``dict``-like
  object when it is not.

Examples
--------
>>> from hamiltonian_modal.utils.logging import get_logger, init_wandb
>>> log = get_logger(__name__)
>>> log.info("hello")
"""

from __future__ import annotations

import logging
import sys
from typing import Any

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a named :class:`logging.Logger` with sensible defaults.

    If the root logger has no handlers configured (i.e. the caller is not
    inside a larger application that already set up logging), a
    ``StreamHandler`` writing to *stderr* is attached to the root logger
    so that ``INFO``-level messages are visible by default.

    Parameters
    ----------
    name : str
        Logger name.  Pass ``__name__`` from the calling module to get a
        hierarchical logger (e.g. ``"hamiltonian_modal.modal.stiffness"``).

    Returns
    -------
    logger : logging.Logger
        Configured logger instance.

    Notes
    -----
    The function is idempotent: calling it multiple times with the same
    ``name`` returns the same logger object (standard Python behaviour).
    The handler is added at most once per process.
    """
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(handler)
        root.setLevel(logging.INFO)

    return logging.getLogger(name)


class _NoOpRun:
    """Minimal no-op replacement for a ``wandb.run`` object.

    Used when ``wandb`` is not installed so that calling code does not
    need to guard every ``run.log(...)`` call.
    """

    def log(self, data: dict, **kwargs: Any) -> None:  # noqa: ANN401
        """Accept and discard metrics silently."""

    def finish(self, **kwargs: Any) -> None:  # noqa: ANN401
        """No-op finish."""

    def __bool__(self) -> bool:
        return False


def init_wandb(
    project: str,
    config: dict,
    run_name: str | None = None,
) -> Any:
    """Initialise a Weights & Biases run, falling back gracefully.

    Parameters
    ----------
    project : str
        W&B project name.
    config : dict
        Hyper-parameter dictionary logged as the run's configuration.
    run_name : str or None, optional
        Human-readable run name shown in the W&B UI.  If ``None``, W&B
        auto-generates a name.

    Returns
    -------
    run : wandb.Run or _NoOpRun
        The active W&B run object if ``wandb`` is installed and
        ``WANDB_MODE`` is not ``"disabled"``, otherwise a :class:`_NoOpRun`
        that silently discards all calls.

    Notes
    -----
    If ``wandb`` raises any exception during initialisation (e.g. network
    error, missing API key), a warning is logged and a :class:`_NoOpRun`
    is returned so training can proceed uninterrupted.
    """
    logger = get_logger(__name__)

    try:
        import wandb  # type: ignore[import-not-found]
    except ImportError:
        logger.warning(
            "wandb not installed; experiment tracking disabled. "
            "Install with: pip install wandb"
        )
        return _NoOpRun()

    try:
        run = wandb.init(
            project=project,
            name=run_name,
            config=config,
            reinit=True,
        )
        logger.info("W&B run initialised: %s/%s", project, run_name or "(auto)")
        return run
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("wandb.init failed (%s); experiment tracking disabled.", exc)
        return _NoOpRun()
