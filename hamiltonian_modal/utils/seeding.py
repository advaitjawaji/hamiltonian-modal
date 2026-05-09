"""Global random-seed utilities.

Provides a single entry-point :func:`set_global_seed` that synchronises
seeds across numpy, Python's ``random`` module, and optionally PyTorch
and JAX when those libraries are present.

Examples
--------
>>> from hamiltonian_modal.utils.seeding import set_global_seed, make_rng_key
>>> set_global_seed(0)
>>> key = make_rng_key(0)
"""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """Set the global random seed for all available RNG sources.

    Sets seeds for:

    * Python's built-in ``random`` module
    * NumPy's global RNG (``numpy.random.seed``)
    * PyTorch CPU and CUDA RNGs (if ``torch`` is importable)
    * JAX (via ``jax.random`` — JAX is functional so this is a no-op
      beyond logging, but we log the seed for reproducibility records)

    Parameters
    ----------
    seed : int
        Non-negative integer seed value.  Values outside [0, 2**32 - 1]
        are accepted by NumPy but may be truncated by other libraries.

    Notes
    -----
    JAX uses explicit key-based randomness rather than a global state.
    Use :func:`make_rng_key` to obtain a JAX PRNG key seeded from
    ``seed`` and thread it through JAX computations explicitly.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")

    random.seed(seed)
    np.random.seed(seed)
    logger.debug("Set numpy and Python random seed to %d", seed)

    # PyTorch (optional)
    try:
        import torch  # type: ignore[import-not-found]

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.debug("Set torch seed to %d", seed)
    except ImportError:
        logger.debug("torch not available; skipping torch seed")

    # JAX (optional — just log, since JAX uses functional RNG)
    try:
        import jax  # type: ignore[import-not-found]  # noqa: F401

        logger.debug(
            "JAX detected; use make_rng_key(%d) for a seeded JAX PRNGKey", seed
        )
    except ImportError:
        logger.debug("JAX not available; skipping JAX seed log")


def make_rng_key(seed: int) -> Any:
    """Return a seeded RNG key suitable for use with JAX or NumPy.

    If JAX is available, returns a ``jax.random.PRNGKey`` seeded with
    ``seed``.  Otherwise returns a ``numpy.random.Generator`` instance
    (``numpy.random.default_rng(seed)``).

    Parameters
    ----------
    seed : int
        Non-negative integer seed.

    Returns
    -------
    key : jax.Array or numpy.random.Generator
        A JAX PRNGKey (shape ``(2,)``, dtype ``uint32``) if JAX is
        installed, otherwise a ``numpy.random.Generator``.

    Examples
    --------
    >>> key = make_rng_key(42)
    >>> # With JAX: key is a jax.Array
    >>> # Without JAX: key is numpy.random.Generator
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")

    try:
        import jax.random as jrandom  # type: ignore[import-not-found]

        key = jrandom.PRNGKey(seed)
        logger.debug("Created JAX PRNGKey with seed %d", seed)
        return key
    except ImportError:
        rng = np.random.default_rng(seed)
        logger.debug("JAX unavailable; returning numpy.random.Generator with seed %d", seed)
        return rng
