"""Random seeding utilities.

Provides a single :func:`seed_all` entry point that consistently seeds Python's
built-in :mod:`random` module and NumPy's global random state, plus a
:func:`make_rng` factory that returns a reproducible
:class:`numpy.random.Generator` for local use.
"""

from __future__ import annotations

import random

import numpy as np

__all__ = [
    "seed_all",
    "make_rng",
]


def seed_all(seed: int) -> None:
    """Seed Python :mod:`random` and NumPy's global RNG with *seed*.

    Parameters
    ----------
    seed:
        Non-negative integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)


def make_rng(seed: int) -> np.random.Generator:
    """Return a fresh :class:`numpy.random.Generator` seeded from *seed*.

    Prefer this over the global NumPy RNG for reproducible experiments
    because the generator state is isolated per call-site.

    Parameters
    ----------
    seed:
        Non-negative integer seed value.

    Returns
    -------
    numpy.random.Generator
        A PCG64-backed generator seeded deterministically.
    """
    return np.random.default_rng(seed)
