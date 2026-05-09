"""Numerical stability utilities for long-horizon gradients.

Long unrolled simulation trajectories often suffer from exploding or vanishing
gradients.  This module provides NumPy-compatible utilities:

* :func:`clip_grad_norm` — clip a flat gradient vector by its L2 norm.
* :func:`gradient_checksum` — diagnostic scalar for monitoring gradient health.
* :func:`stable_log` — numerically stable log with a floor to avoid -inf.
* :func:`running_mean_std` — online Welford estimator for observation normalisation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "clip_grad_norm",
    "gradient_checksum",
    "stable_log",
    "RunningMeanStd",
]


def clip_grad_norm(
    grad: NDArray[np.float64],
    max_norm: float,
) -> tuple[NDArray[np.float64], float]:
    """Clip *grad* so its L2 norm does not exceed *max_norm*.

    Parameters
    ----------
    grad:
        Flat gradient vector.
    max_norm:
        Maximum L2 norm.  Pass ≤ 0 to disable clipping.

    Returns
    -------
    tuple[NDArray, float]
        ``(clipped_grad, grad_norm)``.
    """
    norm = float(np.linalg.norm(grad))
    if max_norm > 0 and norm > max_norm:
        grad = grad * (max_norm / (norm + 1e-8))
    return grad, norm


def gradient_checksum(grad: NDArray[np.float64]) -> float:
    """Return the L2 norm of *grad* as a monitoring scalar.

    A large gradient checksum (e.g. > 100×the typical value) is a useful
    early warning for gradient explosion.

    Parameters
    ----------
    grad:
        Flat gradient vector.

    Returns
    -------
    float
        L2 norm.
    """
    return float(np.linalg.norm(grad))


def stable_log(
    x: NDArray[np.float64],
    floor: float = 1e-8,
) -> NDArray[np.float64]:
    """Compute log(max(x, floor)) element-wise to avoid log(0) = -inf.

    Parameters
    ----------
    x:
        Input array.
    floor:
        Minimum value before taking the log.

    Returns
    -------
    NDArray[np.float64]
    """
    return np.log(np.maximum(np.asarray(x, dtype=np.float64), floor))


class RunningMeanStd:
    """Online Welford estimator for observation normalisation.

    Tracks the running mean and variance of a stream of vectors.

    Parameters
    ----------
    shape:
        Shape of each observation vector.
    clip:
        Maximum absolute value for normalised observations.
    """

    def __init__(self, shape: tuple[int, ...] = (1,), clip: float = 5.0) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count: float = 1e-4
        self.clip = clip

    def update(self, x: NDArray[np.float64]) -> None:
        """Update statistics with a new batch of observations.

        Parameters
        ----------
        x:
            Observations, shape ``(batch, *shape)`` or ``(*shape,)``.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == len(self.mean.shape):
            x = x[np.newaxis]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * batch_count / total
        self.var = (
            self.count * self.var
            + batch_count * batch_var
            + delta**2 * self.count * batch_count / total
        ) / total
        self.count = total

    def normalize(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Normalise *x* using running statistics.

        Parameters
        ----------
        x:
            Input, shape ``(*shape)``.

        Returns
        -------
        NDArray[np.float64]
            Clipped normalised output.
        """
        x = np.asarray(x, dtype=np.float64)
        normed = (x - self.mean) / (np.sqrt(self.var) + 1e-8)
        return np.clip(normed, -self.clip, self.clip)
