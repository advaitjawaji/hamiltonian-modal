"""Benchmark metrics for DiffSim-EffBench.

Three metrics characterise differentiable-simulation gradient efficiency:

* :func:`gradient_snr` — signal-to-noise ratio of the gradient estimate.
* :func:`compute_efficiency_score` — steps-per-second normalised by DOF.
* :func:`eff_summary` — all metrics in a dict.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "gradient_snr",
    "compute_efficiency_score",
    "eff_summary",
]


def gradient_snr(grad_norm: float, grad_variance: float) -> float:
    """Gradient signal-to-noise ratio.

    .. math::

        \\text{SNR} = \\frac{\\|\\bar{g}\\|_2^2}{\\text{Var}(g) + \\epsilon}

    A higher SNR indicates a more reliable gradient estimate.

    Parameters
    ----------
    grad_norm:
        L2 norm of the mean gradient estimate.
    grad_variance:
        Element-wise variance of the gradient, averaged over dimensions.

    Returns
    -------
    float
        Gradient SNR.
    """
    return grad_norm**2 / (grad_variance + 1e-12)


def compute_efficiency_score(
    steps_per_second: float,
    n_dof: int,
) -> float:
    """Normalised simulation throughput.

    Reports ``steps_per_second / n_dof`` so that systems with different DOF
    counts can be fairly compared on a single axis.

    Parameters
    ----------
    steps_per_second:
        Raw simulation throughput.
    n_dof:
        Degrees of freedom.

    Returns
    -------
    float
        Normalised efficiency score.
    """
    return steps_per_second / max(n_dof, 1)


def eff_summary(
    mean_return: float,
    grad_norm: float,
    grad_variance: float,
    steps_per_second: float,
    n_dof: int,
) -> dict[str, float]:
    """Return all efficiency metrics as a dictionary.

    Parameters
    ----------
    mean_return:
        Mean episode return.
    grad_norm:
        L2 norm of mean gradient.
    grad_variance:
        Variance of gradient estimate.
    steps_per_second:
        Simulation throughput.
    n_dof:
        Degrees of freedom.

    Returns
    -------
    dict[str, float]
        Keys: ``"mean_return"``, ``"grad_norm"``, ``"grad_variance"``,
        ``"gradient_snr"``, ``"efficiency_score"``, ``"steps_per_second"``.
    """
    return {
        "mean_return": mean_return,
        "grad_norm": grad_norm,
        "grad_variance": grad_variance,
        "gradient_snr": gradient_snr(grad_norm, grad_variance),
        "efficiency_score": compute_efficiency_score(steps_per_second, n_dof),
        "steps_per_second": steps_per_second,
    }
