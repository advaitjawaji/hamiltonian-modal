"""Benchmark metrics for DriftBench-G1.

Three scalar metrics summarise each :class:`~benchmarks.driftbench_g1.tasks.TaskResult`:

* :func:`relative_energy_drift` — max |ΔH| / |H₀|.
* :func:`mean_drift_rate` — slope of energy error over time (linear fit).
* :func:`drift_summary` — all three metrics in a dict.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "relative_energy_drift",
    "mean_drift_rate",
    "drift_summary",
]


def relative_energy_drift(
    energies: NDArray[np.float64],
    H0: float,
) -> float:
    """Compute the maximum relative energy drift over a trajectory.

    .. math::

        \\delta_H = \\max_t \\frac{|H(t) - H_0|}{|H_0|}

    Parameters
    ----------
    energies:
        Hamiltonian values at each step, shape ``(T+1,)``.
    H0:
        Initial Hamiltonian value.

    Returns
    -------
    float
        Maximum relative energy drift.
    """
    if abs(H0) < 1e-30:
        raise ValueError("H0 is too small to compute relative drift.")
    return float(np.max(np.abs(energies - H0)) / abs(H0))


def mean_drift_rate(
    energies: NDArray[np.float64],
    H0: float,
    dt: float,
) -> float:
    """Estimate the mean rate of energy drift via linear regression.

    Fits |H(t) - H₀| / |H₀| vs t and returns the slope [s⁻¹].

    Parameters
    ----------
    energies:
        Hamiltonian values, shape ``(T+1,)``.
    H0:
        Initial Hamiltonian value.
    dt:
        Integration timestep in seconds.

    Returns
    -------
    float
        Linear drift rate in [relative drift / second].
    """
    n = len(energies)
    t = np.arange(n, dtype=np.float64) * dt
    rel_errors = np.abs(energies - H0) / (abs(H0) + 1e-30)
    # Simple least-squares slope
    t_mean = t.mean()
    e_mean = rel_errors.mean()
    slope = float(np.sum((t - t_mean) * (rel_errors - e_mean)) / (np.sum((t - t_mean) ** 2) + 1e-30))
    return slope


def drift_summary(
    energies: NDArray[np.float64],
    H0: float,
    dt: float,
) -> dict[str, float]:
    """Return all drift metrics as a dictionary.

    Parameters
    ----------
    energies:
        Hamiltonian values, shape ``(T+1,)``.
    H0:
        Initial Hamiltonian value.
    dt:
        Integration timestep in seconds.

    Returns
    -------
    dict[str, float]
        Keys: ``"relative_drift"``, ``"drift_rate"``, ``"final_drift"``.
    """
    rel_drift = relative_energy_drift(energies, H0)
    rate = mean_drift_rate(energies, H0, dt)
    final_drift = float(abs(energies[-1] - H0) / (abs(H0) + 1e-30))
    return {
        "relative_drift": rel_drift,
        "drift_rate": rate,
        "final_drift": final_drift,
    }
