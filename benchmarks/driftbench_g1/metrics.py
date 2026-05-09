"""Energy drift metrics for the DriftBench-G1 benchmark."""
import numpy as np
from numpy.typing import NDArray


def compute_energy_drift(energies: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute relative energy drift at each step: |E(t) - E(0)| / |E(0)|.

    Parameters
    ----------
    energies : shape (n_steps+1,) — energy values starting at t=0

    Returns
    -------
    drift : shape (n_steps+1,) — relative energy drift per step
    """
    energies = np.asarray(energies, dtype=np.float64)
    E0 = energies[0]
    if abs(E0) < 1e-12:
        return np.abs(energies - E0)
    return np.abs(energies - E0) / abs(E0)


def compute_drift_at_depth(energies: NDArray[np.float64], depth: int) -> float:
    """Compute relative energy drift at a specific depth (step index).

    Parameters
    ----------
    energies : shape (n_steps+1,)
    depth : int — step index

    Returns
    -------
    drift : float
    """
    energies = np.asarray(energies, dtype=np.float64)
    if depth >= len(energies):
        depth = len(energies) - 1
    return float(compute_energy_drift(energies)[depth])


def compute_mean_drift(energies: NDArray[np.float64]) -> float:
    """Mean relative energy drift over all steps.

    Parameters
    ----------
    energies : shape (n_steps+1,)

    Returns
    -------
    mean_drift : float
    """
    return float(np.mean(compute_energy_drift(energies)))


def compute_max_drift(energies: NDArray[np.float64]) -> float:
    """Maximum relative energy drift over all steps.

    Parameters
    ----------
    energies : shape (n_steps+1,)

    Returns
    -------
    max_drift : float
    """
    return float(np.max(compute_energy_drift(energies)))


def aggregate_drift_stats(
    energy_histories: list[NDArray[np.float64]],
) -> dict[str, float]:
    """Aggregate energy drift statistics across multiple trials.

    Parameters
    ----------
    energy_histories : list of shape-(n_steps+1,) arrays, one per trial

    Returns
    -------
    stats : dict with keys mean_drift, std_drift, max_drift,
            drift_at_10, drift_at_50, drift_at_100
    """
    drifts = [compute_energy_drift(e) for e in energy_histories]
    mean_drifts = np.array([float(np.mean(d)) for d in drifts])
    max_drifts = np.array([float(np.max(d)) for d in drifts])

    n_steps = len(energy_histories[0]) - 1

    def _mean_at(depth: int) -> float:
        return float(np.mean([compute_drift_at_depth(e, depth) for e in energy_histories]))

    return {
        "mean_drift": float(np.mean(mean_drifts)),
        "std_drift": float(np.std(mean_drifts)),
        "max_drift": float(np.max(max_drifts)),
        "drift_at_10": _mean_at(min(10, n_steps)),
        "drift_at_50": _mean_at(min(50, n_steps)),
        "drift_at_100": _mean_at(min(100, n_steps)),
    }
