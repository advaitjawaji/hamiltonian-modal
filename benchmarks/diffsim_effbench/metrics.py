"""Sample efficiency metrics for the DiffSim-EffBench benchmark."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_sample_efficiency_ratio(
    reward_curve_a: NDArray[np.float64],
    reward_curve_b: NDArray[np.float64],
    threshold: float = 0.8,
) -> float:
    """Ratio of steps needed by method B to reach ``threshold`` vs method A.

    A value > 1 means method A is more sample-efficient.

    Parameters
    ----------
    reward_curve_a : shape (n_steps,) — normalised reward in [0, 1]
    reward_curve_b : shape (n_steps,) — normalised reward in [0, 1]
    threshold : float — reward level to use as reference

    Returns
    -------
    ratio : float — steps_b / steps_a; inf if method A never reaches threshold
    """
    steps_a = _steps_to_threshold(reward_curve_a, threshold)
    steps_b = _steps_to_threshold(reward_curve_b, threshold)
    if steps_a is None:
        return float("inf")
    if steps_b is None:
        # method B never reached threshold; ratio is infinity
        return float("inf")
    return float(steps_b) / float(steps_a)


def time_to_threshold(
    reward_curve: NDArray[np.float64],
    threshold: float = 0.8,
    step_duration: float = 1.0,
) -> float:
    """Wall-clock time (simulated) to first reach reward ``threshold``.

    Parameters
    ----------
    reward_curve : shape (n_steps,)
    threshold : float
    step_duration : float — seconds per step

    Returns
    -------
    t : float — time in seconds; float('inf') if never reached
    """
    steps = _steps_to_threshold(reward_curve, threshold)
    if steps is None:
        return float("inf")
    return float(steps) * step_duration


def _steps_to_threshold(
    reward_curve: NDArray[np.float64],
    threshold: float,
) -> "int | None":
    """Return the index of the first step where reward >= threshold, or None."""
    arr = np.asarray(reward_curve, dtype=np.float64)
    indices = np.where(arr >= threshold)[0]
    if len(indices) == 0:
        return None
    return int(indices[0])


def area_under_curve(reward_curve: NDArray[np.float64]) -> float:
    """Trapezoid area under the reward curve (unnormalised sample efficiency).

    Parameters
    ----------
    reward_curve : shape (n_steps,)

    Returns
    -------
    auc : float
    """
    return float(np.trapz(np.asarray(reward_curve, dtype=np.float64)))


def normalised_auc(
    reward_curve: NDArray[np.float64],
    max_reward: float = 1.0,
) -> float:
    """Area under the reward curve, normalised to [0, 1].

    Parameters
    ----------
    reward_curve : shape (n_steps,)
    max_reward : float — maximum achievable reward

    Returns
    -------
    nauc : float in [0, 1]
    """
    n = len(reward_curve)
    if n == 0:
        return 0.0
    auc = area_under_curve(reward_curve)
    return float(auc / (n * max_reward))
