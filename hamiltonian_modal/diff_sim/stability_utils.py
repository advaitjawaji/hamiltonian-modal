"""Gradient stability utilities for differentiable simulation training."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def clip_grad_norm(
    grads: list[NDArray[np.float64]],
    max_norm: float = 1.0,
) -> tuple[list[NDArray[np.float64]], float]:
    """Clip a list of gradient arrays by their global L2 norm.

    Parameters
    ----------
    grads : list of gradient arrays (any shape)
    max_norm : float — maximum allowable global gradient norm

    Returns
    -------
    clipped_grads : list of clipped gradient arrays
    grad_norm : float — original (pre-clipping) global norm
    """
    total_sq = sum(float(np.sum(g ** 2)) for g in grads)
    grad_norm = float(np.sqrt(total_sq))

    if grad_norm > max_norm and grad_norm > 1e-12:
        scale = max_norm / grad_norm
        clipped = [g * scale for g in grads]
    else:
        clipped = [g.copy() for g in grads]

    return clipped, grad_norm


def trajectory_length_curriculum(
    current_step: int,
    total_steps: int,
    t_start: int = 10,
    t_end: int = 100,
) -> int:
    """Compute the curriculum trajectory length at the current training step.

    Linearly interpolates the horizon from ``t_start`` to ``t_end`` over
    ``total_steps`` gradient steps.  This avoids exploding gradients from
    long rollouts early in training.

    Parameters
    ----------
    current_step : int — current training iteration (0-indexed)
    total_steps : int — total number of training iterations
    t_start : int — initial trajectory length
    t_end : int — final trajectory length

    Returns
    -------
    length : int — trajectory length for this step
    """
    if total_steps <= 0:
        return t_end
    progress = min(current_step / total_steps, 1.0)
    return int(t_start + progress * (t_end - t_start))


def compute_per_timestep_grad_norms(
    trajectory_grads: list[NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Compute the gradient L2 norm at each timestep of a trajectory.

    Useful for diagnosing gradient explosion or vanishing in diff-sim rollouts.

    Parameters
    ----------
    trajectory_grads : list of gradient arrays, one per timestep

    Returns
    -------
    norms : shape (T,) — per-timestep gradient norms
    """
    return np.array([float(np.linalg.norm(g)) for g in trajectory_grads], dtype=np.float64)


def stable_log(x: NDArray[np.float64], eps: float = 1e-8) -> NDArray[np.float64]:
    """Numerically stable natural logarithm.

    Parameters
    ----------
    x : input array
    eps : float — floor value to prevent log(0)

    Returns
    -------
    log_x : clipped log
    """
    return np.log(np.maximum(np.asarray(x, dtype=np.float64), eps))


def compute_grad_variance(
    grad_history: list[list[NDArray[np.float64]]],
) -> list[float]:
    """Compute per-parameter gradient variance over a window of iterations.

    Parameters
    ----------
    grad_history : list of gradient lists, one per iteration
        Each inner list has the same structure: [param_0_grad, param_1_grad, ...]

    Returns
    -------
    variances : list[float] — variance of gradient norm for each parameter
    """
    if not grad_history:
        return []

    n_params = len(grad_history[0])
    variances: list[float] = []
    for p in range(n_params):
        norms = [float(np.linalg.norm(iteration[p])) for iteration in grad_history]
        variances.append(float(np.var(norms)))
    return variances


def check_grad_health(
    grads: list[NDArray[np.float64]],
    nan_threshold: float = 1e6,
) -> dict[str, bool | float]:
    """Sanity-check gradient arrays for NaN, Inf, and explosion.

    Parameters
    ----------
    grads : list of gradient arrays
    nan_threshold : float — norm above which gradient is considered exploded

    Returns
    -------
    report : dict with keys "has_nan", "has_inf", "is_exploded", "global_norm"
    """
    has_nan = any(bool(np.any(np.isnan(g))) for g in grads)
    has_inf = any(bool(np.any(np.isinf(g))) for g in grads)
    total_sq = sum(float(np.sum(g ** 2)) for g in grads if np.all(np.isfinite(g)))
    global_norm = float(np.sqrt(total_sq))
    is_exploded = global_norm > nan_threshold

    return {
        "has_nan": has_nan,
        "has_inf": has_inf,
        "is_exploded": is_exploded,
        "global_norm": global_norm,
    }
