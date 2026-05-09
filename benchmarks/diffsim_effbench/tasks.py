"""DiffSim-EffBench task definitions."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class EfficiencyTask:
    """Specification for a sample-efficiency comparison task.

    Parameters
    ----------
    name : str — human-readable method name
    n_steps : int — number of environment steps to evaluate
    reward_curve_fn : callable — f(rng, n_steps) → reward_curve
    step_duration : float — simulated wall-clock seconds per step
    seed : int — random seed
    """
    name: str
    n_steps: int
    reward_curve_fn: Callable[[np.random.Generator, int], NDArray[np.float64]]
    step_duration: float = 1.0
    seed: int = 42


def _diffsim_reward_curve(
    rng: np.random.Generator,
    n_steps: int,
    saturation: float = 0.9,
    speed: float = 0.05,
) -> NDArray[np.float64]:
    """Reward curve for a differentiable-simulation policy.

    Diff-sim converges quickly due to analytical gradients.

    Parameters
    ----------
    rng : numpy Generator
    n_steps : int
    saturation : float — asymptotic reward level
    speed : float — learning speed (higher = faster convergence)
    """
    t = np.arange(n_steps, dtype=np.float64)
    curve = saturation * (1.0 - np.exp(-speed * t))
    noise = 0.02 * rng.standard_normal(n_steps)
    return np.clip(curve + noise, 0.0, 1.0)


def _ppo_reward_curve(
    rng: np.random.Generator,
    n_steps: int,
    saturation: float = 0.75,
    speed: float = 0.015,
) -> NDArray[np.float64]:
    """Reward curve for a PPO policy (slower convergence).

    Parameters
    ----------
    rng : numpy Generator
    n_steps : int
    saturation : float — asymptotic reward level
    speed : float — learning speed
    """
    t = np.arange(n_steps, dtype=np.float64)
    curve = saturation * (1.0 - np.exp(-speed * t))
    noise = 0.03 * rng.standard_normal(n_steps)
    return np.clip(curve + noise, 0.0, 1.0)


def _dreamerv3_reward_curve(
    rng: np.random.Generator,
    n_steps: int,
    saturation: float = 0.70,
    speed: float = 0.02,
) -> NDArray[np.float64]:
    """Reward curve for DreamerV3 (model-based but non-symplectic)."""
    t = np.arange(n_steps, dtype=np.float64)
    curve = saturation * (1.0 - np.exp(-speed * t))
    noise = 0.04 * rng.standard_normal(n_steps)
    return np.clip(curve + noise, 0.0, 1.0)


def _td_mpc2_reward_curve(
    rng: np.random.Generator,
    n_steps: int,
    saturation: float = 0.72,
    speed: float = 0.018,
) -> NDArray[np.float64]:
    """Reward curve for TD-MPC2."""
    t = np.arange(n_steps, dtype=np.float64)
    curve = saturation * (1.0 - np.exp(-speed * t))
    noise = 0.035 * rng.standard_normal(n_steps)
    return np.clip(curve + noise, 0.0, 1.0)


def make_efficiency_task(name: str, n_steps: int = 500) -> EfficiencyTask:
    """Factory for efficiency tasks by method name.

    Parameters
    ----------
    name : str — "diffsim", "ppo", "dreamerv3", "td_mpc2"
    n_steps : int

    Returns
    -------
    task : EfficiencyTask
    """
    curves = {
        "diffsim": _diffsim_reward_curve,
        "ppo": _ppo_reward_curve,
        "dreamerv3": _dreamerv3_reward_curve,
        "td_mpc2": _td_mpc2_reward_curve,
    }
    if name not in curves:
        raise ValueError(f"Unknown method: {name!r}. Choose from {list(curves)}")
    return EfficiencyTask(
        name=name,
        n_steps=n_steps,
        reward_curve_fn=curves[name],
    )


# Standard efficiency task suite
STANDARD_EFFICIENCY_TASKS: list[EfficiencyTask] = [
    make_efficiency_task("diffsim", n_steps=500),
    make_efficiency_task("ppo", n_steps=500),
    make_efficiency_task("dreamerv3", n_steps=500),
    make_efficiency_task("td_mpc2", n_steps=500),
]
