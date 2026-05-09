"""Benchmark task definitions for DiffSim-EffBench.

DiffSim-EffBench measures the computational efficiency and gradient quality
of the differentiable simulation pipeline.  Each :class:`EffTask` controls:

* **horizon** — rollout length, which scales gradient computation cost.
* **batch_size** — parallel trajectories per gradient estimate.
* **n_dof** — degrees of freedom of the simulated system.
* **n_modes** — retained modal coordinates.

:func:`run_eff_task` returns a :class:`EffTaskResult` with timing and
gradient-variance statistics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "EffTask",
    "EffTaskResult",
    "STANDARD_EFF_TASKS",
    "run_eff_task",
]


@dataclass
class EffTask:
    """Single DiffSim-EffBench configuration.

    Parameters
    ----------
    name:
        Human-readable identifier.
    n_dof:
        Degrees of freedom of the system.
    n_modes:
        Number of retained modal coordinates (≤ n_dof).
    horizon:
        Rollout length in steps.
    batch_size:
        Number of trajectories per gradient estimate.
    seed:
        Random seed.
    """

    name: str
    n_dof: int = 12
    n_modes: int = 6
    horizon: int = 50
    batch_size: int = 4
    seed: int = 0


@dataclass
class EffTaskResult:
    """Output of :func:`run_eff_task`.

    Attributes
    ----------
    task:
        The task specification.
    mean_return:
        Mean episode return across the batch.
    grad_norm:
        L2 norm of the estimated policy gradient.
    grad_variance:
        Variance of the gradient estimate across the batch.
    wallclock_s:
        Wall-clock seconds for the full task.
    steps_per_second:
        Simulation throughput: ``horizon * batch_size / wallclock_s``.
    """

    task: EffTask
    mean_return: float
    grad_norm: float
    grad_variance: float
    wallclock_s: float
    steps_per_second: float


def _simulate_rollout(
    n_dof: int,
    n_modes: int,
    horizon: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate a toy linear Hamiltonian rollout and return (rewards, log_probs).

    Uses a simple harmonic oscillator in modal coordinates — no external
    dependencies required.

    Returns
    -------
    tuple[NDArray, NDArray]
        ``rewards`` of shape ``(horizon,)`` and ``log_probs`` of shape
        ``(horizon,)``.
    """
    # Random positive-definite mass matrix (modal, diagonal approx)
    modal_mass_inv = rng.uniform(0.5, 2.0, size=n_modes)  # M⁻¹ diagonal
    spring_k = rng.uniform(0.1, 1.0, size=n_modes)

    # Initial modal state
    eta = rng.standard_normal(n_modes)
    mu = rng.standard_normal(n_modes)
    dt = 0.01

    rewards = np.empty(horizon, dtype=np.float64)
    log_probs = np.empty(horizon, dtype=np.float64)

    for t in range(horizon):
        # Leapfrog step (diagonal system — cheap)
        mu_half = mu - 0.5 * dt * spring_k * eta
        eta = eta + dt * modal_mass_inv * mu_half
        mu = mu_half - 0.5 * dt * spring_k * eta

        # Reward: negative energy (encourages conservation)
        energy = 0.5 * np.sum(modal_mass_inv * mu**2) + 0.5 * np.sum(spring_k * eta**2)
        rewards[t] = -energy

        # Log-prob of a Gaussian policy (scalar action)
        action = rng.standard_normal(n_dof)
        log_probs[t] = -0.5 * np.sum(action**2) - 0.5 * n_dof * np.log(2 * np.pi)

    return rewards, log_probs


def run_eff_task(task: EffTask) -> EffTaskResult:
    """Execute *task* and return efficiency metrics.

    Parameters
    ----------
    task:
        :class:`EffTask` specification.

    Returns
    -------
    EffTaskResult
    """
    rng = np.random.default_rng(task.seed)
    t0 = time.perf_counter()

    all_returns: list[float] = []
    all_grad_norms: list[NDArray[np.float64]] = []

    for _ in range(task.batch_size):
        rewards, log_probs = _simulate_rollout(task.n_dof, task.n_modes, task.horizon, rng)

        # Compute discounted returns
        G = 0.0
        returns = np.empty(task.horizon, dtype=np.float64)
        for s in reversed(range(task.horizon)):
            G = rewards[s] + 0.99 * G
            returns[s] = G

        all_returns.append(float(returns[0]))

        # REINFORCE gradient (w.r.t. log_probs for monitoring)
        baseline = returns.mean()
        advantages = returns - baseline
        grad = -advantages * log_probs / task.horizon
        all_grad_norms.append(grad)

    elapsed = time.perf_counter() - t0
    total_steps = task.horizon * task.batch_size

    grad_stack = np.stack(all_grad_norms, axis=0)  # (batch, horizon)
    mean_grad = grad_stack.mean(axis=0)
    grad_norm = float(np.linalg.norm(mean_grad))
    grad_variance = float(np.var(grad_stack, axis=0).mean())

    return EffTaskResult(
        task=task,
        mean_return=float(np.mean(all_returns)),
        grad_norm=grad_norm,
        grad_variance=grad_variance,
        wallclock_s=elapsed,
        steps_per_second=total_steps / max(elapsed, 1e-9),
    )


#: Pre-defined suite used by the benchmark runner.
STANDARD_EFF_TASKS: list[EffTask] = [
    EffTask("short_horizon", n_dof=12, n_modes=6, horizon=20, batch_size=4),
    EffTask("medium_horizon", n_dof=12, n_modes=6, horizon=100, batch_size=4),
    EffTask("large_batch", n_dof=12, n_modes=6, horizon=50, batch_size=16),
    EffTask("high_dof", n_dof=30, n_modes=10, horizon=50, batch_size=4),
]
