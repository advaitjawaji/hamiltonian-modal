"""Benchmark task definitions for DriftBench-G1.

DriftBench-G1 measures Hamiltonian energy drift over long integration
trajectories.  Two integrators are compared:

* **Leapfrog** (symplectic) — conserves a shadow Hamiltonian; drift is O(dt²).
* **Euler** (non-symplectic) — energy error grows unboundedly.

Each :class:`DriftTask` specifies the system size, step count, and
timestep; the runner calls :func:`run_task` and passes the result to
:mod:`~benchmarks.driftbench_g1.metrics`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DriftTask",
    "STANDARD_TASKS",
    "run_task",
]


@dataclass
class DriftTask:
    """Single DriftBench-G1 benchmark configuration.

    Parameters
    ----------
    name:
        Human-readable identifier.
    n_dof:
        Degrees of freedom of the harmonic oscillator system.
    n_steps:
        Number of integration steps.
    dt:
        Integration timestep in seconds.
    integrator:
        ``"leapfrog"`` or ``"euler"``.
    seed:
        Random seed for initial conditions.
    """

    name: str
    n_dof: int = 6
    n_steps: int = 5_000
    dt: float = 0.01
    integrator: str = "leapfrog"
    seed: int = 0


@dataclass
class TaskResult:
    """Output of :func:`run_task`.

    Attributes
    ----------
    task:
        The task specification that produced this result.
    H0:
        Initial Hamiltonian value.
    energies:
        Hamiltonian values at each step, shape ``(n_steps + 1,)``.
    """

    task: DriftTask
    H0: float
    energies: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))


def _make_harmonic_system(
    n_dof: int, seed: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return (M, K, q0, p0) for a random harmonic oscillator.

    The mass matrix M and stiffness K are symmetric positive-definite, giving
    H(q,p) = ½ pᵀ M⁻¹ p + ½ qᵀ K q.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n_dof, n_dof))
    M = A @ A.T + np.eye(n_dof) * n_dof
    B = rng.standard_normal((n_dof, n_dof))
    K = B @ B.T + np.eye(n_dof)
    q0 = rng.standard_normal(n_dof)
    p0 = rng.standard_normal(n_dof)
    return M, K, q0, p0


def run_task(task: DriftTask) -> TaskResult:
    """Execute *task* and return energy measurements.

    Parameters
    ----------
    task:
        :class:`DriftTask` specification.

    Returns
    -------
    TaskResult
        Energies at every step for downstream metric computation.
    """
    M, K, q, p = _make_harmonic_system(task.n_dof, task.seed)
    M_inv = np.linalg.inv(M)

    def hamiltonian(q_: NDArray[np.float64], p_: NDArray[np.float64]) -> float:
        T = 0.5 * float(p_ @ M_inv @ p_)
        V = 0.5 * float(q_ @ K @ q_)
        return T + V

    def grad_V(q_: NDArray[np.float64]) -> NDArray[np.float64]:
        return K @ q_

    energies = np.empty(task.n_steps + 1, dtype=np.float64)
    energies[0] = hamiltonian(q, p)
    H0 = energies[0]

    dt = task.dt

    if task.integrator == "leapfrog":
        for i in range(task.n_steps):
            p_half = p - 0.5 * dt * grad_V(q)
            q = q + dt * (M_inv @ p_half)
            p = p_half - 0.5 * dt * grad_V(q)
            energies[i + 1] = hamiltonian(q, p)
    elif task.integrator == "euler":
        for i in range(task.n_steps):
            q_next = q + dt * (M_inv @ p)
            p_next = p - dt * grad_V(q)
            q, p = q_next, p_next
            energies[i + 1] = hamiltonian(q, p)
    else:
        raise ValueError(f"Unknown integrator '{task.integrator}'. Choose 'leapfrog' or 'euler'.")

    return TaskResult(task=task, H0=H0, energies=energies)


#: Pre-defined suite used by the benchmark runner.
STANDARD_TASKS: list[DriftTask] = [
    DriftTask("leapfrog_small", n_dof=4, n_steps=2_000, dt=0.01, integrator="leapfrog"),
    DriftTask("leapfrog_medium", n_dof=10, n_steps=5_000, dt=0.005, integrator="leapfrog"),
    DriftTask("euler_small", n_dof=4, n_steps=2_000, dt=0.01, integrator="euler"),
    DriftTask("euler_medium", n_dof=10, n_steps=5_000, dt=0.005, integrator="euler"),
]
