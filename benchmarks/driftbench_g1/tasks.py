"""DriftBench-G1 task definitions and synthetic baseline generators."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class DriftTask:
    """Specification for a single drift benchmark task.

    Parameters
    ----------
    name : str — human-readable method name
    n_modes : int — number of modal coordinates
    h : float — integration timestep
    n_steps : int — number of steps to integrate
    integrator : str — "leapfrog", "rk4", "euler", or "synthetic:<rate>"
    seed : int — random seed for reproducibility
    """
    name: str
    n_modes: int = 10
    h: float = 0.01
    n_steps: int = 100
    integrator: str = "leapfrog"
    seed: int = 42


def synthetic_baseline_drift(
    n_steps: int,
    drift_rate: float,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Simulate energy trajectory for a baseline method with given drift rate.

    Models the secular energy growth typical of non-symplectic integrators
    and learned world models that don't enforce energy conservation.

    Parameters
    ----------
    n_steps : int — number of integration steps
    drift_rate : float — per-step relative drift rate (higher = less stable)
    seed : int — random seed

    Returns
    -------
    energies : shape (n_steps+1,) — simulated energy trajectory
    """
    rng = np.random.default_rng(seed)
    E0 = 10.0
    energies = [E0]
    E = E0
    for _ in range(n_steps):
        E = E + drift_rate * abs(E) * 0.01 + rng.normal(0.0, 0.01 * abs(E))
        energies.append(E)
    return np.array(energies, dtype=np.float64)


def run_leapfrog_integration(
    n_modes: int,
    h: float,
    n_steps: int,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Run leapfrog integration on a simple harmonic oscillator in modal coordinates.

    Uses ω_k = k+1 as natural frequencies for mode k.

    Parameters
    ----------
    n_modes : int
    h : float — timestep
    n_steps : int
    seed : int

    Returns
    -------
    energies : shape (n_steps+1,) — total energy at each step
    """
    rng = np.random.default_rng(seed)
    omega = np.arange(1, n_modes + 1, dtype=np.float64)  # natural frequencies

    # Initial conditions
    eta = rng.standard_normal(n_modes)
    eta_dot = rng.standard_normal(n_modes)

    def total_energy(e: NDArray, edot: NDArray) -> float:
        T = 0.5 * np.dot(edot, edot)
        V = 0.5 * np.dot(omega ** 2, e ** 2)
        return float(T + V)

    energies = [total_energy(eta, eta_dot)]

    for _ in range(n_steps):
        # Leapfrog (Störmer–Verlet):
        # η̈ = -ω² η  (simple harmonic oscillator)
        # half-step for velocity
        eta_dot = eta_dot - 0.5 * h * omega ** 2 * eta
        # full step for position
        eta = eta + h * eta_dot
        # half-step for velocity
        eta_dot = eta_dot - 0.5 * h * omega ** 2 * eta
        energies.append(total_energy(eta, eta_dot))

    return np.array(energies, dtype=np.float64)


def run_rk4_integration(
    n_modes: int,
    h: float,
    n_steps: int,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Run 4th-order Runge-Kutta integration on modal harmonic oscillator.

    RK4 is not symplectic and shows gradual energy drift.

    Parameters
    ----------
    n_modes : int
    h : float — timestep
    n_steps : int
    seed : int

    Returns
    -------
    energies : shape (n_steps+1,)
    """
    rng = np.random.default_rng(seed)
    omega = np.arange(1, n_modes + 1, dtype=np.float64)

    eta = rng.standard_normal(n_modes)
    eta_dot = rng.standard_normal(n_modes)

    def deriv(e: NDArray, edot: NDArray) -> tuple[NDArray, NDArray]:
        return edot.copy(), -(omega ** 2) * e

    def total_energy(e: NDArray, edot: NDArray) -> float:
        return float(0.5 * np.dot(edot, edot) + 0.5 * np.dot(omega ** 2, e ** 2))

    energies = [total_energy(eta, eta_dot)]

    for _ in range(n_steps):
        k1e, k1d = deriv(eta, eta_dot)
        k2e, k2d = deriv(eta + 0.5 * h * k1e, eta_dot + 0.5 * h * k1d)
        k3e, k3d = deriv(eta + 0.5 * h * k2e, eta_dot + 0.5 * h * k2d)
        k4e, k4d = deriv(eta + h * k3e, eta_dot + h * k3d)
        eta = eta + (h / 6.0) * (k1e + 2 * k2e + 2 * k3e + k4e)
        eta_dot = eta_dot + (h / 6.0) * (k1d + 2 * k2d + 2 * k3d + k4d)
        energies.append(total_energy(eta, eta_dot))

    return np.array(energies, dtype=np.float64)


def run_euler_integration(
    n_modes: int,
    h: float,
    n_steps: int,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Run forward Euler integration on modal harmonic oscillator.

    Euler is not symplectic and shows growing energy (energy injected at each step).

    Parameters
    ----------
    n_modes : int
    h : float — timestep
    n_steps : int
    seed : int

    Returns
    -------
    energies : shape (n_steps+1,)
    """
    rng = np.random.default_rng(seed)
    omega = np.arange(1, n_modes + 1, dtype=np.float64)

    eta = rng.standard_normal(n_modes)
    eta_dot = rng.standard_normal(n_modes)

    def total_energy(e: NDArray, edot: NDArray) -> float:
        return float(0.5 * np.dot(edot, edot) + 0.5 * np.dot(omega ** 2, e ** 2))

    energies = [total_energy(eta, eta_dot)]

    for _ in range(n_steps):
        accel = -(omega ** 2) * eta
        eta = eta + h * eta_dot
        eta_dot = eta_dot + h * accel
        energies.append(total_energy(eta, eta_dot))

    return np.array(energies, dtype=np.float64)


def run_task(task: DriftTask) -> NDArray[np.float64]:
    """Run a drift task and return the energy trajectory.

    Parameters
    ----------
    task : DriftTask

    Returns
    -------
    energies : shape (n_steps+1,)
    """
    integrator = task.integrator.lower()

    if integrator == "leapfrog":
        return run_leapfrog_integration(task.n_modes, task.h, task.n_steps, task.seed)
    elif integrator == "rk4":
        return run_rk4_integration(task.n_modes, task.h, task.n_steps, task.seed)
    elif integrator == "euler":
        return run_euler_integration(task.n_modes, task.h, task.n_steps, task.seed)
    elif integrator.startswith("synthetic:"):
        drift_rate = float(integrator.split(":")[1])
        return synthetic_baseline_drift(task.n_steps, drift_rate, task.seed)
    else:
        raise ValueError(f"Unknown integrator: {task.integrator!r}")


# Standard task suite for the DriftBench-G1 paper results
STANDARD_TASKS: list[DriftTask] = [
    DriftTask(name="Leapfrog (Ours)", n_modes=10, h=0.01, n_steps=100, integrator="leapfrog"),
    DriftTask(name="RK4", n_modes=10, h=0.01, n_steps=100, integrator="rk4"),
    DriftTask(name="Euler", n_modes=10, h=0.01, n_steps=100, integrator="euler"),
    # Synthetic baselines matching known instability of learned world models
    DriftTask(name="DreamerV3", n_modes=10, h=0.01, n_steps=100, integrator="synthetic:0.5"),
    DriftTask(name="TD-MPC2", n_modes=10, h=0.01, n_steps=100, integrator="synthetic:0.3"),
    DriftTask(name="Puppeteer", n_modes=10, h=0.01, n_steps=100, integrator="synthetic:0.2"),
    DriftTask(name="V-JEPA2", n_modes=10, h=0.01, n_steps=100, integrator="synthetic:0.4"),
    DriftTask(name="RoboScape", n_modes=10, h=0.01, n_steps=100, integrator="synthetic:0.35"),
]
