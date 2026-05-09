"""Symplectic leapfrog (Störmer-Verlet) integrator for Hamiltonian ODEs.

Preserves the symplectic 2-form, bounding energy drift to |E(t)-E(0)| ≤ Ch²t.
"""
import logging
from typing import Callable
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def leapfrog_step(
    H_grad_eta: Callable[[NDArray[np.float64], int], NDArray[np.float64]],
    eta: NDArray[np.float64],
    eta_dot: NDArray[np.float64],
    m: int,
    h: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Single symplectic leapfrog (kick-drift-kick) step.

    Algorithm:
        η̇_{t+½} = η̇_t - (h/2) ∂V/∂η|_t
        η_{t+1}  = η_t + h η̇_{t+½}
        η̇_{t+1}  = η̇_{t+½} - (h/2) ∂V/∂η|_{t+1}

    Parameters
    ----------
    H_grad_eta : Callable[[eta, m], grad]
        Returns ∂H/∂η (= ∂V/∂η since T has no η dependence).
    eta : NDArray[np.float64], shape (r,)
    eta_dot : NDArray[np.float64], shape (r,)
    m : int, contact mode
    h : float, timestep

    Returns
    -------
    eta_new : shape (r,)
    eta_dot_new : shape (r,)
    """
    eta = np.asarray(eta, dtype=np.float64)
    eta_dot = np.asarray(eta_dot, dtype=np.float64)

    grad_t = H_grad_eta(eta, m)
    eta_dot_half = eta_dot - (h / 2.0) * grad_t
    eta_new = eta + h * eta_dot_half
    grad_t1 = H_grad_eta(eta_new, m)
    eta_dot_new = eta_dot_half - (h / 2.0) * grad_t1
    return eta_new, eta_dot_new


def stormer_verlet_step(
    H_grad_eta: Callable[[NDArray[np.float64], int], NDArray[np.float64]],
    eta: NDArray[np.float64],
    eta_dot: NDArray[np.float64],
    m: int,
    h: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Alias for leapfrog_step (Störmer-Verlet position-momentum form)."""
    return leapfrog_step(H_grad_eta, eta, eta_dot, m, h)


def rk4_step(
    H_grad_eta: Callable[[NDArray[np.float64], int], NDArray[np.float64]],
    eta: NDArray[np.float64],
    eta_dot: NDArray[np.float64],
    m: int,
    h: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Non-symplectic RK4 step (reference for energy drift comparison)."""
    eta = np.asarray(eta, dtype=np.float64)
    eta_dot = np.asarray(eta_dot, dtype=np.float64)

    def f(
        e: NDArray[np.float64], ed: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return ed, -H_grad_eta(e, m)

    k1e, k1ed = f(eta, eta_dot)
    k2e, k2ed = f(eta + h / 2 * k1e, eta_dot + h / 2 * k1ed)
    k3e, k3ed = f(eta + h / 2 * k2e, eta_dot + h / 2 * k2ed)
    k4e, k4ed = f(eta + h * k3e, eta_dot + h * k3ed)

    eta_new = eta + (h / 6.0) * (k1e + 2 * k2e + 2 * k3e + k4e)
    eta_dot_new = eta_dot + (h / 6.0) * (k1ed + 2 * k2ed + 2 * k3ed + k4ed)
    return eta_new, eta_dot_new


def forward_euler_step(
    H_grad_eta: Callable[[NDArray[np.float64], int], NDArray[np.float64]],
    eta: NDArray[np.float64],
    eta_dot: NDArray[np.float64],
    m: int,
    h: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Non-symplectic forward Euler step (reference, worst drift)."""
    eta = np.asarray(eta, dtype=np.float64)
    eta_dot = np.asarray(eta_dot, dtype=np.float64)

    grad = H_grad_eta(eta, m)
    eta_new = eta + h * eta_dot
    eta_dot_new = eta_dot - h * grad
    return eta_new, eta_dot_new


def integrate_trajectory(
    H_grad_eta: Callable[[NDArray[np.float64], int], NDArray[np.float64]],
    H_energy: Callable[[NDArray[np.float64], NDArray[np.float64], int], float],
    eta0: NDArray[np.float64],
    eta_dot0: NDArray[np.float64],
    m: int,
    h: float,
    n_steps: int,
    integrator: str = "leapfrog",
) -> dict[str, NDArray[np.float64]]:
    """Integrate Hamiltonian dynamics for n_steps.

    Parameters
    ----------
    H_grad_eta : gradient function
    H_energy : Hamiltonian evaluation function
    eta0, eta_dot0 : initial state
    m : contact mode
    h : timestep
    n_steps : number of integration steps
    integrator : "leapfrog", "rk4", or "euler"

    Returns
    -------
    dict with keys "eta", "eta_dot", "energy", "time"
    """
    step_fn = {
        "leapfrog": leapfrog_step,
        "rk4": rk4_step,
        "euler": forward_euler_step,
    }[integrator]

    eta0 = np.asarray(eta0, dtype=np.float64)
    eta_dot0 = np.asarray(eta_dot0, dtype=np.float64)

    etas = np.zeros((n_steps + 1, len(eta0)))
    eta_dots = np.zeros((n_steps + 1, len(eta_dot0)))
    energies = np.zeros(n_steps + 1)

    etas[0] = eta0
    eta_dots[0] = eta_dot0
    energies[0] = H_energy(eta0, eta_dot0, m)

    eta, eta_dot = eta0.copy(), eta_dot0.copy()
    for i in range(n_steps):
        eta, eta_dot = step_fn(H_grad_eta, eta, eta_dot, m, h)
        etas[i + 1] = eta
        eta_dots[i + 1] = eta_dot
        energies[i + 1] = H_energy(eta, eta_dot, m)

    return {
        "eta": etas,
        "eta_dot": eta_dots,
        "energy": energies,
        "time": np.arange(n_steps + 1) * h,
    }
