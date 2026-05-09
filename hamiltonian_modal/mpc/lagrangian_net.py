"""Lagrangian neural network for modal-coordinate dynamics.

L_θ(η, η̇, m) = T(η̇) - V_θ(η, m)

Where T = ½‖η̇‖² (analytical), V_θ is the learned potential energy.
The equations of motion follow from the Euler-Lagrange equations:
  d/dt(∂L/∂η̇) - ∂L/∂η = 0  →  η̈ = -∂V_θ/∂η
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _tanh(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.tanh(x)


def _relu(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.maximum(x, 0.0)


@dataclass
class LagrangianNetParams:
    """Trainable parameters of LagrangianNet."""
    V_weights: list[NDArray[np.float64]]
    V_biases: list[NDArray[np.float64]]
    n_modes: int
    n_contact_modes: int


def _init_mlp(
    layer_sizes: list[int],
    rng: np.random.Generator,
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Glorot uniform initialisation."""
    weights, biases = [], []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        limit = np.sqrt(6.0 / (n_in + n_out))
        W = rng.uniform(-limit, limit, (n_out, n_in)).astype(np.float64)
        b = np.zeros(n_out, dtype=np.float64)
        weights.append(W)
        biases.append(b)
    return weights, biases


class LagrangianNet:
    """Lagrangian neural network for modal-coordinate dynamics.

    L_θ(η, η̇, m) = T(η̇) - V_θ(η, m)

    The kinetic energy T = ½‖η̇‖² is analytical (mass-orthonormal modes).
    V_θ is a learned MLP.

    Equations of motion (Euler-Lagrange):
        η̈ = -∂V_θ/∂η

    Integration uses the explicit midpoint (leapfrog-equivalent) method.

    Parameters
    ----------
    n_modes : int
    n_contact_modes : int
    v_hidden : list[int]
    seed : int
    """

    def __init__(
        self,
        n_modes: int,
        n_contact_modes: int,
        v_hidden: "list[int] | None" = None,
        seed: int = 0,
    ) -> None:
        if v_hidden is None:
            v_hidden = [256, 256, 256]

        self.n_modes = n_modes
        self.n_contact_modes = n_contact_modes

        rng = np.random.default_rng(seed)
        v_in = n_modes + n_contact_modes
        v_sizes = [v_in] + list(v_hidden) + [1]
        self._V_weights, self._V_biases = _init_mlp(v_sizes, rng)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def _mode_embedding(self, m: int) -> NDArray[np.float64]:
        """One-hot encode contact mode."""
        emb = np.zeros(self.n_contact_modes, dtype=np.float64)
        emb[m % self.n_contact_modes] = 1.0
        return emb

    def _V_forward(self, x: NDArray[np.float64]) -> float:
        h = np.asarray(x, dtype=np.float64)
        for W, b in zip(self._V_weights[:-1], self._V_biases[:-1]):
            h = _tanh(h @ W.T + b)
        W, b = self._V_weights[-1], self._V_biases[-1]
        return float((h @ W.T + b).squeeze())

    def kinetic(self, eta_dot: NDArray[np.float64]) -> float:
        """T(η̇) = ½‖η̇‖²."""
        return 0.5 * float(np.dot(eta_dot, eta_dot))

    def potential(self, eta: NDArray[np.float64], m: int) -> float:
        """V_θ(η, m) — learned potential energy."""
        m_emb = self._mode_embedding(m)
        x = np.concatenate([np.asarray(eta, dtype=np.float64), m_emb])
        return self._V_forward(x)

    def __call__(
        self,
        eta: NDArray[np.float64],
        eta_dot: NDArray[np.float64],
        m: int,
    ) -> float:
        """Compute Lagrangian L = T - V.

        Parameters
        ----------
        eta : shape (n_modes,)
        eta_dot : shape (n_modes,)
        m : int

        Returns
        -------
        L : float
        """
        return self.kinetic(eta_dot) - self.potential(eta, m)

    def grad_V_eta(
        self,
        eta: NDArray[np.float64],
        m: int,
        eps: float = 1e-6,
    ) -> NDArray[np.float64]:
        """∂V_θ/∂η via central finite differences.

        Parameters
        ----------
        eta : shape (n_modes,)
        m : int
        eps : float

        Returns
        -------
        grad : shape (n_modes,)
        """
        eta = np.asarray(eta, dtype=np.float64)
        grad = np.zeros_like(eta)
        for i in range(len(eta)):
            e_p = eta.copy(); e_p[i] += eps
            e_m = eta.copy(); e_m[i] -= eps
            grad[i] = (self.potential(e_p, m) - self.potential(e_m, m)) / (2.0 * eps)
        return grad

    def euler_lagrange_step(
        self,
        eta: NDArray[np.float64],
        eta_dot: NDArray[np.float64],
        m: int,
        h: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Single symplectic step using Euler-Lagrange equations.

        η̈ = -∂V_θ/∂η (from Euler-Lagrange with T = ½‖η̇‖²)

        Uses the Störmer-Verlet (leapfrog) scheme:
          η̇_{n+½} = η̇_n - (h/2) ∂V/∂η(η_n)
          η_{n+1}  = η_n + h η̇_{n+½}
          η̇_{n+1}  = η̇_{n+½} - (h/2) ∂V/∂η(η_{n+1})

        Parameters
        ----------
        eta : shape (n_modes,)
        eta_dot : shape (n_modes,)
        m : int
        h : float — timestep

        Returns
        -------
        eta_new : shape (n_modes,)
        eta_dot_new : shape (n_modes,)
        """
        grad_V = self.grad_V_eta(eta, m)
        eta_dot_half = eta_dot - 0.5 * h * grad_V
        eta_new = eta + h * eta_dot_half
        grad_V_new = self.grad_V_eta(eta_new, m)
        eta_dot_new = eta_dot_half - 0.5 * h * grad_V_new
        return eta_new, eta_dot_new

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    def parameters(self) -> list[NDArray[np.float64]]:
        """Return all trainable parameters."""
        params: list[NDArray[np.float64]] = []
        for W, b in zip(self._V_weights, self._V_biases):
            params.append(W)
            params.append(b)
        return params

    def set_parameters(self, params: list[NDArray[np.float64]]) -> None:
        """Set parameters from list in same order as parameters()."""
        n_layers = len(self._V_weights)
        if len(params) != 2 * n_layers:
            raise ValueError(f"Expected {2 * n_layers} arrays, got {len(params)}")
        for i in range(n_layers):
            self._V_weights[i] = np.asarray(params[2 * i], dtype=np.float64).copy()
            self._V_biases[i] = np.asarray(params[2 * i + 1], dtype=np.float64).copy()
