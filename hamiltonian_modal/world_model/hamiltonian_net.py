"""Hamiltonian neural network for modal-coordinate dynamics.

H_θ(η, η̇, m) = T(η̇) + V_θ(η, m) + Δ_contact_θ(η, m)

Where T = ½‖η̇‖² (analytical), V_θ and Δ_contact_θ are MLPs.
"""
import logging
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _relu(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.maximum(x, 0.0)


def _softplus(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.log1p(np.exp(np.clip(x, -50, 50)))


def _tanh(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.tanh(x)


@dataclass
class MLPParams:
    """Parameters of a fully-connected MLP."""
    weights: list[NDArray[np.float64]]
    biases: list[NDArray[np.float64]]
    activation: str = "tanh"  # "relu", "tanh", "softplus"

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        act = {"relu": _relu, "tanh": _tanh, "softplus": _softplus}[self.activation]
        h = np.asarray(x, dtype=np.float64)
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            h = act(h @ W.T + b)
        W, b = self.weights[-1], self.biases[-1]
        return h @ W.T + b


def _init_mlp(
    layer_sizes: list[int],
    rng: np.random.Generator,
    activation: str = "tanh",
) -> MLPParams:
    """Initialize MLP with Glorot uniform weights."""
    weights = []
    biases = []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        limit = np.sqrt(6.0 / (n_in + n_out))
        W = rng.uniform(-limit, limit, (n_out, n_in))
        b = np.zeros(n_out)
        weights.append(W)
        biases.append(b)
    return MLPParams(weights=weights, biases=biases, activation=activation)


@dataclass
class HamiltonianNetParams:
    """All trainable parameters of HamiltonianNet."""
    V_net: MLPParams
    contact_net: MLPParams
    n_modes: int
    n_contact_modes: int


class HamiltonianNet:
    """Hamiltonian neural network for modal-coordinate dynamics.

    H_θ(η, η̇, m) = T(η̇) + V_θ(η, m) + Δ_contact_θ(η, m)

    Parameters
    ----------
    n_modes : int
        Number of modal coordinates.
    n_contact_modes : int
        Number of contact-mode labels.
    v_hidden : list[int]
        Hidden layer sizes for V_θ MLP.
    contact_hidden : list[int]
        Hidden layer sizes for Δ_contact_θ MLP.
    seed : int
        Random seed for parameter initialization.
    """

    def __init__(
        self,
        n_modes: int,
        n_contact_modes: int,
        v_hidden: list[int] | None = None,
        contact_hidden: list[int] | None = None,
        seed: int = 0,
    ) -> None:
        if v_hidden is None:
            v_hidden = [256, 256, 256, 256]
        if contact_hidden is None:
            contact_hidden = [128, 128, 128]

        self.n_modes = n_modes
        self.n_contact_modes = n_contact_modes

        rng = np.random.default_rng(seed)

        v_in = n_modes + n_contact_modes  # η concatenated with one-hot m
        v_sizes = [v_in] + list(v_hidden) + [1]
        self.V_net = _init_mlp(v_sizes, rng, activation="tanh")

        c_in = n_modes + n_contact_modes
        c_sizes = [c_in] + list(contact_hidden) + [1]
        self.contact_net = _init_mlp(c_sizes, rng, activation="tanh")

    def _mode_embedding(self, m: int) -> NDArray[np.float64]:
        """One-hot encode contact mode."""
        emb = np.zeros(self.n_contact_modes)
        emb[m % self.n_contact_modes] = 1.0
        return emb

    def kinetic(self, eta_dot: NDArray[np.float64]) -> float:
        """T(η̇) = ½‖η̇‖² (analytical, mass-orthonormal)."""
        return 0.5 * float(np.dot(eta_dot, eta_dot))

    def potential(self, eta: NDArray[np.float64], m: int) -> float:
        """V_θ(η, m) — learned potential energy."""
        m_emb = self._mode_embedding(m)
        x = np.concatenate([np.asarray(eta, dtype=np.float64), m_emb])
        return float(self.V_net(x).squeeze())

    def contact_correction(self, eta: NDArray[np.float64], m: int) -> float:
        """Δ_contact_θ(η, m) — learned contact correction."""
        m_emb = self._mode_embedding(m)
        x = np.concatenate([np.asarray(eta, dtype=np.float64), m_emb])
        return float(self.contact_net(x).squeeze())

    def __call__(
        self,
        eta: NDArray[np.float64],
        eta_dot: NDArray[np.float64],
        m: int,
    ) -> float:
        """Compute total Hamiltonian H_θ(η, η̇, m).

        Parameters
        ----------
        eta : shape (n_modes,)
        eta_dot : shape (n_modes,)
        m : int, contact mode index

        Returns
        -------
        H : float, Hamiltonian value
        """
        return self.kinetic(eta_dot) + self.potential(eta, m) + self.contact_correction(eta, m)

    def grad_H_eta(
        self,
        eta: NDArray[np.float64],
        eta_dot: NDArray[np.float64],
        m: int,
        eps: float = 1e-6,
    ) -> NDArray[np.float64]:
        """∂H/∂η via central finite differences.

        Parameters
        ----------
        eta : shape (n_modes,)
        eta_dot : shape (n_modes,)
        m : int
        eps : float, finite difference step

        Returns
        -------
        grad : shape (n_modes,)
        """
        eta = np.asarray(eta, dtype=np.float64)
        grad = np.zeros_like(eta)
        for i in range(len(eta)):
            eta_p = eta.copy()
            eta_p[i] += eps
            eta_m = eta.copy()
            eta_m[i] -= eps
            grad[i] = (self(eta_p, eta_dot, m) - self(eta_m, eta_dot, m)) / (2 * eps)
        return grad

    def grad_H_eta_dot(
        self,
        eta: NDArray[np.float64],
        eta_dot: NDArray[np.float64],
        m: int,
    ) -> NDArray[np.float64]:
        """∂H/∂η̇ = η̇ (analytical for T = ½‖η̇‖²)."""
        return np.asarray(eta_dot, dtype=np.float64).copy()

    def hamilton_step(
        self,
        eta: NDArray[np.float64],
        eta_dot: NDArray[np.float64],
        m: int,
        h: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Single symplectic leapfrog step.

        Parameters
        ----------
        eta : shape (n_modes,)
        eta_dot : shape (n_modes,)
        m : int
        h : float, timestep

        Returns
        -------
        eta_new : shape (n_modes,)
        eta_dot_new : shape (n_modes,)
        """
        from hamiltonian_modal.world_model.symplectic import leapfrog_step

        def grad_fn(e: NDArray[np.float64], mode: int) -> NDArray[np.float64]:
            return self.grad_H_eta(e, eta_dot, mode)

        return leapfrog_step(grad_fn, eta, eta_dot, m, h)
