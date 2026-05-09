"""Hamiltonian network interfaces.

A pure-NumPy Hamiltonian neural network that parameterises the total energy

.. math::

    H(\\eta, \\mu) = T(\\mu) + V(\\eta)

where **η** are modal coordinates and **μ** are modal momenta (Phase 2
outputs).  The kinetic term T is a learned positive-definite quadratic form
and V is a small MLP.

No autograd framework is required — gradients are computed via the
finite-difference helper :func:`finite_diff_grad`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "HamiltonianParams",
    "HamiltonianNet",
    "finite_diff_grad",
]

_ACTIVATIONS = {
    "tanh": np.tanh,
    "relu": lambda x: np.maximum(0.0, x),
}


def finite_diff_grad(
    fn: "Callable[[NDArray[np.float64]], float]",
    x: NDArray[np.float64],
    eps: float = 1e-5,
) -> NDArray[np.float64]:
    """Central-difference gradient of scalar *fn* at *x*.

    Parameters
    ----------
    fn:
        Scalar-valued function.
    x:
        Point at which to evaluate the gradient, shape ``(n,)``.
    eps:
        Finite-difference step.

    Returns
    -------
    NDArray[np.float64]
        Gradient ∂fn/∂x, shape ``(n,)``.
    """
    from typing import Callable  # local import to avoid circular

    grad = np.zeros_like(x)
    for i in range(x.shape[0]):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        grad[i] = (fn(xp) - fn(xm)) / (2.0 * eps)
    return grad


@dataclass
class HamiltonianParams:
    """Learnable parameters of :class:`HamiltonianNet`.

    Attributes
    ----------
    W_kinetic:
        Lower-triangular Cholesky factor L of the kinetic-energy mass matrix
        M_modal = L Lᵀ, shape ``(n_modes, n_modes)``.
    V_weights:
        List of weight matrices for the potential-energy MLP.
    V_biases:
        List of bias vectors for the potential-energy MLP.
    """

    W_kinetic: NDArray[np.float64]
    V_weights: list[NDArray[np.float64]] = field(default_factory=list)
    V_biases: list[NDArray[np.float64]] = field(default_factory=list)

    @classmethod
    def random_init(
        cls,
        n_modes: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        seed: int = 0,
    ) -> "HamiltonianParams":
        """Randomly initialise parameters.

        Parameters
        ----------
        n_modes:
            Number of modal coordinates.
        hidden_dim:
            Hidden-layer width for the potential MLP.
        n_layers:
            Number of hidden layers.
        seed:
            Random seed.
        """
        rng = np.random.default_rng(seed)
        # Kinetic: start from identity (stable initial energy landscape)
        W_kinetic = np.eye(n_modes, dtype=np.float64)
        # Potential MLP weights (Xavier-like)
        dims = [n_modes] + [hidden_dim] * n_layers + [1]
        V_weights = []
        V_biases = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            scale = np.sqrt(2.0 / d_in)
            V_weights.append(rng.standard_normal((d_out, d_in)) * scale)
            V_biases.append(np.zeros(d_out, dtype=np.float64))
        return cls(W_kinetic=W_kinetic, V_weights=V_weights, V_biases=V_biases)


class HamiltonianNet:
    """Modal-space Hamiltonian network H(η, μ).

    The total Hamiltonian is the sum of separable kinetic and potential terms:

    * **Kinetic** T(μ) = ½ μᵀ (L Lᵀ)⁻¹ μ  where L is stored in
      ``params.W_kinetic``.
    * **Potential** V(η) = MLP(η) — forced positive by a final softplus output.

    Parameters
    ----------
    params:
        :class:`HamiltonianParams` holding weights.
    activation:
        Activation name for the potential MLP: ``"tanh"`` or ``"relu"``.
    """

    def __init__(
        self,
        params: HamiltonianParams,
        activation: str = "tanh",
    ) -> None:
        self.params = params
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. Choose from {list(_ACTIVATIONS)}.")
        self._act = _ACTIVATIONS[activation]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def kinetic(self, mu: NDArray[np.float64]) -> float:
        """Compute T(μ) = ½ μᵀ M_modal⁻¹ μ.

        Parameters
        ----------
        mu:
            Modal momentum, shape ``(n_modes,)``.

        Returns
        -------
        float
        """
        L = self.params.W_kinetic
        # Solve L Lᵀ v = μ via forward-backward substitution
        v = np.linalg.solve(L @ L.T, mu)
        return 0.5 * float(mu @ v)

    def potential(self, eta: NDArray[np.float64]) -> float:
        """Compute V(η) via the potential MLP.

        Parameters
        ----------
        eta:
            Modal position, shape ``(n_modes,)``.

        Returns
        -------
        float
        """
        x = eta.copy()
        for W, b in zip(self.params.V_weights[:-1], self.params.V_biases[:-1]):
            x = self._act(W @ x + b)
        # Final layer — softplus to ensure V ≥ 0
        W_out, b_out = self.params.V_weights[-1], self.params.V_biases[-1]
        logit = float((W_out @ x + b_out)[0])
        return float(np.log1p(np.exp(logit)))  # softplus

    def __call__(self, eta: NDArray[np.float64], mu: NDArray[np.float64]) -> float:
        """Compute H(η, μ) = T(μ) + V(η).

        Parameters
        ----------
        eta:
            Modal position, shape ``(n_modes,)``.
        mu:
            Modal momentum, shape ``(n_modes,)``.

        Returns
        -------
        float
        """
        return self.kinetic(mu) + self.potential(eta)

    # ------------------------------------------------------------------
    # Gradient helpers used by symplectic integrators
    # ------------------------------------------------------------------

    def grad_H_eta(
        self, eta: NDArray[np.float64], mu: NDArray[np.float64], eps: float = 1e-5
    ) -> NDArray[np.float64]:
        """Compute ∂H/∂η = ∂V/∂η via finite differences."""
        return finite_diff_grad(lambda e: self.potential(e), eta, eps=eps)

    def grad_H_mu(
        self, eta: NDArray[np.float64], mu: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute ∂H/∂μ = M_modal⁻¹ μ (exact)."""
        L = self.params.W_kinetic
        return np.linalg.solve(L @ L.T, mu)
