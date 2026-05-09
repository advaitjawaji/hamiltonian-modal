"""Lagrangian-model interfaces.

A pure-NumPy Lagrangian neural network L(q, q̇) that satisfies:

* L(q, q̇) = T(q, q̇) − V(q)
* Equations of motion are derived via the Euler–Lagrange equations.

The kinetic energy is parameterised as a quadratic form with a
configuration-dependent mass matrix M_θ(q), and the potential energy V(q) is
a small MLP.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "LagrangianParams",
    "LagrangianNet",
]


@dataclass
class LagrangianParams:
    """Learnable parameters for :class:`LagrangianNet`."""

    # Potential MLP (same structure as HamiltonianNet)
    V_weights: list[NDArray[np.float64]] = field(default_factory=list)
    V_biases: list[NDArray[np.float64]] = field(default_factory=list)
    # Mass-matrix MLP: maps q → lower-triangular Cholesky factor L (flattened)
    M_weights: list[NDArray[np.float64]] = field(default_factory=list)
    M_biases: list[NDArray[np.float64]] = field(default_factory=list)

    @classmethod
    def random_init(
        cls,
        n_dof: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        seed: int = 0,
    ) -> "LagrangianParams":
        rng = np.random.default_rng(seed)
        n_lower = n_dof * (n_dof + 1) // 2  # lower-triangle elements

        def _mlp_params(
            d_in: int, d_out: int, hidden: int, layers: int
        ) -> tuple[list, list]:
            dims = [d_in] + [hidden] * layers + [d_out]
            ws, bs = [], []
            for di, do in zip(dims[:-1], dims[1:]):
                ws.append(rng.standard_normal((do, di)) * np.sqrt(2.0 / di))
                bs.append(np.zeros(do, dtype=np.float64))
            return ws, bs

        V_weights, V_biases = _mlp_params(n_dof, 1, hidden_dim, n_layers)
        M_weights, M_biases = _mlp_params(n_dof, n_lower, hidden_dim, n_layers)
        return cls(
            V_weights=V_weights, V_biases=V_biases,
            M_weights=M_weights, M_biases=M_biases,
        )


class LagrangianNet:
    """Configuration-dependent Lagrangian neural network L(q, q̇).

    Parameters
    ----------
    params:
        :class:`LagrangianParams`.
    n_dof:
        Number of degrees of freedom.
    """

    def __init__(self, params: LagrangianParams, n_dof: int) -> None:
        self.params = params
        self.n_dof = n_dof

    def _mass_matrix(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the configuration-dependent mass matrix M(q)."""
        x = q.copy()
        for W, b in zip(self.params.M_weights[:-1], self.params.M_biases[:-1]):
            x = np.tanh(W @ x + b)
        W_out, b_out = self.params.M_weights[-1], self.params.M_biases[-1]
        flat = W_out @ x + b_out  # lower-triangle elements
        # Build lower-triangular L with positive diagonal
        L = np.zeros((self.n_dof, self.n_dof), dtype=np.float64)
        idx = np.tril_indices(self.n_dof)
        L[idx] = flat
        # Force positive diagonal via softplus
        for i in range(self.n_dof):
            L[i, i] = np.log1p(np.exp(L[i, i])) + 1e-4
        return L @ L.T  # M = L Lᵀ

    def _potential(self, q: NDArray[np.float64]) -> float:
        x = q.copy()
        for W, b in zip(self.params.V_weights[:-1], self.params.V_biases[:-1]):
            x = np.tanh(W @ x + b)
        W_out, b_out = self.params.V_weights[-1], self.params.V_biases[-1]
        return float(np.log1p(np.exp((W_out @ x + b_out)[0])))

    def __call__(
        self,
        q: NDArray[np.float64],
        q_dot: NDArray[np.float64],
    ) -> float:
        """Evaluate L(q, q̇) = T(q, q̇) − V(q).

        Parameters
        ----------
        q:
            Joint configuration, shape ``(n_dof,)``.
        q_dot:
            Joint velocity, shape ``(n_dof,)``.

        Returns
        -------
        float
            Lagrangian value.
        """
        M = self._mass_matrix(q)
        T = 0.5 * float(q_dot @ M @ q_dot)
        V = self._potential(q)
        return T - V

    def inertia_matrix(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the configuration-dependent mass matrix M(q).

        Parameters
        ----------
        q:
            Joint configuration, shape ``(n_dof,)``.

        Returns
        -------
        NDArray[np.float64]
            Symmetric positive-definite M(q), shape ``(n_dof, n_dof)``.
        """
        return self._mass_matrix(q)
