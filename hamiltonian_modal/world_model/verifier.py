"""Verifier head interfaces for search.

A lightweight value-and-feasibility verifier used by the MCTS planner to
score leaf nodes without performing full rollouts.  The verifier:

1. Predicts a scalar *value* estimate V(η, μ) ∈ ℝ (higher = better).
2. Predicts a *feasibility* probability F(η, μ) ∈ [0, 1] (1 = fully feasible).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "VerifierOutput",
    "VerifierParams",
    "Verifier",
]

_SIGMOID = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


@dataclass
class VerifierOutput:
    """Output of a single :class:`Verifier` query.

    Attributes
    ----------
    value:
        Estimated long-horizon return.
    feasibility:
        Probability that the state is kinematically / dynamically feasible.
    """

    value: float
    feasibility: float


@dataclass
class VerifierParams:
    """Learnable parameters for the verifier network."""

    weights: list[NDArray[np.float64]] = field(default_factory=list)
    biases: list[NDArray[np.float64]] = field(default_factory=list)

    @classmethod
    def random_init(
        cls,
        n_modes: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        seed: int = 0,
    ) -> "VerifierParams":
        rng = np.random.default_rng(seed)
        # Input: η ⊕ μ → 2*n_modes; Output: [value, feasibility_logit]
        dims = [2 * n_modes] + [hidden_dim] * n_layers + [2]
        weights, biases = [], []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            scale = np.sqrt(2.0 / d_in)
            weights.append(rng.standard_normal((d_out, d_in)) * scale)
            biases.append(np.zeros(d_out, dtype=np.float64))
        return cls(weights=weights, biases=biases)


class Verifier:
    """Dual-head MLP verifier network for MCTS leaf scoring.

    Parameters
    ----------
    params:
        :class:`VerifierParams`.
    """

    def __init__(self, params: VerifierParams) -> None:
        self.params = params

    def predict(
        self,
        eta: NDArray[np.float64],
        mu: NDArray[np.float64],
    ) -> VerifierOutput:
        """Predict value and feasibility for a modal state.

        Parameters
        ----------
        eta:
            Modal position, shape ``(n_modes,)``.
        mu:
            Modal momentum, shape ``(n_modes,)``.

        Returns
        -------
        VerifierOutput
        """
        x = np.concatenate([eta, mu])
        for W, b in zip(self.params.weights[:-1], self.params.biases[:-1]):
            x = np.tanh(W @ x + b)
        W_out, b_out = self.params.weights[-1], self.params.biases[-1]
        out = W_out @ x + b_out  # shape (2,)
        value = float(out[0])
        feasibility = float(_SIGMOID(out[1]))
        return VerifierOutput(value=value, feasibility=feasibility)
