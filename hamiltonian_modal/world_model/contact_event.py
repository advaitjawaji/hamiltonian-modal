"""Contact-event prediction interfaces.

A lightweight classifier that detects / predicts discrete contact events
(foot-ground contact, push, slip) from the modal state (η, μ).

The classifier uses a small MLP with a sigmoid output to produce per-contact
probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ContactState",
    "ContactEventParams",
    "ContactEventPredictor",
]

_SIGMOID = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


@dataclass
class ContactState:
    """Predicted contact state.

    Attributes
    ----------
    probabilities:
        Per-contact probabilities, shape ``(n_contacts,)``.
    active:
        Boolean mask: ``True`` where probability ≥ *threshold*.
    """

    probabilities: NDArray[np.float64]
    active: NDArray[np.bool_]

    @property
    def n_contacts(self) -> int:
        return self.probabilities.shape[0]


@dataclass
class ContactEventParams:
    """Learnable parameters for :class:`ContactEventPredictor`."""

    weights: list[NDArray[np.float64]] = field(default_factory=list)
    biases: list[NDArray[np.float64]] = field(default_factory=list)

    @classmethod
    def random_init(
        cls,
        n_modes: int,
        n_contacts: int,
        hidden_dim: int = 32,
        seed: int = 0,
    ) -> "ContactEventParams":
        rng = np.random.default_rng(seed)
        # Input: concatenation of η and μ → size 2*n_modes
        dims = [2 * n_modes, hidden_dim, n_contacts]
        weights, biases = [], []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            scale = np.sqrt(2.0 / d_in)
            weights.append(rng.standard_normal((d_out, d_in)) * scale)
            biases.append(np.zeros(d_out, dtype=np.float64))
        return cls(weights=weights, biases=biases)


class ContactEventPredictor:
    """MLP that predicts contact probabilities from (η, μ).

    Parameters
    ----------
    params:
        :class:`ContactEventParams`.
    threshold:
        Probability threshold for binary contact detection.
    """

    def __init__(self, params: ContactEventParams, threshold: float = 0.5) -> None:
        self.params = params
        self.threshold = threshold

    def predict(
        self,
        eta: NDArray[np.float64],
        mu: NDArray[np.float64],
    ) -> ContactState:
        """Predict contact state from modal coordinates.

        Parameters
        ----------
        eta:
            Modal position, shape ``(n_modes,)``.
        mu:
            Modal momentum, shape ``(n_modes,)``.

        Returns
        -------
        ContactState
        """
        x = np.concatenate([eta, mu])
        for W, b in zip(self.params.weights[:-1], self.params.biases[:-1]):
            x = np.tanh(W @ x + b)
        W_out, b_out = self.params.weights[-1], self.params.biases[-1]
        logits = W_out @ x + b_out
        probs = _SIGMOID(logits)
        active = probs >= self.threshold
        return ContactState(probabilities=probs, active=active)
