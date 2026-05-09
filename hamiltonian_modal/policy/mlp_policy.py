"""MLP policy interfaces.

A pure-NumPy multi-layer perceptron that maps observations to a vector of
continuous actions.  Outputs are squashed to [-1, 1] by tanh.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from hamiltonian_modal.config import MLPPolicyConfig

__all__ = [
    "MLPParams",
    "MLPPolicy",
]

_ACTIVATIONS = {
    "tanh": np.tanh,
    "relu": lambda x: np.maximum(0.0, x),
}


@dataclass
class MLPParams:
    """Learnable parameters for :class:`MLPPolicy`."""

    weights: list[NDArray[np.float64]] = field(default_factory=list)
    biases: list[NDArray[np.float64]] = field(default_factory=list)

    @classmethod
    def random_init(
        cls,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        seed: int = 0,
    ) -> "MLPParams":
        """Xavier-initialised random parameters.

        Parameters
        ----------
        obs_dim:
            Input (observation) dimension.
        action_dim:
            Output (action) dimension.
        hidden_dim:
            Width of hidden layers.
        n_layers:
            Number of hidden layers.
        seed:
            Random seed.
        """
        rng = np.random.default_rng(seed)
        dims = [obs_dim] + [hidden_dim] * n_layers + [action_dim]
        weights, biases = [], []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            scale = np.sqrt(2.0 / d_in)
            weights.append(rng.standard_normal((d_out, d_in)) * scale)
            biases.append(np.zeros(d_out, dtype=np.float64))
        return cls(weights=weights, biases=biases)


class MLPPolicy:
    """Multi-layer perceptron policy.

    Maps an observation vector to a continuous action via a tanh-squashed MLP.

    Parameters
    ----------
    params:
        :class:`MLPParams` holding weights.
    config:
        :class:`~hamiltonian_modal.config.MLPPolicyConfig`.
    """

    def __init__(
        self,
        params: MLPParams,
        config: MLPPolicyConfig | None = None,
    ) -> None:
        self.params = params
        self.config = config or MLPPolicyConfig()
        if self.config.activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation '{self.config.activation}'.")
        self._act = _ACTIVATIONS[self.config.activation]

    def __call__(self, obs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute action = tanh(MLP(obs)).

        Parameters
        ----------
        obs:
            Observation vector, shape ``(obs_dim,)``.

        Returns
        -------
        NDArray[np.float64]
            Action vector in [-1, 1]^action_dim.
        """
        x = np.asarray(obs, dtype=np.float64).reshape(-1)
        for W, b in zip(self.params.weights[:-1], self.params.biases[:-1]):
            x = self._act(W @ x + b)
        W_out, b_out = self.params.weights[-1], self.params.biases[-1]
        return np.tanh(W_out @ x + b_out)
