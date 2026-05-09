"""Flow-matching policy head interfaces.

Implements a simplified *conditional flow-matching* action head that generates
actions by integrating a learned vector field from a noise sample to a target
action distribution.

The flow is conditioned on the observation vector and integrated over
``n_steps`` Euler steps from t=0 to t=1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "FlowParams",
    "FlowMatchingHead",
]


@dataclass
class FlowParams:
    """Learnable parameters for :class:`FlowMatchingHead`."""

    weights: list[NDArray[np.float64]] = field(default_factory=list)
    biases: list[NDArray[np.float64]] = field(default_factory=list)

    @classmethod
    def random_init(
        cls,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        seed: int = 0,
    ) -> "FlowParams":
        """Random Xavier initialisation.

        Parameters
        ----------
        obs_dim:
            Observation dimension (conditioning signal).
        action_dim:
            Action dimension.
        hidden_dim:
            Hidden-layer width.
        n_layers:
            Number of hidden layers.
        seed:
            Random seed.
        """
        rng = np.random.default_rng(seed)
        # Input: obs ⊕ x_t ⊕ t_scalar → obs_dim + action_dim + 1
        in_dim = obs_dim + action_dim + 1
        dims = [in_dim] + [hidden_dim] * n_layers + [action_dim]
        weights, biases = [], []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            scale = np.sqrt(2.0 / d_in)
            weights.append(rng.standard_normal((d_out, d_in)) * scale)
            biases.append(np.zeros(d_out, dtype=np.float64))
        return cls(weights=weights, biases=biases)


class FlowMatchingHead:
    """Conditional flow-matching policy head.

    Generates an action by integrating the learned velocity field v_θ(obs, x, t)
    from x₀ ~ N(0, I) (t=0) to x₁ (t=1) using n_steps Euler steps.

    Parameters
    ----------
    params:
        :class:`FlowParams`.
    action_dim:
        Dimensionality of the action space.
    n_steps:
        Number of Euler integration steps (higher = more accurate, slower).
    seed:
        Seed for the noise sample at inference time.
    """

    def __init__(
        self,
        params: FlowParams,
        action_dim: int,
        n_steps: int = 10,
        seed: int = 0,
    ) -> None:
        self.params = params
        self.action_dim = action_dim
        self.n_steps = n_steps
        self._rng = np.random.default_rng(seed)

    def _velocity(
        self,
        obs: NDArray[np.float64],
        x: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        """Compute v_θ(obs, x, t)."""
        inp = np.concatenate([obs, x, [t]])
        for W, b in zip(self.params.weights[:-1], self.params.biases[:-1]):
            inp = np.tanh(W @ inp + b)
        W_out, b_out = self.params.weights[-1], self.params.biases[-1]
        return W_out @ inp + b_out

    def __call__(self, obs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Sample an action conditioned on *obs*.

        Parameters
        ----------
        obs:
            Observation vector, shape ``(obs_dim,)``.

        Returns
        -------
        NDArray[np.float64]
            Action vector, shape ``(action_dim,)``.
        """
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        x = self._rng.standard_normal(self.action_dim)
        dt = 1.0 / self.n_steps
        for i in range(self.n_steps):
            t = i * dt
            x = x + dt * self._velocity(obs, x, t)
        return np.tanh(x)
