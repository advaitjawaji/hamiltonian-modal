"""Simple MLP policy for differentiable-simulation training."""
from __future__ import annotations

import logging
from typing import Any
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _relu(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.maximum(x, 0.0)


def _tanh(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.tanh(x)


class MLPPolicy:
    """Simple multi-layer perceptron policy.

    Maps observations to deterministic actions (mean of Gaussian).
    The covariance is a fixed diagonal matrix for stochastic sampling.

    Parameters
    ----------
    obs_dim : int — observation dimensionality
    action_dim : int — action dimensionality
    hidden : list[int] — hidden layer sizes
    seed : int — random seed for parameter initialisation
    log_std_init : float — initial log standard deviation
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: "list[int] | None" = None,
        seed: int = 0,
        log_std_init: float = -0.5,
    ) -> None:
        if hidden is None:
            hidden = [256, 256]

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden = hidden

        rng = np.random.default_rng(seed)
        layer_sizes = [obs_dim] + list(hidden) + [action_dim]

        self._weights: list[NDArray[np.float64]] = []
        self._biases: list[NDArray[np.float64]] = []

        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = np.sqrt(2.0 / n_in)  # He initialisation
            W = rng.uniform(-limit, limit, (n_out, n_in)).astype(np.float64)
            b = np.zeros(n_out, dtype=np.float64)
            self._weights.append(W)
            self._biases.append(b)

        # Fixed log-std for stochastic sampling
        self._log_std = np.full(action_dim, log_std_init, dtype=np.float64)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def __call__(self, obs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the deterministic action (mean) for an observation.

        Parameters
        ----------
        obs : shape (obs_dim,)

        Returns
        -------
        action : shape (action_dim,) in [-1, 1] (tanh squash)
        """
        h = np.asarray(obs, dtype=np.float64).ravel()
        for W, b in zip(self._weights[:-1], self._biases[:-1]):
            h = _tanh(h @ W.T + b)
        W, b = self._weights[-1], self._biases[-1]
        h = h @ W.T + b
        return np.tanh(h)

    def get_distribution(
        self,
        obs: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (mean, covariance) of the action distribution.

        Parameters
        ----------
        obs : shape (obs_dim,)

        Returns
        -------
        mean : shape (action_dim,)
        cov : shape (action_dim, action_dim) — diagonal covariance matrix
        """
        mean = self(obs)
        std = np.exp(self._log_std)
        cov = np.diag(std ** 2)
        return mean, cov

    def sample(
        self,
        obs: NDArray[np.float64],
        rng: "np.random.Generator | None" = None,
    ) -> NDArray[np.float64]:
        """Sample a stochastic action from the policy distribution.

        Parameters
        ----------
        obs : shape (obs_dim,)
        rng : numpy Generator or None (uses default_rng(0))

        Returns
        -------
        action : shape (action_dim,)
        """
        if rng is None:
            rng = np.random.default_rng(0)
        mean, cov = self.get_distribution(obs)
        noise = rng.standard_normal(self.action_dim)
        return mean + np.sqrt(np.diag(cov)) * noise

    def log_prob(
        self,
        obs: NDArray[np.float64],
        action: NDArray[np.float64],
    ) -> float:
        """Log probability of an action under the policy distribution.

        Treats the distribution as a diagonal Gaussian in pre-tanh space
        (simplified: ignores the tanh Jacobian).

        Parameters
        ----------
        obs : shape (obs_dim,)
        action : shape (action_dim,)

        Returns
        -------
        log_p : float
        """
        mean = self(obs)
        std = np.exp(self._log_std)
        action = np.asarray(action, dtype=np.float64)
        log_p = -0.5 * np.sum(((action - mean) / (std + 1e-8)) ** 2)
        log_p -= np.sum(np.log(std + 1e-8))
        log_p -= 0.5 * self.action_dim * np.log(2.0 * np.pi)
        return float(log_p)

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    def parameters(self) -> list[NDArray[np.float64]]:
        """Return a list of all trainable parameter arrays.

        The order is: [W_0, b_0, W_1, b_1, ..., W_L, b_L, log_std].

        Returns
        -------
        params : list of NDArray
        """
        params: list[NDArray[np.float64]] = []
        for W, b in zip(self._weights, self._biases):
            params.append(W)
            params.append(b)
        params.append(self._log_std)
        return params

    def set_parameters(self, params: list[NDArray[np.float64]]) -> None:
        """Set all trainable parameters from a list.

        Parameters must be in the same order as returned by ``parameters()``.

        Parameters
        ----------
        params : list of NDArray
        """
        n_layers = len(self._weights)
        if len(params) != 2 * n_layers + 1:
            raise ValueError(
                f"Expected {2 * n_layers + 1} parameter arrays, got {len(params)}"
            )
        for i in range(n_layers):
            self._weights[i] = np.asarray(params[2 * i], dtype=np.float64).copy()
            self._biases[i] = np.asarray(params[2 * i + 1], dtype=np.float64).copy()
        self._log_std = np.asarray(params[-1], dtype=np.float64).copy()

    def parameter_count(self) -> int:
        """Return total number of scalar parameters."""
        return sum(p.size for p in self.parameters())
