"""Verifier value head for MCTS: V(η, η̇, m, goal) → ℝ."""
import logging
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class VerifierOutput:
    """Output of the verifier head."""
    value: float
    feasibility: float  # probability the goal is reachable


class Verifier:
    """Dual-head MLP: value head + feasibility head.

    Parameters
    ----------
    n_modes : int
    n_contact_modes : int
    goal_dim : int
    hidden : list[int]
    seed : int
    """

    def __init__(
        self,
        n_modes: int,
        n_contact_modes: int,
        goal_dim: int = 3,
        hidden: list[int] | None = None,
        seed: int = 0,
    ) -> None:
        if hidden is None:
            hidden = [256, 256, 256]
        self.n_modes = n_modes
        self.n_contact_modes = n_contact_modes
        self.goal_dim = goal_dim

        rng = np.random.default_rng(seed)
        n_in = n_modes * 2 + n_contact_modes + goal_dim
        layer_sizes = [n_in] + list(hidden)

        self._shared_weights: list[NDArray[np.float64]] = []
        self._shared_biases: list[NDArray[np.float64]] = []
        for n_i, n_o in zip(layer_sizes[:-1], layer_sizes[1:]):
            lim = np.sqrt(6.0 / (n_i + n_o))
            self._shared_weights.append(rng.uniform(-lim, lim, (n_o, n_i)))
            self._shared_biases.append(np.zeros(n_o))

        last_hidden = layer_sizes[-1]
        lim = np.sqrt(6.0 / (last_hidden + 1))
        self._value_W = rng.uniform(-lim, lim, (1, last_hidden))
        self._value_b = np.zeros(1)
        self._feasibility_W = rng.uniform(-lim, lim, (1, last_hidden))
        self._feasibility_b = np.zeros(1)

    def _forward_shared(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        h = np.asarray(x, dtype=np.float64)
        for W, b in zip(self._shared_weights, self._shared_biases):
            h = np.tanh(h @ W.T + b)
        return h

    def __call__(
        self,
        eta: NDArray[np.float64],
        eta_dot: NDArray[np.float64],
        m: int,
        goal: NDArray[np.float64],
    ) -> VerifierOutput:
        """Evaluate value and feasibility.

        Parameters
        ----------
        eta : shape (n_modes,)
        eta_dot : shape (n_modes,)
        m : int
        goal : shape (goal_dim,)

        Returns
        -------
        VerifierOutput
        """
        m_emb = np.zeros(self.n_contact_modes)
        m_emb[m % self.n_contact_modes] = 1.0
        x = np.concatenate([
            np.asarray(eta, dtype=np.float64),
            np.asarray(eta_dot, dtype=np.float64),
            m_emb,
            np.asarray(goal, dtype=np.float64),
        ])
        h = self._forward_shared(x)
        value = float((h @ self._value_W.T + self._value_b).squeeze())
        feasibility = float(
            1.0 / (1.0 + np.exp(-(h @ self._feasibility_W.T + self._feasibility_b).squeeze()))
        )
        return VerifierOutput(value=value, feasibility=feasibility)
