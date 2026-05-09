"""TD-MPC2 baseline adapter for hamiltonian-modal G1 environments.

TD-MPC2 (Hansen et al., 2024) is a model-based RL agent that plans with
a learned latent world model using MPPI at inference time.  This adapter
wraps a G1 environment so that it can be used with a TD-MPC2 training loop.

No TD-MPC2 installation is required to import this module.  The
:meth:`build_agent` method raises :exc:`ImportError` when ``tdmpc2`` is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["TDMPC2Config", "TDMPC2Adapter"]


@dataclass
class TDMPC2Config:
    """Minimal TD-MPC2 hyperparameters.

    Attributes
    ----------
    latent_dim:
        Dimension of the learned latent state.
    horizon:
        MPPI planning horizon.
    n_iterations:
        Number of MPPI sampling iterations.
    n_samples:
        Number of action-sequence samples per MPPI step.
    temperature:
        MPPI temperature (higher = more exploration).
    lr:
        Learning rate for world-model and policy training.
    """

    latent_dim: int = 512
    horizon: int = 3
    n_iterations: int = 6
    n_samples: int = 512
    temperature: float = 0.5
    lr: float = 3e-4


class TDMPC2Adapter:
    """Thin shim between G1 environments and TD-MPC2.

    Parameters
    ----------
    obs_dim:
        Observation dimension.
    act_dim:
        Action dimension.
    config:
        :class:`TDMPC2Config`.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        config: TDMPC2Config | None = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config = config or TDMPC2Config()
        self._total_steps: int = 0

    def encode(self, obs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Encode *obs* into the latent state space (placeholder projection).

        Parameters
        ----------
        obs:
            Observation vector, shape ``(obs_dim,)``.

        Returns
        -------
        NDArray[np.float64]
            Latent state, shape ``(latent_dim,)``.
        """
        obs = np.asarray(obs, dtype=np.float64)
        rng = np.random.default_rng(0)
        W = rng.standard_normal((self.config.latent_dim, self.obs_dim)) / np.sqrt(self.obs_dim)
        return np.tanh(W @ obs)

    def plan(self, obs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the best action from a random MPPI sample (placeholder).

        Parameters
        ----------
        obs:
            Observation vector, shape ``(obs_dim,)``.

        Returns
        -------
        NDArray[np.float64]
            Planned action, shape ``(act_dim,)``.
        """
        rng = np.random.default_rng(self._total_steps)
        # Sample random action sequences and pick by softmax-weighted mean
        samples = rng.standard_normal((self.config.n_samples, self.act_dim))
        weights = np.ones(self.config.n_samples) / self.config.n_samples
        action = (weights[:, None] * samples).sum(axis=0)
        self._total_steps += 1
        return action

    def build_agent(self) -> Any:
        """Instantiate the real TD-MPC2 agent.

        Raises
        ------
        ImportError
            When ``tdmpc2`` is not installed.
        """
        try:
            import tdmpc2  # type: ignore[import]  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "tdmpc2 is not installed.  See https://github.com/nicklashansen/tdmpc2."
            ) from exc
        raise NotImplementedError("Full TD-MPC2 integration is a work in progress.")

    @property
    def total_steps(self) -> int:
        """Total planning steps executed."""
        return self._total_steps
