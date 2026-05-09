"""PPO-Genesis adapter for hamiltonian-modal G1 environments.

PPO-Genesis is Proximal Policy Optimisation (Schulman et al., 2017) running
inside the Genesis physics simulator.  This adapter wraps a G1 environment so
that it presents the observation / action tensors expected by a Genesis-based
PPO training loop.

No Genesis installation is required to import this module.  The
:meth:`build_agent` method raises :exc:`ImportError` when Genesis is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["PPOGenesisConfig", "PPOGenesisAdapter"]


@dataclass
class PPOGenesisConfig:
    """Minimal PPO hyperparameters for Genesis runs.

    Attributes
    ----------
    n_envs:
        Number of parallel environments.
    n_steps:
        Steps collected per environment per update.
    n_epochs:
        Number of SGD epochs per PPO update.
    clip_range:
        PPO clipping parameter ε.
    lr:
        Learning rate.
    gamma:
        Discount factor.
    gae_lambda:
        GAE λ parameter.
    """

    n_envs: int = 4
    n_steps: int = 2048
    n_epochs: int = 10
    clip_range: float = 0.2
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95


class PPOGenesisAdapter:
    """Thin shim between G1 environments and PPO-Genesis.

    Parameters
    ----------
    obs_dim:
        Observation dimension.
    act_dim:
        Action dimension.
    config:
        :class:`PPOGenesisConfig`.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        config: PPOGenesisConfig | None = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config = config or PPOGenesisConfig()
        self._episode_count: int = 0

    def reset(self) -> NDArray[np.float64]:
        """Return a zero observation (placeholder).

        Returns
        -------
        NDArray[np.float64]
            Zero observation, shape ``(obs_dim,)``.
        """
        self._episode_count += 1
        return np.zeros(self.obs_dim, dtype=np.float64)

    def act(self, obs: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
        """Return a random action and its log-probability.

        Parameters
        ----------
        obs:
            Observation vector, shape ``(obs_dim,)``.

        Returns
        -------
        tuple[NDArray, float]
            ``(action, log_prob)``.
        """
        action = np.zeros(self.act_dim, dtype=np.float64)
        log_prob = -0.5 * self.act_dim * float(np.log(2 * np.pi))
        return action, log_prob

    def build_agent(self) -> Any:
        """Instantiate the real PPO-Genesis agent.

        Raises
        ------
        ImportError
            When ``genesis`` is not installed.
        """
        try:
            import genesis  # type: ignore[import]  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "genesis is not installed.  See https://github.com/Genesis-Embodied-AI/Genesis."
            ) from exc
        raise NotImplementedError("Full PPO-Genesis integration is a work in progress.")

    @property
    def episode_count(self) -> int:
        """Total number of episodes started."""
        return self._episode_count
