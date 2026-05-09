"""RoboScape baseline adapter for hamiltonian-modal G1 environments.

RoboScape is a model-free RL baseline operating in a scene-graph physics
environment.  This adapter wraps G1 observations into scene descriptors
expected by RoboScape's policy network.

No RoboScape installation is required to import this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["RoboScapeConfig", "RoboScapeAdapter"]


@dataclass
class RoboScapeConfig:
    """Configuration for RoboScape integration.

    Attributes
    ----------
    scene_embed_dim:
        Dimension of the scene-graph embedding.
    policy_hidden_dim:
        Hidden layer width for the scene-conditioned policy.
    lr:
        Learning rate.
    gamma:
        Discount factor.
    """

    scene_embed_dim: int = 128
    policy_hidden_dim: int = 256
    lr: float = 3e-4
    gamma: float = 0.99


class RoboScapeAdapter:
    """Thin shim between G1 environments and RoboScape.

    Parameters
    ----------
    obs_dim:
        Observation dimension.
    act_dim:
        Action dimension.
    config:
        :class:`RoboScapeConfig`.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        config: RoboScapeConfig | None = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config = config or RoboScapeConfig()

    def embed_observation(self, obs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project *obs* into the scene-embedding space.

        This placeholder uses a random projection matrix seeded at
        construction time.

        Parameters
        ----------
        obs:
            Observation vector, shape ``(obs_dim,)``.

        Returns
        -------
        NDArray[np.float64]
            Scene embedding, shape ``(scene_embed_dim,)``.
        """
        obs = np.asarray(obs, dtype=np.float64)
        if obs.shape != (self.obs_dim,):
            raise ValueError(f"obs must have shape ({self.obs_dim},), got {obs.shape}.")
        rng = np.random.default_rng(0)
        W = rng.standard_normal((self.config.scene_embed_dim, self.obs_dim)) / np.sqrt(self.obs_dim)
        return W @ obs

    def act(self, obs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return a zero action (placeholder).

        Parameters
        ----------
        obs:
            Observation vector, shape ``(obs_dim,)``.

        Returns
        -------
        NDArray[np.float64]
            Zero action, shape ``(act_dim,)``.
        """
        return np.zeros(self.act_dim, dtype=np.float64)

    def build_agent(self) -> Any:
        """Instantiate the real RoboScape agent.

        Raises
        ------
        ImportError
            When the ``roboscape`` package is not installed.
        """
        try:
            import roboscape  # type: ignore[import]  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "roboscape is not installed.  See the RoboScape project repository."
            ) from exc
        raise NotImplementedError("Full RoboScape integration is a work in progress.")
