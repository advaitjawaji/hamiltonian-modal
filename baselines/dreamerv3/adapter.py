"""DreamerV3 adapter for hamiltonian-modal G1 environments.

DreamerV3 (Hafner et al., 2023) is a model-based RL agent using a recurrent
world model with discrete latent states.  This adapter provides a minimal
shim so that G1 environments can be wrapped for use with DreamerV3 without
modifying the baseline's source code.

The adapter is intentionally dependency-free: DreamerV3 itself need not be
installed for this module to import.  The :class:`DreamerV3Adapter` raises a
clear :exc:`ImportError` only when :meth:`build_agent` is called and DreamerV3
is absent from the Python path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["DreamerV3Config", "DreamerV3Adapter"]


@dataclass
class DreamerV3Config:
    """Minimal DreamerV3 hyperparameters exposed to hamiltonian-modal.

    Attributes
    ----------
    batch_size:
        Replay-batch size.
    batch_length:
        Sequence length per batch.
    model_lr:
        World-model learning rate.
    actor_lr:
        Actor learning rate.
    critic_lr:
        Critic learning rate.
    imagination_horizon:
        Rollout length inside the world model.
    """

    batch_size: int = 16
    batch_length: int = 64
    model_lr: float = 1e-4
    actor_lr: float = 3e-5
    critic_lr: float = 3e-5
    imagination_horizon: int = 15


class DreamerV3Adapter:
    """Thin shim between G1 environments and DreamerV3.

    Parameters
    ----------
    obs_dim:
        Observation dimension of the wrapped G1 environment.
    act_dim:
        Action dimension.
    config:
        :class:`DreamerV3Config`.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        config: DreamerV3Config | None = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config = config or DreamerV3Config()
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Minimal environment-facing API
    # ------------------------------------------------------------------

    def observe(self, obs: NDArray[np.float64], reward: float, done: bool) -> None:
        """Record a new environment transition.

        Parameters
        ----------
        obs:
            Observation vector, shape ``(obs_dim,)``.
        reward:
            Scalar reward.
        done:
            Episode termination flag.
        """
        self._step_count += 1

    def act(self, obs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return a random action (placeholder — real agent not installed).

        Parameters
        ----------
        obs:
            Observation vector, shape ``(obs_dim,)``.

        Returns
        -------
        NDArray[np.float64]
            Action vector, shape ``(act_dim,)``.
        """
        return np.zeros(self.act_dim, dtype=np.float64)

    def build_agent(self) -> Any:
        """Instantiate the real DreamerV3 agent.

        Raises
        ------
        ImportError
            When ``dreamerv3`` is not installed.
        """
        try:
            import dreamerv3  # type: ignore[import]  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "dreamerv3 is not installed.  See https://github.com/danijar/dreamerv3."
            ) from exc
        raise NotImplementedError("Full DreamerV3 integration is a work in progress.")

    @property
    def step_count(self) -> int:
        """Total number of transitions observed."""
        return self._step_count
