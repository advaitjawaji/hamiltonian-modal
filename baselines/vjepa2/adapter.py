"""VJEPA2 baseline adapter for hamiltonian-modal G1 environments.

VJEPA2 (Meta AI, 2024) is a video-prediction world model based on Joint-
Embedding Predictive Architecture.  This adapter maps proprioceptive G1
observations to the patch-token format expected by VJEPA2's encoder.

No VJEPA2 installation is required to import this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["VJEPA2Config", "VJEPA2Adapter"]


@dataclass
class VJEPA2Config:
    """Minimal VJEPA2 configuration.

    Attributes
    ----------
    embed_dim:
        Patch-token embedding dimension.
    n_context_frames:
        Number of past frames used as JEPA context.
    n_target_frames:
        Number of future frames predicted by JEPA.
    lr:
        Learning rate for the predictor.
    """

    embed_dim: int = 384
    n_context_frames: int = 8
    n_target_frames: int = 4
    lr: float = 1e-4


class VJEPA2Adapter:
    """Thin shim between G1 proprioceptive observations and VJEPA2.

    Proprioceptive observations are treated as 1-D "tokens" and linearly
    projected into the VJEPA2 embedding space.

    Parameters
    ----------
    obs_dim:
        Observation dimension.
    act_dim:
        Action dimension.
    config:
        :class:`VJEPA2Config`.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        config: VJEPA2Config | None = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config = config or VJEPA2Config()
        rng = np.random.default_rng(0)
        # Fixed random projection: obs_dim → embed_dim
        self._proj = rng.standard_normal(
            (self.config.embed_dim, obs_dim)
        ) / np.sqrt(obs_dim)
        self._context: list[NDArray[np.float64]] = []

    def encode_frame(self, obs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project a single proprioceptive observation into the token space.

        Parameters
        ----------
        obs:
            Observation vector, shape ``(obs_dim,)``.

        Returns
        -------
        NDArray[np.float64]
            Token embedding, shape ``(embed_dim,)``.
        """
        obs = np.asarray(obs, dtype=np.float64)
        if obs.shape != (self.obs_dim,):
            raise ValueError(f"obs must have shape ({self.obs_dim},), got {obs.shape}.")
        token = np.tanh(self._proj @ obs)
        self._context.append(token)
        if len(self._context) > self.config.n_context_frames:
            self._context.pop(0)
        return token

    def predict_next_tokens(self) -> NDArray[np.float64]:
        """Predict the next *n_target_frames* tokens from the current context.

        Returns a zero-filled prediction when there is insufficient context.

        Returns
        -------
        NDArray[np.float64]
            Predicted tokens, shape ``(n_target_frames, embed_dim)``.
        """
        n_target = self.config.n_target_frames
        embed_dim = self.config.embed_dim
        if not self._context:
            return np.zeros((n_target, embed_dim), dtype=np.float64)
        # Simple autoregressive placeholder: mean of context, replicated.
        ctx = np.stack(self._context, axis=0)  # (T, embed_dim)
        mean_token = ctx.mean(axis=0)
        return np.tile(mean_token, (n_target, 1))

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
        """Instantiate the real VJEPA2 agent.

        Raises
        ------
        ImportError
            When the ``vjepa2`` package is not installed.
        """
        try:
            import vjepa2  # type: ignore[import]  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "vjepa2 is not installed.  See https://github.com/facebookresearch/vjepa."
            ) from exc
        raise NotImplementedError("Full VJEPA2 integration is a work in progress.")
