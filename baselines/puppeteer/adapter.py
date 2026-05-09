"""Puppeteer baseline adapter for hamiltonian-modal G1 environments.

Puppeteer is a motion-retargeting and imitation-learning baseline that
drives a humanoid robot by tracking reference motion clips.  This adapter
converts a :class:`~hamiltonian_modal.data.retargeted_motions.MotionClip`
into Puppeteer-compatible reference frames.

No Puppeteer installation is required to import this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["PuppeteerConfig", "PuppeteerAdapter"]


@dataclass
class PuppeteerConfig:
    """Configuration for Puppeteer motion-imitation.

    Attributes
    ----------
    tracking_weight:
        Weight of the motion-tracking reward term.
    style_weight:
        Weight of the style (naturalness) reward term.
    reference_fps:
        Frame rate of the reference motion clip.
    """

    tracking_weight: float = 0.7
    style_weight: float = 0.3
    reference_fps: float = 30.0


class PuppeteerAdapter:
    """Thin shim between G1 environments and Puppeteer.

    Parameters
    ----------
    obs_dim:
        Observation dimension.
    act_dim:
        Action dimension.
    n_dof:
        Number of joint DOF (used to validate reference frames).
    config:
        :class:`PuppeteerConfig`.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        n_dof: int = 23,
        config: PuppeteerConfig | None = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_dof = n_dof
        self.config = config or PuppeteerConfig()
        self._reference_frames: NDArray[np.float64] | None = None

    def set_reference_clip(self, positions: NDArray[np.float64]) -> None:
        """Load a reference motion clip.

        Parameters
        ----------
        positions:
            Joint positions, shape ``(T, n_dof)``.
        """
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != self.n_dof:
            raise ValueError(
                f"positions must have shape (T, {self.n_dof}), got {positions.shape}."
            )
        self._reference_frames = positions

    def tracking_reward(
        self,
        q_current: NDArray[np.float64],
        frame_idx: int,
    ) -> float:
        """Compute pose-tracking reward for *frame_idx*.

        Parameters
        ----------
        q_current:
            Current joint positions, shape ``(n_dof,)``.
        frame_idx:
            Reference frame index.

        Returns
        -------
        float
            Negative L2 distance to reference pose (higher is better).
        """
        if self._reference_frames is None:
            raise RuntimeError("No reference clip loaded. Call set_reference_clip first.")
        idx = int(frame_idx) % len(self._reference_frames)
        ref_q = self._reference_frames[idx]
        diff = q_current - ref_q
        return -float(np.dot(diff, diff))

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
        """Instantiate the real Puppeteer agent.

        Raises
        ------
        ImportError
            When the ``puppeteer`` package is not installed.
        """
        try:
            import puppeteer  # type: ignore[import]  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "puppeteer is not installed.  See the Puppeteer project repository."
            ) from exc
        raise NotImplementedError("Full Puppeteer integration is a work in progress.")
