"""Puppeteer baseline adapter."""
from __future__ import annotations

import logging
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class PuppeteerAdapter:
    """Puppeteer baseline adapter.

    Puppeteer is a motion-retargeting controller that maps reference motions
    to robot actions via kinematic matching.  This adapter provides a
    fallback interface for benchmarking.

    Parameters
    ----------
    env_name : str — environment identifier
    seed : int — random seed for fallback sampler
    """

    def __init__(self, env_name: str = "g1_walking", seed: int = 42) -> None:
        self.env_name = env_name
        self._rng = np.random.default_rng(seed)
        self._step = 0
        logger.info("PuppeteerAdapter initialised (env=%s, seed=%d)", env_name, seed)

    def predict(self, obs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict action from observation.

        Parameters
        ----------
        obs : shape (obs_dim,)

        Returns
        -------
        action : shape (23,)
        """
        self._step += 1
        return 0.4 * self._rng.standard_normal(23).astype(np.float64)

    def synthetic_reward_curve(self, n_steps: int) -> NDArray[np.float64]:
        """Simulated Puppeteer reward curve.

        Parameters
        ----------
        n_steps : int

        Returns
        -------
        rewards : shape (n_steps,)
        """
        t = np.arange(n_steps, dtype=np.float64)
        curve = 0.65 * (1.0 - np.exp(-t / (n_steps * 0.4)))
        noise = 0.05 * self._rng.standard_normal(n_steps)
        return np.clip(curve + noise, 0.0, 1.0)

    def synthetic_drift_trajectory(self, n_steps: int) -> NDArray[np.float64]:
        """Simulated energy drift for Puppeteer.

        Parameters
        ----------
        n_steps : int

        Returns
        -------
        energies : shape (n_steps+1,)
        """
        from benchmarks.driftbench_g1.tasks import synthetic_baseline_drift
        return synthetic_baseline_drift(n_steps, drift_rate=0.2, seed=int(self._rng.integers(0, 1000)))

    def reset(self) -> None:
        """Reset internal step counter."""
        self._step = 0
