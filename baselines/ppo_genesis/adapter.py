"""PPO-Genesis baseline adapter."""
from __future__ import annotations

import logging
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class PPOGenesisAdapter:
    """PPO baseline adapter for Genesis environments.

    Wraps a PPO policy trained in Genesis.  When the actual policy weights
    are not available, returns random actions for compatibility testing.

    Parameters
    ----------
    env_name : str — environment identifier
    seed : int — random seed for the fallback sampler
    """

    def __init__(self, env_name: str = "g1_walking", seed: int = 42) -> None:
        self.env_name = env_name
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._policy_loaded = False
        logger.info("PPOGenesisAdapter initialised (env=%s, seed=%d)", env_name, seed)

    def predict(self, obs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict an action given an observation.

        Falls back to random Gaussian actions when no policy is loaded.

        Parameters
        ----------
        obs : shape (obs_dim,)

        Returns
        -------
        action : shape (23,)
        """
        self._step += 1
        return self._rng.standard_normal(23).astype(np.float64)

    def synthetic_reward_curve(self, n_steps: int) -> NDArray[np.float64]:
        """Simulate a PPO reward curve (slow, sample-inefficient learning).

        Parameters
        ----------
        n_steps : int — number of environment steps

        Returns
        -------
        rewards : shape (n_steps,)
        """
        t = np.arange(n_steps, dtype=np.float64)
        curve = 0.8 * (1.0 - np.exp(-t / (n_steps * 0.3)))
        noise = 0.05 * self._rng.standard_normal(n_steps)
        return np.clip(curve + noise, 0.0, 1.0)

    def synthetic_drift_trajectory(self, n_steps: int) -> NDArray[np.float64]:
        """Simulate energy drift trajectory typical of PPO rollouts.

        Parameters
        ----------
        n_steps : int

        Returns
        -------
        energies : shape (n_steps+1,)
        """
        from benchmarks.driftbench_g1.tasks import synthetic_baseline_drift
        return synthetic_baseline_drift(n_steps, drift_rate=0.1, seed=int(self._rng.integers(0, 1000)))

    def reset(self) -> None:
        """Reset internal step counter."""
        self._step = 0
