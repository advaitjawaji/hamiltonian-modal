"""Base G1 environment wrapping Genesis."""
import logging
from dataclasses import dataclass
from typing import Any
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

G1_N_DOFS = 23
G1_OBS_DIM = 46  # q + qdot


@dataclass
class EnvConfig:
    """Base environment configuration.

    Parameters
    ----------
    dt : float — simulation timestep in seconds
    max_steps : int — maximum episode steps
    n_envs : int — number of parallel environments
    requires_grad : bool — enable gradient computation (Genesis only)
    """
    dt: float = 0.01
    max_steps: int = 1000
    n_envs: int = 1
    requires_grad: bool = False


class G1BaseEnv:
    """Base Unitree G1 environment.

    Provides a numpy-only fallback when Genesis is not available.
    The state vector is [q (23), qdot (23)] with q[2] = height.

    Parameters
    ----------
    config : EnvConfig
    """

    def __init__(self, config: "EnvConfig | None" = None) -> None:
        self.config = config or EnvConfig()
        self.n_dofs = G1_N_DOFS
        self.obs_dim = G1_OBS_DIM
        self.action_dim = G1_N_DOFS
        self._step_count = 0
        self._state = np.zeros(G1_OBS_DIM, dtype=np.float64)
        self._scene: Any = None

    def _try_init_genesis(self) -> None:
        """Attempt to initialise the Genesis physics scene."""
        try:
            from hamiltonian_modal.utils.genesis_wrapper import make_g1_scene
            self._scene = make_g1_scene(
                requires_grad=self.config.requires_grad,
                dt=self.config.dt,
                n_envs=self.config.n_envs,
            )
        except ImportError:
            logger.warning("Genesis not available; running in numpy-only mode")

    def reset(self) -> NDArray[np.float64]:
        """Reset the environment and return initial observation.

        Returns
        -------
        obs : shape (obs_dim,)
        """
        self._step_count = 0
        self._state = np.zeros(self.obs_dim, dtype=np.float64)
        self._state[2] = 0.8  # standing height
        return self._state.copy()

    def step(
        self,
        action: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float, bool, dict]:
        """Step the environment by applying an action.

        Parameters
        ----------
        action : shape (action_dim,) — joint torques or position targets

        Returns
        -------
        obs : shape (obs_dim,)
        reward : float
        done : bool
        info : dict
        """
        self._step_count += 1
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        # Integrate position using forward Euler
        self._state[:G1_N_DOFS] += self.config.dt * self._state[G1_N_DOFS:]
        # Apply action as velocity change
        self._state[G1_N_DOFS:] += self.config.dt * action
        obs = self._state.copy()
        reward = self._compute_reward(action)
        done = self._step_count >= self.config.max_steps or self._is_terminated()
        return obs, reward, done, {"step": self._step_count}

    def _compute_reward(self, action: NDArray[np.float64]) -> float:
        """Compute the step reward. Override in subclasses."""
        return 0.0

    def _is_terminated(self) -> bool:
        """Return True if the robot has fallen (height below threshold)."""
        height = float(self._state[2]) if len(self._state) > 2 else 0.8
        return height < 0.3

    @property
    def observation_space_dim(self) -> int:
        """Observation space dimensionality."""
        return self.obs_dim

    @property
    def action_space_dim(self) -> int:
        """Action space dimensionality."""
        return self.action_dim
