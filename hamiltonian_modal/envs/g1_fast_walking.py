"""G1 fast-walking environment."""
import logging
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from hamiltonian_modal.envs.g1_base import G1BaseEnv, EnvConfig, G1_N_DOFS

logger = logging.getLogger(__name__)


@dataclass
class FastWalkingConfig(EnvConfig):
    """Configuration for the fast-walking task.

    Parameters
    ----------
    target_speed : float — desired forward speed (m/s); higher than walking
    max_steps : int — maximum episode length
    alive_bonus : float — per-step bonus for staying upright
    speed_tolerance : float — speed window around target that is considered "good"
    """
    target_speed: float = 2.5
    max_steps: int = 800
    alive_bonus: float = 0.2
    speed_tolerance: float = 0.2


class G1FastWalkingEnv(G1BaseEnv):
    """G1 fast-walking / running environment.

    Higher target speed than G1WalkingEnv. Penalises slow locomotion more
    aggressively and rewards smooth, high-cadence gaits.

    Parameters
    ----------
    config : FastWalkingConfig or None
    """

    def __init__(self, config: "FastWalkingConfig | None" = None) -> None:
        cfg = config or FastWalkingConfig()
        super().__init__(cfg)
        self._fast_config = cfg
        self._prev_x = 0.0

    def reset(self) -> NDArray[np.float64]:
        obs = super().reset()
        self._prev_x = float(self._state[0])
        return obs

    def _compute_reward(self, action: NDArray[np.float64]) -> float:
        """Reward = speed match + alive bonus - lateral drift - action penalty."""
        x = float(self._state[0])
        forward_speed = (x - self._prev_x) / self.config.dt
        self._prev_x = x

        # Quadratic penalty inside tolerance window, linear outside
        speed_err = abs(forward_speed - self._fast_config.target_speed)
        if speed_err < self._fast_config.speed_tolerance:
            speed_reward = 1.0 - (speed_err / self._fast_config.speed_tolerance) ** 2
        else:
            speed_reward = -(speed_err - self._fast_config.speed_tolerance)

        alive_bonus = self._fast_config.alive_bonus if not self._is_terminated() else -2.0
        lateral_penalty = -0.2 * float(abs(self._state[1]))
        action_penalty = -0.002 * float(np.sum(action ** 2))

        return float(speed_reward + alive_bonus + lateral_penalty + action_penalty)
