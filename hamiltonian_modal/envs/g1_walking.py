"""G1 walking environment."""
import logging
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from hamiltonian_modal.envs.g1_base import G1BaseEnv, EnvConfig, G1_N_DOFS

logger = logging.getLogger(__name__)


@dataclass
class WalkingConfig(EnvConfig):
    """Configuration for the walking task.

    Parameters
    ----------
    target_speed : float — desired forward walking speed in m/s
    max_steps : int — maximum episode length
    alive_bonus : float — per-step bonus for staying upright
    """
    target_speed: float = 1.0
    max_steps: int = 1000
    alive_bonus: float = 0.1


class G1WalkingEnv(G1BaseEnv):
    """G1 bipedal walking environment.

    Reward encourages forward locomotion at the target speed while
    penalising falls, excessive torque, and lateral drift.

    Parameters
    ----------
    config : WalkingConfig or None
    """

    def __init__(self, config: "WalkingConfig | None" = None) -> None:
        cfg = config or WalkingConfig()
        super().__init__(cfg)
        self._walk_config = cfg
        self._prev_x = 0.0

    def reset(self) -> NDArray[np.float64]:
        obs = super().reset()
        self._prev_x = float(self._state[0])
        return obs

    def _compute_reward(self, action: NDArray[np.float64]) -> float:
        """Reward = forward speed + alive bonus - lateral drift - action penalty."""
        x = float(self._state[0])
        forward_speed = (x - self._prev_x) / self.config.dt
        self._prev_x = x

        speed_reward = -abs(forward_speed - self._walk_config.target_speed)
        alive_bonus = self._walk_config.alive_bonus if not self._is_terminated() else -1.0
        lateral_penalty = -0.1 * float(abs(self._state[1]))
        action_penalty = -0.001 * float(np.sum(action ** 2))
        return float(speed_reward + alive_bonus + lateral_penalty + action_penalty)
