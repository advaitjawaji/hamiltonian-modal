"""G1 jumping environment."""
import logging
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from hamiltonian_modal.envs.g1_base import G1BaseEnv, EnvConfig, G1_N_DOFS

logger = logging.getLogger(__name__)


@dataclass
class JumpingConfig(EnvConfig):
    """Configuration for the jumping task.

    Parameters
    ----------
    target_height : float — desired jump apex height above ground (m)
    max_steps : int — maximum episode length
    air_time_bonus : float — bonus per step while airborne
    landing_bonus : float — bonus for landing without falling
    """
    target_height: float = 0.3
    max_steps: int = 200
    air_time_bonus: float = 0.5
    landing_bonus: float = 2.0


class G1JumpingEnv(G1BaseEnv):
    """G1 jumping environment.

    The robot is rewarded for achieving a target jump apex height and
    landing safely.

    Parameters
    ----------
    config : JumpingConfig or None
    """

    def __init__(self, config: "JumpingConfig | None" = None) -> None:
        cfg = config or JumpingConfig()
        super().__init__(cfg)
        self._jump_config = cfg
        self._was_airborne = False
        self._apex_height = 0.0
        self._phase = "crouch"  # "crouch" → "air" → "landing"

    def reset(self) -> NDArray[np.float64]:
        obs = super().reset()
        self._was_airborne = False
        self._apex_height = 0.0
        self._phase = "crouch"
        return obs

    def _is_airborne(self) -> bool:
        """Robot is airborne when height is clearly above standing height."""
        return float(self._state[2]) > 0.9

    def step(
        self,
        action: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float, bool, dict]:
        obs, reward, done, info = super().step(action)

        airborne = self._is_airborne()
        height = float(self._state[2])

        if airborne:
            self._apex_height = max(self._apex_height, height)
            self._phase = "air"
        elif self._phase == "air" and not airborne:
            self._phase = "landing"

        info["phase"] = self._phase
        info["apex_height"] = self._apex_height
        info["airborne"] = airborne

        return obs, reward, done, info

    def _compute_reward(self, action: NDArray[np.float64]) -> float:
        """Reward = apex height match + air-time bonus + landing bonus - action penalty."""
        if self._is_terminated():
            return -3.0

        height = float(self._state[2])
        action_penalty = -0.001 * float(np.sum(action ** 2))

        if self._phase == "air":
            # Reward proportional to height achieved
            height_reward = min(height / (0.8 + self._jump_config.target_height), 1.5)
            return float(self._jump_config.air_time_bonus + height_reward + action_penalty)

        if self._phase == "landing":
            # One-time landing bonus scaled by apex height matching
            apex_err = abs(self._apex_height - (0.8 + self._jump_config.target_height))
            landing_bonus = self._jump_config.landing_bonus * max(0.0, 1.0 - apex_err)
            self._phase = "crouch"  # Reset for next jump
            return float(landing_bonus + action_penalty)

        # Crouch phase: encourage preparatory stance
        upright_penalty = -abs(height - 0.8)
        return float(0.05 + upright_penalty + action_penalty)
