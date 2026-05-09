"""G1 push-recovery environment."""
import logging
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from hamiltonian_modal.envs.g1_base import G1BaseEnv, EnvConfig, G1_N_DOFS

logger = logging.getLogger(__name__)


@dataclass
class PushRecoveryConfig(EnvConfig):
    """Configuration for the push-recovery task.

    Parameters
    ----------
    push_interval : int — steps between random pushes
    push_magnitude : float — maximum impulse magnitude (N·s)
    max_steps : int — maximum episode length
    alive_bonus : float — per-step bonus for remaining upright
    recovery_bonus : float — bonus after recovering from a push
    """
    push_interval: int = 100
    push_magnitude: float = 50.0
    max_steps: int = 500
    alive_bonus: float = 0.2
    recovery_bonus: float = 0.5


class G1PushRecoveryEnv(G1BaseEnv):
    """G1 push-recovery environment.

    Random impulse pushes are applied every ``push_interval`` steps.
    The robot must stay upright and return to the upright stance quickly.

    Parameters
    ----------
    config : PushRecoveryConfig or None
    """

    def __init__(self, config: "PushRecoveryConfig | None" = None) -> None:
        cfg = config or PushRecoveryConfig()
        super().__init__(cfg)
        self._push_config = cfg
        self._rng = np.random.default_rng(42)
        self._last_push_step = -1
        self._recovering = False

    def reset(self) -> NDArray[np.float64]:
        obs = super().reset()
        self._last_push_step = -1
        self._recovering = False
        return obs

    def step(
        self,
        action: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float, bool, dict]:
        # Apply random push if scheduled
        if (self._step_count % self._push_config.push_interval == 0
                and self._step_count > 0):
            self._apply_push()
            self._recovering = True
            self._last_push_step = self._step_count

        obs, reward, done, info = super().step(action)
        info["recovering"] = self._recovering
        info["last_push_step"] = self._last_push_step
        return obs, reward, done, info

    def _apply_push(self) -> None:
        """Apply a random horizontal impulse to the robot."""
        direction = self._rng.standard_normal(3)
        direction[2] = 0.0  # horizontal only
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction = direction / norm
        magnitude = self._push_config.push_magnitude * self._rng.uniform(0.3, 1.0)
        # Apply as velocity change to the base (first 3 velocity DOFs)
        self._state[G1_N_DOFS : G1_N_DOFS + 3] += magnitude * direction

    def _compute_reward(self, action: NDArray[np.float64]) -> float:
        """Reward = alive bonus + recovery bonus - action penalty."""
        if self._is_terminated():
            return -5.0

        alive_bonus = self._push_config.alive_bonus

        # Check if robot has recovered (velocities small, height near nominal)
        base_vel = float(np.linalg.norm(self._state[G1_N_DOFS : G1_N_DOFS + 3]))
        height_err = abs(float(self._state[2]) - 0.8)
        recovery_bonus = 0.0
        if self._recovering and base_vel < 0.3 and height_err < 0.1:
            recovery_bonus = self._push_config.recovery_bonus
            self._recovering = False

        action_penalty = -0.001 * float(np.sum(action ** 2))
        upright_bonus = max(0.0, 1.0 - height_err)

        return float(alive_bonus + recovery_bonus + action_penalty + 0.1 * upright_bonus)
