"""G1 disturbance recovery environment.

G1 traverses 10m to target zone. At t=4s, obstacle appears. Robot must replan.
"""
import logging
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from hamiltonian_modal.envs.g1_base import G1BaseEnv, EnvConfig

logger = logging.getLogger(__name__)


@dataclass
class DisturbanceConfig(EnvConfig):
    """Configuration for the disturbance-recovery task.

    Parameters
    ----------
    target_distance : float — total distance to goal (m)
    disturbance_time : float — time at which obstacle appears (s)
    obstacle_radius : float — collision radius of obstacle (m)
    max_steps : int — maximum episode length
    """
    target_distance: float = 10.0
    disturbance_time: float = 4.0
    obstacle_radius: float = 0.5
    max_steps: int = 1500


class G1DisturbanceEnv(G1BaseEnv):
    """Long-horizon traversal with mid-path obstacle disturbance.

    The robot starts at the origin and must walk 10m forward to a target
    zone.  At t=4s a cylindrical obstacle materialises at the midpoint of the
    path.  The robot must detect the obstacle and plan an avoidance trajectory.

    Parameters
    ----------
    config : DisturbanceConfig or None
    """

    def __init__(self, config: "DisturbanceConfig | None" = None) -> None:
        cfg = config or DisturbanceConfig()
        super().__init__(cfg)
        self._dist_config = cfg
        self._obstacle_active = False
        self._obstacle_pos = np.array([5.0, 0.0, 0.0], dtype=np.float64)
        self._target = np.array([self._dist_config.target_distance, 0.0, 0.0], dtype=np.float64)

    def reset(self) -> NDArray[np.float64]:
        """Reset environment, clearing obstacle state."""
        obs = super().reset()
        self._obstacle_active = False
        return obs

    def step(
        self,
        action: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float, bool, dict]:
        """Step environment; activate obstacle at disturbance_time."""
        obs, reward, done, info = super().step(action)
        t = self._step_count * self.config.dt
        if t >= self._dist_config.disturbance_time:
            self._obstacle_active = True
        info["obstacle_active"] = self._obstacle_active
        info["target"] = self._target.copy()
        info["obstacle_pos"] = self._obstacle_pos.copy()
        return obs, reward, done, info

    def _compute_reward(self, action: NDArray[np.float64]) -> float:
        """Shaped reward: progress to goal minus obstacle-collision penalty."""
        pos = self._state[:3]
        dist_to_target = float(np.linalg.norm(pos - self._target))

        # Progress reward: negative normalised distance
        progress_reward = -dist_to_target / self._dist_config.target_distance

        # Obstacle collision penalty
        collision_penalty = 0.0
        if self._obstacle_active:
            dist_to_obs = float(np.linalg.norm(pos - self._obstacle_pos))
            if dist_to_obs < self._dist_config.obstacle_radius:
                collision_penalty = -2.0

        # Small action-regularisation penalty
        action_penalty = -0.001 * float(np.sum(action ** 2))

        # Success bonus for reaching target
        success_bonus = 1.0 if dist_to_target < 0.5 else 0.0

        return float(progress_reward + collision_penalty + action_penalty + success_bonus)

    def reached_target(self) -> bool:
        """Return True if the robot is within success radius of the target."""
        pos = self._state[:3]
        return float(np.linalg.norm(pos - self._target)) < 0.5

    @property
    def obstacle_active(self) -> bool:
        """Whether the obstacle is currently present."""
        return self._obstacle_active

    @property
    def obstacle_position(self) -> NDArray[np.float64]:
        """Current obstacle position."""
        return self._obstacle_pos.copy()
