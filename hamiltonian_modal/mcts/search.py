"""PUCT MCTS operating in the Hamiltonian-modal world model."""
import logging
from dataclasses import dataclass
from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray
from hamiltonian_modal.mcts.tree import TreeNode
from hamiltonian_modal.mcts.puct import select_leaf

logger = logging.getLogger(__name__)


@dataclass
class MCTSConfig:
    """Configuration for MCTS search.

    Parameters
    ----------
    branching : int — action branching factor
    depth : int — maximum rollout depth
    n_simulations : int — number of MCTS simulations
    c_puct : float — PUCT exploration constant
    rollout_temperature : float — temperature for action sampling
    """
    branching: int = 8
    depth: int = 50
    n_simulations: int = 50
    c_puct: float = 1.5
    rollout_temperature: float = 1.0


@dataclass
class ModalState:
    """Modal state tuple.

    Attributes
    ----------
    eta : modal displacement coordinates, shape (n_modes,)
    eta_dot : modal velocity coordinates, shape (n_modes,)
    contact_mode : int — discrete contact mode index
    """
    eta: NDArray[np.float64]
    eta_dot: NDArray[np.float64]
    contact_mode: int = 0


@dataclass
class Goal:
    """Goal specification for MCTS.

    Attributes
    ----------
    target_position : NDArray — 3-D target Cartesian position
    success_radius : float — distance threshold for success
    """
    target_position: NDArray[np.float64]
    success_radius: float = 0.5


@dataclass
class ActionTrajectory:
    """Planned action sequence from MCTS.

    Attributes
    ----------
    actions : list of actions along the best path
    expected_value : root Q-value estimate
    search_depth_reached : depth reached during best-path extraction
    """
    actions: list[NDArray[np.float64]]
    expected_value: float
    search_depth_reached: int


class MCTSSearch:
    """PUCT MCTS in the Hamiltonian-modal world model.

    Selection: PUCT(s, a) = Q(s, a) + c · P(s, a) · √N(s) / (1 + N(s, a))
    Expansion: roll out world model H_θ for ``depth`` steps via leapfrog
    Value: verifier head V(η, η̇, m, goal)
    Backup: standard MCTS value backup

    Parameters
    ----------
    world_model : HamiltonianNet — learned Hamiltonian neural network
    verifier : callable — value estimator V(η, η̇, m, target) → VerifierOut
    policy : callable — maps state → (mean, cov) action distribution
    config : MCTSConfig
    action_dim : int — dimensionality of the action space
    """

    def __init__(
        self,
        world_model: Any,
        verifier: Any,
        policy: Callable[[NDArray[np.float64]], tuple[NDArray[np.float64], NDArray[np.float64]]],
        config: "MCTSConfig | None" = None,
        action_dim: int = 23,
    ) -> None:
        self.world_model = world_model
        self.verifier = verifier
        self.policy = policy
        self.config = config or MCTSConfig()
        self.action_dim = action_dim
        self._rng = np.random.default_rng(0)

    def _sample_actions(
        self,
        state: NDArray[np.float64],
        n: int,
    ) -> list[NDArray[np.float64]]:
        """Sample n candidate actions from the policy distribution."""
        mean, cov = self.policy(state)
        actions = []
        for _ in range(n):
            noise = self._rng.standard_normal(len(mean))
            a = mean + np.sqrt(np.maximum(np.diag(cov), 0.0)) * noise * self.config.rollout_temperature
            actions.append(a)
        return actions

    def _simulate(
        self,
        node: TreeNode,
        goal: Goal,
        depth: int,
    ) -> float:
        """Roll out world model from node for ``depth`` steps and return value.

        Parameters
        ----------
        node : leaf node to simulate from
        goal : Goal specification
        depth : number of world-model steps to roll out

        Returns
        -------
        value : float — estimated value from verifier
        """
        n_modes = self.world_model.n_modes
        eta = node.state[:n_modes].copy()
        eta_dot = node.state[n_modes:].copy()
        m = node.contact_mode

        for _ in range(max(depth, 0)):
            eta, eta_dot = self.world_model.hamilton_step(eta, eta_dot, m, h=0.01)

        verifier_out = self.verifier(eta, eta_dot, m, goal.target_position)
        return verifier_out.value

    def _expand(self, node: TreeNode, goal: Goal) -> None:
        """Expand a leaf node by sampling child actions and states.

        Parameters
        ----------
        node : leaf node to expand
        goal : Goal specification (unused during expansion, passed for context)
        """
        state = node.state
        actions = self._sample_actions(state, self.config.branching)
        n_modes = self.world_model.n_modes

        eta = state[:n_modes].copy()
        eta_dot = state[n_modes:].copy()
        m = node.contact_mode

        priors = np.ones(self.config.branching) / self.config.branching

        for action, prior in zip(actions, priors):
            eta_new, eta_dot_new = self.world_model.hamilton_step(eta, eta_dot, m, h=0.01)
            child_state = np.concatenate([eta_new, eta_dot_new])
            child = TreeNode(
                state=child_state,
                contact_mode=m,
                action=action,
                prior=float(prior),
            )
            node.add_child(child)

    def run(
        self,
        root_state: ModalState,
        goal: Goal,
    ) -> ActionTrajectory:
        """Run MCTS from root_state towards goal.

        Parameters
        ----------
        root_state : ModalState — starting modal state
        goal : Goal — target goal specification

        Returns
        -------
        ActionTrajectory — best action sequence found
        """
        root_arr = np.concatenate([root_state.eta, root_state.eta_dot])
        root = TreeNode(state=root_arr, contact_mode=root_state.contact_mode)

        for sim_idx in range(self.config.n_simulations):
            leaf = select_leaf(root, self.config.c_puct)
            current_depth = leaf.depth()

            if leaf.visit_count == 0 or current_depth >= self.config.depth:
                # Roll out directly from leaf
                remaining = max(0, min(10, self.config.depth - current_depth))
                value = self._simulate(leaf, goal, depth=remaining)
            else:
                # Expand the leaf and simulate from a child
                self._expand(leaf, goal)
                if leaf.children:
                    child_idx = self._rng.integers(0, len(leaf.children))
                    child = leaf.children[child_idx]
                    value = self._simulate(child, goal, depth=5)
                else:
                    value = 0.0

            leaf.backup(value)

        # Extract best action sequence greedily (highest Q-value at each step)
        actions: list[NDArray[np.float64]] = []
        node = root
        depth_reached = 0
        while node.children and depth_reached < self.config.depth:
            best_child = max(node.children, key=lambda c: c.q_value)
            if best_child.action is not None:
                actions.append(best_child.action)
            node = best_child
            depth_reached += 1

        return ActionTrajectory(
            actions=actions,
            expected_value=root.q_value,
            search_depth_reached=depth_reached,
        )

    def search(self, root_state: ModalState, goal: Goal) -> ActionTrajectory:
        """Alias for run()."""
        return self.run(root_state, goal)


class MCTS:
    """Convenience wrapper matching the spec API.

    Parameters
    ----------
    world_model : HamiltonianNet
    verifier : Verifier
    policy : callable — maps state → (mean, cov)
    config : MCTSConfig or None
    action_dim : int
    """

    def __init__(
        self,
        world_model: Any,
        verifier: Any,
        policy: Callable,
        config: "MCTSConfig | None" = None,
        action_dim: int = 23,
    ) -> None:
        self._search = MCTSSearch(world_model, verifier, policy, config, action_dim)

    def search(self, root_state: ModalState, goal: Goal) -> ActionTrajectory:
        """Run MCTS and return the best action trajectory."""
        return self._search.run(root_state, goal)
