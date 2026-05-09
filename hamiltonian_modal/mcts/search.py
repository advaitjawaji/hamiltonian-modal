"""Search orchestration interfaces.

:class:`MCTSSearch` ties together the PUCT selection, tree expansion,
leaf evaluation, and value back-propagation into a complete MCTS loop.

Usage::

    root = TreeNode(eta=eta0, mu=mu0)
    search = MCTSSearch(config, dynamics_fn, policy_fn, value_fn)
    action = search.run(root)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from hamiltonian_modal.config import MCTSConfig
from hamiltonian_modal.mcts.puct import select_child
from hamiltonian_modal.mcts.tree import TreeNode

__all__ = [
    "MCTSSearch",
]

DynamicsFn = Callable[
    [NDArray[np.float64], NDArray[np.float64], int],
    tuple[NDArray[np.float64], NDArray[np.float64]],
]
PolicyFn = Callable[
    [NDArray[np.float64], NDArray[np.float64]],
    NDArray[np.float64],
]
ValueFn = Callable[[NDArray[np.float64], NDArray[np.float64]], float]


class MCTSSearch:
    """Monte-Carlo Tree Search over the modal state space.

    Parameters
    ----------
    config:
        :class:`~hamiltonian_modal.config.MCTSConfig`.
    n_actions:
        Number of discrete actions (children per node).
    dynamics_fn:
        Callable ``(η, μ, action) → (η', μ')`` that transitions the state.
    policy_fn:
        Callable ``(η, μ) → prior_probs`` of shape ``(n_actions,)``.
    value_fn:
        Callable ``(η, μ) → scalar_value``.
    """

    def __init__(
        self,
        config: MCTSConfig,
        n_actions: int,
        dynamics_fn: DynamicsFn,
        policy_fn: PolicyFn,
        value_fn: ValueFn,
    ) -> None:
        self.config = config
        self.n_actions = n_actions
        self._dynamics = dynamics_fn
        self._policy = policy_fn
        self._value = value_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, root: TreeNode) -> int:
        """Run MCTS from *root* and return the best action index.

        Parameters
        ----------
        root:
            Root :class:`~hamiltonian_modal.mcts.tree.TreeNode` (visit_count
            may be non-zero if we are continuing a previous search).

        Returns
        -------
        int
            Action index with the highest visit count at the root.
        """
        for _ in range(self.config.n_simulations):
            leaf, search_path = self._select(root)
            value = self._expand_and_evaluate(leaf)
            self._backpropagate(search_path, value)
        return self._best_action(root)

    # ------------------------------------------------------------------
    # Internal MCTS phases
    # ------------------------------------------------------------------

    def _select(self, node: TreeNode) -> tuple[TreeNode, list[TreeNode]]:
        """Traverse the tree from *node* to a leaf using PUCT."""
        path = [node]
        depth = 0
        while not node.is_leaf and depth < self.config.max_depth:
            _, node = select_child(node, c_puct=self.config.c_puct)
            path.append(node)
            depth += 1
        return node, path

    def _expand_and_evaluate(self, leaf: TreeNode) -> float:
        """Expand *leaf* (if not terminal) and return a value estimate."""
        prior_probs = self._policy(leaf.eta, leaf.mu)
        # Softmax-normalise raw policy outputs
        prior_probs = prior_probs - np.max(prior_probs)
        prior_probs = np.exp(prior_probs)
        prior_probs /= prior_probs.sum() + 1e-8

        next_states = [
            self._dynamics(leaf.eta, leaf.mu, a) for a in range(self.n_actions)
        ]
        leaf.expand(prior_probs, next_states)
        return self._value(leaf.eta, leaf.mu)

    def _backpropagate(self, path: list[TreeNode], value: float) -> None:
        """Back-propagate *value* up the search path with discounting."""
        discount = 1.0
        for node in reversed(path):
            node.update(value * discount)
            discount *= self.config.discount

    def _best_action(self, root: TreeNode) -> int:
        """Return the action with the highest visit count at the root."""
        return max(root.children, key=lambda a: root.children[a].visit_count)
