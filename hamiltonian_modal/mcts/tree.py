"""Tree data structures for search.

Provides :class:`TreeNode` — the fundamental building block of the MCTS tree.
Each node stores:

* The modal state (η, μ) at this node.
* Visit count N, total value W, and mean value Q = W / N.
* Prior probability P assigned by a policy network.
* Child nodes indexed by discrete action index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "TreeNode",
]


@dataclass
class TreeNode:
    """A single node in the MCTS tree.

    Attributes
    ----------
    eta:
        Modal position at this node, shape ``(n_modes,)``.
    mu:
        Modal momentum at this node, shape ``(n_modes,)``.
    prior:
        Prior probability P(s, a) assigned by the policy (0 for the root).
    parent:
        Parent :class:`TreeNode`, or ``None`` for the root.
    action_taken:
        Index of the action that led to this node (``None`` for root).
    visit_count:
        Number of times this node has been visited.
    value_sum:
        Accumulated value across all visits (W).
    children:
        Dictionary mapping action index → child :class:`TreeNode`.
    """

    eta: NDArray[np.float64]
    mu: NDArray[np.float64]
    prior: float = 0.0
    parent: "TreeNode | None" = field(default=None, repr=False)
    action_taken: int | None = None
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "TreeNode"] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def is_root(self) -> bool:
        """Whether this is the root node."""
        return self.parent is None

    @property
    def is_leaf(self) -> bool:
        """Whether this node has no expanded children."""
        return len(self.children) == 0

    @property
    def q_value(self) -> float:
        """Mean value Q = W / N (0 when never visited)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, value: float) -> None:
        """Back-propagate *value* into this node.

        Parameters
        ----------
        value:
            Value estimate from a leaf evaluation or rollout.
        """
        self.visit_count += 1
        self.value_sum += value

    def expand(
        self,
        action_priors: NDArray[np.float64],
        next_states: list[tuple[NDArray[np.float64], NDArray[np.float64]]],
    ) -> None:
        """Add child nodes for each action.

        Parameters
        ----------
        action_priors:
            Prior probabilities from the policy, shape ``(n_actions,)``.
        next_states:
            List of ``(η', μ')`` tuples resulting from each action.
        """
        for a, ((eta_next, mu_next), prior) in enumerate(
            zip(next_states, action_priors)
        ):
            self.children[a] = TreeNode(
                eta=eta_next,
                mu=mu_next,
                prior=float(prior),
                parent=self,
                action_taken=a,
            )
