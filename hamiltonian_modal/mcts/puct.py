"""PUCT policy for tree search.

Implements the PUCT (Predictor + Upper Confidence bound applied to Trees)
selection criterion used in AlphaZero / MuZero:

.. math::

    U(s, a) = Q(s, a) + c_{\\text{puct}} \\cdot P(s, a) \\cdot
               \\frac{\\sqrt{N(s)}}{1 + N(s, a)}

The action maximising *U* is selected at each tree node.
"""

from __future__ import annotations

import math

import numpy as np

from hamiltonian_modal.mcts.tree import TreeNode

__all__ = [
    "puct_score",
    "select_child",
]


def puct_score(
    parent: TreeNode,
    child: TreeNode,
    c_puct: float = 1.0,
) -> float:
    """Compute the PUCT score for *child* of *parent*.

    Parameters
    ----------
    parent:
        The parent tree node (holds N(s)).
    child:
        The child node for action *a* (holds Q(s,a), P(s,a), N(s,a)).
    c_puct:
        Exploration constant.

    Returns
    -------
    float
        PUCT score U(s, a).
    """
    exploration = (
        c_puct
        * child.prior
        * math.sqrt(parent.visit_count)
        / (1 + child.visit_count)
    )
    return child.q_value + exploration


def select_child(node: TreeNode, c_puct: float = 1.0) -> tuple[int, TreeNode]:
    """Select the child of *node* with the highest PUCT score.

    Parameters
    ----------
    node:
        A non-leaf :class:`~hamiltonian_modal.mcts.tree.TreeNode`.
    c_puct:
        Exploration constant.

    Returns
    -------
    tuple[int, TreeNode]
        ``(action_index, best_child)``.

    Raises
    ------
    ValueError
        If *node* has no children.
    """
    if not node.children:
        raise ValueError("Cannot select child of a leaf node.")
    best_action = max(
        node.children,
        key=lambda a: puct_score(node, node.children[a], c_puct),
    )
    return best_action, node.children[best_action]
