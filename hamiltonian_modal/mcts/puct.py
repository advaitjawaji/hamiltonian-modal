"""PUCT selection rule for MCTS."""
import logging
import math
import numpy as np
from numpy.typing import NDArray
from hamiltonian_modal.mcts.tree import TreeNode

logger = logging.getLogger(__name__)


def puct_score(
    node: TreeNode,
    child: TreeNode,
    c_puct: float = 1.5,
) -> float:
    """PUCT score for a child node.

    PUCT(s, a) = Q(s, a) + c · P(s, a) · √N(s) / (1 + N(s, a))

    Parameters
    ----------
    node : parent node
    child : child node
    c_puct : exploration constant

    Returns
    -------
    score : float
    """
    n_parent = max(node.visit_count, 1)
    exploration = c_puct * child.prior * math.sqrt(n_parent) / (1 + child.visit_count)
    return child.q_value + exploration


def select_child(
    node: TreeNode,
    c_puct: float = 1.5,
) -> TreeNode:
    """Select the child with the highest PUCT score.

    Parameters
    ----------
    node : parent node (must have children)
    c_puct : exploration constant

    Returns
    -------
    best_child : TreeNode
    """
    if not node.children:
        raise ValueError("Cannot select child from leaf node")
    scores = [puct_score(node, child, c_puct) for child in node.children]
    best_idx = int(np.argmax(scores))
    return node.children[best_idx]


def select_leaf(
    root: TreeNode,
    c_puct: float = 1.5,
) -> TreeNode:
    """Traverse from root to a leaf using PUCT selection.

    Parameters
    ----------
    root : root node to start from
    c_puct : exploration constant

    Returns
    -------
    leaf : TreeNode — deepest non-expanded node found
    """
    node = root
    while not node.is_leaf():
        node = select_child(node, c_puct)
    return node
