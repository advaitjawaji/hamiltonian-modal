"""Tree data structure for MCTS."""
import logging
import math
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    """A node in the MCTS tree.

    Attributes
    ----------
    state : NDArray — modal state [η; η̇]
    contact_mode : int
    parent : TreeNode or None
    action : NDArray or None — action that led to this node
    children : list[TreeNode]
    visit_count : int
    total_value : float
    prior : float — prior probability from policy
    """
    state: NDArray[np.float64]
    contact_mode: int = 0
    parent: "TreeNode | None" = field(default=None, repr=False)
    action: "NDArray[np.float64] | None" = None
    children: "list[TreeNode]" = field(default_factory=list)
    visit_count: int = 0
    total_value: float = 0.0
    prior: float = 1.0

    @property
    def q_value(self) -> float:
        """Mean value estimate Q(s, a)."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def is_leaf(self) -> bool:
        """Return True if this node has no children."""
        return len(self.children) == 0

    def is_root(self) -> bool:
        """Return True if this node has no parent."""
        return self.parent is None

    def add_child(self, child: "TreeNode") -> None:
        """Add a child node, setting its parent pointer."""
        child.parent = self
        self.children.append(child)

    def backup(self, value: float) -> None:
        """Propagate value up to root, incrementing visit counts."""
        node: "TreeNode | None" = self
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node = node.parent

    def depth(self) -> int:
        """Return the depth of this node from the root."""
        d = 0
        node = self
        while node.parent is not None:
            d += 1
            node = node.parent
        return d
