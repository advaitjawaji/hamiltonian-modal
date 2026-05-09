"""Unit tests for the MCTS module."""
import math
import numpy as np
import pytest

from hamiltonian_modal.mcts.tree import TreeNode
from hamiltonian_modal.mcts.puct import puct_score, select_child, select_leaf
from hamiltonian_modal.mcts.search import (
    MCTSConfig,
    MCTSSearch,
    MCTS,
    ModalState,
    Goal,
    ActionTrajectory,
)
from hamiltonian_modal.world_model.hamiltonian_net import HamiltonianNet
from hamiltonian_modal.world_model.verifier import Verifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_world_model() -> HamiltonianNet:
    """A small HamiltonianNet for fast tests."""
    return HamiltonianNet(
        n_modes=4,
        n_contact_modes=2,
        v_hidden=[32, 32],
        contact_hidden=[16, 16],
        seed=0,
    )


@pytest.fixture()
def small_verifier() -> Verifier:
    """A small Verifier for fast tests."""
    return Verifier(
        n_modes=4,
        n_contact_modes=2,
        goal_dim=3,
        hidden=[32, 32],
        seed=0,
    )


def _make_uniform_policy(action_dim: int = 4):
    """Return a policy that always outputs zero mean with identity cov."""
    def policy(state: np.ndarray):
        return np.zeros(action_dim), np.eye(action_dim)
    return policy


# ---------------------------------------------------------------------------
# TreeNode tests
# ---------------------------------------------------------------------------

class TestTreeNode:
    def test_tree_node_init(self):
        state = np.array([1.0, 2.0, 3.0, 4.0])
        node = TreeNode(state=state, contact_mode=1)

        assert node.visit_count == 0
        assert node.total_value == 0.0
        assert node.prior == 1.0
        assert node.contact_mode == 1
        assert node.is_leaf()
        assert node.is_root()
        assert node.action is None
        assert np.allclose(node.state, state)

    def test_tree_node_q_value_zero_visits(self):
        node = TreeNode(state=np.zeros(4))
        assert node.q_value == 0.0

    def test_tree_node_q_value_nonzero(self):
        node = TreeNode(state=np.zeros(4))
        node.visit_count = 4
        node.total_value = 8.0
        assert node.q_value == pytest.approx(2.0)

    def test_tree_node_backup(self):
        """Backup should propagate value through the tree to root."""
        root = TreeNode(state=np.zeros(4))
        child = TreeNode(state=np.zeros(4))
        root.add_child(child)

        child.backup(3.0)

        # Root should have been visited once with value 3.0
        assert root.visit_count == 1
        assert root.total_value == pytest.approx(3.0)
        # Child should have been visited once too
        assert child.visit_count == 1
        assert child.total_value == pytest.approx(3.0)

    def test_backup_multiple_values(self):
        root = TreeNode(state=np.zeros(4))
        child = TreeNode(state=np.zeros(4))
        root.add_child(child)

        child.backup(2.0)
        child.backup(4.0)

        assert root.visit_count == 2
        assert root.total_value == pytest.approx(6.0)
        assert root.q_value == pytest.approx(3.0)

    def test_tree_node_depth(self):
        root = TreeNode(state=np.zeros(4))
        child = TreeNode(state=np.zeros(4))
        grandchild = TreeNode(state=np.zeros(4))

        root.add_child(child)
        child.add_child(grandchild)

        assert root.depth() == 0
        assert child.depth() == 1
        assert grandchild.depth() == 2

    def test_is_leaf_and_root(self):
        root = TreeNode(state=np.zeros(4))
        child = TreeNode(state=np.zeros(4))

        assert root.is_leaf()
        assert root.is_root()

        root.add_child(child)
        assert not root.is_leaf()
        assert root.is_root()
        assert child.is_leaf()
        assert not child.is_root()

    def test_add_child_sets_parent(self):
        root = TreeNode(state=np.zeros(4))
        child = TreeNode(state=np.zeros(4))
        root.add_child(child)

        assert child.parent is root
        assert child in root.children


# ---------------------------------------------------------------------------
# PUCT tests
# ---------------------------------------------------------------------------

class TestPUCT:
    def test_puct_score_formula(self):
        """PUCT(s,a) = Q + c * P * sqrt(N_parent) / (1 + N_child)"""
        parent = TreeNode(state=np.zeros(4))
        parent.visit_count = 9
        child = TreeNode(state=np.zeros(4))
        child.visit_count = 2
        child.total_value = 6.0  # Q = 3.0
        child.prior = 0.5

        c_puct = 1.5
        expected_q = 3.0
        expected_u = c_puct * 0.5 * math.sqrt(9) / (1 + 2)
        expected_score = expected_q + expected_u

        score = puct_score(parent, child, c_puct)
        assert score == pytest.approx(expected_score, rel=1e-6)

    def test_puct_score_unvisited_child(self):
        """Unvisited child (Q=0) should still have positive PUCT score."""
        parent = TreeNode(state=np.zeros(4))
        parent.visit_count = 4
        child = TreeNode(state=np.zeros(4), prior=1.0)

        score = puct_score(parent, child, c_puct=1.5)
        assert score > 0.0

    def test_puct_score_exploration_constant(self):
        """Higher c_puct should produce higher score for unvisited child."""
        parent = TreeNode(state=np.zeros(4))
        parent.visit_count = 4
        child = TreeNode(state=np.zeros(4), prior=1.0)

        score_low = puct_score(parent, child, c_puct=0.5)
        score_high = puct_score(parent, child, c_puct=2.0)
        assert score_high > score_low

    def test_select_child_picks_highest_puct(self):
        """select_child should pick the child with the highest PUCT score."""
        parent = TreeNode(state=np.zeros(4))
        parent.visit_count = 10

        # Child A: high Q, low exploration
        child_a = TreeNode(state=np.ones(4), prior=0.5)
        child_a.visit_count = 5
        child_a.total_value = 20.0  # Q = 4.0

        # Child B: zero Q, high exploration (unvisited, high prior)
        child_b = TreeNode(state=-np.ones(4), prior=0.5)

        parent.add_child(child_a)
        parent.add_child(child_b)

        # With c_puct=0 only Q matters → child_a wins
        best = select_child(parent, c_puct=0.0)
        assert best is child_a

    def test_select_child_raises_on_leaf(self):
        """select_child should raise ValueError for leaf nodes."""
        leaf = TreeNode(state=np.zeros(4))
        with pytest.raises(ValueError, match="leaf"):
            select_child(leaf)

    def test_select_leaf_traverses_tree(self):
        """select_leaf should descend to the deepest reachable leaf."""
        root = TreeNode(state=np.zeros(4))
        root.visit_count = 10

        child = TreeNode(state=np.ones(4), prior=1.0)
        root.add_child(child)

        leaf = select_leaf(root, c_puct=1.5)
        # Should have descended past root to child (which is the only child)
        assert leaf is child

    def test_select_leaf_at_root_with_no_children(self):
        """select_leaf on a root with no children returns root immediately."""
        root = TreeNode(state=np.zeros(4))
        leaf = select_leaf(root, c_puct=1.5)
        assert leaf is root


# ---------------------------------------------------------------------------
# MCTS config
# ---------------------------------------------------------------------------

class TestMCTSConfig:
    def test_mcts_config_defaults(self):
        cfg = MCTSConfig()
        assert cfg.branching == 8
        assert cfg.depth == 50
        assert cfg.n_simulations == 50
        assert cfg.c_puct == pytest.approx(1.5)
        assert cfg.rollout_temperature == pytest.approx(1.0)

    def test_mcts_config_custom(self):
        cfg = MCTSConfig(branching=4, depth=20, n_simulations=10, c_puct=2.0)
        assert cfg.branching == 4
        assert cfg.depth == 20
        assert cfg.n_simulations == 10


# ---------------------------------------------------------------------------
# MCTS integration tests
# ---------------------------------------------------------------------------

class TestMCTSSearch:
    def test_mcts_search_returns_action_trajectory(
        self,
        small_world_model,
        small_verifier,
    ):
        """MCTS.search should return an ActionTrajectory with correct types."""
        policy = _make_uniform_policy(action_dim=4)
        config = MCTSConfig(branching=3, depth=5, n_simulations=5)
        mcts = MCTS(small_world_model, small_verifier, policy, config=config, action_dim=4)

        root_state = ModalState(
            eta=np.zeros(4),
            eta_dot=np.zeros(4),
            contact_mode=0,
        )
        goal = Goal(target_position=np.array([1.0, 0.0, 0.0]))

        result = mcts.search(root_state, goal)

        assert isinstance(result, ActionTrajectory)
        assert isinstance(result.actions, list)
        assert isinstance(result.expected_value, float)
        assert isinstance(result.search_depth_reached, int)
        assert result.search_depth_reached >= 0

    def test_mcts_actions_are_arrays(self, small_world_model, small_verifier):
        """Each action in ActionTrajectory should be a numpy array."""
        policy = _make_uniform_policy(action_dim=4)
        config = MCTSConfig(branching=3, depth=3, n_simulations=3)
        mcts = MCTS(small_world_model, small_verifier, policy, config=config, action_dim=4)

        root_state = ModalState(
            eta=np.array([0.1, -0.1, 0.05, 0.0]),
            eta_dot=np.zeros(4),
            contact_mode=0,
        )
        goal = Goal(target_position=np.array([2.0, 0.0, 0.0]))
        result = mcts.search(root_state, goal)

        for action in result.actions:
            assert isinstance(action, np.ndarray)
            assert action.shape == (4,)

    def test_mcts_depth_50_completes(self, small_world_model, small_verifier):
        """MCTS with depth=50 should complete without error."""
        policy = _make_uniform_policy(action_dim=4)
        config = MCTSConfig(branching=2, depth=50, n_simulations=10)
        mcts = MCTSSearch(
            small_world_model, small_verifier, policy, config=config, action_dim=4
        )

        root_state = ModalState(eta=np.zeros(4), eta_dot=np.zeros(4))
        goal = Goal(target_position=np.zeros(3))
        result = mcts.run(root_state, goal)

        assert result is not None
        assert result.search_depth_reached <= 50

    def test_mcts_search_alias(self, small_world_model, small_verifier):
        """MCTSSearch.search should be an alias for run()."""
        policy = _make_uniform_policy(action_dim=4)
        config = MCTSConfig(branching=2, depth=3, n_simulations=3)
        searcher = MCTSSearch(
            small_world_model, small_verifier, policy, config=config, action_dim=4
        )

        root_state = ModalState(eta=np.zeros(4), eta_dot=np.zeros(4))
        goal = Goal(target_position=np.zeros(3))

        result1 = searcher.run(root_state, goal)
        result2 = searcher.search(root_state, goal)

        # Both should be ActionTrajectory instances
        assert isinstance(result1, ActionTrajectory)
        assert isinstance(result2, ActionTrajectory)

    def test_mcts_root_visit_count_increases(self, small_world_model, small_verifier):
        """After n_simulations, root's visit count should be n_simulations."""
        policy = _make_uniform_policy(action_dim=4)
        n_sims = 8
        config = MCTSConfig(branching=2, depth=3, n_simulations=n_sims)
        mcts = MCTSSearch(
            small_world_model, small_verifier, policy, config=config, action_dim=4
        )

        root_state = ModalState(eta=np.zeros(4), eta_dot=np.zeros(4))
        goal = Goal(target_position=np.zeros(3))

        # We need access to the root node — run manually
        root_arr = np.concatenate([root_state.eta, root_state.eta_dot])
        root = TreeNode(state=root_arr, contact_mode=root_state.contact_mode)

        from hamiltonian_modal.mcts.puct import select_leaf
        for _ in range(n_sims):
            leaf = select_leaf(root, config.c_puct)
            if leaf.visit_count == 0:
                value = mcts._simulate(leaf, goal, depth=2)
            else:
                mcts._expand(leaf, goal)
                value = 0.0
            leaf.backup(value)

        assert root.visit_count == n_sims
