"""Unit tests for the differentiable simulation pipeline."""
import numpy as np
import pytest

from hamiltonian_modal.diff_sim.pipeline import DiffSimPipeline, RolloutResult
from hamiltonian_modal.world_model.hamiltonian_net import HamiltonianNet
from hamiltonian_modal.policy.mlp_policy import MLPPolicy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_world_model() -> HamiltonianNet:
    return HamiltonianNet(
        n_modes=4,
        n_contact_modes=2,
        v_hidden=[32, 32],
        contact_hidden=[16, 16],
        seed=0,
    )


@pytest.fixture()
def small_policy() -> MLPPolicy:
    return MLPPolicy(obs_dim=8, action_dim=4, hidden=[32, 32], seed=0)


@pytest.fixture()
def pipeline_no_model(small_policy) -> DiffSimPipeline:
    """Pipeline without a world model (simple linear dynamics)."""
    return DiffSimPipeline(
        policy=small_policy,
        world_model=None,
        n_modes=4,
        n_contact_modes=2,
    )


@pytest.fixture()
def pipeline_with_model(small_policy, small_world_model) -> DiffSimPipeline:
    """Pipeline with a world model rollout."""
    return DiffSimPipeline(
        policy=small_policy,
        world_model=small_world_model,
        n_modes=4,
        n_contact_modes=2,
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestPipelineInit:
    def test_pipeline_init(self, small_policy):
        """DiffSimPipeline should initialise without error."""
        pipeline = DiffSimPipeline(
            policy=small_policy,
            n_modes=4,
            n_contact_modes=2,
        )
        assert pipeline.policy is small_policy
        assert pipeline.world_model is None
        assert pipeline.n_modes == 4
        assert pipeline.n_contact_modes == 2

    def test_pipeline_init_with_world_model(self, small_policy, small_world_model):
        """DiffSimPipeline with a world model should expose it."""
        pipeline = DiffSimPipeline(
            policy=small_policy,
            world_model=small_world_model,
            n_modes=4,
            n_contact_modes=2,
        )
        assert pipeline.world_model is small_world_model

    def test_pipeline_stores_modal_dims(self, pipeline_with_model):
        assert pipeline_with_model.n_modes == 4
        assert pipeline_with_model.n_contact_modes == 2


# ---------------------------------------------------------------------------
# Rollout shapes
# ---------------------------------------------------------------------------

class TestRolloutShape:
    def test_rollout_shape_no_world_model(self, pipeline_no_model):
        """Rollout without world model: states (T+1, obs_dim), actions (T, action_dim)."""
        horizon = 5
        obs_dim = 8
        initial_obs = np.zeros(obs_dim)

        result = pipeline_no_model.rollout(initial_obs, horizon=horizon)

        assert isinstance(result, RolloutResult)
        assert result.states.shape == (horizon + 1, obs_dim)
        assert result.actions.shape == (horizon, 4)
        assert result.rewards.shape == (horizon,)
        assert result.modal_states is None

    def test_rollout_shape_with_world_model(self, pipeline_with_model):
        """Rollout with world model: modal_states (T+1, n_modes*2)."""
        horizon = 5
        n_modes = 4
        initial_obs = np.zeros(n_modes * 2)

        result = pipeline_with_model.rollout(
            initial_obs, horizon=horizon, use_world_model=True
        )

        assert result.states.shape == (horizon + 1, n_modes * 2)
        assert result.actions.shape == (horizon, 4)
        assert result.rewards.shape == (horizon,)
        assert result.modal_states is not None
        assert result.modal_states.shape == (horizon + 1, n_modes * 2)

    def test_rollout_first_state_is_initial(self, pipeline_no_model):
        """The first state in the rollout should equal the initial observation."""
        initial_obs = np.arange(8, dtype=np.float64)
        result = pipeline_no_model.rollout(initial_obs, horizon=3)
        assert np.allclose(result.states[0], initial_obs)

    def test_rollout_total_reward_sum(self, pipeline_no_model):
        """total_reward should equal the sum of per-step rewards."""
        def reward_fn(state, action):
            return 1.0

        result = pipeline_no_model.rollout(
            np.zeros(8), horizon=5, reward_fn=reward_fn
        )
        assert result.total_reward == pytest.approx(5.0)
        assert result.rewards.sum() == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# World model rollout
# ---------------------------------------------------------------------------

class TestWorldModelRollout:
    def test_world_model_rollout(self, pipeline_with_model):
        """World-model rollout should produce finite states."""
        n_modes = 4
        initial_obs = 0.1 * np.ones(n_modes * 2)

        result = pipeline_with_model.rollout(
            initial_obs, horizon=10, use_world_model=True
        )

        assert np.all(np.isfinite(result.states))
        assert np.all(np.isfinite(result.actions))

    def test_world_model_rollout_modal_states_consistent(self, pipeline_with_model):
        """modal_states[:, :n_modes] should equal the η component of states."""
        n_modes = 4
        initial_obs = np.random.default_rng(0).standard_normal(n_modes * 2)

        result = pipeline_with_model.rollout(
            initial_obs, horizon=5, use_world_model=True
        )

        assert np.allclose(
            result.modal_states[:, :n_modes],
            result.states[:, :n_modes],
        )


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

class TestRewardFn:
    def test_reward_fn_called(self, pipeline_no_model):
        """reward_fn should be called at every timestep."""
        call_count = {"n": 0}

        def counting_reward(state, action):
            call_count["n"] += 1
            return float(call_count["n"])

        horizon = 7
        result = pipeline_no_model.rollout(
            np.zeros(8), horizon=horizon, reward_fn=counting_reward
        )

        assert call_count["n"] == horizon
        assert len(result.rewards) == horizon

    def test_reward_fn_values_stored(self, pipeline_no_model):
        """Reward values from reward_fn should be stored in result.rewards."""
        fixed_rewards = [1.0, 2.0, 3.0]
        step = {"i": 0}

        def fixed_reward_fn(state, action):
            r = fixed_rewards[step["i"]]
            step["i"] += 1
            return r

        result = pipeline_no_model.rollout(
            np.zeros(8), horizon=3, reward_fn=fixed_reward_fn
        )

        assert np.allclose(result.rewards, [1.0, 2.0, 3.0])
        assert result.total_reward == pytest.approx(6.0)

    def test_default_reward_is_zero(self, pipeline_no_model):
        """Without a reward_fn, all rewards should be 0.0."""
        result = pipeline_no_model.rollout(np.zeros(8), horizon=4)
        assert np.all(result.rewards == 0.0)
        assert result.total_reward == pytest.approx(0.0)
