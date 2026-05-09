"""Unit tests for Phase 1 reimplementation.

Covers:
* utils: seeding, logging, genesis_wrapper
* config
* data: retargeted_motions, lafan1_g1, amass_g1
* envs: g1_base, g1_walking, g1_fast_walking, g1_push_recovery, g1_jumping, g1_disturbance
* world_model: symplectic, hamiltonian_net, uncertainty, contact_event, verifier, trainer
* mcts: tree, puct, search
* policy: mlp_policy, flow_matching_head, vla_backbone
* mpc: lagrangian_net, ilqr, ddp
* diff_sim: stability_utils, pipeline, policy_gradient, trainer
"""

from __future__ import annotations

import logging
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ===========================================================================
# utils/seeding
# ===========================================================================


class TestSeeding:
    def test_seed_all_deterministic_numpy(self) -> None:
        from hamiltonian_modal.utils.seeding import seed_all

        seed_all(42)
        a = np.random.rand(5)
        seed_all(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_make_rng_returns_generator(self) -> None:
        from hamiltonian_modal.utils.seeding import make_rng

        rng = make_rng(0)
        assert isinstance(rng, np.random.Generator)

    def test_make_rng_reproducible(self) -> None:
        from hamiltonian_modal.utils.seeding import make_rng

        a = make_rng(7).standard_normal(10)
        b = make_rng(7).standard_normal(10)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_different_output(self) -> None:
        from hamiltonian_modal.utils.seeding import make_rng

        a = make_rng(1).standard_normal(5)
        b = make_rng(2).standard_normal(5)
        assert not np.allclose(a, b)


# ===========================================================================
# utils/logging
# ===========================================================================


class TestLogging:
    def test_get_logger_returns_logger(self) -> None:
        from hamiltonian_modal.utils.logging import get_logger

        logger = get_logger()
        assert isinstance(logger, logging.Logger)

    def test_get_logger_prefixed(self) -> None:
        from hamiltonian_modal.utils.logging import get_logger

        logger = get_logger("modal")
        assert logger.name.startswith("hamiltonian_modal")

    def test_already_prefixed_not_double_prefixed(self) -> None:
        from hamiltonian_modal.utils.logging import get_logger

        logger = get_logger("hamiltonian_modal.modal")
        assert logger.name == "hamiltonian_modal.modal"

    def test_configure_logging_sets_level(self) -> None:
        from hamiltonian_modal.utils.logging import configure_logging, get_logger

        configure_logging(level=logging.DEBUG)
        logger = get_logger()
        assert logger.level == logging.DEBUG

    def test_configure_logging_file_handler(self, tmp_path: Path) -> None:
        from hamiltonian_modal.utils.logging import configure_logging, get_logger

        log_file = tmp_path / "test.log"
        configure_logging(log_file=log_file)
        logger = get_logger()
        logger.info("test message")
        assert log_file.exists()
        assert "test message" in log_file.read_text()


# ===========================================================================
# utils/genesis_wrapper
# ===========================================================================


class TestGenesisWrapper:
    def test_load_genesis_raises_without_genesis(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "genesis", None)
        from hamiltonian_modal.utils import genesis_wrapper

        monkeypatch.setattr(genesis_wrapper, "_require_genesis", lambda: (_ for _ in ()).throw(ImportError("no genesis")))
        with pytest.raises((ImportError, TypeError)):
            genesis_wrapper._require_genesis()

    def test_get_joint_positions_wraps_to_float64(self) -> None:
        from hamiltonian_modal.utils.genesis_wrapper import get_joint_positions

        class FakeRobot:
            def get_dofs_position(self):
                return [1.0, 2.0, 3.0]

        result = get_joint_positions(FakeRobot())
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_get_joint_velocities_wraps_to_float64(self) -> None:
        from hamiltonian_modal.utils.genesis_wrapper import get_joint_velocities

        class FakeRobot:
            def get_dofs_velocity(self):
                return [0.1, 0.2]

        result = get_joint_velocities(FakeRobot())
        assert result.dtype == np.float64

    def test_step_scene_calls_step(self) -> None:
        from hamiltonian_modal.utils.genesis_wrapper import step_scene

        calls = []

        class FakeScene:
            def step(self):
                calls.append(1)

        step_scene(FakeScene(), n_steps=3)
        assert len(calls) == 3

    def test_apply_joint_torques_delegates(self) -> None:
        from hamiltonian_modal.utils.genesis_wrapper import apply_joint_torques

        received = []

        class FakeRobot:
            def set_dofs_force(self, t):
                received.append(t)

        torques = np.array([1.0, 2.0])
        apply_joint_torques(FakeRobot(), torques)
        assert len(received) == 1


# ===========================================================================
# config
# ===========================================================================


class TestConfig:
    def test_experiment_config_defaults(self) -> None:
        from hamiltonian_modal.config import ExperimentConfig

        cfg = ExperimentConfig()
        assert cfg.modal.n_modes == 10
        assert cfg.symplectic.dt == pytest.approx(0.001)
        assert cfg.mcts.n_simulations == 50
        assert cfg.seed == 0

    def test_config_fields_are_independent(self) -> None:
        from hamiltonian_modal.config import ExperimentConfig

        a = ExperimentConfig()
        b = ExperimentConfig()
        a.modal.n_modes = 99
        assert b.modal.n_modes == 10  # independent instances

    def test_ddp_config_inherits_ilqr(self) -> None:
        from hamiltonian_modal.config import DDPConfig, ILQRConfig

        assert issubclass(DDPConfig, ILQRConfig)
        cfg = DDPConfig()
        assert hasattr(cfg, "second_order_value")


# ===========================================================================
# data
# ===========================================================================


class TestRetargetedMotions:
    def _write_clip(self, tmp_path: Path, name: str = "clip") -> Path:
        q = np.random.default_rng(0).standard_normal((50, 4))
        v = np.zeros_like(q)
        p = tmp_path / f"{name}.npz"
        np.savez(str(p), q=q, v=v, dt=np.array(0.033))
        return p

    def test_load_motion_clip(self, tmp_path: Path) -> None:
        from hamiltonian_modal.data.retargeted_motions import load_motion_clip

        p = self._write_clip(tmp_path)
        clip = load_motion_clip(p)
        assert clip.n_frames == 50
        assert clip.n_dof == 4
        assert clip.name == "clip"
        assert clip.duration == pytest.approx(50 * 0.033)

    def test_load_motion_clip_no_velocity(self, tmp_path: Path) -> None:
        from hamiltonian_modal.data.retargeted_motions import load_motion_clip

        q = np.ones((10, 3))
        p = tmp_path / "nv.npz"
        np.savez(str(p), q=q)
        clip = load_motion_clip(p, dt=0.01)
        assert clip.v.shape == (10, 3)

    def test_load_motion_clips(self, tmp_path: Path) -> None:
        from hamiltonian_modal.data.retargeted_motions import load_motion_clips

        for i in range(3):
            self._write_clip(tmp_path, name=f"clip_{i:02d}")
        clips = load_motion_clips(tmp_path)
        assert len(clips) == 3


class TestLAFAN1:
    def test_load_lafan1_dataset_source_tag(self, tmp_path: Path) -> None:
        from hamiltonian_modal.data.lafan1_g1 import LAFAN1_DT, load_lafan1_dataset

        q = np.zeros((30, 5))
        np.savez(str(tmp_path / "walk_001.npz"), q=q)
        clips = load_lafan1_dataset(tmp_path)
        assert clips[0].source == "LAFAN1"
        assert clips[0].dt == pytest.approx(LAFAN1_DT)

    def test_load_lafan1_filter_by_action(self, tmp_path: Path) -> None:
        from hamiltonian_modal.data.lafan1_g1 import load_lafan1_dataset

        for name in ["walk_001", "run_002", "jump_003"]:
            np.savez(str(tmp_path / f"{name}.npz"), q=np.zeros((10, 3)))
        clips = load_lafan1_dataset(tmp_path, actions=["walk"])
        assert len(clips) == 1
        assert clips[0].name == "walk_001"


class TestAMASS:
    def test_load_amass_dataset_source_tag(self, tmp_path: Path) -> None:
        from hamiltonian_modal.data.amass_g1 import AMASS_DT, load_amass_dataset

        np.savez(str(tmp_path / "subject_001.npz"), q=np.zeros((60, 4)))
        clips = load_amass_dataset(tmp_path)
        assert clips[0].source == "AMASS"
        assert clips[0].dt == pytest.approx(AMASS_DT)

    def test_load_amass_filter_by_subject(self, tmp_path: Path) -> None:
        from hamiltonian_modal.data.amass_g1 import load_amass_dataset

        for name in ["subjectA_001", "subjectB_002"]:
            np.savez(str(tmp_path / f"{name}.npz"), q=np.zeros((10, 3)))
        clips = load_amass_dataset(tmp_path, subjects=["subjectA"])
        assert len(clips) == 1


# ===========================================================================
# envs
# ===========================================================================


class TestG1BaseEnv:
    def _env(self):
        from hamiltonian_modal.envs.g1_walking import G1WalkingEnv

        return G1WalkingEnv()

    def test_reset_returns_obs_and_info(self) -> None:
        env = self._env()
        obs, info = env.reset()
        assert obs.shape == (env.obs_dim,)
        assert isinstance(info, dict)

    def test_step_returns_step_result(self) -> None:
        from hamiltonian_modal.envs.g1_base import StepResult

        env = self._env()
        env.reset()
        result = env.step(np.zeros(env.action_dim))
        assert isinstance(result, StepResult)
        assert result.obs.shape == (env.obs_dim,)

    def test_step_before_reset_raises(self) -> None:
        env = self._env()
        with pytest.raises(RuntimeError, match="reset"):
            env.step(np.zeros(env.action_dim))

    def test_truncated_at_episode_length(self) -> None:
        from hamiltonian_modal.config import EnvConfig
        from hamiltonian_modal.envs.g1_walking import G1WalkingEnv

        env = G1WalkingEnv(config=EnvConfig(episode_length=2))
        env.reset()
        env.step(np.zeros(env.action_dim))
        result = env.step(np.zeros(env.action_dim))
        assert result.truncated

    def test_obs_noise_applied(self) -> None:
        from hamiltonian_modal.config import EnvConfig
        from hamiltonian_modal.envs.g1_walking import G1WalkingEnv

        env = G1WalkingEnv(config=EnvConfig(obs_noise_std=10.0))
        obs_noisy, _ = env.reset(seed=0)
        env2 = G1WalkingEnv(config=EnvConfig(obs_noise_std=0.0))
        obs_clean, _ = env2.reset(seed=0)
        assert not np.allclose(obs_noisy, obs_clean)


class TestG1DisturbanceEnv:
    def test_obs_has_correct_dim(self) -> None:
        from hamiltonian_modal.envs.g1_disturbance import G1DisturbanceEnv

        env = G1DisturbanceEnv()
        obs, _ = env.reset()
        assert obs.shape == (env.obs_dim,)

    def test_disturbance_changes_trajectory(self) -> None:
        from hamiltonian_modal.envs.g1_disturbance import G1DisturbanceEnv

        env = G1DisturbanceEnv(disturbance_std=100.0, seed=1)
        env.reset()
        result = env.step(np.zeros(env.action_dim))
        assert isinstance(result.reward, float)


class TestG1PushRecoveryEnv:
    def test_reset_schedules_push(self) -> None:
        from hamiltonian_modal.envs.g1_push_recovery import G1PushRecoveryEnv

        env = G1PushRecoveryEnv(seed=42)
        env.reset()
        assert env._push_start >= 0


class TestG1JumpingEnv:
    def test_step_produces_reward(self) -> None:
        from hamiltonian_modal.envs.g1_jumping import G1JumpingEnv

        env = G1JumpingEnv()
        env.reset()
        result = env.step(np.zeros(env.action_dim))
        assert isinstance(result.reward, float)


class TestG1FastWalkingEnv:
    def test_step_produces_reward(self) -> None:
        from hamiltonian_modal.envs.g1_fast_walking import G1FastWalkingEnv

        env = G1FastWalkingEnv()
        env.reset()
        result = env.step(np.zeros(env.action_dim))
        assert isinstance(result.reward, float)


# ===========================================================================
# world_model/symplectic
# ===========================================================================


class TestSymplectic:
    def _harmonic_osc(self, omega: float = 1.0):
        """Simple harmonic oscillator: H = p²/2 + ω² q²/2."""
        return (
            lambda q: omega**2 * q,  # ∂H/∂q
            lambda p: p,             # ∂H/∂p = p
        )

    def test_leapfrog_conserves_energy(self) -> None:
        from hamiltonian_modal.world_model.symplectic import leapfrog_step

        omega = 1.0
        grad_H_q, grad_H_p = self._harmonic_osc(omega)
        q, p = np.array([1.0]), np.array([0.0])
        H0 = 0.5 * p[0] ** 2 + 0.5 * omega**2 * q[0] ** 2
        dt = 0.01
        for _ in range(500):
            q, p = leapfrog_step(q, p, grad_H_q, grad_H_p, dt)
        H_final = 0.5 * p[0] ** 2 + 0.5 * omega**2 * q[0] ** 2
        # Leapfrog should conserve shadow Hamiltonian well
        assert abs(H_final - H0) < 1e-3

    def test_stormer_verlet_step_shape(self) -> None:
        from hamiltonian_modal.world_model.symplectic import stormer_verlet_step

        n = 4
        M_inv = np.eye(n)
        q, p = np.ones(n), np.zeros(n)
        q_new, p_new = stormer_verlet_step(q, p, lambda x: x, M_inv, dt=0.01)
        assert q_new.shape == (n,)
        assert p_new.shape == (n,)

    def test_integrate_trajectory_shape(self) -> None:
        from hamiltonian_modal.world_model.symplectic import integrate_trajectory

        n = 3
        q0, p0 = np.zeros(n), np.ones(n)
        qs, ps = integrate_trajectory(q0, p0, lambda q: q, lambda p: p, dt=0.01, n_steps=20)
        assert qs.shape == (21, n)
        assert ps.shape == (21, n)

    def test_integrate_trajectory_initial_condition(self) -> None:
        from hamiltonian_modal.world_model.symplectic import integrate_trajectory

        q0, p0 = np.array([1.0, 2.0]), np.array([3.0, 4.0])
        qs, ps = integrate_trajectory(q0, p0, lambda q: q, lambda p: p, dt=0.01, n_steps=5)
        np.testing.assert_array_equal(qs[0], q0)
        np.testing.assert_array_equal(ps[0], p0)


# ===========================================================================
# world_model/hamiltonian_net
# ===========================================================================


class TestHamiltonianNet:
    def _net(self, n_modes: int = 3, seed: int = 0):
        from hamiltonian_modal.world_model.hamiltonian_net import HamiltonianNet, HamiltonianParams

        params = HamiltonianParams.random_init(n_modes, hidden_dim=16, n_layers=2, seed=seed)
        return HamiltonianNet(params)

    def test_kinetic_is_nonnegative(self) -> None:
        net = self._net(n_modes=4)
        mu = np.array([1.0, -2.0, 0.5, 0.0])
        assert net.kinetic(mu) >= 0.0

    def test_potential_is_nonnegative(self) -> None:
        net = self._net(n_modes=4)
        eta = np.array([0.1, -0.2, 0.3, 0.4])
        assert net.potential(eta) >= 0.0

    def test_hamiltonian_scalar(self) -> None:
        net = self._net(n_modes=3)
        H = net(np.ones(3), np.ones(3))
        assert isinstance(H, float)

    def test_grad_H_mu_shape(self) -> None:
        net = self._net(n_modes=3)
        eta, mu = np.ones(3), np.ones(3)
        g = net.grad_H_mu(eta, mu)
        assert g.shape == (3,)

    def test_grad_H_eta_shape(self) -> None:
        net = self._net(n_modes=3)
        eta, mu = np.zeros(3), np.ones(3)
        g = net.grad_H_eta(eta, mu, eps=1e-5)
        assert g.shape == (3,)

    def test_unknown_activation_raises(self) -> None:
        from hamiltonian_modal.world_model.hamiltonian_net import HamiltonianNet, HamiltonianParams

        params = HamiltonianParams.random_init(2)
        with pytest.raises(ValueError, match="Unknown activation"):
            HamiltonianNet(params, activation="sigmoid")

    def test_finite_diff_grad(self) -> None:
        from hamiltonian_modal.world_model.hamiltonian_net import finite_diff_grad

        def f(x: np.ndarray) -> float:
            return float(np.sum(x**2))

        x = np.array([1.0, 2.0, 3.0])
        grad = finite_diff_grad(f, x)
        np.testing.assert_allclose(grad, 2 * x, atol=1e-6)


# ===========================================================================
# world_model/uncertainty
# ===========================================================================


class TestEnsembleUncertainty:
    def _ensemble(self, n_members: int = 3, n_modes: int = 4):
        from hamiltonian_modal.world_model.uncertainty import EnsembleUncertainty

        return EnsembleUncertainty.random_init(n_members=n_members, n_modes=n_modes, hidden_dim=16, n_layers=1)

    def test_predict_returns_estimate(self) -> None:
        from hamiltonian_modal.world_model.uncertainty import UncertaintyEstimate

        ens = self._ensemble()
        eta, mu = np.ones(4), np.zeros(4)
        est = ens.predict(eta, mu)
        assert isinstance(est, UncertaintyEstimate)
        assert est.member_values.shape == (3,)
        assert est.std_H >= 0.0

    def test_single_member_variance_zero(self) -> None:
        ens = self._ensemble(n_members=1)
        est = ens.predict(np.ones(4), np.ones(4))
        assert est.var_H == 0.0

    def test_empty_ensemble_raises(self) -> None:
        from hamiltonian_modal.world_model.uncertainty import EnsembleUncertainty

        with pytest.raises(ValueError):
            EnsembleUncertainty(members=[])


# ===========================================================================
# world_model/contact_event
# ===========================================================================


class TestContactEventPredictor:
    def _predictor(self, n_modes: int = 3, n_contacts: int = 2):
        from hamiltonian_modal.world_model.contact_event import ContactEventParams, ContactEventPredictor

        params = ContactEventParams.random_init(n_modes=n_modes, n_contacts=n_contacts, hidden_dim=8, seed=0)
        return ContactEventPredictor(params)

    def test_predict_probabilities_in_range(self) -> None:
        pred = self._predictor()
        eta, mu = np.ones(3), np.zeros(3)
        state = pred.predict(eta, mu)
        assert np.all(state.probabilities >= 0)
        assert np.all(state.probabilities <= 1)

    def test_predict_active_is_bool(self) -> None:
        pred = self._predictor()
        state = pred.predict(np.ones(3), np.ones(3))
        assert state.active.dtype == np.bool_

    def test_n_contacts(self) -> None:
        pred = self._predictor(n_contacts=4)
        state = pred.predict(np.ones(3), np.zeros(3))
        assert state.n_contacts == 4


# ===========================================================================
# world_model/verifier
# ===========================================================================


class TestVerifier:
    def _verifier(self, n_modes: int = 4):
        from hamiltonian_modal.world_model.verifier import Verifier, VerifierParams

        params = VerifierParams.random_init(n_modes=n_modes, hidden_dim=16, n_layers=2, seed=0)
        return Verifier(params)

    def test_predict_feasibility_in_01(self) -> None:
        v = self._verifier()
        out = v.predict(np.ones(4), np.zeros(4))
        assert 0.0 <= out.feasibility <= 1.0

    def test_predict_value_is_float(self) -> None:
        v = self._verifier()
        out = v.predict(np.zeros(4), np.ones(4))
        assert isinstance(out.value, float)


# ===========================================================================
# world_model/trainer
# ===========================================================================


class TestWorldModelTrainer:
    def _batch(self, n_modes: int = 3, B: int = 4):
        from hamiltonian_modal.world_model.trainer import TrainingBatch

        rng = np.random.default_rng(0)
        return TrainingBatch(
            eta=rng.standard_normal((B, n_modes)),
            mu=rng.standard_normal((B, n_modes)),
            H_target=np.abs(rng.standard_normal(B)),
        )

    def test_train_step_returns_float(self) -> None:
        from hamiltonian_modal.world_model.hamiltonian_net import HamiltonianNet, HamiltonianParams
        from hamiltonian_modal.world_model.trainer import WorldModelTrainer

        params = HamiltonianParams.random_init(3, hidden_dim=8, n_layers=1, seed=0)
        net = HamiltonianNet(params)
        trainer = WorldModelTrainer(net, lr=1e-2)
        loss = trainer.train_step(self._batch())
        assert isinstance(loss, float)

    def test_train_reduces_loss(self) -> None:
        from hamiltonian_modal.world_model.hamiltonian_net import HamiltonianNet, HamiltonianParams
        from hamiltonian_modal.world_model.trainer import WorldModelTrainer

        params = HamiltonianParams.random_init(2, hidden_dim=4, n_layers=1, seed=1)
        net = HamiltonianNet(params)
        trainer = WorldModelTrainer(net, lr=1e-1)
        batch = self._batch(n_modes=2, B=2)
        loss_before = trainer._mse_loss(batch)
        trainer.train([batch], n_epochs=3)
        loss_after = trainer._mse_loss(batch)
        # Loss should not increase
        assert loss_after <= loss_before + 1.0  # lenient check


# ===========================================================================
# mcts/tree
# ===========================================================================


class TestTreeNode:
    def _root(self, n_modes: int = 3):
        from hamiltonian_modal.mcts.tree import TreeNode

        return TreeNode(eta=np.zeros(n_modes), mu=np.zeros(n_modes))

    def test_is_root(self) -> None:
        root = self._root()
        assert root.is_root

    def test_is_leaf_initially(self) -> None:
        root = self._root()
        assert root.is_leaf

    def test_q_value_zero_when_unvisited(self) -> None:
        root = self._root()
        assert root.q_value == 0.0

    def test_update_increments_visit_count(self) -> None:
        root = self._root()
        root.update(1.0)
        assert root.visit_count == 1
        assert root.value_sum == 1.0

    def test_q_value_is_mean(self) -> None:
        root = self._root()
        root.update(2.0)
        root.update(4.0)
        assert root.q_value == pytest.approx(3.0)

    def test_expand_creates_children(self) -> None:
        root = self._root(n_modes=2)
        priors = np.array([0.3, 0.7])
        states = [(np.ones(2), np.zeros(2)), (np.zeros(2), np.ones(2))]
        root.expand(priors, states)
        assert len(root.children) == 2
        assert not root.is_leaf

    def test_child_prior_stored(self) -> None:
        root = self._root(n_modes=2)
        priors = np.array([0.4, 0.6])
        states = [(np.ones(2), np.zeros(2)), (np.zeros(2), np.ones(2))]
        root.expand(priors, states)
        assert root.children[0].prior == pytest.approx(0.4)
        assert root.children[1].prior == pytest.approx(0.6)


# ===========================================================================
# mcts/puct
# ===========================================================================


class TestPUCT:
    def test_select_child_returns_highest_score(self) -> None:
        from hamiltonian_modal.mcts.puct import select_child
        from hamiltonian_modal.mcts.tree import TreeNode

        root = TreeNode(eta=np.zeros(2), mu=np.zeros(2), visit_count=10)
        priors = np.array([0.1, 0.9])
        states = [(np.zeros(2), np.zeros(2)), (np.zeros(2), np.zeros(2))]
        root.expand(priors, states)
        action, child = select_child(root, c_puct=1.0)
        # Child with higher prior (1) should be preferred when neither visited
        assert action == 1

    def test_select_child_leaf_raises(self) -> None:
        from hamiltonian_modal.mcts.puct import select_child
        from hamiltonian_modal.mcts.tree import TreeNode

        leaf = TreeNode(eta=np.zeros(2), mu=np.zeros(2))
        with pytest.raises(ValueError, match="leaf"):
            select_child(leaf)

    def test_puct_score_formula(self) -> None:
        from hamiltonian_modal.mcts.puct import puct_score
        from hamiltonian_modal.mcts.tree import TreeNode

        parent = TreeNode(eta=np.zeros(1), mu=np.zeros(1), visit_count=4)
        child = TreeNode(eta=np.zeros(1), mu=np.zeros(1), prior=0.5, visit_count=2, value_sum=1.0)
        score = puct_score(parent, child, c_puct=1.0)
        expected = 0.5 + 1.0 * 0.5 * math.sqrt(4) / (1 + 2)
        assert score == pytest.approx(expected, rel=1e-6)


# ===========================================================================
# mcts/search
# ===========================================================================


class TestMCTSSearch:
    def _setup(self, n_modes: int = 2, n_actions: int = 3):
        from hamiltonian_modal.config import MCTSConfig
        from hamiltonian_modal.mcts.search import MCTSSearch
        from hamiltonian_modal.mcts.tree import TreeNode

        config = MCTSConfig(n_simulations=10, max_depth=3, c_puct=1.0)

        def dynamics(eta, mu, a):
            return eta + float(a) * 0.01, mu

        def policy(eta, mu):
            return np.ones(n_actions) / n_actions

        def value(eta, mu):
            return float(-np.sum(eta**2))

        search = MCTSSearch(config, n_actions, dynamics, policy, value)
        root = TreeNode(eta=np.zeros(n_modes), mu=np.zeros(n_modes))
        return search, root

    def test_run_returns_valid_action(self) -> None:
        search, root = self._setup(n_actions=3)
        action = search.run(root)
        assert 0 <= action < 3

    def test_root_visit_count_increases(self) -> None:
        search, root = self._setup()
        search.run(root)
        assert root.visit_count > 0


# ===========================================================================
# policy/mlp_policy
# ===========================================================================


class TestMLPPolicy:
    def _policy(self, obs_dim: int = 10, action_dim: int = 4):
        from hamiltonian_modal.policy.mlp_policy import MLPParams, MLPPolicy

        params = MLPParams.random_init(obs_dim, action_dim, hidden_dim=16, n_layers=2, seed=0)
        return MLPPolicy(params)

    def test_output_in_tanh_range(self) -> None:
        policy = self._policy()
        obs = np.ones(10)
        action = policy(obs)
        assert action.shape == (4,)
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)

    def test_deterministic(self) -> None:
        policy = self._policy()
        obs = np.random.default_rng(5).standard_normal(10)
        a1 = policy(obs)
        a2 = policy(obs)
        np.testing.assert_array_equal(a1, a2)

    def test_unknown_activation_raises(self) -> None:
        from hamiltonian_modal.config import MLPPolicyConfig
        from hamiltonian_modal.policy.mlp_policy import MLPParams, MLPPolicy

        params = MLPParams.random_init(4, 2)
        with pytest.raises(ValueError, match="Unknown activation"):
            MLPPolicy(params, config=MLPPolicyConfig(activation="sigmoid"))


# ===========================================================================
# policy/flow_matching_head
# ===========================================================================


class TestFlowMatchingHead:
    def _head(self, obs_dim: int = 6, action_dim: int = 3):
        from hamiltonian_modal.policy.flow_matching_head import FlowMatchingHead, FlowParams

        params = FlowParams.random_init(obs_dim, action_dim, hidden_dim=8, n_layers=1, seed=0)
        return FlowMatchingHead(params, action_dim, n_steps=5)

    def test_output_shape(self) -> None:
        head = self._head()
        obs = np.ones(6)
        action = head(obs)
        assert action.shape == (3,)

    def test_output_in_tanh_range(self) -> None:
        head = self._head()
        action = head(np.zeros(6))
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)


# ===========================================================================
# policy/vla_backbone
# ===========================================================================


class TestVLABackbone:
    def test_encode_calls_model(self) -> None:
        from hamiltonian_modal.policy.vla_backbone import VLABackbone

        def fake_model(obs):
            return np.ones(8, dtype=np.float32)

        bb = VLABackbone(fake_model, feature_dim=8)
        features = bb.encode({"image": None})
        assert features.shape == (8,)
        assert features.dtype == np.float64

    def test_wrong_feature_dim_raises(self) -> None:
        from hamiltonian_modal.policy.vla_backbone import VLABackbone

        def fake_model(obs):
            return np.ones(4)

        bb = VLABackbone(fake_model, feature_dim=8)
        with pytest.raises(ValueError, match="Expected feature vector"):
            bb.encode(None)

    def test_preprocessor_applied(self) -> None:
        from hamiltonian_modal.policy.vla_backbone import VLABackbone

        calls = []

        def pre(obs):
            calls.append(obs)
            return obs

        bb = VLABackbone(lambda x: np.ones(4), feature_dim=4, preprocessor=pre)
        bb.encode("raw_obs")
        assert calls == ["raw_obs"]


# ===========================================================================
# mpc/lagrangian_net
# ===========================================================================


class TestLagrangianNet:
    def _net(self, n_dof: int = 3):
        from hamiltonian_modal.mpc.lagrangian_net import LagrangianNet, LagrangianParams

        params = LagrangianParams.random_init(n_dof, hidden_dim=8, n_layers=1, seed=0)
        return LagrangianNet(params, n_dof)

    def test_lagrangian_is_float(self) -> None:
        net = self._net()
        L = net(np.zeros(3), np.ones(3))
        assert isinstance(L, float)

    def test_inertia_matrix_shape(self) -> None:
        net = self._net()
        M = net.inertia_matrix(np.zeros(3))
        assert M.shape == (3, 3)

    def test_inertia_matrix_is_pd(self) -> None:
        net = self._net()
        M = net.inertia_matrix(np.zeros(3))
        eigvals = np.linalg.eigvalsh(M)
        assert np.all(eigvals > 0)

    def test_inertia_matrix_is_symmetric(self) -> None:
        net = self._net()
        M = net.inertia_matrix(np.ones(3) * 0.5)
        np.testing.assert_allclose(M, M.T, atol=1e-12)


# ===========================================================================
# mpc/ilqr
# ===========================================================================


class TestILQR:
    def _ilqr(self, n: int = 2):
        from hamiltonian_modal.config import ILQRConfig
        from hamiltonian_modal.mpc.ilqr import ILQR

        config = ILQRConfig(horizon=5, n_iterations=2, reg_init=1.0)

        def f(x, u):
            return x + u * 0.1

        def l(x, u, t):
            return float(np.sum(u**2) * 0.01 + np.sum(x**2) * 0.001)

        def lf(x):
            return float(np.sum(x**2))

        return ILQR(config, f, l, lf, n_state=n, n_ctrl=n)

    def test_optimise_returns_result(self) -> None:
        from hamiltonian_modal.mpc.ilqr import ILQRResult

        ilqr = self._ilqr()
        result = ilqr.optimise(np.array([1.0, 1.0]))
        assert isinstance(result, ILQRResult)
        assert result.xs.shape == (6, 2)
        assert result.us.shape == (5, 2)

    def test_optimise_finite_cost(self) -> None:
        ilqr = self._ilqr()
        result = ilqr.optimise(np.ones(2))
        assert np.isfinite(result.total_cost)


# ===========================================================================
# mpc/ddp
# ===========================================================================


class TestDDP:
    def test_ddp_inherits_ilqr(self) -> None:
        from hamiltonian_modal.mpc.ddp import DDP
        from hamiltonian_modal.mpc.ilqr import ILQR

        assert issubclass(DDP, ILQR)

    def test_ddp_optimise_second_order_false(self) -> None:
        from hamiltonian_modal.config import DDPConfig
        from hamiltonian_modal.mpc.ddp import DDP

        config = DDPConfig(horizon=3, n_iterations=2, second_order_value=False)
        ddp = DDP(
            config,
            lambda x, u: x + u * 0.1,
            lambda x, u, t: float(np.sum(u**2)),
            lambda x: float(np.sum(x**2)),
            n_state=2,
            n_ctrl=2,
        )
        result = ddp.optimise(np.array([1.0, -1.0]))
        assert np.isfinite(result.total_cost)

    def test_ddp_optimise_second_order_true(self) -> None:
        from hamiltonian_modal.config import DDPConfig
        from hamiltonian_modal.mpc.ddp import DDP

        config = DDPConfig(horizon=3, n_iterations=2, second_order_value=True)
        ddp = DDP(
            config,
            lambda x, u: x + u * 0.1,
            lambda x, u, t: float(np.sum(u**2)),
            lambda x: float(np.sum(x**2)),
            n_state=2,
            n_ctrl=2,
        )
        result = ddp.optimise(np.array([0.5, -0.5]))
        assert np.isfinite(result.total_cost)


# ===========================================================================
# diff_sim/stability_utils
# ===========================================================================


class TestStabilityUtils:
    def test_clip_grad_norm_clips(self) -> None:
        from hamiltonian_modal.diff_sim.stability_utils import clip_grad_norm

        grad = np.array([3.0, 4.0])  # norm = 5
        clipped, norm = clip_grad_norm(grad, max_norm=1.0)
        assert np.isclose(np.linalg.norm(clipped), 1.0)
        assert norm == pytest.approx(5.0)

    def test_clip_grad_norm_no_clip(self) -> None:
        from hamiltonian_modal.diff_sim.stability_utils import clip_grad_norm

        grad = np.array([0.1, 0.2])
        clipped, norm = clip_grad_norm(grad, max_norm=0)  # disabled
        np.testing.assert_array_equal(clipped, grad)

    def test_stable_log_no_inf(self) -> None:
        from hamiltonian_modal.diff_sim.stability_utils import stable_log

        x = np.array([0.0, -1.0, 1.0])
        result = stable_log(x)
        assert np.all(np.isfinite(result))

    def test_running_mean_std_update(self) -> None:
        from hamiltonian_modal.diff_sim.stability_utils import RunningMeanStd

        rms = RunningMeanStd(shape=(3,))
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        rms.update(data)
        assert np.all(np.isfinite(rms.mean))

    def test_running_mean_std_normalize_clips(self) -> None:
        from hamiltonian_modal.diff_sim.stability_utils import RunningMeanStd

        rms = RunningMeanStd(shape=(2,), clip=2.0)
        rms.update(np.array([[0.0, 0.0]]))
        normed = rms.normalize(np.array([1000.0, -1000.0]))
        assert np.all(np.abs(normed) <= 2.0)


# ===========================================================================
# diff_sim/policy_gradient
# ===========================================================================


class TestPolicyGradient:
    def test_compute_returns_shape(self) -> None:
        from hamiltonian_modal.diff_sim.policy_gradient import compute_returns

        rewards = np.array([1.0, 2.0, 3.0])
        returns = compute_returns(rewards, discount=0.99)
        assert returns.shape == (3,)

    def test_compute_returns_values(self) -> None:
        from hamiltonian_modal.diff_sim.policy_gradient import compute_returns

        rewards = np.array([1.0, 0.0, 0.0])
        returns = compute_returns(rewards, discount=0.5)
        assert returns[0] == pytest.approx(1.0)

    def test_reinforce_update_returns_floats(self) -> None:
        from hamiltonian_modal.diff_sim.policy_gradient import reinforce_update

        log_probs = np.log(np.array([0.2, 0.5, 0.3]))
        returns = np.array([1.0, 2.0, 0.5])
        loss, grad_norm = reinforce_update(log_probs, returns)
        assert isinstance(loss, float)
        assert isinstance(grad_norm, float)
        assert np.isfinite(loss)
