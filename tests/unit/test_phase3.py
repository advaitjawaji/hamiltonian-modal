"""Unit tests for Phase 3 — benchmarks and baselines.

Covers:

* ``benchmarks.driftbench_g1.tasks``
* ``benchmarks.driftbench_g1.metrics``
* ``benchmarks.driftbench_g1.runner``
* ``benchmarks.diffsim_effbench.tasks``
* ``benchmarks.diffsim_effbench.metrics``
* ``benchmarks.diffsim_effbench.runner``
* ``baselines.dreamerv3``
* ``baselines.ppo_genesis``
* ``baselines.puppeteer``
* ``baselines.roboscape``
* ``baselines.td_mpc2``
* ``baselines.vjepa2``
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# DriftBench-G1 — tasks
# ---------------------------------------------------------------------------


class TestDriftBenchTasks:
    def test_run_task_leapfrog_returns_result(self) -> None:
        from benchmarks.driftbench_g1.tasks import DriftTask, run_task

        task = DriftTask("test_leapfrog", n_dof=3, n_steps=100, dt=0.01, integrator="leapfrog")
        result = run_task(task)
        assert result.task is task
        assert result.energies.shape == (101,)
        assert np.isfinite(result.H0)

    def test_run_task_euler_returns_result(self) -> None:
        from benchmarks.driftbench_g1.tasks import DriftTask, run_task

        task = DriftTask("test_euler", n_dof=3, n_steps=100, dt=0.01, integrator="euler")
        result = run_task(task)
        assert result.energies.shape == (101,)

    def test_leapfrog_drift_less_than_euler(self) -> None:
        from benchmarks.driftbench_g1.tasks import DriftTask, run_task

        leapfrog = run_task(DriftTask("lf", n_dof=4, n_steps=500, dt=0.01, integrator="leapfrog"))
        euler = run_task(DriftTask("eu", n_dof=4, n_steps=500, dt=0.01, integrator="euler"))
        lf_drift = float(np.max(np.abs(leapfrog.energies - leapfrog.H0))) / abs(leapfrog.H0)
        eu_drift = float(np.max(np.abs(euler.energies - euler.H0))) / abs(euler.H0)
        assert lf_drift < eu_drift

    def test_invalid_integrator_raises(self) -> None:
        from benchmarks.driftbench_g1.tasks import DriftTask, run_task

        task = DriftTask("bad", n_dof=3, n_steps=10, integrator="runge_kutta")
        with pytest.raises(ValueError, match="Unknown integrator"):
            run_task(task)

    def test_standard_tasks_list_nonempty(self) -> None:
        from benchmarks.driftbench_g1.tasks import STANDARD_TASKS

        assert len(STANDARD_TASKS) >= 2

    def test_energies_all_finite(self) -> None:
        from benchmarks.driftbench_g1.tasks import DriftTask, run_task

        task = DriftTask("fin", n_dof=5, n_steps=200, dt=0.005, integrator="leapfrog")
        result = run_task(task)
        assert np.all(np.isfinite(result.energies))


# ---------------------------------------------------------------------------
# DriftBench-G1 — metrics
# ---------------------------------------------------------------------------


class TestDriftBenchMetrics:
    def test_relative_drift_zero_for_constant_energy(self) -> None:
        from benchmarks.driftbench_g1.metrics import relative_energy_drift

        energies = np.full(100, 5.0)
        assert relative_energy_drift(energies, H0=5.0) == pytest.approx(0.0)

    def test_relative_drift_correct(self) -> None:
        from benchmarks.driftbench_g1.metrics import relative_energy_drift

        energies = np.array([10.0, 10.5, 9.8])
        drift = relative_energy_drift(energies, H0=10.0)
        assert drift == pytest.approx(0.05)

    def test_relative_drift_raises_for_zero_H0(self) -> None:
        from benchmarks.driftbench_g1.metrics import relative_energy_drift

        with pytest.raises(ValueError, match="too small"):
            relative_energy_drift(np.ones(5), H0=0.0)

    def test_mean_drift_rate_positive_for_growing_error(self) -> None:
        from benchmarks.driftbench_g1.metrics import mean_drift_rate

        energies = np.linspace(1.0, 2.0, 100)  # growing energy
        rate = mean_drift_rate(energies, H0=1.0, dt=0.01)
        assert rate > 0.0

    def test_drift_summary_keys(self) -> None:
        from benchmarks.driftbench_g1.metrics import drift_summary

        s = drift_summary(np.full(50, 3.0), H0=3.0, dt=0.01)
        for key in ("relative_drift", "drift_rate", "final_drift"):
            assert key in s

    def test_drift_summary_values_finite(self) -> None:
        from benchmarks.driftbench_g1.metrics import drift_summary

        energies = np.array([5.0, 5.1, 4.9, 5.2])
        s = drift_summary(energies, H0=5.0, dt=0.01)
        assert all(np.isfinite(v) for v in s.values())


# ---------------------------------------------------------------------------
# DriftBench-G1 — runner
# ---------------------------------------------------------------------------


class TestDriftBenchRunner:
    def test_run_benchmark_returns_reports(self) -> None:
        from benchmarks.driftbench_g1.tasks import DriftTask
        from benchmarks.driftbench_g1.runner import run_benchmark

        tasks = [DriftTask("t1", n_dof=3, n_steps=50, integrator="leapfrog")]
        reports = run_benchmark(tasks)
        assert len(reports) == 1
        assert "relative_drift" in reports[0].metrics

    def test_run_benchmark_default_tasks(self) -> None:
        from benchmarks.driftbench_g1.runner import run_benchmark

        reports = run_benchmark()
        assert len(reports) >= 2

    def test_report_wallclock_positive(self) -> None:
        from benchmarks.driftbench_g1.tasks import DriftTask
        from benchmarks.driftbench_g1.runner import run_benchmark

        tasks = [DriftTask("t", n_dof=3, n_steps=20, integrator="leapfrog")]
        reports = run_benchmark(tasks)
        assert reports[0].wallclock_s >= 0.0


# ---------------------------------------------------------------------------
# DiffSim-EffBench — tasks
# ---------------------------------------------------------------------------


class TestDiffSimEffTasks:
    def test_run_eff_task_returns_result(self) -> None:
        from benchmarks.diffsim_effbench.tasks import EffTask, run_eff_task

        task = EffTask("small", n_dof=6, n_modes=3, horizon=10, batch_size=2)
        result = run_eff_task(task)
        assert result.task is task
        assert np.isfinite(result.mean_return)
        assert result.grad_norm >= 0.0
        assert result.steps_per_second > 0.0

    def test_standard_eff_tasks_nonempty(self) -> None:
        from benchmarks.diffsim_effbench.tasks import STANDARD_EFF_TASKS

        assert len(STANDARD_EFF_TASKS) >= 2

    def test_larger_horizon_slower(self) -> None:
        from benchmarks.diffsim_effbench.tasks import EffTask, run_eff_task

        short = run_eff_task(EffTask("sh", n_dof=6, n_modes=3, horizon=5, batch_size=1))
        long_ = run_eff_task(EffTask("lo", n_dof=6, n_modes=3, horizon=100, batch_size=1))
        assert short.wallclock_s <= long_.wallclock_s * 100  # sanity bound

    def test_eff_result_attributes_present(self) -> None:
        from benchmarks.diffsim_effbench.tasks import EffTask, run_eff_task

        result = run_eff_task(EffTask("attr", n_dof=4, n_modes=2, horizon=5, batch_size=2))
        assert hasattr(result, "mean_return")
        assert hasattr(result, "grad_norm")
        assert hasattr(result, "grad_variance")
        assert hasattr(result, "wallclock_s")
        assert hasattr(result, "steps_per_second")


# ---------------------------------------------------------------------------
# DiffSim-EffBench — metrics
# ---------------------------------------------------------------------------


class TestDiffSimEffMetrics:
    def test_gradient_snr_positive(self) -> None:
        from benchmarks.diffsim_effbench.metrics import gradient_snr

        snr = gradient_snr(grad_norm=2.0, grad_variance=0.5)
        assert snr > 0.0

    def test_gradient_snr_zero_norm(self) -> None:
        from benchmarks.diffsim_effbench.metrics import gradient_snr

        snr = gradient_snr(grad_norm=0.0, grad_variance=1.0)
        assert snr == pytest.approx(0.0, abs=1e-10)

    def test_compute_efficiency_score_normalised(self) -> None:
        from benchmarks.diffsim_effbench.metrics import compute_efficiency_score

        score = compute_efficiency_score(steps_per_second=1000.0, n_dof=10)
        assert score == pytest.approx(100.0)

    def test_eff_summary_keys(self) -> None:
        from benchmarks.diffsim_effbench.metrics import eff_summary

        s = eff_summary(
            mean_return=-1.0,
            grad_norm=0.5,
            grad_variance=0.1,
            steps_per_second=500.0,
            n_dof=5,
        )
        for key in ("mean_return", "grad_norm", "grad_variance", "gradient_snr",
                    "efficiency_score", "steps_per_second"):
            assert key in s

    def test_eff_summary_values_finite(self) -> None:
        from benchmarks.diffsim_effbench.metrics import eff_summary

        s = eff_summary(-1.0, 0.5, 0.1, 500.0, 5)
        assert all(np.isfinite(v) for v in s.values())


# ---------------------------------------------------------------------------
# DiffSim-EffBench — runner
# ---------------------------------------------------------------------------


class TestDiffSimEffRunner:
    def test_run_eff_benchmark_returns_reports(self) -> None:
        from benchmarks.diffsim_effbench.tasks import EffTask
        from benchmarks.diffsim_effbench.runner import run_eff_benchmark

        tasks = [EffTask("r1", n_dof=4, n_modes=2, horizon=5, batch_size=2)]
        reports = run_eff_benchmark(tasks)
        assert len(reports) == 1
        assert "gradient_snr" in reports[0].metrics

    def test_run_eff_benchmark_default_tasks(self) -> None:
        from benchmarks.diffsim_effbench.runner import run_eff_benchmark

        reports = run_eff_benchmark()
        assert len(reports) >= 2


# ---------------------------------------------------------------------------
# Baselines — DreamerV3
# ---------------------------------------------------------------------------


class TestDreamerV3Adapter:
    def test_imports(self) -> None:
        from baselines.dreamerv3.adapter import DreamerV3Adapter, DreamerV3Config

        cfg = DreamerV3Config()
        assert cfg.batch_size == 16

    def test_act_returns_zeros(self) -> None:
        from baselines.dreamerv3.adapter import DreamerV3Adapter

        adapter = DreamerV3Adapter(obs_dim=10, act_dim=6)
        action = adapter.act(np.zeros(10))
        assert action.shape == (6,)
        np.testing.assert_array_equal(action, 0.0)

    def test_observe_increments_step_count(self) -> None:
        from baselines.dreamerv3.adapter import DreamerV3Adapter

        adapter = DreamerV3Adapter(obs_dim=10, act_dim=6)
        adapter.observe(np.zeros(10), 1.0, False)
        adapter.observe(np.zeros(10), 0.5, True)
        assert adapter.step_count == 2

    def test_build_agent_raises_import_error(self) -> None:
        from baselines.dreamerv3.adapter import DreamerV3Adapter

        adapter = DreamerV3Adapter(obs_dim=10, act_dim=6)
        with pytest.raises(ImportError, match="dreamerv3"):
            adapter.build_agent()


# ---------------------------------------------------------------------------
# Baselines — PPO-Genesis
# ---------------------------------------------------------------------------


class TestPPOGenesisAdapter:
    def test_imports(self) -> None:
        from baselines.ppo_genesis.adapter import PPOGenesisAdapter, PPOGenesisConfig

        cfg = PPOGenesisConfig()
        assert cfg.n_steps == 2048

    def test_reset_returns_obs(self) -> None:
        from baselines.ppo_genesis.adapter import PPOGenesisAdapter

        adapter = PPOGenesisAdapter(obs_dim=8, act_dim=4)
        obs = adapter.reset()
        assert obs.shape == (8,)

    def test_act_returns_action_and_log_prob(self) -> None:
        from baselines.ppo_genesis.adapter import PPOGenesisAdapter

        adapter = PPOGenesisAdapter(obs_dim=8, act_dim=4)
        action, log_prob = adapter.act(np.zeros(8))
        assert action.shape == (4,)
        assert isinstance(log_prob, float)

    def test_episode_count_increments(self) -> None:
        from baselines.ppo_genesis.adapter import PPOGenesisAdapter

        adapter = PPOGenesisAdapter(obs_dim=8, act_dim=4)
        adapter.reset()
        adapter.reset()
        assert adapter.episode_count == 2

    def test_build_agent_raises_import_error(self) -> None:
        from baselines.ppo_genesis.adapter import PPOGenesisAdapter

        adapter = PPOGenesisAdapter(obs_dim=8, act_dim=4)
        with pytest.raises(ImportError, match="genesis"):
            adapter.build_agent()


# ---------------------------------------------------------------------------
# Baselines — Puppeteer
# ---------------------------------------------------------------------------


class TestPuppeteerAdapter:
    def test_imports(self) -> None:
        from baselines.puppeteer.adapter import PuppeteerAdapter, PuppeteerConfig

        cfg = PuppeteerConfig()
        assert cfg.tracking_weight == pytest.approx(0.7)

    def test_set_and_use_reference_clip(self) -> None:
        from baselines.puppeteer.adapter import PuppeteerAdapter

        adapter = PuppeteerAdapter(obs_dim=10, act_dim=6, n_dof=4)
        frames = np.random.randn(30, 4)
        adapter.set_reference_clip(frames)
        q_current = np.zeros(4)
        reward = adapter.tracking_reward(q_current, frame_idx=0)
        assert isinstance(reward, float)
        assert reward <= 0.0

    def test_set_reference_clip_wrong_shape_raises(self) -> None:
        from baselines.puppeteer.adapter import PuppeteerAdapter

        adapter = PuppeteerAdapter(obs_dim=10, act_dim=6, n_dof=4)
        with pytest.raises(ValueError, match="shape"):
            adapter.set_reference_clip(np.zeros((10, 5)))  # wrong n_dof

    def test_tracking_reward_without_clip_raises(self) -> None:
        from baselines.puppeteer.adapter import PuppeteerAdapter

        adapter = PuppeteerAdapter(obs_dim=10, act_dim=6, n_dof=4)
        with pytest.raises(RuntimeError, match="No reference clip"):
            adapter.tracking_reward(np.zeros(4), frame_idx=0)

    def test_build_agent_raises_import_error(self) -> None:
        from baselines.puppeteer.adapter import PuppeteerAdapter

        adapter = PuppeteerAdapter(obs_dim=10, act_dim=6)
        with pytest.raises(ImportError, match="puppeteer"):
            adapter.build_agent()


# ---------------------------------------------------------------------------
# Baselines — RoboScape
# ---------------------------------------------------------------------------


class TestRoboScapeAdapter:
    def test_imports(self) -> None:
        from baselines.roboscape.adapter import RoboScapeAdapter, RoboScapeConfig

        cfg = RoboScapeConfig()
        assert cfg.scene_embed_dim == 128

    def test_embed_observation_shape(self) -> None:
        from baselines.roboscape.adapter import RoboScapeAdapter

        adapter = RoboScapeAdapter(obs_dim=12, act_dim=6)
        embed = adapter.embed_observation(np.ones(12))
        assert embed.shape == (128,)

    def test_embed_observation_wrong_size_raises(self) -> None:
        from baselines.roboscape.adapter import RoboScapeAdapter

        adapter = RoboScapeAdapter(obs_dim=12, act_dim=6)
        with pytest.raises(ValueError, match="shape"):
            adapter.embed_observation(np.ones(5))

    def test_act_returns_zeros(self) -> None:
        from baselines.roboscape.adapter import RoboScapeAdapter

        adapter = RoboScapeAdapter(obs_dim=12, act_dim=6)
        action = adapter.act(np.zeros(12))
        assert action.shape == (6,)

    def test_build_agent_raises_import_error(self) -> None:
        from baselines.roboscape.adapter import RoboScapeAdapter

        adapter = RoboScapeAdapter(obs_dim=12, act_dim=6)
        with pytest.raises(ImportError, match="roboscape"):
            adapter.build_agent()


# ---------------------------------------------------------------------------
# Baselines — TD-MPC2
# ---------------------------------------------------------------------------


class TestTDMPC2Adapter:
    def test_imports(self) -> None:
        from baselines.td_mpc2.adapter import TDMPC2Adapter, TDMPC2Config

        cfg = TDMPC2Config()
        assert cfg.latent_dim == 512

    def test_encode_shape(self) -> None:
        from baselines.td_mpc2.adapter import TDMPC2Adapter

        adapter = TDMPC2Adapter(obs_dim=10, act_dim=6)
        z = adapter.encode(np.zeros(10))
        assert z.shape == (512,)

    def test_plan_shape(self) -> None:
        from baselines.td_mpc2.adapter import TDMPC2Adapter

        adapter = TDMPC2Adapter(obs_dim=10, act_dim=6)
        action = adapter.plan(np.zeros(10))
        assert action.shape == (6,)

    def test_plan_increments_step_count(self) -> None:
        from baselines.td_mpc2.adapter import TDMPC2Adapter

        adapter = TDMPC2Adapter(obs_dim=10, act_dim=6)
        for _ in range(5):
            adapter.plan(np.zeros(10))
        assert adapter.total_steps == 5

    def test_build_agent_raises_import_error(self) -> None:
        from baselines.td_mpc2.adapter import TDMPC2Adapter

        adapter = TDMPC2Adapter(obs_dim=10, act_dim=6)
        with pytest.raises(ImportError, match="tdmpc2"):
            adapter.build_agent()


# ---------------------------------------------------------------------------
# Baselines — VJEPA2
# ---------------------------------------------------------------------------


class TestVJEPA2Adapter:
    def test_imports(self) -> None:
        from baselines.vjepa2.adapter import VJEPA2Adapter, VJEPA2Config

        cfg = VJEPA2Config()
        assert cfg.embed_dim == 384

    def test_encode_frame_shape(self) -> None:
        from baselines.vjepa2.adapter import VJEPA2Adapter

        adapter = VJEPA2Adapter(obs_dim=10, act_dim=6)
        token = adapter.encode_frame(np.zeros(10))
        assert token.shape == (384,)

    def test_encode_frame_wrong_size_raises(self) -> None:
        from baselines.vjepa2.adapter import VJEPA2Adapter

        adapter = VJEPA2Adapter(obs_dim=10, act_dim=6)
        with pytest.raises(ValueError, match="shape"):
            adapter.encode_frame(np.zeros(5))

    def test_predict_next_tokens_shape(self) -> None:
        from baselines.vjepa2.adapter import VJEPA2Adapter

        adapter = VJEPA2Adapter(obs_dim=10, act_dim=6)
        for _ in range(8):
            adapter.encode_frame(np.random.randn(10))
        preds = adapter.predict_next_tokens()
        assert preds.shape == (4, 384)

    def test_predict_next_tokens_empty_context(self) -> None:
        from baselines.vjepa2.adapter import VJEPA2Adapter

        adapter = VJEPA2Adapter(obs_dim=10, act_dim=6)
        preds = adapter.predict_next_tokens()
        assert preds.shape == (4, 384)
        np.testing.assert_array_equal(preds, 0.0)

    def test_build_agent_raises_import_error(self) -> None:
        from baselines.vjepa2.adapter import VJEPA2Adapter

        adapter = VJEPA2Adapter(obs_dim=10, act_dim=6)
        with pytest.raises(ImportError, match="vjepa2"):
            adapter.build_agent()
