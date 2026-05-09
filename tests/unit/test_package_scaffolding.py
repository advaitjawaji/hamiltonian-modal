from __future__ import annotations

from importlib import import_module
from pathlib import Path

import hamiltonian_modal


def test_package_exports_version() -> None:
    assert hamiltonian_modal.__version__ == "0.1.0"


def test_phase_one_scaffold_modules_import() -> None:
    modules = [
        "hamiltonian_modal.config",
        "hamiltonian_modal.modal.decomposition",
        "hamiltonian_modal.modal.encoder",
        "hamiltonian_modal.modal.decoder",
        "hamiltonian_modal.modal.basis_switching",
        "hamiltonian_modal.modal.stiffness",
        "hamiltonian_modal.world_model.hamiltonian_net",
        "hamiltonian_modal.world_model.symplectic",
        "hamiltonian_modal.world_model.uncertainty",
        "hamiltonian_modal.world_model.contact_event",
        "hamiltonian_modal.world_model.verifier",
        "hamiltonian_modal.world_model.trainer",
        "hamiltonian_modal.mcts.puct",
        "hamiltonian_modal.mcts.tree",
        "hamiltonian_modal.mcts.search",
        "hamiltonian_modal.policy.flow_matching_head",
        "hamiltonian_modal.policy.mlp_policy",
        "hamiltonian_modal.policy.vla_backbone",
        "hamiltonian_modal.mpc.lagrangian_net",
        "hamiltonian_modal.mpc.ilqr",
        "hamiltonian_modal.mpc.ddp",
        "hamiltonian_modal.diff_sim.pipeline",
        "hamiltonian_modal.diff_sim.policy_gradient",
        "hamiltonian_modal.diff_sim.trainer",
        "hamiltonian_modal.diff_sim.stability_utils",
        "hamiltonian_modal.envs.g1_base",
        "hamiltonian_modal.envs.g1_walking",
        "hamiltonian_modal.envs.g1_fast_walking",
        "hamiltonian_modal.envs.g1_push_recovery",
        "hamiltonian_modal.envs.g1_jumping",
        "hamiltonian_modal.envs.g1_disturbance",
        "hamiltonian_modal.data.retargeted_motions",
        "hamiltonian_modal.data.lafan1_g1",
        "hamiltonian_modal.data.amass_g1",
        "hamiltonian_modal.utils.genesis_wrapper",
        "hamiltonian_modal.utils.logging",
        "hamiltonian_modal.utils.seeding",
    ]
    for module_name in modules:
        assert import_module(module_name) is not None


def test_phase_one_scaffold_files_exist() -> None:
    root = Path(__file__).resolve().parents[2]
    expected_paths = [
        ".python-version",
        ".pre-commit-config.yaml",
        "AGENTS.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "docs/theory/modal_decomposition.md",
        "docs/theory/hamiltonian_dynamics.md",
        "docs/theory/bounded_drift_theorem.md",
        "docs/theory/symplectic_integration.md",
        "docs/tutorials/01_quickstart.md",
        "docs/tutorials/02_modal_decomp_g1.md",
        "docs/tutorials/03_world_model_training.md",
        "docs/tutorials/04_diff_sim_policy_grad.md",
        "docs/tutorials/05_mcts_disturbance_recovery.md",
        "benchmarks/diffsim_effbench/tasks.py",
        "benchmarks/diffsim_effbench/runner.py",
        "benchmarks/diffsim_effbench/metrics.py",
        "benchmarks/driftbench_g1/tasks.py",
        "benchmarks/driftbench_g1/runner.py",
        "benchmarks/driftbench_g1/metrics.py",
        "baselines/ppo_genesis/.gitkeep",
        "baselines/dreamerv3/.gitkeep",
        "baselines/td_mpc2/.gitkeep",
        "baselines/puppeteer/.gitkeep",
        "baselines/vjepa2/.gitkeep",
        "baselines/roboscape/.gitkeep",
    ]
    for relative_path in expected_paths:
        assert (root / relative_path).exists(), relative_path
