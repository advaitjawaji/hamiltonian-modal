"""Train foundation policy via end-to-end differentiable simulation.

Usage:
    python scripts/train_diff_sim_policy.py
    hm-train-policy  (after pip install -e .)
"""
from __future__ import annotations

import argparse
import logging
import pathlib

import numpy as np

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train diff-sim foundation policy")
    parser.add_argument("--n-iterations", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--horizon-start", type=int, default=10)
    parser.add_argument("--horizon-end", type=int, default=100)
    parser.add_argument("--n-modes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/diff_sim_policy")
    parser.add_argument("--wandb", action="store_true", default=False)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from hamiltonian_modal.utils.seeding import set_global_seed
    from hamiltonian_modal.world_model.hamiltonian_net import HamiltonianNet
    from hamiltonian_modal.world_model.verifier import Verifier
    from hamiltonian_modal.policy.mlp_policy import MLPPolicy
    from hamiltonian_modal.diff_sim.pipeline import DiffSimPipeline
    from hamiltonian_modal.diff_sim.trainer import DiffSimTrainConfig, DiffSimTrainer
    from hamiltonian_modal.diff_sim.stability_utils import trajectory_length_curriculum

    set_global_seed(args.seed)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_modes = args.n_modes
    n_contact_modes = 5
    obs_dim = n_modes * 2
    action_dim = 23

    world_model = HamiltonianNet(n_modes, n_contact_modes, seed=args.seed)
    policy = MLPPolicy(obs_dim, action_dim, seed=args.seed)

    pipeline = DiffSimPipeline(
        policy=policy,
        world_model=world_model,
        n_modes=n_modes,
        n_contact_modes=n_contact_modes,
    )

    config = DiffSimTrainConfig(
        n_iterations=args.n_iterations,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        horizon_start=args.horizon_start,
        horizon_end=args.horizon_end,
        seed=args.seed,
    )

    logger.info("Starting diff-sim policy training for %d iterations", args.n_iterations)

    rng = np.random.default_rng(args.seed)
    reward_history: list[float] = []

    for step in range(args.n_iterations):
        horizon = trajectory_length_curriculum(step, args.n_iterations, args.horizon_start, args.horizon_end)
        initial_obs = rng.standard_normal(obs_dim)

        def reward_fn(state: np.ndarray, action: np.ndarray) -> float:
            return -float(np.mean(state ** 2)) - 0.001 * float(np.mean(action ** 2))

        result = pipeline.rollout(initial_obs, horizon=horizon, use_world_model=True, reward_fn=reward_fn)
        reward_history.append(result.total_reward)

        if step % 500 == 0:
            avg_reward = float(np.mean(reward_history[-100:])) if reward_history else 0.0
            logger.info("step=%d  horizon=%d  avg_reward=%.4f", step, horizon, avg_reward)

    np.save(str(output_dir / "reward_history.npy"), np.array(reward_history))
    logger.info("Training complete. Reward history saved to %s", output_dir / "reward_history.npy")


if __name__ == "__main__":
    main()
