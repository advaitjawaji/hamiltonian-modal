"""Train Hamiltonian-modal world model on G1 trajectory data.

Usage:
    python scripts/train_world_model.py
    python scripts/train_world_model.py --n-iterations 10000 --n-modes 20
    hm-train-wm  (after pip install -e .)
"""
from __future__ import annotations

import argparse
import logging
import pathlib

import numpy as np

logger = logging.getLogger(__name__)


def _make_synthetic_dataset(
    n_trajectories: int,
    seq_len: int,
    n_modes: int,
    n_contact_modes: int,
    seed: int,
) -> "TrajectoryDataset":  # type: ignore[name-defined]
    from hamiltonian_modal.world_model.trainer import TrajectoryDataset

    rng = np.random.default_rng(seed)
    eta = rng.standard_normal((n_trajectories, seq_len, n_modes)).astype(np.float64)
    eta_dot = rng.standard_normal((n_trajectories, seq_len, n_modes)).astype(np.float64)
    contact_mode = rng.integers(0, n_contact_modes, size=(n_trajectories, seq_len)).astype(np.int32)
    return TrajectoryDataset(eta=eta, eta_dot=eta_dot, contact_mode=contact_mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Hamiltonian-modal world model")
    parser.add_argument("--n-iterations", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-modes", type=int, default=20)
    parser.add_argument("--n-contact-modes", type=int, default=5)
    parser.add_argument("--n-train-trajectories", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/world_model")
    parser.add_argument("--wandb", action="store_true", default=False)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from hamiltonian_modal.world_model.trainer import WorldModelTrainConfig, train_world_model
    from hamiltonian_modal.utils.seeding import set_global_seed

    set_global_seed(args.seed)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building synthetic dataset (%d trajectories, seq_len=%d)", args.n_train_trajectories, args.seq_len)
    train_data = _make_synthetic_dataset(
        args.n_train_trajectories, args.seq_len, args.n_modes, args.n_contact_modes, args.seed
    )
    val_data = _make_synthetic_dataset(
        max(100, args.n_train_trajectories // 10), args.seq_len, args.n_modes, args.n_contact_modes, args.seed + 1
    )

    config = WorldModelTrainConfig(
        n_iterations=args.n_iterations,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
    )

    logger.info("Training world model for %d iterations ...", args.n_iterations)
    model, history = train_world_model(config, train_data, val_data, log_to_wandb=args.wandb)

    checkpoint_path = output_dir / "world_model_checkpoint.npz"
    np.save(str(checkpoint_path) + ".npy", {"history": history})
    logger.info("Training complete. History saved to %s", checkpoint_path)

    final_loss = history["total_loss"][-1] if history["total_loss"] else float("nan")
    logger.info("Final training loss: %.6f", final_loss)


if __name__ == "__main__":
    main()
