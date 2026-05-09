"""Run DiffSim-EffBench: sample efficiency vs. PPO baseline.

Head-to-head comparison: end-to-end diff-sim policy gradient vs. PPO.
Plots reward vs. wall-clock time.

This produces Fig 5 in the paper (headline figure for Headline A).

Usage:
    python scripts/run_diffsim_effbench.py
    hm-effbench  (after pip install -e .)
"""
from __future__ import annotations

import argparse
import logging
import pathlib

import numpy as np

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DiffSim-EffBench sample efficiency benchmark")
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/effbench")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from benchmarks.diffsim_effbench.runner import run_benchmark

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running DiffSim-EffBench (%d steps) ...", args.n_steps)
    reports = run_benchmark(n_steps=args.n_steps, seed=args.seed)

    print("\n=== DiffSim-EffBench Results ===")
    print(f"{'Method':<25}  {'Final Reward':>14}  {'80% Threshold Step':>20}")
    print("-" * 65)
    for r in reports:
        print(f"{r.method_name:<25}  {r.final_reward:>14.4f}  {r.steps_to_80pct_threshold:>20}")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ["tab:blue", "tab:red"]
        for r, color in zip(reports, colors):
            steps = np.arange(len(r.reward_curve))
            ax.plot(steps, r.reward_curve, color=color, linewidth=2, label=r.method_name)

        ax.set_xlabel("Training steps", fontsize=12)
        ax.set_ylabel("Task reward", fontsize=12)
        ax.set_title("DiffSim-EffBench: Sample Efficiency\n(Fig 5 — Headline A)", fontsize=12)
        ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5, label="80% quality threshold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = output_dir / "diffsim_effbench_results.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Figure saved to %s", out_path)
        plt.close()
    except ImportError:
        logger.warning("matplotlib not installed; skipping figure")


if __name__ == "__main__":
    main()
