"""Run DriftBench-G1: energy drift comparison across world models.

Rolls out 1000 random initial conditions x 100 timesteps and plots
energy-drift-vs-depth on log scale with Ch²t theoretical bound.

This produces Fig 6 in the paper (headline figure for Headline B).

Usage:
    python scripts/run_driftbench.py
    hm-driftbench  (after pip install -e .)
"""
from __future__ import annotations

import argparse
import logging
import pathlib

import numpy as np

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DriftBench-G1 energy drift benchmark")
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--n-modes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/driftbench")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from benchmarks.driftbench_g1.runner import run_benchmark

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running DriftBench-G1 (%d trials, %d steps) ...", args.n_trials, args.n_steps)
    reports = run_benchmark(n_trials=args.n_trials, n_steps=args.n_steps, n_modes=args.n_modes, seed=args.seed)

    print("\n=== DriftBench-G1 Results ===")
    print(f"{'Method':<25}  {'Drift@10':>10}  {'Drift@50':>10}  {'Drift@100':>10}")
    print("-" * 60)
    for r in reports:
        print(f"{r.method_name:<25}  {r.drift_at_10:>9.3%}  {r.drift_at_50:>9.3%}  {r.drift_at_100:>9.3%}")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        steps = np.arange(args.n_steps + 1)
        h = 0.01
        C = 0.1
        theoretical_bound = C * h**2 * steps
        ax.plot(steps, theoretical_bound, "k--", linewidth=1.5, label=r"Theoretical bound $Ch^2t$")

        colors = ["tab:blue", "tab:red", "tab:orange", "tab:green", "tab:purple", "tab:brown"]
        for r, color in zip(reports, colors):
            from benchmarks.driftbench_g1.metrics import compute_energy_drift
            drift = compute_energy_drift(r.mean_energy_history)
            ax.semilogy(steps[: len(drift)], np.maximum(drift, 1e-6), color=color, linewidth=2, label=r.method_name)

        ax.set_xlabel("Rollout depth (steps)", fontsize=12)
        ax.set_ylabel("Relative energy drift |E(t)-E(0)| / |E(0)|", fontsize=12)
        ax.set_title("DriftBench-G1: Energy Drift vs. Depth\n(Fig 6 — Headline B)", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = output_dir / "driftbench_g1_results.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Figure saved to %s", out_path)
        plt.close()
    except ImportError:
        logger.warning("matplotlib not installed; skipping figure")


if __name__ == "__main__":
    main()
