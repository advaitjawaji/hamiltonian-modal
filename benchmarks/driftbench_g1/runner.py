"""DriftBench-G1 benchmark runner."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

from benchmarks.driftbench_g1.tasks import DriftTask, STANDARD_TASKS, run_task
from benchmarks.driftbench_g1.metrics import compute_drift_at_depth, compute_energy_drift

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkReport:
    """Results for a single method in the DriftBench-G1 benchmark.

    Attributes
    ----------
    method_name : str
    n_trials : int
    drift_at_10 : float — mean relative energy drift at step 10
    drift_at_50 : float — mean relative energy drift at step 50
    drift_at_100 : float — mean relative energy drift at step 100
    mean_energy_history : NDArray — mean energy over time across trials
    std_energy_history : NDArray — std of energy over time across trials
    """
    method_name: str
    n_trials: int
    drift_at_10: float
    drift_at_50: float
    drift_at_100: float
    mean_energy_history: NDArray[np.float64]
    std_energy_history: NDArray[np.float64] = field(default_factory=lambda: np.zeros(1))


def run_benchmark(
    n_trials: int = 100,
    n_steps: int = 100,
    tasks: list[DriftTask] | None = None,
) -> list[BenchmarkReport]:
    """Run the full DriftBench-G1 benchmark across all methods.

    Each method is run ``n_trials`` times with different random seeds.
    Energy drift is computed at steps 10, 50, and 100.

    Parameters
    ----------
    n_trials : int — number of independent trials per method
    n_steps : int — number of integration steps per trial
    tasks : list of DriftTask or None (defaults to STANDARD_TASKS)

    Returns
    -------
    reports : list[BenchmarkReport] — one report per method
    """
    if tasks is None:
        tasks = STANDARD_TASKS

    reports: list[BenchmarkReport] = []

    for task in tasks:
        logger.info("Running benchmark for method: %s (%d trials)", task.name, n_trials)
        energy_histories: list[NDArray[np.float64]] = []

        for trial in range(n_trials):
            trial_task = DriftTask(
                name=task.name,
                n_modes=task.n_modes,
                h=task.h,
                n_steps=n_steps,
                integrator=task.integrator,
                seed=task.seed + trial,
            )
            energies = run_task(trial_task)
            energy_histories.append(energies)

        # Stack trials: shape (n_trials, n_steps+1)
        E_mat = np.vstack(energy_histories)
        mean_E = np.mean(E_mat, axis=0)
        std_E = np.std(E_mat, axis=0)

        drift_at_10 = float(np.mean([
            compute_drift_at_depth(e, min(10, n_steps)) for e in energy_histories
        ]))
        drift_at_50 = float(np.mean([
            compute_drift_at_depth(e, min(50, n_steps)) for e in energy_histories
        ]))
        drift_at_100 = float(np.mean([
            compute_drift_at_depth(e, min(100, n_steps)) for e in energy_histories
        ]))

        report = BenchmarkReport(
            method_name=task.name,
            n_trials=n_trials,
            drift_at_10=drift_at_10,
            drift_at_50=drift_at_50,
            drift_at_100=drift_at_100,
            mean_energy_history=mean_E,
            std_energy_history=std_E,
        )
        reports.append(report)
        logger.info(
            "  %s: drift@10=%.4f, drift@50=%.4f, drift@100=%.4f",
            task.name,
            drift_at_10,
            drift_at_50,
            drift_at_100,
        )

    return reports


def summarise_reports(reports: list[BenchmarkReport]) -> str:
    """Format a human-readable summary table of benchmark results.

    Parameters
    ----------
    reports : list[BenchmarkReport]

    Returns
    -------
    table : str
    """
    header = f"{'Method':<25} {'Drift@10':>10} {'Drift@50':>10} {'Drift@100':>10}"
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in reports:
        lines.append(
            f"{r.method_name:<25} {r.drift_at_10:>10.6f} {r.drift_at_50:>10.6f} {r.drift_at_100:>10.6f}"
        )
    lines.append(sep)
    return "\n".join(lines)
