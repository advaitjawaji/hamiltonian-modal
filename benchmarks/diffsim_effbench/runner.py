"""DiffSim-EffBench benchmark runner."""
from __future__ import annotations

import logging
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from benchmarks.diffsim_effbench.tasks import (
    EfficiencyTask,
    STANDARD_EFFICIENCY_TASKS,
)
from benchmarks.diffsim_effbench.metrics import (
    compute_sample_efficiency_ratio,
    time_to_threshold,
    normalised_auc,
)

logger = logging.getLogger(__name__)


@dataclass
class EfficiencyReport:
    """Results for a single method in the DiffSim-EffBench benchmark.

    Attributes
    ----------
    method_name : str
    n_trials : int
    mean_reward_curve : shape (n_steps,)
    std_reward_curve : shape (n_steps,)
    nauc : float — normalised area under reward curve (higher = better)
    time_to_80pct : float — simulated steps to reach 80 % of max reward
    """
    method_name: str
    n_trials: int
    mean_reward_curve: NDArray[np.float64]
    std_reward_curve: NDArray[np.float64]
    nauc: float
    time_to_80pct: float


@dataclass
class ComparisonReport:
    """Head-to-head comparison between diff-sim and a baseline method.

    Attributes
    ----------
    method_a : str — name of method A (typically "diffsim")
    method_b : str — name of method B (baseline)
    efficiency_ratio : float — steps_b / steps_a at threshold; > 1 means A is better
    nauc_ratio : float — nauc_a / nauc_b; > 1 means A is better
    time_to_80pct_a : float
    time_to_80pct_b : float
    """
    method_a: str
    method_b: str
    efficiency_ratio: float
    nauc_ratio: float
    time_to_80pct_a: float
    time_to_80pct_b: float


def run_efficiency_benchmark(
    tasks: "list[EfficiencyTask] | None" = None,
    n_trials: int = 50,
    threshold: float = 0.8,
) -> list[EfficiencyReport]:
    """Run the DiffSim-EffBench efficiency benchmark for all methods.

    Parameters
    ----------
    tasks : list of EfficiencyTask or None (defaults to STANDARD_EFFICIENCY_TASKS)
    n_trials : int — independent trials per method
    threshold : float — reward threshold for time-to-threshold metric

    Returns
    -------
    reports : list[EfficiencyReport]
    """
    if tasks is None:
        tasks = STANDARD_EFFICIENCY_TASKS

    reports: list[EfficiencyReport] = []

    for task in tasks:
        logger.info("Running efficiency benchmark for: %s (%d trials)", task.name, n_trials)
        all_curves: list[NDArray[np.float64]] = []

        for trial in range(n_trials):
            rng = np.random.default_rng(task.seed + trial)
            curve = task.reward_curve_fn(rng, task.n_steps)
            all_curves.append(curve)

        curves_mat = np.vstack(all_curves)
        mean_curve = np.mean(curves_mat, axis=0)
        std_curve = np.std(curves_mat, axis=0)

        nauc = float(np.mean([normalised_auc(c) for c in all_curves]))
        t80 = float(np.mean([
            time_to_threshold(c, threshold=threshold, step_duration=task.step_duration)
            for c in all_curves
        ]))

        report = EfficiencyReport(
            method_name=task.name,
            n_trials=n_trials,
            mean_reward_curve=mean_curve,
            std_reward_curve=std_curve,
            nauc=nauc,
            time_to_80pct=t80,
        )
        reports.append(report)
        logger.info("  %s: nAUC=%.4f, time_to_80pct=%.1f", task.name, nauc, t80)

    return reports


def compare_diffsim_vs_baselines(
    reports: list[EfficiencyReport],
    reference_method: str = "diffsim",
) -> list[ComparisonReport]:
    """Generate head-to-head comparison reports.

    Parameters
    ----------
    reports : list from run_efficiency_benchmark
    reference_method : str — method name to use as reference (numerator)

    Returns
    -------
    comparisons : list[ComparisonReport]
    """
    ref = next((r for r in reports if r.method_name == reference_method), None)
    if ref is None:
        raise ValueError(f"Reference method {reference_method!r} not found in reports")

    comparisons: list[ComparisonReport] = []
    for report in reports:
        if report.method_name == reference_method:
            continue

        eff_ratio = compute_sample_efficiency_ratio(
            ref.mean_reward_curve, report.mean_reward_curve
        )
        nauc_ratio = ref.nauc / report.nauc if report.nauc > 1e-10 else float("inf")

        comparisons.append(
            ComparisonReport(
                method_a=reference_method,
                method_b=report.method_name,
                efficiency_ratio=eff_ratio,
                nauc_ratio=nauc_ratio,
                time_to_80pct_a=ref.time_to_80pct,
                time_to_80pct_b=report.time_to_80pct,
            )
        )

    return comparisons


def summarise_comparison(comparisons: list[ComparisonReport]) -> str:
    """Format a human-readable comparison summary table.

    Parameters
    ----------
    comparisons : list[ComparisonReport]

    Returns
    -------
    table : str
    """
    header = f"{'Method A':<15} vs {'Method B':<15} {'EffRatio':>10} {'nAUC_A':>8} {'nAUC_B':>8}"
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for c in comparisons:
        lines.append(
            f"{c.method_a:<15} vs {c.method_b:<15} {c.efficiency_ratio:>10.3f} "
            f"{c.time_to_80pct_a:>8.1f} {c.time_to_80pct_b:>8.1f}"
        )
    lines.append(sep)
    return "\n".join(lines)
