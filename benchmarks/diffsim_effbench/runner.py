"""Benchmark runner for DiffSim-EffBench.

:func:`run_eff_benchmark` executes the efficiency suite and returns a list of
:class:`EffReport` objects containing all metrics.

Typical usage::

    from benchmarks.diffsim_effbench.runner import run_eff_benchmark
    from benchmarks.diffsim_effbench.tasks import STANDARD_EFF_TASKS

    reports = run_eff_benchmark(STANDARD_EFF_TASKS)
    for r in reports:
        print(r.task.name, r.metrics)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from benchmarks.diffsim_effbench.metrics import eff_summary
from benchmarks.diffsim_effbench.tasks import EffTask, EffTaskResult, run_eff_task

__all__ = [
    "EffReport",
    "run_eff_benchmark",
]


@dataclass
class EffReport:
    """Result for a single :class:`~benchmarks.diffsim_effbench.tasks.EffTask`.

    Attributes
    ----------
    task:
        The task specification.
    result:
        Raw :class:`~benchmarks.diffsim_effbench.tasks.EffTaskResult`.
    metrics:
        Dict of scalar metrics from :func:`~benchmarks.diffsim_effbench.metrics.eff_summary`.
    """

    task: EffTask
    result: EffTaskResult
    metrics: dict[str, float] = field(default_factory=dict)


def run_eff_benchmark(
    tasks: list[EffTask] | None = None,
    *,
    verbose: bool = False,
) -> list[EffReport]:
    """Execute all *tasks* and return a list of :class:`EffReport`.

    Parameters
    ----------
    tasks:
        Tasks to run.  Defaults to
        :data:`~benchmarks.diffsim_effbench.tasks.STANDARD_EFF_TASKS`.
    verbose:
        Print one line per task when ``True``.

    Returns
    -------
    list[EffReport]
    """
    from benchmarks.diffsim_effbench.tasks import STANDARD_EFF_TASKS

    if tasks is None:
        tasks = STANDARD_EFF_TASKS

    reports: list[EffReport] = []
    for task in tasks:
        result = run_eff_task(task)
        metrics = eff_summary(
            result.mean_return,
            result.grad_norm,
            result.grad_variance,
            result.steps_per_second,
            task.n_dof,
        )
        report = EffReport(task=task, result=result, metrics=metrics)
        if verbose:
            print(
                f"[DiffSim-EffBench] {task.name:25s}  "
                f"snr={metrics['gradient_snr']:.2e}  "
                f"eff={metrics['efficiency_score']:.1f}  "
                f"sps={metrics['steps_per_second']:.0f}"
            )
        reports.append(report)
    return reports
