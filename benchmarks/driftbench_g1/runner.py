"""Benchmark runner for DriftBench-G1.

:func:`run_benchmark` executes a list of :class:`~benchmarks.driftbench_g1.tasks.DriftTask`
objects and collects :class:`BenchmarkReport` instances with all metrics.

Typical usage::

    from benchmarks.driftbench_g1.runner import run_benchmark
    from benchmarks.driftbench_g1.tasks import STANDARD_TASKS

    reports = run_benchmark(STANDARD_TASKS)
    for r in reports:
        print(r.task.name, r.metrics)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from benchmarks.driftbench_g1.metrics import drift_summary
from benchmarks.driftbench_g1.tasks import DriftTask, TaskResult, run_task

__all__ = [
    "BenchmarkReport",
    "run_benchmark",
]


@dataclass
class BenchmarkReport:
    """Result for a single :class:`~benchmarks.driftbench_g1.tasks.DriftTask`.

    Attributes
    ----------
    task:
        The task that was run.
    result:
        Raw :class:`~benchmarks.driftbench_g1.tasks.TaskResult`.
    metrics:
        Dict of scalar metrics from :func:`~benchmarks.driftbench_g1.metrics.drift_summary`.
    wallclock_s:
        Wall-clock time in seconds to complete the task.
    """

    task: DriftTask
    result: TaskResult
    metrics: dict[str, float] = field(default_factory=dict)
    wallclock_s: float = 0.0


def run_benchmark(
    tasks: list[DriftTask] | None = None,
    *,
    verbose: bool = False,
) -> list[BenchmarkReport]:
    """Execute all *tasks* and return a list of :class:`BenchmarkReport`.

    Parameters
    ----------
    tasks:
        Tasks to run.  Defaults to :data:`~benchmarks.driftbench_g1.tasks.STANDARD_TASKS`.
    verbose:
        Print one line per task when ``True``.

    Returns
    -------
    list[BenchmarkReport]
    """
    from benchmarks.driftbench_g1.tasks import STANDARD_TASKS

    if tasks is None:
        tasks = STANDARD_TASKS

    reports: list[BenchmarkReport] = []
    for task in tasks:
        t0 = time.perf_counter()
        result = run_task(task)
        elapsed = time.perf_counter() - t0
        metrics = drift_summary(result.energies, result.H0, task.dt)
        report = BenchmarkReport(
            task=task,
            result=result,
            metrics=metrics,
            wallclock_s=elapsed,
        )
        if verbose:
            print(
                f"[DriftBench-G1] {task.name:30s}  "
                f"drift={metrics['relative_drift']:.2e}  "
                f"rate={metrics['drift_rate']:.2e}  "
                f"time={elapsed:.3f}s"
            )
        reports.append(report)
    return reports
