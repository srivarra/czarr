"""Instrumentation: NVTX stage spans, an nsys capture window, and ncu metrics.

Three tools, three jobs — none replaces the wall-clock harness:

* :func:`span` — free-form, nestable NVTX ranges (``czarr._nvtx``) that label
  stages in an nsys timeline.  Always cheap; no-op when ``CZARR_NVTX=0``.
* :func:`capture_window` — ``cudaProfilerStart/Stop`` around the steady-state
  reps so ``nsys profile --capture-range=cudaProfilerApi`` records only those.
* :func:`ncu_metrics` — drive ``nsight.analyze.kernel`` (Nsight Compute) over a
  body to collect kernel hardware metrics.  Requires ``ncu`` on PATH (bundled
  with the cuda module); a no-op stub is returned if Nsight Python is absent.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from czarr._nvtx import nvtx_range as span  # re-export: stage spans use the existing layer

__all__ = ["capture_window", "ncu_metrics", "span"]


@contextmanager
def capture_window(active: bool):
    """Bracket the timed reps with cudaProfilerStart/Stop when profiling.

    Pairs with ``nsys profile --capture-range=cudaProfilerApi`` so warmup and
    setup stay out of the captured timeline.  A no-op when ``active`` is False.
    """
    if not active:
        yield
        return
    from cuda.bindings import runtime as cudart

    cudart.cudaProfilerStart()
    try:
        yield
    finally:
        cudart.cudaProfilerStop()


def ncu_metrics(
    body: Callable[[], Any],
    *,
    name: str,
    metrics: list[str] | None = None,
    runs: int = 1,
) -> dict[str, Any]:
    """Profile ``body``'s kernels with Nsight Compute, return a metrics dict.

    Wraps ``nsight.analyze.kernel``; the body is re-run under ``ncu`` (it
    re-execs this process), and the resulting per-kernel metric rows are folded
    into a flat dict keyed ``<kernel>.<metric>``.  Returns ``{"ncu": "<reason>"}``
    when ncu/Nsight Python is unavailable so a sweep degrades instead of dying.
    """
    metrics = metrics or ["gpu__time_duration.sum"]
    try:
        import nsight
    except ImportError:
        return {"ncu_skipped": "nsight-python not installed"}

    @nsight.analyze.kernel(metrics=metrics, runs=runs, output="quiet")
    def _run():
        with nsight.annotate(name):
            return body()

    try:
        res = _run()
    except Exception as exc:  # ncu not on PATH / no perf-counter permission
        return {"ncu_skipped": f"{type(exc).__name__}: {exc}"[:120]}
    df = res.to_dataframe()
    return {f"{row.Kernel}.{row.Metric}": float(row.AvgValue) for row in df.itertuples()}
