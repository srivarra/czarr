"""The harness: correctness gate → warmup → timed reps → median, GiB/s, row.

All the timing/instrumentation boilerplate lives here, once, so benches stay
declarative.  Order is deliberate: correctness is checked before the timing
loop, so a wrong result fails fast and never gets reported as a fast number.
"""

import statistics
import time

from .context import BenchContext
from .fixtures import Fixtures
from .instrument import capture_window, ncu_metrics, span
from .output import Row, git_commit
from .registry import BenchSpec
from .telemetry import gds_active, gpu_info, rmm_peak


class CorrectnessError(RuntimeError):
    """Raised when a bench's verify() gate fails — fast-but-wrong is still wrong."""


def run_bench(
    spec: BenchSpec,
    params: dict[str, object],
    *,
    reps: int = 5,
    warmup: int = 2,
    nsys: bool = False,
    ncu: bool = False,
    ncu_metric_names: list[str] | None = None,
) -> Row:
    """Drive one bench: gate correctness, warm up, time ``reps``, return a row."""
    ctx = BenchContext(params=params, fixture=Fixtures())
    plan = spec.fn(ctx)

    if plan.setup is not None:
        plan.setup()

    # --- correctness gate (before any timing) ---
    first = plan.body()
    if plan.verify is not None:
        ok = False
        try:
            ok = bool(plan.verify(first))
        except Exception as exc:  # a raising check is a failed check
            raise CorrectnessError(f"{spec.name}: verify() raised {type(exc).__name__}: {exc}") from exc
        if not ok:
            raise CorrectnessError(f"{spec.name}: result not bit-exact vs reference")

    # --- warmup (JIT, allocation, buffer registration) ---
    for _ in range(warmup):
        plan.body()

    # --- timed reps, bracketed by the profiler window + RMM peak tracking ---
    samples: list[float] = []
    ctx.sync()
    with rmm_peak() as peak, capture_window(nsys):
        for _ in range(reps):
            ctx.sync()
            t0 = time.perf_counter()
            with span("body", **{k: params[k] for k in list(params)[:3]}):
                plan.body()
            ctx.sync()
            samples.append(time.perf_counter() - t0)

    median_s = statistics.median(samples)
    extra: dict[str, object] = {**plan.regime, **peak}

    if ncu:
        extra.update(ncu_metrics(plan.body, name=spec.name, metrics=ncu_metric_names))

    info = gpu_info()
    return Row(
        bench=spec.name,
        params=params,
        median_ms=median_s * 1e3,
        gib_s=(plan.nbytes / 2**30) / median_s,
        reps=reps,
        warmup=warmup,
        gpu=info.get("gpu"),
        gpu_mem_gib=info.get("gpu_mem_gib"),
        gds_active=gds_active(),
        commit=git_commit(),
        extra=extra,
    )
