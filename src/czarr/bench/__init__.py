"""czarr-bench — one CLI + one harness in place of a dozen one-off scripts.

Public surface for authoring benches::

    from czarr.bench import benchmark, BenchContext


    @benchmark("my-bench", params={"chunk_mib": 8}, sweep={"chunk_mib": [8, 128]})
    def my_bench(ctx: BenchContext):
        fx = ctx.fixture.blosc_store(ctx.p("chunk_mib"))
        arr = ...  # open fx.store
        return ctx.plan(
            body=lambda: arr[:],
            nbytes=fx.ref.nbytes,
            verify=lambda out: (ctx.to_np(out) == fx.ref).all(),
            regime=fx.regime,
        )

The CLI (``czarr-bench``) lives in :mod:`czarr.bench.cli`.
"""

from .context import BenchContext, BenchPlan
from .registry import benchmark, load_all

__all__ = ["BenchContext", "BenchPlan", "benchmark", "load_all"]
