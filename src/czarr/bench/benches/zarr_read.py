"""zarr-read — the phase-4 gate: full-array read A/B across the four stacks.

All arms read the same blosc ``[bitshuffle, zstd]`` store end to end and land
a device-resident result; only the machinery differs (param ``impl``):

* ``lowlevel`` — ``czarr.core.Array`` (DecodePlan + coalesced cuFile +
  batched GPU decode).  The explicit tier-2 path; no global state.
* ``tier1``    — ``czarr.open_cuda_array(...)[:]``: the CudaZarrArray fast
  path, which routes through a cached core.Array.  Should track
  ``lowlevel`` within noise — the delta is the wrapper cost.
* ``pipeline`` — ``configure_gpu()`` + stock zarr machinery over
  GPULocalStore.  The pre-fast-path default; tier 1's fallback.
* ``kvikio``   — ``kvikio.zarr.GDSStore`` + ``configure_gpu()`` codecs:
  RAPIDS' GDS read with czarr's GPU decode.  The external baseline;
  requires kvikio (``uv run --with kvikio-cu12``) or the row fails.

Every arm bit-checks against the CPU-decoded reference before timing.
GiB/s is on the logical (uncompressed) payload.
"""

import zarr

from czarr.bench.context import BenchContext
from czarr.bench.registry import benchmark


@benchmark(
    "zarr-read",
    params={"chunk_mib": 128, "n_chunks": 8, "impl": "lowlevel"},
    sweep={"impl": ["lowlevel", "tier1", "pipeline", "kvikio"], "chunk_mib": [8, 128, 256]},
)
def zarr_read(ctx: BenchContext):
    """Full-array read: lowlevel vs tier1 fast path vs zarr pipeline vs kvikio."""
    chunk_mib, n_chunks, impl = ctx.p("chunk_mib"), ctx.p("n_chunks"), ctx.p("impl")

    if impl == "pipeline":
        fx = ctx.fixture.blosc_store(chunk_mib, "gpu", n_chunks=n_chunks)
        arr = zarr.open_array(store=fx.store, mode="r")
    else:
        # "local" builds the fixture + CPU reference without configure_gpu,
        # keeping the lowlevel/tier1 arms free of global zarr state.
        fx = ctx.fixture.blosc_store(chunk_mib, "local", n_chunks=n_chunks)
        if impl == "lowlevel":
            from czarr.core import Array

            arr = Array.open(fx.path)
        elif impl == "tier1":
            import czarr

            arr = czarr.open_cuda_array(str(fx.path))
            if arr._fast_array() is None:
                raise RuntimeError("tier1 fast path did not engage — arm would silently measure the fallback")
        elif impl == "kvikio":
            try:
                import kvikio.zarr
            except ImportError as err:
                raise RuntimeError("kvikio arm needs kvikio installed (uv run --with kvikio-cu12)") from err
            import czarr

            czarr.configure_gpu()
            arr = zarr.open_array(store=kvikio.zarr.GDSStore(str(fx.path)), mode="r")
        else:
            raise ValueError(f"unknown impl {impl!r}")

    def body():
        out = arr[:]
        ctx.sync()
        return out

    def verify(out) -> bool:
        return bool((ctx.to_np(out) == fx.ref).all())

    regime = dict(fx.regime, impl=impl)
    return ctx.plan(body=body, nbytes=fx.ref.nbytes, verify=verify, regime=regime)
