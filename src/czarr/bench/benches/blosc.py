"""Blosc decode benches — GPU-direct vs CPU, ported from bench/blosc/e2e_h100.py (git history).

``blosc-e2e`` reads a blosc ``[bitshuffle, zstd]`` store end to end:
* ``store=gpu``   -> GPULocalStore (cuFile) + on-device GPU decode  (the path under test)
* ``store=local`` -> plain LocalStore (numcodecs decode + H2D)       (the CPU baseline)

Both validate bit-exact against the same CPU reference before timing.  GiB/s is
on the logical (uncompressed) payload; ``compressed_ratio`` rides every row.
"""

import zarr

from czarr.bench.context import BenchContext
from czarr.bench.registry import benchmark

_CHUNK_AXIS = [8, 128, 256]


@benchmark(
    "blosc-e2e",
    params={"chunk_mib": 128, "store": "gpu", "n_chunks": 8},
    sweep={"chunk_mib": _CHUNK_AXIS, "store": ["gpu", "local"]},
)
def blosc_e2e(ctx: BenchContext):
    """End-to-end blosc read: GPU-direct decode (store=gpu) vs CPU+H2D (store=local)."""
    fx = ctx.fixture.blosc_store(ctx.p("chunk_mib"), ctx.p("store"), n_chunks=ctx.p("n_chunks"))
    arr = zarr.open_array(store=fx.store, mode="r")

    def body():
        out = arr[:]
        ctx.sync()
        return out

    def verify(out) -> bool:
        return bool((ctx.to_np(out) == fx.ref).all())

    return ctx.plan(body=body, nbytes=fx.ref.nbytes, verify=verify, regime=fx.regime)
