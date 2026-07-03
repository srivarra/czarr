"""Does ``CudaZarrArray.iter_gpu`` hide read+decode behind compute?

NEGATIVE / NARROW RESULT — KEPT AS A RECORD.  ``iter_gpu`` (a background-
thread prefetcher) was REVERTED after this bench.  H100/GDS, 1 GiB, 64x16MiB
slices:

    consumer          k    block ms   pipe ms   speedup
    async (no sync)   1      630.2     642.4     0.98x   ← redundant
    async             16    8891.1    8930.7     1.00x
    sync-per-iter     1      824.6     633.7     1.30x   ← only real win
    sync-per-iter     4     2570.3    2346.5     1.10x
    sync-per-iter     16    9155.1    8924.4     1.03x

Verdict: for the realistic async consumer (no per-iteration sync), CUDA's
async command queue ALREADY overlaps the next slice's read with the current
slice's compute — an explicit prefetch thread adds only overhead.  iter_gpu
helps (1.10-1.30x, peak when read≈compute) ONLY when the consumer syncs every
iteration (e.g. loss.item()), which drains the queue.  That's rescuing an
anti-pattern, so the feature was dropped; the win is available for free by
just not syncing inside the loop.  To re-run, restore iter_gpu (git history).

The streaming use case: iterate the outer axis, stream each inner slice to
the GPU, run compute, repeat.  ``iter_gpu`` prefetches slice i+1 on a
background thread while the caller computes on slice i.  If czarr's I/O +
orchestration cost overlaps the compute, the pipelined loop trends toward
``max(read_total, compute_total)`` instead of the blocking
``read_total + compute_total``.

This is the bench multistream never was: the overlap is with the *caller's*
GPU work, not internal decode batches.  It can only win when per-slice
compute is GPU-bound (long kernels, brief launch) so the GIL is free for the
prefetch thread.  Hence the compute-intensity sweep (matmul iterations `k`):

* k=0   — pure read, nothing to hide behind: expect tie (or slight loss to
          thread overhead).  Sanity control.
* k grows — compute-bound: pipelined should approach max(read, compute).

Reference rows (read-only, compute-only) bracket the ideal so the speedup is
interpretable, not just a ratio.

Run::

    uv run --extra cu12 python -m bench.pipeline.stream_pipeline_h100 \
        --store gpu --rewrite
"""

import argparse
import os
import statistics
import time
from pathlib import Path

import cupy as cp
import numpy as np
import zarr
from zarr.storage import LocalStore

import czarr
from czarr import CudaZarrArray

SHAPE = (64, 2048, 2048)  # 64 slices x 16 MiB = 1 GiB
CHUNKS = (1, 512, 512)
DTYPE = "float32"
SLICE_BYTES = SHAPE[1] * SHAPE[2] * 4
# Compute-intensity sweep: each unit = _BURN_UNIT inner FMA/transcendental
# iters per element.  k spans read-bound (compute << read) to compute-bound.
KSWEEP = [0, 1, 4, 16]
_BURN_UNIT = 2000

# cuBLAS-free GPU-bound kernel: a tight per-element FMA + transcendental loop.
# One long kernel per slice (minimal launch overhead) is the ideal regime for
# the prefetch thread to grab the GIL and read ahead while this runs.
_BURN = cp.ElementwiseKernel(
    "float32 x, int32 iters",
    "float32 y",
    """
    float v = x;
    for (int i = 0; i < iters; i++) {
        v = v * 1.0000001f + 0.0000001f;
        v = sinf(v) + 1.0f;
    }
    y = v;
    """,
    "czarr_bench_burn",
)


def _store(path: Path, kind: str, *, read_only: bool = False):
    if kind == "gpu":
        return czarr.GPULocalStore(path, read_only=read_only)
    return LocalStore(path, read_only=read_only)


def write_store(path: Path, store_kind: str) -> None:
    czarr.configure_gpu()
    store = _store(path, store_kind)
    arr = zarr.create_array(
        store=store,
        shape=SHAPE,
        chunks=CHUNKS,
        dtype=DTYPE,
        compressors=[czarr.ANS()],
        overwrite=True,
    )
    rng = np.random.default_rng(0)
    base = np.linspace(0, 1, int(np.prod(SHAPE)), dtype="float32").reshape(SHAPE)
    arr[:] = cp.asarray(base + rng.standard_normal(SHAPE).astype("float32") * 0.01)
    cp.cuda.Stream.null.synchronize()
    zarr.config.reset()


def _compute(x: cp.ndarray, k: int) -> cp.ndarray:
    """GPU-bound per-slice work: one burn kernel of ``k * _BURN_UNIT`` iters.

    ``k=0`` still launches the kernel with 0 inner iters (≈ passthrough) so
    both paths pay the same launch overhead — the difference measured is
    overlap, not kernel count.
    """
    return _BURN(x.astype(cp.float32, copy=False), np.int32(k * _BURN_UNIT))


def _open(path: Path, store_kind: str) -> CudaZarrArray:
    return CudaZarrArray.wrap(zarr.open_array(store=_store(path, store_kind, read_only=True), mode="r"))


def run_blocking(arr: CudaZarrArray, k: int, *, sync_each: bool = False) -> float:
    sink = cp.zeros((), cp.float32)
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    for i in range(arr.shape[0]):
        x = arr[i]
        sink += _compute(x, k).sum()
        if sync_each:
            # Simulate a consumer that syncs every step (e.g. loss.item()),
            # which drains the async queue and defeats the free overlap.
            cp.cuda.runtime.deviceSynchronize()
    cp.cuda.runtime.deviceSynchronize()
    return time.perf_counter() - t0


def run_pipelined(arr: CudaZarrArray, k: int, prefetch: int, *, sync_each: bool = False) -> float:
    sink = cp.zeros((), cp.float32)
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    for x in arr.iter_gpu(prefetch=prefetch):
        sink += _compute(x, k).sum()
        if sync_each:
            cp.cuda.runtime.deviceSynchronize()
    cp.cuda.runtime.deviceSynchronize()
    return time.perf_counter() - t0


def _median(fn, *, reps: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    return statistics.median([fn() for _ in range(reps)])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    default_path = Path(os.environ.get("MYDATA", "/hpc/mydata/" + os.environ["USER"])) / ".czarr_stream_bench.zarr"
    p.add_argument("--path", type=Path, default=default_path)
    p.add_argument("--store", choices=["gpu", "local"], default="gpu")
    p.add_argument("--prefetch", type=int, default=2)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--rewrite", action="store_true")
    p.add_argument(
        "--sync-consumer",
        action="store_true",
        help="sync the device every iteration (simulates loss.item()-style consumers "
        "that drain the async queue; this is the regime where prefetch can win)",
    )
    args = p.parse_args()

    path = args.path.with_name(f"{args.path.name}.{args.store}")
    n = SHAPE[0]

    print(f"store={args.store}  slab={SHAPE} chunks={CHUNKS}  prefetch={args.prefetch}")
    if args.rewrite or not path.exists():
        print(f"writing store at {path} ...")
        t0 = time.perf_counter()
        write_store(path, args.store)
        print(f"  wrote {np.prod(SHAPE) * 4 / (1 << 30):.2f} GiB in {time.perf_counter() - t0:.2f}s")
    else:
        print(f"reusing store at {path}")

    czarr.configure_gpu()
    arr = _open(path, args.store)

    # Reference: pure read (k=0) and pure compute on a resident slice.
    read_only = _median(lambda: run_blocking(arr, 0), reps=args.reps, warmup=args.warmup)
    resident = arr[0].copy()

    def compute_only(k: int) -> float:
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        sink = cp.zeros((), cp.float32)
        for _ in range(n):
            sink += _compute(resident, k).sum()
        cp.cuda.runtime.deviceSynchronize()
        return time.perf_counter() - t0

    sync_each = args.sync_consumer
    mode = "sync-per-iter consumer" if sync_each else "async consumer (no per-iter sync)"
    print(
        f"\nread-only (blocking, k=0): {read_only * 1e3:.1f} ms  ({n} slices, {SLICE_BYTES / (1 << 20):.0f} MiB each)"
    )
    print(f"consumer mode: {mode}")
    print(
        f"\n{'k':>4}  {'compute ms':>10}  {'block ms':>9}  {'pipe ms':>9}  {'ideal ms':>9}  {'speedup':>8}  {'of ideal':>8}"
    )
    print("-" * 70)
    for k in KSWEEP:
        comp = compute_only(k)
        block = _median(lambda k=k: run_blocking(arr, k, sync_each=sync_each), reps=args.reps, warmup=args.warmup)
        pipe = _median(
            lambda k=k: run_pipelined(arr, k, args.prefetch, sync_each=sync_each), reps=args.reps, warmup=args.warmup
        )
        ideal = max(read_only, comp)
        speedup = block / pipe
        of_ideal = ideal / pipe  # 1.0 = perfectly hidden; <1 = pipe slower than ideal
        print(
            f"{k:>4}  {comp * 1e3:>10.1f}  {block * 1e3:>9.1f}  {pipe * 1e3:>9.1f}  "
            f"{ideal * 1e3:>9.1f}  {speedup:>7.2f}x  {of_ideal:>7.2f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
