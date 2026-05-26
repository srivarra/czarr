"""Sweep ``codec_pipeline.batch_size`` to measure read↔decode overlap.

The profile on H200 showed reads + decodes serialised when the
pipeline ran one big batch (``sys.maxsize``).  Lowering the batch size
lets zarr's :func:`concurrent_map` run multiple ``read_batch`` calls
concurrently — decode of batch K then overlaps the reads of batch K+1
naturally.  This script measures the win across batch sizes.

Workload reuses the same fixture as ``bench/zarr/slice_compare.py``
(1 GiB Z-slab, 16x512x512 zstd chunks = 64 chunks per read).

Run::

    uv run --extra cu12 python -m bench.overlap.sweep_h200 \
        [--path ...] [--reps 6] [--rewrite]
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
import zarr
from zarr.storage import LocalStore

import czarr

SHAPE = (16, 4096, 4096)
CHUNKS = (16, 512, 512)
DTYPE = "float32"
TOTAL_GIB = (np.prod(SHAPE) * 4) / (1 << 30)

# 64 chunks total (1x8x8). batch_sizes spanning "one big batch" → "many
# small batches" is what we want to see in the sweep.
SWEEP_BATCH_SIZES = [sys.maxsize, 32, 16, 8, 4, 2]


def _gibs(nbytes: int, seconds: float) -> float:
    return (nbytes / (1 << 30)) / seconds


def write_store(path: Path) -> None:
    czarr.configure_gpu()
    store = czarr.GPULocalStore(path)
    arr = zarr.create_array(
        store=store,
        shape=SHAPE,
        chunks=CHUNKS,
        dtype=DTYPE,
        compressors=[czarr.Zstd()],
        overwrite=True,
    )
    rng = np.random.default_rng(0)
    arr[:] = cp.asarray(rng.standard_normal(SHAPE).astype("float32"))
    cp.cuda.Stream.null.synchronize()
    zarr.config.reset()


def _run_once(path: Path, *, batch_size: int, reps: int, warmup: int) -> list[float]:
    czarr.configure_gpu(batch_size=batch_size)
    try:
        arr = zarr.open(czarr.GPULocalStore(path, read_only=True), mode="r")
        for _ in range(warmup):
            _ = arr[:]
            cp.cuda.Stream.null.synchronize()
        samples: list[float] = []
        for _ in range(reps):
            t0 = time.perf_counter()
            out = arr[:]
            cp.cuda.Stream.null.synchronize()
            samples.append(time.perf_counter() - t0)
        assert isinstance(out, cp.ndarray)
        return samples
    finally:
        zarr.config.reset()


def bench_cpu_then_h2d(path: Path, *, reps: int, warmup: int) -> list[float]:
    arr = zarr.open(LocalStore(path), mode="r")
    for _ in range(warmup):
        _ = cp.asarray(arr[:])
        cp.cuda.Stream.null.synchronize()
    samples: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        gpu = cp.asarray(arr[:])
        cp.cuda.Stream.null.synchronize()
        samples.append(time.perf_counter() - t0)
    assert isinstance(gpu, cp.ndarray)
    return samples


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    default_path = Path(os.environ.get("MYDATA", "/hpc/mydata/" + os.environ["USER"])) / ".czarr_overlap_bench.zarr"
    p.add_argument("--path", type=Path, default=default_path)
    p.add_argument("--reps", type=int, default=6)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--rewrite", action="store_true")
    args = p.parse_args()

    nbytes = int(np.prod(SHAPE) * 4)

    if args.rewrite or not args.path.exists():
        print(f"writing store at {args.path} (shape={SHAPE}, chunks={CHUNKS}) ...")
        t0 = time.perf_counter()
        write_store(args.path)
        dt = time.perf_counter() - t0
        print(f"  wrote {TOTAL_GIB:.2f} GiB in {dt:.2f}s ({TOTAL_GIB / dt:.2f} GiB/s)")
    else:
        print(f"reusing store at {args.path}")

    print(f"\nbench: {args.reps} timed reps + {args.warmup} warmup, slab = {SHAPE}")
    print(f"{'batch_size':>12}  {'median (ms)':>12}  {'min (ms)':>10}  {'GiB/s':>10}")
    print("-" * 52)

    results: list[tuple[str, float]] = []
    for bs in SWEEP_BATCH_SIZES:
        label = "maxsize" if bs == sys.maxsize else str(bs)
        samples = _run_once(args.path, batch_size=bs, reps=args.reps, warmup=args.warmup)
        med = statistics.median(samples)
        mn = min(samples)
        gibs = _gibs(nbytes, med)
        print(f"{label:>12}  {med * 1e3:>12.2f}  {mn * 1e3:>10.2f}  {gibs:>10.2f}")
        results.append((label, med))

    cpu_samples = bench_cpu_then_h2d(args.path, reps=args.reps, warmup=args.warmup)
    cpu_med = statistics.median(cpu_samples)
    print(f"{'cpu+h2d':>12}  {cpu_med * 1e3:>12.2f}  {min(cpu_samples) * 1e3:>10.2f}  {_gibs(nbytes, cpu_med):>10.2f}")

    best_label, best_med = min(results, key=lambda r: r[1])
    print(f"\nbest: batch_size={best_label} at {best_med * 1e3:.2f} ms ({_gibs(nbytes, best_med):.2f} GiB/s)")
    print(f"speedup vs maxsize: {results[0][1] / best_med:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
