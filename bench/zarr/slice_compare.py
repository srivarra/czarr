"""czarr (GPU end-to-end) vs regular-zarr-then-cupy on a single Z-slab read.

A real-world Talon-style slab shape: ``(16, 4096, 4096)`` float32 = 1 GiB
per read.  Two paths:

* PATH A — czarr: GPULocalStore + CzarrPipeline.  cuFile reads compressed
  bytes straight to GPU, nvCOMP decodes, result is a ``cupy.ndarray``.

* PATH B — zarr default + manual H2D: vanilla LocalStore, numcodecs.Zstd
  on host, then ``cp.asarray(...)`` to land on GPU.

Both decode the same zstd bitstream (czarr.Zstd is bitstream-compatible
with numcodecs.Zstd).  The store is written once with the czarr GPU
encoder and reused across runs.

Run:
    uv run --extra cu12 --group test python -m bench.zarr.slice_compare \
        [--path /tmp/store.zarr] [--reps 10] [--rewrite]
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

SHAPE = (16, 4096, 4096)
CHUNKS = (16, 512, 512)
DTYPE = "float32"
TOTAL_GIB = (np.prod(SHAPE) * 4) / (1 << 30)


def _gibs(nbytes: int, seconds: float) -> float:
    return (nbytes / (1 << 30)) / seconds


def write_store(path: Path) -> None:
    """One-time generate + write of the test store."""
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
    data = rng.standard_normal(SHAPE).astype("float32")
    arr[:] = cp.asarray(data)
    cp.cuda.Stream.null.synchronize()
    zarr.config.reset()


def bench_czarr(path: Path, *, reps: int, warmup: int) -> list[float]:
    czarr.configure_gpu()
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
    assert isinstance(out, cp.ndarray), f"expected cupy.ndarray, got {type(out).__name__}"
    zarr.config.reset()
    return samples


def bench_cpu_then_h2d(path: Path, *, reps: int, warmup: int) -> list[float]:
    zarr.config.reset()  # vanilla pipeline + CPU buffer prototype
    arr = zarr.open(LocalStore(path), mode="r")
    for _ in range(warmup):
        _ = cp.asarray(arr[:])
        cp.cuda.Stream.null.synchronize()
    samples: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        host = arr[:]
        gpu = cp.asarray(host)
        cp.cuda.Stream.null.synchronize()
        samples.append(time.perf_counter() - t0)
    assert isinstance(gpu, cp.ndarray)
    return samples


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    default_path = Path(os.environ.get("MYDATA", "/hpc/mydata/" + os.environ["USER"])) / ".czarr_slice_bench.zarr"
    p.add_argument("--path", type=Path, default=default_path)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--rewrite", action="store_true", help="Force re-write the store.")
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
    czarr_samples = bench_czarr(args.path, reps=args.reps, warmup=args.warmup)
    cpu_samples = bench_cpu_then_h2d(args.path, reps=args.reps, warmup=args.warmup)

    czarr_med = statistics.median(czarr_samples)
    cpu_med = statistics.median(cpu_samples)
    czarr_min = min(czarr_samples)
    cpu_min = min(cpu_samples)

    print(f"\n{'path':<25}{'median':>12}{'min':>12}{'GiB/s':>10}")
    print("-" * 60)
    print(
        f"{'czarr (GPU e2e)':<25}{czarr_med * 1e3:>9.2f} ms{czarr_min * 1e3:>9.2f} ms{_gibs(nbytes, czarr_med):>10.2f}"
    )
    print(f"{'zarr + cp.asarray':<25}{cpu_med * 1e3:>9.2f} ms{cpu_min * 1e3:>9.2f} ms{_gibs(nbytes, cpu_med):>10.2f}")
    print(f"\nspeedup (czarr vs cpu+h2d): {cpu_med / czarr_med:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
