"""Quick perf probe: czarr default (zarr.gpu.Buffer) vs CzarrGpuBuffer.

Reuses the same fixture as ``bench/zarr/slice_compare.py`` (1 GiB
Z-slab, 16x512x512 zstd chunks) so the only thing changing between
runs is the buffer prototype. Phase 3+ (cuFile direct I/O via the
4 KiB-aligned VMR pointer) is *not* wired in yet, so we expect the
two paths to land within noise — this run establishes that
CzarrGpuBuffer doesn't regress on the decode-only path before we
plumb the cuFile side.

Run::

    uv run --extra cu12 python -m bench.buffer.quick_h200 \
        [--path /tmp/store.zarr] [--reps 8] [--rewrite]
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

# Side-effect import: registers CzarrGpuBuffer / CzarrGpuNDBuffer in
# zarr's buffer registry under their qualnames.
import czarr.core.buffer

SHAPE = (16, 4096, 4096)
CHUNKS = (16, 512, 512)
DTYPE = "float32"
TOTAL_GIB = (np.prod(SHAPE) * 4) / (1 << 30)


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
    data = rng.standard_normal(SHAPE).astype("float32")
    arr[:] = cp.asarray(data)
    cp.cuda.Stream.null.synchronize()
    zarr.config.reset()


def _run(path: Path, *, reps: int, warmup: int) -> list[float]:
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
    return samples


def bench_czarr_default(path: Path, *, reps: int, warmup: int) -> list[float]:
    czarr.configure_gpu()
    try:
        return _run(path, reps=reps, warmup=warmup)
    finally:
        zarr.config.reset()


def bench_czarr_cuda_core(path: Path, *, reps: int, warmup: int) -> list[float]:
    czarr.configure_gpu()
    try:
        zarr.config.set(
            {
                "buffer": "czarr.core.buffer.CzarrGpuBuffer",
                "ndbuffer": "czarr.core.buffer.CzarrGpuNDBuffer",
            }
        )
        return _run(path, reps=reps, warmup=warmup)
    finally:
        zarr.config.reset()


def bench_cpu_then_h2d(path: Path, *, reps: int, warmup: int) -> list[float]:
    zarr.config.reset()
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
    default_path = Path(os.environ.get("MYDATA", "/hpc/mydata/" + os.environ["USER"])) / ".czarr_buffer_bench.zarr"
    p.add_argument("--path", type=Path, default=default_path)
    p.add_argument("--reps", type=int, default=8)
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
    a = bench_czarr_default(args.path, reps=args.reps, warmup=args.warmup)
    b = bench_czarr_cuda_core(args.path, reps=args.reps, warmup=args.warmup)
    c = bench_cpu_then_h2d(args.path, reps=args.reps, warmup=args.warmup)

    rows = [
        ("czarr (gpu.Buffer)", a),
        ("czarr (CzarrGpuBuffer)", b),
        ("zarr + cp.asarray", c),
    ]
    print(f"\n{'path':<28}{'median':>12}{'min':>12}{'GiB/s':>10}")
    print("-" * 64)
    for name, samples in rows:
        med = statistics.median(samples)
        mn = min(samples)
        print(f"{name:<28}{med * 1e3:>9.2f} ms{mn * 1e3:>9.2f} ms{_gibs(nbytes, med):>10.2f}")

    med_a, med_b, med_c = (statistics.median(s) for _, s in rows)
    print(f"\ncuda-core vs gpu.Buffer: {med_a / med_b:.3f}x (>1.0 = win)")
    print(f"czarr default vs cpu+h2d: {med_c / med_a:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
