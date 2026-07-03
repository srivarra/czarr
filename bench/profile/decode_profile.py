"""Decompression profiling target for Nsight Systems.

Drives a repeated ``arr[:]`` GPU decode under NVTX so the nsys timeline
segments cleanly into czarr's existing ranges (``czarr.pipeline.read_batch``
-> ``czarr.codec.nvcomp_decode`` / ``alloc_outs`` / ``wrap_*``).

Reports two headline numbers without a profiler attached, so the script
is also useful standalone:

* **Throughput** — uncompressed GiB / median wall time over the read.
* **Peak device memory** — via ``rmm.statistics`` when an RMM pool is
  active (``--rmm-gb``).  Captures nvCOMP scratch + compressed inputs +
  decoded outputs at their concurrent peak, which is the number that
  decides how big a budget a deployment needs.

Run under nsys (see ``bench/run_decode_profile.sbatch``)::

    nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true \\
        -o decode python bench/profile/decode_profile.py --rmm-gb 4 ...
"""

import argparse
import time
from pathlib import Path

import cupy as cp
import numpy as np
import rmm.statistics
import zarr
from zarr.storage import LocalStore

import czarr
from czarr._nvtx import mark, nvtx_range

_COMPRESSORS = {
    "zstd": lambda: czarr.Zstd(),
    "lz4": lambda: czarr.LZ4(),
    "ans": lambda: czarr.ANS(),
    "gdeflate": lambda: czarr.GDeflate(),
}


def _build_store(path: Path, *, shape, inner_chunk, dtype, compressor: str) -> int:
    """Write a compressed store if missing; return uncompressed byte size."""
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if (path / "zarr.json").exists():
        return nbytes
    rng = np.random.default_rng(0)
    # Structured-but-compressible data: a smooth ramp + mild noise so the
    # compressor does real work (pure-random would defeat it, all-zeros
    # would be unrealistically fast).
    base = np.linspace(0, 1000, num=int(np.prod(shape)), dtype=dtype).reshape(shape)
    noise = rng.standard_normal(shape).astype(dtype) * 5
    data = base + noise
    store = LocalStore(str(path))
    arr = zarr.create_array(
        store=store,
        shape=shape,
        chunks=inner_chunk,
        dtype=dtype,
        compressors=[_COMPRESSORS[compressor]()],
    )
    arr[:] = data
    return nbytes


def _compressed_bytes(path: Path) -> int:
    """Sum the on-disk chunk bytes (everything except metadata)."""
    total = 0
    for p in path.rglob("*"):
        if p.is_file() and p.name != "zarr.json":
            total += p.stat().st_size
    return total


def main() -> int:
    """Build, warm up, then profile repeated GPU decode reads."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--compressor", choices=list(_COMPRESSORS), default="zstd")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rmm-gb", type=float, default=4.0)
    # 1 GiB float32 Z-slab, 16x512x512 chunks — the canonical czarr bench shape.
    parser.add_argument("--zdim", type=int, default=64)
    args = parser.parse_args()

    shape = (args.zdim, 512, 512)
    inner_chunk = (16, 512, 512)
    dtype = np.float32

    czarr.configure_gpu(rmm_pool_gb=args.rmm_gb)
    rmm.statistics.enable_statistics()

    nbytes = _build_store(args.path, shape=shape, inner_chunk=inner_chunk, dtype=dtype, compressor=args.compressor)
    comp = _compressed_bytes(args.path)
    print(f"workload: {shape} {np.dtype(dtype).name}  ({nbytes / (1 << 20):.1f} MiB uncompressed)")
    print(f"compressor: {args.compressor}   on-disk: {comp / (1 << 20):.1f} MiB   ratio: {nbytes / max(comp, 1):.2f}x")

    arr = zarr.open_array(store=LocalStore(str(args.path)), mode="r")

    for _ in range(args.warmup):
        _ = arr[:]
        cp.cuda.Stream.null.synchronize()

    # Reset stats after warmup so peak reflects steady-state decode only.
    rmm.statistics.push_statistics()
    samples: list[float] = []
    for i in range(args.reps):
        mark(f"profile.rep.{i}")
        t0 = time.perf_counter()
        with nvtx_range("profile.decode_read", rep=i):
            out = arr[:]
            cp.cuda.Stream.null.synchronize()
        samples.append(time.perf_counter() - t0)
    stats = rmm.statistics.pop_statistics()

    assert isinstance(out, cp.ndarray), f"expected cupy.ndarray, got {type(out).__name__}"
    samples.sort()
    median = samples[len(samples) // 2]
    fastest = samples[0]
    gibs_med = (nbytes / (1 << 30)) / median
    gibs_best = (nbytes / (1 << 30)) / fastest

    print()
    print(f"{'metric':<28}{'value':>16}")
    print("-" * 44)
    print(f"{'median read':<28}{median * 1e3:>13.3f} ms")
    print(f"{'fastest read':<28}{fastest * 1e3:>13.3f} ms")
    print(f"{'throughput (median)':<28}{gibs_med:>13.2f} GiB/s")
    print(f"{'throughput (best)':<28}{gibs_best:>13.2f} GiB/s")
    if stats is not None:
        print(f"{'peak device mem':<28}{stats.peak_bytes / (1 << 20):>13.1f} MiB")
        print(f"{'peak / uncompressed':<28}{stats.peak_bytes / max(nbytes, 1):>13.2f}x")
        print(f"{'allocations (total)':<28}{stats.total_count:>13}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
