"""Register-once slab pool vs per-call cuFile buffer allocation.

The decision this answers: does the :class:`czarr.core.slab.CuFileSlabPool`
recover the ~5x per-chunk regression that the naive
"fresh ``VirtualMemoryResource`` allocate + ``buf_register`` per chunk"
path showed on H200 slice_compare?

Three strategies, each reading N chunks of fixed size from one shard
file into device memory via cuFile:

* ``per_call_vmr`` — the *old* ``CzarrGpuBuffer.empty`` behaviour:
  fresh VMR allocate + ``buf_register`` + read + ``buf_deregister`` per
  chunk.  Pays ``cuMemCreate``/``cuMemMap`` (2 MiB granularity) and a
  cuFile registration on every chunk.
* ``slab`` — the *new* path: one pre-registered slab, per-chunk buffers
  are free-list sub-regions; read into the slab's already-registered
  memory.  No per-chunk VMR alloc, no per-chunk registration.
* ``stock_cupy`` — baseline: plain ``cp.empty`` + ``read_into`` (opens +
  registers an fd handle per call, but the *buffer* is cupy-pooled and
  unregistered, so cuFile copies through its own bounce path).

Smaller chunks make the per-chunk fixed cost dominate, so the slab win
should grow as chunk size shrinks.

Run (A40 compat-mode smoke or H100 real GDS)::

    uv run --extra cu12 python bench/buffer/slab_vs_percall.py \\
        --shard /local/scratch/users/$USER/shard.bin
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cupy as cp
import numpy as np
from cuda.core import Device, VirtualMemoryResource, VirtualMemoryResourceOptions

from czarr import cufile as cufile_runtime
from czarr.core.slab import CuFileSlabPool


def _build_shard(path: Path, *, n_chunks: int, chunk_bytes: int) -> list[tuple[int, int]]:
    """Write a shard of ``n_chunks`` contiguous chunks; return (offset, size) list."""
    rng = np.random.default_rng(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets: list[tuple[int, int]] = []
    with path.open("wb") as f:
        for _ in range(n_chunks):
            chunk = rng.integers(0, 256, size=chunk_bytes, dtype=np.uint8)
            offsets.append((f.tell(), chunk.nbytes))
            f.write(chunk.tobytes())
    return offsets


def _time(fn, *, reps: int, warmup: int) -> tuple[float, float]:
    """Median + min wall time over ``reps`` runs after ``warmup``."""
    for _ in range(warmup):
        fn()
        cp.cuda.Stream.null.synchronize()
    samples: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        cp.cuda.Stream.null.synchronize()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2], samples[0]


def _vmr() -> VirtualMemoryResource:
    dev = Device()
    dev.set_current()
    return VirtualMemoryResource(dev, VirtualMemoryResourceOptions(addr_align=4096, gpu_direct_rdma=True))


def read_per_call_vmr(path: Path, ranges: list[tuple[int, int]], mr: VirtualMemoryResource) -> list[cp.ndarray]:
    """Old path: fresh VMR alloc + register + read + deregister per chunk."""
    stream = Device().default_stream
    out: list[cp.ndarray] = []
    for offset, size in ranges:
        buf = mr.allocate(size, stream=stream)
        view = cp.from_dlpack(buf).view(cp.uint8)
        ptr = int(view.data.ptr)
        cufile_runtime.ensure_buf_registered(ptr, size)
        cufile_runtime.read_into(path, ptr, size, offset)
        cufile_runtime.deregister_buf(ptr)
        out.append(view)
    return out


def read_slab(path: Path, ranges: list[tuple[int, int]], pool: CuFileSlabPool) -> list[object]:
    """New path: free-list sub-region from a pre-registered slab; read in."""
    allocs = []
    for offset, size in ranges:
        a = pool.allocate(size)
        cufile_runtime.read_into(path, a.device_ptr, size, offset)
        allocs.append(a)
    return allocs


def read_stock_cupy(path: Path, ranges: list[tuple[int, int]]) -> list[cp.ndarray]:
    """Baseline: plain cupy buffer + per-call cuFile read (unregistered buffer)."""
    out: list[cp.ndarray] = []
    for offset, size in ranges:
        dev = cp.empty(size, dtype=cp.uint8)
        cufile_runtime.read_into(path, int(dev.data.ptr), size, offset)
        out.append(dev)
    return out


def _verify(path: Path, ranges: list[tuple[int, int]], slab_allocs: list) -> bool:
    """Slab reads must match a host read of the same bytes."""
    with path.open("rb") as f:
        for (offset, size), a in zip(ranges, slab_allocs, strict=True):
            f.seek(offset)
            host = np.frombuffer(f.read(size), dtype=np.uint8)
            got = cp.asnumpy(a.array)
            if not np.array_equal(host, got):
                return False
    return True


def main() -> int:
    """Run the slab vs per-call comparison."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=4)
    args = parser.parse_args()

    if not cufile_runtime.is_available():
        print("cuFile unavailable — aborting")
        return 1

    gds = Path("/proc/driver/nvidia-fs").exists()
    print(f"cuFile available: True   real GDS (nvidia_fs): {gds}")
    print(f"{'chunks':>8} {'KiB':>8} {'per_call_vmr':>14} {'slab':>10} {'stock_cupy':>12} {'slab speedup':>14}")
    print("-" * 72)

    for n_chunks, chunk_kib in [(64, 64), (64, 256), (32, 1024), (16, 4096)]:
        chunk_bytes = chunk_kib << 10
        ranges = _build_shard(args.shard, n_chunks=n_chunks, chunk_bytes=chunk_bytes)

        mr = _vmr()
        pool = CuFileSlabPool(slab_bytes=max(64 << 20, n_chunks * chunk_bytes), register=True)
        try:
            # Correctness check on the slab path once.
            chk = read_slab(args.shard, ranges, pool)
            ok = _verify(args.shard, ranges, chk)
            del chk
            pool.close()
            if not ok:
                print(f"  MISMATCH at chunks={n_chunks} kib={chunk_kib}")
                return 1
            pool = CuFileSlabPool(slab_bytes=max(64 << 20, n_chunks * chunk_bytes), register=True)

            # Default-arg binding captures the current loop values (ruff B023).
            t_pc, _ = _time(
                lambda p=args.shard, r=ranges, m=mr: read_per_call_vmr(p, r, m),
                reps=args.reps,
                warmup=args.warmup,
            )
            # Fresh pool each rep would re-pay slab alloc; we want steady-state
            # reuse, so allocate into the same pool and let GC recycle.
            t_slab, _ = _time(
                lambda p=args.shard, r=ranges, pl=pool: read_slab(p, r, pl),
                reps=args.reps,
                warmup=args.warmup,
            )
            t_stock, _ = _time(
                lambda p=args.shard, r=ranges: read_stock_cupy(p, r),
                reps=args.reps,
                warmup=args.warmup,
            )
        finally:
            pool.close()

        speedup = t_pc / t_slab if t_slab else 0.0
        print(
            f"{n_chunks:>8} {chunk_kib:>8} "
            f"{t_pc * 1e3:>12.3f}ms {t_slab * 1e3:>8.3f}ms {t_stock * 1e3:>10.3f}ms {speedup:>12.2f}x"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
