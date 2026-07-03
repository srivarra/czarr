"""Coalesce-vs-serial benchmark on a sharded zarr v3 store.

Compares two ways to read N chunks out of one shard file:

* **Serial**: N sequential ``cuFile`` range reads of (offset, length)
  per chunk — mirrors zarr v3's ``ShardingCodec`` partial-shard loop.
* **Coalesced**: one fused ``cuFile`` read covering the union of the
  N ranges; slice the per-chunk views out of the device buffer locally.

Workload:

* Single shard file with 32 chunks of fixed size (default 256 KiB).
* All chunks contiguous in the shard so the coalesced read fuses to
  one window.
* Read all 32 chunks in one batch.

The "serial" path issues real cuFile reads back-to-back to be apples-
to-apples with the ShardingCodec hot path; we deliberately do NOT run
them concurrently because the codec's loop is a serial ``await``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cupy as cp
import numpy as np

from czarr import cufile as cufile_runtime
from czarr.lowlevel.coalesce import ByteRange, coalesce_ranges


def _build_shard(path: Path, *, n_chunks: int, chunk_bytes: int) -> list[tuple[int, int]]:
    """Write a synthetic shard file: ``n_chunks`` contiguous chunks of random bytes.

    Returns the per-chunk (offset, length) tuples — what a real shard
    index would yield.
    """
    rng = np.random.default_rng(0)
    chunks = [rng.integers(0, 256, size=chunk_bytes, dtype=np.uint8) for _ in range(n_chunks)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        offsets = []
        for c in chunks:
            offsets.append((f.tell(), c.nbytes))
            f.write(c.tobytes())
    return offsets


def _time(fn, *, reps: int, warmup: int) -> tuple[float, float]:
    """Return (median, min) wall time over ``reps`` calls after ``warmup``."""
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


def bench_serial(path: Path, ranges: list[tuple[int, int]]) -> cp.ndarray:
    """One cuFile read per chunk, sequentially — mirrors ShardingCodec."""
    out = cp.empty(sum(length for _, length in ranges), dtype=cp.uint8)
    cursor = 0
    for offset, length in ranges:
        cufile_runtime.read_into(path, int(out[cursor : cursor + length].data.ptr), length, offset)
        cursor += length
    return out


def bench_coalesced(
    path: Path,
    ranges: list[tuple[int, int]],
    *,
    max_fused_bytes: int,
    max_gap_bytes: int,
) -> list[cp.ndarray]:
    """One cuFile read per fused window; slice per-chunk views out locally."""
    fused = coalesce_ranges(
        [ByteRange(offset, length) for offset, length in ranges],
        max_fused_bytes=max_fused_bytes,
        max_gap_bytes=max_gap_bytes,
    )
    buffers: list[cp.ndarray] = []
    for w in fused:
        buf = cp.empty(w.length, dtype=cp.uint8) if w.length > 0 else cp.empty(0, dtype=cp.uint8)
        if w.length > 0:
            cufile_runtime.read_into(path, int(buf.data.ptr), w.length, w.offset)
        buffers.append(buf)
    # Slice per-chunk views (no copies).
    out: list[cp.ndarray | None] = [None] * len(ranges)
    for buf, w in zip(buffers, fused, strict=True):
        for orig_idx, intra_off, length in w.members:
            out[orig_idx] = buf[intra_off : intra_off + length]
    return [o for o in out if o is not None]


def main() -> int:
    """Run coalesce vs serial benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, required=True, help="Where to write the shard file")
    parser.add_argument("--n-chunks", type=int, default=32)
    parser.add_argument("--chunk-bytes", type=int, default=256 << 10)
    parser.add_argument("--max-fused-bytes", type=int, default=64 << 20)
    parser.add_argument("--max-gap-bytes", type=int, default=0)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    print(f"workload: {args.n_chunks} chunks of {args.chunk_bytes / (1 << 10):.0f} KiB each")
    print(f"shard total: {args.n_chunks * args.chunk_bytes / (1 << 20):.1f} MiB")
    print(f"fuse cap: {args.max_fused_bytes / (1 << 20):.0f} MiB; gap cap: {args.max_gap_bytes} B")

    ranges = _build_shard(args.shard, n_chunks=args.n_chunks, chunk_bytes=args.chunk_bytes)
    if not cufile_runtime.is_available():
        print("cuFile unavailable on this host — aborting")
        return 1
    print(f"cuFile available: {cufile_runtime.is_available()}")

    # Correctness: both paths must yield the same bytes per chunk.
    serial_out = bench_serial(args.shard, ranges)
    coalesced_out = bench_coalesced(
        args.shard,
        ranges,
        max_fused_bytes=args.max_fused_bytes,
        max_gap_bytes=args.max_gap_bytes,
    )
    cp.cuda.Stream.null.synchronize()
    cursor = 0
    for i, (_offset, length) in enumerate(ranges):
        serial_slice = serial_out[cursor : cursor + length]
        cursor += length
        if not bool((serial_slice == coalesced_out[i]).all()):
            print(f"  MISMATCH at chunk {i}")
            return 1
    print("correctness: OK")

    # Time both paths.
    t_serial_med, t_serial_min = _time(lambda: bench_serial(args.shard, ranges), reps=args.reps, warmup=args.warmup)
    t_coal_med, t_coal_min = _time(
        lambda: bench_coalesced(
            args.shard,
            ranges,
            max_fused_bytes=args.max_fused_bytes,
            max_gap_bytes=args.max_gap_bytes,
        ),
        reps=args.reps,
        warmup=args.warmup,
    )

    total_bytes = args.n_chunks * args.chunk_bytes
    print()
    print(f"{'path':<14}{'median ms':>14}{'min ms':>12}{'GiB/s (med)':>14}")
    print("-" * 54)
    for name, med, mn in [("serial", t_serial_med, t_serial_min), ("coalesced", t_coal_med, t_coal_min)]:
        gibs = (total_bytes / (1 << 30)) / med
        print(f"{name:<14}{med * 1e3:>14.4f}{mn * 1e3:>12.4f}{gibs:>14.2f}")
    print(f"speedup (serial/coalesced): {t_serial_med / t_coal_med:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
