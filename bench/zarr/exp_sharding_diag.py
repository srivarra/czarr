"""Diagnostic: why is Zarr sharding 12-46× slower than non-sharded for our codec?

Hypothesis tests:
  H1. Per-chunk aligned-copy overhead (from _ensure_aligned)
  H2. Inner chunks too small (256 KiB lands in CPU-wins zone)
  H3. Sharding codec dispatches inner chunks one-at-a-time (no batch to us)
  H4. Sharding adds index/framing overhead

We measure the same data:
  * No sharding, 16 MiB chunks
  * Sharded, inner 1 MiB / outer 16 MiB
  * Sharded, inner 4 MiB / outer 16 MiB

For each, count how many times our `_batch_sync` is called and with what
batch sizes — that confirms or rejects H3 directly.

Run:
    uv run --extra cu12 --group test python -m bench.zarr.exp_sharding_diag
"""

from __future__ import annotations

import statistics
import time
import warnings
from contextlib import contextmanager

import cupy as cp
import numpy as np
import rmm
import zarr
from rmm.allocators.cupy import rmm_cupy_allocator
from zarr.core.buffer import gpu as gpu_buffer
from zarr.storage import MemoryStore

import czarr
import czarr.codecs.base as cb
from czarr.codecs import ANS

warnings.filterwarnings("ignore", category=DeprecationWarning)

TOTAL_SHAPE = (8192, 8192)  # 256 MiB
DTYPE = np.float32
REPEATS = 3


@contextmanager
def _gpu_config():
    with zarr.config.set({"buffer": "zarr.core.buffer.gpu.Buffer", "ndbuffer": "zarr.core.buffer.gpu.NDBuffer"}):
        yield


def _instrument():
    """Wrap _batch_sync to count call count + batch size distribution."""
    orig = cb.Codec._batch_sync
    stats = {"calls": 0, "total_chunks": 0, "batch_sizes": []}

    def traced(self, items, op):
        if op == "decode":
            n_chunks = sum(1 for c, _ in items if c is not None)
            stats["calls"] += 1
            stats["total_chunks"] += n_chunks
            stats["batch_sizes"].append(n_chunks)
        return orig(self, items, op)

    cb.Codec._batch_sync = traced
    return stats, orig


def _restore(orig):
    cb.Codec._batch_sync = orig


def _measure(payload, codec, chunks, shards):
    kwargs = dict(shape=payload.shape, chunks=chunks, dtype="float32", compressors=[codec])
    if shards is not None:
        kwargs["shards"] = shards

    # Warm up
    store = MemoryStore()
    arr = zarr.create_array(store=store, **kwargs)
    arr[:] = payload
    _ = arr.get_basic_selection(prototype=gpu_buffer.buffer_prototype)

    write_times, read_times = [], []
    for _ in range(REPEATS):
        store2 = MemoryStore()
        arr2 = zarr.create_array(store=store2, **kwargs)
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter_ns()
        arr2[:] = payload
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter_ns()
        write_times.append((t1 - t0) / 1e6)

    for _ in range(REPEATS):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter_ns()
        _ = arr.get_basic_selection(prototype=gpu_buffer.buffer_prototype)
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter_ns()
        read_times.append((t1 - t0) / 1e6)

    return statistics.median(write_times), statistics.median(read_times)


def main() -> None:
    rmm.reinitialize(pool_allocator=True, initial_pool_size=4 << 30)
    cp.cuda.set_allocator(rmm_cupy_allocator)
    czarr.register_nvcomp_allocator()

    nbytes = int(np.prod(TOTAL_SHAPE) * np.dtype(DTYPE).itemsize)
    print("# Diagnostic: Zarr sharding regression with our codec")
    print(f"# Array {TOTAL_SHAPE} {DTYPE.__name__} = {nbytes // 1024 // 1024} MiB total")
    print(f"# Repeats: {REPEATS} (median); GPU prototype")
    print()

    rng = np.random.default_rng(0)
    data = rng.integers(0, 64, np.prod(TOTAL_SHAPE), dtype=np.int32).astype(np.float32).reshape(TOTAL_SHAPE)
    data_dev = cp.asarray(data)

    configs = [
        ("no shard, 16 MiB chunks", (512, 8192), None),
        ("shard 16 MiB / inner 4 MiB", (128, 8192), (512, 8192)),
        ("shard 16 MiB / inner 1 MiB", (32, 8192), (512, 8192)),
        ("shard 16 MiB / inner 256 KiB", (8, 8192), (512, 8192)),
    ]

    print(
        f"{'config':<35s} | {'write GB/s':>11s} {'read GB/s':>10s} | "
        f"{'#calls':>7s} {'#chunks':>8s} {'avg batch':>10s} {'max batch':>10s}"
    )
    print("-" * 100)

    with _gpu_config():
        for label, chunks, shards in configs:
            stats, orig = _instrument()
            try:
                w, r = _measure(data_dev, ANS(), chunks, shards)
                avg_batch = stats["total_chunks"] / max(stats["calls"], 1)
                max_batch = max(stats["batch_sizes"]) if stats["batch_sizes"] else 0
                print(
                    f"{label:<35s} | "
                    f"{nbytes / (w / 1000) / 1e9:>11.2f} {nbytes / (r / 1000) / 1e9:>10.2f} | "
                    f"{stats['calls']:>7d} {stats['total_chunks']:>8d} {avg_batch:>10.1f} {max_batch:>10d}",
                    flush=True,
                )
            except Exception as e:
                print(f"{label:<35s} | FAILED: {type(e).__name__}: {e}", flush=True)
            finally:
                _restore(orig)

    print()
    print("Interpretation:")
    print("  - avg batch = 1  → sharding calls our codec one chunk at a time (no batching)")
    print("  - avg batch ≫ 1  → sharding batches well; perf hit is something else")


if __name__ == "__main__":
    main()
