"""Experiment: Zarr sharding codec — bigger batch to nvCOMP per file.

Sharding bundles many small chunks into one shard (one file). The decode
path passes the *whole shard's* chunks to nvCOMP in a single batch call,
which our prior experiments showed is up to 39x faster than serial.

Compare:
  * No sharding: 16 MiB chunks, one file each (16 files)
  * Sharded: 1 MiB chunks, 16 per shard, one file per shard (16 files)
  * Tiny+sharded: 256 KiB chunks, 64 per shard, one file per shard (16 files)

Total payload constant (256 MiB) so we measure the *sharding* effect, not
size-dependent overhead.

Run:
    uv run --extra cu12 --group test python -m bench.zarr.exp_sharding
"""

import statistics
import time
import warnings

import cupy as cp
import numpy as np
import rmm
import zarr
from rmm.allocators.cupy import rmm_cupy_allocator
from zarr.core.buffer import gpu as gpu_buffer
from zarr.storage import MemoryStore

import czarr
from czarr.codecs import (
    ANS,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

TOTAL_SHAPE = (8192, 8192)  # 256 MiB float32
DTYPE = np.float32
REPEATS = 3


def _gpu_config():
    return zarr.config.set({"buffer": "zarr.core.buffer.gpu.Buffer", "ndbuffer": "zarr.core.buffer.gpu.NDBuffer"})


def _create_array(store, shape, dtype, chunks, shards, codec):
    """Create_array with optional sharding (``shards=`` outer, ``chunks=`` inner)."""
    kwargs = dict(store=store, shape=shape, dtype=dtype, chunks=chunks, compressors=[codec])
    if shards is not None:
        kwargs["shards"] = shards
    return zarr.create_array(**kwargs)


def _measure(data, codec, chunks, shards, use_gpu_proto):
    """Run write/read REPEATS times under the right buffer prototype."""
    ctx = _gpu_config() if use_gpu_proto else _no_op_ctx()

    with ctx:
        # Warm-up
        store = MemoryStore()
        arr = _create_array(store, data.shape, str(data.dtype), chunks, shards, codec)
        arr[:] = data
        if use_gpu_proto:
            _ = arr.get_basic_selection(prototype=gpu_buffer.buffer_prototype)
        else:
            _ = arr[:]

        write_times, read_times = [], []
        for _ in range(REPEATS):
            store2 = MemoryStore()
            arr2 = _create_array(store2, data.shape, str(data.dtype), chunks, shards, codec)
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter_ns()
            arr2[:] = data
            cp.cuda.Stream.null.synchronize()
            t1 = time.perf_counter_ns()
            write_times.append((t1 - t0) / 1e6)

        for _ in range(REPEATS):
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter_ns()
            if use_gpu_proto:
                _ = arr.get_basic_selection(prototype=gpu_buffer.buffer_prototype)
            else:
                _ = arr[:]
            cp.cuda.Stream.null.synchronize()
            t1 = time.perf_counter_ns()
            read_times.append((t1 - t0) / 1e6)

    return statistics.median(write_times), statistics.median(read_times)


from contextlib import contextmanager


@contextmanager
def _no_op_ctx():
    yield


def main() -> None:
    rmm.reinitialize(pool_allocator=True, initial_pool_size=2 << 30)
    cp.cuda.set_allocator(rmm_cupy_allocator)
    czarr.register_nvcomp_allocator()

    nbytes = int(np.prod(TOTAL_SHAPE) * np.dtype(DTYPE).itemsize)
    print("# Experiment: Zarr sharding codec impact on GPU throughput")
    print(f"# Array {TOTAL_SHAPE} {DTYPE.__name__} = {nbytes // 1024 // 1024} MiB total")
    print(f"# Repeats: {REPEATS} (median)")
    print()

    rng = np.random.default_rng(0)
    data = rng.integers(0, 64, np.prod(TOTAL_SHAPE), dtype=np.int32).astype(np.float32).reshape(TOTAL_SHAPE)

    # Configs: (label, codec, inner-chunks, outer-shards-or-None, gpu_proto)
    # In zarr 3: chunks=inner-shape; shards=outer-shape (None disables sharding).
    configs = [
        ("no shard, 16 MiB chunks, GPU proto", ANS(), (1024, 8192), None, True),
        ("shard outer 16 MiB / inner 1 MiB, GPU", ANS(), (64, 8192), (1024, 8192), True),
        ("shard outer 16 MiB / inner 256 KiB, GPU", ANS(), (16, 8192), (1024, 8192), True),
        ("no shard, 16 MiB chunks, host proto", ANS(), (1024, 8192), None, False),
        ("shard outer 16 MiB / inner 1 MiB, host", ANS(), (64, 8192), (1024, 8192), False),
    ]

    print(f"{'config':<45s} | {'write ms':>10s} {'read ms':>10s} | {'write GB/s':>11s} {'read GB/s':>10s}")
    print("-" * 100)

    for label, codec, chunks, shards, gpu_proto in configs:
        try:
            payload = cp.asarray(data) if gpu_proto else data
            w, r = _measure(payload, codec, chunks, shards, gpu_proto)
            print(
                f"{label:<45s} | "
                f"{w:>10.1f} {r:>10.1f} | "
                f"{nbytes / (w / 1000) / 1e9:>11.2f} {nbytes / (r / 1000) / 1e9:>10.2f}",
                flush=True,
            )
        except Exception as e:
            print(f"{label:<45s} FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
