"""Experiment: end-to-end Zarr write/read performance.

Measures the realistic user-facing path: ``arr[:] = data`` (write) and
``arr[:]`` (read), through Zarr's full codec pipeline.  Three configs:

  1. Host prototype + numcodecs Blosc-lz4 (the canonical CPU baseline)
  2. Host prototype + GPU nvCOMP codec (data flows host → GPU → decode → host)
  3. GPU prototype  + GPU nvCOMP codec (data stays on GPU end-to-end)

Run:
    uv run --extra cu12 --group test python -m bench.zarr.exp_zarr_e2e
"""

from __future__ import annotations

import statistics
import time
import warnings

import cupy as cp
import numpy as np
import rmm
import zarr
from rmm.allocators.cupy import rmm_cupy_allocator
from zarr.codecs import BloscCodec
from zarr.core.buffer import gpu as gpu_buffer
from zarr.storage import MemoryStore

import czarr
from czarr.codecs import (
    ANS,
    LZ4,
    Bitcomp,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Big-enough array that GPU has something to chew on, small enough to run fast.
# 256 MiB total, 16 MiB chunks → 16 chunks
TOTAL_SHAPE = (16384, 4096)  # 16384 * 4096 * 4 bytes (float32) = 256 MiB
CHUNK_SHAPE = (1024, 4096)  # 1024 * 4096 * 4 bytes = 16 MiB / chunk → 16 chunks
DTYPE = np.float32
REPEATS = 3


def _bench_host_cpu(data_host: np.ndarray) -> tuple[float, float]:
    """Host prototype + Blosc-lz4 CPU codec."""
    store = MemoryStore()
    arr = zarr.create_array(
        store=store,
        shape=data_host.shape,
        chunks=CHUNK_SHAPE,
        dtype=str(data_host.dtype),
        compressors=[BloscCodec(cname="lz4", clevel=5)],
    )
    # Warm up
    arr[:] = data_host
    _ = arr[:]

    write_times = []
    for _ in range(REPEATS):
        store2 = MemoryStore()
        arr2 = zarr.create_array(
            store=store2,
            shape=data_host.shape,
            chunks=CHUNK_SHAPE,
            dtype=str(data_host.dtype),
            compressors=[BloscCodec(cname="lz4", clevel=5)],
        )
        t0 = time.perf_counter_ns()
        arr2[:] = data_host
        t1 = time.perf_counter_ns()
        write_times.append((t1 - t0) / 1e6)

    read_times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter_ns()
        _ = arr[:]
        t1 = time.perf_counter_ns()
        read_times.append((t1 - t0) / 1e6)

    return statistics.median(write_times), statistics.median(read_times)


def _bench_host_gpu(data_host: np.ndarray, codec_cls) -> tuple[float, float]:
    """Host prototype + GPU codec — bytes round-trip through GPU."""
    store = MemoryStore()
    arr = zarr.create_array(
        store=store,
        shape=data_host.shape,
        chunks=CHUNK_SHAPE,
        dtype=str(data_host.dtype),
        compressors=[codec_cls()],
    )
    arr[:] = data_host
    _ = arr[:]

    write_times = []
    for _ in range(REPEATS):
        store2 = MemoryStore()
        arr2 = zarr.create_array(
            store=store2,
            shape=data_host.shape,
            chunks=CHUNK_SHAPE,
            dtype=str(data_host.dtype),
            compressors=[codec_cls()],
        )
        t0 = time.perf_counter_ns()
        arr2[:] = data_host
        t1 = time.perf_counter_ns()
        write_times.append((t1 - t0) / 1e6)

    read_times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter_ns()
        _ = arr[:]
        t1 = time.perf_counter_ns()
        read_times.append((t1 - t0) / 1e6)

    return statistics.median(write_times), statistics.median(read_times)


def _bench_gpu_gpu(data_host: np.ndarray, codec_cls) -> tuple[float, float]:
    """GPU prototype + GPU codec — fully on-device pipeline."""
    data_dev = cp.asarray(data_host)
    with zarr.config.set({"buffer": "zarr.core.buffer.gpu.Buffer", "ndbuffer": "zarr.core.buffer.gpu.NDBuffer"}):
        store = MemoryStore()
        arr = zarr.create_array(
            store=store,
            shape=data_host.shape,
            chunks=CHUNK_SHAPE,
            dtype=str(data_host.dtype),
            compressors=[codec_cls()],
        )
        arr[:] = data_dev
        _ = arr.get_basic_selection(prototype=gpu_buffer.buffer_prototype)

        write_times = []
        for _ in range(REPEATS):
            store2 = MemoryStore()
            arr2 = zarr.create_array(
                store=store2,
                shape=data_host.shape,
                chunks=CHUNK_SHAPE,
                dtype=str(data_host.dtype),
                compressors=[codec_cls()],
            )
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter_ns()
            arr2[:] = data_dev
            cp.cuda.Stream.null.synchronize()
            t1 = time.perf_counter_ns()
            write_times.append((t1 - t0) / 1e6)

        read_times = []
        for _ in range(REPEATS):
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter_ns()
            _ = arr.get_basic_selection(prototype=gpu_buffer.buffer_prototype)
            cp.cuda.Stream.null.synchronize()
            t1 = time.perf_counter_ns()
            read_times.append((t1 - t0) / 1e6)

    return statistics.median(write_times), statistics.median(read_times)


def main() -> None:
    rmm.reinitialize(pool_allocator=True, initial_pool_size=2 << 30)
    cp.cuda.set_allocator(rmm_cupy_allocator)
    czarr.register_nvcomp_allocator()

    print("# Experiment: end-to-end Zarr arr[:] write/read")
    nbytes = int(np.prod(TOTAL_SHAPE) * np.dtype(DTYPE).itemsize)
    print(
        f"# Array: {TOTAL_SHAPE} {DTYPE.__name__} = {nbytes // 1024 // 1024} MiB total, "
        f"chunks {CHUNK_SHAPE} = {int(np.prod(CHUNK_SHAPE) * np.dtype(DTYPE).itemsize) // 1024 // 1024} MiB"
    )
    print(f"# Repeats: {REPEATS} (median)")
    print()

    rng = np.random.default_rng(0)
    # Mildly compressible data — ramp pattern
    data_host = (
        rng.integers(0, 64, size=int(np.prod(TOTAL_SHAPE)), dtype=np.int32).astype(np.float32).reshape(TOTAL_SHAPE)
    )

    rows = []

    # Config 1: CPU baseline
    w, r = _bench_host_cpu(data_host)
    rows.append(("Blosc-lz4 (CPU)", "host", w, r))

    # Config 2: GPU codec via host prototype
    for codec_cls in [LZ4, Bitcomp, ANS]:
        try:
            w, r = _bench_host_gpu(data_host, codec_cls)
            rows.append((codec_cls.__name__ + " (GPU)", "host", w, r))
        except Exception as e:
            print(f"  ! {codec_cls.__name__} host-prototype failed: {e}")

    # Config 3: GPU codec via GPU prototype (fully on-device)
    for codec_cls in [LZ4, Bitcomp, ANS]:
        try:
            w, r = _bench_gpu_gpu(data_host, codec_cls)
            rows.append((codec_cls.__name__ + " (GPU)", "gpu", w, r))
        except Exception as e:
            print(f"  ! {codec_cls.__name__} gpu-prototype failed: {e}")

    print(f"{'codec':<22s} {'proto':<6s} | {'write ms':>10s} {'read ms':>10s} | {'write GB/s':>12s} {'read GB/s':>11s}")
    print("-" * 80)
    for codec_name, proto, w, r in rows:
        print(
            f"{codec_name:<22s} {proto:<6s} | "
            f"{w:>10.1f} {r:>10.1f} | "
            f"{nbytes / (w / 1000) / 1e9:>12.2f} {nbytes / (r / 1000) / 1e9:>11.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
