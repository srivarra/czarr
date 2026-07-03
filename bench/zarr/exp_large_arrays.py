"""Experiment: scaling sweep across array sizes.

All previous experiments used 256 MiB. Real workloads run from 1 GiB to
many TiB. This sweep verifies:
  * The chunk-size sweet spot doesn't shift with total volume
  * nvCOMP scratch allocations don't OOM (Zstd was 110× input on small data)
  * GPU throughput stays linear with array size

Sweep: 256 MiB → 1 GiB → 4 GiB. (Anything bigger hits the A40's 48 GB ceiling
once we account for compressed copies, decode outputs, and pool overhead.)

Run:
    uv run --extra cu12 --group test python -m bench.zarr.exp_large_arrays
"""

import statistics
import time
import warnings
from contextlib import contextmanager

import cupy as cp
import numpy as np
import rmm
import rmm.statistics as rs
import zarr
from rmm.allocators.cupy import rmm_cupy_allocator
from zarr.core.buffer import gpu as gpu_buffer
from zarr.storage import MemoryStore

import czarr
from czarr.codecs import (
    ANS,
    LZ4,
    Bitcomp,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

# 16 MiB chunks (the sweet spot from earlier sweep)
CHUNK_BYTES = 16 * 1024 * 1024
DTYPE = np.float32
ITEMSIZE = np.dtype(DTYPE).itemsize  # 4
# Each chunk is (rows, 8192) so chunk_bytes = rows * 8192 * 4
CHUNK_ROWS = CHUNK_BYTES // (8192 * ITEMSIZE)  # = 512
CHUNK_SHAPE = (CHUNK_ROWS, 8192)

# Sweep total array size by varying outer dim. Total bytes = N_CHUNKS * CHUNK_BYTES
# 1 GiB = 64 chunks; 4 GiB = 256 chunks
SIZE_CONFIGS = [
    ("256 MiB", 16),  # 16 chunks  = 256 MiB
    ("1 GiB", 64),  # 64 chunks  = 1 GiB
    ("4 GiB", 256),  # 256 chunks = 4 GiB
]

REPEATS = 2  # bigger arrays — fewer reps to keep runtime sane


@contextmanager
def _gpu_config():
    with zarr.config.set({"buffer": "zarr.core.buffer.gpu.Buffer", "ndbuffer": "zarr.core.buffer.gpu.NDBuffer"}):
        yield


def _measure_one(codec_cls, total_chunks: int) -> dict:
    shape = (total_chunks * CHUNK_ROWS, 8192)
    nbytes = total_chunks * CHUNK_BYTES
    rng = np.random.default_rng(0)
    src = rng.integers(0, 64, np.prod(shape), dtype=np.int32).astype(np.float32).reshape(shape)
    src_dev = cp.asarray(src)

    with _gpu_config():
        # Warm up
        store = MemoryStore()
        arr = zarr.create_array(
            store=store,
            shape=shape,
            chunks=CHUNK_SHAPE,
            dtype="float32",
            compressors=[codec_cls()],
        )
        arr[:] = src_dev
        _ = arr.get_basic_selection(prototype=gpu_buffer.buffer_prototype)

        write_times, read_times = [], []
        for _ in range(REPEATS):
            store2 = MemoryStore()
            arr2 = zarr.create_array(
                store=store2,
                shape=shape,
                chunks=CHUNK_SHAPE,
                dtype="float32",
                compressors=[codec_cls()],
            )
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter_ns()
            arr2[:] = src_dev
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

    w = statistics.median(write_times)
    r = statistics.median(read_times)
    return {
        "write_ms": w,
        "read_ms": r,
        "write_gbps": nbytes / (w / 1000) / 1e9,
        "read_gbps": nbytes / (r / 1000) / 1e9,
    }


def main() -> None:
    rmm.reinitialize(pool_allocator=True, initial_pool_size=8 << 30, maximum_pool_size=24 << 30)
    cp.cuda.set_allocator(rmm_cupy_allocator)
    czarr.register_nvcomp_allocator()
    rs.enable_statistics()

    print("# Experiment: scaling sweep across array sizes")
    print(f"# Chunk size: {CHUNK_BYTES // 1024 // 1024} MiB. Repeats per cell: {REPEATS} (median)")
    print("# RMM pool: 8 GiB initial, 24 GiB max")
    print()

    print(
        f"{'codec':<10s} | {'size':<8s} | {'write GB/s':>11s} {'read GB/s':>10s} | "
        f"{'write ms':>10s} {'read ms':>10s} | RMM peak"
    )
    print("-" * 95)

    for codec_cls in [LZ4, Bitcomp, ANS]:
        for label, n_chunks in SIZE_CONFIGS:
            try:
                with rs.profiler(name=f"{codec_cls.__name__}::{label}"):
                    r = _measure_one(codec_cls, n_chunks)
                rec = rs.default_profiler_records.records.get(f"{codec_cls.__name__}::{label}")
                peak = rec.memory_peak if rec else 0
                print(
                    f"{codec_cls.__name__:<10s} | "
                    f"{label:<8s} | "
                    f"{r['write_gbps']:>11.2f} {r['read_gbps']:>10.2f} | "
                    f"{r['write_ms']:>10.1f} {r['read_ms']:>10.1f} | "
                    f"{peak / 1024 // 1024} MiB",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"{codec_cls.__name__:<10s} | {label:<8s} | FAILED: {type(e).__name__}: {e}",
                    flush=True,
                )

    print()
    print("## RMM peak per (codec, size) — quick sanity")
    print(rs.default_profiler_records.report(ordered_by="memory_peak"))


if __name__ == "__main__":
    main()
