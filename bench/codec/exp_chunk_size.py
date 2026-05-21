"""Experiment: GPU vs CPU at varying chunk sizes.

Holds total bytes constant (256 MiB), sweeps chunk count to find the
chunk-size sweet spot. Uses batch encode/decode for GPU (best case),
multi-threaded Blosc for CPU (best case).

We expect:
  * Tiny chunks (64 KiB): GPU launch overhead kills it; CPU wins
  * Sweet spot somewhere mid-MiB: GPU catches up
  * Large chunks (multi-MiB): GPU pulls ahead, especially for Bitcomp/ANS

Run:
    uv run --extra cu12 --group test python -m bench.codec.exp_chunk_size
"""

from __future__ import annotations

import statistics
import time
import warnings

import cupy as cp
import numcodecs
import numpy as np
import rmm
from nvidia import nvcomp
from rmm.allocators.cupy import rmm_cupy_allocator

import czarr
from czarr.codecs import (
    ANS,
    LZ4,
    Bitcomp,
    Zstd,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

TOTAL_BYTES = 256 * 1024 * 1024
CHUNK_SIZES_KB = [64, 256, 1024, 4096, 16384, 65536]  # 64 KiB → 64 MiB
REPEATS = 3


def _gpu_decode_time_ms(codec_cls, chunks_dev) -> float:
    codec = nvcomp.Codec(algorithm=codec_cls.algorithm.value)
    nv_chunks = [nvcomp.as_array(c) for c in chunks_dev]
    compressed = codec.encode(nv_chunks)
    cp.cuda.Stream.null.synchronize()

    # Warm up
    codec.decode(compressed)
    cp.cuda.Stream.null.synchronize()

    times = []
    for _ in range(REPEATS):
        start = cp.cuda.Event()
        stop = cp.cuda.Event()
        start.record()
        codec.decode(compressed)
        stop.record()
        stop.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, stop))
    return statistics.median(times)


def _cpu_blosc_decode_time_ms(cname, chunks_host) -> float:
    codec = numcodecs.Blosc(cname=cname, clevel=5)
    encoded = [codec.encode(c) for c in chunks_host]
    # Warm up
    for e in encoded:
        codec.decode(e)
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter_ns()
        for e in encoded:
            codec.decode(e)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e6)
    return statistics.median(times)


def main() -> None:
    rmm.reinitialize(pool_allocator=True, initial_pool_size=2 << 30)
    cp.cuda.set_allocator(rmm_cupy_allocator)
    czarr.register_nvcomp_allocator()

    print("# Experiment: GPU vs CPU at varying chunk sizes")
    print(f"# Total payload: {TOTAL_BYTES // 1024 // 1024} MiB held constant; chunk count varies")
    print("# GPU = nvCOMP batch decode; CPU = Blosc-lz4 16-thread")
    print(f"# Repeats: {REPEATS} (median)")
    print()

    rng = np.random.default_rng(0)

    headers = (
        f"{'chunk':<8s} {'n_chunks':>9s} | "
        + "  ".join(f"{c.__name__:>7s}" for c in [LZ4, Zstd, Bitcomp, ANS])
        + f"  | {'CPU-lz4':>7s} {'CPU-zstd':>8s}"
    )
    print(headers)
    print("-" * len(headers))

    for chunk_kb in CHUNK_SIZES_KB:
        chunk_bytes = chunk_kb * 1024
        n_chunks = TOTAL_BYTES // chunk_bytes
        chunks_host = [rng.integers(0, 255, chunk_bytes, dtype=np.uint8) for _ in range(n_chunks)]
        chunks_dev = [cp.asarray(c) for c in chunks_host]

        gpu_results = {}
        for codec_cls in [LZ4, Zstd, Bitcomp, ANS]:
            try:
                ms = _gpu_decode_time_ms(codec_cls, chunks_dev)
                gpu_results[codec_cls.__name__] = TOTAL_BYTES / (ms / 1000) / 1e9
            except Exception:
                gpu_results[codec_cls.__name__] = float("nan")

        try:
            cpu_lz4 = TOTAL_BYTES / (_cpu_blosc_decode_time_ms("lz4", chunks_host) / 1000) / 1e9
        except Exception:
            cpu_lz4 = float("nan")
        try:
            cpu_zstd = TOTAL_BYTES / (_cpu_blosc_decode_time_ms("zstd", chunks_host) / 1000) / 1e9
        except Exception:
            cpu_zstd = float("nan")

        # Format chunk size label
        if chunk_kb < 1024:
            label = f"{chunk_kb}KiB"
        else:
            label = f"{chunk_kb // 1024}MiB"

        print(
            f"{label:<8s} {n_chunks:>9d} | "
            + "  ".join(f"{gpu_results[k]:>7.1f}" for k in ["LZ4", "Zstd", "Bitcomp", "ANS"])
            + f"  | {cpu_lz4:>7.1f} {cpu_zstd:>8.1f}",
            flush=True,
        )

    print()
    print("Legend: GB/s decode throughput on uncompressed bytes.")


if __name__ == "__main__":
    main()
