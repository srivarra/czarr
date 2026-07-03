"""Experiment: pre-allocated output buffers via ``Codec.decode(out=...)``.

nvCOMP 5.2.0 added an ``out`` argument to the Python ``Codec.decode`` that
accepts a list of pre-allocated, device-accessible buffers — bypassing
nvCOMP's internal output allocation. Combined with our RMM pool, this should
flatten allocation cost off the hot path.

Compare:
  * Default decode:       codec.decode(srcs)                    — internal alloc
  * Pre-alloc decode:     codec.decode(srcs, out=[...])         — user-supplied buffers
  * Pre-alloc + reuse:    same out= list reused across REPEATS  — zero alloc per call

Run:
    uv run --extra cu12 --group test python -m bench.codec.exp_hlif_out
"""

import statistics
import warnings

import cupy as cp
import numpy as np
import rmm
import rmm.statistics as rs
from nvidia import nvcomp
from rmm.allocators.cupy import rmm_cupy_allocator

import czarr
from czarr.codecs import (
    ANS,
    LZ4,
    Bitcomp,
    Snappy,
    Zstd,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

CODECS = [LZ4, Zstd, Snappy, Bitcomp, ANS]
N_CHUNKS = 64
CHUNK_BYTES = 1 * 1024 * 1024
REPEATS = 5


def _gpu_time(fn, repeats=REPEATS) -> float:
    fn()
    cp.cuda.Stream.null.synchronize()
    times = []
    for _ in range(repeats):
        start = cp.cuda.Event()
        stop = cp.cuda.Event()
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, stop))
    return statistics.median(times)


def _measure(codec_cls, chunks_dev) -> dict:
    codec = nvcomp.Codec(algorithm=codec_cls.algorithm.value)
    nv_chunks = [nvcomp.as_array(c) for c in chunks_dev]
    compressed = codec.encode(nv_chunks)
    cp.cuda.Stream.null.synchronize()

    total_bytes = sum(int(c.nbytes) for c in chunks_dev)

    # 1. Default decode — nvCOMP allocates output internally
    with rs.profiler(name=f"{codec_cls.__name__}::default"):
        default_ms = _gpu_time(lambda: codec.decode(compressed))

    # 2. Pre-alloc with fresh allocations each call
    def fresh_prealloc():
        outs = [cp.empty(CHUNK_BYTES, dtype=cp.uint8) for _ in compressed]
        codec.decode(compressed, out=outs)

    with rs.profiler(name=f"{codec_cls.__name__}::prealloc_each"):
        prealloc_ms = _gpu_time(fresh_prealloc)

    # 3. Pre-alloc with one persistent buffer set, reused across calls
    persistent_outs = [cp.empty(CHUNK_BYTES, dtype=cp.uint8) for _ in compressed]
    cp.cuda.Stream.null.synchronize()
    with rs.profiler(name=f"{codec_cls.__name__}::prealloc_reuse"):
        reuse_ms = _gpu_time(lambda: codec.decode(compressed, out=persistent_outs))

    return {
        "codec": codec_cls.__name__,
        "default_gbps": total_bytes / (default_ms / 1000) / 1e9,
        "prealloc_gbps": total_bytes / (prealloc_ms / 1000) / 1e9,
        "reuse_gbps": total_bytes / (reuse_ms / 1000) / 1e9,
        "default_ms": default_ms,
        "prealloc_ms": prealloc_ms,
        "reuse_ms": reuse_ms,
    }


def main() -> None:
    rmm.reinitialize(pool_allocator=True, initial_pool_size=2 << 30)
    cp.cuda.set_allocator(rmm_cupy_allocator)
    czarr.register_nvcomp_allocator()
    rs.enable_statistics()

    print("# Experiment: Codec.decode(out=...) — pre-allocated output buffers")
    print(f"# {N_CHUNKS} chunks of {CHUNK_BYTES // 1024} KiB — total {N_CHUNKS * CHUNK_BYTES // 1024 // 1024} MiB")
    print(f"# Repeats: {REPEATS} (median)")
    print()

    rng = np.random.default_rng(0)
    chunks_host = [rng.integers(0, 255, CHUNK_BYTES, dtype=np.uint8) for _ in range(N_CHUNKS)]
    chunks_dev = [cp.asarray(c) for c in chunks_host]

    print(
        f"{'codec':<10s} | {'default ms':>11s} {'prealloc ms':>12s} {'reuse ms':>10s} | "
        f"{'default GB/s':>13s} {'prealloc GB/s':>14s} {'reuse GB/s':>11s} | speedup"
    )
    print("-" * 110)

    for codec_cls in CODECS:
        try:
            r = _measure(codec_cls, chunks_dev)
            speedup = r["default_ms"] / r["reuse_ms"]
            print(
                f"{r['codec']:<10s} | "
                f"{r['default_ms']:>11.2f} {r['prealloc_ms']:>12.2f} {r['reuse_ms']:>10.2f} | "
                f"{r['default_gbps']:>13.2f} {r['prealloc_gbps']:>14.2f} {r['reuse_gbps']:>11.2f} | "
                f"{speedup:>4.2f}x",
                flush=True,
            )
        except Exception as e:
            print(f"  ! {codec_cls.__name__} failed: {type(e).__name__}: {e}")

    print()
    print("## RMM memory profile (per-codec phase)")
    print(rs.default_profiler_records.report(ordered_by="memory_peak"))


if __name__ == "__main__":
    main()
