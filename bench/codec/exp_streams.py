"""Experiment: concurrent decode across N CUDA streams.

We have 64 MiB of compressed chunks. Distribute them across N independent
streams and decode in parallel. Each stream gets its own ``nvcomp.Codec``
(thread-safety + scratch isolation per nvCOMP docs).

We compare 1, 2, 4, 8, 16 streams. Each codec gets a batch of chunks via
the batch API and runs async on its stream; we block on all streams and
time the wall-clock window.

Important: streams + codecs are pre-built once and reused. Recreating
them per measurement triggers a segfault during the second iteration's
codec construction (exact root cause unknown — likely a stream-lifetime
issue between cupy and nvCOMP).

Run:
    LD_LIBRARY_PATH=...:$LD_LIBRARY_PATH \\
    uv run --extra cu12 --group test python -m bench.codec.exp_streams
"""

import statistics
import warnings

import cupy as cp
import numpy as np
import rmm
from nvidia import nvcomp
from rmm.allocators.cupy import rmm_cupy_allocator

import czarr
from czarr.codecs import (
    ANS,
    LZ4,
    Bitcomp,
    Cascaded,
    Snappy,
    Zstd,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

CODECS = [
    LZ4,
    Zstd,
    Snappy,
    Bitcomp,
    ANS,
    Cascaded,
]

N_CHUNKS = 64
CHUNK_BYTES = 1 * 1024 * 1024
STREAM_COUNTS = [1, 2, 4, 8, 16]
REPEATS = 5


def _measure_streams_with_resources(
    codec_cls,
    n_streams: int,
    chunks_dev: list,
    all_streams: list,
) -> dict:
    """Decode all chunks split across N CUDA streams concurrently.

    Each call creates one fresh ``nvcomp.Codec`` per stream (codec scratch
    is per-instance, but the underlying streams are reused).
    """
    streams = all_streams[:n_streams]
    codecs = [nvcomp.Codec(algorithm=codec_cls.algorithm.value, cuda_stream=int(s.ptr)) for s in streams]

    chunk_groups: list[list] = [[] for _ in range(n_streams)]
    for i, c in enumerate(chunks_dev):
        chunk_groups[i % n_streams].append(nvcomp.as_array(c))

    compressed_groups = [codecs[0].encode(group) for group in chunk_groups]
    cp.cuda.Stream.null.synchronize()

    # Warm up
    for codec, group in zip(codecs, compressed_groups, strict=True):
        codec.decode(group)
    for s in streams:
        s.synchronize()

    times = []
    for _ in range(REPEATS):
        start = cp.cuda.Event()
        stop = cp.cuda.Event()
        start.record()
        for codec, group in zip(codecs, compressed_groups, strict=True):
            codec.decode(group)
        for s in streams:
            s.synchronize()
        stop.record()
        stop.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, stop))

    total_bytes = sum(int(c.nbytes) for c in chunks_dev)
    elapsed = statistics.median(times)
    return {
        "n_streams": n_streams,
        "elapsed_ms": elapsed,
        "gbps": total_bytes / (elapsed / 1000) / 1e9,
    }


def main() -> None:
    rmm.reinitialize(pool_allocator=True, initial_pool_size=2 << 30)
    cp.cuda.set_allocator(rmm_cupy_allocator)
    czarr.register_nvcomp_allocator()

    print("# Experiment: multi-stream concurrent decode")
    print(f"# {N_CHUNKS} chunks of {CHUNK_BYTES // 1024} KiB each — total {N_CHUNKS * CHUNK_BYTES // 1024 // 1024} MiB")
    print(f"# Repeats per config: {REPEATS} (median)")
    print()

    rng = np.random.default_rng(0)
    chunks_host = [rng.integers(0, 255, CHUNK_BYTES, dtype=np.uint8) for _ in range(N_CHUNKS)]
    chunks_dev = [cp.asarray(c) for c in chunks_host]

    # Build the maximum number of streams we'll need ONCE; subsets get sliced.
    all_streams = [cp.cuda.Stream(non_blocking=True) for _ in range(max(STREAM_COUNTS))]

    header = f"{'codec':<10s} | " + " ".join(f"{'n=' + str(n):>10s}" for n in STREAM_COUNTS) + "  | scaling"
    print(header)
    print("-" * len(header))

    for codec_cls in CODECS:
        rows = []
        for n_streams in STREAM_COUNTS:
            try:
                r = _measure_streams_with_resources(codec_cls, n_streams, chunks_dev, all_streams)
                rows.append(r["gbps"])
            except Exception as e:
                rows.append(float("nan"))
                print(f"  ! {codec_cls.__name__} n={n_streams} failed: {type(e).__name__}: {e}")
        baseline = rows[0] if rows else float("nan")
        scaling = max(rows) / baseline if baseline > 0 else float("nan")
        print(
            f"{codec_cls.__name__:<10s} | " + " ".join(f"{v:>10.2f}" for v in rows) + f"  | {scaling:>4.2f}x",
            flush=True,
        )

    print()
    print("Legend: GB/s uncompressed decode throughput. n = stream count. scaling = max / n=1.")


if __name__ == "__main__":
    main()
