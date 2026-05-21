"""Experiment: nvCOMP batch encode/decode parallelism on one stream.

Runs each codec three ways for the same total payload:
  1. Serial loop:   for chunk in chunks: codec.encode(chunk)
  2. Batch API:     codec.encode([chunks...])
  3. (Reference)    single mega-chunk = sum of chunks

Each row = one codec; columns = throughput (GB/s) for serial / batch / mono.
Speedup = batch / serial on decode side.

Run:
    LD_LIBRARY_PATH=...:$LD_LIBRARY_PATH \\
    uv run --extra cu12 --group test python -m bench.codec.exp_batch
"""

from __future__ import annotations

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
    Deflate,
    GDeflate,
    Snappy,
    Zstd,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

ALL_CODECS = [
    LZ4,
    Zstd,
    Snappy,
    Deflate,
    GDeflate,
    Bitcomp,
    ANS,
    Cascaded,
]

# 64 chunks of 1 MB each = 64 MB total — small enough to be quick, big enough
# that batch parallelism should pay off if it works at all.
N_CHUNKS = 64
CHUNK_BYTES = 1 * 1024 * 1024
REPEATS = 5


def _gpu_time(fn, repeats=REPEATS) -> float:
    """Median GPU elapsed time in ms over `repeats`."""
    # Warm-up
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


def _measure(codec_cls, chunks_dev: list, mono_dev) -> dict:
    """Run serial, batch, and mono single-buffer modes for one codec."""
    codec = codec_cls()._get_codec()
    total_bytes = sum(int(c.nbytes) for c in chunks_dev)
    nv_chunks = [nvcomp.as_array(c) for c in chunks_dev]
    nv_mono = nvcomp.as_array(mono_dev)

    # ENCODE — serial loop
    enc_serial_ms = _gpu_time(lambda: [codec.encode(a) for a in nv_chunks])
    # ENCODE — batch (single call)
    enc_batch_ms = _gpu_time(lambda: codec.encode(nv_chunks))
    # ENCODE — single mega-buffer (reference upper bound for amortization)
    enc_mono_ms = _gpu_time(lambda: codec.encode(nv_mono))

    # Get a compressed list to decode against
    compressed_list = codec.encode(nv_chunks)
    compressed_mono = codec.encode(nv_mono)

    # DECODE — serial loop
    dec_serial_ms = _gpu_time(lambda: [codec.decode(c) for c in compressed_list])
    # DECODE — batch (single call)
    dec_batch_ms = _gpu_time(lambda: codec.decode(compressed_list))
    # DECODE — single mega-buffer
    dec_mono_ms = _gpu_time(lambda: codec.decode(compressed_mono))

    return {
        "codec": codec_cls.__name__,
        "total_bytes": total_bytes,
        "enc_serial_gbps": total_bytes / (enc_serial_ms / 1000) / 1e9,
        "enc_batch_gbps": total_bytes / (enc_batch_ms / 1000) / 1e9,
        "enc_mono_gbps": total_bytes / (enc_mono_ms / 1000) / 1e9,
        "dec_serial_gbps": total_bytes / (dec_serial_ms / 1000) / 1e9,
        "dec_batch_gbps": total_bytes / (dec_batch_ms / 1000) / 1e9,
        "dec_mono_gbps": total_bytes / (dec_mono_ms / 1000) / 1e9,
        "dec_speedup_batch_vs_serial": dec_serial_ms / dec_batch_ms,
        "enc_speedup_batch_vs_serial": enc_serial_ms / enc_batch_ms,
    }


def main() -> None:
    rmm.reinitialize(pool_allocator=True, initial_pool_size=1 << 30)
    cp.cuda.set_allocator(rmm_cupy_allocator)
    czarr.register_nvcomp_allocator()

    print("# Experiment: batch vs serial vs mono")
    print(f"# {N_CHUNKS} chunks of {CHUNK_BYTES // 1024} KiB each — total {N_CHUNKS * CHUNK_BYTES // 1024 // 1024} MiB")
    print(f"# Repeats per measurement: {REPEATS} (median)")
    print()

    rng = np.random.default_rng(0)
    # Use uint8 random — nvCOMP byte-codecs handle this without dtype quirks.
    chunks_host = [rng.integers(0, 255, CHUNK_BYTES, dtype=np.uint8) for _ in range(N_CHUNKS)]
    chunks_dev = [cp.asarray(c) for c in chunks_host]
    mono_host = np.concatenate(chunks_host)
    mono_dev = cp.asarray(mono_host)

    print(
        f"{'codec':<10s} | {'enc serial':>10s} {'enc batch':>10s} {'enc mono':>10s} {'enc spd':>8s} | {'dec serial':>10s} {'dec batch':>10s} {'dec mono':>10s} {'dec spd':>8s}"
    )
    print("-" * 110)
    for codec_cls in ALL_CODECS:
        try:
            r = _measure(codec_cls, chunks_dev, mono_dev)
            print(
                f"{r['codec']:<10s} | "
                f"{r['enc_serial_gbps']:>10.2f} {r['enc_batch_gbps']:>10.2f} {r['enc_mono_gbps']:>10.2f} "
                f"{r['enc_speedup_batch_vs_serial']:>8.2f}x | "
                f"{r['dec_serial_gbps']:>10.2f} {r['dec_batch_gbps']:>10.2f} {r['dec_mono_gbps']:>10.2f} "
                f"{r['dec_speedup_batch_vs_serial']:>8.2f}x"
            )
        except Exception as e:
            print(f"{codec_cls.__name__:<30s} FAILED: {type(e).__name__}: {e}")

    print()
    print("Legend: GB/s of UNCOMPRESSED throughput. spd = batch / serial.")


if __name__ == "__main__":
    main()
