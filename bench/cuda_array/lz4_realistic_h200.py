"""Realistic czarr workload — Shuffle + BitRound + LZ4 on smooth float32.

The previous bench (``lz4_native_e2e.py``) used 64 chunks of 16 MiB random
float32 — incompressible, large blocks — and found native LZ4 at parity
with nvcomp.  The spike at 173 GiB/s was at the opposite extreme: 1024
chunks of 64 KiB highly-compressible ASCII.

This bench mirrors what scientific Zarr users actually run: smooth
float32 with light noise, fed through the typical filter chain
(Shuffle for byte-stream regrouping, BitRound to drop precision) before
LZ4 sees it.  After filtering, the LZ4 input is small and uniform; the
spike's regime returns.

Two passes:

A. Phase 1 baseline geometry — 64 chunks of (16, 512, 512) = 16 MiB
   raw per chunk.  Measures the same chunk count we shipped on with
   data that *should* compress.
B. Spike-like geometry — 1024 chunks of (1, 256, 256) = 256 KiB raw
   per chunk.  Closer to the spike's 64 KiB shape; more blocks fill
   H200's 132 SMs.

For each pass: compression ratio of the filter chain output; native
vs nvcomp decode-only timings; bit-exact validation off the hot loop.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from typing import Any

import cupy as cp
import numpy as np
import zarr
from numcodecs import LZ4 as NumcodecsLZ4
from numcodecs import BitRound as NumcodecsBitRound
from numcodecs import Shuffle as NumcodecsShuffle
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import gpu as gpu_buffer

import czarr

DTYPE = np.float32


def _make_scientific_data(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Smooth float32 with light noise — typical scientific-data shape.

    Produces a 3-D array with a low-frequency trend (sin/cos in each axis)
    plus Gaussian noise.  Shuffle + BitRound compresses this much better
    than pure random — exactly the regime where LZ4 wins.
    """
    z, y, x = shape
    zz = np.arange(z, dtype=np.float32)[:, None, None]
    yy = np.arange(y, dtype=np.float32)[None, :, None]
    xx = np.arange(x, dtype=np.float32)[None, None, :]
    smooth = np.sin(zz / 4) + np.cos(yy / 32) + np.sin(xx / 32)
    noise = rng.standard_normal(shape).astype(np.float32) * 0.05
    return (smooth + noise).astype(np.float32)


def make_filtered_fixture(
    *,
    n_chunks: int,
    chunk_shape: tuple[int, ...],
    keepbits: int = 8,
    elementsize: int = 4,
) -> tuple[list[bytes], list[bytes], list[np.ndarray]]:
    """Build ``n_chunks`` worth of scientific data, run through filter chain.

    Returns
    -------
    encoded_lz4: list[bytes]
        N LZ4-encoded chunks (the bytes that would land on disk).
    filter_output: list[bytes]
        N filter-chain outputs (Shuffle + BitRound; the LZ4 inputs).  These
        are what LZ4 decode should reproduce.
    raw_arrays: list[np.ndarray]
        N original float32 arrays (just for completeness; not used by the bench).
    """
    rng = np.random.default_rng(0)
    lz4 = NumcodecsLZ4(acceleration=1)
    shuffle = NumcodecsShuffle(elementsize=elementsize)
    bitround = NumcodecsBitRound(keepbits=keepbits)

    encoded_lz4: list[bytes] = []
    filter_output: list[bytes] = []
    raw_arrays: list[np.ndarray] = []
    for _ in range(n_chunks):
        arr = _make_scientific_data(chunk_shape, rng)
        # BitRound operates on the array; output stays float32 but with
        # cleared mantissa bits.  Then Shuffle reinterprets the byte stream.
        rounded = bitround.encode(arr)
        # Shuffle is byte-shuffle: numcodecs.Shuffle works on byte sequences.
        shuffled = shuffle.encode(rounded.tobytes())
        # LZ4-encode the shuffled bytes.
        compressed = bytes(lz4.encode(shuffled))
        encoded_lz4.append(compressed)
        filter_output.append(bytes(shuffled))
        raw_arrays.append(arr)
    return encoded_lz4, filter_output, raw_arrays


def _make_array_spec(nbytes: int) -> ArraySpec:
    return ArraySpec(
        shape=(nbytes,),
        dtype=zarr.dtype.parse_dtype("uint8", zarr_format=3),
        fill_value=0,
        prototype=gpu_buffer.buffer_prototype,
        config=zarr.core.array.ArrayConfig.from_dict({}),
    )


def upload_to_device(encoded: list[bytes]) -> list[Any]:
    """Pre-upload encoded bytes as GPU-prototype Buffers."""
    prototype = gpu_buffer.buffer_prototype
    return [prototype.buffer.from_bytes(enc) for enc in encoded]


def decode_once(codec: Any, items: list[tuple[Any, ArraySpec]]) -> list[Any]:
    return asyncio.run(codec.decode(items))


def validate_bitstream(items: list[tuple[Any, ArraySpec]], filter_output: list[bytes]) -> None:
    """LZ4 decode output must match the pre-LZ4 filter output bit-exact."""
    for backend in ("native", "nvcomp"):
        codec = czarr.LZ4(backend=backend)
        decoded = decode_once(codec, items)
        for i, (buf, exp) in enumerate(zip(decoded, filter_output, strict=True)):
            actual = buf.to_bytes()
            if actual != exp:
                bad_idx = next(
                    (j for j, (a, b) in enumerate(zip(actual, exp, strict=True)) if a != b),
                    -1,
                )
                raise AssertionError(
                    f"{backend} chunk {i} mismatch at byte {bad_idx}: "
                    f"got {actual[bad_idx : bad_idx + 8]!r}, expected {exp[bad_idx : bad_idx + 8]!r}"
                )
        print(f"  {backend}: {len(decoded)} chunks bit-exact ✓")


def time_decode(codec: Any, items: list[tuple[Any, ArraySpec]], *, reps: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        decode_once(codec, items)
        cp.cuda.Stream.null.synchronize()
    samples: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        decode_once(codec, items)
        cp.cuda.Stream.null.synchronize()
        samples.append(time.perf_counter() - t0)
    return samples


def _gibs(nbytes: int, seconds: float) -> float:
    return (nbytes / (1 << 30)) / seconds


def run_pass(
    *,
    label: str,
    n_chunks: int,
    chunk_shape: tuple[int, ...],
    reps: int,
    warmup: int,
) -> None:
    nbytes_per_chunk = int(np.prod(chunk_shape) * DTYPE().itemsize)
    nbytes_total = nbytes_per_chunk * n_chunks
    print(f"\n=== pass {label} — {n_chunks} chunks of {chunk_shape} ({nbytes_per_chunk / (1 << 20):.2f} MiB each) ===")

    encoded, filter_output, _raw = make_filtered_fixture(n_chunks=n_chunks, chunk_shape=chunk_shape)
    avg_comp = sum(len(e) for e in encoded) // n_chunks
    ratio = avg_comp / nbytes_per_chunk * 100
    print(f"  raw total: {nbytes_total / (1 << 30):.3f} GiB")
    print(f"  avg compressed: {avg_comp / 1024:.1f} KiB ({ratio:.1f}% of raw)")
    print(f"  pre-LZ4 filter output per chunk: {len(filter_output[0]) / (1 << 20):.2f} MiB")

    device_buffers = upload_to_device(encoded)
    items = [(buf, _make_array_spec(nbytes_per_chunk)) for buf in device_buffers]
    cp.cuda.Stream.null.synchronize()

    print("  bit-exact validation (off hot path):")
    validate_bitstream(items, filter_output)

    print(f"  timing (reps={reps}, warmup={warmup}):")
    native_samples = time_decode(czarr.LZ4(backend="native"), items, reps=reps, warmup=warmup)
    nvcomp_samples = time_decode(czarr.LZ4(backend="nvcomp"), items, reps=reps, warmup=warmup)

    def _stats(s: list[float]) -> tuple[float, float]:
        return statistics.median(s), min(s)

    n_med, n_min = _stats(native_samples)
    c_med, c_min = _stats(nvcomp_samples)

    print()
    # GiB/s reported on the **filter-chain output** size, not the LZ4-compressed
    # input — that's the throughput of the decode kernel from the user's view.
    decoded_total = sum(len(b) for b in filter_output)
    print(f"  {'backend':<10}{'median ms':>12}{'min ms':>10}{'GiB/s (decoded)':>18}")
    print(f"  {'-' * 50}")
    print(f"  {'native':<10}{n_med * 1e3:>12.2f}{n_min * 1e3:>10.2f}{_gibs(decoded_total, n_med):>18.2f}")
    print(f"  {'nvcomp':<10}{c_med * 1e3:>12.2f}{c_min * 1e3:>10.2f}{_gibs(decoded_total, c_med):>18.2f}")
    print(f"  speedup (native vs nvcomp): {c_med / n_med:.2f}x")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--warmup", type=int, default=3)
    args = p.parse_args()

    zarr.config.set({"buffer": "zarr.core.buffer.gpu.Buffer", "ndbuffer": "zarr.core.buffer.gpu.NDBuffer"})
    try:
        # Pass A: Phase 1 geometry.  Should show some compression from the
        # filter chain but the per-chunk size is still 16 MiB, so the
        # kernel's per-warp parallelism limit still applies.
        run_pass(
            label="A (Phase 1 baseline geometry)",
            n_chunks=64,
            chunk_shape=(16, 512, 512),
            reps=args.reps,
            warmup=args.warmup,
        )

        # Pass B: Spike-like geometry.  1024 chunks of 1 MiB lights up
        # H200's 132 SMs (8 blocks/SM at full occupancy).
        run_pass(
            label="B (spike-like geometry)",
            n_chunks=1024,
            chunk_shape=(1, 256, 256),
            reps=args.reps,
            warmup=args.warmup,
        )
    finally:
        zarr.config.reset()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
