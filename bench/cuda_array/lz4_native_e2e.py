"""End-to-end bench for the native LZ4 backend on H200.

Validates the kernel runs against real ``numcodecs.LZ4``-encoded data,
the bitstream is identical between the two backends, and the native path
actually saturates the GPU.

Workload mirrors the 1 GiB Z-slab from ``bench/zarr/slice_compare.py``:
64 chunks of 16x512x512 float32 ≈ 16 MiB each.  Each chunk is encoded
on the CPU via ``numcodecs.LZ4`` (the on-disk format czarr's
``WITH_UNCOMPRESSED_SIZE`` mode matches).

The encoded bytes are uploaded to device **once**, outside the timing
loop, and wrapped as Zarr v3 GPU-prototype ``Buffer`` instances.  The
timed region runs only ``codec.decode``; output stays on device.  This
mirrors what an actual ``CudaZarrArray`` read looks like — cuFile lands
the encoded bytes on device, decode happens entirely on device, the
user-facing output is a ``cupy.ndarray`` ready for downstream kernels.

Outputs:
* Bit-exact comparison vs numcodecs decode (sanity check, off the hot loop)
* Per-rep wall times for native + nvcomp paths (decode-only, device-resident I/O)
* Median GiB/s for each path
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
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import gpu as gpu_buffer

import czarr

SHAPE_PER_CHUNK = (16, 512, 512)
N_CHUNKS = 64
DTYPE = np.float32


def make_fixture(*, n_chunks: int, shape: tuple[int, ...]) -> tuple[list[bytes], list[np.ndarray]]:
    """Build N chunks of random float32 data + their numcodecs.LZ4 encodings.

    Returns ``(encoded_chunks, raw_arrays)``.  Each encoded chunk has the
    4-byte LE size prefix prepended by ``numcodecs.LZ4`` — the same
    bitstream czarr's ``WITH_UNCOMPRESSED_SIZE`` mode reads.
    """
    rng = np.random.default_rng(0)
    encoded: list[bytes] = []
    raw_arrays: list[np.ndarray] = []
    codec = NumcodecsLZ4(acceleration=1)
    for _ in range(n_chunks):
        arr = rng.standard_normal(shape).astype(DTYPE)
        enc = bytes(codec.encode(arr.tobytes()))
        encoded.append(enc)
        raw_arrays.append(arr)
    return encoded, raw_arrays


def _make_array_spec(nbytes: int) -> ArraySpec:
    return ArraySpec(
        shape=(nbytes,),
        dtype=zarr.dtype.parse_dtype("uint8", zarr_format=3),
        fill_value=0,
        prototype=gpu_buffer.buffer_prototype,
        config=zarr.core.array.ArrayConfig.from_dict({}),
    )


def upload_to_device(encoded: list[bytes]) -> list[Any]:
    """Pre-upload encoded bytes as Zarr v3 GPU-prototype Buffers.

    Mirrors what cuFile + GPULocalStore would produce in a real read.
    """
    prototype = gpu_buffer.buffer_prototype
    return [prototype.buffer.from_bytes(enc) for enc in encoded]


def decode_once(codec: Any, items: list[tuple[Any, ArraySpec]]) -> list[Any]:
    """One decode call; outputs stay on device."""
    return asyncio.run(codec.decode(items))


def validate_bitstream_compat(items: list[tuple[Any, ArraySpec]], raw_arrays: list[np.ndarray]) -> None:
    """Bit-exact sanity check off the timing loop."""
    expected_bytes = [arr.tobytes() for arr in raw_arrays]
    for backend in ("native", "nvcomp"):
        codec = czarr.LZ4(backend=backend)
        decoded = decode_once(codec, items)
        for i, (buf, exp) in enumerate(zip(decoded, expected_bytes, strict=True)):
            actual = buf.to_bytes()  # OFF the hot path
            if actual != exp:
                bad_idx = next(
                    (j for j, (a, b) in enumerate(zip(actual, exp, strict=True)) if a != b),
                    -1,
                )
                raise AssertionError(
                    f"{backend} chunk {i} mismatch at byte {bad_idx}: "
                    f"got {actual[bad_idx : bad_idx + 8]!r}, expected {exp[bad_idx : bad_idx + 8]!r}"
                )
        print(f"  {backend}: {len(decoded)} chunks bit-exact vs numcodecs.LZ4 ✓")


def time_decode_only(
    codec: Any,
    items: list[tuple[Any, ArraySpec]],
    *,
    reps: int,
    warmup: int,
) -> list[float]:
    """Time pure codec.decode on device-resident inputs; outputs stay on device."""
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--warmup", type=int, default=3)
    args = p.parse_args()

    print("=== building fixture ===")
    encoded, raw_arrays = make_fixture(n_chunks=N_CHUNKS, shape=SHAPE_PER_CHUNK)
    nbytes_per_chunk = int(np.prod(SHAPE_PER_CHUNK) * DTYPE().itemsize)
    nbytes_total = nbytes_per_chunk * N_CHUNKS
    avg_comp_size = sum(len(e) for e in encoded) // N_CHUNKS
    ratio = avg_comp_size / nbytes_per_chunk * 100
    print(f"  {N_CHUNKS} chunks of {SHAPE_PER_CHUNK} float32 ≈ {nbytes_per_chunk / (1 << 20):.1f} MiB each")
    print(f"  total raw: {nbytes_total / (1 << 30):.2f} GiB")
    print(f"  avg compressed size: {avg_comp_size / (1 << 20):.1f} MiB ({ratio:.1f}%)")

    # Activate the GPU buffer prototype so outputs land on device.
    zarr.config.set({"buffer": "zarr.core.buffer.gpu.Buffer", "ndbuffer": "zarr.core.buffer.gpu.NDBuffer"})
    try:
        print("\n=== uploading inputs to device (once) ===")
        device_buffers = upload_to_device(encoded)
        items = [(buf, _make_array_spec(nbytes_per_chunk)) for buf in device_buffers]
        cp.cuda.Stream.null.synchronize()
        print(f"  {len(device_buffers)} buffers on device")

        print("\n=== bit-exact validation (off hot path) ===")
        validate_bitstream_compat(items, raw_arrays)

        print(f"\n=== timing decode-only (reps={args.reps}, warmup={args.warmup}) ===")
        native_samples = time_decode_only(czarr.LZ4(backend="native"), items, reps=args.reps, warmup=args.warmup)
        nvcomp_samples = time_decode_only(czarr.LZ4(backend="nvcomp"), items, reps=args.reps, warmup=args.warmup)
    finally:
        zarr.config.reset()

    def _stats(s: list[float]) -> tuple[float, float, float]:
        return statistics.median(s), min(s), max(s)

    n_med, n_min, n_max = _stats(native_samples)
    c_med, c_min, c_max = _stats(nvcomp_samples)

    print()
    print(f"{'backend':<10}{'median ms':>12}{'min ms':>10}{'max ms':>10}{'GiB/s':>10}")
    print("-" * 52)
    print(
        f"{'native':<10}{n_med * 1e3:>12.2f}{n_min * 1e3:>10.2f}{n_max * 1e3:>10.2f}{_gibs(nbytes_total, n_med):>10.2f}"
    )
    print(
        f"{'nvcomp':<10}{c_med * 1e3:>12.2f}{c_min * 1e3:>10.2f}{c_max * 1e3:>10.2f}{_gibs(nbytes_total, c_med):>10.2f}"
    )
    print()
    print(f"speedup (native vs nvcomp): {c_med / n_med:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
