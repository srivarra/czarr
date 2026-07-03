"""Baseline perf measurement for czarr codecs vs CPU equivalents.

Captures encode/decode throughput and compression ratio across all 8 nvCOMP
codecs (GPU) and matching CPU codecs from numcodecs (LZ4, Snappy, Zstd,
Deflate, Blosc-lz4, Blosc-zstd) on identical workloads. RMM allocation
statistics (peak bytes, total bytes, ncalls) are tracked for the GPU side.

Run:
    uv run --extra cu12 --group test python -m bench.baseline

Output: a sorted GPU table + a sorted CPU table per workload + RMM profile.
"""

import dataclasses
import statistics
import time
import warnings
from typing import Any

import cupy as cp
import numcodecs
import numpy as np
import rmm
import rmm.statistics as rs
from cuda.core import Device
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


# Workloads: (name, dtype, shape, generator) — shape sized so total bytes
# stays modest for a quick baseline (<50 MB each).
def _gen_random(rng, shape, dtype):
    if np.issubdtype(np.dtype(dtype), np.integer):
        info = np.iinfo(dtype)
        return rng.integers(info.min // 2, info.max // 2, size=shape, dtype=dtype)
    return rng.standard_normal(shape).astype(dtype)


def _gen_zeros(_rng, shape, dtype):
    return np.zeros(shape, dtype=dtype)


def _gen_structured(_rng, shape, dtype):
    """Repeating ramp 0..255 — somewhat compressible, more realistic than zeros."""
    n = int(np.prod(shape))
    base = (np.arange(n, dtype=np.int64) % 256).astype(dtype)
    return base.reshape(shape)


WORKLOADS: list[tuple[str, str, tuple[int, ...], Any]] = [
    ("random_u8_4MB", "uint8", (4 * 1024 * 1024,), _gen_random),
    ("random_f32_4MB", "float32", (1024 * 1024,), _gen_random),
    ("zeros_f32_4MB", "float32", (1024 * 1024,), _gen_zeros),
    ("structured_u8_4MB", "uint8", (4 * 1024 * 1024,), _gen_structured),
]


@dataclasses.dataclass
class Result:
    workload: str
    codec: str
    enc_ms: float
    dec_ms: float
    ratio: float
    enc_gbps: float
    dec_gbps: float


def _gpu_time_ms(fn, stream) -> float:
    """Time a GPU operation using cuda.core.Event for sub-µs accuracy."""
    start = stream.record(options={"enable_timing": True})
    fn()
    end = stream.record(options={"enable_timing": True})
    end.sync()
    return start - end  # cuda.core elapsed, returns ms


# CPU codecs from numcodecs to compare against. Names mirror the GPU codec
# names so the two tables read in parallel. Blosc variants run multi-threaded
# (16 threads on this host), giving a fair "best CPU" baseline against the
# single-stream GPU runs above.
def _make_cpu_codecs():
    return [
        ("LZ4-cpu", numcodecs.LZ4()),
        ("Zstd-cpu", numcodecs.Zstd(level=3)),
        ("Deflate-cpu", numcodecs.Zlib(level=6)),
        ("Blosc-lz4-cpu", numcodecs.Blosc(cname="lz4", clevel=5)),
        ("Blosc-zstd-cpu", numcodecs.Blosc(cname="zstd", clevel=3)),
    ]


def _measure_cpu(name: str, codec, data: np.ndarray, repeats: int = 5) -> Result | None:
    """Measure CPU codec round-trip on host bytes via wall-clock timing."""
    raw = data.tobytes()
    nbytes = len(raw)

    # Warm-up
    try:
        encoded = codec.encode(np.frombuffer(raw, dtype=np.uint8))
        codec.decode(encoded)
    except Exception as e:
        print(f"  ! {name} failed warm-up: {type(e).__name__}: {e}")
        return None

    src = np.frombuffer(raw, dtype=np.uint8)

    enc_times = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        encoded = codec.encode(src)
        t1 = time.perf_counter_ns()
        enc_times.append((t1 - t0) / 1e6)  # ms

    dec_times = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        codec.decode(encoded)
        t1 = time.perf_counter_ns()
        dec_times.append((t1 - t0) / 1e6)

    enc_ms = statistics.median(enc_times)
    dec_ms = statistics.median(dec_times)
    encoded_size = len(encoded) if isinstance(encoded, (bytes, bytearray)) else encoded.nbytes
    return Result(
        workload="",
        codec=name,
        enc_ms=enc_ms,
        dec_ms=dec_ms,
        ratio=nbytes / encoded_size,
        enc_gbps=nbytes / (enc_ms / 1000) / 1e9,
        dec_gbps=nbytes / (dec_ms / 1000) / 1e9,
    )


def _measure_codec(codec_cls, data: np.ndarray, repeats: int = 5) -> Result:
    codec = codec_cls()
    dev_in = cp.asarray(data.view(np.uint8).ravel())
    nbytes = int(dev_in.nbytes)

    # Warm-up: codec creation, scratch alloc, JIT compile of any cupy ops
    # Use the actual codec methods on raw nvcomp arrays for a tight loop
    from nvidia import nvcomp

    nvcodec = codec._get_codec()
    nv_arr = nvcomp.as_array(dev_in)

    # Warm-up
    compressed = nvcodec.encode(nv_arr)
    _ = nvcodec.decode(compressed)
    cp.cuda.Stream.null.synchronize()

    # Encode timing
    enc_times = []
    for _ in range(repeats):
        start = cp.cuda.Event()
        stop = cp.cuda.Event()
        start.record()
        compressed = nvcodec.encode(nv_arr)
        stop.record()
        stop.synchronize()
        enc_times.append(cp.cuda.get_elapsed_time(start, stop))

    # Decode timing
    dec_times = []
    for _ in range(repeats):
        start = cp.cuda.Event()
        stop = cp.cuda.Event()
        start.record()
        decoded = nvcodec.decode(compressed)
        stop.record()
        stop.synchronize()
        dec_times.append(cp.cuda.get_elapsed_time(start, stop))

    enc_ms = statistics.median(enc_times)
    dec_ms = statistics.median(dec_times)
    ratio = nbytes / compressed.buffer_size

    return Result(
        workload="",
        codec=codec_cls.__name__,
        enc_ms=enc_ms,
        dec_ms=dec_ms,
        ratio=ratio,
        enc_gbps=nbytes / (enc_ms / 1000) / 1e9,
        dec_gbps=nbytes / (dec_ms / 1000) / 1e9,
    )


def _print_workload(workload: str, results: list[Result]) -> None:
    print(f"\n## {workload}")
    print(f"{'codec':<10s} {'enc ms':>8s} {'dec ms':>8s} {'enc GB/s':>10s} {'dec GB/s':>10s} {'ratio':>8s}")
    print("-" * 60)
    for r in sorted(results, key=lambda r: -r.dec_gbps):
        print(
            f"{r.codec:<10s} {r.enc_ms:>8.2f} {r.dec_ms:>8.2f} {r.enc_gbps:>10.2f} {r.dec_gbps:>10.2f} {r.ratio:>8.2f}"
        )


def main() -> None:
    # Setup: RMM pool + statistics, route everything through one allocator
    rmm.reinitialize(pool_allocator=True, initial_pool_size=512 * 1024 * 1024)
    cp.cuda.set_allocator(rmm_cupy_allocator)
    czarr.register_nvcomp_allocator()
    rs.enable_statistics()

    dev = Device()
    dev.set_current()

    rng = np.random.default_rng(0)
    print(f"# czarr baseline — device {dev.name if hasattr(dev, 'name') else dev.device_id}")
    print("# RMM pool: 512 MiB initial, statistics tracking enabled")
    print("# Repeats per measurement: 5 (median reported)")

    cpu_codecs = _make_cpu_codecs()

    for workload_name, dtype, shape, gen in WORKLOADS:
        data = gen(rng, shape, dtype)

        # GPU
        gpu_results: list[Result] = []
        for codec_cls in ALL_CODECS:
            with rs.profiler(name=f"{workload_name}::{codec_cls.__name__}"):
                try:
                    r = _measure_codec(codec_cls, data)
                    r.workload = workload_name
                    gpu_results.append(r)
                except Exception as e:
                    print(f"  ! {codec_cls.__name__} failed: {type(e).__name__}: {e}")
        _print_workload(f"{workload_name} — GPU (nvCOMP)", gpu_results)

        # CPU
        cpu_results: list[Result] = []
        for name, codec in cpu_codecs:
            r = _measure_cpu(name, codec, data)
            if r is not None:
                r.workload = workload_name
                cpu_results.append(r)
        _print_workload(f"{workload_name} — CPU (numcodecs)", cpu_results)

    print("\n## RMM memory profile (GPU codecs only)")
    print(rs.default_profiler_records.report(ordered_by="memory_peak"))


if __name__ == "__main__":
    main()
