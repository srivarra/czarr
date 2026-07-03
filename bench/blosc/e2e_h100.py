"""End-to-end GPU blosc decode through GPULocalStore (cuFile/GDS) vs CPU.

Validates the full read path the `Blosc` codec is meant for: cuFile reads
compressed blosc chunks straight to the GPU, czarr decodes them on-device.
Compares to the CPU baseline (numcodecs blosc decode + H2D) and checks
bit-exactness. Needs real GDS (H100 + nvidia-fs); A40 cuFile is compat-broken.

Run: module load cuda && uv run --extra cu12 python -m bench.blosc.e2e_h100
"""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path

import cupy as cp
import numpy as np
import zarr
from zarr.codecs import BloscCodec, BloscShuffle
from zarr.storage import LocalStore

import czarr

SHAPE = (64, 2048, 2048)  # 512 MiB f16
CHUNKS = (1, 2048, 2048)  # 8 MiB/chunk -> 256 blocks/chunk @ 32 KiB
DTYPE = "float16"
NBYTES = int(np.prod(SHAPE)) * 2
REPS, WARMUP = 5, 2


def _to_np(x):
    return cp.asnumpy(x) if isinstance(x, cp.ndarray) else np.asarray(x)


def write_store(path: Path) -> None:
    arr = zarr.create_array(
        store=LocalStore(path),
        shape=SHAPE,
        chunks=CHUNKS,
        dtype=DTYPE,
        compressors=[BloscCodec(cname="zstd", clevel=1, shuffle=BloscShuffle.bitshuffle, typesize=2, blocksize=32768)],
        overwrite=True,
    )
    rng = np.random.default_rng(0)
    base = np.linspace(0, 1, int(np.prod(SHAPE)), dtype="float32").reshape(SHAPE)
    arr[:] = (base + rng.standard_normal(SHAPE).astype("float32") * 0.01).astype("float16")


def _gibs(s: float) -> float:
    return (NBYTES / 2**30) / s


def main() -> int:
    base = os.environ.get("TMPDIR", "/tmp")
    path = Path(base) / "blosc_e2e.zarr"
    print(f"writing {NBYTES / 2**20:.0f} MiB f16 blosc store at {path} ...")
    write_store(path)

    # ---- CPU baseline (before configure_gpu): numcodecs decode + H2D ----
    def cpu_read():
        host = np.asarray(zarr.open_array(store=LocalStore(path), mode="r")[:])
        cp.cuda.runtime.deviceSynchronize()
        t = time.perf_counter()
        g = cp.asarray(host)
        cp.cuda.runtime.deviceSynchronize()
        return host, (time.perf_counter() - t)

    ref, _ = cpu_read()
    cpu = []
    for _ in range(WARMUP + REPS):
        t0 = time.perf_counter()
        _h, _ = cpu_read()
        cpu.append(time.perf_counter() - t0)
    cpu_ms = statistics.median(cpu[WARMUP:]) * 1e3

    # ---- czarr GPU-direct: GPULocalStore (cuFile) + GPU blosc decode ----
    czarr.configure_gpu()
    store = czarr.GPULocalStore(path, read_only=True)
    print(f"GDS available: {getattr(store, 'gds_available', '?')}")

    def gpu_read():
        out = zarr.open_array(store=store, mode="r")[:]
        cp.cuda.runtime.deviceSynchronize()
        return out

    out = gpu_read()
    assert isinstance(out, cp.ndarray), "expected cupy output"
    exact = bool(np.array_equal(_to_np(out), ref))
    print(f"bit-exact vs CPU: {exact}")
    if not exact:
        print("CORRECTNESS FAIL")
        return 1

    for _ in range(WARMUP):
        gpu_read()
    gpu = []
    for _ in range(REPS):
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        gpu_read()
        cp.cuda.runtime.deviceSynchronize()
        gpu.append(time.perf_counter() - t0)
    gpu_ms = statistics.median(gpu) * 1e3

    print(f"\n{'=' * 56}\ne2e blosc read ({NBYTES / 2**20:.0f} MiB, {REPS} reps):")
    print(f"  czarr GPU-direct (GPULocalStore + GPU decode) : {gpu_ms:7.1f} ms  ({_gibs(gpu_ms / 1e3):.1f} GiB/s)")
    print(f"  CPU (numcodecs blosc + H2D)                    : {cpu_ms:7.1f} ms  ({_gibs(cpu_ms / 1e3):.1f} GiB/s)")
    print(f"  speedup: {cpu_ms / gpu_ms:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
