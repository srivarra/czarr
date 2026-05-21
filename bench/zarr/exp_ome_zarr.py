"""Experiment: codec performance on real OME-Zarr microscopy data.

Random and "structured" payloads from earlier experiments don't capture how
codecs behave on real scientific data. This loads a real OME-Zarr dataset
via iohub and runs the codec sweep against it for comparison.

Real microscopy data is highly compressible (correlated neighbors, lots of
background near zero, integer-quantized intensities) — Zstd and Deflate
typically pull ahead here.

Run:
    uv run --extra cu12 --group test python -m bench.zarr.exp_ome_zarr
"""

from __future__ import annotations

import statistics
import warnings

import cupy as cp
import numpy as np
import rmm
from iohub import open_ome_zarr
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

DATASET_PATH = "/hpc/websites/public.czbiohub.org/comp.micro/nd-embedding-atlas-test-data/dataset.zarr"
N_TIMEPOINTS = 32  # 32 timepoints * 9 MiB ≈ 288 MiB total
REPEATS = 3

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


def _load_real_data():
    print(f"# Loading {N_TIMEPOINTS} timepoints from {DATASET_PATH} ...")
    ds = open_ome_zarr(DATASET_PATH, mode="r")
    name, pos = next(iter(ds.positions()))
    arr = pos.data
    data = np.asarray(arr[:N_TIMEPOINTS])  # (T, C, Z, Y, X)
    print(
        f"# real data: {name} shape={data.shape} dtype={data.dtype} "
        f"size={data.nbytes // 1024 // 1024} MiB "
        f"min={data.min():.2f} max={data.max():.2f} mean={data.mean():.1f}"
    )
    return data


def _measure_codec(codec_cls, data: np.ndarray) -> dict:
    """Single-shot encode/decode of the whole array as one big buffer."""
    flat = data.tobytes()
    src = cp.asarray(np.frombuffer(flat, dtype=np.uint8))
    nbytes = src.nbytes

    codec = nvcomp.Codec(algorithm=codec_cls.algorithm.value)
    nv_in = nvcomp.as_array(src)

    # Warm up
    compressed = codec.encode(nv_in)
    _ = codec.decode(compressed)
    cp.cuda.Stream.null.synchronize()

    enc_times, dec_times = [], []
    for _ in range(REPEATS):
        start = cp.cuda.Event()
        stop = cp.cuda.Event()
        start.record()
        compressed = codec.encode(nv_in)
        stop.record()
        stop.synchronize()
        enc_times.append(cp.cuda.get_elapsed_time(start, stop))

    for _ in range(REPEATS):
        start = cp.cuda.Event()
        stop = cp.cuda.Event()
        start.record()
        codec.decode(compressed)
        stop.record()
        stop.synchronize()
        dec_times.append(cp.cuda.get_elapsed_time(start, stop))

    enc_ms = statistics.median(enc_times)
    dec_ms = statistics.median(dec_times)
    ratio = nbytes / compressed.buffer_size
    return {
        "codec": codec_cls.__name__,
        "enc_gbps": nbytes / (enc_ms / 1000) / 1e9,
        "dec_gbps": nbytes / (dec_ms / 1000) / 1e9,
        "ratio": ratio,
    }


def main() -> None:
    rmm.reinitialize(pool_allocator=True, initial_pool_size=2 << 30)
    cp.cuda.set_allocator(rmm_cupy_allocator)
    czarr.register_nvcomp_allocator()

    print("# Experiment: codec performance on real OME-Zarr microscopy data")

    data = _load_real_data()

    print()
    print("## Real OME-Zarr microscopy data")
    print(f"{'codec':<10s} | {'enc GB/s':>10s} {'dec GB/s':>10s} | {'ratio':>8s}")
    print("-" * 50)
    real_results = []
    for codec_cls in ALL_CODECS:
        try:
            r = _measure_codec(codec_cls, data)
            real_results.append(r)
        except Exception as e:
            print(f"  ! {codec_cls.__name__} failed: {type(e).__name__}: {e}")
    for r in sorted(real_results, key=lambda x: -x["ratio"]):
        print(f"{r['codec']:<10s} | {r['enc_gbps']:>10.2f} {r['dec_gbps']:>10.2f} | {r['ratio']:>8.2f}")

    print()
    print("## Random uint8 (same total bytes — for reference)")
    rng = np.random.default_rng(0)
    random_data = rng.integers(0, 255, data.nbytes, dtype=np.uint8)
    print(f"{'codec':<10s} | {'enc GB/s':>10s} {'dec GB/s':>10s} | {'ratio':>8s}")
    print("-" * 50)
    rand_results = []
    for codec_cls in ALL_CODECS:
        try:
            r = _measure_codec(codec_cls, random_data)
            rand_results.append(r)
        except Exception as e:
            print(f"  ! {codec_cls.__name__} failed: {type(e).__name__}: {e}")
    for r in sorted(rand_results, key=lambda x: -x["ratio"]):
        print(f"{r['codec']:<10s} | {r['enc_gbps']:>10.2f} {r['dec_gbps']:>10.2f} | {r['ratio']:>8.2f}")

    # Side-by-side
    print()
    print("## Side-by-side ratio: real / random")
    print(f"{'codec':<10s} | {'real ratio':>10s} {'rand ratio':>10s} | {'real/rand':>10s}")
    print("-" * 50)
    rand_by_name = {r["codec"]: r for r in rand_results}
    for r in sorted(real_results, key=lambda x: -x["ratio"]):
        rand_r = rand_by_name.get(r["codec"], {"ratio": 1.0})
        print(
            f"{r['codec']:<10s} | {r['ratio']:>10.2f} {rand_r['ratio']:>10.2f} | {r['ratio'] / rand_r['ratio']:>9.2f}x"
        )


if __name__ == "__main__":
    main()
