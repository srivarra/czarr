"""Experiment: chunk-size sweep through full Zarr pipeline.

Full Zarr ``arr[:] = data`` and ``arr[:]`` measurements as we vary chunk size,
on a fixed 256 MiB float32 array. Three configs:

  * CPU Blosc-lz4 + host prototype
  * GPU ANS    + host prototype  (round-trips host)
  * GPU ANS    + GPU prototype   (fully on-device)

Confirms whether the `chunk size ≥ 1 MiB → GPU wins` finding from the
codec-only sweep also holds end-to-end through the Zarr pipeline.

Run:
    uv run --extra cu12 --group test python -m bench.zarr.exp_zarr_chunksize
"""

from __future__ import annotations

import statistics
import time
import warnings
from contextlib import contextmanager

import cupy as cp
import numpy as np
import rmm
import zarr
from rmm.allocators.cupy import rmm_cupy_allocator
from zarr.codecs import BloscCodec
from zarr.core.buffer import gpu as gpu_buffer
from zarr.storage import MemoryStore

import czarr
from czarr.codecs import ANS

warnings.filterwarnings("ignore", category=DeprecationWarning)

TOTAL_SHAPE = (8192, 8192)  # 256 MiB float32
DTYPE = np.float32
REPEATS = 3


@contextmanager
def _no_op_ctx():
    yield


def _gpu_config():
    return zarr.config.set({"buffer": "zarr.core.buffer.gpu.Buffer", "ndbuffer": "zarr.core.buffer.gpu.NDBuffer"})


def _measure(data, codec, chunk_shape, gpu_proto: bool):
    ctx = _gpu_config() if gpu_proto else _no_op_ctx()
    payload = cp.asarray(data) if gpu_proto else data

    with ctx:
        store = MemoryStore()
        arr = zarr.create_array(
            store=store,
            shape=data.shape,
            chunks=chunk_shape,
            dtype=str(data.dtype),
            compressors=[codec],
        )
        arr[:] = payload
        if gpu_proto:
            _ = arr.get_basic_selection(prototype=gpu_buffer.buffer_prototype)
        else:
            _ = arr[:]

        write_times, read_times = [], []
        for _ in range(REPEATS):
            store2 = MemoryStore()
            arr2 = zarr.create_array(
                store=store2,
                shape=data.shape,
                chunks=chunk_shape,
                dtype=str(data.dtype),
                compressors=[codec],
            )
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter_ns()
            arr2[:] = payload
            cp.cuda.Stream.null.synchronize()
            t1 = time.perf_counter_ns()
            write_times.append((t1 - t0) / 1e6)

        for _ in range(REPEATS):
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter_ns()
            if gpu_proto:
                _ = arr.get_basic_selection(prototype=gpu_buffer.buffer_prototype)
            else:
                _ = arr[:]
            cp.cuda.Stream.null.synchronize()
            t1 = time.perf_counter_ns()
            read_times.append((t1 - t0) / 1e6)

    return statistics.median(write_times), statistics.median(read_times)


def main() -> None:
    rmm.reinitialize(pool_allocator=True, initial_pool_size=4 << 30)
    cp.cuda.set_allocator(rmm_cupy_allocator)
    czarr.register_nvcomp_allocator()

    nbytes = int(np.prod(TOTAL_SHAPE) * np.dtype(DTYPE).itemsize)
    print("# Experiment: chunk-size sweep through Zarr pipeline")
    print(f"# Array {TOTAL_SHAPE} {DTYPE.__name__} = {nbytes // 1024 // 1024} MiB total")
    print(f"# Repeats: {REPEATS} (median)")
    print()

    rng = np.random.default_rng(0)
    data = rng.integers(0, 64, np.prod(TOTAL_SHAPE), dtype=np.int32).astype(np.float32).reshape(TOTAL_SHAPE)

    # Chunk size as (rows, 8192_cols).  rows × 8192 × 4 bytes = chunk bytes.
    chunk_configs = [
        ("256 KiB", (8, 8192)),
        ("1 MiB", (32, 8192)),
        ("4 MiB", (128, 8192)),
        ("16 MiB", (512, 8192)),
        ("32 MiB", (1024, 8192)),
        ("64 MiB", (2048, 8192)),
    ]

    blosc = BloscCodec(cname="lz4", clevel=5)
    ans = ANS()

    print(
        f"{'chunk':<8s} | {'CPU r GB/s':>10s} {'GPU+host r':>10s} {'GPU+gpu r':>10s} | "
        f"{'CPU w GB/s':>10s} {'GPU+host w':>10s} {'GPU+gpu w':>10s}"
    )
    print("-" * 90)

    for label, chunk_shape in chunk_configs:
        # CPU baseline
        try:
            cw, cr = _measure(data, blosc, chunk_shape, gpu_proto=False)
        except Exception:
            cw = cr = float("nan")

        # GPU codec, host prototype (slow path: H2D each chunk)
        try:
            gw_h, gr_h = _measure(data, ans, chunk_shape, gpu_proto=False)
        except Exception:
            gw_h = gr_h = float("nan")

        # GPU codec, GPU prototype (fast path: stays on device)
        try:
            gw_g, gr_g = _measure(data, ans, chunk_shape, gpu_proto=True)
        except Exception:
            gw_g = gr_g = float("nan")

        def gbps(ms):
            return nbytes / (ms / 1000) / 1e9 if ms == ms else float("nan")

        print(
            f"{label:<8s} | "
            f"{gbps(cr):>10.2f} {gbps(gr_h):>10.2f} {gbps(gr_g):>10.2f} | "
            f"{gbps(cw):>10.2f} {gbps(gw_h):>10.2f} {gbps(gw_g):>10.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
