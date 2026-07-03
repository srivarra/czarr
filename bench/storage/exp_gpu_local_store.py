"""Experiment: GPULocalStore (cuFile) under disk-load.

Measures the realistic write-to-disk + read-from-disk path. Three configs:

  1. zarr.LocalStore + Blosc-lz4 (CPU codec, host buffer)            — baseline
  2. zarr.LocalStore + GPU codec (host prototype, then upload)       — GPU compute, host I/O
  3. czarr.GPULocalStore + GPU codec (GPU prototype, cuFile)         — full on-device path

On Bruno's Lustre (cuFile compat mode), config 3 won't see real GDS DMA but
will still benefit from libcufile's pinned-host bounce + thread pool over
naive read+upload. The flag we want to confirm: GPULocalStore matches or
beats LocalStore for the GPU-codec path.

Run:
    LD_LIBRARY_PATH=...:$LD_LIBRARY_PATH \\
    CUFILE_FORCE_COMPAT_MODE=true \\
    uv run --extra cu12 --group test python -m bench.storage.exp_gpu_local_store
"""

import shutil
import statistics
import tempfile
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import cupy as cp
import numpy as np
import rmm
import zarr
from rmm.allocators.cupy import rmm_cupy_allocator
from zarr.codecs import BloscCodec
from zarr.core.buffer import gpu as gpu_buffer
from zarr.storage import LocalStore

import czarr
from czarr import GPULocalStore
from czarr.codecs import ANS

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Lustre tmpdir (cuFile compat mode rejects /tmp tmpfs)
_TMP_PARENT = "/hpc/mydata/sricharan.varra/Dev/czarr"

DTYPE = np.float32
CHUNK_SHAPE = (1024, 8192)  # 32 MiB per chunk — in GPU sweet spot
TOTAL_SHAPE = (8192, 8192)  # 256 MiB total
REPEATS = 3


@contextmanager
def _gpu_config():
    with zarr.config.set({"buffer": "zarr.core.buffer.gpu.Buffer", "ndbuffer": "zarr.core.buffer.gpu.NDBuffer"}):
        yield


@contextmanager
def _no_op():
    yield


@contextmanager
def _fresh_dir():
    d = tempfile.mkdtemp(dir=_TMP_PARENT, prefix=".bench_")
    try:
        yield Path(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _measure(make_store, payload, codec, gpu_proto: bool):
    """Single measurement: build a fresh store, write, then re-open and read."""
    ctx = _gpu_config() if gpu_proto else _no_op()
    with ctx, _fresh_dir() as path:
        # WRITE
        store_w = make_store(path, read_only=False)
        arr = zarr.create_array(
            store=store_w,
            shape=payload.shape,
            chunks=CHUNK_SHAPE,
            dtype="float32",
            compressors=[codec],
        )
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter_ns()
        arr[:] = payload
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter_ns()
        write_ms = (t1 - t0) / 1e6

        # READ — fresh store, force on-disk path (no in-memory cache)
        store_r = make_store(path, read_only=True)
        arr2 = zarr.open_array(store=store_r, mode="r")
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter_ns()
        if gpu_proto:
            _ = arr2.get_basic_selection(prototype=gpu_buffer.buffer_prototype)
        else:
            _ = arr2[:]
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter_ns()
        read_ms = (t1 - t0) / 1e6

    return write_ms, read_ms


def _bench(label: str, make_store, payload, codec, gpu_proto: bool, nbytes: int):
    # Warm-up (filesystem caches)
    _measure(make_store, payload, codec, gpu_proto)
    write_times, read_times = [], []
    for _ in range(REPEATS):
        w, r = _measure(make_store, payload, codec, gpu_proto)
        write_times.append(w)
        read_times.append(r)
    w = statistics.median(write_times)
    r = statistics.median(read_times)
    print(
        f"{label:<40s} | {w:>10.1f} {r:>10.1f} | {nbytes / (w / 1000) / 1e9:>11.2f} {nbytes / (r / 1000) / 1e9:>10.2f}",
        flush=True,
    )


def main() -> None:
    rmm.reinitialize(pool_allocator=True, initial_pool_size=2 << 30)
    cp.cuda.set_allocator(rmm_cupy_allocator)
    czarr.register_nvcomp_allocator()

    nbytes = int(np.prod(TOTAL_SHAPE) * np.dtype(DTYPE).itemsize)
    print("# Experiment: GPULocalStore (cuFile) under disk-load")
    print(
        f"# Array {TOTAL_SHAPE} {DTYPE.__name__} = {nbytes // 1024 // 1024} MiB total, "
        f"chunks {CHUNK_SHAPE} = {int(np.prod(CHUNK_SHAPE) * np.dtype(DTYPE).itemsize) // 1024 // 1024} MiB"
    )
    print(f"# Tmp on Lustre at {_TMP_PARENT}")
    print(f"# Repeats: {REPEATS} (median)")

    # Verify cuFile is available
    from czarr import cufile as cufile_runtime

    print(f"# cuFile available: {cufile_runtime.is_available()}")
    print()

    rng = np.random.default_rng(0)
    data = rng.integers(0, 64, np.prod(TOTAL_SHAPE), dtype=np.int32).astype(np.float32).reshape(TOTAL_SHAPE)
    data_dev = cp.asarray(data)

    blosc = BloscCodec(cname="lz4", clevel=5)
    ans = ANS()

    print(f"{'config':<40s} | {'write ms':>10s} {'read ms':>10s} | {'write GB/s':>11s} {'read GB/s':>10s}")
    print("-" * 95)

    # Config 1: vanilla LocalStore + CPU codec + host array
    _bench(
        "LocalStore   + Blosc-lz4 (CPU)        ",
        lambda p, read_only: LocalStore(p, read_only=read_only),
        data,
        blosc,
        gpu_proto=False,
        nbytes=nbytes,
    )

    # Config 2: vanilla LocalStore + GPU codec + host buffer
    _bench(
        "LocalStore   + ANS  (host proto, GPU) ",
        lambda p, read_only: LocalStore(p, read_only=read_only),
        data,
        ans,
        gpu_proto=False,
        nbytes=nbytes,
    )

    # Config 3: GPULocalStore + GPU codec + host buffer
    _bench(
        "GPULocalStore + ANS (host proto, GPU) ",
        lambda p, read_only: GPULocalStore(p, read_only=read_only),
        data,
        ans,
        gpu_proto=False,
        nbytes=nbytes,
    )

    # Config 4: vanilla LocalStore + GPU codec + GPU buffer
    _bench(
        "LocalStore   + ANS  (GPU proto)       ",
        lambda p, read_only: LocalStore(p, read_only=read_only),
        data_dev,
        ans,
        gpu_proto=True,
        nbytes=nbytes,
    )

    # Config 5: GPULocalStore + GPU codec + GPU buffer (the cuFile path)
    _bench(
        "GPULocalStore + ANS (GPU proto)       ",
        lambda p, read_only: GPULocalStore(p, read_only=read_only),
        data_dev,
        ans,
        gpu_proto=True,
        nbytes=nbytes,
    )


if __name__ == "__main__":
    main()
