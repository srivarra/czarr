"""Compare cuFile read strategies: threaded sync vs async-stream vs batch_io_submit.

Four contenders for "read N independent chunks from disk into GPU memory":

1. ``threaded_sync``      — N OS threads, each calls ``cufile.read`` (the
                            shape zarr's pipeline drives via ``concurrent_map``).
2. ``async_1stream``      — N ``read_async`` calls on ONE CUDA stream.  We
                            measured this serialises reads (the killer in the
                            scrapped ``stream_crops_async``).
3. ``async_Nstream``      — round-robin N reads across K=4 CUDA streams,
                            breaking single-stream FIFO ordering.
4. ``batch_io_submit``    — one batch of N IOCBs to libcufile, which schedules
                            them across its internal worker pool.  Operations
                            within a batch can complete out of order = real
                            parallelism without our thread coordination cost.

Decision criterion: if (4) beats (1) by >=30% on the realistic chunk shape we
care about, plumb ``GPULocalStore._get_many`` on top of ``batch_io_submit``.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cupy as cp
import numpy as np
from cuda.bindings import cufile

from czarr.storage import cufile_runtime
from czarr.storage.cufile_runtime import _CUfileDescr

N_FILES = 32
CHUNK_BYTES = 8 * 1024 * 1024  # 8 MiB per file, mirrors a realistic Zarr chunk
WARMUP = 2
RUNS = 5

# Bench on NFS — the filesystem czarr workloads actually live on.
BASE = Path("/hpc/mydata/sricharan.varra/czarr_batch_io_probe")


def _open_and_register(path: Path) -> tuple[int, int, cufile.Descr]:
    """Open file + register handle with cuFile; return (fd, fh, descr_keepalive)."""
    fd = os.open(str(path), os.O_RDONLY)
    descr = cufile.Descr()
    s = _CUfileDescr.from_address(int(descr))
    s.type = int(cufile.FileHandleType.OPAQUE_FD)
    s.handle.fd = fd
    s.fs_ops = 0
    fh = cufile.handle_register(int(descr))
    return fd, fh, descr


def _close_and_dereg(fd: int, fh: int) -> None:
    try:
        cufile.handle_deregister(fh)
    except Exception:
        pass
    try:
        os.close(fd)
    except Exception:
        pass


def _prepare_files() -> list[Path]:
    BASE.mkdir(parents=True, exist_ok=True)
    paths = [BASE / f"chunk_{i:04d}.bin" for i in range(N_FILES)]
    payload = np.random.default_rng(0).integers(0, 255, CHUNK_BYTES, dtype=np.uint8).tobytes()
    for p in paths:
        if not p.exists() or p.stat().st_size != CHUNK_BYTES:
            p.write_bytes(payload)
    return paths


def _bench_threaded_sync(paths: list[Path], arena: cp.ndarray, offsets: list[int]) -> float:
    """N OS threads each issue a sync ``cufile.read``."""

    def _one(idx: int) -> None:
        fd, fh, _descr = _open_and_register(paths[idx])
        try:
            cufile.read(fh, int(arena.data.ptr), CHUNK_BYTES, 0, offsets[idx])
        finally:
            _close_and_dereg(fd, fh)

    arena[:] = 0
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=N_FILES) as ex:
        list(ex.map(_one, range(N_FILES)))
    return time.perf_counter() - t0


def _bench_async_streams(
    handles: list[tuple[int, int]],
    arena: cp.ndarray,
    offsets: list[int],
    n_streams: int,
) -> float:
    """Round-robin N ``read_async`` calls across ``n_streams`` CUDA streams."""
    arena[:] = 0
    cp.cuda.runtime.deviceSynchronize()
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(n_streams)]
    for s in streams:
        cufile_runtime.ensure_stream_registered(s.ptr)

    args_holder = []
    t0 = time.perf_counter()
    for i, (_fd, fh) in enumerate(handles):
        args = cufile_runtime.make_io_args()
        cufile_runtime.read_async(
            fh,
            int(arena.data.ptr),
            CHUNK_BYTES,
            streams[i % n_streams].ptr,
            args=args,
            file_offset=0,
            dev_offset=offsets[i],
        )
        args_holder.append(args)
    for s in streams:
        s.synchronize()
    elapsed = time.perf_counter() - t0
    del args_holder
    return elapsed


def _bench_batch_io(
    handles: list[tuple[int, int]],
    arena: cp.ndarray,
    offsets: list[int],
) -> float:
    """One ``batch_io_submit`` covering all N IOCBs."""
    arena[:] = 0
    cp.cuda.runtime.deviceSynchronize()
    arena_ptr = int(arena.data.ptr)

    iocbs = np.zeros(N_FILES, dtype=cufile.io_params_dtype)
    for i, (_fd, fh) in enumerate(handles):
        iocbs[i]["mode"] = int(cufile.BatchMode.BATCH)
        iocbs[i]["u"]["batch"]["dev_ptr_base"] = arena_ptr
        iocbs[i]["u"]["batch"]["file_offset"] = 0
        iocbs[i]["u"]["batch"]["dev_ptr_offset"] = offsets[i]
        iocbs[i]["u"]["batch"]["size_"] = CHUNK_BYTES
        iocbs[i]["fh"] = fh
        iocbs[i]["opcode"] = int(cufile.Opcode.READ)
        iocbs[i]["cookie"] = i

    t0 = time.perf_counter()
    batch_id_holder = (ctypes.c_void_p * 1)()
    cufile.batch_io_set_up(int(N_FILES))  # returns batch_id as intptr_t
    # Wait — batch_io_set_up returns the batch id directly per the C API.
    # Re-do via the binding's actual signature.
    batch_id = cufile.batch_io_set_up(int(N_FILES))
    try:
        cufile.batch_io_submit(int(batch_id), int(N_FILES), iocbs.ctypes.data, 0)

        # Poll until all complete.
        events = np.zeros(N_FILES, dtype=cufile.io_events_dtype)
        nr_arr = (ctypes.c_uint * 1)(N_FILES)
        completed = 0
        while completed < N_FILES:
            nr_arr[0] = N_FILES - completed
            cufile.batch_io_get_status(
                int(batch_id),
                0,
                ctypes.addressof(nr_arr),
                events[completed:].ctypes.data,
                0,
            )
            completed += int(nr_arr[0])
    finally:
        try:
            cufile.batch_io_destroy(int(batch_id))
        except Exception:
            pass
    return time.perf_counter() - t0


def _measure(label: str, fn) -> tuple[float, float]:
    for _ in range(WARMUP):
        fn()
    times = [fn() for _ in range(RUNS)]
    arr = np.array(times)
    return float(arr.mean()), float(arr.std())


def main() -> int:
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"nvidia_fs loaded: {os.path.exists('/proc/driver/nvidia-fs')}")
    print(f"N_FILES={N_FILES}  CHUNK_BYTES={CHUNK_BYTES}  total={N_FILES * CHUNK_BYTES / 1e6:.1f} MB")
    print(f"backing FS: {BASE} (NFS / vast)")
    print()

    cufile_runtime.ensure_driver_open()
    paths = _prepare_files()

    # Pre-allocate one big arena.  Buf-register once for the async + batch paths.
    arena = cp.empty(N_FILES * CHUNK_BYTES, dtype=cp.uint8)
    arena_ptr = int(arena.data.ptr)
    cufile.buf_register(arena_ptr, N_FILES * CHUNK_BYTES, 0)
    offsets = [i * CHUNK_BYTES for i in range(N_FILES)]

    # Pre-open + pre-register handles for the no-overhead variants.
    handles = [(fd, fh) for fd, fh, _descr in (_open_and_register(p) for p in paths)]
    # Keep descr alive — they're stored alongside fd/fh in a tuple via closure.

    total_bytes = N_FILES * CHUNK_BYTES

    print(f"{'method':<24} {'mean ms':>10} {'std ms':>9} {'GB/s':>9}")
    print("-" * 60)
    try:
        for label, fn in [
            ("threaded_sync (N=32)", lambda: _bench_threaded_sync(paths, arena, offsets)),
            ("async_1stream", lambda: _bench_async_streams(handles, arena, offsets, 1)),
            ("async_4streams", lambda: _bench_async_streams(handles, arena, offsets, 4)),
            ("async_8streams", lambda: _bench_async_streams(handles, arena, offsets, 8)),
            ("batch_io_submit", lambda: _bench_batch_io(handles, arena, offsets)),
        ]:
            try:
                mean, std = _measure(label, fn)
                gbps = total_bytes / mean / 1e9
                print(f"{label:<24} {mean * 1000:>10.2f} {std * 1000:>9.2f} {gbps:>9.2f}")
            except Exception as e:
                print(f"{label:<24} FAILED: {type(e).__name__}: {str(e)[:80]}")
    finally:
        # Teardown
        for fd, fh in handles:
            _close_and_dereg(fd, fh)
        try:
            cufile.buf_deregister(arena_ptr)
        except Exception:
            pass
        if BASE.exists():
            shutil.rmtree(BASE, ignore_errors=True)

    print()
    print("Decision rule: ship batch_io in GPULocalStore._get_many if it beats")
    print("threaded_sync by >=30% on this config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
