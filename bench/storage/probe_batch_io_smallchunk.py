"""Adapt probe_batch_io.py for the SMALL-chunk regime (N=2048 x 32KB).

The earlier batch_io probe used N=32 x 8MB and saw batch_io_submit lose
6.5x vs threaded_sync.  Phase 0 of the cross-chunk-batching epic showed
the real bottleneck is at the opposite end (many tiny chunks).  Test
whether batch_io_submit wins in *that* regime before committing to a
Phase 3 implementation.
"""

from __future__ import annotations

import ctypes
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cupy as cp
import numpy as np
from cuda.bindings import cufile

from czarr import cufile as cufile_runtime
from czarr.cufile import _CUfileDescr

N_FILES = 2048
CHUNK_BYTES = 32 * 1024
WARMUP = 1
RUNS = 3
BASE = Path("/hpc/mydata/sricharan.varra/czarr_batch_io_probe_small")


def _open_and_register(path: Path):
    fd = os.open(str(path), os.O_RDONLY)
    descr = cufile.Descr()
    s = _CUfileDescr.from_address(int(descr))
    s.type = int(cufile.FileHandleType.OPAQUE_FD)
    s.handle.fd = fd
    s.fs_ops = 0
    fh = cufile.handle_register(int(descr))
    return fd, fh, descr


def _close_and_dereg(fd: int, fh) -> None:
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
    paths = [BASE / f"chunk_{i:05d}.bin" for i in range(N_FILES)]
    payload = np.random.default_rng(0).integers(0, 255, CHUNK_BYTES, dtype=np.uint8).tobytes()
    missing = [p for p in paths if not p.exists() or p.stat().st_size != CHUNK_BYTES]
    if missing:
        for p in missing:
            p.write_bytes(payload)
    return paths


def _bench_threaded_sync(paths: list[Path], arena: cp.ndarray, offsets: list[int]) -> float:
    """N OS threads, each opens + registers + reads + deregisters + closes (current path)."""

    def _one(idx: int) -> None:
        fd, fh, _descr = _open_and_register(paths[idx])
        try:
            cufile.read(fh, int(arena.data.ptr), CHUNK_BYTES, 0, offsets[idx])
        finally:
            _close_and_dereg(fd, fh)

    arena[:] = 0
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=32) as ex:
        list(ex.map(_one, range(N_FILES)))
    return time.perf_counter() - t0


def _bench_get_many(paths: list[Path], arena: cp.ndarray, offsets: list[int]) -> float:
    """Our new cufile_runtime.read_into_many path (serial open, parallel read)."""
    arena[:] = 0
    cp.cuda.runtime.deviceSynchronize()
    arena_ptr = int(arena.data.ptr)
    requests = [(path, arena_ptr + off, CHUNK_BYTES, 0) for path, off in zip(paths, offsets, strict=True)]
    t0 = time.perf_counter()
    cufile_runtime.read_into_many(requests, max_workers=32)
    return time.perf_counter() - t0


def _bench_batch_io(handles, arena: cp.ndarray, offsets: list[int]) -> float:
    """cuFile batched I/O — one batch_io_submit call for all N reads."""
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
    batch_id = cufile.batch_io_set_up(int(N_FILES))
    try:
        cufile.batch_io_submit(int(batch_id), int(N_FILES), iocbs.ctypes.data, 0)
        events = np.zeros(N_FILES, dtype=cufile.io_events_dtype)
        nr_arr = (ctypes.c_uint * 1)(N_FILES)
        completed = 0
        while completed < N_FILES:
            nr_arr[0] = N_FILES - completed
            cufile.batch_io_get_status(int(batch_id), 0, ctypes.addressof(nr_arr), events[completed:].ctypes.data, 0)
            completed += int(nr_arr[0])
    finally:
        try:
            cufile.batch_io_destroy(int(batch_id))
        except Exception:
            pass
    return time.perf_counter() - t0


def _measure(label: str, fn) -> float:
    for _ in range(WARMUP):
        fn()
    samples = [fn() for _ in range(RUNS)]
    return float(np.array(samples).mean())


def main() -> int:
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"nvidia_fs loaded: {os.path.exists('/proc/driver/nvidia-fs')}")
    print(f"N_FILES={N_FILES}  CHUNK_BYTES={CHUNK_BYTES}  total={N_FILES * CHUNK_BYTES / 1e6:.1f} MB")
    print()

    cufile_runtime.ensure_driver_open()
    paths = _prepare_files()
    arena = cp.empty(N_FILES * CHUNK_BYTES, dtype=cp.uint8)
    cufile.buf_register(int(arena.data.ptr), N_FILES * CHUNK_BYTES, 0)
    offsets = [i * CHUNK_BYTES for i in range(N_FILES)]

    handles = [(fd, fh) for fd, fh, _descr in (_open_and_register(p) for p in paths)]
    try:
        ts = _measure("threaded_sync", lambda: _bench_threaded_sync(paths, arena, offsets))
        gm = _measure("get_many (read_into_many)", lambda: _bench_get_many(paths, arena, offsets))
        bi = _measure("batch_io_submit", lambda: _bench_batch_io(handles, arena, offsets))
    finally:
        for fd, fh in handles:
            _close_and_dereg(fd, fh)
        cufile.buf_deregister(int(arena.data.ptr))

    total_gb = N_FILES * CHUNK_BYTES / 1e9
    print(f"{'method':<32}{'mean ms':>12}{'GB/s':>10}")
    print("-" * 60)
    for label, t in (("threaded_sync", ts), ("get_many", gm), ("batch_io_submit", bi)):
        print(f"{label:<32}{t * 1000:>10.1f} ms{total_gb / t:>9.2f}")
    print()
    print(f"speedup batch_io / threaded_sync: {ts / bi:.2f}x")
    print(f"speedup batch_io / get_many:      {gm / bi:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
