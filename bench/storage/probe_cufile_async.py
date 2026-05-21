"""End-to-end probe: cuda.bindings.cufile.read_async into a cupy device buffer.

Verifies (a) async API is reachable from Python, (b) stream-ordered execution
works, (c) returned bytes match what was written.  Run from sbatch on an H100
node where nvidia_fs is loaded.
"""

from __future__ import annotations

import ctypes
import os
import sys

import cupy as cp
import numpy as np
from cuda.bindings import cufile as cf


class _FdHandle(ctypes.Union):
    _fields_ = (("fd", ctypes.c_int), ("handle", ctypes.c_void_p))


class _CUfileDescr(ctypes.Structure):
    _pack_ = 1
    _fields_ = (
        ("type", ctypes.c_int),
        ("_pad", ctypes.c_int),
        ("handle", _FdHandle),
        ("fs_ops", ctypes.c_void_p),
    )


def _set_pinned_i64(p, v: int) -> None:
    ctypes.cast(int(p), ctypes.POINTER(ctypes.c_int64))[0] = v


def _get_pinned_ssize(p) -> int:
    return ctypes.cast(int(p), ctypes.POINTER(ctypes.c_ssize_t))[0]


def _set_pinned_size(p, v: int) -> None:
    ctypes.cast(int(p), ctypes.POINTER(ctypes.c_size_t))[0] = v


def _try_path(label: str, path: str, *, use_buf_register: bool, use_o_direct: bool) -> bool:
    print(f"\n----- {label} | path={path} | buf_register={use_buf_register} | O_DIRECT={use_o_direct} -----")
    payload = np.arange(2048, dtype=np.uint32).tobytes()
    with open(path, "wb") as f:
        f.write(payload)
    flags = os.O_RDONLY | (os.O_DIRECT if use_o_direct else 0)
    fd = os.open(path, flags)
    descr = cf.Descr()
    s = _CUfileDescr.from_address(int(descr))
    s.type = int(cf.FileHandleType.OPAQUE_FD)
    s.handle.fd = fd
    s.fs_ops = 0
    fh = cf.handle_register(int(descr))

    dbuf = cp.empty(len(payload), dtype=cp.uint8)
    dbuf.fill(0xAA)  # poison to detect "no write happened"
    cp.cuda.runtime.deviceSynchronize()

    # Sync baseline first
    sync_n = cf.read(fh, int(dbuf.data.ptr), len(payload), 0, 0)
    sync_back = np.frombuffer(dbuf.get(), dtype=np.uint32)
    sync_ok = np.array_equal(sync_back, np.arange(len(sync_back), dtype=np.uint32))
    print(f"sync read: bytes={sync_n} match={sync_ok} first8={sync_back[:8].tolist()}")

    # Reset buffer
    dbuf.fill(0xAA)
    cp.cuda.runtime.deviceSynchronize()

    stream = cp.cuda.Stream()
    cf.stream_register(stream.ptr, 0)

    if use_buf_register:
        cf.buf_register(int(dbuf.data.ptr), len(payload), 0)

    pin_size = cp.cuda.alloc_pinned_memory(8)
    pin_off = cp.cuda.alloc_pinned_memory(8)
    pin_doff = cp.cuda.alloc_pinned_memory(8)
    pin_bytes = cp.cuda.alloc_pinned_memory(8)
    _set_pinned_size(pin_size, len(payload))
    _set_pinned_i64(pin_off, 0)
    _set_pinned_i64(pin_doff, 0)
    _set_pinned_i64(pin_bytes, -1)

    cf.read_async(
        fh,
        int(dbuf.data.ptr),
        int(pin_size),
        int(pin_off),
        int(pin_doff),
        int(pin_bytes),
        stream.ptr,
    )
    stream.synchronize()
    bytes_read = _get_pinned_ssize(pin_bytes)
    async_back = np.frombuffer(dbuf.get(), dtype=np.uint32)
    async_ok = np.array_equal(async_back, np.arange(len(async_back), dtype=np.uint32))
    print(f"async read: bytes={bytes_read} match={async_ok} first8={async_back[:8].tolist()}")

    if use_buf_register:
        cf.buf_deregister(int(dbuf.data.ptr))
    cf.stream_deregister(stream.ptr)
    cf.handle_deregister(fh)
    os.close(fd)
    os.unlink(path)
    return async_ok


def main() -> int:
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"nvidia_fs loaded: {os.path.exists('/proc/driver/nvidia-fs')}")
    cf.driver_open()
    print("cuFile driver opened")

    # Probe several locations to find one that supports real GDS.
    candidates = []
    for env_var in ("TMPDIR", "SLURM_TMPDIR"):
        v = os.environ.get(env_var)
        if v and os.path.isdir(v):
            candidates.append((f"$ {env_var}", os.path.join(v, "probe_async.bin")))
    for d in ("/local/scratch", "/tmp", os.getcwd()):
        if os.path.isdir(d):
            candidates.append((d, os.path.join(d, f"probe_async_{os.getpid()}.bin")))

    seen_paths = set()
    results = {}
    for label, path in candidates:
        if path in seen_paths:
            continue
        seen_paths.add(path)
        for use_buf_register in (True, False):
            for use_o_direct in (False, True):
                key = (path, use_buf_register, use_o_direct)
                try:
                    results[key] = _try_path(
                        label,
                        path,
                        use_buf_register=use_buf_register,
                        use_o_direct=use_o_direct,
                    )
                except Exception as e:
                    results[key] = f"raised: {type(e).__name__}: {e}"
                    print(f"  -> raised: {e}")

    print("\n=== summary ===")
    for (path, br, od), ok in results.items():
        marker = "OK " if ok is True else "FAIL"
        print(f"  {marker} | path={path} buf_register={br} O_DIRECT={od} -> {ok}")
    any_ok = any(v is True for v in results.values())
    return 0 if any_ok else 1


if __name__ == "__main__":
    sys.exit(main())
