"""Thin wrapper around ``cuda.bindings.cufile`` for direct file → GPU I/O.

cuFile silently degrades to "compatibility mode" (pinned host bounce buffer +
async memcpy) when the host or filesystem does not support real GDS DMA — the
call surface is identical, so the same code path serves both deployments.

Sync-only by design: the stream-ordered async wrappers were deleted after
benching 1.8-14x slower than threaded sync reads on Bruno NFS (see git
history, including ``bench/storage/probe_cufile_async.py``, for the constraints).
"""

import atexit
import ctypes
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from cuda.bindings import cufile


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


_lock = threading.Lock()
_opened = False


def _close() -> None:
    global _opened
    if not _opened:
        return
    clear_handle_cache()  # deregister cached handles before the driver goes away
    try:
        cufile.driver_close()
    except cufile.cuFileError:
        pass
    _opened = False


def ensure_driver_open() -> None:
    """Open the cuFile driver exactly once per process. Idempotent + thread-safe."""
    global _opened
    if _opened:
        return
    with _lock:
        if _opened:
            return
        cufile.driver_open()
        _opened = True
        atexit.register(_close)


def is_available() -> bool:
    """Return True when cuFile can be opened on this host (real GDS or compat)."""
    try:
        ensure_driver_open()
    except Exception:  # noqa: BLE001 — opaque cuFile failures => unavailable
        return False
    return True


# Friendly names for the cufile.json knobs reachable via set_parameter_*.
# Values mirror the cuFile configuration guide; tune via czarr.cufile.configure.
_SIZET_PARAMS = {
    "max_io_threads": cufile.SizeTConfigParameter.EXECUTION_MAX_IO_THREADS,
    "max_io_queue_depth": cufile.SizeTConfigParameter.EXECUTION_MAX_IO_QUEUE_DEPTH,
    "min_io_threshold_size_kb": cufile.SizeTConfigParameter.EXECUTION_MIN_IO_THRESHOLD_SIZE_KB,
    "max_request_parallelism": cufile.SizeTConfigParameter.EXECUTION_MAX_REQUEST_PARALLELISM,
    "max_direct_io_size_kb": cufile.SizeTConfigParameter.PROPERTIES_MAX_DIRECT_IO_SIZE_KB,
    "max_device_cache_size_kb": cufile.SizeTConfigParameter.PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB,
}
_BOOL_PARAMS = {
    "parallel_io": cufile.BoolConfigParameter.EXECUTION_PARALLEL_IO,
    "allow_compat_mode": cufile.BoolConfigParameter.PROPERTIES_ALLOW_COMPAT_MODE,
}


def configure(**knobs: int | bool) -> None:
    """Set cuFile config knobs programmatically (the cufile.json equivalents).

    Must be called BEFORE the driver opens (first read / ``is_available``
    call) — libcufile reads its configuration at ``driver_open``, and the
    setting is process-wide and permanent for the driver's lifetime.
    Raises ``RuntimeError`` if the driver is already open, ``KeyError``
    for an unknown knob.

    Knobs (see the GDS configuration guide for semantics):
    ``max_io_threads`` (internal pool, default 4), ``max_io_queue_depth``,
    ``min_io_threshold_size_kb`` (large-read split threshold, default
    8192), ``max_request_parallelism`` (default 4),
    ``max_direct_io_size_kb`` (per-request IO chunk, default 16384),
    ``max_device_cache_size_kb``, ``parallel_io`` (default True),
    ``allow_compat_mode`` (set False to fail loudly instead of silently
    staging through the CPU).
    """
    if _opened:
        raise RuntimeError("cuFile driver already open — configure() must run before the first cuFile call")
    for name, value in knobs.items():
        if name in _SIZET_PARAMS:
            cufile.set_parameter_size_t(_SIZET_PARAMS[name], int(value))
        elif name in _BOOL_PARAMS:
            cufile.set_parameter_bool(_BOOL_PARAMS[name], bool(value))
        else:
            raise KeyError(f"unknown cuFile knob {name!r}; known: {sorted(_SIZET_PARAMS | _BOOL_PARAMS)}")


def set_poll_mode(poll: bool, threshold_kb: int = 4) -> None:
    """Toggle cuFile's polling-vs-IRQ completion mode.

    With ``poll=True``, cuFile spins the calling thread until each I/O up to
    ``threshold_kb`` finishes, avoiding the wakeup cost of IRQ-driven
    completion.  Useful when many small reads dominate (sub-MiB chunks) and
    you have CPU to burn.  For our typical ≥1 MiB chunks the IRQ path is
    fine — the polling threshold caps which I/Os get the spin treatment.

    Idempotent.  Opens the driver if needed.
    """
    ensure_driver_open()
    cufile.driver_set_poll_mode(bool(poll), int(threshold_kb))


@contextmanager
def registered_handle(fd: int):
    """Register an OS file descriptor with cuFile and yield the cuFile handle.

    The cuFile handle is deregistered on exit; the caller still owns ``fd``.
    """
    descr = cufile.Descr()
    s = _CUfileDescr.from_address(int(descr))
    s.type = int(cufile.FileHandleType.OPAQUE_FD)
    s.handle.fd = fd
    s.fs_ops = 0
    handle = cufile.handle_register(int(descr))
    try:
        yield handle
    finally:
        try:
            cufile.handle_deregister(handle)
        except cufile.cuFileError:
            pass


# ---------------------------------------------------------------------------
# Persistent fd/handle cache
#
# open + handle_register cost ~1 ms per call on H100 (~3 ms in compat mode),
# paid per file per read before this cache.  Zarr reads hit the same chunk
# and shard files over and over, so registered handles are kept in a
# refcounted LRU keyed by path.  Every acquire revalidates the file's stat
# signature (inode, mtime, size): a rewritten/replaced file is reopened, so
# cached handles never serve stale data.  Entries evicted while a read is in
# flight are closed by the last releaser, never under a reader.
# ---------------------------------------------------------------------------

_HANDLE_CACHE_CAP = 128  # max cached fds; well under default ulimits


class _CachedHandle:
    __slots__ = ("dead", "descr", "fd", "handle", "refs", "sig")

    def __init__(self, fd: int, handle: object, descr: object, sig: tuple[int, int, int]) -> None:
        self.fd = fd
        self.handle = handle
        self.descr = descr  # keep the ctypes Descr alive alongside the handle
        self.sig = sig
        self.refs = 0
        self.dead = False


_handle_cache: OrderedDict[str, _CachedHandle] = OrderedDict()
_handle_cache_lock = threading.Lock()


def _destroy_entry(entry: _CachedHandle) -> None:
    try:
        cufile.handle_deregister(entry.handle)
    except cufile.cuFileError:
        pass
    try:
        os.close(entry.fd)
    except OSError:
        pass


def _register_path(path: str) -> tuple[int, object, object]:
    fd = os.open(path, os.O_RDONLY)
    descr = cufile.Descr()
    s = _CUfileDescr.from_address(int(descr))
    s.type = int(cufile.FileHandleType.OPAQUE_FD)
    s.handle.fd = fd
    s.fs_ops = 0
    try:
        handle = cufile.handle_register(int(descr))
    except cufile.cuFileError:
        os.close(fd)
        raise
    return fd, handle, descr


def _acquire_handle(path) -> _CachedHandle:
    """Cached-or-fresh registered handle for ``path``; pair with :func:`_release_handle`."""
    key = os.fspath(path)
    st = os.stat(key)  # noqa: PTH116 — key is already a plain string; Path() round-trip buys nothing
    sig = (st.st_ino, st.st_mtime_ns, st.st_size)
    with _handle_cache_lock:
        entry = _handle_cache.get(key)
        if entry is not None:
            if entry.sig == sig:
                entry.refs += 1
                _handle_cache.move_to_end(key)
                return entry
            # Same path, different file (rewrite/replace) — retire the entry.
            entry.dead = True
            del _handle_cache[key]
            if entry.refs == 0:
                _destroy_entry(entry)
        fd, handle, descr = _register_path(key)
        entry = _CachedHandle(fd, handle, descr, sig)
        entry.refs = 1
        _handle_cache[key] = entry
        while len(_handle_cache) > _HANDLE_CACHE_CAP:
            _evict_key, evicted = _handle_cache.popitem(last=False)
            evicted.dead = True
            if evicted.refs == 0:
                _destroy_entry(evicted)
        return entry


def _release_handle(entry: _CachedHandle) -> None:
    with _handle_cache_lock:
        entry.refs -= 1
        if entry.dead and entry.refs == 0:
            _destroy_entry(entry)


def clear_handle_cache() -> None:
    """Deregister and close every cached handle (in-flight reads finish first)."""
    with _handle_cache_lock:
        for entry in _handle_cache.values():
            entry.dead = True
            if entry.refs == 0:
                _destroy_entry(entry)
        _handle_cache.clear()


def handle_cache_len() -> int:
    """Number of live cached handles (introspection for tests/telemetry)."""
    with _handle_cache_lock:
        return len(_handle_cache)


def read_into(path, dev_ptr: int, size: int, file_offset: int = 0) -> int:
    """Read ``size`` bytes from ``path`` (at ``file_offset``) into device memory at ``dev_ptr``.

    File handles are served from the process-wide registered-handle cache;
    the file's stat signature is revalidated per call.
    """
    ensure_driver_open()
    entry = _acquire_handle(path)
    try:
        return cufile.read(entry.handle, dev_ptr, size, file_offset, 0)
    finally:
        _release_handle(entry)


def read_into_many(
    requests: list[tuple[str | os.PathLike[str], int, int, int]],
    *,
    max_workers: int | None = None,
) -> list[int]:
    """Batched cuFile reads with cached registered handles + threaded dispatch.

    Per-chunk profiling showed ~3 ms of fixed overhead per call to
    :func:`read_into` before handle caching (open + handle_register +
    read + handle_deregister + close).  This entry point acquires every
    handle up front from the process-wide cache (repeat reads and
    same-file requests within one batch pay the open+register cost only
    once), then issues all reads in parallel via a threadpool.

    Parameters
    ----------
    requests
        Sequence of ``(path, dev_ptr, size, file_offset)`` tuples.
        ``size`` may be zero — those entries return 0 without I/O.
    max_workers
        Thread-pool size.  ``None`` (default) lets ``ThreadPoolExecutor``
        pick (``min(32, cpu_count + 4)``).

    Returns
    -------
    list[int]
        One byte count per request, in input order.  Missing files raise
        ``FileNotFoundError`` synchronously — same semantics as
        :func:`read_into`.
    """
    if not requests:
        return []
    ensure_driver_open()

    n = len(requests)
    entries: list[_CachedHandle | None] = [None] * n

    try:
        # Phase 1: serial handle acquisition.  Tried parallelising this
        # with a ThreadPoolExecutor — on Bruno's VAST NFS it regressed
        # (cufile.handle_register holds a driver-level lock; threads just
        # added scheduling overhead).  Cache hits make it near-free; the
        # real speedup source is the parallel-read phase.
        for i, (path, _dev_ptr, size, _offset) in enumerate(requests):
            if size == 0:
                continue
            entries[i] = _acquire_handle(path)

        # Phase 2: parallel reads.  cuFile read internally releases the
        # GIL during the syscall, so this scales with thread count up to
        # what the driver's submission queue allows.
        results: list[int] = [0] * n

        def _do_read(i: int) -> int:
            _path, dev_ptr, size, file_offset = requests[i]
            entry = entries[i]
            if size == 0 or entry is None:
                return 0
            return cufile.read(entry.handle, dev_ptr, size, file_offset, 0)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for i, r in zip(range(n), ex.map(_do_read, range(n)), strict=True):
                results[i] = r

        return results
    finally:
        for entry in entries:
            if entry is not None:
                _release_handle(entry)


def write_from(path, dev_ptr: int, size: int, file_offset: int = 0) -> int:
    """Write ``size`` bytes from device memory at ``dev_ptr`` to ``path`` at ``file_offset``."""
    ensure_driver_open()
    flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if file_offset == 0 else 0)
    fd = os.open(os.fspath(path), flags, 0o644)
    try:
        with registered_handle(fd) as h:
            return cufile.write(h, dev_ptr, size, file_offset, 0)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Buffer registration (register-once reads in the bench harness)
# ---------------------------------------------------------------------------


_registered_bufs: dict[int, int] = {}  # ptr -> size
_registration_lock = threading.Lock()


def ensure_buf_registered(dev_ptr: int, size: int) -> None:
    """Register a device buffer with cuFile (idempotent per pointer).

    Re-registers if the size changed (caller reused the pointer for a larger
    region — rare but defensive).
    """
    existing = _registered_bufs.get(dev_ptr)
    if existing is not None and existing >= size:
        return
    with _registration_lock:
        existing = _registered_bufs.get(dev_ptr)
        if existing is not None:
            try:
                cufile.buf_deregister(dev_ptr)
            except cufile.cuFileError:
                pass
        cufile.buf_register(dev_ptr, size, 0)
        _registered_bufs[dev_ptr] = size


def deregister_buf(dev_ptr: int) -> None:
    """Best-effort buffer deregistration; silently ignores unknown pointers."""
    if dev_ptr not in _registered_bufs:
        return
    with _registration_lock:
        if dev_ptr not in _registered_bufs:
            return
        try:
            cufile.buf_deregister(dev_ptr)
        except cufile.cuFileError:
            pass
        _registered_bufs.pop(dev_ptr, None)
