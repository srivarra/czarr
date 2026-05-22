"""Thin wrapper around ``cuda.bindings.cufile`` for direct file → GPU I/O.

cuFile silently degrades to "compatibility mode" (pinned host bounce buffer +
async memcpy) when the host or filesystem does not support real GDS DMA — the
call surface is identical, so the same code path serves both deployments.

Async (``read_into_async``) needs strictly more setup than sync: stream must be
registered, device buffer must be registered, and the size/offset/bytes-read
arguments are *pointers to pinned host memory* (the GDS kernel reads them when
the operation runs on-stream).  See module ``probe_cufile_async.py`` for the
empirical constraints we discovered on Bruno.
"""

from __future__ import annotations

import atexit
import ctypes
import os
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import cupy as cp
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


def is_async_available() -> bool:
    """True when the async read/write path can be used reliably.

    Requires real GDS — ``nvidia_fs`` kernel module loaded.  On compat-mode-only
    hosts (e.g. Bruno A40 nodes), libcufile crashes inside its worker thread on
    the first ``read_async`` call.
    """
    return is_available() and os.path.exists("/proc/driver/nvidia-fs")


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


def read_into(path, dev_ptr: int, size: int, file_offset: int = 0) -> int:
    """Read ``size`` bytes from ``path`` (at ``file_offset``) into device memory at ``dev_ptr``."""
    ensure_driver_open()
    fd = os.open(os.fspath(path), os.O_RDONLY)
    try:
        with registered_handle(fd) as h:
            return cufile.read(h, dev_ptr, size, file_offset, 0)
    finally:
        os.close(fd)


def read_into_many(
    requests: list[tuple[object, int, int, int]],
    *,
    max_workers: int | None = None,
) -> list[int]:
    """Batched cuFile reads with pre-registered handles + threaded dispatch.

    Per-chunk profiling showed ~3 ms of fixed overhead per call to
    :func:`read_into` (open + handle_register + read + handle_deregister
    + close).  For multi-chunk reads the register/deregister fraction
    dominates.  This entry point pays the per-fd setup costs up front,
    issues all the actual reads in parallel via a threadpool, then tears
    everything down in one pass.

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
    fds: list[int | None] = [None] * n
    handles: list[object | None] = [None] * n
    descrs: list[object | None] = [None] * n

    try:
        # Phase 1: serial open + register.  Tried parallelising this with
        # a ThreadPoolExecutor — on Bruno's VAST NFS it regressed
        # (cufile.handle_register holds a driver-level lock; threads just
        # added scheduling overhead).  Keep serial; the real speedup
        # source is the parallel-read phase plus future cuFile batched I/O.
        for i, (path, _dev_ptr, size, _offset) in enumerate(requests):
            if size == 0:
                continue
            fd = os.open(os.fspath(path), os.O_RDONLY)
            descr = cufile.Descr()
            s = _CUfileDescr.from_address(int(descr))
            s.type = int(cufile.FileHandleType.OPAQUE_FD)
            s.handle.fd = fd
            s.fs_ops = 0
            handles[i] = cufile.handle_register(int(descr))
            fds[i] = fd
            descrs[i] = descr

        # Phase 2: parallel reads.  cuFile read internally releases the
        # GIL during the syscall, so this scales with thread count up to
        # what the driver's submission queue allows.
        results: list[int] = [0] * n

        def _do_read(i: int) -> int:
            _path, dev_ptr, size, file_offset = requests[i]
            h = handles[i]
            if size == 0 or h is None:
                return 0
            return cufile.read(h, dev_ptr, size, file_offset, 0)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for i, r in zip(range(n), ex.map(_do_read, range(n)), strict=True):
                results[i] = r

        return results
    finally:
        # Phase 3: deregister + close everything we touched.
        for h in handles:
            if h is None:
                continue
            try:
                cufile.handle_deregister(h)
            except cufile.cuFileError:
                pass
        for fd in fds:
            if fd is None:
                continue
            try:
                os.close(fd)
            except OSError:
                pass


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
# Async path
# ---------------------------------------------------------------------------


_registered_streams: set[int] = set()
_registered_bufs: dict[int, int] = {}  # ptr -> size
_async_lock = threading.Lock()


def ensure_stream_registered(stream_ptr: int) -> None:
    """Register a CUDA stream with cuFile (idempotent per stream pointer)."""
    if stream_ptr in _registered_streams:
        return
    with _async_lock:
        if stream_ptr in _registered_streams:
            return
        cufile.stream_register(stream_ptr, 0)
        _registered_streams.add(stream_ptr)


def ensure_buf_registered(dev_ptr: int, size: int) -> None:
    """Register a device buffer with cuFile (idempotent per pointer).

    Re-registers if the size changed (caller reused the pointer for a larger
    region — rare but defensive).
    """
    existing = _registered_bufs.get(dev_ptr)
    if existing is not None and existing >= size:
        return
    with _async_lock:
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
    with _async_lock:
        if dev_ptr not in _registered_bufs:
            return
        try:
            cufile.buf_deregister(dev_ptr)
        except cufile.cuFileError:
            pass
        _registered_bufs.pop(dev_ptr, None)


class _AsyncIOArgs:
    """Pinned host scalars for one in-flight ``read_async`` / ``write_async``.

    cuFile's stream-ordered API takes pointers to size/offset/bytes-completed —
    the GDS kernel reads them when the work actually runs.  These pointers
    must outlive the call until the stream syncs, so the caller stashes one of
    these objects per outstanding submission.

    Optional ``fd`` / ``fh`` / ``descr`` slots are populated by
    :func:`read_into_async` to keep the OS fd, cuFile handle, and descriptor
    alive until ``args`` is GC'd.  When set directly via :func:`read_async`
    (caller owns the handle), they stay ``None`` and ``deregister``-on-finalize
    is the caller's responsibility.
    """

    __slots__ = ("size_p", "off_p", "doff_p", "bytes_p", "_fd", "_fh", "_descr", "__weakref__")

    def __init__(self) -> None:
        self.size_p = cp.cuda.alloc_pinned_memory(8)
        self.off_p = cp.cuda.alloc_pinned_memory(8)
        self.doff_p = cp.cuda.alloc_pinned_memory(8)
        self.bytes_p = cp.cuda.alloc_pinned_memory(8)
        self._fd: int | None = None
        self._fh: int | None = None
        self._descr: object | None = None

    def fill(self, size: int, file_offset: int, dev_offset: int) -> None:
        """Populate the pinned scalars before submitting the async I/O."""
        ctypes.cast(int(self.size_p), ctypes.POINTER(ctypes.c_size_t))[0] = size
        ctypes.cast(int(self.off_p), ctypes.POINTER(ctypes.c_int64))[0] = file_offset
        ctypes.cast(int(self.doff_p), ctypes.POINTER(ctypes.c_int64))[0] = dev_offset
        ctypes.cast(int(self.bytes_p), ctypes.POINTER(ctypes.c_ssize_t))[0] = -1

    @property
    def bytes_done(self) -> int:
        """Bytes actually transferred (read after stream sync)."""
        return ctypes.cast(int(self.bytes_p), ctypes.POINTER(ctypes.c_ssize_t))[0]


def make_io_args() -> _AsyncIOArgs:
    """Allocate pinned-host scalar set for one async submission."""
    return _AsyncIOArgs()


def read_async(
    fh: int,
    dev_ptr: int,
    size: int,
    stream_ptr: int,
    *,
    args: _AsyncIOArgs,
    file_offset: int = 0,
    dev_offset: int = 0,
) -> None:
    """Submit a stream-ordered cuFile read into ``dev_ptr``.

    Caller is responsible for: (1) pre-registering the device buffer via
    :func:`ensure_buf_registered`, (2) pre-registering the stream via
    :func:`ensure_stream_registered`, (3) holding ``args`` alive until the
    stream syncs, then reading ``args.bytes_done``.

    Note: this function assumes the caller has already opened the file +
    registered its handle.  See :func:`read_into_async` for the path-based
    convenience wrapper.
    """
    args.fill(size, file_offset, dev_offset)
    cufile.read_async(
        fh,
        dev_ptr,
        int(args.size_p),
        int(args.off_p),
        int(args.doff_p),
        int(args.bytes_p),
        stream_ptr,
    )


def read_into_async(
    path,
    dev_ptr: int,
    size: int,
    stream_ptr: int,
    *,
    file_offset: int = 0,
) -> _AsyncIOArgs:
    """Path-based async read for one-shot use; opens + registers + submits.

    Returns the in-flight ``_AsyncIOArgs`` — caller must keep it alive until
    the stream syncs, then call ``args.bytes_done`` to get the byte count.
    The file handle is deregistered + closed when ``args`` is GC'd; for hot
    paths use :func:`registered_handle` + :func:`read_async` directly.
    """
    if not is_async_available():
        raise RuntimeError(
            "cuFile async path unavailable (need nvidia_fs kernel module). Use read_into for sync fallback."
        )
    ensure_driver_open()
    ensure_stream_registered(stream_ptr)
    ensure_buf_registered(dev_ptr, size)
    fd = os.open(os.fspath(path), os.O_RDONLY)
    descr = cufile.Descr()
    s = _CUfileDescr.from_address(int(descr))
    s.type = int(cufile.FileHandleType.OPAQUE_FD)
    s.handle.fd = fd
    s.fs_ops = 0
    fh = cufile.handle_register(int(descr))
    args = make_io_args()
    # Keep fd + fh + descr alive on the args object so they outlive the
    # async call.  ``__slots__`` declares them so this is type-clean.
    args._fd = fd
    args._fh = fh
    args._descr = descr

    weakref.finalize(args, _close_async_handle, fd, fh)
    read_async(fh, dev_ptr, size, stream_ptr, args=args, file_offset=file_offset)
    return args


def _close_async_handle(fd: int, fh: int) -> None:
    """Deregister a cuFile handle + close its fd; called from ``_AsyncIOArgs`` finalize."""
    try:
        cufile.handle_deregister(fh)
    except cufile.cuFileError:
        pass
    try:
        os.close(fd)
    except OSError:
        pass
