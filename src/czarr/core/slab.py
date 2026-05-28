"""``CuFileSlabPool`` — register-once cuFile arena over cuda.core VMR slabs.

The motivation: ``CzarrGpuBuffer.empty()`` allocating a *fresh*
``VirtualMemoryResource`` block per chunk pays two fixed costs every
call —

1. VMR allocation: ``cuMemCreate`` + ``cuMemAddressReserve`` +
   ``cuMemMap``, padded up to the device's VMM granularity (2 MiB on
   Hopper/Ampere).  ~ms-scale.
2. cuFile ``buf_register`` of the fresh pointer.

On the H200 ``slice_compare`` workload that per-chunk tax was a ~5x
regression versus stock cupy, which is why ``configure_gpu`` defaults
to cupy buffers today.

This pool amortises both costs.  It pre-allocates large 4 KiB-aligned,
GPU-direct-RDMA-tagged slabs through ``VirtualMemoryResource`` and
registers each slab with cuFile exactly **once**.  Per-chunk buffers are
zero-copy ``cupy`` slices into a slab, so:

* No per-chunk VMR allocation — sub-region hand-out is a free-list pop.
* No per-chunk cuFile ``buf_register`` — a ``cuFileRead`` into a slab
  sub-region reuses the slab's registration.
* Every sub-region inherits the slab's 4 KiB alignment (sizes are
  rounded up to 4 KiB so neighbouring regions stay aligned).

Memory safety is handled by cupy: each :class:`SlabAllocation` holds a
``cupy`` slice of the slab's base view, and cupy keeps the underlying
``cuda.core.Buffer`` alive as long as any slice references it.  The
free-list is purely for *reuse* — a region returns to it when its
:class:`SlabAllocation` is garbage-collected, never before.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cupy as cp

from czarr.core.buffer import _current_stream, _device_mr

if TYPE_CHECKING:
    from cuda.core import Stream


_ALIGN = 4096


def _round_up(n: int, align: int = _ALIGN) -> int:
    """Round ``n`` up to the next multiple of ``align``."""
    return (n + align - 1) // align * align


@dataclass
class _Slab:
    """One registered VMR slab plus its free-region book-keeping.

    ``view`` is a uint8 ``cupy`` view over the whole slab; sub-regions
    are slices of it.  Holding ``view`` (and the producing
    ``cuda.core.Buffer`` it imported from) keeps the device mapping
    alive for the slab's lifetime.
    """

    cuda_buffer: object  # cuda.core.Buffer — kept alive
    view: cp.ndarray  # uint8 view over the entire slab
    base_ptr: int
    size: int
    registered: bool
    # Sorted, coalesced list of free ``(offset, length)`` regions.
    free: list[tuple[int, int]] = field(default_factory=list)


class SlabAllocation:
    """A 4 KiB-aligned sub-region of a slab, returned to the pool on GC.

    Exposes :attr:`array` (a uint8 ``cupy`` view of exactly ``size``
    bytes) and :attr:`device_ptr` (4 KiB-aligned).  The region is
    returned to its slab's free-list when this object is collected —
    never sooner, so any ``cupy`` view derived from :attr:`array`
    remains valid for its own lifetime.
    """

    __slots__ = ("_pool", "_slab", "_offset", "_rounded", "array", "size", "__weakref__")

    def __init__(self, pool: CuFileSlabPool, slab: _Slab, offset: int, size: int, rounded: int) -> None:
        self._pool = pool
        self._slab = slab
        self._offset = offset
        self._rounded = rounded
        self.size = size
        self.array: cp.ndarray = slab.view[offset : offset + size]

    @property
    def device_ptr(self) -> int:
        """4 KiB-aligned device pointer to the start of the sub-region."""
        return self._slab.base_ptr + self._offset

    def __del__(self) -> None:
        # Return the rounded region to the free-list.  Guard against
        # interpreter-shutdown races where the pool is already gone.
        pool = getattr(self, "_pool", None)
        if pool is None:
            return
        try:
            pool._release(self._slab, self._offset, self._rounded)
        except Exception:  # noqa: BLE001 — best-effort reclaim at GC time
            pass


class CuFileSlabPool:
    """Process-friendly arena of register-once cuFile slabs.

    Parameters
    ----------
    slab_bytes:
        Size of each slab.  A request larger than this gets its own
        right-sized slab.  Default 64 MiB.
    register:
        Register each slab with cuFile when available.  Set ``False`` to
        get the alignment/sub-allocation benefits without touching
        cuFile (e.g. on hosts without GDS, or for decode-only paths).
    """

    def __init__(self, slab_bytes: int = 64 << 20, *, register: bool = True) -> None:
        self._slab_bytes = _round_up(slab_bytes)
        self._slabs: list[_Slab] = []
        self._lock = threading.Lock()
        # Resolve cuFile availability lazily so importing this module
        # never forces a driver open.
        self._register_requested = register
        self._cufile_ready: bool | None = None

    # ------------------------------------------------------------------
    # cuFile availability (lazy)
    # ------------------------------------------------------------------

    def _should_register(self) -> bool:
        if not self._register_requested:
            return False
        if self._cufile_ready is None:
            from czarr.storage import cufile_runtime

            self._cufile_ready = cufile_runtime.is_available()
        return self._cufile_ready

    # ------------------------------------------------------------------
    # slab management
    # ------------------------------------------------------------------

    def _new_slab(self, min_size: int, stream: Stream) -> _Slab:
        size = max(self._slab_bytes, _round_up(min_size))
        cuda_buf = _device_mr().allocate(size, stream=stream)
        view = cp.from_dlpack(cuda_buf).view(cp.uint8)
        base = int(view.data.ptr)
        registered = False
        if self._should_register():
            from czarr.storage import cufile_runtime

            cufile_runtime.ensure_buf_registered(base, size)
            registered = True
        slab = _Slab(
            cuda_buffer=cuda_buf,
            view=view,
            base_ptr=base,
            size=size,
            registered=registered,
            free=[(0, size)],
        )
        self._slabs.append(slab)
        return slab

    @staticmethod
    def _take(slab: _Slab, need: int) -> int | None:
        """First-fit: carve ``need`` bytes from ``slab``'s free list.

        Returns the offset, or ``None`` if no free region is large
        enough.  ``need`` is already 4 KiB-rounded so every returned
        offset stays aligned.
        """
        for i, (off, length) in enumerate(slab.free):
            if length >= need:
                if length == need:
                    slab.free.pop(i)
                else:
                    slab.free[i] = (off + need, length - need)
                return off
        return None

    @staticmethod
    def _give_back(slab: _Slab, offset: int, length: int) -> None:
        """Insert a freed region and coalesce with neighbours."""
        free = slab.free
        # Insert in offset order.
        lo, hi = 0, len(free)
        while lo < hi:
            mid = (lo + hi) // 2
            if free[mid][0] < offset:
                lo = mid + 1
            else:
                hi = mid
        free.insert(lo, (offset, length))
        # Coalesce left + right.
        merged: list[tuple[int, int]] = []
        for off, ln in free:
            if merged and merged[-1][0] + merged[-1][1] == off:
                p_off, p_ln = merged[-1]
                merged[-1] = (p_off, p_ln + ln)
            else:
                merged.append((off, ln))
        slab.free = merged

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def allocate(self, size: int, *, stream: Stream | None = None) -> SlabAllocation:
        """Hand out a 4 KiB-aligned ``size``-byte sub-region.

        Reuses a free region in an existing slab when one fits;
        otherwise grows by one slab.  Thread-safe.
        """
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        need = _round_up(size)
        s = stream or _current_stream()
        with self._lock:
            for slab in self._slabs:
                off = self._take(slab, need)
                if off is not None:
                    return SlabAllocation(self, slab, off, size, need)
            slab = self._new_slab(need, s)
            off = self._take(slab, need)
            assert off is not None  # fresh slab always fits
            return SlabAllocation(self, slab, off, size, need)

    def _release(self, slab: _Slab, offset: int, rounded: int) -> None:
        with self._lock:
            self._give_back(slab, offset, rounded)

    # ------------------------------------------------------------------
    # introspection (tests + bench)
    # ------------------------------------------------------------------

    @property
    def n_slabs(self) -> int:
        """Number of slabs currently allocated."""
        return len(self._slabs)

    @property
    def committed_bytes(self) -> int:
        """Total device bytes reserved across all slabs."""
        return sum(s.size for s in self._slabs)

    def free_bytes(self) -> int:
        """Total currently-free bytes across all slabs."""
        with self._lock:
            return sum(ln for slab in self._slabs for _, ln in slab.free)

    def close(self) -> None:
        """Deregister + drop all slabs.

        Caller must ensure no live :class:`SlabAllocation` references
        remain — outstanding cupy views would otherwise alias freed
        registrations.  Intended for test teardown / pool replacement.
        """
        with self._lock:
            if self._cufile_ready:
                from czarr.storage import cufile_runtime

                for slab in self._slabs:
                    if slab.registered:
                        cufile_runtime.deregister_buf(slab.base_ptr)
            self._slabs.clear()


# ---------------------------------------------------------------------------
# Process-global default pool
# ---------------------------------------------------------------------------

_DEFAULT_POOL: CuFileSlabPool | None = None
_DEFAULT_POOL_LOCK = threading.Lock()


def get_default_slab_pool() -> CuFileSlabPool:
    """Lazily-created process-global :class:`CuFileSlabPool`.

    ``CzarrGpuBuffer.empty`` draws from this so every chunk buffer shares
    the slabs' single cuFile registration.
    """
    global _DEFAULT_POOL
    if _DEFAULT_POOL is None:
        with _DEFAULT_POOL_LOCK:
            if _DEFAULT_POOL is None:
                _DEFAULT_POOL = CuFileSlabPool()
    return _DEFAULT_POOL


def set_default_slab_pool(pool: CuFileSlabPool | None) -> None:
    """Replace (or clear) the process-global pool.  Test + tuning hook."""
    global _DEFAULT_POOL
    with _DEFAULT_POOL_LOCK:
        _DEFAULT_POOL = pool
