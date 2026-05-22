"""Pinned-host buffer pool — pre-allocated slab + on-demand growth.

Use case: staging compressed chunk bytes for H2D transfers inside
:class:`CzarrPipeline` (Phase 2).  Each ``arr[...]`` call copies many
chunks; reusing pinned allocations across calls avoids the ~ms cost of
each fresh ``cudaHostAlloc``.

Strategy:

* Pre-allocate one or more *prealloc* buffers at construction (typical
  imaging chunk sizes — caller passes the hint).  These land in the
  free list immediately.
* :meth:`acquire` returns a Buffer of the requested size, preferring
  the free list (exact-size match) before falling back to a fresh
  ``cudaHostAlloc``.
* :meth:`release` returns the buffer to the free list for the next
  ``acquire`` of the same size.
* :meth:`close` frees everything.

Exact-size matching keeps the implementation tiny.  Imaging chunks are
typically uniform-sized within an array (Zarr enforces this for inner
chunks, modulo edge chunks), so exact match handles most traffic.
Future refinement: size-bucketed pool with rounding.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from cuda.core import Buffer, LegacyPinnedMemoryResource

if TYPE_CHECKING:
    from collections.abc import Iterable


class PinnedHostPool:
    """Recycling pool of pinned-host ``cuda.core.Buffer`` objects.

    Parameters
    ----------
    prealloc:
        Optional iterable of ``(size, count)`` tuples — pre-allocate
        ``count`` buffers of ``size`` bytes each at construction.  These
        amortise the ``cudaHostAlloc`` cost over the lifetime of the
        pool.  Default no pre-allocation.

    Notes
    -----
    Uses ``LegacyPinnedMemoryResource`` (non-stream-ordered) so the
    pool can allocate without binding to a specific stream — buffers
    flow between streams via stream-ordered ``copy_from``/``copy_to``.
    """

    def __init__(self, prealloc: Iterable[tuple[int, int]] | None = None) -> None:
        self._mr = LegacyPinnedMemoryResource()
        self._free: dict[int, list[Buffer]] = defaultdict(list)
        self._live: int = 0  # count of buffers handed out and not yet released
        if prealloc:
            for size, count in prealloc:
                for _ in range(count):
                    self._free[size].append(self._mr.allocate(size, stream=None))

    @property
    def live_count(self) -> int:
        """Number of buffers handed out by :meth:`acquire` and not yet released."""
        return self._live

    def free_count(self, size: int) -> int:
        """Number of recyclable buffers of ``size`` bytes on the free list."""
        return len(self._free.get(size, []))

    def acquire(self, size: int) -> Buffer:
        """Return a pinned ``Buffer`` of ``size`` bytes.

        Prefers a recycled buffer from the free list (exact-size match);
        falls back to a fresh allocation if none available.
        """
        if size <= 0:
            raise ValueError(f"PinnedHostPool.acquire: size must be > 0, got {size}")
        bucket = self._free[size]
        buf = bucket.pop() if bucket else self._mr.allocate(size, stream=None)
        self._live += 1
        return buf

    def release(self, buf: Buffer) -> None:
        """Return ``buf`` to the free list for future ``acquire(buf.size)``."""
        self._free[buf.size].append(buf)
        self._live -= 1

    def close(self) -> None:
        """Free all buffers, both live-on-loan and on the free list.

        Caller is responsible for ensuring no buffer is still in use
        when this is called — typically after a ``StreamPool.sync_all()``.
        """
        for bucket in self._free.values():
            for buf in bucket:
                buf.close()
        self._free.clear()
        self._live = 0
