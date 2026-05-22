"""Device buffer pool — thin wrapper over RMM.

Phase 1 keeps this minimal: a single ``acquire(size, stream=...)``
method that allocates a ``uint8`` device buffer.  RMM (already wired in
via :func:`czarr.alloc.register_nvcomp_allocator`) absorbs the
recycling work under the hood — cupy's allocator hits the RMM pool, so
repeated allocations of the same size are effectively free after warmup.

Future work:

* Direct ``cuda.core.DeviceMemoryResource`` integration if/when we drop
  the cupy dependency in hot paths.
* Stream-ordered allocations bound to a particular ``Stream`` from the
  :class:`StreamPool` for true async lifetime.
* Per-stream scratch pools for nvCOMP intermediate buffers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp

if TYPE_CHECKING:
    from cuda.core import Stream


class DeviceBufferPool:
    """Pipeline-facing API for device-resident byte buffers.

    Currently a one-line wrapper around ``cp.empty`` (RMM-backed) but
    exposed as a class so Phase 2's pipeline imports a stable API while
    we tune the underlying allocator in later phases.
    """

    def acquire(self, size: int, *, stream: Stream | None = None) -> cp.ndarray:
        """Allocate a ``size``-byte uint8 device buffer.

        ``stream`` is accepted for forward-compatibility with a future
        stream-ordered allocator; ignored today because RMM/cupy
        already do their own internal stream handling.
        """
        if size <= 0:
            raise ValueError(f"DeviceBufferPool.acquire: size must be > 0, got {size}")
        if stream is not None:
            with cp.cuda.Stream.from_external(stream):
                return cp.empty(size, dtype=cp.uint8)
        return cp.empty(size, dtype=cp.uint8)
