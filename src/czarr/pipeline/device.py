"""Device buffer pool — stream-ordered ``cuda.core`` allocation.

``acquire(size, stream=...)`` allocates a ``uint8`` device buffer through
``cuda.core``'s :class:`~cuda.core.DeviceMemoryResource`, stream-ordered on
the supplied :class:`~cuda.core.Stream` (the device default stream when
none is given).  The resulting ``cuda.core.Buffer`` is wrapped zero-copy as
a ``cupy.ndarray`` via DLPack, so callers keep the cupy ergonomics while
the allocation itself goes through cuda.core rather than cupy's current
stream context.

cupy's DLPack import holds a reference to the producer's deleter, so the
underlying ``cuda.core.Buffer`` stays alive for the lifetime of the
returned view — we never close it directly.

Future work:

* Free-list recycling on top of the memory resource (the previous
  cupy/RMM path recycled same-size allocations after warmup; the plain
  cuda.core resource does not pool, so hot loops that churn buffers may
  want a small per-size cache here).
* Per-stream scratch pools for nvCOMP intermediate buffers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
from cuda.core import Device

if TYPE_CHECKING:
    from cuda.core import Stream


class DeviceBufferPool:
    """Pipeline-facing API for device-resident byte buffers.

    Allocates through ``cuda.core``'s default
    :class:`~cuda.core.DeviceMemoryResource` and returns the buffer as a
    ``cupy.ndarray`` view.  Exposed as a class so Phase 2's pipeline imports
    a stable API while the underlying allocator evolves in later phases.
    """

    def acquire(self, size: int, *, stream: Stream | None = None) -> cp.ndarray:
        """Allocate a ``size``-byte uint8 device buffer.

        The allocation is stream-ordered on ``stream`` (the device default
        stream when ``None``) via ``cuda.core``'s
        :class:`~cuda.core.DeviceMemoryResource`, then wrapped zero-copy as
        a ``cupy.ndarray`` through DLPack.  The cupy view keeps the
        underlying ``cuda.core.Buffer`` alive, so the returned array owns
        its memory for its full lifetime.
        """
        if size <= 0:
            raise ValueError(f"DeviceBufferPool.acquire: size must be > 0, got {size}")
        device = Device()
        device.set_current()
        order = stream if stream is not None else device.default_stream
        buffer = device.memory_resource.allocate(size, stream=order)
        return cp.from_dlpack(buffer).view(cp.uint8)
