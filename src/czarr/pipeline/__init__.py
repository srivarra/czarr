"""GPU-native zarr v3 codec pipeline.

Phase 1 of the refactor (epic ianyfe7m) introduces the ``cuda.core``
substrate:

* :class:`StreamPool` — round-robin assignment of N CUDA streams.
* :class:`PinnedHostPool` — recycling pool of pinned-host buffers used
  for H2D staging of compressed chunks.
* :class:`DeviceBufferPool` — thin wrapper around RMM-backed device
  allocations.

Phase 2 layers ``CzarrPipeline`` (a :class:`zarr.abc.codec.CodecPipeline`
subclass) on top of these primitives.
"""

from czarr.pipeline.device import DeviceBufferPool
from czarr.pipeline.pinned import PinnedHostPool
from czarr.pipeline.streams import StreamPool

__all__ = [
    "DeviceBufferPool",
    "PinnedHostPool",
    "StreamPool",
]
