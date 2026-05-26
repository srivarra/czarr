"""GPU-resident Zarr Array facade.

Public surface:

* :class:`CudaZarrArray` — :class:`zarr.Array` subclass with a
  basic-indexing fast path that returns :class:`cupy.ndarray` on device.
  Advanced indexing falls through to :meth:`zarr.Array.__getitem__` and
  returns :class:`numpy.ndarray` on host.
* :func:`open_cuda_array` — open an existing array path and wrap it.
* :func:`create_cuda_array` — create + wrap in one call.

Implementation lives in :mod:`czarr.array._impl`; the public API
intentionally keeps the orchestrator off the user-facing namespace.
"""

from __future__ import annotations

from czarr.array._factories import (
    create_cuda_array,
    open_cuda_array,
)
from czarr.array.cuda_array import CudaZarrArray

__all__ = ["CudaZarrArray", "create_cuda_array", "open_cuda_array"]
