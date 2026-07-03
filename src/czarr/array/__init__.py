"""GPU-resident Zarr Array facade.

Public surface:

* :class:`CudaZarrArray` — :class:`zarr.Array` subclass returned by the
  factories; GPU behavior comes from the globally configured pipeline,
  prototypes, and store, not from the subclass.
* :func:`open_cuda_array` — open an existing array path and wrap it.
* :func:`create_cuda_array` — create + wrap in one call.
"""

from czarr.array._factories import (
    create_cuda_array,
    open_cuda_array,
)
from czarr.array.cuda_array import CudaZarrArray

__all__ = ["CudaZarrArray", "create_cuda_array", "open_cuda_array"]
