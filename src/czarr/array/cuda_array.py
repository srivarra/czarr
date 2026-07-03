"""``CudaZarrArray`` — :class:`zarr.Array` opened against the GPU stack.

A trivial :class:`zarr.Array` subclass: it adds no read-path behavior of
its own — GPU decode comes from the globally configured pipeline, buffer
prototypes, and store (see :func:`czarr.configure_gpu`).  The subclass
exists as the documented return type of :func:`open_cuda_array` /
:func:`create_cuda_array` and as the anchor for a future explicit
low-level read path (``czarr.lowlevel``).

Construction is through :meth:`CudaZarrArray.wrap` or the factories —
the constructor exists for parity with :class:`zarr.Array` but is not
the documented path.
"""

from __future__ import annotations

import zarr


class CudaZarrArray(zarr.Array):
    """:class:`zarr.Array` returned by the czarr factories.

    Reads return :class:`cupy.ndarray` on device when a GPU buffer
    prototype is configured globally (via :func:`czarr.configure_gpu`);
    otherwise the return type follows the active prototype, exactly as
    with a stock :class:`zarr.Array`.
    """

    @classmethod
    def wrap(cls, array: zarr.Array, /) -> CudaZarrArray:
        """Wrap an existing :class:`zarr.Array`.

        The preferred constructor — re-uses the existing array's
        metadata, store, and codec chain.  No second open.
        """
        return cls(array._async_array)
