"""``_CudaArrayImpl`` — the orchestrator behind :class:`CudaZarrArray`.

The implementation is intentionally minimal in the v0.1 skeleton: it holds
the underlying :class:`zarr.AsyncArray` reference plus tuning knobs and
delegates :meth:`retrieve_gpu` to zarr's existing async machinery via
:func:`zarr.core.sync.sync`.  Subsequent subtasks replace the body with the
bounded read-queue producer + single-decode-call pipeline (``lfw5ftx9``)
and the native-LZ4 codec backend (``s7ucov1a``).

The orchestrator owns:

* a reference to the underlying :class:`zarr.AsyncArray` — needed for
  metadata, codec pipeline, store path, chunk grid, and to drive
  selection without going through :class:`CudaZarrArray.__getitem__`
  (which would recurse).
* runtime knobs (``queue_depth``, ``microbatch_size``) that the I/O
  pipeline will consume in a later subtask.

Nothing here owns CUDA streams or buffers yet — that comes with the
I/O queue subtask.  This file is deliberately small.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict, Unpack

from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync

if TYPE_CHECKING:
    import cupy as cp
    import zarr


class _ImplKwargs(TypedDict, total=False):
    """Tuning knobs threaded through from :class:`CudaZarrArray`.

    All optional; defaults live in :class:`_CudaArrayImpl`.
    """

    queue_depth: int
    stream_pool_size: int
    microbatch_size: int


@dataclass(slots=True)
class _CudaArrayImpl:
    """Read-path orchestrator for :class:`CudaZarrArray`.

    The v0.1 skeleton drives selection through the underlying
    :class:`zarr.AsyncArray` so we never recurse into the parent
    :meth:`zarr.Array.__getitem__`.  Later iterations replace this with
    a bounded I/O queue feeding a single decode call.
    """

    async_array: zarr.AsyncArray
    queue_depth: int = 16
    stream_pool_size: int = 4
    microbatch_size: int = 8

    @classmethod
    def from_array(
        cls,
        array: zarr.Array,
        /,
        **kwargs: Unpack[_ImplKwargs],
    ) -> _CudaArrayImpl:
        """Build an impl bound to an existing :class:`zarr.Array`'s async surface."""
        return cls(async_array=array._async_array, **kwargs)

    def retrieve_gpu(self, key: Any) -> cp.ndarray:
        """Read a basic-indexed selection into device memory.

        v0.1 delegates to ``AsyncArray.getitem``, which routes through the
        configured codec pipeline.  When :func:`czarr.configure_gpu` has
        set the GPU buffer prototype, the result is a :class:`cupy.ndarray`
        on the device.  When called without that configuration, the result
        is a :class:`numpy.ndarray` — the caller is expected to verify the
        active prototype before promising device output.

        Subsequent subtasks (``lfw5ftx9``) replace the body with the
        bounded-queue read pipeline that bypasses :class:`BatchedCodecPipeline.read_batch`.
        """
        return sync(self.async_array.getitem(key, prototype=default_buffer_prototype()))
