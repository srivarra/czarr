"""Factory helpers — :func:`open_cuda_array` and :func:`create_cuda_array`.

Single-call entry points that bundle the underlying :func:`zarr.open_array`
or :func:`zarr.create_array` with the :class:`CudaZarrArray.wrap` step.
Match the parent function signatures where possible so users can switch
``zarr.open_array`` → ``czarr.open_cuda_array`` with no other changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Unpack

import zarr

from czarr.array.cuda_array import CudaZarrArray, CudaZarrArrayKwargs

if TYPE_CHECKING:
    import numpy.typing as npt
    import zarr.abc.store


def open_cuda_array(
    store: zarr.abc.store.Store | str,
    *,
    path: str | None = None,
    mode: Literal["r", "r+", "a", "w", "w-"] = "r",
    **kwargs: Unpack[CudaZarrArrayKwargs],
) -> CudaZarrArray:
    """Open an existing array and wrap it as :class:`CudaZarrArray`.

    Equivalent to ``CudaZarrArray.wrap(zarr.open_array(store, path=path, mode=mode))``.
    Tuning kwargs (``queue_depth``, ``microbatch_size``, ``stream_pool_size``)
    are threaded into the orchestrator; non-tuning kwargs raise
    :class:`TypeError`.
    """
    arr = zarr.open_array(store=store, path=path, mode=mode)
    return CudaZarrArray.wrap(arr, **kwargs)


def create_cuda_array(
    store: zarr.abc.store.Store | str,
    *,
    shape: tuple[int, ...],
    dtype: npt.DTypeLike,
    chunks: tuple[int, ...] | Literal["auto"] = "auto",
    compressors: list[Any] | Literal["auto"] | None = "auto",
    filters: list[Any] | Literal["auto"] | None = "auto",
    fill_value: Any | None = None,
    overwrite: bool = False,
    path: str | None = None,
    **kwargs: Unpack[CudaZarrArrayKwargs],
) -> CudaZarrArray:
    """Create + wrap in one call.

    Mirrors :func:`zarr.create_array`'s positional/kwarg signature for the
    fields czarr users typically touch.  Less-common kwargs (storage
    options, codec pipeline overrides, etc.) are not threaded — use
    :func:`zarr.create_array` + :meth:`CudaZarrArray.wrap` for those.
    """
    arr = zarr.create_array(
        store=store,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        compressors=compressors,
        filters=filters,
        fill_value=fill_value,
        overwrite=overwrite,
        name=path,
    )
    return CudaZarrArray.wrap(arr, **kwargs)
