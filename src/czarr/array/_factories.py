"""Factory helpers — :func:`open_cuda_array` and :func:`create_cuda_array`.

Single-call entry points that bundle the underlying :func:`zarr.open_array`
or :func:`zarr.create_array` with the :class:`CudaZarrArray.wrap` step.
Match the parent function signatures where possible so users can switch
``zarr.open_array`` → ``czarr.open_cuda_array`` with no other changes.

Default to cuFile via :class:`czarr.GPULocalStore` when called with a
string/``Path`` argument — that's the whole point of CudaZarrArray, and
without the GPU store the read path falls through to host bytes.  Users
who want a different store (``MemoryStore`` for tests, ``LocalStore``
for non-cuFile workflows) pass an explicit :class:`zarr.abc.store.Store`
instance and the factories pass it straight through.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Unpack

import zarr
import zarr.abc.store

from czarr.array.cuda_array import CudaZarrArray, CudaZarrArrayKwargs
from czarr.storage import GPULocalStore

if TYPE_CHECKING:
    import numpy.typing as npt


def _resolve_store(
    store_or_path: zarr.abc.store.Store | str | Path,
    *,
    mode: str,
) -> zarr.abc.store.Store:
    """Auto-wrap string/Path inputs in :class:`GPULocalStore`.

    A :class:`Store` instance passes through unchanged — explicit user
    choice (e.g. ``MemoryStore()`` for hermetic tests).
    """
    if isinstance(store_or_path, (str, Path)):
        return GPULocalStore(str(store_or_path), read_only=(mode == "r"))
    return store_or_path


def open_cuda_array(
    store: zarr.abc.store.Store | str | Path,
    *,
    path: str | None = None,
    mode: Literal["r", "r+", "a", "w", "w-"] = "r",
    **kwargs: Unpack[CudaZarrArrayKwargs],
) -> CudaZarrArray:
    """Open an existing array and wrap it as :class:`CudaZarrArray`.

    A string or :class:`pathlib.Path` is automatically wrapped in
    :class:`czarr.GPULocalStore` so the cuFile read path is the default;
    pass a :class:`zarr.abc.store.Store` instance to opt into a different
    backend.

    Tuning kwargs (``queue_depth``, ``microbatch_size``,
    ``stream_pool_size``) are threaded into the orchestrator; non-tuning
    kwargs raise :class:`TypeError`.
    """
    resolved = _resolve_store(store, mode=mode)
    arr = zarr.open_array(store=resolved, path=path, mode=mode)
    return CudaZarrArray.wrap(arr, **kwargs)


def create_cuda_array(
    store: zarr.abc.store.Store | str | Path,
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

    Mirrors :func:`zarr.create_array`'s positional/kwarg signature for
    the fields czarr users typically touch.  String/``Path`` ``store``
    arguments are auto-wrapped in :class:`czarr.GPULocalStore`; pass an
    explicit store instance to opt into a different backend.  Less-common
    kwargs (storage options, codec pipeline overrides, etc.) are not
    threaded — use :func:`zarr.create_array` + :meth:`CudaZarrArray.wrap`
    for those.
    """
    # Create mode — ``read_only=False`` regardless of the requested ``mode``.
    resolved = _resolve_store(store, mode="w")
    arr = zarr.create_array(
        store=resolved,
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
