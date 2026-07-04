"""GPU-resident Zarr Array facade — ``CudaZarrArray`` and its factories.

* :class:`CudaZarrArray` — :class:`zarr.Array` subclass returned by the
  factories; basic-indexing reads route through the lowlevel fast path
  (a cached :class:`czarr.core.Array`), everything else falls back to
  zarr's machinery.
* :func:`open_cuda_array` — open an existing array path and wrap it.
* :func:`create_cuda_array` — create + wrap in one call.

The factories mirror :func:`zarr.open_array` / :func:`zarr.create_array`
signatures where possible and wrap string/``Path`` stores in
:class:`czarr.GPULocalStore`, so the cuFile read path is the default.
The fallback read path's return type follows the active buffer prototype
(``cupy`` after :func:`czarr.configure_gpu`, numpy otherwise).
"""

from pathlib import Path
from typing import Any, Literal, Self, override

import numpy as np
import numpy.typing as npt
import zarr
import zarr.abc.store
from zarr.storage import LocalStore

from czarr.core.array import Array as CoreArray
from czarr.storage import GPULocalStore

__all__ = ["CudaZarrArray", "create_cuda_array", "open_cuda_array"]

# ``zarr.Array`` is a frozen dataclass, so the memo lives in __dict__
# directly: (metadata_object, CoreArray | None).  ``None`` records
# "lowlevel can't serve this metadata" so the plan build isn't retried
# on every read; a new metadata object (resize, reopen) invalidates.
_FAST_CACHE = "_czarr_fast_array"


def _int_axes(selection: Any, ndim: int) -> tuple[int, ...]:
    """Axes selected by a plain integer — squeezed for numpy semantics.

    Mirrors :func:`czarr.lowlevel.plan.normalize_selection`'s padding
    rules; only called after that normalization has already accepted the
    selection, so no validation here.
    """
    items: list[Any] = list(selection) if isinstance(selection, tuple) else [selection]
    if sum(1 for it in items if it is Ellipsis) == 1:
        at = items.index(Ellipsis)
        items[at : at + 1] = [slice(None)] * (ndim - (len(items) - 1))
    return tuple(
        axis
        for axis, it in enumerate(items[:ndim])
        if isinstance(it, int | np.integer) and not isinstance(it, bool | np.bool_)
    )


class CudaZarrArray(zarr.Array):
    """:class:`zarr.Array` returned by the czarr factories.

    Basic-indexing reads on supported arrays return
    :class:`cupy.ndarray` via the lowlevel fast path (0-d for full-int
    selections, matching cupy semantics).  Reads the fast path cannot
    serve fall back to :meth:`zarr.Array.__getitem__`, whose return
    type follows the active buffer prototype — run
    :func:`czarr.configure_gpu` for uniformly-GPU results.
    """

    @classmethod
    def wrap(cls, array: zarr.Array, /) -> Self:
        """Wrap an existing :class:`zarr.Array`.

        The preferred constructor — re-uses the existing array's
        metadata, store, and codec chain.  No second open.
        """
        return cls(array._async_array)

    def _fast_array(self) -> CoreArray | None:
        """The cached :class:`czarr.core.Array`, or ``None`` when out of scope."""
        meta = self._async_array.metadata
        cached = self.__dict__.get(_FAST_CACHE)
        if cached is not None and cached[0] is meta:
            return cached[1]
        core: CoreArray | None = None
        store = self.store_path.store
        if isinstance(store, LocalStore) and meta.zarr_format == 3:
            try:
                core = CoreArray.from_metadata(meta.to_dict(), Path(store.root) / self.path)
            except (NotImplementedError, ValueError):
                core = None  # metadata outside lowlevel scope — zarr serves all reads
        self.__dict__[_FAST_CACHE] = (meta, core)
        return core

    @override
    def __getitem__(self, selection: Any) -> Any:
        """Read ``selection`` — lowlevel fast path first, zarr fallback second."""
        core = self._fast_array()
        if core is not None:
            try:
                out = core.retrieve_array_subset(selection)
            except (NotImplementedError, TypeError, IndexError):
                pass  # non-basic selection or v1 decode gap — zarr handles it
            else:
                axes = _int_axes(selection, core.ndim)
                return out.squeeze(axis=axes) if axes else out
        return super().__getitem__(selection)

    @override
    def __setitem__(self, selection: Any, value: Any) -> None:
        """Write through zarr; drop the cached plan (shard indexes may change)."""
        self.__dict__.pop(_FAST_CACHE, None)
        super().__setitem__(selection, value)


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
) -> CudaZarrArray:
    """Open an existing array and wrap it as :class:`CudaZarrArray`.

    A string or :class:`pathlib.Path` is automatically wrapped in
    :class:`czarr.GPULocalStore` so the cuFile read path is the default;
    pass a :class:`zarr.abc.store.Store` instance to opt into a different
    backend.
    """
    resolved = _resolve_store(store, mode=mode)
    arr = zarr.open_array(store=resolved, path=path or "", mode=mode)
    return CudaZarrArray.wrap(arr)


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
    return CudaZarrArray.wrap(arr)
