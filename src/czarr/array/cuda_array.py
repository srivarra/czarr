"""``CudaZarrArray`` — :class:`zarr.Array` with a lowlevel GPU fast path.

Reads that ``czarr.lowlevel`` can serve — basic indexing on a
local-store zarr v3 array whose codec chain is in the v1 decode scope —
route through a cached :class:`czarr.core.Array` (derive-once metadata,
coalesced cuFile reads, one batched GPU decode) and return
``cupy.ndarray``.  Everything else falls back to zarr's own machinery
unchanged: fancy indexing, unsupported codecs, non-local stores, and
all writes.  The fallback's return type follows the active buffer
prototype (``cupy`` after :func:`czarr.configure_gpu`, numpy otherwise).

Construction is through :meth:`CudaZarrArray.wrap` or the factories —
the constructor exists for parity with :class:`zarr.Array` but is not
the documented path.
"""

from pathlib import Path
from typing import Any, Self, override

import numpy as np
import zarr
from zarr.storage import LocalStore

from czarr.core.array import Array as CoreArray

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
