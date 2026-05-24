"""``CudaZarrArray`` — :class:`zarr.Array` subclass with GPU fast path.

Mirrors the pattern from zarrs-python PR #147 (``ZarrsArray``): subclass
:class:`zarr.Array`, intercept :meth:`__getitem__` on the basic-indexing
fast path, fall through to :meth:`super().__getitem__` for advanced
indexing.

The fast path returns a :class:`cupy.ndarray` on device.  The fallback
returns whatever :meth:`zarr.Array.__getitem__` produces (typically
:class:`numpy.ndarray` on host unless a GPU buffer prototype is active
globally).  The dual return-type contract is intentional and documented
in the public docstring; callers that need consistency wrap the result
in :func:`cupy.asarray`.

Construction is through :meth:`CudaZarrArray.wrap` or the
:func:`open_cuda_array` / :func:`create_cuda_array` factories — the
constructor exists for parity with :class:`zarr.Array` but is not the
documented path.
"""

from __future__ import annotations

from types import EllipsisType
from typing import TYPE_CHECKING, TypedDict, TypeGuard, Unpack, override

import zarr

from czarr.array._impl import _CudaArrayImpl

if TYPE_CHECKING:
    from typing import Any

    import cupy as cp
    import numpy as np

# PEP 695 type aliases for indexing.  Kept local to this module — the
# public ``czarr.types`` namespace can re-export later when the project
# has multiple call sites.
type _SimpleKey = int | slice | EllipsisType
type _TupleKey = tuple[_SimpleKey, ...]
type BasicKey = _SimpleKey | _TupleKey


class CudaZarrArrayKwargs(TypedDict, total=False):
    """Tuning knobs for the per-array orchestrator.

    Mirrors :class:`czarr.array._impl._ImplKwargs`.
    """

    queue_depth: int
    stream_pool_size: int
    microbatch_size: int


def _is_basic_indexing(key: object, /) -> TypeGuard[BasicKey]:
    """Return ``True`` for keys our fast path handles.

    The fast path covers ``int``, contiguous ``slice`` (step 1 or
    ``None``), single ``Ellipsis``, and tuples thereof.  Anything else
    (bool masks, ndarray indices, structured-dtype fields, multi-axis
    advanced indexing) falls through to :meth:`zarr.Array.__getitem__`.
    """
    if isinstance(key, int | EllipsisType):
        return True
    if isinstance(key, slice):
        return key.step is None or key.step == 1
    if isinstance(key, tuple):
        seen_ellipsis = False
        for k in key:
            if isinstance(k, int):
                continue
            if isinstance(k, slice):
                if k.step is None or k.step == 1:
                    continue
                return False
            if isinstance(k, EllipsisType):
                if seen_ellipsis:
                    return False
                seen_ellipsis = True
                continue
            return False
        return True
    return False


class CudaZarrArray(zarr.Array):
    """zarr.Array subclass with a GPU-resident basic-indexing fast path.

    Public surface beyond :class:`zarr.Array`:

    * ``arr[basic_key]`` returns :class:`cupy.ndarray` on device when a
      GPU buffer prototype is configured globally (via
      :func:`czarr.configure_gpu`).  Without that configuration the
      fast path still executes, but the return type follows the active
      prototype.
    * ``arr[advanced_key]`` falls through to :meth:`super().__getitem__`
      and returns whatever the parent class returns (typically
      :class:`numpy.ndarray` on host).
    * :meth:`wrap` is the documented constructor — takes an existing
      opened :class:`zarr.Array` and wraps it with the orchestrator.

    The dual return-type contract is intentional.  Callers that need
    consistency wrap the basic-path result in :func:`cupy.asarray`.
    """

    @classmethod
    def wrap(
        cls,
        array: zarr.Array,
        /,
        **kwargs: Unpack[CudaZarrArrayKwargs],
    ) -> CudaZarrArray:
        """Wrap an existing :class:`zarr.Array` with a GPU fast path.

        The preferred constructor — re-uses the existing array's
        metadata, store, and codec chain.  No second open.
        """
        instance = cls(array._async_array)
        # frozen-style assign: dataclass field on the implementation,
        # plain attribute on the wrapper.
        object.__setattr__(instance, "_impl", _CudaArrayImpl.from_array(instance, **kwargs))
        return instance

    @property
    def _orchestrator(self) -> _CudaArrayImpl:
        """The private :class:`_CudaArrayImpl` instance.

        Created lazily on first access for arrays constructed directly
        through the parent constructor (without going through
        :meth:`wrap`).  Tests can inject a mock here.
        """
        impl = getattr(self, "_impl", None)
        if impl is None:
            impl = _CudaArrayImpl.from_array(self)
            object.__setattr__(self, "_impl", impl)
        return impl

    @override
    def __getitem__(self, key: Any, /) -> cp.ndarray | np.ndarray:  # type: ignore[override]
        """Read a selection.

        Basic indexing (int / step-1 slice / Ellipsis and tuples thereof)
        routes through the GPU fast path and returns
        :class:`cupy.ndarray` on device when a GPU prototype is
        configured.  Advanced indexing falls through to
        :meth:`zarr.Array.__getitem__`.
        """
        if _is_basic_indexing(key):
            return self._orchestrator.retrieve_gpu(key)
        return super().__getitem__(key)
