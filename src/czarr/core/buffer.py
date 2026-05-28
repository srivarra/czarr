"""``cuda.core.Buffer``-backed Zarr buffers.

`CzarrGpuBuffer` and `CzarrGpuNDBuffer` are drop-in replacements for
``zarr.core.buffer.gpu.Buffer`` / ``...gpu.NDBuffer`` that allocate
device memory through ``cuda.core``'s
``VirtualMemoryResource(addr_align=4096, gpu_direct_rdma=True)`` so the
underlying device pointer is 4 KiB-aligned and tagged GPU-direct-RDMA.
That is exactly what cuFile direct I/O wants, and it gives us a clean
substrate for Phase 3 (register-once cuFile).

The wrapper holds a ``cupy.ndarray`` view that was imported zero-copy
from the producer via DLPack. cupy's DLPack import takes a reference to
the producer's deleter, so the lifetime of the underlying
``cuda.core.Buffer`` follows the cupy view automatically — we never
call ``close()`` directly, which keeps slicing safe.

``__cuda_array_interface__`` is synthesised on the wrapper so nvCOMP /
Zarr code that consumes CAI can wrap the buffer zero-copy without going
through the cupy view first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import cupy as cp
import numpy as np
import numpy.typing as npt
from cuda.core import (
    Device,
    VirtualMemoryResource,
    VirtualMemoryResourceOptions,
)
from zarr.core.buffer import core
from zarr.core.buffer.core import ArrayLike, BufferPrototype, NDArrayLike
from zarr.registry import register_buffer, register_ndbuffer

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Self

    from cuda.core import Stream
    from zarr.core.common import BytesLike


_DEVICE_MR: VirtualMemoryResource | None = None


def _device_mr() -> VirtualMemoryResource:
    """Lazy singleton for the 4 KiB-aligned device allocator.

    ``VirtualMemoryResource`` is the only ``cuda.core`` allocator that
    delivers a fresh 4 KiB-aligned virtual address per call (verified
    in ``bench/buffer/alignment_probe.py``). The cost is granularity:
    every allocation pads up to the device's VMM granularity (commonly
    2 MiB on Hopper / Ampere). Phase 5 bench will tell us whether we
    want to slab + sub-allocate on top.
    """
    global _DEVICE_MR
    if _DEVICE_MR is None:
        dev = Device()
        dev.set_current()
        _DEVICE_MR = VirtualMemoryResource(
            dev,
            VirtualMemoryResourceOptions(addr_align=4096, gpu_direct_rdma=True),
        )
    return _DEVICE_MR


def _current_stream() -> Stream:
    return Device().default_stream


class CzarrGpuBuffer(core.Buffer):
    """A flat byte buffer backed by a ``cuda.core.Buffer``.

    For freshly-allocated buffers the underlying device pointer is
    4 KiB-aligned and tagged GPU-direct-RDMA. Slices and combined
    buffers share or copy from those allocations and are not
    necessarily aligned themselves; check ``device_ptr % 4096`` if you
    plan to feed a slice to cuFile.
    """

    def __init__(self, array_like: ArrayLike) -> None:
        # The ABC requires us to accept a 1-D byte array_like. We honour
        # that for compatibility with code that constructs the buffer
        # directly from a cupy / numpy slice; the primary entry point
        # callers should use is ``CzarrGpuBuffer.empty``.
        arr = cp.asarray(array_like)
        if arr.ndim != 1:
            raise ValueError("CzarrGpuBuffer: only 1-dim array_like allowed")
        if arr.dtype != np.dtype("B") and arr.dtype != np.dtype("int8"):
            raise ValueError(f"CzarrGpuBuffer: only byte dtype allowed, got {arr.dtype}")
        # Always present as uint8 so downstream code sees one dtype.
        self._data: cp.ndarray = arr.view(cp.uint8)
        # Set by :meth:`empty` to the slab sub-region backing this buffer.
        # Holding it keeps the region reserved; dropping the buffer
        # returns the region to the slab free-list.  ``None`` for buffers
        # that wrap external memory (``from_array_like``).  cuFile
        # registration is owned by the slab, not the individual buffer.
        self._slab_alloc: object | None = None

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def empty(cls, size: int, *, stream: Stream | None = None) -> Self:
        """Allocate a ``size``-byte buffer from the register-once slab pool.

        Sub-allocates a 4 KiB-aligned region from the process-global
        :class:`czarr.core.slab.CuFileSlabPool`.  The slab — not this
        buffer — owns the cuFile ``buf_register``, so allocation here is
        a free-list pop with no per-chunk VMR ``cuMemCreate``/``cuMemMap``
        and no per-chunk cuFile registration.  That is the fix for the
        ~5x per-chunk regression the naive ``VirtualMemoryResource``
        allocate-per-call path showed on the H200 slice_compare bench.

        The returned buffer holds its :class:`SlabAllocation` alive; when
        the buffer is GC'd the region returns to the slab's free-list for
        reuse.  Slices via :meth:`__getitem__` share the region and keep
        it pinned for their own lifetime (cupy view refcounting).
        """
        if size < 0:
            raise ValueError(f"size must be non-negative, got {size}")
        if size == 0:
            return cls.create_zero_length()
        from czarr.core.slab import get_default_slab_pool

        alloc = get_default_slab_pool().allocate(size, stream=stream)
        instance = cls(alloc.array)
        # Keep the sub-region reserved for this buffer's lifetime.
        instance._slab_alloc = alloc
        return instance

    @classmethod
    def create_zero_length(cls) -> Self:
        """Empty 0-byte buffer — no device allocation."""
        return cls(cp.empty(0, dtype=cp.uint8))

    @classmethod
    def from_array_like(cls, array_like: ArrayLike) -> Self:
        """Wrap a CAI- or DLPack-compatible array zero-copy.

        The 4 KiB-aligned + RDMA guarantee only holds for buffers
        produced by :meth:`empty` / :meth:`from_bytes`. Callers that
        need the alignment guarantee should allocate explicitly and
        copy in; wrapping someone else's pointer keeps their alignment.
        """
        src = cp.asarray(array_like).view(cp.uint8).ravel()
        return cls(src)

    @classmethod
    def from_buffer(cls, buffer: core.Buffer) -> Self:
        """Wrap or copy an arbitrary zarr Buffer; zero-copy when already ours."""
        if isinstance(buffer, cls):
            return buffer
        return cls.from_array_like(buffer.as_array_like())

    @classmethod
    def from_bytes(cls, bytes_like: BytesLike) -> Self:
        """Copy host bytes into a fresh aligned device buffer."""
        host = np.frombuffer(bytes_like, dtype=np.uint8)
        if host.size == 0:
            return cls.create_zero_length()
        out = cls.empty(int(host.size))
        out._data.set(host)
        return out

    # ------------------------------------------------------------------
    # zarr Buffer protocol
    # ------------------------------------------------------------------

    def as_numpy_array(self) -> npt.NDArray[Any]:
        """Copy device bytes to a fresh numpy array."""
        return cast("npt.NDArray[Any]", cp.asnumpy(self._data))

    def combine(self, others: Iterable[core.Buffer]) -> Self:
        """Concatenate self + ``others`` into a fresh aligned device buffer."""
        parts = [self._data]
        for other in others:
            parts.append(cp.asarray(other.as_array_like()).view(cp.uint8))
        total = int(sum(p.size for p in parts))
        out = type(self).empty(total)
        offset = 0
        for p in parts:
            n = int(p.size)
            out._data[offset : offset + n] = p
            offset += n
        return out

    # ------------------------------------------------------------------
    # cuda.core extras
    # ------------------------------------------------------------------

    @property
    def device_ptr(self) -> int:
        """Raw device pointer.

        For buffers allocated via ``empty`` / ``from_bytes`` this is the
        slab sub-region's 4 KiB-aligned pointer.  Slice-views inherit the
        parent's pointer plus the slice offset.
        """
        if self._data.size == 0:
            return 0
        return int(self._data.data.ptr)

    @property
    def is_device_accessible(self) -> bool:
        """True when the buffer holds device memory we can read from the GPU."""
        return self._data.size > 0

    @property
    def is_host_accessible(self) -> bool:
        """Always False — device path; pinned host buffers come in a follow-up."""
        return False

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]:
        """Synthesise CAI v3 over the logical byte range.

        ``cuda.core.Buffer`` does not expose CAI natively; we build it
        from the cupy view's pointer + size so that nvCOMP and other
        CAI consumers can wrap the buffer directly.
        """
        return {
            "version": 3,
            "shape": (int(self._data.size),),
            "typestr": "|u1",
            "data": (self.device_ptr, False),
            "strides": None,
            "stream": None,
        }


class CzarrGpuNDBuffer(core.NDBuffer):
    """n-dimensional GPU buffer.

    Backed by a plain ``cupy.ndarray``. NDBuffers carry decoded array
    payloads and don't see cuFile, so the 4 KiB-alignment / RDMA flag
    matters only for the flat ``CzarrGpuBuffer`` that the cuFile path
    writes into.
    """

    def __init__(self, array: NDArrayLike) -> None:
        if array.dtype == object:
            raise ValueError("CzarrGpuNDBuffer: object dtype not supported")
        self._data: NDArrayLike = cp.asarray(array)

    @classmethod
    def create(
        cls,
        *,
        shape: Iterable[int],
        dtype: npt.DTypeLike,
        order: Literal["C", "F"] = "C",
        fill_value: Any | None = None,
    ) -> Self:
        """New cupy-backed ndbuffer; ``fill_value`` is applied when not None."""
        arr = cp.empty(shape=tuple(shape), dtype=dtype, order=order)
        if fill_value is not None:
            arr.fill(fill_value)
        return cls(arr)

    @classmethod
    def empty(cls, shape: tuple[int, ...], dtype: npt.DTypeLike, order: Literal["C", "F"] = "C") -> Self:
        """Uninitialised cupy-backed ndbuffer."""
        return cls(cp.empty(shape=shape, dtype=dtype, order=order))

    @classmethod
    def from_numpy_array(cls, array_like: npt.ArrayLike) -> Self:
        """H2D copy via ``cp.asarray``."""
        return cls(cp.asarray(array_like))

    @classmethod
    def from_ndarray_like(cls, ndarray_like: NDArrayLike) -> Self:
        """Wrap / coerce an existing ndarray-like to cupy."""
        return cls(cp.asarray(ndarray_like))

    def as_numpy_array(self) -> npt.NDArray[Any]:
        """Copy device data back to numpy."""
        return cast("npt.NDArray[Any]", cp.asnumpy(self._data))

    def __getitem__(self, key: Any) -> Self:
        return type(self)(self._data.__getitem__(key))

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(value, CzarrGpuNDBuffer):
            value = value._data
        elif isinstance(value, core.NDBuffer):
            value = cp.asarray(value.as_ndarray_like())
        self._data.__setitem__(key, value)


buffer_prototype = BufferPrototype(buffer=CzarrGpuBuffer, nd_buffer=CzarrGpuNDBuffer)

register_buffer(CzarrGpuBuffer, qualname="czarr.core.buffer.CzarrGpuBuffer")
register_ndbuffer(CzarrGpuNDBuffer, qualname="czarr.core.buffer.CzarrGpuNDBuffer")
