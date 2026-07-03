"""cupy-backed Zarr buffers registered as an opt-in prototype.

`CzarrGpuBuffer` and `CzarrGpuNDBuffer` are drop-in replacements for
``zarr.core.buffer.gpu.Buffer`` / ``...gpu.NDBuffer``.  They are
registered with zarr's registry at import but NOT wired by default —
``configure_gpu`` selects zarr's stock gpu buffers; opt in via
``zarr.config.set({"buffer": "czarr.core.buffer.CzarrGpuBuffer", ...})``.

History: these classes previously allocated through ``cuda.core``'s
``VirtualMemoryResource`` (4 KiB-aligned, GPU-direct-RDMA-tagged) with a
register-once ``CuFileSlabPool`` on top.  That architecture was benched
~5% SLOWER than stock cupy allocation on real GDS (H100/H200), so the
slab pool and VMR allocator were deleted — git history has both if a
register-once retry ever becomes evidence-backed.

``__cuda_array_interface__`` is synthesised on the wrapper so nvCOMP /
Zarr code that consumes CAI can wrap the buffer zero-copy without going
through the cupy view first.
"""

from collections.abc import Iterable
from typing import Any, Literal, Self, cast

import cupy as cp
import numpy as np
import numpy.typing as npt
from zarr.core.buffer import core
from zarr.core.buffer import gpu as gpu_buffer
from zarr.core.buffer.core import ArrayLike, BufferPrototype, NDArrayLike
from zarr.core.common import BytesLike
from zarr.registry import register_buffer, register_ndbuffer


class CzarrGpuBuffer(core.Buffer):
    """A flat byte buffer over device memory (cupy-backed)."""

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

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def empty(cls, size: int) -> Self:
        """Allocate a ``size``-byte device buffer via ``cp.empty``.

        Same allocation path production uses (``GPULocalStore`` reads
        into plain ``cp.empty``); cuFile registers pointers internally
        on first use.
        """
        if size < 0:
            raise ValueError(f"size must be non-negative, got {size}")
        if size == 0:
            return cls.create_zero_length()
        return cls(cp.empty(size, dtype=cp.uint8))

    @classmethod
    def create_zero_length(cls) -> Self:
        """Empty 0-byte buffer — no device allocation."""
        return cls(cp.empty(0, dtype=cp.uint8))

    @classmethod
    def from_array_like(cls, array_like: ArrayLike) -> Self:
        """Wrap a CAI- or DLPack-compatible array zero-copy."""
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
        parts.extend(cp.asarray(other.as_array_like()).view(cp.uint8) for other in others)
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
        """Raw device pointer; slice-views inherit the parent's pointer plus offset."""
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


# Canonical "is this GPU-resident" predicate — CzarrGpuBuffer subclasses
# zarr's core.Buffer, not gpu.Buffer, so a plain gpu.Buffer issubclass
# check would miss it.  Import these instead of re-deriving the tuple.
GPU_BUFFER_TYPES: tuple[type, ...] = (gpu_buffer.Buffer, CzarrGpuBuffer)


def is_gpu_prototype(prototype: BufferPrototype) -> bool:
    """True when ``prototype.buffer`` is a device-resident Buffer class."""
    return issubclass(prototype.buffer, GPU_BUFFER_TYPES)


def is_gpu_buffer(obj: object) -> bool:
    """True when ``obj`` is a device-resident Buffer instance."""
    return isinstance(obj, GPU_BUFFER_TYPES)
