"""Conversions between Zarr ``Buffer``, device arrays, and ``nvcomp.Array``."""

import cupy as cp
import numpy as np
from nvidia import nvcomp
from zarr.core.buffer import Buffer, BufferPrototype

from czarr.core.buffer import is_gpu_prototype

# nvCOMP's batched encode/decode kernels read with vector loads that need
# at least 16-byte alignment of the source pointer.  RMM/cupy pools usually
# return 256-aligned blocks, but the Zarr sharding codec hands us slices of
# a larger device buffer whose offset is not necessarily 16-aligned, which
# causes ``cudaErrorMisalignedAddress`` deep inside nvCOMP.  We sniff the
# pointer and force a fresh contiguous copy when we'd otherwise feed nvCOMP
# something it can't safely vectorise.
_NVCOMP_ALIGNMENT = 16


def _ensure_aligned(arr: cp.ndarray) -> cp.ndarray:
    if int(arr.data.ptr) % _NVCOMP_ALIGNMENT == 0 and arr.flags.c_contiguous:
        return arr
    fresh = cp.empty(arr.nbytes, dtype=cp.uint8)
    fresh[:] = cp.ascontiguousarray(arr).view(cp.uint8)
    return fresh


def buffer_to_nvarray(buffer: Buffer) -> nvcomp.Array:
    """Wrap a Zarr ``Buffer`` as an ``nvcomp.Array`` on the GPU.

    Zero-copy when ``buffer`` is already device-resident and 16-byte aligned;
    otherwise either uploads from host or makes a fresh aligned copy.
    """
    backing = buffer.as_array_like()
    if isinstance(backing, cp.ndarray):
        backing = _ensure_aligned(backing)
    nv = nvcomp.as_array(backing)
    if nv.buffer_kind.name.startswith("STRIDED_HOST"):
        nv = nv.cuda()
    return nv


def nvarray_to_buffer(nv: nvcomp.Array, prototype: BufferPrototype) -> Buffer:
    """Wrap an ``nvcomp.Array`` as a Zarr ``Buffer`` matching ``prototype``.

    With a GPU prototype the result is a zero-copy ``cupy`` view of the device
    payload (via ``__cuda_array_interface__``). With a host prototype the bytes
    are downloaded with ``nv.cpu()`` and wrapped as the prototype's buffer.
    """
    if is_gpu_prototype(prototype):
        device = cp.asarray(nv).view(cp.uint8)
        return prototype.buffer.from_array_like(device)
    host = np.ascontiguousarray(np.asarray(nv.cpu())).view(np.uint8)
    return prototype.buffer.from_bytes(host.tobytes())


def device_to_buffer(dev: cp.ndarray, prototype: BufferPrototype) -> Buffer:
    """Wrap a decoded device array as a Zarr ``Buffer`` matching ``prototype``.

    Zero-copy for GPU prototypes; D2H download + bytes copy for host ones.
    """
    if is_gpu_prototype(prototype):
        return prototype.buffer.from_array_like(dev)
    return prototype.buffer.from_bytes(cp.asnumpy(dev).tobytes())
