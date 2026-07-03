"""GPU-aware local-filesystem store: cuFile reads/writes when prototype is GPU."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

import cupy as cp
from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest
from zarr.core.buffer import default_buffer_prototype
from zarr.core.buffer import gpu as gpu_buffer
from zarr.storage import LocalStore

from czarr._nvtx import nvtx_range
from czarr.storage import cufile_runtime

if TYPE_CHECKING:
    from pathlib import Path

    from zarr.abc.store import ByteRequest
    from zarr.core.buffer import Buffer, BufferPrototype


def _gpu_prototype_requested(prototype: BufferPrototype) -> bool:
    from czarr.core.buffer import CzarrGpuBuffer

    return issubclass(prototype.buffer, (gpu_buffer.Buffer, CzarrGpuBuffer))


def _resolve_byte_range(byte_range: ByteRequest | None, file_size: int) -> tuple[int, int]:
    """Translate Zarr's byte-range descriptor into ``(offset, size)``."""
    if byte_range is None:
        return 0, file_size
    if isinstance(byte_range, RangeByteRequest):
        return byte_range.start, max(0, byte_range.end - byte_range.start)
    if isinstance(byte_range, OffsetByteRequest):
        return byte_range.offset, max(0, file_size - byte_range.offset)
    if isinstance(byte_range, SuffixByteRequest):
        return max(0, file_size - byte_range.suffix), min(byte_range.suffix, file_size)
    raise TypeError(f"unsupported byte-range type: {type(byte_range)!r}")


def _gds_get_sync(path: Path, prototype: BufferPrototype, byte_range: ByteRequest | None) -> Buffer | None:
    """Synchronous cuFile read into a fresh GPU buffer.

    Allocates via plain ``cp.empty`` regardless of prototype; cuFile
    handles registration internally on first use.  An earlier Phase 3
    iteration routed CzarrGpuBuffer prototypes through
    ``CzarrGpuBuffer.empty`` (VMR-aligned + pre-registered with cuFile)
    but VMR's per-allocation cost (cuMemCreate / cuMemAddressReserve /
    cuMemMap + 2 MiB granularity) dominates at ~5 ms per chunk — net
    5× regression on the H200 slice_compare workload.  The
    register-once architecture only pays off with a pre-allocated
    buffer pool (callers reuse one big registered slab across many
    reads); without that pool the VMR allocation cost outweighs the
    cuFile-register savings (which are also negligible in compat-mode
    cuFile on VAST/Lustre).  Tracked as a follow-up to v0.1.
    """
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    offset, size = _resolve_byte_range(byte_range, st.st_size)
    if size == 0:
        return prototype.buffer.create_zero_length()
    with nvtx_range("czarr.GPULocalStore.cufile_read", size=size):
        dev = cp.empty(size, dtype=cp.uint8)
        n = cufile_runtime.read_into(path, int(dev.data.ptr), size, offset)
    if n != size:
        # Truncate to what was actually read; cuFile returns a short count
        # only at EOF or hardware error.
        dev = dev[:n]
    return prototype.buffer.from_array_like(dev)


def _gds_set_sync(path: Path, value: Buffer) -> None:
    """Synchronous cuFile write from a GPU buffer (creates/overwrites file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = value.as_array_like()  # cupy.ndarray for gpu.Buffer
    nbytes = int(arr.nbytes)
    if nbytes == 0:
        # cuFile cannot write zero bytes; touch the file instead.
        path.touch()
        return
    with nvtx_range("czarr.GPULocalStore.cufile_write", size=nbytes):
        cufile_runtime.write_from(path, int(arr.data.ptr), nbytes, 0)


class GPULocalStore(LocalStore):
    """Local-filesystem store that reads/writes GPU buffers via cuFile when possible.

    Falls back to :class:`zarr.storage.LocalStore` semantics when the requested
    prototype is host-side, or when cuFile is unavailable on this host.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        read_only: bool = False,
        force_gpu: bool = False,
    ) -> None:
        super().__init__(root, read_only=read_only)
        self._gds_available = cufile_runtime.is_available()
        if force_gpu and not self._gds_available:
            raise RuntimeError("cuFile is not available on this host but force_gpu=True was requested.")

    @property
    def gds_available(self) -> bool:
        """Whether cuFile is usable in this process (real GDS or compat mode)."""
        return self._gds_available

    async def get(
        self,
        key: str,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        """Read a key into a Buffer; uses cuFile when prototype is GPU."""
        if prototype is None:
            prototype = default_buffer_prototype()
        if not self._is_open:
            await self._open()
        if not self._gds_available or not _gpu_prototype_requested(prototype):
            return await super().get(key, prototype, byte_range)
        path = self.root / key
        try:
            return await asyncio.to_thread(_gds_get_sync, path, prototype, byte_range)
        except FileNotFoundError:
            return None

    async def set(self, key: str, value: Buffer) -> None:
        """Write a Buffer to a key; uses cuFile when value is a gpu Buffer."""
        self._check_writable()
        if not self._is_open:
            await self._open()
        if not self._gds_available or not isinstance(value, gpu_buffer.Buffer):
            await super().set(key, value)
            return
        path = self.root / key
        await asyncio.to_thread(_gds_set_sync, path, value)
