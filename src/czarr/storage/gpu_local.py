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
    from collections.abc import Sequence
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
    """Synchronous cuFile read into a fresh GPU buffer."""
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


def _gds_get_many_sync(
    paths: list[Path],
    prototype: BufferPrototype,
    byte_ranges: list[ByteRequest | None],
) -> list[Buffer | None]:
    """Allocate per-key device buffers + one batched cuFile read.

    Returns ``None`` for any path that doesn't exist; matches
    :func:`_gds_get_sync` semantics per slot.
    """
    # Resolve every (offset, size); also detect missing files up-front so
    # we don't half-set the cuFile batch and then have to unwind.
    requests: list[tuple[Path, int, int, int] | None] = []
    sizes: list[int] = []
    for path, br in zip(paths, byte_ranges, strict=True):
        try:
            st = os.stat(path)
        except FileNotFoundError:
            requests.append(None)
            sizes.append(-1)
            continue
        offset, size = _resolve_byte_range(br, st.st_size)
        requests.append((path, offset, size, st.st_size))
        sizes.append(size)

    # Per-key device buffers — empty for zero-length and missing slots so
    # downstream code can still index by position.
    bufs: list[Buffer | None] = []
    devs: list[cp.ndarray | None] = []
    for size, req in zip(sizes, requests, strict=True):
        if req is None:
            bufs.append(None)
            devs.append(None)
        elif size == 0:
            bufs.append(prototype.buffer.create_zero_length())
            devs.append(None)
        else:
            dev = cp.empty(size, dtype=cp.uint8)
            devs.append(dev)
            bufs.append(prototype.buffer.from_array_like(dev))

    # Build the flat batched-read request list (skip missing slots).
    submit: list[tuple[Path, int, int, int]] = []
    submit_indices: list[int] = []
    for i, (size, req, dev) in enumerate(zip(sizes, requests, devs, strict=True)):
        if req is None or size <= 0 or dev is None:
            continue
        path, offset, _size, _file_size = req
        submit.append((path, int(dev.data.ptr), size, offset))
        submit_indices.append(i)

    if not submit:
        return bufs

    with nvtx_range("czarr.GPULocalStore.cufile_batched_read", n=len(submit)):
        results = cufile_runtime.read_into_many(submit)

    # If cuFile returned a short count for any slot, truncate the cupy
    # buffer (matches the per-slot logic in _gds_get_sync).
    for slot_idx, n in zip(submit_indices, results, strict=True):
        if n != sizes[slot_idx]:
            dev = devs[slot_idx][:n]
            bufs[slot_idx] = prototype.buffer.from_array_like(dev)

    return bufs


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

    async def get_many(
        self,
        keys: Sequence[str],
        prototype: BufferPrototype | None = None,
        byte_ranges: Sequence[ByteRequest | None] | None = None,
    ) -> list[Buffer | None]:
        """Batched read of many keys via a single cuFile-handle setup pass.

        Falls back to a list of :meth:`get` calls when prototype is host
        or cuFile is unavailable.  When the GPU prototype is requested,
        all device buffers are allocated up-front, then one batched
        :func:`czarr.storage.cufile_runtime.read_into_many` call services
        every read with pre-registered handles — eliminating the per-key
        ``open + register + deregister + close`` cycle that dominates
        small-chunk workloads (see Phase 0 measurements).

        Missing files come back as ``None`` in their slot (matching
        :meth:`get`).  Caller is responsible for slicing if any byte
        range is partial.
        """
        if prototype is None:
            prototype = default_buffer_prototype()
        if not self._is_open:
            await self._open()
        if byte_ranges is None:
            byte_ranges = [None] * len(keys)
        if len(byte_ranges) != len(keys):
            raise ValueError(f"byte_ranges length {len(byte_ranges)} != keys length {len(keys)}")
        if not self._gds_available or not _gpu_prototype_requested(prototype):
            # Per-key fan-out via asyncio.gather — same as zarr's default.
            return list(
                await asyncio.gather(*(self.get(k, prototype, br) for k, br in zip(keys, byte_ranges, strict=True)))
            )
        return await asyncio.to_thread(
            _gds_get_many_sync,
            [self.root / k for k in keys],
            prototype,
            list(byte_ranges),
        )

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
