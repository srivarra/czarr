"""Phase 1 tests for ``czarr.core.buffer.CzarrGpuBuffer``."""

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

from czarr.core.buffer import (
    CzarrGpuBuffer,
    CzarrGpuNDBuffer,
    buffer_prototype,
)

# ----------------------------------------------------------------------
# CzarrGpuBuffer
# ----------------------------------------------------------------------


def test_empty_buffer_is_4kib_aligned() -> None:
    buf = CzarrGpuBuffer.empty(1024)
    try:
        assert buf.device_ptr != 0
        assert buf.device_ptr % 4096 == 0
        assert len(buf) == 1024
    finally:
        del buf


@pytest.mark.parametrize("size", [1, 16, 4096, 1 << 20, 16 << 20])
def test_alignment_across_sizes(size: int) -> None:
    buf = CzarrGpuBuffer.empty(size)
    try:
        assert buf.device_ptr % 4096 == 0, f"size={size} ptr={hex(buf.device_ptr)}"
        assert len(buf) == size
    finally:
        del buf


def test_zero_length_buffer() -> None:
    buf = CzarrGpuBuffer.create_zero_length()
    assert len(buf) == 0
    assert buf.device_ptr == 0
    assert buf.as_numpy_array().shape == (0,)
    cai = buf.__cuda_array_interface__
    assert cai["shape"] == (0,)
    # Zero-length CAI: pointer is 0, writability is intentionally False
    # since there is nothing to write to.
    ptr, _ = cai["data"]
    assert ptr == 0


def test_memory_class_flags() -> None:
    buf = CzarrGpuBuffer.empty(4096)
    try:
        assert buf.is_device_accessible is True
        assert buf.is_host_accessible is False
    finally:
        del buf


def test_cuda_array_interface_shape_matches_logical_size() -> None:
    """CAI must expose ``size`` (not the rounded-up cuda.core size)."""
    buf = CzarrGpuBuffer.empty(1024)
    try:
        cai = buf.__cuda_array_interface__
        assert cai["version"] == 3
        assert cai["shape"] == (1024,)
        assert cai["typestr"] == "|u1"
        ptr, readonly = cai["data"]
        assert ptr == buf.device_ptr
        assert readonly is False
    finally:
        del buf


def test_from_bytes_round_trip() -> None:
    payload = b"czarr-buffer-roundtrip-payload" * 17
    buf = CzarrGpuBuffer.from_bytes(payload)
    try:
        # device-side data
        host = buf.as_numpy_array()
        assert host.tobytes() == payload
        # to_bytes goes through as_numpy_array
        assert buf.to_bytes() == payload
        assert len(buf) == len(payload)
    finally:
        del buf


def test_from_array_like_copies_aligned() -> None:
    # Build on host then push to device to avoid cupy JIT (the cluster
    # has CUDA 13.1 system libs but CuPy is built against cu12, so any
    # JIT-triggering cupy op fails on libnvrtc.so.12 lookup).
    src = cp.asarray(np.arange(1024, dtype=np.uint8))
    # cupy's default allocator is not 4 KiB-aligned for sub-page slabs,
    # but CzarrGpuBuffer must always be 4 KiB-aligned.
    buf = CzarrGpuBuffer.from_array_like(src)
    try:
        assert buf.device_ptr % 4096 == 0
        assert len(buf) == 1024
        host = buf.as_numpy_array()
        np.testing.assert_array_equal(host, np.arange(1024, dtype=np.uint8))
    finally:
        del buf


def test_as_array_like_returns_cupy_view_same_pointer() -> None:
    buf = CzarrGpuBuffer.empty(4096)
    try:
        arr = buf.as_array_like()
        assert isinstance(arr, cp.ndarray)
        assert int(arr.data.ptr) == buf.device_ptr
        assert arr.dtype == cp.uint8
        assert arr.size == 4096
    finally:
        del buf


def test_slice_via_getitem_keeps_dtype_byte() -> None:
    buf = CzarrGpuBuffer.from_bytes(b"abcdefghij")
    try:
        sl = buf[2:7]
        assert isinstance(sl, CzarrGpuBuffer)
        # Slicing wraps the same _data; ABC guarantees byte dtype.
        assert sl.as_numpy_array().tobytes() == b"cdefg"
    finally:
        del buf


def test_combine_concatenates_payloads() -> None:
    a = CzarrGpuBuffer.from_bytes(b"hello-")
    b = CzarrGpuBuffer.from_bytes(b"world")
    try:
        combined = a.combine([b])
        try:
            assert combined.as_numpy_array().tobytes() == b"hello-world"
            assert combined.device_ptr % 4096 == 0
            assert len(combined) == 11
        finally:
            del combined
    finally:
        del a, b


def test_non_byte_dtype_rejected() -> None:
    """Constructing directly from an int32 cupy array should raise."""
    src = cp.empty(8, dtype=cp.int32)
    with pytest.raises(ValueError, match="byte dtype"):
        CzarrGpuBuffer(src)


def test_non_1d_rejected() -> None:
    """Constructing directly from a 2-D array should raise."""
    src = cp.empty((4, 4), dtype=cp.uint8)
    with pytest.raises(ValueError, match="1-dim"):
        CzarrGpuBuffer(src)


def test_registry_round_trip() -> None:
    """The qualname we register under must round-trip through zarr's registry."""
    import zarr
    from zarr.registry import get_buffer_class, get_ndbuffer_class

    cfg_buffer = zarr.config.get("buffer")
    cfg_ndbuffer = zarr.config.get("ndbuffer")
    try:
        zarr.config.set({"buffer": "czarr.core.buffer.CzarrGpuBuffer"})
        zarr.config.set({"ndbuffer": "czarr.core.buffer.CzarrGpuNDBuffer"})
        assert get_buffer_class() is CzarrGpuBuffer
        assert get_ndbuffer_class() is CzarrGpuNDBuffer
    finally:
        zarr.config.set({"buffer": cfg_buffer})
        zarr.config.set({"ndbuffer": cfg_ndbuffer})


# ----------------------------------------------------------------------
# CzarrGpuNDBuffer
# ----------------------------------------------------------------------


def test_ndbuffer_create_shape_dtype() -> None:
    # Skip fill_value here — cupy's elementwise fill JITs through nvrtc,
    # and the cluster has CUDA 13 system libs vs CuPy cu12. The fill
    # path itself is exercised by the codec tests on a JIT-warm worker.
    nd = CzarrGpuNDBuffer.create(shape=(4, 8), dtype="float32")
    assert nd.shape == (4, 8)
    assert nd.dtype == np.float32


def test_ndbuffer_from_numpy_array() -> None:
    src = np.arange(24, dtype=np.int16).reshape(4, 6)
    nd = CzarrGpuNDBuffer.from_numpy_array(src)
    np.testing.assert_array_equal(nd.as_numpy_array(), src)


def test_ndbuffer_setitem_from_ndbuffer() -> None:
    nd = CzarrGpuNDBuffer.empty(shape=(4,), dtype="int32")
    nd[:] = CzarrGpuNDBuffer.from_numpy_array(np.arange(4, dtype=np.int32))
    np.testing.assert_array_equal(nd.as_numpy_array(), np.arange(4, dtype=np.int32))


# ----------------------------------------------------------------------
# buffer_prototype
# ----------------------------------------------------------------------


def test_buffer_prototype_exports_pair() -> None:
    assert buffer_prototype.buffer is CzarrGpuBuffer
    assert buffer_prototype.nd_buffer is CzarrGpuNDBuffer
