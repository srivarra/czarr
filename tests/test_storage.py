"""Tests for ``czarr.storage.GPULocalStore`` (cuFile-backed local store)."""

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
import zarr

from czarr import LZ4, cufile
from czarr.storage import GPULocalStore


def test_gpu_local_store_roundtrip_via_zarr(gpustore_tmpdir):
    """Write/read a Zarr array through GPULocalStore using GPU buffer prototype."""
    store = GPULocalStore(gpustore_tmpdir)
    if not store.gds_available:
        pytest.skip("cuFile not available on this host")

    with zarr.config.set({"buffer": "zarr.core.buffer.gpu.Buffer", "ndbuffer": "zarr.core.buffer.gpu.NDBuffer"}):
        data = np.arange(64 * 64, dtype="float32").reshape(64, 64)
        arr = zarr.create_array(
            store=store,
            shape=data.shape,
            chunks=(32, 32),
            dtype="float32",
            compressors=[LZ4()],
        )
        arr[:] = cp.asarray(data)

        store_ro = GPULocalStore(gpustore_tmpdir)
        arr_ro = zarr.open_array(store=store_ro, mode="r")
        out = arr_ro[:]

    assert isinstance(out, cp.ndarray)
    np.testing.assert_array_equal(cp.asnumpy(out), data)


def test_gpu_local_store_falls_back_for_host_prototype(gpustore_tmpdir):
    """A host buffer prototype should bypass cuFile and use the LocalStore path."""
    store = GPULocalStore(gpustore_tmpdir)
    if not store.gds_available:
        pytest.skip("cuFile not available on this host")

    data = np.arange(64 * 64, dtype="float32").reshape(64, 64)
    arr = zarr.create_array(
        store=store,
        shape=data.shape,
        chunks=(32, 32),
        dtype="float32",
        compressors=[LZ4()],
    )
    arr[:] = data
    out = arr[:]
    np.testing.assert_array_equal(out, data)


def test_arr_getitem_drives_cufile(gpustore_tmpdir):
    """A full-array read through zarr's pipeline must actually exercise cuFile."""
    import czarr

    store = GPULocalStore(gpustore_tmpdir)
    if not store.gds_available:
        pytest.skip("cuFile not available on this host")

    czarr.configure_gpu()
    try:
        rng = np.random.default_rng(0)
        src = rng.integers(0, 64, 8 * 2 * 4 * 64 * 64, dtype=np.int32).astype(np.float32).reshape(8, 2, 4, 64, 64)

        from czarr import ANS

        arr = zarr.create_array(
            store=store,
            shape=src.shape,
            chunks=(1, 1, 4, 64, 64),
            dtype="float32",
            compressors=[ANS()],
        )
        arr[:] = cp.asarray(src)

        # Multi-chunk reads fan out to per-chunk read_into via zarr's
        # concurrent_map; trace it to prove cuFile actually served the read.
        orig_into = cufile.read_into
        calls: list[int] = []

        def _traced_into(path, dev_ptr, size, file_offset=0):
            calls.append(size)
            return orig_into(path, dev_ptr, size, file_offset)

        cufile.read_into = _traced_into
        try:
            store_r = GPULocalStore(gpustore_tmpdir, read_only=True)
            arr_r = zarr.open_array(store=store_r, mode="r")
            out = arr_r[2:6]  # 4 chunks
        finally:
            cufile.read_into = orig_into
    finally:
        zarr.config.reset()

    assert len(calls) > 0, "cuFile read_into was never invoked from arr[:]"
    np.testing.assert_array_equal(cp.asnumpy(out), src[2:6])


def test_set_poll_mode_toggles_without_error():
    """``set_poll_mode`` should be idempotent and not crash regardless of fs type."""
    if not cufile.is_available():
        pytest.skip("cuFile not available on this host")
    # Toggle on then off — both must return cleanly.
    cufile.set_poll_mode(True, threshold_kb=4)
    cufile.set_poll_mode(False, threshold_kb=4)
