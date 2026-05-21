"""Tests for ``czarr.storage.GPULocalStore`` (cuFile-backed local store)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cupy as cp
import numpy as np
import pytest
import zarr

from czarr import LZ4
from czarr.storage import GPULocalStore, cufile_runtime

# cuFile in compat mode rejects tmpfs/ramfs; force compat path on hosts without
# nvidia_fs so the sync tests still run.  When nvidia_fs IS loaded we want real
# GDS — the async path requires it and silently corrupts under forced compat.
if not os.path.exists("/proc/driver/nvidia-fs"):
    os.environ.setdefault("CUFILE_FORCE_COMPAT_MODE", "true")
_LUSTRE_TMP_PARENT = "/hpc/mydata/sricharan.varra/Dev/czarr"


@pytest.fixture
def lustre_tmpdir():
    # ignore_cleanup_errors: NFS metadata caching can leave stale .nfs* sentinel
    # files briefly after handle close, making rmdir() race-fail.
    with tempfile.TemporaryDirectory(
        dir=_LUSTRE_TMP_PARENT, prefix=".gpustore_test_", ignore_cleanup_errors=True
    ) as td:
        yield Path(td)


def test_gpu_local_store_roundtrip_via_zarr(lustre_tmpdir):
    """Write/read a Zarr array through GPULocalStore using GPU buffer prototype."""
    store = GPULocalStore(lustre_tmpdir)
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

        store_ro = GPULocalStore(lustre_tmpdir)
        arr_ro = zarr.open_array(store=store_ro, mode="r")
        out = arr_ro[:]

    assert isinstance(out, cp.ndarray)
    np.testing.assert_array_equal(cp.asnumpy(out), data)


def test_gpu_local_store_falls_back_for_host_prototype(lustre_tmpdir):
    """A host buffer prototype should bypass cuFile and use the LocalStore path."""
    store = GPULocalStore(lustre_tmpdir)
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


def test_arr_getitem_drives_cufile(lustre_tmpdir):
    """A full-array read through zarr's pipeline must actually exercise cuFile."""
    import czarr

    store = GPULocalStore(lustre_tmpdir)
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

        orig = cufile_runtime.read_into
        calls: list[int] = []

        def _traced(path, dev_ptr, size, file_offset=0):
            calls.append(size)
            return orig(path, dev_ptr, size, file_offset)

        cufile_runtime.read_into = _traced
        try:
            store_r = GPULocalStore(lustre_tmpdir, read_only=True)
            arr_r = zarr.open_array(store=store_r, mode="r")
            out = arr_r[2:6]  # 4 chunks
        finally:
            cufile_runtime.read_into = orig
    finally:
        zarr.config.reset()

    assert len(calls) > 0, "cuFile read_into was never invoked from arr[:]"
    np.testing.assert_array_equal(cp.asnumpy(out), src[2:6])


def test_cufile_runtime_read_into_async_roundtrip(lustre_tmpdir):
    """Async cuFile read submits, completes on stream sync, returns correct bytes."""
    if not cufile_runtime.is_async_available():
        pytest.skip("cuFile async path requires nvidia_fs (real GDS)")

    payload = np.arange(4096, dtype=np.uint32)
    path = lustre_tmpdir / "async_probe.bin"
    path.write_bytes(payload.tobytes())

    dbuf = cp.empty(payload.nbytes, dtype=cp.uint8)
    dbuf.fill(0xAA)
    cp.cuda.runtime.deviceSynchronize()

    stream = cp.cuda.Stream()
    args = cufile_runtime.read_into_async(
        path,
        int(dbuf.data.ptr),
        payload.nbytes,
        stream.ptr,
    )
    stream.synchronize()

    assert args.bytes_done == payload.nbytes
    got = np.frombuffer(dbuf.get(), dtype=np.uint32)
    np.testing.assert_array_equal(got, payload)
    cufile_runtime.deregister_buf(int(dbuf.data.ptr))


def test_set_poll_mode_toggles_without_error():
    """``set_poll_mode`` should be idempotent and not crash regardless of fs type."""
    if not cufile_runtime.is_available():
        pytest.skip("cuFile not available on this host")
    # Toggle on then off — both must return cleanly.
    cufile_runtime.set_poll_mode(True, threshold_kb=4)
    cufile_runtime.set_poll_mode(False, threshold_kb=4)
