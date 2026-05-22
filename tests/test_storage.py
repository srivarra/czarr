"""Tests for ``czarr.storage.GPULocalStore`` (cuFile-backed local store)."""

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
import zarr

from czarr import LZ4
from czarr.storage import GPULocalStore, cufile_runtime


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

        # CzarrPipeline routes multi-chunk reads through read_into_many;
        # the single-chunk path still uses read_into.  Trace both so the
        # assertion catches whichever path zarr picks.
        orig_into = cufile_runtime.read_into
        orig_many = cufile_runtime.read_into_many
        calls: list[int] = []

        def _traced_into(path, dev_ptr, size, file_offset=0):
            calls.append(size)
            return orig_into(path, dev_ptr, size, file_offset)

        def _traced_many(requests, **kw):
            calls.extend(size for _p, _d, size, _o in requests)
            return orig_many(requests, **kw)

        cufile_runtime.read_into = _traced_into
        cufile_runtime.read_into_many = _traced_many
        try:
            store_r = GPULocalStore(gpustore_tmpdir, read_only=True)
            arr_r = zarr.open_array(store=store_r, mode="r")
            out = arr_r[2:6]  # 4 chunks
        finally:
            cufile_runtime.read_into = orig_into
            cufile_runtime.read_into_many = orig_many
    finally:
        zarr.config.reset()

    assert len(calls) > 0, "cuFile read_into was never invoked from arr[:]"
    np.testing.assert_array_equal(cp.asnumpy(out), src[2:6])


def test_cufile_runtime_read_into_async_roundtrip(gpustore_tmpdir):
    """Async cuFile read submits, completes on stream sync, returns correct bytes."""
    if not cufile_runtime.is_async_available():
        pytest.skip("cuFile async path requires nvidia_fs (real GDS)")

    payload = np.arange(4096, dtype=np.uint32)
    path = gpustore_tmpdir / "async_probe.bin"
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


def test_get_many_matches_get_loop_byte_for_byte(gpustore_tmpdir):
    """GPULocalStore.get_many() must return the same bytes as N independent
    get() calls.  Compares against the per-key path on a real on-disk
    Zarr store (so the keys are real zarr metadata + chunk files)."""
    import asyncio

    import czarr

    czarr.configure_gpu()
    try:
        store = GPULocalStore(gpustore_tmpdir)
        if not store.gds_available:
            pytest.skip("cuFile not available on this host")
        rng = np.random.default_rng(0)
        # Multi-chunk array so we have several chunk keys to fetch.
        data = rng.integers(0, 1000, size=(64, 64), dtype=np.int32)
        arr = zarr.create_array(
            store=store,
            shape=data.shape,
            chunks=(16, 16),  # -> 16 chunk files
            dtype=np.int32,
            compressors=[LZ4()],
        )
        arr[:] = cp.asarray(data)

        # Enumerate the keys zarr actually wrote so we exercise real
        # paths.  Use a fresh read-only handle for clean state.
        store_r = GPULocalStore(gpustore_tmpdir, read_only=True)
        keys = sorted(asyncio.run(_collect_keys(store_r)))
        # Filter to chunk-data files (skip metadata blobs like zarr.json).
        keys = [k for k in keys if not k.endswith(".json")]
        assert len(keys) > 1, "test setup expected multiple chunk files"

        from zarr.core.buffer import gpu as gpu_buffer

        proto = gpu_buffer.buffer_prototype

        # Compare get_many() against a per-key get() loop.
        async def _both():
            many = await store_r.get_many(keys, prototype=proto)
            one_by_one = [await store_r.get(k, prototype=proto) for k in keys]
            return many, one_by_one

        many, one_by_one = asyncio.run(_both())
        assert len(many) == len(one_by_one)
        for m, o in zip(many, one_by_one, strict=True):
            assert m is not None and o is not None
            np.testing.assert_array_equal(cp.asnumpy(m.as_array_like()), cp.asnumpy(o.as_array_like()))
    finally:
        zarr.config.reset()


async def _collect_keys(store):
    keys = []
    async for k in store.list_prefix(""):
        keys.append(k)
    return keys


def test_set_poll_mode_toggles_without_error():
    """``set_poll_mode`` should be idempotent and not crash regardless of fs type."""
    if not cufile_runtime.is_available():
        pytest.skip("cuFile not available on this host")
    # Toggle on then off — both must return cleanly.
    cufile_runtime.set_poll_mode(True, threshold_kb=4)
    cufile_runtime.set_poll_mode(False, threshold_kb=4)
