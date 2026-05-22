"""Pipeline tests: Phase 1 substrate + Phase 2 CzarrPipeline integration."""

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
import zarr

import czarr
from czarr.pipeline import CzarrPipeline, DeviceBufferPool, PinnedHostPool, StreamPool


class TestStreamPool:
    def test_default_size(self):
        pool = StreamPool()
        try:
            assert pool.size == 4
            assert len(pool) == 4
            assert len(list(pool)) == 4
        finally:
            pool.close()

    def test_custom_size(self):
        pool = StreamPool(size=8)
        try:
            assert pool.size == 8
        finally:
            pool.close()

    def test_invalid_size_rejected(self):
        with pytest.raises(ValueError, match="size must be >= 1"):
            StreamPool(size=0)

    def test_round_robin_acquire(self):
        pool = StreamPool(size=3)
        try:
            seen = [pool.acquire() for _ in range(7)]
            # After 7 acquires of a 3-pool we expect to see indices
            # 0,1,2,0,1,2,0 i.e. exactly 3 distinct stream identities.
            assert len({id(s) for s in seen}) == 3
            # First three and second three should be the same triple in order.
            assert seen[0] is seen[3]
            assert seen[1] is seen[4]
            assert seen[2] is seen[5]
            assert seen[6] is seen[0]
        finally:
            pool.close()

    def test_sync_all_noops_on_idle(self):
        pool = StreamPool(size=2)
        try:
            pool.sync_all()  # nothing in flight; should not throw
        finally:
            pool.close()

    def test_close_empties_pool(self):
        pool = StreamPool(size=2)
        pool.close()
        assert len(pool) == 0


class TestPinnedHostPool:
    def test_acquire_releases_recycle(self):
        pool = PinnedHostPool()
        try:
            buf = pool.acquire(1024)
            assert buf.size == 1024
            assert pool.live_count == 1
            pool.release(buf)
            assert pool.live_count == 0
            assert pool.free_count(1024) == 1
            buf2 = pool.acquire(1024)
            # Recycled buffer should be the same one we just released.
            assert buf2 is buf
            pool.release(buf2)
        finally:
            pool.close()

    def test_prealloc_populates_free_list(self):
        pool = PinnedHostPool(prealloc=[(4096, 3), (1024, 2)])
        try:
            assert pool.free_count(4096) == 3
            assert pool.free_count(1024) == 2
            assert pool.live_count == 0
        finally:
            pool.close()

    def test_zero_size_rejected(self):
        pool = PinnedHostPool()
        try:
            with pytest.raises(ValueError, match="size must be > 0"):
                pool.acquire(0)
        finally:
            pool.close()

    def test_different_sizes_use_distinct_buckets(self):
        pool = PinnedHostPool()
        try:
            a = pool.acquire(1024)
            b = pool.acquire(2048)
            pool.release(a)
            pool.release(b)
            assert pool.free_count(1024) == 1
            assert pool.free_count(2048) == 1
        finally:
            pool.close()


class TestCzarrPipeline:
    def test_pipeline_registers_as_default(self):
        """configure_gpu() should set codec_pipeline.path to CzarrPipeline."""
        czarr.configure_gpu()
        assert zarr.config.get("codec_pipeline.path") == f"{CzarrPipeline.__module__}.{CzarrPipeline.__qualname__}"

    def test_pipeline_round_trip_with_compat_codec(self, gpustore_tmpdir):
        """End-to-end round-trip via CzarrPipeline + GPULocalStore + Zstd.

        Exercises the GPU-direct path: GPULocalStore returns gpu.Buffer
        from cuFile reads, which CudaBytesBytesCodec passes to nvcomp
        without any host round-trip.
        """
        czarr.configure_gpu()

        store = czarr.GPULocalStore(gpustore_tmpdir / "rt_zstd.zarr")
        if not store.gds_available:
            pytest.skip("cuFile not available on this host")

        src = np.arange(1024, dtype=np.float32).reshape(32, 32)
        arr = zarr.create_array(
            store=store,
            shape=src.shape,
            chunks=(16, 16),
            dtype=src.dtype,
            compressors=[czarr.Zstd()],
            overwrite=True,
        )
        arr[:] = cp.asarray(src)
        out = arr[:]

        assert isinstance(out, cp.ndarray)
        np.testing.assert_array_equal(cp.asnumpy(out), src)

    def test_pipeline_round_trip_with_native_codec(self, gpustore_tmpdir):
        """Round-trip via CzarrPipeline + GPULocalStore + nvCOMP-native ANS."""
        czarr.configure_gpu()

        store = czarr.GPULocalStore(gpustore_tmpdir / "rt_ans.zarr")
        if not store.gds_available:
            pytest.skip("cuFile not available on this host")

        src = np.arange(2048, dtype=np.uint16).reshape(32, 64)
        arr = zarr.create_array(
            store=store,
            shape=src.shape,
            chunks=(8, 32),
            dtype=src.dtype,
            compressors=[czarr.ANS()],
            overwrite=True,
        )
        arr[:] = cp.asarray(src)
        out = arr[:]

        assert isinstance(out, cp.ndarray)
        np.testing.assert_array_equal(cp.asnumpy(out), src)

    def test_stream_pool_shared_across_pipeline_instances(self):
        """Each pipeline instance shares the class-level StreamPool."""
        czarr.configure_gpu(stream_pool_size=4)
        p1 = CzarrPipeline.get_stream_pool()
        p2 = CzarrPipeline.get_stream_pool()
        assert p1 is p2
        assert p1.size == 4

    def test_reconfigure_replaces_pools(self):
        """CzarrPipeline.configure() can resize the substrate."""
        CzarrPipeline.configure(stream_pool_size=2)
        assert CzarrPipeline.get_stream_pool().size == 2
        CzarrPipeline.configure(stream_pool_size=8)
        assert CzarrPipeline.get_stream_pool().size == 8


class TestDeviceBufferPool:
    def test_basic_alloc(self):
        pool = DeviceBufferPool()
        buf = pool.acquire(4096)
        assert isinstance(buf, cp.ndarray)
        assert buf.size == 4096
        assert buf.dtype == cp.uint8

    def test_zero_size_rejected(self):
        pool = DeviceBufferPool()
        with pytest.raises(ValueError, match="size must be > 0"):
            pool.acquire(0)

    def test_stream_param_accepted(self):
        sp = StreamPool(size=1)
        try:
            stream = sp.acquire()
            pool = DeviceBufferPool()
            buf = pool.acquire(2048, stream=stream)
            assert buf.size == 2048
        finally:
            sp.close()
