"""CzarrPipeline integration tests."""

import cupy as cp
import numpy as np
import pytest
import zarr

import czarr
from czarr.pipeline import CzarrPipeline


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
