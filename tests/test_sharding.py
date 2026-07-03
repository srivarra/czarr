"""Phase 4 sharding tests — verify zarr's ShardingCodec routes inner chunks
through our CzarrPipeline transparently.

Zarr 3.2's ``ShardingCodec`` reaches the codec pipeline lazily via
``get_pipeline_class()`` (reads ``codec_pipeline.path`` from
``zarr.config``).  Once :func:`czarr.configure_gpu` is called, that
resolves to :class:`czarr.pipeline.CzarrPipeline` — so we get full
GPU-pipeline behaviour for inner-chunk decode *without* writing a new
ShardingCodec.
"""

import cupy as cp
import numpy as np
import zarr
from zarr.codecs import ShardingCodec

import czarr


def _build_sharded_array(store, shape, chunk_shape, shard_shape, dtype):
    """Helper: create a zarr v3 array with sharding+zstd."""
    return zarr.create_array(
        store=store,
        shape=shape,
        chunks=shard_shape,
        shards=shard_shape,
        dtype=dtype,
        # Inner chunks are smaller than the shard; sharding bundles them.
        compressors=[czarr.Zstd()],
        # Sharding lives in the metadata as a serializer; zarr handles wiring.
        serializer="auto",
    )


class TestSharding:
    def test_v3_sharded_round_trip_via_czarr_pipeline(self, gpustore_tmpdir):
        """Round-trip through GPULocalStore + ShardingCodec + CzarrPipeline."""
        czarr.configure_gpu()

        store = czarr.GPULocalStore(gpustore_tmpdir / "sharded.zarr")
        if not store.gds_available:
            import pytest

            pytest.skip("cuFile not available on this host")

        # Shard = 4x4 inner-chunks of 8x8 elements; total 32x32 array.
        rng = np.random.default_rng(0)
        data = rng.standard_normal((32, 32)).astype("float32")

        arr = zarr.create_array(
            store=store,
            shape=data.shape,
            chunks=(8, 8),
            shards=(32, 32),
            dtype="float32",
            compressors=[czarr.Zstd()],
            overwrite=True,
        )
        arr[:] = cp.asarray(data)
        out = arr[:]

        assert isinstance(out, cp.ndarray)
        np.testing.assert_array_equal(cp.asnumpy(out), data)

    def test_sharded_partial_read(self, gpustore_tmpdir):
        """Selecting one shard out of a multi-shard array goes through the same pipeline."""
        czarr.configure_gpu()

        store = czarr.GPULocalStore(gpustore_tmpdir / "sharded_partial.zarr")
        if not store.gds_available:
            import pytest

            pytest.skip("cuFile not available on this host")

        rng = np.random.default_rng(1)
        data = rng.integers(0, 1000, size=(64, 64), dtype=np.int32)

        arr = zarr.create_array(
            store=store,
            shape=data.shape,
            chunks=(8, 8),  # inner chunk
            shards=(32, 32),  # shard wraps 16 inner chunks
            dtype="int32",
            compressors=[czarr.Zstd()],
            overwrite=True,
        )
        arr[:] = cp.asarray(data)

        # Read a region that spans two shards along both axes.
        sub = arr[16:48, 16:48]
        np.testing.assert_array_equal(cp.asnumpy(sub), data[16:48, 16:48])

    def test_sharding_codec_uses_czarr_pipeline(self):
        """After configure_gpu(), ShardingCodec.codec_pipeline is CzarrPipeline."""
        czarr.configure_gpu()
        sharding = ShardingCodec(chunk_shape=(8, 8))
        from czarr.pipeline import CzarrPipeline

        assert isinstance(sharding.codec_pipeline, CzarrPipeline)
