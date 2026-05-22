"""Phase 5 crc32c tests — verify zarr-stock Crc32cCodec works under CzarrPipeline.

A bespoke GPU CRC32C kernel is deferred (non-trivial — needs the GF(2)
tree-reduce dance to parallelise across warps).  In the meantime,
zarr's stock :class:`zarr.codecs.Crc32cCodec` works correctly with our
pipeline because:

* The compressed bytes are materialised to host for ``google_crc32c``
  to hash them.  That's a real GPU->host transfer per chunk, but the
  checksum payload is small (4 bytes) and crc32c is computed after
  compression so the host transfer is over already-shrunk bytes.
* No GPU acceleration today; tracked as follow-up.
"""

from __future__ import annotations

import cupy as cp
import numpy as np
import zarr
from zarr.codecs import Crc32cCodec
from zarr.storage import MemoryStore

import czarr


class TestCrc32c:
    def test_v3_store_with_crc32c_round_trips_under_czarr_pipeline(self, gpustore_tmpdir):
        """zarr-stock Crc32cCodec validates correctly when our pipeline runs."""
        czarr.configure_gpu()

        store = czarr.GPULocalStore(gpustore_tmpdir / "crc32c_rt.zarr")
        if not store.gds_available:
            import pytest

            pytest.skip("cuFile not available on this host")

        rng = np.random.default_rng(0)
        data = rng.standard_normal((16, 16)).astype("float32")

        arr = zarr.create_array(
            store=store,
            shape=data.shape,
            chunks=(8, 8),
            dtype="float32",
            compressors=[czarr.Zstd(), Crc32cCodec()],
            overwrite=True,
        )
        arr[:] = cp.asarray(data)
        out = arr[:]

        np.testing.assert_array_equal(cp.asnumpy(out), data)

    def test_crc32c_with_memory_store(self):
        """End-to-end via MemoryStore (no cuFile), exercises crc32c on GPU buffer."""
        czarr.configure_gpu()

        rng = np.random.default_rng(1)
        data = rng.integers(0, 1000, size=(8, 8), dtype=np.int32)

        arr = zarr.create_array(
            store=MemoryStore(),
            shape=data.shape,
            chunks=data.shape,
            dtype=data.dtype,
            compressors=[czarr.Zstd(), Crc32cCodec()],
        )
        arr[:] = cp.asarray(data)
        out = arr[:]
        np.testing.assert_array_equal(cp.asnumpy(out), data)
