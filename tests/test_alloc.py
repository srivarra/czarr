"""Tests for the cupy/RMM allocator wiring exposed by ``czarr.alloc``."""

from __future__ import annotations

import cupy as cp
import numpy as np
import rmm
import zarr
from zarr.core.buffer import gpu as gpu_buffer
from zarr.storage import MemoryStore

from czarr import LZ4, register_nvcomp_allocator, use_rmm_pool


def test_register_nvcomp_allocator_is_idempotent():
    """Calling the default registration twice should be a no-op."""
    register_nvcomp_allocator()
    register_nvcomp_allocator()


def test_use_rmm_pool_routes_through_rmm_and_roundtrips():
    """After ``use_rmm_pool``, the current RMM MR must be a pool, and a Zarr
    GPU-prototype roundtrip through nvCOMP must still produce bit-exact output.
    """
    use_rmm_pool(initial_size=32 * 1024 * 1024)

    mr = rmm.mr.get_current_device_resource()
    # ``PoolMemoryResource`` (or any adapter wrapping one) signals that the
    # pool reinit succeeded.
    assert "Pool" in type(mr).__name__, f"expected a pool MR, got {type(mr).__name__}"

    data = np.arange(64 * 64, dtype="float32").reshape(64, 64)
    with zarr.config.set({"buffer": "zarr.core.buffer.gpu.Buffer", "ndbuffer": "zarr.core.buffer.gpu.NDBuffer"}):
        store = MemoryStore()
        arr = zarr.create_array(
            store=store,
            shape=data.shape,
            chunks=(32, 32),
            dtype="float32",
            compressors=[LZ4()],
        )
        arr[:] = cp.asarray(data)
        out = arr.get_basic_selection(prototype=gpu_buffer.buffer_prototype)

    np.testing.assert_array_equal(cp.asnumpy(out), data)
