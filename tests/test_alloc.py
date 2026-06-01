"""Tests for the cupy/RMM allocator wiring exposed by ``czarr.alloc``."""

from __future__ import annotations

import cupy as cp
import numpy as np
import rmm
import rmm.statistics
import zarr
from zarr.core.buffer import gpu as gpu_buffer
from zarr.storage import MemoryStore

from czarr import LZ4, register_nvcomp_allocator, use_rmm_pool


def _decode_roundtrip() -> None:
    """Bit-exact GPU-prototype roundtrip through nvCOMP (LZ4) via a MemoryStore.

    Allocates nvCOMP scratch + decode output buffers on the device, which is
    what we want to observe flowing (or not) through RMM.
    """
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


def test_register_nvcomp_allocator_is_idempotent():
    """Calling the default registration twice should be a no-op."""
    register_nvcomp_allocator()
    register_nvcomp_allocator()


def test_use_rmm_pool_actually_routes_allocations_through_rmm():
    """``use_rmm_pool`` must *route* czarr's device allocations through RMM —
    not merely install a pool MR.

    Proven with RMM's own allocation counters: a decode on the default path
    records zero RMM traffic, while the same decode after ``use_rmm_pool``
    records non-zero traffic.  Both phases must still roundtrip bit-exact.

    ``use_rmm_pool`` mutates process-global state (pool MR + cupy allocator)
    that persists for the session, so the control phase explicitly resets to a
    vanilla baseline first — keeping the test independent of test order.
    """
    register_nvcomp_allocator()

    # --- Control: vanilla cupy pool + plain CUDA MR; RMM must see nothing. ---
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
    rmm.mr.set_current_device_resource(rmm.mr.CudaMemoryResource())
    rmm.statistics.enable_statistics()
    with rmm.statistics.statistics():
        _decode_roundtrip()
        control = rmm.statistics.get_statistics()
    assert control.total_count == 0, f"default path routed {control.total_count} allocs through RMM, expected 0"

    # --- Opt in: pool MR + cupy->RMM allocator; RMM must capture the traffic. ---
    use_rmm_pool(initial_size=32 * 1024 * 1024)
    mr = rmm.mr.get_current_device_resource()
    assert "Pool" in type(mr).__name__, f"expected a pool MR, got {type(mr).__name__}"
    rmm.statistics.enable_statistics()
    with rmm.statistics.statistics():
        _decode_roundtrip()
        opted = rmm.statistics.get_statistics()
    assert opted.total_count > 0, "use_rmm_pool installed a pool but no allocations routed through it"
