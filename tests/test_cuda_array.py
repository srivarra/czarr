"""v0.1 skeleton tests for :class:`czarr.CudaZarrArray`.

Round-trip + fast-path/fallback split, run against an in-memory store to
keep the test suite hermetic.  cuFile + cupy JIT are not exercised here;
real-GPU benches live in ``bench/cuda_array/``.

The orchestrator currently delegates :meth:`_CudaArrayImpl.retrieve_gpu`
to zarr's async machinery, so the fast and fallback paths produce the
same bytes; later subtasks (``lfw5ftx9``, ``s7ucov1a``) replace the
fast-path body and these tests pin the contract.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

import czarr
from czarr.array.cuda_array import _is_basic_indexing


@pytest.fixture
def small_array(tmp_path) -> zarr.Array:
    """A 16x16x16 float32 array with deterministic content, in MemoryStore."""
    store = MemoryStore()
    arr = zarr.create_array(
        store=store,
        shape=(16, 16, 16),
        chunks=(4, 4, 4),
        dtype="float32",
    )
    data = np.arange(16 * 16 * 16, dtype="float32").reshape(16, 16, 16)
    arr[:] = data
    return arr


# ---------------------------------------------------------------------------
# _is_basic_indexing — the type-guard for the fast path
# ---------------------------------------------------------------------------


class TestIsBasicIndexing:
    """The fast-path eligibility predicate."""

    @pytest.mark.parametrize(
        "key",
        [
            0,
            -1,
            slice(None),
            slice(0, 8),
            slice(2, 10, 1),
            slice(2, 10, None),
            ...,
            (0, slice(None), ...),
            (slice(0, 8), 4, slice(None)),
            (..., 0),
            (),
        ],
    )
    def test_accepts_basic(self, key) -> None:
        assert _is_basic_indexing(key)

    @pytest.mark.parametrize(
        "key",
        [
            slice(0, 8, 2),  # step != 1
            slice(None, None, -1),  # reversed
            np.array([0, 2, 4]),  # integer-array advanced
            np.array([True, False] * 8),  # bool mask
            "field",  # structured-dtype field name
            (slice(None), slice(0, 8, 2)),  # nested non-step-1
            (..., ..., 0),  # multi-ellipsis
            (np.array([0, 1]),),  # ndarray inside tuple
        ],
    )
    def test_rejects_advanced(self, key) -> None:
        assert not _is_basic_indexing(key)


# ---------------------------------------------------------------------------
# CudaZarrArray.wrap and the fast path
# ---------------------------------------------------------------------------


class TestWrap:
    """``CudaZarrArray.wrap`` and basic-indexing fast path."""

    def test_wrap_returns_cuda_array(self, small_array) -> None:
        cuda = czarr.CudaZarrArray.wrap(small_array)
        assert isinstance(cuda, czarr.CudaZarrArray)
        assert isinstance(cuda, zarr.Array)

    def test_wrap_preserves_metadata(self, small_array) -> None:
        cuda = czarr.CudaZarrArray.wrap(small_array)
        assert cuda.shape == small_array.shape
        assert cuda.dtype == small_array.dtype
        assert cuda.chunks == small_array.chunks

    def test_wrap_takes_tuning_kwargs(self, small_array) -> None:
        cuda = czarr.CudaZarrArray.wrap(small_array, queue_depth=32, microbatch_size=4)
        assert cuda._orchestrator.queue_depth == 32
        assert cuda._orchestrator.microbatch_size == 4

    def test_orchestrator_lazy_init(self, small_array) -> None:
        cuda = czarr.CudaZarrArray.wrap(small_array)
        # _impl is set eagerly via wrap; the property returns it.
        assert cuda._orchestrator is cuda._impl


class TestBasicIndexingFastPath:
    """Round-trip parity between fast path and zarr.Array baseline."""

    def test_full_slice(self, small_array) -> None:
        cuda = czarr.CudaZarrArray.wrap(small_array)
        expected = small_array[:]
        actual = cuda[:]
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_scalar_index(self, small_array) -> None:
        cuda = czarr.CudaZarrArray.wrap(small_array)
        expected = small_array[3]
        actual = cuda[3]
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_tuple_slice(self, small_array) -> None:
        cuda = czarr.CudaZarrArray.wrap(small_array)
        expected = small_array[2:10, 4, :]
        actual = cuda[2:10, 4, :]
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_ellipsis(self, small_array) -> None:
        cuda = czarr.CudaZarrArray.wrap(small_array)
        expected = small_array[..., 0]
        actual = cuda[..., 0]
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


class TestAdvancedIndexingFallback:
    """Advanced indexing should fall through to ``zarr.Array.__getitem__``."""

    def test_strided_slice_falls_through(self, small_array) -> None:
        cuda = czarr.CudaZarrArray.wrap(small_array)
        expected = small_array[::2]
        actual = cuda[::2]
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_bool_mask_falls_through(self, small_array) -> None:
        cuda = czarr.CudaZarrArray.wrap(small_array)
        mask = np.zeros(16, dtype=bool)
        mask[::4] = True
        expected = small_array.oindex[mask, :, :]
        actual = cuda.oindex[mask, :, :]
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


class TestFactories:
    """``open_cuda_array`` and ``create_cuda_array``."""

    def test_create_round_trip(self) -> None:
        store = MemoryStore()
        arr = czarr.create_cuda_array(
            store=store,
            shape=(8, 8),
            chunks=(4, 4),
            dtype="int32",
            compressors=None,
            filters=None,
        )
        assert isinstance(arr, czarr.CudaZarrArray)
        arr[:] = np.arange(64, dtype="int32").reshape(8, 8)
        np.testing.assert_array_equal(np.asarray(arr[:]), np.arange(64, dtype="int32").reshape(8, 8))

    def test_open_returns_cuda_array(self, small_array) -> None:
        # small_array fixture used a MemoryStore; re-open by path off its store.
        store = small_array.store_path.store
        opened = czarr.open_cuda_array(store=store, path=small_array.path)
        assert isinstance(opened, czarr.CudaZarrArray)
        assert opened.shape == small_array.shape

    def test_string_path_auto_wraps_to_gpu_local_store(self, tmp_path) -> None:
        """A string ``store`` argument should default to GPULocalStore (cuFile)."""
        # Create + read back via path strings.  The factories should
        # autowrap the path in czarr.GPULocalStore.
        arr_path = str(tmp_path / "data.zarr")
        arr = czarr.create_cuda_array(
            store=arr_path,
            shape=(4, 4),
            chunks=(2, 2),
            dtype="int16",
            compressors=None,
            filters=None,
        )
        assert isinstance(arr.store_path.store, czarr.GPULocalStore)
        arr[:] = np.arange(16, dtype="int16").reshape(4, 4)

        opened = czarr.open_cuda_array(store=arr_path)
        assert isinstance(opened, czarr.CudaZarrArray)
        assert isinstance(opened.store_path.store, czarr.GPULocalStore)

    def test_pathlib_path_auto_wraps_to_gpu_local_store(self, tmp_path) -> None:
        """A ``pathlib.Path`` should be treated like a string."""
        arr = czarr.create_cuda_array(
            store=tmp_path / "data.zarr",
            shape=(4,),
            chunks=(2,),
            dtype="int16",
            compressors=None,
            filters=None,
        )
        assert isinstance(arr.store_path.store, czarr.GPULocalStore)

    def test_explicit_store_passes_through(self, tmp_path) -> None:
        """An explicit Store instance should NOT be autowrapped."""
        store = MemoryStore()
        arr = czarr.create_cuda_array(
            store=store,
            shape=(4,),
            chunks=(2,),
            dtype="int16",
            compressors=None,
            filters=None,
        )
        # MemoryStore stays as MemoryStore; not autowrapped.
        assert arr.store_path.store is store
        assert isinstance(arr.store_path.store, MemoryStore)
