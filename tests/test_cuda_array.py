"""Tests for :class:`czarr.CudaZarrArray` and its factories.

Round-trip parity against a stock :class:`zarr.Array`, run against an
in-memory store to keep the test suite hermetic.  cuFile + cupy JIT are
not exercised here; real-GPU coverage lives in the storage/pipeline
tests and the bench harness.
"""

import cupy as cp
import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

import czarr


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


class TestWrap:
    """``CudaZarrArray.wrap`` parity with the wrapped array."""

    def test_wrap_returns_cuda_array(self, small_array) -> None:
        cuda = czarr.CudaZarrArray.wrap(small_array)
        assert isinstance(cuda, czarr.CudaZarrArray)
        assert isinstance(cuda, zarr.Array)

    def test_wrap_preserves_metadata(self, small_array) -> None:
        cuda = czarr.CudaZarrArray.wrap(small_array)
        assert cuda.shape == small_array.shape
        assert cuda.dtype == small_array.dtype
        assert cuda.chunks == small_array.chunks


class TestIndexingParity:
    """Reads through the subclass match the stock zarr.Array byte-for-byte."""

    @pytest.mark.parametrize(
        "key",
        [
            np.s_[:],
            np.s_[3],
            np.s_[2:10, 4, :],
            np.s_[..., 0],
            np.s_[::2],
        ],
    )
    def test_read_parity(self, small_array, key) -> None:
        cuda = czarr.CudaZarrArray.wrap(small_array)
        np.testing.assert_array_equal(np.asarray(cuda[key]), np.asarray(small_array[key]))

    def test_bool_mask_oindex(self, small_array) -> None:
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


# ---------------------------------------------------------------------------
# Lowlevel fast path (phase-4 step 5)
# ---------------------------------------------------------------------------


class TestFastPath:
    """``__getitem__`` routes through czarr.core.Array when lowlevel can serve."""

    @pytest.fixture
    def disk_array(self, gpustore_tmpdir):
        """(CudaZarrArray, ndarray) — default-zstd array on real disk."""
        root = str(gpustore_tmpdir / "fast.zarr")
        arr = zarr.create_array(store=root, shape=(8, 8, 8), chunks=(4, 4, 4), dtype="float32")
        data = np.arange(512, dtype="float32").reshape(8, 8, 8)
        arr[:] = data
        return czarr.open_cuda_array(root), data

    @pytest.mark.parametrize(
        "key",
        [
            np.s_[:],
            np.s_[1:5, :, 2:7],
            np.s_[3],
            np.s_[..., 2],
            (2, slice(1, 3), 4),
        ],
    )
    def test_returns_cupy_with_numpy_shape_semantics(self, disk_array, key) -> None:
        cuda, data = disk_array
        out = cuda[key]
        # cupy result proves the fast path ran: configure_gpu was never
        # called, so the zarr fallback would have produced numpy.
        assert isinstance(out, cp.ndarray)
        assert out.shape == np.shape(data[key])  # int axes squeezed
        np.testing.assert_array_equal(cp.asnumpy(out), data[key])

    def test_full_int_selection_is_zero_dim(self, disk_array) -> None:
        cuda, data = disk_array
        out = cuda[1, 2, 3]
        assert isinstance(out, cp.ndarray)
        assert out.shape == ()
        assert float(out) == data[1, 2, 3]

    def test_fancy_indexing_falls_back(self, disk_array) -> None:
        cuda, data = disk_array
        out = cuda[[0, 2]]
        np.testing.assert_array_equal(np.asarray(out), data[[0, 2]])

    def test_unsupported_codec_falls_back(self, gpustore_tmpdir) -> None:
        root = str(gpustore_tmpdir / "gzip.zarr")
        arr = zarr.create_array(
            store=root,
            shape=(8, 8),
            chunks=(4, 4),
            dtype="float32",
            compressors=[zarr.codecs.GzipCodec()],
        )
        data = np.arange(64, dtype="float32").reshape(8, 8)
        arr[:] = data
        cuda = czarr.open_cuda_array(root)
        np.testing.assert_array_equal(np.asarray(cuda[1:6, :]), data[1:6, :])

    def test_memory_store_falls_back(self, small_array) -> None:
        cuda = czarr.CudaZarrArray.wrap(small_array)
        assert cuda._fast_array() is None
        np.testing.assert_array_equal(np.asarray(cuda[2:5]), np.asarray(small_array[2:5]))

    def test_write_invalidates_cached_plan(self, gpustore_tmpdir) -> None:
        root = str(gpustore_tmpdir / "rw.zarr")
        cuda = czarr.create_cuda_array(store=root, shape=(8, 8), chunks=(4, 4), dtype="float32")
        cuda[:] = np.zeros((8, 8), dtype="float32")
        assert float(cp.asnumpy(cuda[0, 0])) == 0.0  # populate the plan cache
        cuda[0] = np.full(8, 7.0, dtype="float32")
        np.testing.assert_array_equal(cp.asnumpy(cuda[0]), np.full(8, 7.0, dtype="float32"))
