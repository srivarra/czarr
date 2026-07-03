"""Phase 3 filter codec tests: Shuffle, Delta, FixedScaleOffset, BitRound."""

import cupy as cp
import numcodecs
import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

import czarr


def _roundtrip(
    data,
    *,
    filters: list | None = None,
    compressors: list | None = None,
    chunks=None,
):
    """Write + read ``data`` through a v3 MemoryStore with the given chain."""
    czarr.configure_gpu()
    store = MemoryStore()
    arr = zarr.create_array(
        store=store,
        shape=data.shape,
        chunks=chunks or data.shape,
        dtype=data.dtype,
        filters=filters or [],
        compressors=compressors or [czarr.Zstd()],
    )
    arr[:] = cp.asarray(data) if isinstance(data, np.ndarray) else data
    return arr[:]


class TestShuffle:
    @pytest.mark.parametrize("elementsize", [2, 4, 8])
    def test_shuffle_round_trip_via_v3_chain(self, elementsize):
        rng = np.random.default_rng(0)
        n = (32 * 32) * elementsize
        flat = rng.integers(0, 256, size=n, dtype=np.uint8)
        dtype = {2: np.uint16, 4: np.uint32, 8: np.uint64}[elementsize]
        data = flat.view(dtype).reshape(32, 32)
        # Shuffle is a BytesBytesCodec — sits in `compressors`.
        out = _roundtrip(
            data,
            compressors=[czarr.Shuffle(elementsize=elementsize), czarr.Zstd()],
        )
        np.testing.assert_array_equal(cp.asnumpy(out), data)

    def test_shuffle_matches_numcodecs_bitstream(self):
        """czarr.Shuffle encode bytes should match numcodecs.Shuffle byte-for-byte."""
        from czarr.kernels.byteshuffle import byteshuffle

        rng = np.random.default_rng(1)
        data = rng.integers(0, 65535, size=128, dtype=np.uint16)
        cpu_encoded = numcodecs.Shuffle(elementsize=2).encode(data)
        gpu_bytes = byteshuffle(cp.asarray(data).view(cp.uint8), 2, data.nbytes)
        np.testing.assert_array_equal(cp.asnumpy(gpu_bytes), np.asarray(cpu_encoded))


class TestDelta:
    def test_delta_round_trip_int(self):
        rng = np.random.default_rng(0)
        data = rng.integers(-1000, 1000, size=(16, 16), dtype=np.int32)
        out = _roundtrip(data, filters=[czarr.Delta(dtype="<i4")])
        np.testing.assert_array_equal(cp.asnumpy(out), data)

    def test_delta_round_trip_float(self):
        # float32 cumsum accumulates rounding error; tolerate ~ULP-scale drift.
        rng = np.random.default_rng(1)
        data = rng.standard_normal((8, 8)).astype("float32")
        out = _roundtrip(data, filters=[czarr.Delta(dtype="<f4")])
        np.testing.assert_allclose(cp.asnumpy(out), data, atol=1e-5)


class TestFixedScaleOffset:
    def test_affine_round_trip_no_dtype_change(self):
        """Same-dtype FSO round-trip: affine math without quantisation.

        The dtype-changing path (e.g. float32 -> int16 storage) requires
        ``resolve_metadata`` on the codec so downstream codecs see the
        storage dtype — deferred follow-up after Phase 3.
        """
        rng = np.random.default_rng(0)
        data = (rng.standard_normal((16, 16)) * 100).astype("float32")
        out = _roundtrip(
            data,
            filters=[czarr.FixedScaleOffset(offset=10.0, scale=0.5, dtype="<f4")],
        )
        np.testing.assert_allclose(cp.asnumpy(out), data, rtol=1e-5)


class TestBitRound:
    @pytest.mark.parametrize("keepbits", [10, 15, 20])
    def test_bitround_truncates_only_low_bits(self, keepbits):
        rng = np.random.default_rng(0)
        data = (rng.standard_normal((16, 16)) * 100).astype("float32")
        out = _roundtrip(data, filters=[czarr.BitRound(keepbits=keepbits)])
        # Relative error after rounding is bounded by 2^-(keepbits).
        rel_err = np.abs(cp.asnumpy(out) - data) / np.maximum(np.abs(data), 1e-12)
        assert rel_err.max() <= 2**-(keepbits)

    def test_bitround_rejects_int_dtype(self):
        """BitRound only applies to floats; encoding ints should TypeError."""
        rng = np.random.default_rng(0)
        data = rng.integers(0, 1000, size=(8, 8), dtype=np.int32)
        with pytest.raises(TypeError, match="float16/32/64"):
            _roundtrip(data, filters=[czarr.BitRound(keepbits=10)])
