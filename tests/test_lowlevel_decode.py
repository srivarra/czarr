"""End-to-end tests for czarr.lowlevel.decode / read_array (GPU required).

Every parity test is bit-exact against a zarr-python CPU read of the same
store — the hard correctness gate from the design doc.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

cp = pytest.importorskip("cupy")

from czarr import lowlevel  # noqa: E402

SHAPE = (16, 12)


def _write(root, data, **kwargs):
    arr = zarr.create_array(store=str(root), shape=data.shape, fill_value=0, zarr_format=3, **kwargs)
    arr[:] = data
    return arr


@pytest.fixture
def zstd_sharded(gpustore_tmpdir):
    data = (np.arange(np.prod(SHAPE)) % 251).astype("uint16").reshape(SHAPE)
    root = gpustore_tmpdir / "zstd_sharded.zarr"
    _write(root, data, dtype="uint16", chunks=(4, 4), shards=(8, 8))
    return root, data


@pytest.fixture
def zstd_plain(gpustore_tmpdir):
    """Plain zstd array, shape NOT divisible by chunks, one chunk unwritten."""
    data = np.arange(15 * 10, dtype="float32").reshape(15, 10)
    root = gpustore_tmpdir / "zstd_plain.zarr"
    arr = zarr.create_array(
        store=str(root), shape=(15, 10), chunks=(4, 4), dtype="float32", fill_value=-1, zarr_format=3
    )
    arr[:8, :] = data[:8, :]  # chunk rows 2+ untouched -> fill_value
    expected = np.full((15, 10), -1, dtype="float32")
    expected[:8, :] = data[:8, :]
    return root, expected


@pytest.mark.parametrize(
    "selection",
    [np.s_[:], np.s_[:8, :8], np.s_[2:14, 3:11], np.s_[5], np.s_[..., 0], np.s_[3:4, 4:12]],
)
def test_sharded_parity_with_zarr(zstd_sharded, selection) -> None:
    root, data = zstd_sharded
    got = lowlevel.read_array(root, selection)
    assert isinstance(got, cp.ndarray)
    expected = data[selection]
    # lowlevel reads are ndim-preserving; numpy squeezes int axes.
    assert got.size == expected.size
    np.testing.assert_array_equal(cp.asnumpy(got).reshape(expected.shape), expected)


def test_ndim_preserving_int_axis(zstd_sharded) -> None:
    root, data = zstd_sharded
    got = lowlevel.read_array(root, np.s_[5])
    assert got.shape == (1, SHAPE[1])
    np.testing.assert_array_equal(cp.asnumpy(got)[0], data[5])


def test_plain_fill_value_and_edge_chunks(zstd_plain) -> None:
    root, expected = zstd_plain
    got = lowlevel.read_array(root, np.s_[:])
    np.testing.assert_array_equal(cp.asnumpy(got), expected)


def test_plan_reuse_and_out_param(zstd_sharded) -> None:
    root, data = zstd_sharded
    plan = lowlevel.open_plan(root)
    out = cp.empty((8, 8), dtype="uint16")
    got = lowlevel.read_array(None, np.s_[:8, :8], plan=plan, out=out)
    assert got is out
    np.testing.assert_array_equal(cp.asnumpy(got), data[:8, :8])
    # Second read reuses the cached shard index (no fetch => no error).
    got2 = lowlevel.read_array(None, np.s_[8:16, :8], plan=plan)
    np.testing.assert_array_equal(cp.asnumpy(got2), data[8:16, :8])


def test_out_shape_mismatch_raises(zstd_sharded) -> None:
    root, _ = zstd_sharded
    plan = lowlevel.open_plan(root)
    with pytest.raises(ValueError, match="out must be shape"):
        lowlevel.read_array(None, np.s_[:8, :8], plan=plan, out=cp.empty((3, 3), dtype="uint16"))


def test_shuffle_zstd_chain(gpustore_tmpdir) -> None:
    """[Shuffle, Zstd] chain written by czarr codecs decodes bit-exact."""
    import czarr

    data = (np.arange(np.prod(SHAPE)) % 4999).astype("uint16").reshape(SHAPE)
    root = gpustore_tmpdir / "shuffled.zarr"
    with czarr.configure_gpu():
        arr = zarr.create_array(
            store=str(root),
            shape=SHAPE,
            chunks=(8, 8),
            dtype="uint16",
            fill_value=0,
            zarr_format=3,
            compressors=[czarr.Shuffle(elementsize=2), czarr.Zstd()],
        )
        arr[:] = cp.asarray(data)
    got = lowlevel.read_array(root, np.s_[:])
    np.testing.assert_array_equal(cp.asnumpy(got), data)


def test_blosc_store_decodes(gpustore_tmpdir) -> None:
    """CPU-written blosc(zstd+byteshuffle) store decodes through the native path."""
    data = (np.arange(64 * 64) % 1009).astype("uint16").reshape(64, 64)
    root = gpustore_tmpdir / "blosc.zarr"
    _write(
        root,
        data,
        dtype="uint16",
        chunks=(32, 64),
        compressors=[zarr.codecs.BloscCodec(cname="zstd", shuffle="shuffle", typesize=2)],
    )
    got = lowlevel.read_array(root, np.s_[:])
    np.testing.assert_array_equal(cp.asnumpy(got), data)


def test_unsupported_codec_raises(gpustore_tmpdir) -> None:
    data = np.arange(64, dtype="int32").reshape(8, 8)
    root = gpustore_tmpdir / "gzip.zarr"
    _write(root, data, dtype="int32", chunks=(4, 4), compressors=[zarr.codecs.GzipCodec()])
    with pytest.raises(NotImplementedError, match="gzip"):
        lowlevel.read_array(root, np.s_[:])


def test_empty_selection(zstd_sharded) -> None:
    root, _ = zstd_sharded
    got = lowlevel.read_array(root, np.s_[3:3, :])
    assert got.shape == (0, SHAPE[1])
