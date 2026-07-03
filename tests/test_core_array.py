"""Tests for czarr.core.Array / AsyncArray (GPU required for reads)."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
import zarr

cp = pytest.importorskip("cupy")

from czarr.core import Array, AsyncArray  # noqa: E402

SHAPE = (16, 12)


@pytest.fixture
def store(gpustore_tmpdir):
    root = gpustore_tmpdir / "core.zarr"
    arr = zarr.create_array(
        store=str(root),
        shape=SHAPE,
        chunks=(4, 4),
        shards=(8, 8),
        dtype="uint16",
        fill_value=7,
        zarr_format=3,
        attributes={"who": "czarr"},
    )
    data = (np.arange(np.prod(SHAPE)) % 251).astype("uint16").reshape(SHAPE)
    arr[:8, :] = data[:8, :]  # shard row 1 untouched -> fill_value
    expected = np.full(SHAPE, 7, dtype="uint16")
    expected[:8, :] = data[:8, :]
    return root, expected


class TestMetadata:
    def test_open_properties(self, store) -> None:
        root, _ = store
        arr = Array.open(root)
        assert arr.shape == SHAPE
        assert arr.dtype == np.uint16
        assert arr.ndim == 2
        assert arr.chunk_shape == (4, 4)  # inner chunks, not shards
        assert arr.grid_shape == (4, 3)
        assert arr.attrs == {"who": "czarr"}
        assert arr.metadata["zarr_format"] == 3
        assert "sharded" in repr(arr)

    def test_from_metadata_no_io(self, store) -> None:
        root, _ = store
        metadata = Array.open(root).metadata
        arr = Array.from_metadata(metadata, root)
        assert arr.shape == SHAPE


class TestReads:
    def test_getitem_parity(self, store) -> None:
        root, expected = store
        arr = Array.open(root)
        np.testing.assert_array_equal(cp.asnumpy(arr[:]), expected)
        np.testing.assert_array_equal(cp.asnumpy(arr[2:10, 3:9]), expected[2:10, 3:9])

    def test_ndim_preserving(self, store) -> None:
        root, expected = store
        arr = Array.open(root)
        got = arr[5]
        assert got.shape == (1, SHAPE[1])
        np.testing.assert_array_equal(cp.asnumpy(got)[0], expected[5])

    def test_retrieve_chunk(self, store) -> None:
        root, expected = store
        arr = Array.open(root)
        got = arr.retrieve_chunk((1, 2))
        np.testing.assert_array_equal(cp.asnumpy(got), expected[4:8, 8:12])

    def test_retrieve_chunk_oob(self, store) -> None:
        root, _ = store
        with pytest.raises(IndexError, match="chunk index"):
            Array.open(root).retrieve_chunk((9, 0))

    def test_retrieve_missing_chunk_is_fill(self, store) -> None:
        root, _ = store
        got = Array.open(root).retrieve_chunk((3, 0))  # shard row 1: unwritten
        assert bool((got == 7).all())

    def test_retrieve_encoded_chunk(self, store) -> None:
        root, expected = store
        arr = Array.open(root)
        raw = arr.retrieve_encoded_chunk((0, 0))
        assert raw is not None
        assert raw.dtype == cp.uint8
        # Round-trip: nvCOMP-decode the raw zstd bytes and compare.
        from nvidia import nvcomp

        codec = nvcomp.Codec(algorithm="Zstd", bitstream_kind=nvcomp.BitstreamKind.RAW)
        (decoded,) = codec.decode([nvcomp.as_array(raw)])
        got = cp.asarray(decoded).view(cp.uint16).reshape(4, 4)
        np.testing.assert_array_equal(cp.asnumpy(got), expected[:4, :4])

    def test_retrieve_encoded_missing_is_none(self, store) -> None:
        root, _ = store
        assert Array.open(root).retrieve_encoded_chunk((3, 0)) is None

    def test_read_options_out(self, store) -> None:
        root, expected = store
        arr = Array.open(root)
        out = cp.empty((8, 8), dtype="uint16")
        got = arr.retrieve_array_subset(np.s_[:8, :8], out=out)
        assert got is out
        np.testing.assert_array_equal(cp.asnumpy(out), expected[:8, :8])


class TestAsync:
    def test_async_dual(self, store) -> None:
        root, expected = store

        async def go():
            arr = AsyncArray.open(root)
            assert arr.shape == SHAPE  # delegated property
            sub, chunk, raw = await asyncio.gather(
                arr.retrieve_array_subset(np.s_[:8, :8]),
                arr.retrieve_chunk((0, 0)),
                arr.retrieve_encoded_chunk((3, 0)),
            )
            return sub, chunk, raw

        sub, chunk, raw = asyncio.run(go())
        np.testing.assert_array_equal(cp.asnumpy(sub), expected[:8, :8])
        np.testing.assert_array_equal(cp.asnumpy(chunk), expected[:4, :4])
        assert raw is None
