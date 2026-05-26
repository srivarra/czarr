"""Smoke + correctness tests for :class:`CzarrShardingCodec`.

Validates that the coalescing override produces bit-identical output
to zarr's stock ShardingCodec on the same sharded store.  Real perf
coverage lives in ``bench/storage/coalesce_compare.py`` (microbench)
and a follow-up end-to-end sbatch (TODO).
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr
from zarr.codecs.sharding import ShardingCodec
from zarr.storage import MemoryStore

from czarr.codecs.sharding import CzarrShardingCodec


@pytest.fixture
def sharded_array(tmp_path):
    """Create a v3 sharded zarr with a known pattern.

    32×32 array, 32-wide shards along axis 0, 4-wide inner chunks —
    yields a 4-chunk-per-shard partial-shard read when we slice a
    sub-region.
    """
    rng = np.random.default_rng(0)
    data = rng.integers(0, 100, size=(32, 32), dtype=np.int32)
    store = MemoryStore()
    arr = zarr.create_array(
        store=store,
        shape=data.shape,
        chunks=(4, 32),  # inner chunk shape
        shards=(32, 32),  # shard shape — one shard covers the whole array
        dtype=data.dtype,
        serializer=ShardingCodec(chunk_shape=(4, 32)),
    )
    arr[:] = data
    return store, data


def test_czarr_sharding_codec_matches_stock_full_read(sharded_array):
    """Full-array read through CzarrShardingCodec returns identical bytes."""
    store, data = sharded_array
    # Re-open the array but replace the serializer.  Stock ShardingCodec
    # wrote the bytes; CzarrShardingCodec must read them back unchanged.
    arr = zarr.open_array(
        store=store,
        mode="r",
        serializer=CzarrShardingCodec(chunk_shape=(4, 32)),
    )
    np.testing.assert_array_equal(np.asarray(arr[:]), data)


def test_czarr_sharding_codec_matches_stock_partial_read(sharded_array):
    """Partial-shard read (the path coalesce intercepts) is bit-identical."""
    store, data = sharded_array
    arr = zarr.open_array(
        store=store,
        mode="r",
        serializer=CzarrShardingCodec(chunk_shape=(4, 32)),
    )
    # Slice spans 3 of 8 inner chunks within the shard — exercises the
    # _decode_partial_single override.
    out = np.asarray(arr[4:16, :])
    np.testing.assert_array_equal(out, data[4:16, :])


def test_czarr_sharding_codec_constructor_defaults():
    """Defaults match the values documented in the module docstring."""
    codec = CzarrShardingCodec(chunk_shape=(4, 32))
    assert codec.max_fused_bytes == 64 << 20
    assert codec.max_gap_bytes == 0


def test_czarr_sharding_codec_constructor_explicit():
    """Explicit knobs override defaults."""
    codec = CzarrShardingCodec(
        chunk_shape=(4, 32),
        max_fused_bytes=8 << 20,
        max_gap_bytes=4096,
    )
    assert codec.max_fused_bytes == 8 << 20
    assert codec.max_gap_bytes == 4096


def test_czarr_sharding_codec_metadata_omits_knobs(sharded_array):
    """``to_dict`` must NOT include the coalesce knobs.

    The override inherits ``to_dict`` from the parent unchanged — these
    are runtime-only fields.  Pin that behaviour so a future refactor
    can't accidentally leak them.
    """
    codec = CzarrShardingCodec(chunk_shape=(4, 32), max_fused_bytes=8 << 20)
    d = codec.to_dict()
    assert d["name"] == "sharding_indexed"
    cfg = d["configuration"]
    assert "max_fused_bytes" not in cfg
    assert "max_gap_bytes" not in cfg


def test_czarr_sharding_codec_aggressive_coalesce_still_correct(sharded_array):
    """Setting a huge gap cap fuses everything; output must still match."""
    store, data = sharded_array
    arr = zarr.open_array(
        store=store,
        mode="r",
        serializer=CzarrShardingCodec(
            chunk_shape=(4, 32),
            max_fused_bytes=64 << 20,
            max_gap_bytes=64 << 20,  # swallow any gap up to 64 MiB
        ),
    )
    np.testing.assert_array_equal(np.asarray(arr[4:16, :]), data[4:16, :])
