"""Sharding coalesce override is wired into zarr's codec registry.

``CzarrShardingCodec`` is an internal class — users never construct it
directly.  After ``configure_gpu()`` runs, zarr's registry resolves
``"sharding_indexed"`` to our coalescing variant, so any existing v3
sharded store reads through it transparently.

These tests validate the registry plumbing and bit-exact parity
against the stock decode path.  Real perf coverage lives in
``bench/storage/coalesce_compare.py`` (microbench).
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr
from zarr.codecs.sharding import ShardingCodec
from zarr.registry import get_codec_class
from zarr.storage import MemoryStore

import czarr
from czarr.codecs.sharding import CzarrShardingCodec


@pytest.fixture
def sharded_store():
    """Create a v3 sharded zarr via stock ShardingCodec.

    32x32 array, 32-wide shards along axis 0, 4-wide inner chunks —
    a sub-region slice yields a 3-of-8 partial-shard read that
    exercises the coalesce override.
    """
    rng = np.random.default_rng(0)
    data = rng.integers(0, 100, size=(32, 32), dtype=np.int32)
    store = MemoryStore()
    # ``compressors=()`` sidesteps an orthogonal czarr nvCOMP buffer-resize
    # bug on the GPU partial-shard decode path (tracked under
    # ``project_coalesce_status``).  The coalesce code we want to test
    # lives on the I/O side, so uncompressed inner chunks isolate the
    # signal cleanly.
    arr = zarr.create_array(
        store=store,
        shape=data.shape,
        chunks=(4, 32),
        shards=(32, 32),
        dtype=data.dtype,
        serializer=ShardingCodec(chunk_shape=(4, 32)),
        compressors=(),
        filters=(),
    )
    arr[:] = data
    return store, data


def test_registry_resolves_sharding_to_czarr_variant():
    """``configure_gpu()`` makes ``"sharding_indexed"`` resolve to our class."""
    czarr.configure_gpu()
    try:
        resolved = get_codec_class("sharding_indexed")
        assert resolved is CzarrShardingCodec
    finally:
        zarr.config.reset()


def test_stock_written_array_decodes_through_coalesce(sharded_store):
    """Stock-written sharded zarr is read through CzarrShardingCodec.

    Full-array read covers the total-shard branch (no coalesce engaged
    but the same code path); partial slices exercise
    :meth:`CzarrShardingCodec._decode_partial_single`.
    """
    store, data = sharded_store
    czarr.configure_gpu()
    try:
        arr = zarr.open_array(store=store, mode="r")
        # Full read.
        np.testing.assert_array_equal(_to_numpy(arr[:]), data)
        # Partial-shard read — coalesce path engaged.
        np.testing.assert_array_equal(_to_numpy(arr[4:16, :]), data[4:16, :])
    finally:
        zarr.config.reset()


def test_decode_partial_single_overridden():
    """Class hierarchy carries our override despite same codec_name."""
    assert issubclass(CzarrShardingCodec, ShardingCodec)
    # The override lives on the subclass; the parent's method is intact.
    parent_fn = ShardingCodec._decode_partial_single
    child_fn = CzarrShardingCodec._decode_partial_single
    assert parent_fn is not child_fn


def test_czarr_sharding_codec_constructor_defaults():
    """Knobs default to the values documented in the module docstring."""
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


def test_metadata_omits_knobs():
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


def test_from_dict_via_registry_uses_knob_defaults(sharded_store):
    """When zarr deserialises metadata our class picks up its defaults."""
    store, _ = sharded_store
    czarr.configure_gpu()
    try:
        arr = zarr.open_array(store=store, mode="r")
        # zarr stashes the codec instance under array.metadata.codecs.
        codecs = arr.metadata.codecs
        sharding = next(c for c in codecs if isinstance(c, ShardingCodec))
        assert isinstance(sharding, CzarrShardingCodec)
        assert sharding.max_fused_bytes == 64 << 20
        assert sharding.max_gap_bytes == 0
    finally:
        zarr.config.reset()


def _to_numpy(x):
    """Pull array contents to host regardless of cupy/numpy backing."""
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x)
