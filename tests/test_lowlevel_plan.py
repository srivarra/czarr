"""Tests for czarr.lowlevel.plan — metadata parse, selection, range planning.

Host-only: fixture stores are written with zarr-python (CPU path), and
plan.py itself never imports cupy.
"""

import numpy as np
import pytest
import zarr

from czarr.lowlevel import DecodePlan, normalize_selection, open_plan

SHAPE = (16, 12)
CHUNKS = (4, 4)


@pytest.fixture
def plain_store(tmp_path):
    """Plain (unsharded) zstd array; chunk row 3 left unwritten (fill value)."""
    root = tmp_path / "plain.zarr"
    arr = zarr.create_array(
        store=str(root),
        shape=SHAPE,
        chunks=CHUNKS,
        dtype="float32",
        fill_value=0,
        zarr_format=3,
    )
    data = np.arange(12 * 12, dtype="float32").reshape(12, 12)
    arr[:12, :] = data  # rows 12:16 (chunk row 3) never written
    return root, data


@pytest.fixture
def sharded_store(tmp_path):
    """Sharded array: 8x8 shards of 4x4 inner chunks; one shard untouched."""
    root = tmp_path / "sharded.zarr"
    arr = zarr.create_array(
        store=str(root),
        shape=SHAPE,
        chunks=(4, 4),
        shards=(8, 8),
        dtype="uint16",
        fill_value=0,
        zarr_format=3,
    )
    data = (np.arange(16 * 12) % 4999).astype("uint16").reshape(16, 12)
    arr[:8, :] = data[:8, :]  # shard row 0 fully written
    arr[8:12, :4] = data[8:12, :4]  # shard (1,0) partially written
    return root, data


# ---------------------------------------------------------------------------
# normalize_selection — zarrista/zarrs semantics
# ---------------------------------------------------------------------------


class TestNormalizeSelection:
    SHAPE3 = (10, 6, 8)

    @pytest.mark.parametrize(
        ("sel", "expected"),
        [
            (np.s_[:], ((0, 10), (0, 6), (0, 8))),
            (np.s_[...], ((0, 10), (0, 6), (0, 8))),
            (3, ((3, 4), (0, 6), (0, 8))),
            (-1, ((9, 10), (0, 6), (0, 8))),
            (np.s_[2:5], ((2, 5), (0, 6), (0, 8))),
            (np.s_[2:100], ((2, 10), (0, 6), (0, 8))),  # stop clamped
            (np.s_[-4:-1], ((6, 9), (0, 6), (0, 8))),
            (np.s_[5:2], ((5, 5), (0, 6), (0, 8))),  # empty, start kept sane
            (np.s_[1, 2:4], ((1, 2), (2, 4), (0, 8))),
            (np.s_[..., 0], ((0, 10), (0, 6), (0, 1))),
            (np.s_[1, ..., 2:], ((1, 2), (0, 6), (2, 8))),
            ((), ((0, 10), (0, 6), (0, 8))),
            (np.s_[2:4, 1, 3], ((2, 4), (1, 2), (3, 4))),  # ndim-preserving int axes
        ],
    )
    def test_accepted(self, sel, expected) -> None:
        assert normalize_selection(sel, self.SHAPE3) == expected

    @pytest.mark.parametrize(
        ("sel", "exc"),
        [
            (np.s_[::2], NotImplementedError),  # strided
            (np.s_[::-1], NotImplementedError),  # reversed
            (None, NotImplementedError),  # newaxis
            (True, TypeError),  # bool is not 1
            (np.True_, TypeError),
            ("field", TypeError),
            (np.array([0, 1]), TypeError),  # fancy
            (10, IndexError),  # OOB
            (-11, IndexError),
            (np.s_[0, 0, 0, 0], IndexError),  # too many indices
            (np.s_[..., 0, ...], IndexError),  # multi-ellipsis
        ],
    )
    def test_rejected(self, sel, exc) -> None:
        with pytest.raises(exc):
            normalize_selection(sel, self.SHAPE3)


# ---------------------------------------------------------------------------
# open_plan — metadata parse
# ---------------------------------------------------------------------------


class TestOpenPlan:
    def test_plain_metadata(self, plain_store) -> None:
        root, _ = plain_store
        plan = open_plan(root)
        assert isinstance(plan, DecodePlan)
        assert plan.shape == SHAPE
        assert plan.dtype == np.float32
        assert plan.chunk_shape == CHUNKS
        assert plan.shard is None
        assert plan.decode_chunk_shape == CHUNKS
        assert plan.grid_shape == (4, 3)
        assert any(c.get("name") == "zstd" for c in plan.codecs)

    def test_sharded_metadata(self, sharded_store) -> None:
        root, _ = sharded_store
        plan = open_plan(root)
        assert plan.chunk_shape == (8, 8)  # outer grid = shard shape
        assert plan.shard is not None
        assert plan.shard.inner_chunk_shape == (4, 4)
        assert plan.decode_chunk_shape == (4, 4)
        assert plan.units_per_shard == (2, 2)
        assert plan.grid_shape == (4, 3)
        assert plan.shard.index_location in ("end", "start")

    def test_chunk_key_default_encoding(self, plain_store) -> None:
        root, _ = plain_store
        plan = open_plan(root)
        assert plan.chunk_key((1, 2)) == "c/1/2"

    def test_non_v3_rejected(self, tmp_path) -> None:
        (tmp_path / "zarr.json").write_text('{"zarr_format": 2}')
        with pytest.raises(ValueError, match="zarr v3"):
            open_plan(tmp_path)


# ---------------------------------------------------------------------------
# ranges — plain arrays
# ---------------------------------------------------------------------------


class TestPlainRanges:
    def test_full_selection_lists_written_chunks(self, plain_store) -> None:
        root, _ = plain_store
        plan = open_plan(root)
        reqs = plan.ranges(np.s_[:])
        # 4x3 grid selected; chunk row 3 (4 chunk coords... 3 files) unwritten.
        coords = {m[0] for r in reqs for m in r.members}
        assert coords == {(i, j) for i in range(3) for j in range(3)}
        for r in reqs:
            assert r.offset == 0
            assert r.nbytes == r.path.stat().st_size
            assert len(r.members) == 1

    def test_selection_maps_to_chunk_files(self, plain_store) -> None:
        root, _ = plain_store
        plan = open_plan(root)
        reqs = plan.ranges(np.s_[0:4, 4:8])
        assert len(reqs) == 1
        assert reqs[0].path == root / "c/0/1"

    def test_missing_chunks_omitted(self, plain_store) -> None:
        root, _ = plain_store
        plan = open_plan(root)
        reqs = plan.ranges(np.s_[12:16, :])  # entirely unwritten
        assert reqs == []

    def test_empty_selection(self, plain_store) -> None:
        root, _ = plain_store
        plan = open_plan(root)
        assert plan.ranges(np.s_[5:5, :]) == []


# ---------------------------------------------------------------------------
# ranges — sharded arrays
# ---------------------------------------------------------------------------


class TestShardedRanges:
    def test_members_cover_selected_units_exactly_once(self, sharded_store) -> None:
        root, _ = sharded_store
        plan = open_plan(root)
        reqs = plan.ranges(np.s_[:8, :8])  # shard (0,0): 4 inner chunks, all written
        members = [m for r in reqs for m in r.members]
        coords = [m[0] for m in members]
        assert sorted(coords) == sorted({(i, j) for i in range(2) for j in range(2)})
        assert len(coords) == len(set(coords))
        for r in reqs:
            assert r.path == root / "c/0/0"

    def test_adjacent_ranges_fuse(self, sharded_store) -> None:
        root, _ = sharded_store
        plan = open_plan(root)
        reqs = plan.ranges(np.s_[:8, :8])
        # zarr writes a full shard contiguously — expect fewer requests than
        # inner chunks (gap=0 fuses only touching ranges, so allow >=1).
        n_members = sum(len(r.members) for r in reqs)
        assert n_members == 4
        assert len(reqs) < 4, f"no fusion happened: {reqs}"

    def test_intra_offsets_slice_correct_bytes(self, sharded_store) -> None:
        root, _ = sharded_store
        plan = open_plan(root)
        reqs = plan.ranges(np.s_[:8, :8])
        # Slicing the fused window at (intra_offset, length) must equal the
        # bytes at the index-recorded absolute offset.
        for r in reqs:
            with open(r.path, "rb") as f:
                f.seek(r.offset)
                window = f.read(r.nbytes)
            index = plan._shard_index((0, 0), r.path, None)
            for coords, intra, length in r.members:
                within = (coords[0] % 2, coords[1] % 2)
                abs_offset, abs_len = (int(x) for x in index[within])
                assert abs_len == length
                assert window[intra : intra + length] == open(r.path, "rb").read()[abs_offset : abs_offset + length]

    def test_partial_shard_missing_inner_chunks_omitted(self, sharded_store) -> None:
        root, _ = sharded_store
        plan = open_plan(root)
        # Shard (1,0) has only inner chunk (2,0) written (rows 8:12, cols 0:4).
        reqs = plan.ranges(np.s_[8:16, :8])
        coords = {m[0] for r in reqs for m in r.members}
        assert coords == {(2, 0)}

    def test_missing_shard_file_omitted(self, sharded_store) -> None:
        root, _ = sharded_store
        plan = open_plan(root)
        reqs = plan.ranges(np.s_[8:16, 8:12])  # shard (1,1) never written
        assert reqs == []

    def test_shard_index_cached(self, sharded_store) -> None:
        root, _ = sharded_store
        plan = open_plan(root)
        plan.ranges(np.s_[:8, :8])
        assert (0, 0) in plan._shard_indexes
        calls = []
        plan.ranges(np.s_[:4, :4], fetch=lambda *a: calls.append(a))
        assert calls == []  # cache hit — fetch never called

    def test_max_fused_bytes_limits_windows(self, sharded_store) -> None:
        root, _ = sharded_store
        plan = open_plan(root)
        fused = plan.ranges(np.s_[:8, :8])
        split = plan.ranges(np.s_[:8, :8], max_fused_bytes=1)  # nothing can fuse
        assert len(split) >= len(fused)
        assert sum(len(r.members) for r in split) == 4
