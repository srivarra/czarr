"""Unit tests for the byte-range coalescer."""

import pytest

from czarr.lowlevel.coalesce import ByteRange, coalesce_ranges


def _slice_members(buffers, fused, n_originals):
    """The production slicing pattern (see sharding + lowlevel.decode)."""
    out = [None] * n_originals
    for buf, win in zip(buffers, fused, strict=True):
        for orig_idx, intra, length in win.members:
            out[orig_idx] = buf[intra : intra + length]
    return out


class TestCoalesceRanges:
    """Single-shard byte-range fusion behaviour."""

    def test_empty_input(self) -> None:
        assert coalesce_ranges([]) == []

    def test_single_range_passthrough(self) -> None:
        r = ByteRange(offset=100, length=50)
        out = coalesce_ranges([r])
        assert len(out) == 1
        assert out[0].offset == 100
        assert out[0].length == 50
        assert out[0].members == ((0, 0, 50),)

    def test_exactly_adjacent_fuse(self) -> None:
        """Contiguous ranges fuse into one read regardless of max_gap."""
        rs = [
            ByteRange(offset=0, length=100),
            ByteRange(offset=100, length=200),
            ByteRange(offset=300, length=50),
        ]
        out = coalesce_ranges(rs, max_gap_bytes=0)
        assert len(out) == 1
        assert out[0].offset == 0
        assert out[0].length == 350
        # Members preserve original order via orig_index.
        assert out[0].members == ((0, 0, 100), (1, 100, 200), (2, 300, 50))

    def test_gap_blocks_fusion_when_max_gap_zero(self) -> None:
        """A 1-byte gap with default max_gap=0 splits the fusion."""
        rs = [
            ByteRange(offset=0, length=100),
            ByteRange(offset=101, length=50),
        ]
        out = coalesce_ranges(rs, max_gap_bytes=0)
        assert len(out) == 2

    def test_gap_within_cap_fuses(self) -> None:
        """A small gap within max_gap_bytes fuses into one window."""
        rs = [
            ByteRange(offset=0, length=100),
            ByteRange(offset=200, length=50),
        ]
        out = coalesce_ranges(rs, max_gap_bytes=4096)
        assert len(out) == 1
        assert out[0].offset == 0
        assert out[0].length == 250

    def test_cap_splits_fusion(self) -> None:
        """``max_fused_bytes`` stops fusion before the cap."""
        rs = [
            ByteRange(offset=0, length=400),
            ByteRange(offset=400, length=400),
            ByteRange(offset=800, length=400),
        ]
        out = coalesce_ranges(rs, max_fused_bytes=1000)
        # First pair fuses to 800B (under 1000); 3rd would push to 1200.
        assert len(out) == 2
        assert out[0].length == 800
        assert out[1].length == 400

    def test_oversized_request_emitted_alone(self) -> None:
        """A single request larger than the cap gets its own window."""
        rs = [
            ByteRange(offset=0, length=100),
            ByteRange(offset=200, length=10_000),
            ByteRange(offset=10_500, length=100),
        ]
        out = coalesce_ranges(rs, max_fused_bytes=1024, max_gap_bytes=4096)
        # Three windows: first range, oversized middle, last range.
        assert len(out) == 3
        assert out[1].length == 10_000
        assert out[1].members == ((1, 0, 10_000),)

    def test_unsorted_input_sorted_internally(self) -> None:
        """Input order doesn't matter; orig_index is preserved in members."""
        rs = [
            ByteRange(offset=200, length=100),  # orig_index=0
            ByteRange(offset=0, length=100),  # orig_index=1
            ByteRange(offset=100, length=100),  # orig_index=2
        ]
        out = coalesce_ranges(rs, max_gap_bytes=0)
        # All three contiguous after sort.
        assert len(out) == 1
        # orig_index in members must match the input position.
        orig_indices = sorted(m[0] for m in out[0].members)
        assert orig_indices == [0, 1, 2]

    def test_zero_length_passthrough(self) -> None:
        """Zero-length requests are kept as standalone zero-length windows.

        They flush any open fusion window — a zero-length read has no
        bytes to slice out of a fused buffer, so the simplest contract
        is that it never participates in fusion.  Callers index by
        position so the placeholder must still appear.
        """
        rs = [
            ByteRange(offset=0, length=100),
            ByteRange(offset=100, length=0),
            ByteRange(offset=100, length=100),
        ]
        out = coalesce_ranges(rs)
        zero_windows = [r for r in out if r.length == 0]
        # All three originals must appear exactly once across windows.
        seen = sorted(m[0] for w in out for m in w.members)
        assert seen == [0, 1, 2]
        assert len(zero_windows) == 1
        assert zero_windows[0].members == ((1, 0, 0),)

    def test_overlapping_fuses_without_growth(self) -> None:
        """Overlapping ranges fuse to the union, not the sum."""
        rs = [
            ByteRange(offset=0, length=100),
            ByteRange(offset=50, length=100),  # overlaps first 50B
        ]
        out = coalesce_ranges(rs, max_gap_bytes=0)
        assert len(out) == 1
        assert out[0].offset == 0
        assert out[0].length == 150  # union, not 200

    def test_invalid_max_fused_bytes_raises(self) -> None:
        with pytest.raises(ValueError, match="max_fused_bytes"):
            coalesce_ranges([ByteRange(0, 1)], max_fused_bytes=0)

    def test_invalid_max_gap_raises(self) -> None:
        with pytest.raises(ValueError, match="max_gap_bytes"):
            coalesce_ranges([ByteRange(0, 1)], max_gap_bytes=-1)

    def test_every_input_appears_once(self) -> None:
        """Coverage invariant: each original request shows up exactly once."""
        rs = [ByteRange(offset=i * 100, length=80) for i in range(20)]
        out = coalesce_ranges(rs, max_fused_bytes=512, max_gap_bytes=32)
        seen = sorted(m[0] for win in out for m in win.members)
        assert seen == list(range(20))


class TestMemberSlicing:
    """Round-trip: coalesce + read + member slicing yields the original payloads."""

    def test_round_trip_three_chunks(self) -> None:
        """Simulate a fused read by concatenating chunk bytes."""
        # Three pretend chunks, each 10 bytes, at offsets 0, 10, 20.
        chunks = [bytes([i] * 10) for i in range(3)]
        rs = [ByteRange(offset=i * 10, length=10) for i in range(3)]
        fused = coalesce_ranges(rs, max_gap_bytes=0)
        assert len(fused) == 1
        # Simulated buffer: the bytes the fused read would have returned.
        buffer = b"".join(chunks)
        out = _slice_members([buffer], fused, n_originals=3)
        for i, view in enumerate(out):
            assert view is not None
            assert bytes(view) == chunks[i]

    def test_round_trip_with_gap_waste(self) -> None:
        """Gap bytes are read but ignored; original slices come out clean."""
        # Chunks at 0..10 and 20..30; bytes 10..20 are "waste" inside fuse.
        chunk_a = b"A" * 10
        chunk_b = b"B" * 10
        waste = b"X" * 10
        rs = [ByteRange(offset=0, length=10), ByteRange(offset=20, length=10)]
        fused = coalesce_ranges(rs, max_gap_bytes=16)
        assert len(fused) == 1
        buffer = chunk_a + waste + chunk_b
        out = _slice_members([buffer], fused, n_originals=2)
        assert bytes(out[0]) == chunk_a  # type: ignore[arg-type]
        assert bytes(out[1]) == chunk_b  # type: ignore[arg-type]

    def test_round_trip_uncoalesced(self) -> None:
        """Cap forces no fusion; each request gets its own buffer + slice."""
        chunks = [bytes([1] * 5), bytes([2] * 5)]
        rs = [ByteRange(offset=0, length=5), ByteRange(offset=1_000_000, length=5)]
        fused = coalesce_ranges(rs, max_fused_bytes=10, max_gap_bytes=0)
        assert len(fused) == 2
        out = _slice_members(chunks, fused, n_originals=2)
        assert bytes(out[0]) == chunks[0]  # type: ignore[arg-type]
        assert bytes(out[1]) == chunks[1]  # type: ignore[arg-type]
