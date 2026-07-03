"""Read-range coalescing — damacy-inspired adjacent-chunk fusion.

zarr v3's ``ShardingCodec`` calls ``byte_getter.get`` once per chunk
within a shard in a serial ``await`` loop (zarr/codecs/sharding.py
~L504).  With cuFile on H100 each call costs ~1 ms, so a 32-chunk
partial shard read serializes 32 ms before any decode starts.

This module collapses a set of (offset, length) requests against the
same shard file into a smaller set of *fused* reads.  Adjacent or
near-adjacent ranges merge into a single read up to a configurable
byte cap; the caller slices originals out of the fused buffer.

This is purely the planning step — the read itself happens elsewhere.
The algorithm mirrors damacy's ``coalesce_chunks`` (see
``src/planner/coalesce.h`` upstream): sort → greedy fuse with a cap →
return fused windows + a per-request slicer.

Performance gain comes from three places, ordered by importance:

1. Fewer kernel-syscall round trips (one ``cuFileReadAsync`` instead
   of N).
2. No serial ``await`` chain — ShardingCodec's per-chunk loop awaits
   each get individually.
3. Larger reads better-saturate the storage backend (NVMe, NFS, GDS).

When the requests are already widely separated (e.g. sparse chunk
selections) the algorithm degrades cleanly to N un-fused windows.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ByteRange:
    """Closed-open byte range request.  ``length`` may be 0 for sentinel slots."""

    offset: int
    length: int

    @property
    def end(self) -> int:
        """One past the last byte — convenient for sort + merge math."""
        return self.offset + self.length


@dataclass(frozen=True)
class FusedRead:
    """One coalesced read window plus the mapping back to original requests.

    ``members`` lists ``(orig_index, intra_offset, length)`` tuples — the
    caller slices ``buffer[intra_offset : intra_offset + length]`` to
    recover request ``orig_index``'s payload.

    ``waste`` is bytes inside the fused window that no original request
    asked for.  A pure scheduler metric; not used at slice time.
    """

    offset: int
    length: int
    members: tuple[tuple[int, int, int], ...] = field(default_factory=tuple)
    waste: int = 0

    @property
    def end(self) -> int:
        """End offset (exclusive)."""
        return self.offset + self.length


def coalesce_ranges(
    ranges: Sequence[ByteRange],
    *,
    max_fused_bytes: int = 4 << 20,
    max_gap_bytes: int = 0,
) -> list[FusedRead]:
    """Sort + greedy-fuse byte ranges into fewer larger reads.

    Parameters
    ----------
    ranges:
        Input requests.  Each carries its original list index by position
        in the input ``ranges`` sequence.  Empty or zero-length entries
        are passed through as their own fused read so callers can still
        index by position.
    max_fused_bytes:
        Cap on a single fused read's length, in bytes.  A request that's
        already larger than this is emitted as its own window (the cap is
        best-effort — fusion never grows past it, never shrinks an
        individual request).
    max_gap_bytes:
        Bytes of "no one asked for this" the fuser will swallow between
        two adjacent ranges in order to merge them.  ``0`` (default) only
        fuses ranges that are exactly contiguous; larger values trade
        wasted bytes for fewer reads (typical: page size = 4096).

    Returns
    -------
    list[FusedRead]
        Sorted by fused offset.  ``sum(len(r.members) for r in result)``
        equals ``len(ranges)`` — every input request appears exactly
        once across all fused reads.
    """
    if max_fused_bytes <= 0:
        raise ValueError(f"max_fused_bytes must be > 0, got {max_fused_bytes}")
    if max_gap_bytes < 0:
        raise ValueError(f"max_gap_bytes must be >= 0, got {max_gap_bytes}")

    # Stamp each range with its original index; sort by offset then by
    # original index to keep the merge deterministic for zero-length
    # placeholders.
    indexed = sorted(
        ((r, i) for i, r in enumerate(ranges)),
        key=lambda pair: (pair[0].offset, pair[1]),
    )

    fused: list[FusedRead] = []
    cur_offset = 0
    cur_end = 0
    cur_members: list[tuple[int, int, int]] = []
    cur_waste = 0
    cur_open = False

    def _emit() -> None:
        if not cur_open:
            return
        fused.append(
            FusedRead(
                offset=cur_offset,
                length=cur_end - cur_offset,
                members=tuple(cur_members),
                waste=cur_waste,
            )
        )

    for r, orig_idx in indexed:
        if r.length == 0:
            # Pass through zero-length placeholders untouched — preserves
            # the caller's slot ordering.  Flush any open window first
            # since a zero-length read can't be part of a fused window.
            _emit()
            fused.append(FusedRead(offset=r.offset, length=0, members=((orig_idx, 0, 0),)))
            cur_open = False
            cur_members = []
            cur_waste = 0
            continue

        if r.length > max_fused_bytes:
            # Single request already larger than the cap — emit as its
            # own window after flushing whatever's open.
            _emit()
            fused.append(
                FusedRead(
                    offset=r.offset,
                    length=r.length,
                    members=((orig_idx, 0, r.length),),
                    waste=0,
                )
            )
            cur_open = False
            cur_members = []
            cur_waste = 0
            continue

        if not cur_open:
            cur_offset = r.offset
            cur_end = r.end
            cur_members = [(orig_idx, 0, r.length)]
            cur_waste = 0
            cur_open = True
            continue

        gap = r.offset - cur_end
        # Gap == 0 means exactly contiguous; gap > 0 means there's
        # un-requested bytes between current window and new range; gap
        # < 0 means overlap (a request fully inside the current window
        # — still fuse, just point intra_offset back).
        candidate_end = max(cur_end, r.end)
        candidate_len = candidate_end - cur_offset
        ok_gap = gap <= 0 or gap <= max_gap_bytes
        ok_cap = candidate_len <= max_fused_bytes

        if ok_gap and ok_cap:
            cur_members.append((orig_idx, r.offset - cur_offset, r.length))
            if gap > 0:
                cur_waste += gap
            cur_end = candidate_end
        else:
            _emit()
            cur_offset = r.offset
            cur_end = r.end
            cur_members = [(orig_idx, 0, r.length)]
            cur_waste = 0
            cur_open = True

    _emit()
    return fused


def slice_into_outputs(
    buffers: Sequence[memoryview | bytes | bytearray],
    fused: Sequence[FusedRead],
    n_originals: int,
) -> list[memoryview | bytes | bytearray | None]:
    """Slice fused-read buffers back into per-original-request views.

    Each entry of ``buffers`` corresponds to ``fused[i]`` and holds the
    bytes read for that fused window.  Returns a list of length
    ``n_originals`` where index ``j`` is a view (or sub-buffer) covering
    the bytes originally requested as ``ranges[j]``.

    Slices are returned as the same type as the input buffer (memoryview
    if input is memoryview / bytes / bytearray).  Caller is responsible
    for handing the views onward before the parent buffer is freed.
    """
    if len(buffers) != len(fused):
        raise ValueError(f"buffers length {len(buffers)} != fused length {len(fused)}")
    out: list[memoryview | bytes | bytearray | None] = [None] * n_originals
    for buf, win in zip(buffers, fused, strict=True):
        if isinstance(buf, (bytes, bytearray)):
            mv = memoryview(buf)
        else:
            mv = buf
        for orig_idx, intra_off, length in win.members:
            if length == 0:
                out[orig_idx] = mv[:0] if mv is not None else None
            else:
                out[orig_idx] = mv[intra_off : intra_off + length]
    return out
