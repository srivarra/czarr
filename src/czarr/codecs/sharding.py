"""Coalescing override of zarr's ShardingCodec.

zarr v3's ``ShardingCodec._decode_partial_single`` reads chunks-within-
a-shard in a serial ``await`` loop — one ``byte_getter.get`` per chunk
(zarr/codecs/sharding.py ~L504).  With cuFile on H100 each call costs
~1 ms, so a 32-chunk partial-shard read serializes 30+ ms before any
decode starts.

This subclass collects all chunk byte slices, runs them through
:func:`czarr.lowlevel.coalesce.coalesce_ranges`, and issues fused
reads — then slices per-chunk Buffer views out locally.

Microbench on H100 (``bench/storage/coalesce_compare.py``, untracked):

* 32 x 64 KiB chunks   → 22.2x faster (26.5 ms → 1.20 ms)
* 32 x 256 KiB chunks  → 17.7x
* 32 x 1 MiB chunks    →  6.6x
* 8 x 4 MiB chunks     →  2.3x

The coalesce params (``max_fused_bytes`` / ``max_gap_bytes``) are
runtime knobs — not persisted in metadata.  The on-disk name stays
``"sharding_indexed"`` so existing zarr v3 sharded stores round-trip
unchanged; opt-in by passing ``serializer=CzarrShardingCodec(...)`` at
array creation, or globally via :func:`czarr.configure_gpu` (TODO).
"""

import asyncio
from collections.abc import Iterable
from typing import ClassVar

from zarr.abc.codec import Codec
from zarr.abc.store import ByteGetter, RangeByteRequest
from zarr.codecs.sharding import ShardingCodec, ShardingCodecIndexLocation, _ShardingByteGetter
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer, BufferPrototype, NDBuffer
from zarr.core.chunk_grids import ChunkGrid
from zarr.core.common import JSON, ShapeLike
from zarr.core.indexing import SelectorTuple, get_indexer

from czarr.lowlevel.coalesce import ByteRange, coalesce_ranges

# Default knobs — tuned from the H100 microbench.  64 MiB is large
# enough to fuse a whole shard (typical shards are 32-128 MiB) without
# pinning a wasteful amount of device memory; gap=0 only fuses ranges
# that touch.
_DEFAULT_MAX_FUSED_BYTES = 64 << 20
_DEFAULT_MAX_GAP_BYTES = 0


class CzarrShardingCodec(ShardingCodec):
    """ShardingCodec subclass that coalesces partial-shard reads.

    Internal class — registered against the same codec name
    (``"sharding_indexed"``) as :class:`zarr.codecs.ShardingCodec` so
    zarr's registry resolves all existing v3 sharded metadata to this
    class on read.  Users don't import or construct it directly;
    :func:`czarr.configure_gpu` selects it via ``zarr.config``.

    Same on-disk format, same codec name, same metadata schema.  Only
    the partial-shard chunk-fetch loop changes — see
    :meth:`_decode_partial_single`.

    Two extra knobs control the coalesce, set at construction time
    (runtime-only — never written to metadata):

    * ``max_fused_bytes`` — cap on a single fused read.  Default 64 MiB.
    * ``max_gap_bytes`` — bytes of "no one asked for this" the fuser
      will swallow to merge two near-adjacent ranges.  Default 0.
    """

    # ClassVar surfaced for the registration loop in ``codecs/__init__``;
    # zarr's parent ShardingCodec uses a hard-coded "sharding_indexed"
    # literal and doesn't expose codec_name as a class attribute.
    codec_name: ClassVar[str] = "sharding_indexed"

    # Reserve slots so the frozen-dataclass parent's __slots__-like
    # behaviour (via object.__setattr__) carries through.
    max_fused_bytes: int
    max_gap_bytes: int

    def __init__(
        self,
        *,
        chunk_shape: ShapeLike,
        codecs: Iterable[Codec | dict[str, JSON]] | None = None,
        index_codecs: Iterable[Codec | dict[str, JSON]] | None = None,
        index_location: ShardingCodecIndexLocation | str | None = None,
        max_fused_bytes: int = _DEFAULT_MAX_FUSED_BYTES,
        max_gap_bytes: int = _DEFAULT_MAX_GAP_BYTES,
    ) -> None:
        # Build the kwarg dict carefully: pass only what was supplied
        # so we don't shadow zarr's defaults (e.g. BytesCodec() for
        # codecs).  The parent does its own object.__setattr__ on the
        # frozen fields.
        parent_kwargs: dict = {"chunk_shape": chunk_shape}
        if codecs is not None:
            parent_kwargs["codecs"] = codecs
        if index_codecs is not None:
            parent_kwargs["index_codecs"] = index_codecs
        if index_location is not None:
            parent_kwargs["index_location"] = index_location
        super().__init__(**parent_kwargs)
        object.__setattr__(self, "max_fused_bytes", int(max_fused_bytes))
        object.__setattr__(self, "max_gap_bytes", int(max_gap_bytes))

    async def _decode_partial_single(
        self,
        byte_getter: ByteGetter,
        selection: SelectorTuple,
        shard_spec: ArraySpec,
    ) -> NDBuffer | None:
        """Coalesced override of zarr's partial-shard decode.

        Diff vs upstream: the per-chunk serial ``await byte_getter.get``
        loop is replaced with a coalesce + ``asyncio.gather`` over the
        fused windows.  Everything else (total-shard branch, codec
        pipeline call, reshape) matches the parent.
        """
        shard_shape = shard_spec.shape
        chunk_shape = self.chunk_shape
        chunks_per_shard = self._get_chunks_per_shard(shard_spec)
        chunk_spec = self._get_chunk_spec(shard_spec)

        indexer = get_indexer(
            selection,
            shape=shard_shape,
            chunk_grid=ChunkGrid.from_sizes(shard_shape, chunk_shape),
        )

        out = shard_spec.prototype.nd_buffer.empty(
            shape=indexer.shape,
            dtype=shard_spec.dtype.to_native_dtype(),
            order=shard_spec.order,
        )

        indexed_chunks = list(indexer)
        all_chunk_coords = {chunk_coords for chunk_coords, *_ in indexed_chunks}

        if self._is_total_shard(all_chunk_coords, chunks_per_shard):
            # Whole-shard read is already one I/O; no coalesce needed.
            shard_dict_maybe = await self._load_full_shard_maybe(
                byte_getter=byte_getter,
                prototype=chunk_spec.prototype,
                chunks_per_shard=chunks_per_shard,
            )
            if shard_dict_maybe is None:
                return None
            shard_dict = shard_dict_maybe
        else:
            shard_index = await self._load_shard_index_maybe(byte_getter, chunks_per_shard)
            if shard_index is None:
                return None
            shard_dict = await self._coalesced_chunk_fetch(
                byte_getter,
                shard_index,
                all_chunk_coords,
                chunk_spec.prototype,
            )

        await self.codec_pipeline.read(
            [
                (
                    _ShardingByteGetter(shard_dict, chunk_coords),
                    chunk_spec,
                    chunk_selection,
                    out_selection,
                    is_complete_shard,
                )
                for chunk_coords, chunk_selection, out_selection, is_complete_shard in indexer
            ],
            out,
        )

        sel_shape = getattr(indexer, "sel_shape", None)
        if sel_shape is not None:
            return out.reshape(tuple(sel_shape))
        return out

    async def _coalesced_chunk_fetch(
        self,
        byte_getter: ByteGetter,
        shard_index,
        all_chunk_coords,
        prototype: BufferPrototype,
    ) -> dict:
        """Build coalesced reads + slice per-chunk Buffer views out.

        Walks the requested chunk coordinates, looks each up in the
        shard index, coalesces byte ranges, issues fused reads via
        ``asyncio.gather``, and slices per-chunk views into a
        ``shard_dict`` keyed by chunk coordinate.
        """
        coords_with_slices: list[tuple[tuple[int, ...], tuple[int, int]]] = []
        for chunk_coords in all_chunk_coords:
            chunk_byte_slice = shard_index.get_chunk_slice(chunk_coords)
            if chunk_byte_slice:
                coords_with_slices.append((chunk_coords, chunk_byte_slice))

        if not coords_with_slices:
            return {}

        ranges = [ByteRange(offset=s[0], length=s[1] - s[0]) for _, s in coords_with_slices]
        fused = coalesce_ranges(
            ranges,
            max_fused_bytes=self.max_fused_bytes,
            max_gap_bytes=self.max_gap_bytes,
        )

        # asyncio.gather over only the non-empty windows.  The store
        # may parallelise these (thread pool, cuFile batched I/O); the
        # primary win is still that one fused read replaces N
        # per-chunk reads regardless of how the store schedules them.
        nonempty_windows = [w for w in fused if w.length > 0]
        nonempty_buffers: list[Buffer | None] = list(
            await asyncio.gather(
                *(
                    byte_getter.get(
                        prototype=prototype,
                        byte_range=RangeByteRequest(w.offset, w.offset + w.length),
                    )
                    for w in nonempty_windows
                )
            )
        )

        # Re-align with the full fused list so zero-length windows
        # remain in position.
        nonempty_iter = iter(nonempty_buffers)
        materialized: list[Buffer | None] = [next(nonempty_iter) if w.length > 0 else None for w in fused]

        shard_dict: dict = {}
        for fbuf, w in zip(materialized, fused, strict=True):
            if fbuf is None:
                continue
            for orig_idx, intra_off, length in w.members:
                chunk_coords = coords_with_slices[orig_idx][0]
                shard_dict[chunk_coords] = fbuf[intra_off : intra_off + length]
        return shard_dict
