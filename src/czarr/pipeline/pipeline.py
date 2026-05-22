"""``CzarrPipeline`` — GPU-native subclass of zarr v3's BatchedCodecPipeline.

Phase 2 of the refactor (epic ianyfe7m).

In zarr-python 3.x, a :class:`zarr.abc.codec.CodecPipeline` is
instantiated per array call and orchestrates the codec chain.  The
stock :class:`zarr.core.codec_pipeline.BatchedCodecPipeline` already
batches per-codec decode calls across all chunks of a selection, so the
expensive cross-chunk fan-out is *not* the bottleneck.

What this subclass changes:

* Owns a class-level :class:`StreamPool` and :class:`PinnedHostPool`
  (from Phase 1) so individual codecs can reach across into shared
  substrate without having to thread these instances through every
  call site.
* Provides :func:`configure` for tuning the substrate from
  :func:`czarr.configure_gpu`.

The actual GPU-direct passthrough that eliminates the
host-round-trip lives in :class:`CudaBytesBytesCodec._batch_sync` — when
the input chunk is already a ``gpu.Buffer`` it bypasses
``to_bytes() → host → .cuda()`` entirely.  This is the largest
per-chunk win and applies whether the user opts in via
``codec_pipeline.path`` or per-array.

Register globally via :func:`czarr.configure_gpu` (sets
``codec_pipeline.path = "czarr.pipeline.CzarrPipeline"``).  Per-array
opt-in is also supported by passing ``codec_pipeline=CzarrPipeline`` to
``zarr.create_array`` / ``zarr.open``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from zarr.core.codec_pipeline import BatchedCodecPipeline, fill_value_or_default

from czarr.pipeline.pinned import PinnedHostPool
from czarr.pipeline.streams import StreamPool

if TYPE_CHECKING:
    from collections.abc import Iterable

    from zarr.abc.codec import GetResult
    from zarr.abc.store import ByteGetter
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import NDBuffer
    from zarr.core.indexing import SelectorTuple


class CzarrPipeline(BatchedCodecPipeline):
    """zarr v3 CodecPipeline with GPU-aware staging + shared substrate."""

    # Class-level shared substrate.  Lazily initialised on first
    # ``get_*_pool`` access; reconfigurable via :func:`configure`.  Static
    # state is fine here — these primitives are device-bound and
    # process-global by nature.
    _stream_pool: ClassVar[StreamPool | None] = None
    _pinned_pool: ClassVar[PinnedHostPool | None] = None

    @classmethod
    def get_stream_pool(cls) -> StreamPool:
        """Return the shared StreamPool, lazily creating it on first call."""
        if cls._stream_pool is None:
            cls._stream_pool = StreamPool(size=4)
        return cls._stream_pool

    @classmethod
    def get_pinned_pool(cls) -> PinnedHostPool:
        """Return the shared PinnedHostPool, lazily creating it on first call."""
        if cls._pinned_pool is None:
            cls._pinned_pool = PinnedHostPool()
        return cls._pinned_pool

    @classmethod
    def configure(
        cls,
        *,
        stream_pool_size: int = 4,
        pinned_prealloc: Iterable[tuple[int, int]] | None = None,
    ) -> None:
        """Reconfigure the shared substrate; closes existing pools.

        Call from :func:`czarr.configure_gpu` to set defaults at process
        startup.  Subsequent reconfiguration is allowed but should be
        done from a quiescent state (no in-flight pipeline calls).
        """
        if cls._stream_pool is not None:
            cls._stream_pool.close()
        if cls._pinned_pool is not None:
            cls._pinned_pool.close()
        cls._stream_pool = StreamPool(size=stream_pool_size)
        cls._pinned_pool = PinnedHostPool(prealloc=list(pinned_prealloc) if pinned_prealloc else None)

    async def read_batch(
        self,
        batch_info: Iterable[tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        out: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> tuple[GetResult, ...]:
        """Override of zarr's per-chunk fetch with a batched ``store.get_many`` path.

        Whenever the entire batch's byte-getters resolve to the same
        :class:`czarr.storage.GPULocalStore`, we call ``store.get_many``
        once — cutting per-chunk Python + cuFile setup overhead that
        Phase 0 measured at ~3 ms per chunk.  Otherwise the call falls
        through to the parent's ``concurrent_map`` path.
        """
        from zarr.abc.codec import GetResult

        from czarr.storage import GPULocalStore

        batch_info_list = list(batch_info)
        if self.supports_partial_decode or not batch_info_list:
            return await super().read_batch(batch_info_list, out, drop_axes)

        # All byte-getters in a single zarr selection always share the
        # same store; check via the first one and assert downstream.  If
        # the store doesn't expose get_many or isn't a GPULocalStore,
        # punt back to zarr's default path.  Sharding wraps byte-getters
        # in a `_ShardingByteGetter` that has no `.store` attribute —
        # those go through the parent path too (the shard is read in one
        # shot already; per-inner-chunk batching happens via the inner
        # pipeline's own read_batch).
        first_bg = batch_info_list[0][0]
        first_store = getattr(first_bg, "store", None)
        if not isinstance(first_store, GPULocalStore) or not hasattr(first_store, "get_many"):
            return await super().read_batch(batch_info_list, out, drop_axes)

        # Single batched fetch.  zarr's default uses concurrent_map; we
        # use store.get_many which underneath pre-registers handles + does
        # parallel cuFile reads.
        prototype = batch_info_list[0][1].prototype
        chunk_bytes_batch = await first_store.get_many(
            [bg.path for bg, _, _, _, _ in batch_info_list],
            prototype=prototype,
        )

        chunk_array_batch = await self.decode_batch(
            [
                (chunk_bytes, chunk_spec)
                for chunk_bytes, (_, chunk_spec, *_) in zip(chunk_bytes_batch, batch_info_list, strict=False)
            ],
        )

        results: list[GetResult] = []
        for chunk_array, (_, chunk_spec, chunk_selection, out_selection, _) in zip(
            chunk_array_batch, batch_info_list, strict=False
        ):
            if chunk_array is not None:
                tmp = chunk_array[chunk_selection]
                if drop_axes:
                    tmp = tmp.squeeze(axis=drop_axes)
                out[out_selection] = tmp
                results.append(GetResult(status="present"))
            else:
                out[out_selection] = fill_value_or_default(chunk_spec)
                results.append(GetResult(status="missing"))
        return tuple(results)
