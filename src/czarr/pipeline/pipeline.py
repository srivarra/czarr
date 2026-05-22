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

from zarr.core.codec_pipeline import BatchedCodecPipeline

from czarr.pipeline.pinned import PinnedHostPool
from czarr.pipeline.streams import StreamPool

if TYPE_CHECKING:
    from collections.abc import Iterable


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

    # NOTE: a read_batch override that routes through GPULocalStore.get_many
    # was tried (Phase 2 of epic z3hd9ph7) and benched on Bruno H100.
    # Result: a 2-4x REGRESSION versus zarr's stock concurrent_map path.
    # Root cause: on H100 with real GDS, per-cuFile-call overhead is ~1 ms,
    # while the serial open + handle_register loop inside get_many costs
    # ~1 ms per chunk too.  The default concurrent_map path runs the
    # per-call overhead 32-way parallel and ends up faster.  On A40
    # (compat mode, ~3 ms per call) the override won 1.34x — but the
    # H100 regression made the override a net negative across the
    # hardware we target.
    #
    # GPULocalStore.get_many stays in place as a building block; the real
    # speedup will come from Phase 3 (cuFile batched I/O API:
    # cuFileBatchIOSetUp/Submit/GetStatus) which collapses the open +
    # register cycle into one driver call.  Until then, no override.
