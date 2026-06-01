"""``CzarrPipeline`` — GPU-native subclass of zarr v3's BatchedCodecPipeline.

In zarr-python 3.x, a :class:`zarr.abc.codec.CodecPipeline` is
instantiated per array call and orchestrates the codec chain.

What this subclass changes:

* Owns a class-level :class:`PinnedHostPool` so individual codecs can
  reach across into shared substrate without having to thread the
  instance through every call site.
* Adds NVTX-instrumented ``read_batch`` / ``write_batch`` so the nsys
  timeline shows the read + decode interleave that fires when
  ``codec_pipeline.batch_size`` is set below ``len(chunks)``.
* Provides :func:`configure` for tuning the substrate from
  :func:`czarr.configure_gpu`.

The actual GPU-direct passthrough that eliminates the
host-round-trip lives in :class:`CudaBytesBytesCodec._batch_sync` — when
the input chunk is already a ``gpu.Buffer`` it bypasses
``to_bytes() → host → .cuda()`` entirely.  This is the largest
per-chunk win and applies whether the user opts in via
``codec_pipeline.path`` or per-array.

Read/decode overlap: zarr's :meth:`BatchedCodecPipeline.read` already
splits ``batch_info`` into batches of ``self.batch_size`` and dispatches
them via ``concurrent_map`` up to ``async.concurrency``.  Inside each
batch, reads happen concurrently and decode runs after they all
arrive.  Across batches, the decode of batch K runs concurrently with
the reads of batch K+1.  So the lever for overlap is just ``batch_size``
— ``configure_gpu(decode_batch_size=8)`` enables it by default.

The NVTX wrappers below make the overlap visible in nsys; the actual
concurrency comes from zarr's pipeline machinery.

Register globally via :func:`czarr.configure_gpu` (sets
``codec_pipeline.path = "czarr.pipeline.CzarrPipeline"``).  Per-array
opt-in is also supported by passing ``codec_pipeline=CzarrPipeline`` to
``zarr.create_array`` / ``zarr.open``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from zarr.core.codec_pipeline import BatchedCodecPipeline

from czarr._nvtx import nvtx_range
from czarr.pipeline.pinned import PinnedHostPool

if TYPE_CHECKING:
    from collections.abc import Iterable

    from zarr.abc.codec import CodecPipeline  # noqa: F401
    from zarr.abc.store import ByteGetter, ByteSetter
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import NDBuffer
    from zarr.core.codec_pipeline import GetResult
    from zarr.core.indexing import SelectorTuple


class CzarrPipeline(BatchedCodecPipeline):
    """zarr v3 CodecPipeline with GPU-aware staging + shared substrate."""

    # Class-level shared substrate.  Lazily initialised on first
    # ``get_pinned_pool`` access; reconfigurable via :func:`configure`.
    # Static state is fine here — device-bound and process-global by nature.
    _pinned_pool: ClassVar[PinnedHostPool | None] = None

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
        pinned_prealloc: Iterable[tuple[int, int]] | None = None,
    ) -> None:
        """Reconfigure the shared substrate; closes existing pools.

        Call from :func:`czarr.configure_gpu` to set defaults at process
        startup.  Subsequent reconfiguration is allowed but should be
        done from a quiescent state (no in-flight pipeline calls).
        """
        if cls._pinned_pool is not None:
            cls._pinned_pool.close()
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
    # speedup will come from a future phase (cuFile batched I/O API:
    # cuFileBatchIOSetUp/Submit/GetStatus) which collapses the open +
    # register cycle into one driver call.  Until then, no override.

    # ------------------------------------------------------------------
    # NVTX-instrumented read/write — same logic as the parent, but each
    # micro-batch gets a clearly-bounded range on the nsys timeline so the
    # read↔decode interleave is visible.  Without this every batch shows
    # up as an interleaved blur of unrelated chunks.
    # ------------------------------------------------------------------

    async def read_batch(
        self,
        batch_info: Iterable[tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        out: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> tuple[GetResult, ...]:
        """Run zarr's batched read with a per-microbatch NVTX range."""
        # Materialise so we can count without consuming the iterator twice.
        items = list(batch_info)
        with nvtx_range("czarr.pipeline.read_batch", n=len(items)):
            return await super().read_batch(items, out, drop_axes)

    async def write_batch(
        self,
        batch_info: Iterable[tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        value: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        """Run zarr's batched write with a per-microbatch NVTX range."""
        items = list(batch_info)
        with nvtx_range("czarr.pipeline.write_batch", n=len(items)):
            await super().write_batch(items, value, drop_axes)
