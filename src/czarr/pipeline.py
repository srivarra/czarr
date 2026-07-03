"""``CzarrPipeline`` — GPU-native subclass of zarr v3's BatchedCodecPipeline.

In zarr-python 3.x, a :class:`zarr.abc.codec.CodecPipeline` is
instantiated per array call and orchestrates the codec chain.

What this subclass changes:

* Adds NVTX-instrumented ``read_batch`` / ``write_batch`` so the nsys
  timeline shows the read + decode interleave that fires when
  ``codec_pipeline.batch_size`` is set below ``len(chunks)``.

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
— the default is one bulk batch (no overlap; measured near-optimal on
GDS), with ``configure_gpu(batch_size=...)`` as the opt-in.

The NVTX wrappers below make the overlap visible in nsys; the actual
concurrency comes from zarr's pipeline machinery.

Register globally via :func:`czarr.configure_gpu` (sets
``codec_pipeline.path = "czarr.pipeline.CzarrPipeline"``).  To scope it,
use ``with czarr.configure_gpu(): ...`` or set
``zarr.config.set({"codec_pipeline.path": "czarr.pipeline.CzarrPipeline"})``
directly — zarr 3.x does not accept a ``codec_pipeline=`` kwarg per array.
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING

from zarr.abc.codec import GetResult
from zarr.abc.store import ByteGetter, ByteSetter
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import NDBuffer
from zarr.core.codec_pipeline import BatchedCodecPipeline
from zarr.core.indexing import SelectorTuple

from czarr._nvtx import nvtx_range

if TYPE_CHECKING:
    pass


class CzarrPipeline(BatchedCodecPipeline):
    """zarr v3 CodecPipeline with NVTX-instrumented batched read/write."""

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
    # GPULocalStore.get_many was removed with the dead-code sweep (git
    # history has it); cuFile batched I/O (cuFileBatchIOSetUp/Submit)
    # was probed separately and also lost to threaded sync reads.  zarr's
    # stock concurrent_map path is the read dispatcher.

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
