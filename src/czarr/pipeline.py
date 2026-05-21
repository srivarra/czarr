"""Custom codec pipeline that hands the whole chunk batch to the codec at once.

Zarr's default :class:`zarr.core.codec_pipeline.BatchedCodecPipeline` chunks
the read into groups of ``codec_pipeline.batch_size`` and calls
``Codec.decode`` once per group.  The default ``batch_size`` is **1**, so
every chunk gets its own decode call — losing nvCOMP's batched-decode win.

:class:`CzarrCodecPipeline` defaults ``batch_size`` to ``sys.maxsize`` so a
single ``arr[:]`` triggers one decode call across all chunks — every selection
through zarr's normal indexing path gets the full nvCOMP batched-decode win.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from zarr.core.codec_pipeline import BatchedCodecPipeline
from zarr.registry import register_pipeline

from czarr._nvtx import nvtx_range

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Self

    from zarr.abc.codec import Codec
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer, NDBuffer


class CzarrCodecPipeline(BatchedCodecPipeline):
    """Drop-in replacement that maximises batch size + adds NVTX traces."""

    @classmethod
    def from_codecs(cls, codecs: Iterable[Codec], *, batch_size: int | None = None) -> Self:
        """Build the pipeline; default batch_size = sys.maxsize so one decode runs per read."""
        return super().from_codecs(codecs, batch_size=batch_size or sys.maxsize)

    async def decode_batch(
        self,
        chunk_bytes_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[NDBuffer | None]:
        """Decode every chunk in this batch through a single ``Codec.decode([all])`` call."""
        items = list(chunk_bytes_and_specs)
        with nvtx_range("czarr.pipeline.decode_batch", n=len(items)):
            return await super().decode_batch(items)

    async def encode_batch(
        self,
        chunk_arrays_and_specs: Iterable[tuple[NDBuffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        """Encode every chunk in this batch through a single ``Codec.encode([all])`` call."""
        items = list(chunk_arrays_and_specs)
        with nvtx_range("czarr.pipeline.encode_batch", n=len(items)):
            return await super().encode_batch(items)


register_pipeline(CzarrCodecPipeline)
