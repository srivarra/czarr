"""Shuffle — GPU byteshuffle BytesBytesCodec.

numcodecs-compatible byteshuffle filter that operates on the encoded
byte buffer.  Each chunk is treated as one big shuffle block of size
``chunk_byte_length`` with element width ``elementsize``; the codec
swaps adjacent ``elementsize`` byte rows so each plane is contiguous.

Wraps the cuTile kernel in :mod:`czarr.kernels.byteshuffle`.  Matches the
on-disk bitstream produced by ``numcodecs.Shuffle(elementsize=N)`` so
v3 stores written by either CPU or GPU pipelines interoperate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import cupy as cp
from zarr.abc.codec import BytesBytesCodec

from czarr.kernels.byteshuffle import byteshuffle_batched, byteunshuffle_batched

if TYPE_CHECKING:
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer
    from zarr.core.common import JSON


@dataclass(frozen=True)
class Shuffle(BytesBytesCodec):
    """GPU byteshuffle — numcodecs-compatible BytesBytesCodec.

    Parameters
    ----------
    elementsize:
        Number of bytes per element to shuffle across.  Must divide the
        chunk byte length.
    """

    is_fixed_size: ClassVar[bool] = False
    codec_name: ClassVar[str] = "shuffle"
    elementsize: int = 4

    async def _decode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> Buffer:
        arr = chunk_data.as_array_like()
        cp_arr = cp.asarray(arr).view(cp.uint8) if not isinstance(arr, cp.ndarray) else arr.view(cp.uint8)
        out = byteunshuffle_batched(cp_arr, self.elementsize, cp_arr.size)
        return chunk_spec.prototype.buffer.from_array_like(out)

    async def _encode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> Buffer:
        arr = chunk_data.as_array_like()
        cp_arr = cp.asarray(arr).view(cp.uint8) if not isinstance(arr, cp.ndarray) else arr.view(cp.uint8)
        out = byteshuffle_batched(cp_arr, self.elementsize, cp_arr.size)
        return chunk_spec.prototype.buffer.from_array_like(out)

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        """Shuffle is a permutation — same size in and out."""
        return input_byte_length

    def to_dict(self) -> dict[str, JSON]:
        """Serialise codec config for storage in Zarr v3 metadata."""
        return {
            "name": self.codec_name,
            "configuration": {"elementsize": self.elementsize},
        }

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Shuffle:
        """Reconstruct from Zarr v3 metadata: {'name': ..., 'configuration': {...}}."""
        if "configuration" in data:
            return cls(**data["configuration"])
        # Be tolerant of someone passing the raw configuration dict.
        return cls(**{k: v for k, v in data.items() if k != "name"})
