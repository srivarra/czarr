"""Shuffle — GPU byteshuffle BytesBytesCodec.

numcodecs-compatible byteshuffle filter that operates on the encoded
byte buffer.  Each chunk is treated as one big shuffle block of size
``chunk_byte_length`` with element width ``elementsize``; the codec
swaps adjacent ``elementsize`` byte rows so each plane is contiguous.

Single ``"cupy"`` backend — ``reshape/transpose/ascontiguousarray``,
works on every supported GPU including H100/H200.  A cuTile transpose
variant (~2.5x more bandwidth on A40) was deleted: cuda-tile 1.3 fails
to compile on sm_90, the real-GDS targets, and shuffle is not a
read-path bottleneck.  Re-add as a cupy RawKernel if a filter sweep
ever shows it mattering (see git history).  The ``backend`` field is
runtime — not persisted in Zarr v3 metadata.
"""

from dataclasses import dataclass, field
from typing import ClassVar, Self

import cupy as cp
from zarr.abc.codec import BytesBytesCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer
from zarr.core.common import JSON

from czarr.codecs._backend import CodecBackend, resolve_backend_for_filter
from czarr.kernels.byteshuffle import byteshuffle, byteunshuffle


@dataclass(frozen=True)
class Shuffle(BytesBytesCodec):
    """GPU byteshuffle — numcodecs-compatible BytesBytesCodec.

    Parameters
    ----------
    elementsize:
        Number of bytes per element to shuffle across.  Must divide the
        chunk byte length.
    backend:
        Runtime impl choice.  Only ``"cupy"`` today.  Not persisted.
    """

    is_fixed_size: ClassVar[bool] = False
    codec_name: ClassVar[str] = "shuffle"
    _supported_backends: ClassVar[tuple[CodecBackend, ...]] = ("cupy",)
    _default_backend: ClassVar[CodecBackend] = "cupy"

    elementsize: int = 4
    backend: CodecBackend | None = field(default=None, compare=False, repr=True)

    def __post_init__(self) -> None:
        chosen = resolve_backend_for_filter(
            self.codec_name,
            instance_backend=self.backend,
            supported=self._supported_backends,
            default=self._default_backend,
        )
        object.__setattr__(self, "backend", chosen)

    async def _decode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> Buffer:
        arr = chunk_data.as_array_like()
        cp_arr = cp.asarray(arr).view(cp.uint8) if not isinstance(arr, cp.ndarray) else arr.view(cp.uint8)
        out = byteunshuffle(cp_arr, self.elementsize, cp_arr.size)
        return chunk_spec.prototype.buffer.from_array_like(out)

    async def _encode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> Buffer:
        arr = chunk_data.as_array_like()
        cp_arr = cp.asarray(arr).view(cp.uint8) if not isinstance(arr, cp.ndarray) else arr.view(cp.uint8)
        out = byteshuffle(cp_arr, self.elementsize, cp_arr.size)
        return chunk_spec.prototype.buffer.from_array_like(out)

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        """Shuffle is a permutation — same size in and out."""
        return input_byte_length

    def to_dict(self) -> dict[str, JSON]:
        """Serialise codec config for storage in Zarr v3 metadata.

        ``backend`` is intentionally omitted — bitstream is the only
        persisted identity.
        """
        return {
            "name": self.codec_name,
            "configuration": {"elementsize": self.elementsize},
        }

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        """Reconstruct from Zarr v3 metadata: {'name': ..., 'configuration': {...}}.

        Tolerant of writers that leak ``backend`` into the configuration.
        """
        cfg = dict(data.get("configuration", {k: v for k, v in data.items() if k != "name"}))
        cfg.pop("backend", None)
        return cls(**cfg)
