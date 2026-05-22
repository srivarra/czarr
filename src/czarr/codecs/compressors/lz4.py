"""LZ4 — block format compatible with numcodecs.LZ4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from czarr.codecs.base import CudaBytesBytesCodec, _Algorithm, _BitstreamKind

if TYPE_CHECKING:
    from zarr.core.common import JSON


@dataclass(frozen=True)
class LZ4(CudaBytesBytesCodec):
    """LZ4 block format compatible with numcodecs.LZ4."""

    codec_name: ClassVar[str] = "lz4"
    _algorithm: ClassVar[_Algorithm] = _Algorithm.LZ4
    _bitstream_kind: ClassVar[_BitstreamKind] = _BitstreamKind.WITH_UNCOMPRESSED_SIZE

    acceleration: int = 1  # numcodecs metadata field; ignored by nvCOMP

    def to_dict(self) -> dict[str, JSON]:
        """Emit the numcodecs LZ4 schema (no czarr-internal fields)."""
        return {
            "name": self.codec_name,
            "configuration": {"acceleration": int(self.acceleration)},
        }
