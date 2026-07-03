"""LZ4 — block format compatible with numcodecs.LZ4.

Backed by nvCOMP.  czarr's ``WITH_UNCOMPRESSED_SIZE`` bitstream — a
4-byte little-endian uncompressed-size prefix followed by the LZ4 block,
matching ``numcodecs.LZ4`` — is parsed by nvCOMP internally, so the
whole buffer is handed to nvCOMP unchanged.

(czarr previously shipped a hand-written cuda-python LZ4 decoder behind
a ``backend="native"`` switch.  Profiling showed the codec is rarely the
read-path bottleneck — data movement and buffer plumbing dominate — so
the custom kernel was dropped in favour of wrapping nvCOMP, the same as
every other compressor.)
"""

from dataclasses import dataclass
from typing import ClassVar, Self

from zarr.core.common import JSON

from czarr.codecs.base import CudaBytesBytesCodec, _Algorithm, _BitstreamKind, codec_config


@dataclass(frozen=True)
class LZ4(CudaBytesBytesCodec):
    """LZ4 block format compatible with ``numcodecs.LZ4``.

    Decodes the ``WITH_UNCOMPRESSED_SIZE`` bitstream (4-byte LE
    uncompressed-size prefix + LZ4 block) via nvCOMP, which parses the
    prefix internally.

    ``acceleration`` is accepted for metadata round-trip with
    ``numcodecs.LZ4``; nvCOMP picks its own internal settings.
    """

    codec_name: ClassVar[str] = "lz4"
    _algorithm: ClassVar[_Algorithm] = _Algorithm.LZ4
    _bitstream_kind: ClassVar[_BitstreamKind] = _BitstreamKind.WITH_UNCOMPRESSED_SIZE

    acceleration: int = 1  # numcodecs metadata field; ignored by nvCOMP

    def to_dict(self) -> dict[str, JSON]:
        """Emit the numcodecs LZ4 schema (no czarr-internal fields).

        The base ``CudaBytesBytesCodec.to_dict`` would also serialise
        ``chunk_size`` / ``checksum_policy`` / ``device_id``, which break
        round-trip when the store is opened by a non-czarr reader.
        """
        return {
            "name": self.codec_name,
            "configuration": {"acceleration": int(self.acceleration)},
        }

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        """Reconstruct from Zarr v3 metadata."""
        return cls(**codec_config(data))
