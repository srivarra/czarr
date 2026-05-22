"""Zstd — RFC 8478 frame compatible with libzstd / numcodecs.Zstd."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from czarr.codecs.base import CudaBytesBytesCodec, _Algorithm, _BitstreamKind

if TYPE_CHECKING:
    from zarr.core.common import JSON


@dataclass(frozen=True)
class Zstd(CudaBytesBytesCodec):
    """Zstd — RFC 8478 frame compatible with libzstd / numcodecs.Zstd.

    ``level`` / ``checksum`` are accepted for metadata round-trip with
    zarr's built-in ``ZstdCodec``; nvCOMP picks its own internal level.
    """

    codec_name: ClassVar[str] = "zstd"
    _algorithm: ClassVar[_Algorithm] = _Algorithm.ZSTD
    _bitstream_kind: ClassVar[_BitstreamKind] = _BitstreamKind.RAW

    level: int = 0
    checksum: bool = False

    def to_dict(self) -> dict[str, JSON]:
        """Emit the zarr-v3 Zstd codec schema (no czarr-internal fields).

        The base ``CudaBytesBytesCodec.to_dict`` would also serialise
        ``chunk_size`` / ``checksum_policy`` / ``device_id``, which break
        round-trip when the store is opened by a non-czarr reader
        (``zarr.codecs.ZstdCodec`` rejects unknown configuration keys).
        """
        return {
            "name": self.codec_name,
            "configuration": {"level": int(self.level), "checksum": bool(self.checksum)},
        }
