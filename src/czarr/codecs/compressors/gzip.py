"""Gzip — RFC 1952 wrapper around nvCOMP raw deflate."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from czarr.codecs.base import CudaBytesBytesCodec, _Algorithm, _BitstreamKind

if TYPE_CHECKING:
    from zarr.core.common import JSON


@dataclass(frozen=True)
class Gzip(CudaBytesBytesCodec):
    """RFC 1952 gzip — compatible with zarr's ``GzipCodec`` / numcodecs.GZip.

    Wraps nvCOMP's raw deflate output with the 10-byte gzip header and the
    8-byte CRC32 + ISIZE trailer on encode; strips them on decode.

    ``level`` is accepted for metadata round-trip; nvCOMP picks its own.
    """

    codec_name: ClassVar[str] = "gzip"
    _algorithm: ClassVar[_Algorithm] = _Algorithm.DEFLATE
    _bitstream_kind: ClassVar[_BitstreamKind] = _BitstreamKind.RAW
    _frame_strip_head: ClassVar[int] = 10
    _frame_strip_tail: ClassVar[int] = 8

    level: int = 5

    def _wrap_frame(self, compressed: bytes, original: memoryview) -> bytes:
        # gzip RFC 1952 header: ID1 ID2 CM FLG MTIME(4) XFL OS
        # Use canonical "no extra flags, MTIME=0, OS=unknown(255)".
        header = b"\x1f\x8b\x08\x00" + b"\x00\x00\x00\x00" + b"\x00\xff"
        crc = zlib.crc32(bytes(original)) & 0xFFFFFFFF
        isize = len(original) & 0xFFFFFFFF
        trailer = struct.pack("<II", crc, isize)
        return header + compressed + trailer

    def to_dict(self) -> dict[str, JSON]:
        """Emit the numcodecs Gzip schema (no czarr-internal fields)."""
        return {
            "name": self.codec_name,
            "configuration": {"level": int(self.level)},
        }
