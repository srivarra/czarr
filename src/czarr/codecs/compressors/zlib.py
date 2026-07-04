"""Zlib — RFC 1950 wrapper around nvCOMP raw deflate."""

import struct
import zlib
from dataclasses import dataclass
from typing import ClassVar, override

from zarr.core.common import JSON

from czarr.codecs.base import CudaBytesBytesCodec, _Algorithm, _BitstreamKind


@dataclass(frozen=True)
class Zlib(CudaBytesBytesCodec):
    """RFC 1950 zlib — compatible with numcodecs.Zlib.

    Wraps nvCOMP's raw deflate output with the 2-byte zlib header and the
    4-byte Adler-32 trailer on encode; strips them on decode.
    """

    codec_name: ClassVar[str] = "zlib"
    _algorithm: ClassVar[_Algorithm] = _Algorithm.DEFLATE
    _bitstream_kind: ClassVar[_BitstreamKind] = _BitstreamKind.RAW
    _frame_strip_head: ClassVar[int] = 2
    _frame_strip_tail: ClassVar[int] = 4

    level: int = 5

    @override
    def _wrap_frame(self, compressed: bytes, original: memoryview) -> bytes:
        # zlib RFC 1950 header for default level + 32K window: 0x789C.
        header = b"\x78\x9c"
        adler = zlib.adler32(bytes(original)) & 0xFFFFFFFF
        trailer = struct.pack(">I", adler)
        return header + compressed + trailer

    @override
    def to_dict(self) -> dict[str, JSON]:
        """Emit the numcodecs Zlib schema (no czarr-internal fields)."""
        return {
            "name": self.codec_name,
            "configuration": {"level": int(self.level)},
        }
