"""CPU-format-compatible GPU codecs.

These codecs produce/consume the same on-disk bitstream as numcodecs (and
zarr's bundled CPU codecs).  Registering them under the standard codec_id
(``"zstd"``, ``"lz4"``, etc.) means existing zarr stores written by any
CPU implementation decode transparently on the GPU once czarr is imported
and :func:`czarr.configure_gpu` selects the GPU registration.

| Class    | codec_id   | nvCOMP algo | bitstream                    | framing                       |
|----------|------------|-------------|------------------------------|-------------------------------|
| `Zstd`   | ``zstd``   | Zstd        | ``RAW`` (RFC 8478 frame)     | none                          |
| `LZ4`    | ``lz4``    | LZ4         | ``WITH_UNCOMPRESSED_SIZE``   | 4-byte usize prefix (auto)    |
| `Gzip`   | ``gzip``   | Deflate     | ``RAW`` (raw deflate)        | 10-byte hdr + CRC32/ISIZE     |
| `Zlib`   | ``zlib``   | Deflate     | ``RAW`` (raw deflate)        | 2-byte hdr + Adler-32         |
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import ClassVar

from czarr.codecs.base import Codec, _Algorithm, _BitstreamKind


@dataclass(frozen=True)
class Zstd(Codec):
    """Zstd — RFC 8478 frame compatible with libzstd / numcodecs.Zstd.

    ``level`` / ``checksum`` are accepted for metadata round-trip with
    zarr's built-in ``ZstdCodec``; nvCOMP picks its own internal level.
    """

    codec_name: ClassVar[str] = "zstd"
    _algorithm: ClassVar[_Algorithm] = _Algorithm.ZSTD
    _bitstream_kind: ClassVar[_BitstreamKind] = _BitstreamKind.RAW

    level: int = 0
    checksum: bool = False


@dataclass(frozen=True)
class LZ4(Codec):
    """LZ4 block format compatible with numcodecs.LZ4."""

    codec_name: ClassVar[str] = "lz4"
    _algorithm: ClassVar[_Algorithm] = _Algorithm.LZ4
    _bitstream_kind: ClassVar[_BitstreamKind] = _BitstreamKind.WITH_UNCOMPRESSED_SIZE

    acceleration: int = 1  # numcodecs metadata field; ignored by nvCOMP


@dataclass(frozen=True)
class Gzip(Codec):
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


@dataclass(frozen=True)
class Zlib(Codec):
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

    def _wrap_frame(self, compressed: bytes, original: memoryview) -> bytes:
        # zlib RFC 1950 header for default level + 32K window: 0x789C.
        header = b"\x78\x9c"
        adler = zlib.adler32(bytes(original)) & 0xFFFFFFFF
        trailer = struct.pack(">I", adler)
        return header + compressed + trailer
