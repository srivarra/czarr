"""Zstd — RFC 8478 frame compatible with libzstd / numcodecs.Zstd."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from czarr.codecs.base import CudaBytesBytesCodec, _Algorithm, _BitstreamKind


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
