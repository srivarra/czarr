"""GPU codecs for Zarr v3 — nvCOMP-backed.

* :mod:`czarr.codecs.compressors` — BytesBytesCodec implementations
  (Zstd, LZ4, Gzip, Zlib, ANS, Bitcomp, Cascaded, Deflate, GDeflate,
  Snappy).
* :mod:`czarr.codecs.filters` — ArrayArrayCodec implementations
  (populated in Phase 3 — Bitshuffle, Shuffle, Delta, FixedScaleOffset,
  BitRound).
* :mod:`czarr.codecs.checksum` — checksum BytesBytesCodecs (populated in
  Phase 5 — Crc32c).

Public symbols are re-exported from this module so users can write
``from czarr.codecs import Zstd`` (or just ``from czarr import Zstd``).
"""

from zarr.registry import register_codec

from czarr.codecs.base import Checksum, CudaBytesBytesCodec
from czarr.codecs.compressors import (
    ANS,
    LZ4,
    Bitcomp,
    Cascaded,
    Deflate,
    GDeflate,
    Gzip,
    Snappy,
    Zlib,
    Zstd,
)
from czarr.codecs.filters import BitRound, Delta, FixedScaleOffset, Shuffle
from czarr.codecs.sharding import CzarrShardingCodec

# Register all codec classes with zarr's codec registry.  Compat codecs
# and filters shadow stdlib codec_ids ("zstd", "lz4", "gzip", "zlib",
# "shuffle", "delta", "fixedscaleoffset", "bitround"); native codecs use
# "czarr.*" so they never collide.
for _cls in (
    ANS,
    Bitcomp,
    Cascaded,
    Deflate,
    GDeflate,
    Snappy,
    Zstd,
    LZ4,
    Gzip,
    Zlib,
    Shuffle,
    Delta,
    FixedScaleOffset,
    BitRound,
):
    register_codec(_cls.codec_name, _cls)

__all__ = [
    "ANS",
    "Bitcomp",
    "BitRound",
    "Cascaded",
    "Checksum",
    "CudaBytesBytesCodec",
    "CzarrShardingCodec",
    "Delta",
    "Deflate",
    "FixedScaleOffset",
    "GDeflate",
    "Gzip",
    "LZ4",
    "Shuffle",
    "Snappy",
    "Zlib",
    "Zstd",
]
