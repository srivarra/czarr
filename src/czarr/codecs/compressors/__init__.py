"""BytesBytesCodec compressors backed by nvCOMP.

Two families:

* **Native** (:mod:`czarr.codecs.compressors.native`) — nvCOMP-only
  formats.  Maximum throughput, but only consumable by another
  nvCOMP-using process.  Classes: :class:`ANS`, :class:`Bitcomp`,
  :class:`Cascaded`, :class:`Deflate`, :class:`GDeflate`, :class:`Snappy`.

* **Compat** (one module per codec) — standard bitstreams readable by
  CPU implementations.  These shadow the corresponding stdlib codecs in
  zarr's registry once :func:`czarr.configure_gpu` is called, so
  existing CPU-written zarr v3 stores decode on the GPU transparently.
  Classes: :class:`Zstd`, :class:`LZ4`, :class:`Gzip`, :class:`Zlib`,
  :class:`Blosc` (decode-only).
"""

from czarr.codecs.compressors.blosc import Blosc
from czarr.codecs.compressors.gzip import Gzip
from czarr.codecs.compressors.lz4 import LZ4
from czarr.codecs.compressors.native import (
    ANS,
    Bitcomp,
    Cascaded,
    Deflate,
    GDeflate,
    Snappy,
)
from czarr.codecs.compressors.zlib import Zlib
from czarr.codecs.compressors.zstd import Zstd

__all__ = [
    "ANS",
    "LZ4",
    "Bitcomp",
    "Blosc",
    "Cascaded",
    "Deflate",
    "GDeflate",
    "Gzip",
    "Snappy",
    "Zlib",
    "Zstd",
]
