"""GPU codecs for Zarr — nvCOMP-backed.

Two families:

* **Native** (:mod:`czarr.codecs.native`) — nvCOMP-only formats.  Maximum
  throughput, but only consumable by another nvCOMP-using process.
  Classes: :class:`ANS`, :class:`Bitcomp`, :class:`Cascaded`, :class:`GDeflate`.

* **Compat** (:mod:`czarr.codecs.compat`) — standard bitstreams readable by
  CPU implementations.  These shadow the corresponding stdlib codecs in
  Zarr's registry once :func:`czarr.configure_gpu` is called, so existing
  CPU-written zarr stores decode on the GPU transparently.
  Classes: :class:`Zstd`, :class:`LZ4`, :class:`Gzip`, :class:`Zlib`.
"""

from zarr.registry import register_codec

from czarr.codecs.base import Checksum, Codec
from czarr.codecs.compat import LZ4, Gzip, Zlib, Zstd
from czarr.codecs.native import ANS, Bitcomp, Cascaded, Deflate, GDeflate, Snappy

# Register all codec classes with zarr's codec registry.  Compat codecs
# shadow stdlib codec_ids ("zstd", "lz4", "gzip", "zlib"); native codecs
# use "czarr.*" so they never collide.
for _cls in (ANS, Bitcomp, Cascaded, Deflate, GDeflate, Snappy, Zstd, LZ4, Gzip, Zlib):
    register_codec(_cls.codec_name, _cls)

__all__ = [
    "ANS",
    "Bitcomp",
    "Cascaded",
    "Checksum",
    "Codec",
    "Deflate",
    "GDeflate",
    "Gzip",
    "LZ4",
    "Snappy",
    "Zlib",
    "Zstd",
]
