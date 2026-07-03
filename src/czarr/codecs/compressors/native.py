"""nvCOMP-exclusive codecs.

These produce nvCOMP's native chunked bitstream — fastest path, but
**not** consumable by any CPU implementation.  Registered under
``czarr.*`` codec names so they never collide with stdlib codecs.
"""

from dataclasses import dataclass
from typing import ClassVar

from czarr.codecs.base import CudaBytesBytesCodec, _Algorithm


@dataclass(frozen=True)
class ANS(CudaBytesBytesCodec):
    """Asymmetric Numeral Systems — high throughput, balanced ratio."""

    codec_name: ClassVar[str] = "czarr.ans"
    _algorithm: ClassVar[_Algorithm] = _Algorithm.ANS


@dataclass(frozen=True)
class Bitcomp(CudaBytesBytesCodec):
    """Bitcomp — highest throughput on numeric data.

    ``algorithm_type``:

    * 0 — default, best compression ratio
    * 1 — sparse mode, faster on data with many zeros
    """

    codec_name: ClassVar[str] = "czarr.bitcomp"
    _algorithm: ClassVar[_Algorithm] = _Algorithm.BITCOMP

    algorithm_type: int = 0

    def _codec_kwargs(self) -> dict:
        return {"algorithm_type": self.algorithm_type}


@dataclass(frozen=True)
class Cascaded(CudaBytesBytesCodec):
    """Cascaded — RLE + delta + bit-packing for integer columns."""

    codec_name: ClassVar[str] = "czarr.cascaded"
    _algorithm: ClassVar[_Algorithm] = _Algorithm.CASCADED

    num_rles: int = 2
    num_deltas: int = 1
    use_bitpack: bool = True

    def _codec_kwargs(self) -> dict:
        return {
            "num_rles": self.num_rles,
            "num_deltas": self.num_deltas,
            "use_bitpack": self.use_bitpack,
        }


@dataclass(frozen=True)
class Snappy(CudaBytesBytesCodec):
    """nvCOMP Snappy — chunked nvCOMP-native bitstream.

    NOT bytewise-compatible with the standard Snappy block format.  Listed
    for algorithm parity with the original codec set.
    """

    codec_name: ClassVar[str] = "czarr.snappy"
    _algorithm: ClassVar[_Algorithm] = _Algorithm.SNAPPY


@dataclass(frozen=True)
class Deflate(CudaBytesBytesCodec):
    """Raw deflate (RFC 1951) — nvCOMP-native chunked bitstream.

    For CPU-interop deflate use :class:`czarr.Gzip` or :class:`czarr.Zlib`
    (both wrap raw deflate with the appropriate framing).

    ``algorithm_type`` (0-5) trades throughput for compression ratio.
    """

    codec_name: ClassVar[str] = "czarr.deflate"
    _algorithm: ClassVar[_Algorithm] = _Algorithm.DEFLATE

    algorithm_type: int = 1

    def _codec_kwargs(self) -> dict:
        return {"algorithm_type": self.algorithm_type}


@dataclass(frozen=True)
class GDeflate(CudaBytesBytesCodec):
    """nvCOMP GDeflate — chunked deflate variant optimised for GPU parallelism.

    NOT bitstream-compatible with standard deflate/gzip.  Use :class:`Gzip`
    or :class:`Zlib` if you need CPU interop.

    ``algorithm_type`` (0-5): trades throughput for compression ratio.
    """

    codec_name: ClassVar[str] = "czarr.gdeflate"
    _algorithm: ClassVar[_Algorithm] = _Algorithm.GDEFLATE

    algorithm_type: int = 1

    def _codec_kwargs(self) -> dict:
        return {"algorithm_type": self.algorithm_type}
