"""ArrayArrayCodec filters + filter-shaped BytesBytesCodecs (Shuffle).

Phase 3 of the GPU-native zarr v3 pipeline refactor (epic ianyfe7m).

* :class:`Shuffle` — ``BytesBytesCodec``, codec_name ``"shuffle"``.
  GPU byteshuffle via the cuTile kernel in :mod:`czarr.kernels.byteshuffle`.
* :class:`Delta` — ``ArrayArrayCodec``, codec_name ``"delta"``.
  First-differences encode; cumsum decode.
* :class:`FixedScaleOffset` — ``ArrayArrayCodec``, codec_name
  ``"fixedscaleoffset"``.  Affine quantisation.
* :class:`BitRound` — ``ArrayArrayCodec``, codec_name ``"bitround"``.
  IEEE-754 mantissa truncation.
"""

from czarr.codecs.filters.bitround import BitRound
from czarr.codecs.filters.delta import Delta
from czarr.codecs.filters.fixedscaleoffset import FixedScaleOffset
from czarr.codecs.filters.shuffle import Shuffle

__all__ = [
    "BitRound",
    "Delta",
    "FixedScaleOffset",
    "Shuffle",
]
